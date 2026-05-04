"""Tests for SemanticCache decoupling from asyncpg-specific assumptions.

These tests exercise the storage surface of
``atlas_brain.reasoning.semantic_cache`` that doesn't require a live
Postgres -- ``CacheEntry`` / ``SemanticCachePool`` / ``SemanticCache``
plus the atlas-namespace ``_apply_decay`` and ``compute_evidence_hash``
re-exports -- and the canonical ``row_to_cache_entry`` from
``extracted_reasoning_core.semantic_cache_keys`` (PR-C2 promoted that
helper from a private staticmethod into core). The async query methods
are tested against a fake pool that records its inputs.

The atlas ``DatabasePool`` and a raw asyncpg ``Pool`` both satisfy the
Protocol; tests here use a plain duck-typed fake so this suite can
run in unit-test mode without the database fixtures the
``test_reasoning_live.py`` suite needs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from atlas_brain.reasoning.semantic_cache import (
    CacheEntry,
    SemanticCache,
    SemanticCachePool,
    _apply_decay,
    compute_evidence_hash,
)

# PR-C2: ``row_to_cache_entry`` was promoted from a private staticmethod
# on ``SemanticCache`` to a public function in
# ``extracted_reasoning_core.semantic_cache_keys``. The decoupling tests
# below import it from the canonical home rather than reaching into
# ``SemanticCache._row_to_entry`` (which no longer exists).
from extracted_reasoning_core.semantic_cache_keys import row_to_cache_entry


class _FakePool:
    """Duck-typed pool that records calls + returns scripted rows."""

    def __init__(self) -> None:
        self.fetchrow_calls: list[tuple[str, tuple]] = []
        self.fetch_calls: list[tuple[str, tuple]] = []
        self.execute_calls: list[tuple[str, tuple]] = []
        self.fetchrow_return: Any = None
        self.fetch_return: list[Any] = []

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.fetchrow_calls.append((query, args))
        return self.fetchrow_return

    async def fetch(self, query: str, *args: Any) -> Any:
        self.fetch_calls.append((query, args))
        return self.fetch_return

    async def execute(self, query: str, *args: Any) -> Any:
        self.execute_calls.append((query, args))
        return None


def _row(**overrides: Any) -> dict[str, Any]:
    """Build a row dict matching the SELECT * shape of reasoning_semantic_cache."""
    base: dict[str, Any] = {
        "pattern_sig": "sig1",
        "pattern_class": "cls1",
        "vendor_name": "vendor_x",
        "product_category": None,
        "conclusion": {"summary": "test"},
        "confidence": 0.9,
        "reasoning_steps": [{"step": 1}],
        "boundary_conditions": {"max_age_days": 30},
        "falsification_conditions": ["c1", "c2"],
        "uncertainty_sources": ["u1"],
        "decay_half_life_days": 90,
        "conclusion_type": "T1",
        "evidence_hash": "abc123",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "last_validated_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
        "validation_count": 3,
    }
    base.update(overrides)
    return base


# ---- compute_evidence_hash ----


def test_evidence_hash_is_deterministic():
    h1 = compute_evidence_hash({"a": 1, "b": [1, 2]})
    h2 = compute_evidence_hash({"b": [1, 2], "a": 1})  # different key order
    assert h1 == h2


def test_evidence_hash_is_16_hex_chars():
    h = compute_evidence_hash({"a": 1})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_evidence_hash_changes_on_value_change():
    h1 = compute_evidence_hash({"a": 1})
    h2 = compute_evidence_hash({"a": 2})
    assert h1 != h2


def test_evidence_hash_handles_non_json_safe_values():
    # default=str: dates / arbitrary objects serialise via repr/str
    h = compute_evidence_hash({"date": datetime(2026, 5, 1, tzinfo=timezone.utc)})
    assert len(h) == 16


# ---- _apply_decay ----


def test_decay_returns_full_confidence_on_no_elapsed_days():
    now = datetime.now(timezone.utc)
    out = _apply_decay(0.9, now, half_life_days=90)
    # Days elapsed is ~0, so the function returns the input unchanged.
    assert out == pytest.approx(0.9, abs=1e-6)


def test_decay_halves_confidence_after_one_half_life():
    half_life = 30
    long_ago = datetime.now(timezone.utc) - timedelta(days=half_life)
    out = _apply_decay(0.8, long_ago, half_life_days=half_life)
    assert out == pytest.approx(0.4, abs=1e-3)


def test_decay_handles_naive_datetime():
    # Naive datetime gets the UTC tzinfo applied; do not crash.
    naive = datetime.now() - timedelta(days=10)
    out = _apply_decay(0.8, naive, half_life_days=30)
    assert 0 < out < 0.8


def test_decay_zero_half_life_returns_input():
    long_ago = datetime.now(timezone.utc) - timedelta(days=100)
    out = _apply_decay(0.5, long_ago, half_life_days=0)
    assert out == 0.5


# ---- _row_to_entry: row-shape duck typing ----


def test_row_to_entry_accepts_plain_dict():
    entry = row_to_cache_entry(_row())
    assert isinstance(entry, CacheEntry)
    assert entry.pattern_sig == "sig1"
    assert entry.confidence == 0.9
    assert entry.validation_count == 3


def test_row_to_entry_jsonb_decodes_when_string():
    # Vanilla adapter shape: JSONB returned as a string.
    row = _row(
        conclusion='{"k": "v"}',
        reasoning_steps='[{"step": 9}]',
        boundary_conditions='{"max": 10}',
        falsification_conditions='["x"]',
    )
    entry = row_to_cache_entry(row)
    assert entry.conclusion == {"k": "v"}
    assert entry.reasoning_steps == [{"step": 9}]
    assert entry.boundary_conditions == {"max": 10}
    assert entry.falsification_conditions == ["x"]


def test_row_to_entry_jsonb_passes_through_when_dict():
    # Asyncpg shape: JSONB pre-decoded to dict.
    row = _row(conclusion={"already": "decoded"})
    entry = row_to_cache_entry(row)
    assert entry.conclusion == {"already": "decoded"}


def test_row_to_entry_falsification_dict_collapses_to_list():
    # Defensive case: caller stored a dict rather than a list.
    row = _row(falsification_conditions={"a": "x", "b": "y"})
    entry = row_to_cache_entry(row)
    # Either of "x", "y" -- order is dict insertion order in Py3.7+
    assert sorted(entry.falsification_conditions) == ["x", "y"]


def test_row_to_entry_handles_empty_uncertainty_sources():
    row = _row(uncertainty_sources=None)
    entry = row_to_cache_entry(row)
    assert entry.uncertainty_sources == []


# ---- Protocol satisfaction (compile-time + runtime) ----


def test_fake_pool_is_usable_as_semantic_cache_pool():
    """Duck-typing: any object with the three coroutine methods works,
    no nominal subclass required.
    """
    pool: SemanticCachePool = _FakePool()  # static check
    cache = SemanticCache(pool)
    assert cache._pool is pool


# ---- Async query plumbing through the fake pool ----


@pytest.mark.asyncio
async def test_lookup_returns_none_on_miss():
    pool = _FakePool()
    pool.fetchrow_return = None
    cache = SemanticCache(pool)
    out = await cache.lookup("missing-sig")
    assert out is None
    # Did issue the query
    assert len(pool.fetchrow_calls) == 1
    assert pool.fetchrow_calls[0][1] == ("missing-sig",)


@pytest.mark.asyncio
async def test_lookup_returns_none_when_stale():
    """Effective confidence below STALE_THRESHOLD should not surface."""
    pool = _FakePool()
    long_ago = datetime.now(timezone.utc) - timedelta(days=400)
    pool.fetchrow_return = _row(confidence=0.6, last_validated_at=long_ago)
    cache = SemanticCache(pool)
    out = await cache.lookup("sig1")
    assert out is None  # decayed below 0.5


@pytest.mark.asyncio
async def test_lookup_returns_entry_when_fresh():
    pool = _FakePool()
    pool.fetchrow_return = _row(
        confidence=0.95,
        last_validated_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    cache = SemanticCache(pool)
    out = await cache.lookup("sig1")
    assert out is not None
    assert out.pattern_sig == "sig1"
    assert out.effective_confidence is not None
    assert out.effective_confidence >= SemanticCache.STALE_THRESHOLD


@pytest.mark.asyncio
async def test_invalidate_issues_update_query():
    pool = _FakePool()
    cache = SemanticCache(pool)
    await cache.invalidate("sig1", reason="test")
    assert len(pool.execute_calls) == 1
    query, args = pool.execute_calls[0]
    assert "UPDATE reasoning_semantic_cache" in query
    assert "invalidated_at = NOW()" in query
    assert args == ("sig1",)


@pytest.mark.asyncio
async def test_validate_with_new_confidence_passes_value():
    pool = _FakePool()
    cache = SemanticCache(pool)
    await cache.validate("sig1", new_confidence=0.85)
    assert len(pool.execute_calls) == 1
    _, args = pool.execute_calls[0]
    assert args == ("sig1", 0.85)


@pytest.mark.asyncio
async def test_validate_without_new_confidence_omits_value():
    pool = _FakePool()
    cache = SemanticCache(pool)
    await cache.validate("sig1")
    assert len(pool.execute_calls) == 1
    _, args = pool.execute_calls[0]
    assert args == ("sig1",)


@pytest.mark.asyncio
async def test_lookup_by_class_vendor_only_path():
    """When vendor is set and class is empty, the vendor-only query is used."""
    pool = _FakePool()
    pool.fetch_return = []
    cache = SemanticCache(pool)
    await cache.lookup_by_class("", vendor_name="vendor_x", limit=5)
    assert len(pool.fetch_calls) == 1
    query, args = pool.fetch_calls[0]
    assert "WHERE vendor_name = $1" in query
    assert args == ("vendor_x", 5)


@pytest.mark.asyncio
async def test_lookup_for_tier_filters_stale():
    """Tier inheritance only returns entries above the stale threshold."""
    pool = _FakePool()
    fresh = _row(
        pattern_sig="fresh",
        confidence=0.95,
        last_validated_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    stale = _row(
        pattern_sig="stale",
        confidence=0.6,
        last_validated_at=datetime.now(timezone.utc) - timedelta(days=500),
    )
    pool.fetch_return = [fresh, stale]
    cache = SemanticCache(pool)
    entries = await cache.lookup_for_tier("T1", limit=10)
    sigs = [e.pattern_sig for e in entries]
    assert "fresh" in sigs
    assert "stale" not in sigs


@pytest.mark.asyncio
async def test_get_cache_stats_returns_dict_for_none_row():
    pool = _FakePool()
    pool.fetchrow_return = None
    cache = SemanticCache(pool)
    stats = await cache.get_cache_stats()
    assert stats == {
        "active": 0,
        "invalidated": 0,
        "avg_confidence": 0,
        "avg_validations": 0,
    }


@pytest.mark.asyncio
async def test_get_cache_stats_passes_through_row_dict():
    pool = _FakePool()
    pool.fetchrow_return = {
        "active": 5,
        "invalidated": 2,
        "avg_confidence": 0.8,
        "avg_validations": 3.2,
    }
    cache = SemanticCache(pool)
    stats = await cache.get_cache_stats()
    assert stats["active"] == 5
    assert stats["avg_confidence"] == 0.8


# ---- Backwards-compat: public API surface unchanged ----


def test_public_methods_signature_unchanged():
    """Rule 14 guard: the public methods exposed by SemanticCache
    must keep the same names and parameter shapes that callers
    (b2b_battle_cards, falsification_check, ecosystem_analysis,
    test_reasoning_live) import.
    """
    expected = {
        "lookup": ["pattern_sig"],
        "store": ["entry"],
        "validate": ["pattern_sig", "new_confidence"],
        "invalidate": ["pattern_sig", "reason"],
        "lookup_by_class": ["pattern_class", "vendor_name", "limit"],
        "lookup_for_tier": ["conclusion_type", "product_category", "vendor_name", "limit"],
        "get_cache_stats": [],
    }
    import inspect
    for method_name, params in expected.items():
        method = getattr(SemanticCache, method_name)
        sig = inspect.signature(method)
        actual = [p for p in sig.parameters if p != "self"]
        assert actual == params, f"{method_name}: expected {params}, got {actual}"
