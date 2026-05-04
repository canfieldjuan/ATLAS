"""Unit tests for extracted_reasoning_core.semantic_cache_keys.

Exercises the pure semantic-cache primitives (PR-C2 / PR 4):

  - ``compute_evidence_hash`` determinism + key-order stability
  - ``apply_decay`` math + edge cases (naive datetimes, non-positive
    elapsed days, non-positive half-life)
  - ``CacheEntry`` default-field behaviour
  - ``row_to_cache_entry`` handles pre-decoded vs string-shaped JSONB
    and the ``falsification_conditions`` dict-vs-list drift

Storage-side concerns (Postgres queries, ``SemanticCache`` class)
remain in the LLM infrastructure adapter and are tested separately
(see ``tests/test_semantic_cache_decoupling.py`` for the atlas-side
copy and the LLM-infra standalone smoke tests).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from extracted_reasoning_core.semantic_cache_keys import (
    CacheEntry,
    STALE_THRESHOLD,
    apply_decay,
    compute_evidence_hash,
    row_to_cache_entry,
)


# ----------------------------------------------------------------------
# compute_evidence_hash
# ----------------------------------------------------------------------


def test_compute_evidence_hash_returns_16_char_hex() -> None:
    h = compute_evidence_hash({"vendor": "Acme", "score": 0.7})
    assert isinstance(h, str)
    assert len(h) == 16
    int(h, 16)  # raises if not valid hex


def test_compute_evidence_hash_is_key_order_stable() -> None:
    h1 = compute_evidence_hash({"a": 1, "b": 2, "c": 3})
    h2 = compute_evidence_hash({"c": 3, "a": 1, "b": 2})
    assert h1 == h2


def test_compute_evidence_hash_distinguishes_distinct_evidence() -> None:
    h1 = compute_evidence_hash({"vendor": "Acme"})
    h2 = compute_evidence_hash({"vendor": "Beta"})
    assert h1 != h2


def test_compute_evidence_hash_handles_non_json_native_values() -> None:
    # default=str means datetimes/decimals don't crash.
    moment = datetime(2026, 5, 4, 12, 0, 0, tzinfo=timezone.utc)
    h = compute_evidence_hash({"validated_at": moment})
    assert len(h) == 16


# ----------------------------------------------------------------------
# apply_decay
# ----------------------------------------------------------------------


def test_apply_decay_returns_input_for_zero_elapsed() -> None:
    just_now = datetime.now(timezone.utc)
    out = apply_decay(0.9, just_now, half_life_days=90)
    # Tiny clock drift means we can't assert ==, but it must be
    # essentially unchanged (>= 0.999 of input).
    assert out >= 0.9 * 0.999


def test_apply_decay_halves_at_one_half_life() -> None:
    one_period_ago = datetime.now(timezone.utc) - timedelta(days=14)
    out = apply_decay(0.8, one_period_ago, half_life_days=14)
    # Allow small drift for clock skew during test execution.
    assert 0.39 < out < 0.41


def test_apply_decay_handles_naive_last_validated_as_utc() -> None:
    naive_one_day_ago = datetime.utcnow() - timedelta(days=1)
    out = apply_decay(1.0, naive_one_day_ago, half_life_days=14)
    # 1 day at 14-day half-life: 2^(-1/14) ~= 0.952
    assert 0.94 < out < 0.96


def test_apply_decay_short_circuits_on_zero_half_life() -> None:
    # A zero or negative half-life means "no decay model" -- return
    # the raw confidence rather than dividing by zero.
    long_ago = datetime.now(timezone.utc) - timedelta(days=365)
    assert apply_decay(0.7, long_ago, half_life_days=0) == 0.7
    assert apply_decay(0.7, long_ago, half_life_days=-1) == 0.7


def test_apply_decay_returns_input_for_future_last_validated() -> None:
    # Negative elapsed (future timestamp) is a clock-drift / data
    # corruption case; return confidence unchanged.
    in_the_future = datetime.now(timezone.utc) + timedelta(days=30)
    assert apply_decay(0.6, in_the_future, half_life_days=14) == 0.6


# ----------------------------------------------------------------------
# CacheEntry
# ----------------------------------------------------------------------


def test_cache_entry_minimal_construction() -> None:
    entry = CacheEntry(
        pattern_sig="acme:pricing_shock:v1",
        pattern_class="pricing_shock",
        conclusion={"label": "Pricing Shock", "score": 0.84},
        confidence=0.84,
    )
    assert entry.pattern_sig == "acme:pricing_shock:v1"
    assert entry.reasoning_steps == []
    assert entry.boundary_conditions == {}
    assert entry.falsification_conditions == []
    assert entry.uncertainty_sources == []
    assert entry.vendor_name is None
    assert entry.decay_half_life_days == 90
    assert entry.validation_count == 1
    assert entry.effective_confidence is None


def test_cache_entry_default_lists_are_independent() -> None:
    # Regression guard: dataclass field(default_factory=list) means
    # each instance gets its own list -- mutation should not bleed.
    a = CacheEntry(pattern_sig="a", pattern_class="x", conclusion={}, confidence=0.5)
    b = CacheEntry(pattern_sig="b", pattern_class="x", conclusion={}, confidence=0.5)
    a.reasoning_steps.append({"step": 1})
    assert b.reasoning_steps == []


# ----------------------------------------------------------------------
# row_to_cache_entry
# ----------------------------------------------------------------------


def _base_row(**overrides) -> dict:
    """Build a Postgres-shaped row mapping with sensible defaults."""
    row = {
        "pattern_sig": "vendor_x:pricing_shock:v1",
        "pattern_class": "pricing_shock",
        "vendor_name": "Vendor X",
        "product_category": "billing",
        "conclusion": {"label": "Pricing Shock", "score": 0.84},
        "confidence": 0.84,
        "reasoning_steps": [{"step": "evidence", "value": 0.84}],
        "boundary_conditions": {"min_reviews": 50},
        "falsification_conditions": ["recommend_ratio > 0.5"],
        "uncertainty_sources": ["small_sample"],
        "decay_half_life_days": 60,
        "conclusion_type": "archetype",
        "evidence_hash": "abc123def4567890",
        "created_at": datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        "last_validated_at": datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
        "validation_count": 3,
    }
    row.update(overrides)
    return row


def test_row_to_cache_entry_handles_pre_decoded_jsonb() -> None:
    entry = row_to_cache_entry(_base_row())
    assert entry.pattern_sig == "vendor_x:pricing_shock:v1"
    assert entry.conclusion == {"label": "Pricing Shock", "score": 0.84}
    assert entry.reasoning_steps == [{"step": "evidence", "value": 0.84}]
    assert entry.boundary_conditions == {"min_reviews": 50}
    assert entry.falsification_conditions == ["recommend_ratio > 0.5"]
    assert entry.uncertainty_sources == ["small_sample"]
    assert entry.validation_count == 3


def test_row_to_cache_entry_handles_string_shaped_jsonb() -> None:
    # A vanilla adapter that hasn't installed the json codec returns
    # JSONB columns as raw strings -- the coercion must handle this.
    row = _base_row(
        conclusion=json.dumps({"label": "Pricing Shock", "score": 0.84}),
        reasoning_steps=json.dumps([{"step": "evidence"}]),
        boundary_conditions=json.dumps({"min_reviews": 50}),
        falsification_conditions=json.dumps(["recommend_ratio > 0.5"]),
    )
    entry = row_to_cache_entry(row)
    assert entry.conclusion == {"label": "Pricing Shock", "score": 0.84}
    assert entry.reasoning_steps == [{"step": "evidence"}]
    assert entry.boundary_conditions == {"min_reviews": 50}
    assert entry.falsification_conditions == ["recommend_ratio > 0.5"]


def test_row_to_cache_entry_normalizes_dict_shaped_falsification() -> None:
    # Older rows persisted falsification_conditions as a dict
    # (label -> condition); coerce to a flat list of values.
    row = _base_row(falsification_conditions={"a": "cond1", "b": "cond2"})
    entry = row_to_cache_entry(row)
    assert sorted(entry.falsification_conditions) == ["cond1", "cond2"]


def test_row_to_cache_entry_handles_empty_falsification_dict() -> None:
    row = _base_row(falsification_conditions={})
    entry = row_to_cache_entry(row)
    assert entry.falsification_conditions == []


def test_row_to_cache_entry_handles_null_uncertainty_sources() -> None:
    row = _base_row(uncertainty_sources=None)
    entry = row_to_cache_entry(row)
    assert entry.uncertainty_sources == []


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------


def test_stale_threshold_is_documented_value() -> None:
    # Storage adapters depend on this constant; pin it so changes are
    # explicit. Originally 0.5 in atlas_brain.reasoning.semantic_cache.
    assert STALE_THRESHOLD == 0.5
