"""Smoke tests for the stratified reasoning engine (Phase 1).

Tests semantic cache (Postgres) and episodic store (Neo4j) end-to-end.
Requires running Postgres (port 5433) and Neo4j (port 7687).
"""

import asyncio
import json
import pytest

from atlas_brain.reasoning.semantic_cache import (
    CacheEntry,
    SemanticCache,
    _apply_decay,
    compute_evidence_hash,
)
from atlas_brain.reasoning.episodic_store import (
    ConclusionNode,
    EpisodicStore,
    EvidenceNode,
    ReasoningTrace,
)

TEST_SIG = "test_smoke:deadbeef"
TEST_GROUP = "test-reasoning"


# ---------------------------------------------------------------------------
# Pure logic tests (no DB required)
# ---------------------------------------------------------------------------


class TestPureLogic:
    def test_evidence_hash_deterministic(self):
        h1 = compute_evidence_hash({"a": 1, "b": 2})
        h2 = compute_evidence_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_evidence_hash_changes(self):
        h1 = compute_evidence_hash({"a": 1})
        h2 = compute_evidence_hash({"a": 2})
        assert h1 != h2

    def test_decay_fresh(self):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        assert abs(_apply_decay(1.0, now, 90) - 1.0) < 0.001

    def test_decay_half_life(self):
        from datetime import datetime, timedelta, timezone

        past = datetime.now(timezone.utc) - timedelta(days=90)
        assert abs(_apply_decay(1.0, past, 90) - 0.5) < 0.001

    def test_decay_two_half_lives(self):
        from datetime import datetime, timedelta, timezone

        past = datetime.now(timezone.utc) - timedelta(days=180)
        assert abs(_apply_decay(1.0, past, 90) - 0.25) < 0.001


# ---------------------------------------------------------------------------
# Integration tests (require Postgres + Neo4j)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_pool():
    from atlas_brain.storage.database import DatabasePool

    pool = DatabasePool()
    await pool.initialize()
    yield pool
    # Cleanup test data
    await pool.execute(
        "DELETE FROM reasoning_semantic_cache WHERE pattern_sig = $1", TEST_SIG
    )
    await pool.execute(
        "DELETE FROM reasoning_metacognition WHERE period_start > '2099-01-01'"
    )
    await pool.close()


@pytest.fixture
async def episodic():
    store = EpisodicStore()
    await store.ensure_indexes()
    yield store
    await store.delete_by_group(TEST_GROUP)
    await store.close()


@pytest.mark.asyncio
async def test_semantic_cache_store_and_recall(db_pool):
    cache = SemanticCache(db_pool)
    entry = CacheEntry(
        pattern_sig=TEST_SIG,
        pattern_class="pricing_shock",
        vendor_name="Test Vendor",
        conclusion={"archetype": "pricing_shock"},
        confidence=0.9,
        evidence_hash="abc123",
    )
    await cache.store(entry)

    hit = await cache.lookup(TEST_SIG)
    assert hit is not None
    assert hit.confidence == 0.9
    assert hit.effective_confidence > 0.85
    assert hit.conclusion["archetype"] == "pricing_shock"


@pytest.mark.asyncio
async def test_semantic_cache_invalidation(db_pool):
    cache = SemanticCache(db_pool)
    entry = CacheEntry(
        pattern_sig=TEST_SIG,
        pattern_class="test",
        conclusion={"test": True},
        confidence=0.9,
    )
    await cache.store(entry)
    assert await cache.lookup(TEST_SIG) is not None

    await cache.invalidate(TEST_SIG, reason="test")
    assert await cache.lookup(TEST_SIG) is None


@pytest.mark.asyncio
async def test_semantic_cache_validation_bump(db_pool):
    cache = SemanticCache(db_pool)
    entry = CacheEntry(
        pattern_sig=TEST_SIG,
        pattern_class="test",
        conclusion={"test": True},
        confidence=0.7,
    )
    await cache.store(entry)
    await cache.validate(TEST_SIG, new_confidence=0.95)

    hit = await cache.lookup(TEST_SIG)
    assert hit is not None
    assert hit.confidence == 0.95
    assert hit.validation_count == 2


@pytest.mark.asyncio
async def test_episodic_store_roundtrip(episodic):
    trace = ReasoningTrace(
        vendor_name="Test Vendor",
        category="CRM",
        conclusion_type="pricing_shock",
        confidence=0.85,
        pattern_sig=TEST_SIG,
        trace_embedding=[0.1] * 1024,
        evidence=[
            EvidenceNode(type="review", source="g2", value="Too expensive"),
        ],
        conclusions=[
            ConclusionNode(claim="Pricing shock detected", confidence=0.85),
        ],
    )
    # Override group_id for test isolation
    import atlas_brain.reasoning.episodic_store as es
    original_group = es.GROUP_ID
    es.GROUP_ID = TEST_GROUP
    try:
        trace_id = await episodic.store_trace(trace)
        assert trace_id

        loaded = await episodic.get_trace(trace_id)
        assert loaded is not None
        assert loaded.vendor_name == "Test Vendor"
        assert len(loaded.evidence) == 1
        assert len(loaded.conclusions) == 1

        vendor_traces = await episodic.get_traces_for_vendor("Test Vendor")
        assert len(vendor_traces) >= 1
    finally:
        es.GROUP_ID = original_group


@pytest.mark.asyncio
async def test_cache_stats(db_pool):
    cache = SemanticCache(db_pool)
    stats = await cache.get_cache_stats()
    assert "active" in stats
    assert "invalidated" in stats


# ---------------------------------------------------------------------------
# Phase 2 tests: Differential engine, metacognition, tiers
# ---------------------------------------------------------------------------


class TestDifferentialEngine:
    def test_identical_evidence(self):
        from atlas_brain.reasoning.differential import classify_evidence

        old = {"a": 1, "b": 2}
        diff = classify_evidence(old, old)
        assert diff.diff_ratio == 0.0
        assert diff.should_reconstitute

    def test_large_delta_triggers_full_reason(self):
        from atlas_brain.reasoning.differential import classify_evidence

        old = {"a": 1, "b": 2}
        new = {"a": 99, "b": 99, "c": "new"}
        diff = classify_evidence(old, new)
        assert not diff.should_reconstitute
        assert len(diff.contradicted) == 2
        assert len(diff.novel) == 1

    def test_small_numeric_within_tolerance(self):
        from atlas_brain.reasoning.differential import classify_evidence

        old = {"score": 7.0}
        new = {"score": 7.3}  # 4.3% change, within 5% tolerance
        diff = classify_evidence(old, new)
        assert len(diff.confirmed) == 1

    def test_missing_evidence(self):
        from atlas_brain.reasoning.differential import classify_evidence

        old = {"a": 1, "b": 2, "c": 3}
        new = {"a": 1, "b": 2}
        diff = classify_evidence(old, new)
        assert len(diff.missing) == 1
        assert "c" in diff.missing

    def test_list_jaccard_similarity(self):
        from atlas_brain.reasoning.differential import classify_evidence

        old = {"items": ["a", "b", "c", "d", "e"]}
        new = {"items": ["a", "b", "c", "d", "e", "f"]}  # Jaccard 5/6 > 0.8
        diff = classify_evidence(old, new)
        assert len(diff.confirmed) == 1


class TestTiers:
    def test_tier_ordering(self):
        from atlas_brain.reasoning.tiers import Tier

        assert Tier.VENDOR_STATE < Tier.VENDOR_ARCHETYPE < Tier.CATEGORY_PATTERN < Tier.MARKET_DYNAMICS

    def test_pattern_sig_building(self):
        from atlas_brain.reasoning.tiers import Tier, build_tiered_pattern_sig

        sig = build_tiered_pattern_sig(Tier.VENDOR_ARCHETYPE, "Slack", "abc")
        assert sig == "t2:vendor:slack:abc"

    def test_refresh_needed(self):
        from datetime import datetime, timedelta, timezone

        from atlas_brain.reasoning.tiers import Tier, needs_refresh

        now = datetime.now(timezone.utc)
        assert needs_refresh(Tier.VENDOR_STATE, now - timedelta(hours=25))
        assert not needs_refresh(Tier.VENDOR_STATE, now - timedelta(hours=12))

    def test_inheritance_chain(self):
        from atlas_brain.reasoning.tiers import Tier, get_tier_config

        cfg = get_tier_config(Tier.VENDOR_STATE)
        assert Tier.MARKET_DYNAMICS in cfg.inherits_from
        assert get_tier_config(Tier.MARKET_DYNAMICS).inherits_from == []


class TestMetacognition:
    def test_recording(self):
        from atlas_brain.reasoning.metacognition import MetacognitiveMonitor

        m = MetacognitiveMonitor(pool=None)
        m.record("recall", 0, "pricing_shock")
        m.record("reconstitute", 600, "feature_gap")
        m.record("reason", 2000, "support_collapse")
        assert m._state.total_queries == 3
        assert m._state.recall_hits == 1
        assert m._state.reconstitute_hits == 1
        assert m._state.full_reasons == 1
        assert m._state.tokens_saved == 2000 + 1400  # recall + reconstitute

    def test_exploration_rate(self):
        from atlas_brain.reasoning.metacognition import MetacognitiveMonitor

        m = MetacognitiveMonitor(pool=None)
        # Run 1000 checks, expect ~12% to trigger
        triggered = sum(1 for _ in range(1000) if m.should_force_exploration())
        assert 60 < triggered < 200, f"Expected ~120 triggers, got {triggered}"
