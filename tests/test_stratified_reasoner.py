"""Smoke tests for the stratified reasoning engine (Phases 1-3).

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


# ---------------------------------------------------------------------------
# Phase 3 tests: Temporal reasoning + Churn archetypes
# ---------------------------------------------------------------------------


class TestTemporalEngine:
    """Pure-logic tests for temporal.py (no DB)."""

    def test_velocity_basic(self):
        from datetime import date

        from atlas_brain.reasoning.temporal import TemporalEngine

        engine = TemporalEngine(pool=None)
        snapshots = [
            {"snapshot_date": date(2026, 3, 1), "churn_density": 0.3, "avg_urgency": 5.0},
            {"snapshot_date": date(2026, 3, 8), "churn_density": 0.6, "avg_urgency": 7.0},
        ]
        velocities = engine._compute_velocities("TestVendor", snapshots)
        assert len(velocities) == 2
        cd = next(v for v in velocities if v.metric == "churn_density")
        assert abs(cd.velocity - (0.3 / 7)) < 0.001  # 0.0429/day
        assert cd.acceleration is None  # need 3+ snapshots

    def test_velocity_with_acceleration(self):
        from datetime import date

        from atlas_brain.reasoning.temporal import TemporalEngine

        engine = TemporalEngine(pool=None)
        snapshots = [
            {"snapshot_date": date(2026, 3, 1), "avg_urgency": 4.0},
            {"snapshot_date": date(2026, 3, 8), "avg_urgency": 5.0},
            {"snapshot_date": date(2026, 3, 15), "avg_urgency": 7.0},
        ]
        velocities = engine._compute_velocities("TestVendor", snapshots)
        v = next(v for v in velocities if v.metric == "avg_urgency")
        assert v.velocity > 0  # urgency increasing
        assert v.acceleration is not None
        assert v.acceleration > 0  # accelerating

    def test_velocity_insufficient_data(self):
        from atlas_brain.reasoning.temporal import TemporalEngine

        engine = TemporalEngine(pool=None)
        snapshots = [{"snapshot_date": None, "churn_density": 0.5}]
        velocities = engine._compute_velocities("TestVendor", snapshots)
        assert velocities == []

    def test_recency_weight(self):
        from atlas_brain.reasoning.temporal import TemporalEngine

        assert TemporalEngine.recency_weight(0) == 1.0
        assert abs(TemporalEngine.recency_weight(14) - 0.5) < 0.001  # half-life
        assert abs(TemporalEngine.recency_weight(28) - 0.25) < 0.001

    def test_evidence_dict_insufficient(self):
        from atlas_brain.reasoning.temporal import TemporalEngine, TemporalEvidence

        ev = TemporalEvidence(vendor_name="X", snapshot_days=1, insufficient_data=True)
        d = TemporalEngine.to_evidence_dict(ev)
        assert d["temporal_status"] == "insufficient_data"
        assert d["snapshot_days"] == 1

    def test_evidence_dict_with_velocities(self):
        from atlas_brain.reasoning.temporal import (
            TemporalEngine,
            TemporalEvidence,
            VendorVelocity,
        )

        ev = TemporalEvidence(
            vendor_name="X",
            snapshot_days=7,
            velocities=[
                VendorVelocity(
                    vendor_name="X", metric="churn_density",
                    current_value=0.6, previous_value=0.3,
                    velocity=0.0429, days_between=7, acceleration=0.01,
                ),
            ],
        )
        d = TemporalEngine.to_evidence_dict(ev)
        assert "velocity_churn_density" in d
        assert "accel_churn_density" in d
        assert d["snapshot_days"] == 7

    def test_normal_sf(self):
        from atlas_brain.reasoning.temporal import _normal_sf

        # sf(0) should be ~0.5
        assert abs(_normal_sf(0) - 0.5) < 0.01
        # sf(2) should be ~0.023
        assert abs(_normal_sf(2.0) - 0.0228) < 0.005
        # sf(-2) should be ~0.977
        assert abs(_normal_sf(-2.0) - 0.977) < 0.005


class TestArchetypes:
    """Pure-logic tests for archetypes.py (no DB)."""

    def test_all_archetypes_defined(self):
        from atlas_brain.reasoning.archetypes import ARCHETYPES

        expected = {
            "pricing_shock", "feature_gap", "acquisition_decay",
            "leadership_redesign", "integration_break", "support_collapse",
            "category_disruption", "compliance_gap",
        }
        assert set(ARCHETYPES.keys()) == expected

    def test_pricing_shock_high_signals(self):
        from atlas_brain.reasoning.archetypes import best_match

        evidence = {
            "avg_urgency": 7.5,
            "top_pain": "pricing is way too expensive after renewal hike",
            "competitor_count": 5,
            "recommend_ratio": 0.25,
            "displacement_edge_count": 4,
            "positive_review_pct": 28.0,
            "churn_density": 0.8,
        }
        m = best_match(evidence)
        assert m is not None
        assert m.archetype == "pricing_shock"
        assert m.score > 0.5

    def test_feature_gap_signals(self):
        from atlas_brain.reasoning.archetypes import best_match

        evidence = {
            "top_pain": "missing SSO, lack of API, need webhook support",
            "competitor_count": 4,
            "displacement_edge_count": 3,
            "pain_count": 6,
            "recommend_ratio": 0.4,
            "avg_urgency": 4.0,
        }
        m = best_match(evidence)
        assert m is not None
        assert m.archetype == "feature_gap"

    def test_support_collapse_signals(self):
        from atlas_brain.reasoning.archetypes import best_match

        evidence = {
            "top_pain": "support response time is terrible, no response for weeks",
            "avg_urgency": 8.0,
            "positive_review_pct": 22.0,
            "recommend_ratio": 0.2,
            "churn_density": 0.85,
        }
        m = best_match(evidence)
        assert m is not None
        assert m.archetype == "support_collapse"

    def test_compliance_gap_signals(self):
        from atlas_brain.reasoning.archetypes import best_match

        evidence = {
            "top_pain": "no SOC2 certification, GDPR compliance missing",
            "high_intent_company_count": 5,
            "avg_urgency": 6.0,
            "churn_density": 0.6,
        }
        m = best_match(evidence)
        assert m is not None
        assert m.archetype == "compliance_gap"

    def test_no_match_below_threshold(self):
        from atlas_brain.reasoning.archetypes import best_match

        evidence = {
            "avg_urgency": 2.0,
            "positive_review_pct": 85.0,
            "recommend_ratio": 0.9,
            "competitor_count": 0,
            "pain_count": 0,
        }
        m = best_match(evidence)
        # Should either be None or very low score
        if m is not None:
            assert m.score < 0.3

    def test_velocity_bonus(self):
        from atlas_brain.reasoning.archetypes import score_evidence

        base = {
            "avg_urgency": 7.0,
            "top_pain": "pricing increase is outrageous",
            "competitor_count": 4,
            "recommend_ratio": 0.3,
            "displacement_edge_count": 3,
            "positive_review_pct": 35.0,
        }
        # Without velocity
        scores_no_vel = score_evidence(base)
        ps_no = next(m for m in scores_no_vel if m.archetype == "pricing_shock")

        # With velocity signals matching pricing_shock hints
        temporal = {
            "velocity_avg_urgency": 0.5,       # increasing
            "velocity_competitor_count": 0.3,   # increasing
            "velocity_recommend_ratio": -0.05,  # decreasing
        }
        scores_vel = score_evidence(base, temporal)
        ps_vel = next(m for m in scores_vel if m.archetype == "pricing_shock")

        assert ps_vel.score > ps_no.score

    def test_anomaly_bonus(self):
        from atlas_brain.reasoning.archetypes import score_evidence

        base = {
            "avg_urgency": 7.0,
            "top_pain": "way too expensive after the price hike",
            "competitor_count": 4,
            "recommend_ratio": 0.3,
        }
        temporal_with_anom = {
            "anomalies": [
                {"metric": "avg_urgency", "z_score": 2.5, "value": 7.0},
                {"metric": "competitor_count", "z_score": 2.1, "value": 4},
            ],
        }
        scores_no = score_evidence(base)
        scores_anom = score_evidence(base, temporal_with_anom)
        ps_no = next(m for m in scores_no if m.archetype == "pricing_shock")
        ps_anom = next(m for m in scores_anom if m.archetype == "pricing_shock")
        assert ps_anom.score > ps_no.score

    def test_enrich_evidence(self):
        from atlas_brain.reasoning.archetypes import enrich_evidence_with_archetypes

        evidence = {
            "avg_urgency": 7.5,
            "top_pain": "price hike is killing us",
            "competitor_count": 5,
            "recommend_ratio": 0.2,
            "displacement_edge_count": 4,
            "positive_review_pct": 25.0,
        }
        enriched = enrich_evidence_with_archetypes(evidence)
        assert "archetype_scores" in enriched
        assert len(enriched["archetype_scores"]) > 0
        top = enriched["archetype_scores"][0]
        assert "archetype" in top
        assert "signal_score" in top
        assert "matched_signals" in top

    def test_falsification_conditions(self):
        from atlas_brain.reasoning.archetypes import get_falsification_conditions

        conds = get_falsification_conditions("pricing_shock")
        assert len(conds) == 3
        assert any("pricing" in c.lower() or "price" in c.lower() for c in conds)

        assert get_falsification_conditions("nonexistent") == []

    def test_top_matches_limit(self):
        from atlas_brain.reasoning.archetypes import top_matches

        evidence = {
            "avg_urgency": 8.0,
            "top_pain": "support is terrible, pricing too high, missing features, no compliance",
            "competitor_count": 6,
            "displacement_edge_count": 5,
            "high_intent_company_count": 4,
            "positive_review_pct": 20.0,
            "recommend_ratio": 0.15,
            "churn_density": 0.9,
            "pain_count": 8,
        }
        matches = top_matches(evidence, limit=3)
        assert len(matches) <= 3
        # Scores should be descending
        for i in range(len(matches) - 1):
            assert matches[i].score >= matches[i + 1].score
