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

# Lazy-import episodic store types to avoid neo4j -> numpy __version__
# AttributeError that can occur during pytest collection depending on
# import ordering.  The types are simple dataclasses and only needed by
# the integration tests that already skip when Neo4j is unavailable.
try:
    from atlas_brain.reasoning.episodic_store import (
        ConclusionNode,
        EpisodicStore,
        EvidenceNode,
        ReasoningTrace,
    )
except (ImportError, AttributeError):
    EpisodicStore = None  # type: ignore[misc,assignment]
    ReasoningTrace = None  # type: ignore[misc,assignment]
    EvidenceNode = None  # type: ignore[misc,assignment]
    ConclusionNode = None  # type: ignore[misc,assignment]

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

    def test_reasoning_validation_accepts_grounded_output(self):
        from atlas_brain.reasoning.stratified_reasoner import StratifiedReasoner

        conclusion = StratifiedReasoner._normalize_conclusion({
            "archetype": "pricing_shock",
            "secondary_archetype": None,
            "confidence": 0.82,
            "risk_level": "high",
            "executive_summary": "Pricing shock is the dominant pattern. It is surfacing now because complaint rates and urgency are rising together. Buyers should watch whether churn density normalizes over the next two snapshots.",
            "key_signals": [
                "churn_density: 38.6%",
                "avg_urgency: 7.4",
            ],
            "falsification_conditions": [
                "churn_density falls below 20% for two consecutive snapshots",
            ],
            "uncertainty_sources": [
                "limited temporal depth",
            ],
        })
        errors = StratifiedReasoner._validate_conclusion(
            conclusion,
            evidence_keys={"churn_density", "avg_urgency", "snapshot_days"},
        )
        assert errors == []

    def test_reasoning_validation_rejects_parse_fallback(self):
        from atlas_brain.reasoning.stratified_reasoner import StratifiedReasoner

        errors = StratifiedReasoner._validate_conclusion({
            "_parse_fallback": True,
            "analysis_text": "not json",
        })
        assert "llm output was not valid json" in errors

    def test_reasoning_validation_rejects_ungrounded_signals(self):
        from atlas_brain.reasoning.stratified_reasoner import StratifiedReasoner

        conclusion = StratifiedReasoner._normalize_conclusion({
            "archetype": "feature_gap",
            "secondary_archetype": None,
            "confidence": 0.76,
            "risk_level": "medium",
            "executive_summary": "Feature gap is the likeliest pattern. It is appearing now because missing capability mentions are rising. Buyers should watch whether competitor mentions keep clustering around the same gap.",
            "key_signals": ["high churn"],
            "falsification_conditions": ["feature gap mentions stop appearing in new reviews"],
            "uncertainty_sources": ["review sample may lag product releases"],
        })
        errors = StratifiedReasoner._validate_conclusion(
            conclusion,
            evidence_keys={"feature_gaps", "competitors"},
        )
        assert "key_signals must contain at least one grounded signal" in errors or (
            "key_signals do not reference known evidence metrics" in errors
        )


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


# ---------------------------------------------------------------------------
# Phase 4 tests: Knowledge graph + Trigger events
# ---------------------------------------------------------------------------


class TestKnowledgeGraphEntities:
    """Pure-logic tests for knowledge_graph.py entity definitions."""

    def test_vendor_node_defaults(self):
        from atlas_brain.reasoning.knowledge_graph import VendorNode

        v = VendorNode(canonical_name="Slack")
        assert v.canonical_name == "Slack"
        assert v.aliases == []
        assert v.churn_density == 0.0

    def test_displacement_edge(self):
        from atlas_brain.reasoning.knowledge_graph import DisplacementEdge

        e = DisplacementEdge(
            from_vendor="Mailchimp", to_vendor="Klaviyo",
            mention_count=28, primary_driver="pricing",
            signal_strength="strong", confidence_score=0.85,
        )
        assert e.from_vendor == "Mailchimp"
        assert e.primary_driver == "pricing"

    def test_integration_edge(self):
        from atlas_brain.reasoning.knowledge_graph import IntegrationEdge

        e = IntegrationEdge(
            vendor_name="HubSpot", integration_name="Salesforce",
            mention_count=15, confidence_score=0.9,
        )
        assert e.vendor_name == "HubSpot"
        assert e.integration_name == "Salesforce"

    def test_group_id(self):
        from atlas_brain.reasoning.knowledge_graph import GROUP_ID

        assert GROUP_ID == "b2b-knowledge-graph"
        # Must differ from episodic store group
        try:
            from atlas_brain.reasoning.episodic_store import GROUP_ID as EPISODIC_GID
            assert GROUP_ID != EPISODIC_GID
        except (ImportError, AttributeError):
            pytest.skip("episodic_store import unavailable (neo4j dependency)")


class TestTriggerEvents:
    """Pure-logic tests for trigger_events.py."""

    def test_all_trigger_types_defined(self):
        from atlas_brain.reasoning.trigger_events import EVENT_TAXONOMY, TriggerType

        assert len(EVENT_TAXONOMY) == 8
        for tt in TriggerType:
            assert tt in EVENT_TAXONOMY

    def test_archetype_affinity(self):
        from atlas_brain.reasoning.trigger_events import EVENT_TAXONOMY, TriggerType

        pricing = EVENT_TAXONOMY[TriggerType.PRICING_CHANGE]
        assert "pricing_shock" in pricing.archetype_affinity
        assert pricing.urgency_boost > 1.0

        outage = EVENT_TAXONOMY[TriggerType.OUTAGE_INCIDENT]
        assert "support_collapse" in outage.archetype_affinity
        assert outage.urgency_boost == 2.0  # highest urgency

    def test_get_archetype_triggers(self):
        from atlas_brain.reasoning.trigger_events import (
            TriggerType,
            get_archetype_triggers,
        )

        triggers = get_archetype_triggers("pricing_shock")
        assert TriggerType.PRICING_CHANGE in triggers
        assert TriggerType.CONTRACT_CYCLE in triggers

        triggers = get_archetype_triggers("compliance_gap")
        assert TriggerType.COMPLIANCE_UPDATE in triggers

    def test_map_event_type(self):
        from atlas_brain.reasoning.trigger_events import TriggerType, _map_event_type

        assert _map_event_type("Series B funding round") == TriggerType.FUNDING_ROUND
        assert _map_event_type("CEO departure announced") == TriggerType.LEADERSHIP_CHANGE
        assert _map_event_type("SOC2 certification achieved") == TriggerType.COMPLIANCE_UPDATE
        assert _map_event_type("v3.0 feature release") == TriggerType.PRODUCT_LAUNCH
        assert _map_event_type("pricing tier restructure") == TriggerType.PRICING_CHANGE
        assert _map_event_type("acquired by Salesforce") == TriggerType.ACQUISITION
        assert _map_event_type("major outage 12h downtime") == TriggerType.OUTAGE_INCIDENT
        assert _map_event_type("annual contract renewal") == TriggerType.CONTRACT_CYCLE
        assert _map_event_type("random unrelated text") is None

    def test_correlation_strength_classification(self):
        from atlas_brain.reasoning.trigger_events import _classify_correlation_strength

        # avg_urgency thresholds: (0.05, 0.15, 0.3)
        assert _classify_correlation_strength(0.35, "avg_urgency") == "strong"
        assert _classify_correlation_strength(0.2, "avg_urgency") == "moderate"
        assert _classify_correlation_strength(0.08, "avg_urgency") == "weak"
        assert _classify_correlation_strength(0.01, "avg_urgency") == "none"

    def test_risk_level_from_score(self):
        from atlas_brain.reasoning.trigger_events import _risk_level_from_score

        assert _risk_level_from_score(0.9) == "critical"
        assert _risk_level_from_score(0.7) == "high"
        assert _risk_level_from_score(0.4) == "medium"
        assert _risk_level_from_score(0.2) == "low"

    def test_trigger_event_creation(self):
        from datetime import date

        from atlas_brain.reasoning.trigger_events import TriggerEvent, TriggerType

        ev = TriggerEvent(
            vendor_name="Slack",
            trigger_type=TriggerType.PRICING_CHANGE,
            event_date=date(2026, 3, 1),
            source="press",
            description="Slack raises Pro plan price by 30%",
            confidence=0.9,
        )
        assert ev.vendor_name == "Slack"
        assert ev.trigger_type == TriggerType.PRICING_CHANGE

    def test_event_taxonomy_serializable(self):
        from atlas_brain.reasoning.trigger_events import get_event_taxonomy

        taxonomy = get_event_taxonomy()
        assert isinstance(taxonomy, dict)
        assert "pricing_change" in taxonomy
        assert "outage_incident" in taxonomy

    @pytest.mark.asyncio
    async def test_correlator_no_temporal(self):
        """Correlator returns None when no temporal data provided."""
        from datetime import date

        from atlas_brain.reasoning.trigger_events import (
            TriggerCorrelator,
            TriggerEvent,
            TriggerType,
        )

        correlator = TriggerCorrelator(pool=None)
        event = TriggerEvent(
            vendor_name="Test",
            trigger_type=TriggerType.PRICING_CHANGE,
            event_date=date.today(),
        )
        result = await correlator.correlate_event(event, temporal_dict=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_correlator_with_velocity(self):
        """Correlator detects correlation when velocity matches expected direction."""
        from datetime import date

        from atlas_brain.reasoning.trigger_events import (
            TriggerCorrelator,
            TriggerEvent,
            TriggerType,
        )

        correlator = TriggerCorrelator(pool=None)
        event = TriggerEvent(
            vendor_name="Test",
            trigger_type=TriggerType.PRICING_CHANGE,
            event_date=date.today(),
            confidence=0.9,
        )
        temporal = {
            "velocity_avg_urgency": 0.5,  # urgency increasing (expected for pricing)
            "velocity_competitor_count": 0.3,  # competitors increasing
        }
        result = await correlator.correlate_event(event, temporal_dict=temporal)
        assert result is not None
        assert result.correlation_strength in ("strong", "moderate", "weak")

    @pytest.mark.asyncio
    async def test_composite_risk_score(self):
        """Composite risk combines archetype score + event boosts."""
        from datetime import date

        from atlas_brain.reasoning.trigger_events import (
            TriggerCorrelator,
            TriggerEvent,
            TriggerType,
        )

        correlator = TriggerCorrelator(pool=None)
        events = [
            TriggerEvent(
                vendor_name="Test",
                trigger_type=TriggerType.PRICING_CHANGE,
                event_date=date.today(),
                confidence=0.9,
            ),
            TriggerEvent(
                vendor_name="Test",
                trigger_type=TriggerType.OUTAGE_INCIDENT,
                event_date=date.today(),
                confidence=0.8,
            ),
        ]
        temporal = {"velocity_avg_urgency": 0.4}

        score = await correlator.compute_composite_risk(
            "Test", events, archetype_score=0.6, temporal_dict=temporal,
        )
        assert score.composite_risk > 0.6  # should be boosted above base
        assert score.risk_level in ("medium", "high", "critical")
        assert len(score.events) == 2


# ---------------------------------------------------------------------------
# Phase 5 tests: Ecosystem + Narrative
# ---------------------------------------------------------------------------


class TestEcosystem:
    """Pure-logic tests for ecosystem.py."""

    def test_category_health_defaults(self):
        from atlas_brain.reasoning.ecosystem import CategoryHealth

        h = CategoryHealth(category="CRM")
        assert h.category == "CRM"
        assert h.hhi == 0.0
        assert h.market_structure == ""

    def test_hhi_computation(self):
        from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer

        analyzer = EcosystemAnalyzer(pool=None)
        # Two vendors with equal reviews -> HHI = 2 * 50^2 = 5000
        vendors = [
            {"vendor_name": "A", "total_reviews": 100, "churn_density": 30,
             "avg_urgency": 4.0, "positive_review_pct": 60,
             "displacement_edge_count": 1},
            {"vendor_name": "B", "total_reviews": 100, "churn_density": 40,
             "avg_urgency": 5.0, "positive_review_pct": 50,
             "displacement_edge_count": 2},
        ]
        health = analyzer._compute_health("CRM", vendors)
        assert health.vendor_count == 2
        assert health.total_reviews == 200
        assert abs(health.hhi - 5000.0) < 1  # 50^2 + 50^2

    def test_hhi_dominant_player(self):
        from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer

        analyzer = EcosystemAnalyzer(pool=None)
        # One dominant vendor (90% share) -> HHI = 90^2 + 10^2 = 8200
        vendors = [
            {"vendor_name": "Big", "total_reviews": 900, "churn_density": 20,
             "avg_urgency": 3.0, "positive_review_pct": 70,
             "displacement_edge_count": 0},
            {"vendor_name": "Small", "total_reviews": 100, "churn_density": 50,
             "avg_urgency": 6.0, "positive_review_pct": 30,
             "displacement_edge_count": 3},
        ]
        health = analyzer._compute_health("CRM", vendors)
        assert health.hhi > 8000  # concentrated market

    def test_market_structure_consolidating(self):
        from atlas_brain.reasoning.ecosystem import (
            CategoryHealth,
            CategoryVendorSlice,
            EcosystemAnalyzer,
        )

        analyzer = EcosystemAnalyzer(pool=None)
        health = CategoryHealth(
            category="CRM", vendor_count=3, hhi=3000,
            displacement_intensity=0.5, avg_churn_density=25,
        )
        slices = [
            CategoryVendorSlice(vendor_name="Big", review_share=50, net_displacement=2),
            CategoryVendorSlice(vendor_name="Med", review_share=30, net_displacement=0),
            CategoryVendorSlice(vendor_name="Small", review_share=20, net_displacement=-2),
        ]
        assert analyzer._classify_market(health, slices) == "consolidating"

    def test_market_structure_displacing(self):
        from atlas_brain.reasoning.ecosystem import (
            CategoryHealth,
            CategoryVendorSlice,
            EcosystemAnalyzer,
        )

        analyzer = EcosystemAnalyzer(pool=None)
        health = CategoryHealth(
            category="CRM", vendor_count=5, hhi=2000,
            displacement_intensity=3.0, avg_churn_density=50,
        )
        slices = [
            CategoryVendorSlice(vendor_name="Winner", review_share=25, net_displacement=5),
            CategoryVendorSlice(vendor_name="Loser", review_share=25, net_displacement=-5),
            CategoryVendorSlice(vendor_name="C", review_share=20, net_displacement=0),
            CategoryVendorSlice(vendor_name="D", review_share=15, net_displacement=1),
            CategoryVendorSlice(vendor_name="E", review_share=15, net_displacement=-1),
        ]
        assert analyzer._classify_market(health, slices) == "displacing"

    def test_market_structure_stable(self):
        from atlas_brain.reasoning.ecosystem import (
            CategoryHealth,
            CategoryVendorSlice,
            EcosystemAnalyzer,
        )

        analyzer = EcosystemAnalyzer(pool=None)
        health = CategoryHealth(
            category="CRM", vendor_count=4, hhi=2200,
            displacement_intensity=0.5, avg_churn_density=20,
        )
        slices = [
            CategoryVendorSlice(vendor_name="A", review_share=30, net_displacement=0),
            CategoryVendorSlice(vendor_name="B", review_share=28, net_displacement=0),
            CategoryVendorSlice(vendor_name="C", review_share=22, net_displacement=0),
            CategoryVendorSlice(vendor_name="D", review_share=20, net_displacement=0),
        ]
        assert analyzer._classify_market(health, slices) == "stable"

    def test_evidence_dict_serialization(self):
        from atlas_brain.reasoning.ecosystem import (
            CategoryHealth,
            CategoryVendorSlice,
            EcosystemAnalyzer,
            EcosystemEvidence,
        )

        ev = EcosystemEvidence(
            category="CRM",
            health=CategoryHealth(
                category="CRM", vendor_count=5, total_reviews=500,
                avg_churn_density=35.0, hhi=2000, market_structure="displacing",
                dominant_archetype="pricing_shock",
            ),
            vendor_slices=[
                CategoryVendorSlice(vendor_name="X", review_share=40, churn_density=50),
            ],
            pain_distribution={"pricing": 100, "support": 50},
            archetype_distribution={"pricing_shock": 3, "feature_gap": 2},
        )
        d = EcosystemAnalyzer.to_evidence_dict(ev)
        assert d["market_structure"] == "displacing"
        assert d["hhi"] == 2000
        assert d["dominant_archetype"] == "pricing_shock"
        assert "vendor_positions" in d
        assert "category_pains" in d


class TestNarrative:
    """Pure-logic tests for narrative.py."""

    def test_evidence_chain_basics(self):
        from atlas_brain.reasoning.narrative import EvidenceChain, EvidenceLink

        chain = EvidenceChain(
            claim="High churn density",
            confidence=0.8,
            vendor_name="TestCo",
            links=[
                EvidenceLink(source_type="snapshot", metric="churn_density", value=65.0),
                EvidenceLink(source_type="temporal", metric="velocity_churn_density", value=0.05),
            ],
        )
        assert chain.source_count == 2
        assert chain.is_well_supported

    def test_single_source_not_well_supported(self):
        from atlas_brain.reasoning.narrative import EvidenceChain, EvidenceLink

        chain = EvidenceChain(
            claim="Some claim",
            links=[EvidenceLink(source_type="snapshot", metric="x", value=1)],
        )
        assert chain.source_count == 1
        assert not chain.is_well_supported

    def test_rule_engine_triggers(self):
        from atlas_brain.reasoning.narrative import NarrativeEngine

        engine = NarrativeEngine()
        evidence = {
            "churn_density": 75.0,  # >= 70 -> critical_churn_density
            "avg_urgency": 8.0,  # >= 7 -> urgency_spike
            "displacement_edge_count": 3,  # < 5 -> no trigger
            "positive_review_pct": 20.0,  # < 25 -> positive_collapse
        }
        triggered = engine.evaluate_rules("TestVendor", evidence)
        names = [t.rule.name for t in triggered]
        assert "critical_churn_density" in names
        assert "urgency_spike" in names
        assert "positive_collapse" in names
        assert "mass_displacement" not in names

    def test_rule_engine_no_triggers(self):
        from atlas_brain.reasoning.narrative import NarrativeEngine

        engine = NarrativeEngine()
        evidence = {
            "churn_density": 20.0,
            "avg_urgency": 3.0,
            "positive_review_pct": 70.0,
        }
        triggered = engine.evaluate_rules("HealthyVendor", evidence)
        assert len(triggered) == 0

    def test_build_narrative(self):
        from atlas_brain.reasoning.narrative import NarrativeEngine

        engine = NarrativeEngine()
        narrative = engine.build_vendor_narrative(
            "Mailchimp",
            reasoning_result={
                "archetype": "pricing_shock",
                "executive_summary": "Mailchimp shows pricing shock signals",
                "risk_level": "high",
                "falsification_conditions": ["Price reverts"],
                "uncertainty_sources": ["Limited data"],
            },
            archetype_match={
                "archetype": "pricing_shock",
                "signal_score": 0.67,
                "risk_level": "high",
            },
            snapshot={"churn_density": 45.6, "avg_urgency": 4.3},
            temporal_dict={
                "snapshot_days": 2,
                "velocity_churn_density": 0.03,
                "velocity_avg_urgency": 0.1,
            },
            competitive_landscape={
                "losing_to": [
                    {"name": "Klaviyo", "mentions": 15, "driver": "pricing"},
                    {"name": "Brevo", "mentions": 8, "driver": "pricing"},
                ],
            },
        )
        assert narrative.vendor_name == "Mailchimp"
        assert narrative.archetype == "pricing_shock"
        assert narrative.archetype_score == 0.67
        assert len(narrative.evidence_chains) >= 3  # churn + competitive + urgency
        assert narrative.falsification_conditions == ["Price reverts"]

    def test_explainability(self):
        from atlas_brain.reasoning.narrative import (
            EvidenceChain,
            EvidenceLink,
            NarrativeEngine,
            VendorNarrative,
        )

        engine = NarrativeEngine()
        narrative = VendorNarrative(
            vendor_name="TestCo",
            archetype="feature_gap",
            archetype_score=0.6,
            risk_level="medium",
            evidence_chains=[
                EvidenceChain(
                    claim="Missing SSO",
                    confidence=0.8,
                    links=[
                        EvidenceLink(source_type="review", value="no SSO"),
                        EvidenceLink(source_type="displacement", value="3 mentions"),
                    ],
                ),
            ],
            temporal_context={"snapshot_days": 14},
            competitive_context={"losing_to": [{"name": "Competitor"}]},
            falsification_conditions=["Vendor releases SSO"],
        )
        explain = engine.build_explainability(narrative)
        assert explain["archetype"] == "feature_gap"
        assert explain["confidence_assessment"] in ("low", "medium", "high")
        assert "what_would_change_conclusion" in explain
        assert explain["evidence_summary"]["well_supported"] == 1

    def test_intelligence_payload(self):
        from atlas_brain.reasoning.narrative import NarrativeEngine, VendorNarrative

        engine = NarrativeEngine()
        narrative = engine.build_vendor_narrative(
            "Slack",
            reasoning_result={
                "archetype": "category_disruption",
                "risk_level": "high",
                "executive_summary": "AI-native tools disrupting Slack",
            },
            archetype_match={"signal_score": 0.55},
            snapshot={"churn_density": 30},
            temporal_dict={"snapshot_days": 7, "velocity_churn_density": 0.02},
            competitive_landscape={
                "losing_to": [{"name": "Teams", "mentions": 27}],
                "winning_from": [],
            },
            ecosystem_evidence={"market_structure": "displacing", "hhi": 1800},
        )
        payload = NarrativeEngine.to_intelligence_payload(narrative)
        assert payload["vendor_name"] == "Slack"
        assert payload["archetype"] == "category_disruption"
        assert payload["competitive"]["losing_to_count"] == 1
        assert payload["ecosystem"]["market_structure"] == "displacing"
