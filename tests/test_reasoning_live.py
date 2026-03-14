"""Live integration tests for Stratified Reasoning Architecture.

Hits real Postgres, real Neo4j, real LLM (DeepSeek via OpenRouter).
No mocks. Tests what the system actually does vs what it should do.

Run:
    python -m pytest tests/test_reasoning_live.py -v -s --tb=short

Requires:
    - Postgres running (ATLAS_DB_* env vars)
    - Neo4j running (bolt://localhost:7687)
    - OPENROUTER_API_KEY set in .env
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import asyncpg
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("test.reasoning.live")

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
async def pool():
    """Create a real asyncpg connection pool (fresh per test to avoid loop issues)."""
    from atlas_brain.storage.config import db_settings

    p = await asyncpg.create_pool(
        host=db_settings.host,
        port=db_settings.port,
        database=db_settings.database,
        user=db_settings.user,
        password=db_settings.password,
        min_size=1,
        max_size=3,
    )
    yield p
    await p.close()


@pytest.fixture
async def test_vendor(pool):
    """Pick the vendor with the most data for testing."""
    row = await pool.fetchrow("""
        SELECT cs.vendor_name, cs.total_reviews, cs.avg_urgency_score,
               cs.confidence_score
        FROM b2b_churn_signals cs
        ORDER BY cs.total_reviews DESC
        LIMIT 1
    """)
    assert row, "No vendors in b2b_churn_signals"
    name = row["vendor_name"]
    logger.info("Test vendor: %s (%d reviews, urgency=%.1f)",
                 name, row["total_reviews"], row["avg_urgency_score"] or 0)
    return name


@pytest.fixture
async def vendor_snapshot(pool, test_vendor):
    """Load the latest snapshot for the test vendor."""
    row = await pool.fetchrow("""
        SELECT * FROM b2b_vendor_snapshots
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY snapshot_date DESC
        LIMIT 1
    """, test_vendor)
    return dict(row) if row else {}


@pytest.fixture
async def vendor_evidence(pool, test_vendor, vendor_snapshot):
    """Build a complete evidence dict for the test vendor."""
    evidence = {}

    # Snapshot data
    if vendor_snapshot:
        evidence.update({
            "churn_density": vendor_snapshot.get("churn_density"),
            "avg_urgency": vendor_snapshot.get("avg_urgency"),
            "positive_review_pct": vendor_snapshot.get("positive_review_pct"),
            "recommend_ratio": vendor_snapshot.get("recommend_ratio"),
            "competitor_count": vendor_snapshot.get("competitor_count"),
            "displacement_edge_count": vendor_snapshot.get("displacement_edge_count"),
            "high_intent_company_count": vendor_snapshot.get("high_intent_company_count"),
            "pain_count": vendor_snapshot.get("pain_count"),
        })

    # Churn signals
    cs = await pool.fetchrow("""
        SELECT total_reviews, avg_urgency_score, confidence_score, product_category
        FROM b2b_churn_signals WHERE LOWER(vendor_name) = LOWER($1)
    """, test_vendor)
    if cs:
        evidence["total_reviews"] = cs["total_reviews"]
        evidence["avg_urgency_score"] = cs["avg_urgency_score"]
        evidence["confidence_score"] = cs["confidence_score"]
        evidence["product_category"] = cs["product_category"]

    # Pain points
    pains = await pool.fetch("""
        SELECT pain_category, mention_count
        FROM b2b_vendor_pain_points
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY mention_count DESC LIMIT 10
    """, test_vendor)
    evidence["top_pain_points"] = [
        {"category": r["pain_category"], "mentions": r["mention_count"]}
        for r in pains
    ]

    # Displacement edges
    disps = await pool.fetch("""
        SELECT to_vendor, mention_count, primary_driver
        FROM b2b_displacement_edges
        WHERE LOWER(from_vendor) = LOWER($1)
        ORDER BY mention_count DESC LIMIT 5
    """, test_vendor)
    evidence["losing_to"] = [
        {"name": r["to_vendor"], "mentions": r["mention_count"],
         "driver": r["primary_driver"]}
        for r in disps
    ]

    gains = await pool.fetch("""
        SELECT from_vendor, mention_count, primary_driver
        FROM b2b_displacement_edges
        WHERE LOWER(to_vendor) = LOWER($1)
        ORDER BY mention_count DESC LIMIT 5
    """, test_vendor)
    evidence["winning_from"] = [
        {"name": r["from_vendor"], "mentions": r["mention_count"],
         "driver": r["primary_driver"]}
        for r in gains
    ]

    logger.info("Evidence for %s: %d keys", test_vendor, len(evidence))
    return evidence


@pytest.fixture
def openrouter_llm():
    """Create an OpenRouter LLM with DeepSeek for cheap reasoning tests."""
    from atlas_brain.services.llm.openrouter import OpenRouterLLM

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    llm = OpenRouterLLM(
        model="deepseek/deepseek-chat-v3-0324",
        api_key=api_key,
    )
    llm.load()
    yield llm
    llm.unload()


# ------------------------------------------------------------------
# Test 1: Archetype scoring against real vendor data
# ------------------------------------------------------------------


class TestArchetypesLive:

    @pytest.mark.asyncio
    async def test_score_real_vendor(self, test_vendor, vendor_evidence):
        """Archetype scoring should produce at least one match for a real vendor."""
        from atlas_brain.reasoning.archetypes import score_evidence, best_match

        matches = score_evidence(vendor_evidence)

        logger.info("Archetype matches for %s:", test_vendor)
        for m in matches[:5]:
            logger.info(
                "  %s: score=%.3f risk=%s matched=%s",
                m.archetype, m.score, m.risk_level, m.matched_signals,
            )

        assert len(matches) > 0, "No archetype matches for a vendor with data"
        top = matches[0]
        assert top.score > 0, "Top match score should be positive"
        assert top.archetype in (
            "pricing_shock", "feature_gap", "acquisition_decay",
            "leadership_redesign", "integration_break", "support_collapse",
            "category_disruption", "compliance_gap",
        ), f"Unknown archetype: {top.archetype}"
        assert top.risk_level in ("low", "medium", "high", "critical")

    @pytest.mark.asyncio
    async def test_enrich_evidence(self, vendor_evidence):
        """enrich_evidence_with_archetypes should add archetype_scores key."""
        from atlas_brain.reasoning.archetypes import enrich_evidence_with_archetypes

        enriched = enrich_evidence_with_archetypes(vendor_evidence)
        assert "archetype_scores" in enriched
        assert len(enriched["archetype_scores"]) > 0
        top = enriched["archetype_scores"][0]
        assert "archetype" in top
        assert "signal_score" in top  # field is signal_score, not score

    @pytest.mark.asyncio
    async def test_all_vendors_score(self, pool):
        """Score every vendor - none should crash."""
        from atlas_brain.reasoning.archetypes import score_evidence

        vendors = await pool.fetch("""
            SELECT cs.vendor_name, snap.churn_density, snap.avg_urgency,
                   snap.positive_review_pct, snap.displacement_edge_count,
                   snap.competitor_count, snap.pain_count
            FROM b2b_churn_signals cs
            LEFT JOIN (
                SELECT DISTINCT ON (vendor_name) *
                FROM b2b_vendor_snapshots ORDER BY vendor_name, snapshot_date DESC
            ) snap ON LOWER(cs.vendor_name) = LOWER(snap.vendor_name)
        """)

        scored = 0
        for v in vendors:
            ev = {k: v[k] for k in v.keys() if v[k] is not None}
            matches = score_evidence(ev)
            scored += 1

        logger.info("Scored %d/%d vendors without error", scored, len(vendors))
        assert scored == len(vendors)


# ------------------------------------------------------------------
# Test 2: Temporal engine with real snapshot data
# ------------------------------------------------------------------


class TestTemporalLive:

    @pytest.mark.asyncio
    async def test_analyze_vendor(self, pool, test_vendor):
        """Temporal engine should handle real snapshot data (even if insufficient)."""
        from atlas_brain.reasoning.temporal import TemporalEngine

        engine = TemporalEngine(pool)
        result = await engine.analyze_vendor(test_vendor)

        logger.info("Temporal result for %s: insufficient=%s, snapshots=%d",
                     test_vendor, result.insufficient_data, result.snapshot_days)

        assert result.vendor_name == test_vendor
        # With only 1 snapshot, should say insufficient data
        if result.snapshot_days < 2:
            assert result.insufficient_data is True

    @pytest.mark.asyncio
    async def test_evidence_dict(self, pool, test_vendor):
        """to_evidence_dict should return a structured dict."""
        from atlas_brain.reasoning.temporal import TemporalEngine

        engine = TemporalEngine(pool)
        te = await engine.analyze_vendor(test_vendor)
        evidence = TemporalEngine.to_evidence_dict(te)

        logger.info("Temporal evidence dict: %s", json.dumps(evidence, indent=2, default=str))
        assert isinstance(evidence, dict)
        assert "temporal_status" in evidence or "snapshot_days" in evidence


# ------------------------------------------------------------------
# Test 3: Knowledge graph queries against live Neo4j
# ------------------------------------------------------------------


class TestKnowledgeGraphLive:

    @pytest.mark.asyncio
    async def test_graph_stats(self):
        """Knowledge graph should have real data from prior sync."""
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError:
            pytest.skip("neo4j driver not installed")

        driver = AsyncGraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "password123"),
        )
        try:
            from atlas_brain.reasoning.knowledge_graph import KnowledgeGraphQuery
            gq = KnowledgeGraphQuery(driver)
            stats = await gq.graph_stats()

            logger.info("Knowledge graph stats: %s", stats)
            assert stats.get("B2bVendor", 0) > 0, "No vendor nodes in Neo4j"
            assert stats.get("B2bPainPoint", 0) > 0, "No pain point nodes"
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_competitive_landscape(self, test_vendor):
        """Query competitive landscape for a real vendor."""
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError:
            pytest.skip("neo4j driver not installed")

        driver = AsyncGraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "password123"),
        )
        try:
            from atlas_brain.reasoning.knowledge_graph import KnowledgeGraphQuery
            gq = KnowledgeGraphQuery(driver)
            landscape = await gq.vendor_competitive_landscape(test_vendor)

            logger.info("Competitive landscape for %s:", test_vendor)
            logger.info("  losing_to: %s", landscape.get("losing_to", [])[:3])
            logger.info("  winning_from: %s", landscape.get("winning_from", [])[:3])
            logger.info("  shared_pains: %d", len(landscape.get("shared_pains", [])))

            assert isinstance(landscape, dict)
            assert "losing_to" in landscape
            assert "winning_from" in landscape
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_displacement_chain(self, test_vendor):
        """Multi-hop displacement chain query."""
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError:
            pytest.skip("neo4j driver not installed")

        driver = AsyncGraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "password123"),
        )
        try:
            from atlas_brain.reasoning.knowledge_graph import KnowledgeGraphQuery
            gq = KnowledgeGraphQuery(driver)
            chains = await gq.displacement_chain(test_vendor, max_hops=2)

            logger.info("Displacement chains for %s: %d paths", test_vendor, len(chains))
            for c in chains[:3]:
                logger.info("  %s", c)

            assert isinstance(chains, list)
        finally:
            await driver.close()


# ------------------------------------------------------------------
# Test 4: Trigger event correlation with real data
# ------------------------------------------------------------------


class TestTriggerEventsLive:

    @pytest.mark.asyncio
    async def test_load_recent_events(self, pool, test_vendor):
        """Load real trigger events from b2b_change_events."""
        from atlas_brain.reasoning.trigger_events import TriggerCorrelator

        correlator = TriggerCorrelator(pool)
        events = await correlator.load_recent_events(test_vendor, days=180)

        logger.info("Trigger events for %s: %d events", test_vendor, len(events))
        for e in events[:5]:
            logger.info("  %s: %s (%s)", e.trigger_type.value, e.description, e.event_date)

        assert isinstance(events, list)
        # May be empty if this vendor has no change events

    @pytest.mark.asyncio
    async def test_composite_risk_any_vendor(self, pool):
        """Find a vendor with change events and compute composite risk."""
        from atlas_brain.reasoning.trigger_events import TriggerCorrelator

        correlator = TriggerCorrelator(pool)

        # Find a vendor that has change events
        row = await pool.fetchrow("""
            SELECT DISTINCT vendor_name FROM b2b_change_events LIMIT 1
        """)
        if not row:
            pytest.skip("No change events in database")

        vendor = row["vendor_name"]
        events = await correlator.load_recent_events(vendor, days=365)
        assert len(events) > 0, f"No parsed events for {vendor}"

        risk = await correlator.compute_composite_risk(
            vendor, events, archetype_score=0.4,
        )

        logger.info("Composite risk for %s:", vendor)
        logger.info("  events: %d", len(risk.events))
        logger.info("  correlations: %d", len(risk.correlations))
        logger.info("  base_risk: %.3f", risk.base_risk)
        logger.info("  event_boost: %.3f", risk.event_boost)
        logger.info("  composite_risk: %.3f", risk.composite_risk)
        logger.info("  risk_level: %s", risk.risk_level)
        logger.info("  explanation: %s", risk.explanation)

        assert risk.composite_risk >= risk.base_risk, \
            "Composite should be >= base risk"
        assert risk.risk_level in ("low", "medium", "high", "critical")


# ------------------------------------------------------------------
# Test 5: Ecosystem analysis against real categories
# ------------------------------------------------------------------


class TestEcosystemLive:

    @pytest.mark.asyncio
    async def test_analyze_real_category(self, pool):
        """Full ecosystem analysis on a real product category."""
        from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer

        # Find a category with multiple vendors
        row = await pool.fetchrow("""
            SELECT product_category, COUNT(*) as cnt
            FROM b2b_churn_signals
            WHERE product_category IS NOT NULL AND product_category != ''
            GROUP BY product_category
            ORDER BY cnt DESC
            LIMIT 1
        """)
        if not row:
            pytest.skip("No product categories with vendors")

        category = row["product_category"]
        count = row["cnt"]
        logger.info("Testing ecosystem for category: %s (%d vendors)", category, count)

        analyzer = EcosystemAnalyzer(pool)
        evidence = await analyzer.analyze_category(category)

        logger.info("Ecosystem results for '%s':", category)
        logger.info("  vendor_count: %d", evidence.health.vendor_count)
        logger.info("  total_reviews: %d", evidence.health.total_reviews)
        logger.info("  avg_churn_density: %.2f", evidence.health.avg_churn_density)
        logger.info("  HHI: %.1f", evidence.health.hhi)
        logger.info("  displacement_intensity: %.2f", evidence.health.displacement_intensity)
        logger.info("  market_structure: %s", evidence.health.market_structure)
        logger.info("  dominant_archetype: %s", evidence.health.dominant_archetype)
        logger.info("  pain_distribution: %s", dict(list(evidence.pain_distribution.items())[:5]))
        logger.info("  top_flows: %d", len(evidence.top_displacement_flows))

        assert evidence.health.vendor_count > 0
        assert evidence.health.market_structure in (
            "consolidating", "fragmenting", "displacing", "stable",
            "insufficient_data",
        )

    @pytest.mark.asyncio
    async def test_analyze_all_categories(self, pool):
        """analyze_all_categories should not crash."""
        from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer

        analyzer = EcosystemAnalyzer(pool)
        results = await analyzer.analyze_all_categories()

        logger.info("Analyzed %d categories", len(results))
        for cat, eco in results.items():
            logger.info("  %s: %d vendors, HHI=%.0f, structure=%s",
                        cat, eco.health.vendor_count, eco.health.hhi,
                        eco.health.market_structure)

        assert len(results) > 0


# ------------------------------------------------------------------
# Test 6: Narrative engine with real vendor data
# ------------------------------------------------------------------


class TestNarrativeLive:

    @pytest.mark.asyncio
    async def test_build_narrative(self, pool, test_vendor, vendor_snapshot, vendor_evidence):
        """Build a real vendor narrative from live data."""
        from atlas_brain.reasoning.narrative import NarrativeEngine
        from atlas_brain.reasoning.archetypes import best_match

        engine = NarrativeEngine(pool)

        # Get archetype match
        match = best_match(vendor_evidence)
        archetype_match = None
        if match:
            archetype_match = {
                "archetype": match.archetype,
                "signal_score": match.score,
                "risk_level": match.risk_level,
            }

        narrative = engine.build_vendor_narrative(
            test_vendor,
            archetype_match=archetype_match,
            snapshot=vendor_snapshot,
            competitive_landscape={
                "losing_to": vendor_evidence.get("losing_to", []),
                "winning_from": vendor_evidence.get("winning_from", []),
            },
        )

        logger.info("Narrative for %s:", test_vendor)
        logger.info("  archetype: %s (score=%.2f)", narrative.archetype, narrative.archetype_score)
        logger.info("  risk_level: %s", narrative.risk_level)
        logger.info("  evidence_chains: %d", len(narrative.evidence_chains))
        for chain in narrative.evidence_chains:
            logger.info("    claim: %s", chain.claim[:80])
            logger.info("    confidence: %.2f, well_supported: %s, sources: %d",
                        chain.confidence, chain.is_well_supported, chain.source_count)

        assert narrative.vendor_name == test_vendor
        assert len(narrative.evidence_chains) > 0, "Should have at least one evidence chain"

    @pytest.mark.asyncio
    async def test_rule_engine_real_data(self, pool, test_vendor, vendor_snapshot):
        """Evaluate threshold rules against real vendor snapshot."""
        from atlas_brain.reasoning.narrative import NarrativeEngine

        engine = NarrativeEngine(pool)
        if not vendor_snapshot:
            pytest.skip("No snapshot for test vendor")

        triggered = engine.evaluate_rules(test_vendor, vendor_snapshot)

        logger.info("Rules triggered for %s: %d", test_vendor, len(triggered))
        for t in triggered:
            logger.info("  %s: %s (actual=%.1f, threshold=%.1f, priority=%s)",
                        t.rule.name, t.rule.description,
                        t.actual_value, t.rule.threshold, t.rule.priority)

        # Don't assert specific triggers - just ensure it doesn't crash
        assert isinstance(triggered, list)

    @pytest.mark.asyncio
    async def test_explainability(self, pool, test_vendor, vendor_snapshot, vendor_evidence):
        """Explainability audit trail should be well-formed."""
        from atlas_brain.reasoning.narrative import NarrativeEngine
        from atlas_brain.reasoning.archetypes import best_match

        engine = NarrativeEngine(pool)
        match = best_match(vendor_evidence)

        narrative = engine.build_vendor_narrative(
            test_vendor,
            archetype_match={"archetype": match.archetype, "signal_score": match.score} if match else None,
            snapshot=vendor_snapshot,
        )
        explain = engine.build_explainability(narrative)

        logger.info("Explainability for %s:", test_vendor)
        logger.info("  confidence_assessment: %s", explain.get("confidence_assessment"))
        logger.info("  confidence_factors: %s", explain.get("confidence_factors"))
        logger.info("  evidence_summary: %s", explain.get("evidence_summary"))

        assert "vendor" in explain
        assert "confidence_assessment" in explain
        assert explain["confidence_assessment"] in ("high", "medium", "low")

    @pytest.mark.asyncio
    async def test_intelligence_payload(self, pool, test_vendor, vendor_snapshot, vendor_evidence):
        """to_intelligence_payload should produce valid JSON-serializable output."""
        from atlas_brain.reasoning.narrative import NarrativeEngine
        from atlas_brain.reasoning.archetypes import best_match

        engine = NarrativeEngine(pool)
        match = best_match(vendor_evidence)

        narrative = engine.build_vendor_narrative(
            test_vendor,
            archetype_match={"archetype": match.archetype, "signal_score": match.score} if match else None,
            snapshot=vendor_snapshot,
            competitive_landscape={
                "losing_to": vendor_evidence.get("losing_to", []),
                "winning_from": vendor_evidence.get("winning_from", []),
            },
        )
        payload = NarrativeEngine.to_intelligence_payload(narrative)

        # Must be JSON-serializable
        serialized = json.dumps(payload, default=str)
        assert len(serialized) > 100, "Payload too small"

        logger.info("Intelligence payload (%d bytes): %s", len(serialized),
                     json.dumps(payload, indent=2, default=str)[:500])

        assert payload["vendor_name"] == test_vendor
        assert "archetype" in payload
        assert "risk_level" in payload
        assert "evidence_chain_count" in payload


# ------------------------------------------------------------------
# Test 7: Full stratified reasoner with real LLM
# ------------------------------------------------------------------


class TestStratifiedReasonerLive:

    @pytest.mark.asyncio
    async def test_full_reason_mode(self, pool, test_vendor, vendor_evidence, openrouter_llm):
        """Full Reason mode: cache is empty, should call LLM and store result."""
        from atlas_brain.reasoning.semantic_cache import SemanticCache
        from atlas_brain.reasoning.episodic_store import EpisodicStore
        from atlas_brain.reasoning.stratified_reasoner import StratifiedReasoner
        from atlas_brain.pipelines import llm as llm_module

        cache = SemanticCache(pool)
        episodic = EpisodicStore()

        reasoner = StratifiedReasoner(cache, episodic)

        # Monkey-patch get_pipeline_llm to return our OpenRouter LLM
        original = llm_module.get_pipeline_llm

        def _patched(**kwargs):
            return openrouter_llm

        llm_module.get_pipeline_llm = _patched

        try:
            t0 = time.monotonic()
            result = await reasoner.analyze(
                test_vendor,
                vendor_evidence,
                product_category=vendor_evidence.get("product_category", ""),
                force_reason=True,
            )
            elapsed = time.monotonic() - t0

            logger.info("=== REASON MODE RESULT (%s) ===", test_vendor)
            logger.info("  mode: %s", result.mode)
            logger.info("  archetype: %s", result.conclusion.get("archetype"))
            logger.info("  confidence: %.2f", result.confidence)
            logger.info("  risk_level: %s", result.conclusion.get("risk_level"))
            logger.info("  tokens_used: %d", result.tokens_used)
            logger.info("  cached: %s", result.cached)
            logger.info("  trace_id: %s", result.trace_id)
            logger.info("  elapsed: %.1fs", elapsed)
            logger.info("  executive_summary: %s",
                        result.conclusion.get("executive_summary", "")[:200])
            logger.info("  key_signals: %s", result.conclusion.get("key_signals", []))
            logger.info("  falsification: %s",
                        result.conclusion.get("falsification_conditions", []))

            assert result.mode == "reason", f"Expected reason mode, got {result.mode}"
            assert result.tokens_used > 0, "Should have used tokens"
            assert result.confidence > 0, "Confidence should be positive"

            # Validate conclusion structure
            conc = result.conclusion
            assert "archetype" in conc, "Missing archetype in conclusion"
            assert "executive_summary" in conc, "Missing executive_summary"
            assert "risk_level" in conc, "Missing risk_level"

            # Archetype should be one of the valid ones
            valid_archetypes = {
                "pricing_shock", "feature_gap", "acquisition_decay",
                "leadership_redesign", "integration_break", "support_collapse",
                "category_disruption", "compliance_gap", "mixed", "stable",
            }
            assert conc["archetype"] in valid_archetypes, \
                f"Invalid archetype: {conc['archetype']}"

        finally:
            llm_module.get_pipeline_llm = original

    @pytest.mark.asyncio
    async def test_recall_mode_after_reason(self, pool, test_vendor, vendor_evidence):
        """After reasoning, same evidence should trigger recall (cache hit)."""
        from atlas_brain.reasoning.semantic_cache import SemanticCache, compute_evidence_hash
        from atlas_brain.reasoning.episodic_store import EpisodicStore
        from atlas_brain.reasoning.stratified_reasoner import StratifiedReasoner

        cache = SemanticCache(pool)
        episodic = EpisodicStore()
        reasoner = StratifiedReasoner(cache, episodic)

        # This should hit the cache from the previous test
        ev_hash = compute_evidence_hash(vendor_evidence)
        pattern_sig = reasoner._build_pattern_sig(test_vendor, ev_hash)

        # Check if cache has the entry
        entry = await cache.lookup(pattern_sig)
        if entry is None:
            pytest.skip("Cache empty -- test_full_reason_mode may not have run first")

        result = await reasoner.analyze(
            test_vendor,
            vendor_evidence,
            product_category=vendor_evidence.get("product_category", ""),
        )

        logger.info("=== RECALL MODE RESULT ===")
        logger.info("  mode: %s", result.mode)
        logger.info("  tokens_used: %d", result.tokens_used)
        logger.info("  cached: %s", result.cached)

        assert result.mode == "recall", f"Expected recall, got {result.mode}"
        assert result.tokens_used == 0, "Recall should use 0 tokens"
        assert result.cached is True

    @pytest.mark.asyncio
    async def test_reason_multiple_vendors(self, pool, openrouter_llm):
        """Run reason mode on 3 vendors and compare results."""
        from atlas_brain.reasoning.semantic_cache import SemanticCache
        from atlas_brain.reasoning.episodic_store import EpisodicStore
        from atlas_brain.reasoning.stratified_reasoner import StratifiedReasoner
        from atlas_brain.pipelines import llm as llm_module

        cache = SemanticCache(pool)
        episodic = EpisodicStore()
        reasoner = StratifiedReasoner(cache, episodic)

        original = llm_module.get_pipeline_llm
        llm_module.get_pipeline_llm = lambda **kwargs: openrouter_llm

        try:
            # Pick 3 diverse vendors
            rows = await pool.fetch("""
                SELECT cs.vendor_name, cs.product_category,
                       snap.churn_density, snap.avg_urgency,
                       snap.positive_review_pct, snap.displacement_edge_count,
                       snap.competitor_count, snap.pain_count,
                       snap.high_intent_company_count, snap.recommend_ratio
                FROM b2b_churn_signals cs
                LEFT JOIN (
                    SELECT DISTINCT ON (vendor_name) *
                    FROM b2b_vendor_snapshots ORDER BY vendor_name, snapshot_date DESC
                ) snap ON LOWER(cs.vendor_name) = LOWER(snap.vendor_name)
                WHERE snap.churn_density IS NOT NULL
                ORDER BY RANDOM()
                LIMIT 3
            """)

            results = []
            for row in rows:
                vendor = row["vendor_name"]
                evidence = {k: row[k] for k in row.keys() if row[k] is not None}
                category = evidence.pop("product_category", "")

                result = await reasoner.analyze(
                    vendor, evidence,
                    product_category=category,
                    force_reason=True,
                )
                results.append((vendor, result))
                logger.info(
                    "  %s -> %s (conf=%.2f, risk=%s, %d tok)",
                    vendor,
                    result.conclusion.get("archetype", "?"),
                    result.confidence,
                    result.conclusion.get("risk_level", "?"),
                    result.tokens_used,
                )

            assert len(results) == 3, "Should have reasoned about 3 vendors"

            # Each should have a valid conclusion
            for vendor, result in results:
                assert result.mode == "reason"
                assert "archetype" in result.conclusion
                assert result.tokens_used > 0

        finally:
            llm_module.get_pipeline_llm = original


# ------------------------------------------------------------------
# Test 8: Full pipeline integration (narrative + archetype + LLM)
# ------------------------------------------------------------------


class TestFullPipelineLive:

    @pytest.mark.asyncio
    async def test_end_to_end_vendor_intelligence(
        self, pool, test_vendor, vendor_evidence, vendor_snapshot, openrouter_llm,
    ):
        """End-to-end: archetype scoring -> narrative -> LLM reasoning -> payload.

        This is what a real intelligence report should do.
        """
        from atlas_brain.reasoning.archetypes import score_evidence, best_match
        from atlas_brain.reasoning.narrative import NarrativeEngine
        from atlas_brain.reasoning.semantic_cache import SemanticCache
        from atlas_brain.reasoning.episodic_store import EpisodicStore
        from atlas_brain.reasoning.stratified_reasoner import StratifiedReasoner
        from atlas_brain.pipelines import llm as llm_module

        logger.info("=" * 60)
        logger.info("END-TO-END VENDOR INTELLIGENCE: %s", test_vendor)
        logger.info("=" * 60)

        # Step 1: Archetype pre-scoring
        matches = score_evidence(vendor_evidence)
        top_match = matches[0] if matches else None
        logger.info("Step 1 - Archetype: %s (score=%.3f)",
                     top_match.archetype if top_match else "none",
                     top_match.score if top_match else 0)

        # Step 2: Build narrative with evidence chains
        narrative_engine = NarrativeEngine(pool)
        narrative = narrative_engine.build_vendor_narrative(
            test_vendor,
            archetype_match={
                "archetype": top_match.archetype,
                "signal_score": top_match.score,
                "risk_level": top_match.risk_level,
            } if top_match else None,
            snapshot=vendor_snapshot,
            competitive_landscape={
                "losing_to": vendor_evidence.get("losing_to", []),
                "winning_from": vendor_evidence.get("winning_from", []),
            },
        )
        logger.info("Step 2 - Narrative: %d evidence chains, %d well-supported",
                     len(narrative.evidence_chains),
                     sum(1 for c in narrative.evidence_chains if c.is_well_supported))

        # Step 3: Evaluate threshold rules
        triggered = narrative_engine.evaluate_rules(test_vendor, vendor_snapshot or {})
        logger.info("Step 3 - Rules: %d triggered", len(triggered))
        for t in triggered:
            logger.info("  TRIGGERED: %s (%s, actual=%.1f)",
                        t.rule.name, t.rule.priority, t.actual_value)

        # Step 4: Build explainability
        explain = narrative_engine.build_explainability(narrative)
        logger.info("Step 4 - Confidence: %s (factors: %s)",
                     explain["confidence_assessment"],
                     explain["confidence_factors"])

        # Step 5: Stratified reasoning (LLM call)
        cache = SemanticCache(pool)
        episodic = EpisodicStore()
        reasoner = StratifiedReasoner(cache, episodic)

        original = llm_module.get_pipeline_llm
        llm_module.get_pipeline_llm = lambda **kwargs: openrouter_llm

        try:
            result = await reasoner.analyze(
                test_vendor,
                vendor_evidence,
                product_category=vendor_evidence.get("product_category", ""),
                force_reason=True,
            )
            logger.info("Step 5 - LLM Reasoning:")
            logger.info("  archetype: %s", result.conclusion.get("archetype"))
            logger.info("  confidence: %.2f", result.confidence)
            logger.info("  risk_level: %s", result.conclusion.get("risk_level"))
            logger.info("  executive_summary: %s",
                        result.conclusion.get("executive_summary", "")[:300])
            logger.info("  tokens: %d", result.tokens_used)
        finally:
            llm_module.get_pipeline_llm = original

        # Step 6: Generate intelligence payload
        payload = NarrativeEngine.to_intelligence_payload(narrative)
        payload["llm_conclusion"] = result.conclusion
        payload["explainability"] = explain
        payload["triggered_rules"] = [
            {"rule": t.rule.name, "actual": t.actual_value,
             "priority": t.rule.priority}
            for t in triggered
        ]

        logger.info("Step 6 - Final payload: %d bytes",
                     len(json.dumps(payload, default=str)))

        # Validate the complete output
        assert payload["vendor_name"] == test_vendor
        assert "archetype" in payload
        assert "llm_conclusion" in payload
        assert "explainability" in payload
        assert payload["evidence_chain_count"] > 0

        # Check archetype agreement between pre-scoring and LLM
        pre_archetype = top_match.archetype if top_match else ""
        llm_archetype = result.conclusion.get("archetype", "")
        if pre_archetype and llm_archetype:
            agreement = pre_archetype == llm_archetype
            logger.info(
                "ARCHETYPE AGREEMENT: pre-score=%s, llm=%s, agree=%s",
                pre_archetype, llm_archetype, agreement,
            )

        logger.info("=" * 60)
        logger.info("END-TO-END COMPLETE")
        logger.info("=" * 60)


# ------------------------------------------------------------------
# Test 9: Semantic cache round-trip with real Postgres
# ------------------------------------------------------------------


class TestSemanticCacheLive:

    @pytest.mark.asyncio
    async def test_cache_store_and_lookup(self, pool):
        """Store a cache entry and retrieve it from real Postgres."""
        from atlas_brain.reasoning.semantic_cache import (
            SemanticCache, CacheEntry, compute_evidence_hash,
        )

        cache = SemanticCache(pool)
        test_evidence = {"churn_density": 42.0, "avg_urgency": 5.5}
        ev_hash = compute_evidence_hash(test_evidence)
        pattern_sig = f"__test_vendor__:{ev_hash}"

        entry = CacheEntry(
            pattern_sig=pattern_sig,
            pattern_class="pricing_shock",
            vendor_name="__test_vendor__",
            product_category="testing",
            conclusion={"archetype": "pricing_shock", "confidence": 0.8},
            confidence=0.8,
            evidence_hash=ev_hash,
        )

        await cache.store(entry)
        retrieved = await cache.lookup(pattern_sig)

        assert retrieved is not None, "Cache lookup should find stored entry"
        assert retrieved.pattern_class == "pricing_shock"
        assert retrieved.confidence == 0.8
        logger.info("Cache round-trip: PASS")

        # Clean up
        await pool.execute(
            "DELETE FROM reasoning_semantic_cache WHERE pattern_sig = $1",
            pattern_sig,
        )

    @pytest.mark.asyncio
    async def test_cache_stats(self, pool):
        """Cache stats query should work against real table."""
        from atlas_brain.reasoning.semantic_cache import SemanticCache

        cache = SemanticCache(pool)
        stats = await cache.get_cache_stats()
        logger.info("Cache stats: %s", stats)
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_lookup_for_tier(self, pool):
        """lookup_for_tier should find entries by conclusion_type + product_category."""
        import json as _json
        from datetime import datetime, timezone
        from atlas_brain.reasoning.semantic_cache import SemanticCache

        cache = SemanticCache(pool)
        now = datetime.now(timezone.utc)
        t4_sig = "__test_t4__:test_cat"
        t3_sig = "__test_t3__:test_cat"

        # Insert T4 (market_dynamics) and T3 (category_pattern) entries
        for sig, ctype, data in [
            (t4_sig, "market_dynamics", {"market_structure": "stable", "hhi": 1200}),
            (t3_sig, "category_pattern", {"dominant_archetype": "pricing_shock", "avg_churn_density": 0.3}),
        ]:
            await pool.execute(
                """
                INSERT INTO reasoning_semantic_cache (
                    pattern_sig, pattern_class, vendor_name, product_category,
                    conclusion_type, confidence, conclusion,
                    created_at, last_validated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
                ON CONFLICT (pattern_sig) DO UPDATE SET
                    conclusion_type = EXCLUDED.conclusion_type,
                    last_validated_at = EXCLUDED.last_validated_at,
                    invalidated_at = NULL
                """,
                sig, "stable", "__ecosystem__", "Test Category",
                ctype, 0.8,
                _json.dumps(data),
                now,
            )

        try:
            # T4 lookup should only find market_dynamics
            t4_entries = await cache.lookup_for_tier(
                "market_dynamics", product_category="Test Category", limit=3,
            )
            assert len(t4_entries) >= 1, "Should find market_dynamics entry"
            assert t4_entries[0].conclusion_type == "market_dynamics"

            # T3 lookup should only find category_pattern
            t3_entries = await cache.lookup_for_tier(
                "category_pattern", product_category="Test Category", limit=3,
            )
            assert len(t3_entries) >= 1, "Should find category_pattern entry"
            assert t3_entries[0].conclusion_type == "category_pattern"

            # T4 and T3 should NOT cross-match
            wrong = await cache.lookup_for_tier(
                "market_dynamics", product_category="Nonexistent", limit=3,
            )
            assert len(wrong) == 0, "Should not find entries for wrong category"

            logger.info("lookup_for_tier: T4=%d entries, T3=%d entries (correctly separated)",
                        len(t4_entries), len(t3_entries))
        finally:
            await pool.execute(
                "DELETE FROM reasoning_semantic_cache WHERE pattern_sig IN ($1, $2)",
                t4_sig, t3_sig,
            )


# ------------------------------------------------------------------
# Test 10: Tier context gathering
# ------------------------------------------------------------------


class TestTierContextLive:

    @pytest.mark.asyncio
    async def test_gather_tier_context_t2(self, pool, test_vendor):
        """gather_tier_context for T2 should return tier metadata and any T4 priors."""
        from atlas_brain.reasoning.semantic_cache import SemanticCache
        from atlas_brain.reasoning.tiers import Tier, gather_tier_context

        cache = SemanticCache(pool)
        ctx = await gather_tier_context(
            cache, Tier.VENDOR_ARCHETYPE,
            vendor_name=test_vendor, product_category="Project Management",
        )

        logger.info("Tier context for T2 %s: %s", test_vendor, ctx)

        assert ctx["tier"] == 2
        assert ctx["tier_name"] == "Vendor Archetype"
        # inherited_priors may or may not exist depending on cache state
        if "inherited_priors" in ctx:
            for prior in ctx["inherited_priors"]:
                assert "tier" in prior
                assert "confidence" in prior
                logger.info("  Prior: tier=%d type=%s conf=%.2f",
                            prior["tier"], prior.get("conclusion_type", ""), prior["confidence"])

    @pytest.mark.asyncio
    async def test_gather_tier_context_t4_has_no_parents(self, pool):
        """T4 (Market Dynamics) should inherit nothing."""
        from atlas_brain.reasoning.semantic_cache import SemanticCache
        from atlas_brain.reasoning.tiers import Tier, gather_tier_context

        cache = SemanticCache(pool)
        ctx = await gather_tier_context(cache, Tier.MARKET_DYNAMICS)

        assert ctx["tier"] == 4
        assert "inherited_priors" not in ctx, "T4 should have no parents"
