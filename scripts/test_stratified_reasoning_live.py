#!/usr/bin/env python3
"""Live stratified reasoning test.

Initializes the reasoning engine against real Postgres + Neo4j,
picks 3 vendors with the most churn data, builds evidence for each,
and runs them through the full analyze() pipeline.

Usage:
    python scripts/test_stratified_reasoning_live.py

Expects:
    - Postgres running with b2b_churn_signals populated
    - Neo4j running on bolt://localhost:7687
    - vLLM server running on configured URL (or Anthropic key in env)
    - ATLAS_B2B_CHURN_STRATIFIED_REASONING_ENABLED=true in .env
"""

import asyncio
import json
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
# Quiet noisy loggers
for name in ("neo4j", "httpx", "httpcore", "urllib3"):
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger("test_reasoning")


async def main():
    from atlas_brain.storage.database import DatabasePool
    from atlas_brain.reasoning import init_stratified_reasoner, get_stratified_reasoner
    from atlas_brain.reasoning.tiers import Tier, gather_tier_context

    # 1. Init DB
    pool = DatabasePool()
    await pool.initialize()
    logger.info("DB connected")

    # 2. Init reasoning engine
    await init_stratified_reasoner(pool)
    reasoner = get_stratified_reasoner()
    if reasoner is None:
        logger.error("Reasoner failed to initialize")
        return 1

    # Check degraded state
    if getattr(reasoner._episodic, "_degraded", False):
        logger.warning("Episodic store is DEGRADED (Neo4j unavailable) -- reasoning will work but no trace storage")
    else:
        logger.info("Episodic store healthy")

    logger.info("Reasoner initialized")

    # 3. Pick top 3 vendors by data richness
    vendors = await pool.fetch("""
        SELECT vendor_name, product_category, total_reviews, churn_intent_count,
               avg_urgency_score, price_complaint_rate, decision_maker_churn_rate
        FROM b2b_churn_signals
        WHERE total_reviews > 10
        ORDER BY total_reviews DESC
        LIMIT 3
    """)
    if not vendors:
        logger.error("No vendors with sufficient data in b2b_churn_signals")
        return 1

    logger.info("Selected %d vendors: %s", len(vendors), [v["vendor_name"] for v in vendors])

    # 4. Build evidence and run reasoning for each vendor
    results = []
    for v in vendors:
        vendor_name = v["vendor_name"]
        category = v["product_category"] or ""

        # Build evidence from real DB data
        evidence = {
            "vendor_name": vendor_name,
            "product_category": category,
            "total_reviews": v["total_reviews"],
            "churn_intent": v["churn_intent_count"],
            "churn_density": round(
                (v["churn_intent_count"] / max(v["total_reviews"], 1)) * 100, 1
            ),
            "avg_urgency": round(float(v["avg_urgency_score"] or 0), 1),
            "dm_churn_rate": round(float(v["decision_maker_churn_rate"] or 0), 2),
            "price_complaint_rate": round(float(v["price_complaint_rate"] or 0), 2),
        }

        # Enrich with pain categories from DB
        pains = await pool.fetch("""
            SELECT pain_category, mention_count
            FROM b2b_vendor_pain_points
            WHERE vendor_name = $1
            ORDER BY mention_count DESC LIMIT 5
        """, vendor_name)
        if pains:
            evidence["pain_categories"] = [
                {"category": p["pain_category"], "count": p["mention_count"]}
                for p in pains
            ]

        # Enrich with competitors
        comps = await pool.fetch("""
            SELECT to_vendor, mention_count
            FROM b2b_displacement_edges
            WHERE from_vendor = $1
            ORDER BY mention_count DESC LIMIT 5
        """, vendor_name)
        if comps:
            evidence["competitors"] = [
                {"name": c["to_vendor"], "mentions": c["mention_count"]}
                for c in comps
            ]
            evidence["displacement_mention_count"] = sum(c["mention_count"] for c in comps)

        # Enrich with temporal data
        try:
            from atlas_brain.reasoning.temporal import TemporalEngine
            te = TemporalEngine(pool)
            temporal = await te.analyze_vendor(vendor_name)
            td = TemporalEngine.to_evidence_dict(temporal)
            evidence.update(td)
        except Exception as e:
            logger.debug("Temporal enrichment skipped for %s: %s", vendor_name, e)

        logger.info("Evidence for %s: %d fields", vendor_name, len(evidence))

        # Gather tier context
        tier_ctx = await gather_tier_context(
            reasoner._cache, Tier.VENDOR_ARCHETYPE,
            vendor_name=vendor_name, product_category=category,
        )

        # Run reasoning
        t0 = time.monotonic()
        result = await reasoner.analyze(
            vendor_name=vendor_name,
            evidence=evidence,
            product_category=category,
            tier_context=tier_ctx,
        )
        elapsed = time.monotonic() - t0

        results.append({
            "vendor": vendor_name,
            "mode": result.mode,
            "archetype": result.conclusion.get("archetype", ""),
            "confidence": result.confidence,
            "risk_level": result.conclusion.get("risk_level", ""),
            "tokens_used": result.tokens_used,
            "elapsed_s": round(elapsed, 2),
            "cached": result.cached,
            "trace_id": result.trace_id,
            "exec_summary": result.conclusion.get("executive_summary", "")[:200],
            "key_signals": result.conclusion.get("key_signals", []),
            "falsification": result.conclusion.get("falsification_conditions", [])[:2],
            "error": result.conclusion.get("error"),
        })

    # 5. Print results
    print("\n" + "=" * 80)
    print("STRATIFIED REASONING RESULTS")
    print("=" * 80)

    total_tokens = 0
    for r in results:
        print(f"\n--- {r['vendor']} ---")
        print(f"  Mode:       {r['mode']}")
        print(f"  Archetype:  {r['archetype']}")
        print(f"  Confidence: {r['confidence']}")
        print(f"  Risk:       {r['risk_level']}")
        print(f"  Tokens:     {r['tokens_used']}")
        print(f"  Elapsed:    {r['elapsed_s']}s")
        print(f"  Cached:     {r['cached']}")
        print(f"  Trace ID:   {r['trace_id']}")
        if r["error"]:
            print(f"  ERROR:      {r['error']}")
        if r["exec_summary"]:
            print(f"  Summary:    {r['exec_summary']}")
        if r["key_signals"]:
            print(f"  Signals:    {r['key_signals']}")
        if r["falsification"]:
            print(f"  Falsify:    {r['falsification']}")
        total_tokens += r["tokens_used"]

    # 6. Verify cache state
    cache_count = await pool.fetchval(
        "SELECT COUNT(*) FROM reasoning_semantic_cache WHERE invalidated_at IS NULL"
    )
    print(f"\n--- Cache ---")
    print(f"  Active entries: {cache_count}")
    print(f"  Total tokens:   {total_tokens}")

    # 7. Check for errors
    errors = [r for r in results if r["error"]]
    successes = [r for r in results if not r["error"]]

    print(f"\n--- Summary ---")
    print(f"  Vendors:    {len(results)}")
    print(f"  Successes:  {len(successes)}")
    print(f"  Errors:     {len(errors)}")

    if errors:
        print(f"\n  FAILED vendors:")
        for r in errors:
            print(f"    {r['vendor']}: {r['error']}")

    await pool.close()

    # Return 0 on all success, 1 on any error
    return 1 if errors else 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
