#!/usr/bin/env python3
"""Benchmark stratified reasoning across LLM models.

Replicates the EXACT pipeline path from b2b_churn_intelligence.py:
  _fetch_vendor_churn_scores -> batch lookups -> _build_vendor_evidence
  -> reasoner.analyze() with asyncio.gather + semaphore.

Default mode is LIVE: cache recall/reconstitute/reason as normal so you
see what each model actually does in production (including cache effects).
Use --fresh to invalidate cache and force full reasoning for both models.

Usage:
    python scripts/benchmark_reasoning_models.py
    python scripts/benchmark_reasoning_models.py --vendors 10 --fresh
    python scripts/benchmark_reasoning_models.py --model-a openai/o4-mini --model-b moonshotai/kimi-k2.5
    python scripts/benchmark_reasoning_models.py --output data/reasoning_benchmark.json

Expects:
    - Postgres with b2b_churn_signals populated
    - Neo4j on bolt://localhost:7687 (degrades gracefully)
    - OPENROUTER_API_KEY or ATLAS_B2B_CHURN_OPENROUTER_API_KEY in env
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
for name in ("neo4j", "httpx", "httpcore", "urllib3"):
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger("benchmark_reasoning")

DEFAULT_VENDOR_CAP = 60
DEFAULT_CONCURRENCY = 5

# Sigs created during --fresh runs, cleaned up on exit.
_benchmark_sigs: list[str] = []


# ------------------------------------------------------------------
# Batch evidence assembly (mirrors pipeline exactly)
# ------------------------------------------------------------------

async def build_batch_evidence(pool, vendor_cap: int) -> list[dict]:
    """Replicate the pipeline's batch evidence assembly.

    Calls _fetch_vendor_churn_scores with the same ordering + cap,
    then builds lookups from bulk fetches and runs _build_vendor_evidence
    for each vendor -- identical to b2b_churn_intelligence.py:1263-1313.
    """
    from atlas_brain.config import settings
    from atlas_brain.autonomous.tasks._b2b_shared import (
        _canonicalize_vendor,
        _build_vendor_evidence,
        _build_pain_lookup,
        _build_competitor_lookup,
        _build_feature_gap_lookup,
        _build_keyword_spike_lookup,
        _build_buyer_auth_lookup,
        _build_use_case_lookup,
        _build_insider_lookup,
        _fetch_vendor_churn_scores,
        _fetch_pain_distribution,
        _fetch_competitive_displacement,
        _fetch_feature_gaps,
        _fetch_price_complaint_rates,
        _fetch_dm_churn_rates,
        _fetch_keyword_spikes,
        _fetch_insider_aggregates,
        _fetch_quotable_evidence,
        _fetch_budget_signals,
        _fetch_use_case_distribution,
        _fetch_buyer_authority_summary,
    )

    cfg = settings.b2b_churn
    window_days = cfg.intelligence_window_days
    min_reviews = cfg.intelligence_min_reviews

    # 1. Vendor scores (same query, same avg_urgency DESC ordering)
    vendor_scores = await _fetch_vendor_churn_scores(pool, window_days, min_reviews)
    if not vendor_scores:
        return []

    capped = vendor_scores[:vendor_cap]
    logger.info(
        "Vendor scores: %d total, capped to %d (ordered by avg_urgency DESC)",
        len(vendor_scores), len(capped),
    )

    # 2. Bulk data fetches (parallel, matches pipeline)
    (
        pain_dist, competitive_disp, feature_gaps,
        price_rates, dm_rates, keyword_spikes,
        insider_raw, quotable_evidence, budget_signals,
        use_case_dist, buyer_auth,
    ) = await asyncio.gather(
        _fetch_pain_distribution(pool, window_days),
        _fetch_competitive_displacement(pool, window_days),
        _fetch_feature_gaps(pool, window_days, min_mentions=cfg.feature_gap_min_mentions),
        _fetch_price_complaint_rates(pool, window_days),
        _fetch_dm_churn_rates(pool, window_days),
        _fetch_keyword_spikes(pool),
        _fetch_insider_aggregates(pool, window_days),
        _fetch_quotable_evidence(pool, window_days, min_urgency=cfg.quotable_phrase_min_urgency),
        _fetch_budget_signals(pool, window_days),
        _fetch_use_case_distribution(pool, window_days),
        _fetch_buyer_authority_summary(pool, window_days),
    )

    # 3. Build lookups (same as pipeline)
    pain_lookup = _build_pain_lookup(pain_dist)
    comp_lookup = _build_competitor_lookup(competitive_disp)
    fg_lookup = _build_feature_gap_lookup(feature_gaps)
    kw_lookup = _build_keyword_spike_lookup(keyword_spikes)
    insider_lookup = _build_insider_lookup(insider_raw)
    dm_lookup = {r["vendor"]: r["dm_churn_rate"] for r in dm_rates}
    price_lookup = {r["vendor"]: r["price_complaint_rate"] for r in price_rates}
    quote_lookup = {r["vendor"]: r["quotes"] for r in quotable_evidence}
    budget_lookup = {
        r["vendor"]: {k: v for k, v in r.items() if k != "vendor"}
        for r in budget_signals
    }
    ba_lookup = _build_buyer_auth_lookup(buyer_auth)
    uc_lookup = _build_use_case_lookup(use_case_dist)

    # 4. Temporal + archetype pre-scores (non-fatal, same as pipeline)
    temporal_lookup: dict[str, dict] = {}
    archetype_lookup: dict[str, list[dict]] = {}
    try:
        from atlas_brain.reasoning.temporal import TemporalEngine
        from atlas_brain.reasoning.archetypes import enrich_evidence_with_archetypes

        te = TemporalEngine(pool)
        for vs in capped:
            vname = _canonicalize_vendor(vs.get("vendor_name") or "")
            if not vname:
                continue
            try:
                td_result = await te.analyze_vendor(vname)
                td = TemporalEngine.to_evidence_dict(td_result)
                temporal_lookup[vname] = td
                enriched = enrich_evidence_with_archetypes(
                    {"vendor_name": vname, **td}, td,
                )
                arch_scores = enriched.get("archetype_scores", [])
                if arch_scores:
                    archetype_lookup[vname] = arch_scores
            except Exception:
                logger.debug("Temporal enrichment skipped for %s", vname)
    except Exception:
        logger.debug("Temporal engine unavailable", exc_info=True)

    # 5. Displacement velocity (same as pipeline)
    vel_rows = await pool.fetch(
        """
        SELECT from_vendor,
               SUM(COALESCE(velocity_7d, 0)) AS velocity_7d,
               SUM(COALESCE(velocity_30d, 0)) AS velocity_30d
        FROM b2b_displacement_edges
        WHERE computed_date = (SELECT MAX(computed_date) FROM b2b_displacement_edges)
        GROUP BY from_vendor
        """,
    )
    velocity_lookup = {
        r["from_vendor"]: {
            "velocity_7d": float(r["velocity_7d"]),
            "velocity_30d": float(r["velocity_30d"]),
        }
        for r in vel_rows
    }

    # 6. Market regime analysis (same as pipeline b2b_churn_intelligence.py:1209-1254)
    market_regime_lookup: dict[str, dict] = {}
    if temporal_lookup:
        try:
            from dataclasses import asdict
            from atlas_brain.reasoning.market_pulse import MarketPulseReasoner
            from atlas_brain.reasoning.temporal import TemporalEvidence, VendorVelocity

            mp_reasoner = MarketPulseReasoner()
            vendors_by_cat: dict[str, list[tuple[str, dict]]] = {}
            for vs in capped:
                vname = _canonicalize_vendor(vs.get("vendor_name") or "")
                td = temporal_lookup.get(vname)
                if not td:
                    continue
                category = str(vs.get("product_category") or "")
                if category:
                    vendors_by_cat.setdefault(category, []).append((vname, td))

            market_regimes_by_cat: dict = {}
            for category, vendor_tds in vendors_by_cat.items():
                te_list = []
                for vname, td in vendor_tds:
                    velocities = []
                    for key, value in td.items():
                        if not key.startswith("velocity_"):
                            continue
                        velocities.append(VendorVelocity(
                            vendor_name=vname,
                            metric=key.replace("velocity_", ""),
                            current_value=0,
                            previous_value=0,
                            velocity=float(value),
                            days_between=1,
                        ))
                    te_list.append(TemporalEvidence(
                        vendor_name=vname,
                        snapshot_days=td.get("snapshot_days", 0),
                        velocities=velocities,
                    ))
                regime = mp_reasoner.analyze_category(category, te_list)
                market_regimes_by_cat[category] = regime

            for vs in capped:
                vname = _canonicalize_vendor(vs.get("vendor_name") or "")
                category = str(vs.get("product_category") or "")
                regime = market_regimes_by_cat.get(category)
                if vname and regime is not None:
                    market_regime_lookup[vname] = asdict(regime)
        except Exception:
            logger.debug("Market pulse analysis skipped", exc_info=True)

    # 7. Build per-vendor evidence dicts (same call as pipeline)
    evidence_list = []
    for vs in capped:
        vname = _canonicalize_vendor(vs.get("vendor_name") or "")
        if not vname:
            continue
        ev = _build_vendor_evidence(
            vs,
            pain_lookup=pain_lookup,
            competitor_lookup=comp_lookup,
            feature_gap_lookup=fg_lookup,
            insider_lookup=insider_lookup,
            keyword_spike_lookup=kw_lookup,
            temporal_lookup=temporal_lookup or None,
            archetype_lookup=archetype_lookup or None,
            dm_lookup=dm_lookup,
            price_lookup=price_lookup,
            quote_lookup=quote_lookup,
            budget_lookup=budget_lookup,
            buyer_auth_lookup=ba_lookup,
            use_case_lookup=uc_lookup,
            velocity_lookup=velocity_lookup or None,
            market_regime_lookup=market_regime_lookup or None,
        )
        evidence_list.append(ev)

    return evidence_list


# ------------------------------------------------------------------
# Per-vendor reasoning (mirrors pipeline's _analyze_one)
# ------------------------------------------------------------------

async def _analyze_one(
    reasoner,
    ev: dict,
    model_id: str,
    sem: asyncio.Semaphore,
    *,
    fresh: bool,
) -> dict:
    """Run reasoning for one vendor under the semaphore.

    Mirrors the pipeline's _analyze_one in b2b_churn_intelligence.py:1272.
    """
    from atlas_brain.reasoning.tiers import Tier, gather_tier_context

    vendor_name = ev["vendor_name"]
    category = ev.get("product_category", "")

    async with sem:
        tier_ctx = await gather_tier_context(
            reasoner._cache, Tier.VENDOR_ARCHETYPE,
            vendor_name=vendor_name, product_category=category,
        )

        t0 = time.monotonic()
        try:
            sr = await reasoner.analyze(
                vendor_name=vendor_name,
                evidence=ev,
                product_category=category,
                tier_context=tier_ctx,
                force_reason=fresh,
            )
            elapsed = time.monotonic() - t0
            conclusion = sr.conclusion or {}

            if fresh and sr.pattern_sig:
                _benchmark_sigs.append(sr.pattern_sig)

            return {
                "vendor": vendor_name,
                "category": category,
                "model": model_id,
                "mode": sr.mode,
                "archetype": conclusion.get("archetype", ""),
                "secondary_archetype": conclusion.get("secondary_archetype"),
                "confidence": sr.confidence,
                "risk_level": conclusion.get("risk_level", ""),
                "executive_summary": conclusion.get("executive_summary", ""),
                "key_signals": conclusion.get("key_signals", []),
                "falsification_conditions": conclusion.get("falsification_conditions", []),
                "uncertainty_sources": conclusion.get("uncertainty_sources", []),
                "tokens_used": sr.tokens_used,
                "elapsed_s": round(elapsed, 2),
                "cached": sr.cached,
                "error": conclusion.get("error"),
            }
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.exception("Reasoning failed for %s with %s", vendor_name, model_id)
            return {
                "vendor": vendor_name,
                "category": category,
                "model": model_id,
                "mode": "error",
                "archetype": "",
                "secondary_archetype": None,
                "confidence": 0,
                "risk_level": "",
                "executive_summary": "",
                "key_signals": [],
                "falsification_conditions": [],
                "uncertainty_sources": [],
                "tokens_used": 0,
                "elapsed_s": round(elapsed, 2),
                "cached": False,
                "error": str(exc)[:200],
            }


# ------------------------------------------------------------------
# Phase runner
# ------------------------------------------------------------------

async def run_reasoning_phase(
    reasoner,
    model_id: str,
    evidence_list: list[dict],
    *,
    concurrency: int,
    max_tokens: int,
    fresh: bool,
) -> list[dict]:
    """Run stratified reasoning for all vendors concurrently.

    Sets model via env var so resolve_stratified_llm() inside _reason()
    picks it up through a fresh ReasoningConfig() instantiation.
    """
    os.environ["ATLAS_REASONING__STRATIFIED_OPENROUTER_MODEL"] = model_id
    os.environ["ATLAS_REASONING__STRATIFIED_LLM_WORKLOAD"] = "openrouter"
    os.environ["ATLAS_REASONING__MAX_TOKENS"] = str(max_tokens)

    # Force re-activation by deactivating so model mismatch check triggers
    from atlas_brain.services import llm_registry
    try:
        llm_registry.deactivate()
    except Exception:
        pass

    logger.info(
        "Phase: model=%s, max_tokens=%d, concurrency=%d, fresh=%s",
        model_id, max_tokens, concurrency, fresh,
    )

    # In fresh mode, invalidate cache for all benchmark vendors
    if fresh:
        async def _invalidate(vn: str):
            prior = await reasoner._cache.lookup_by_class("", vendor_name=vn, limit=20)
            for entry in prior:
                try:
                    await reasoner._cache.invalidate(entry.pattern_sig, reason="benchmark_clear")
                except Exception:
                    pass

        await asyncio.gather(*[
            _invalidate(ev["vendor_name"]) for ev in evidence_list
        ])

    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*[
        _analyze_one(reasoner, ev, model_id, sem, fresh=fresh)
        for ev in evidence_list
    ])

    return list(results)


# ------------------------------------------------------------------
# Comparison output
# ------------------------------------------------------------------

def print_comparison(results_a: list[dict], results_b: list[dict]) -> None:
    model_a = results_a[0]["model"] if results_a else "?"
    model_b = results_b[0]["model"] if results_b else "?"

    by_vendor_a = {r["vendor"]: r for r in results_a}
    by_vendor_b = {r["vendor"]: r for r in results_b}
    all_vendors = list(dict.fromkeys(
        [r["vendor"] for r in results_a] + [r["vendor"] for r in results_b]
    ))

    print("\n" + "=" * 100)
    print(f"REASONING BENCHMARK: {model_a} vs {model_b}")
    print("=" * 100)

    agree_count = 0
    risk_agree = 0
    total = 0

    for vendor in all_vendors:
        a = by_vendor_a.get(vendor, {})
        b = by_vendor_b.get(vendor, {})
        total += 1

        arch_match = a.get("archetype") == b.get("archetype")
        risk_match = a.get("risk_level") == b.get("risk_level")
        if arch_match:
            agree_count += 1
        if risk_match:
            risk_agree += 1

        print(f"\n--- {vendor} ---")
        col_a = f"[{model_a}]"
        col_b = f"[{model_b}]"
        print(f"  {'':30s} {col_a:>35s}   {col_b:>35s}")
        print(f"  {'Archetype':30s} {a.get('archetype', '-'):>35s}   {b.get('archetype', '-'):>35s}  {'OK' if arch_match else 'DIFFER'}")
        print(f"  {'Secondary':30s} {str(a.get('secondary_archetype') or '-'):>35s}   {str(b.get('secondary_archetype') or '-'):>35s}")
        print(f"  {'Confidence':30s} {a.get('confidence', 0):>35.2f}   {b.get('confidence', 0):>35.2f}  delta={abs(a.get('confidence', 0) - b.get('confidence', 0)):.2f}")
        print(f"  {'Risk Level':30s} {a.get('risk_level', '-'):>35s}   {b.get('risk_level', '-'):>35s}  {'OK' if risk_match else 'DIFFER'}")
        print(f"  {'Mode':30s} {a.get('mode', '-'):>35s}   {b.get('mode', '-'):>35s}")
        print(f"  {'Tokens':30s} {a.get('tokens_used', 0):>35d}   {b.get('tokens_used', 0):>35d}")
        print(f"  {'Latency':30s} {str(a.get('elapsed_s', 0)) + 's':>35s}   {str(b.get('elapsed_s', 0)) + 's':>35s}")

        # Key signals diff
        sigs_a = set(a.get("key_signals") or [])
        sigs_b = set(b.get("key_signals") or [])
        shared = sigs_a & sigs_b
        only_a = sigs_a - sigs_b
        only_b = sigs_b - sigs_a
        if shared or only_a or only_b:
            print(f"  Key Signals:")
            for s in sorted(shared):
                print(f"    = {s[:90]}")
            for s in sorted(only_a):
                print(f"    A {s[:90]}")
            for s in sorted(only_b):
                print(f"    B {s[:90]}")

        sum_a = (a.get("executive_summary") or "")[:120]
        sum_b = (b.get("executive_summary") or "")[:120]
        if sum_a or sum_b:
            print(f"  Summary (A): {sum_a}")
            print(f"  Summary (B): {sum_b}")

        if a.get("error"):
            print(f"  ERROR (A): {a['error']}")
        if b.get("error"):
            print(f"  ERROR (B): {b['error']}")

    # Aggregate
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  Vendors benchmarked:   {total}")
    print(f"  Archetype agreement:   {agree_count}/{total} ({agree_count/max(total,1)*100:.0f}%)")
    print(f"  Risk level agreement:  {risk_agree}/{total} ({risk_agree/max(total,1)*100:.0f}%)")

    conf_a = [r["confidence"] for r in results_a if not r.get("error")]
    conf_b = [r["confidence"] for r in results_b if not r.get("error")]
    if conf_a:
        print(f"  Avg confidence (A):    {sum(conf_a)/len(conf_a):.2f}")
    if conf_b:
        print(f"  Avg confidence (B):    {sum(conf_b)/len(conf_b):.2f}")

    tokens_a = sum(r.get("tokens_used", 0) for r in results_a)
    tokens_b = sum(r.get("tokens_used", 0) for r in results_b)
    time_a = sum(r.get("elapsed_s", 0) for r in results_a)
    time_b = sum(r.get("elapsed_s", 0) for r in results_b)
    cached_a = sum(1 for r in results_a if r.get("cached"))
    cached_b = sum(1 for r in results_b if r.get("cached"))
    print(f"  Total tokens (A):      {tokens_a}")
    print(f"  Total tokens (B):      {tokens_b}")
    print(f"  Total time (A):        {time_a:.1f}s")
    print(f"  Total time (B):        {time_b:.1f}s")
    print(f"  Cache recalls (A):     {cached_a}")
    print(f"  Cache recalls (B):     {cached_b}")

    errors_a = sum(1 for r in results_a if r.get("error"))
    errors_b = sum(1 for r in results_b if r.get("error"))
    if errors_a or errors_b:
        print(f"  Errors (A):            {errors_a}")
        print(f"  Errors (B):            {errors_b}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Benchmark reasoning models")
    parser.add_argument("--model-a", default="openai/o4-mini",
                        help="First model (OpenRouter ID)")
    parser.add_argument("--model-b", default="moonshotai/kimi-k2.5",
                        help="Second model (OpenRouter ID)")
    parser.add_argument("--vendors", type=int, default=None,
                        help=f"Override vendor cap (default: {DEFAULT_VENDOR_CAP})")
    parser.add_argument("--concurrency", type=int, default=None,
                        help=f"Override concurrency (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Max output tokens per LLM call (includes reasoning tokens)")
    parser.add_argument("--fresh", action="store_true",
                        help="Invalidate cache and force full reasoning (default: live mode with cache)")
    parser.add_argument("--output", default=None,
                        help="Save full results to JSON file")
    args = parser.parse_args()

    from atlas_brain.config import settings
    from atlas_brain.storage.database import DatabasePool
    from atlas_brain.reasoning import (
        init_stratified_reasoner,
        get_stratified_reasoner,
        close_stratified_reasoner,
    )

    cfg = settings.b2b_churn
    vendor_cap = args.vendors or DEFAULT_VENDOR_CAP
    concurrency = args.concurrency or DEFAULT_CONCURRENCY

    # Save original env vars for restoration
    _saved_env = {}
    _env_keys = (
        "ATLAS_REASONING__STRATIFIED_OPENROUTER_MODEL",
        "ATLAS_REASONING__STRATIFIED_LLM_WORKLOAD",
        "ATLAS_REASONING__MAX_TOKENS",
    )
    for k in _env_keys:
        _saved_env[k] = os.environ.get(k)

    pool = DatabasePool()
    try:
        await pool.initialize()
        logger.info("DB connected")

        await init_stratified_reasoner(pool)
        reasoner = get_stratified_reasoner()
        if reasoner is None:
            logger.error("Reasoner failed to initialize")
            return 1
        logger.info(
            "Reasoner initialized (episodic degraded=%s)",
            getattr(reasoner._episodic, "_degraded", False),
        )

        # 1. Build evidence using the exact batch pipeline path
        logger.info("Building batch evidence (vendor_cap=%d)...", vendor_cap)
        evidence_list = await build_batch_evidence(pool, vendor_cap)
        if not evidence_list:
            logger.error("No vendors produced evidence")
            return 1
        for ev in evidence_list:
            logger.info("  %s: %d fields", ev["vendor_name"], len(ev))

        # 2. Phase A
        logger.info("=" * 60)
        logger.info(
            "PHASE A: %s (%d vendors, concurrency=%d, max_tokens=%d, fresh=%s)",
            args.model_a, len(evidence_list), concurrency, args.max_tokens, args.fresh,
        )
        logger.info("=" * 60)
        t0 = time.monotonic()
        results_a = await run_reasoning_phase(
            reasoner, args.model_a, evidence_list,
            concurrency=concurrency, max_tokens=args.max_tokens, fresh=args.fresh,
        )
        wall_a = time.monotonic() - t0

        # 3. Phase B
        logger.info("=" * 60)
        logger.info(
            "PHASE B: %s (%d vendors, concurrency=%d, max_tokens=%d, fresh=%s)",
            args.model_b, len(evidence_list), concurrency, args.max_tokens, args.fresh,
        )
        logger.info("=" * 60)
        t0 = time.monotonic()
        results_b = await run_reasoning_phase(
            reasoner, args.model_b, evidence_list,
            concurrency=concurrency, max_tokens=args.max_tokens, fresh=args.fresh,
        )
        wall_b = time.monotonic() - t0

        # 4. Comparison
        print_comparison(results_a, results_b)
        print(f"\n  Wall time (A):         {wall_a:.1f}s")
        print(f"  Wall time (B):         {wall_b:.1f}s")

        # 5. Save JSON
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_a": args.model_a,
                "model_b": args.model_b,
                "max_tokens": args.max_tokens,
                "concurrency": concurrency,
                "vendor_cap": vendor_cap,
                "fresh": args.fresh,
                "vendors": len(evidence_list),
                "wall_time_a_s": round(wall_a, 2),
                "wall_time_b_s": round(wall_b, 2),
                "evidence": evidence_list,
                "results_a": results_a,
                "results_b": results_b,
            }
            output_path.write_text(json.dumps(payload, indent=2, default=str))
            logger.info("Results saved to %s", output_path)

        # 6. Clean up benchmark cache entries (--fresh mode only)
        if _benchmark_sigs:
            cleaned = 0
            for sig in _benchmark_sigs:
                try:
                    await reasoner._cache.invalidate(sig, reason="benchmark_cleanup")
                    cleaned += 1
                except Exception:
                    pass
            logger.info("Cleaned up %d benchmark cache entries", cleaned)

    finally:
        # Restore original env vars
        for k in _env_keys:
            orig = _saved_env.get(k)
            if orig is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig

        # Proper shutdown
        try:
            await close_stratified_reasoner()
        except Exception:
            pass
        try:
            await pool.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
