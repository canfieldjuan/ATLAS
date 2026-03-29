#!/usr/bin/env python3
"""Run b2b_churn_intelligence + vendor_scorecard for a small vendor subset.

Usage:
    ATLAS_B2B_CHURN_ENABLED=true python scripts/test_intelligence_subset.py
    ATLAS_B2B_CHURN_ENABLED=true python scripts/test_intelligence_subset.py --vendors "Shopify,Salesforce,Jira"
    ATLAS_B2B_CHURN_ENABLED=true python scripts/test_intelligence_subset.py --vendors Shopify --report vendor_scorecard
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.config import settings
from atlas_brain.storage.database import close_database, get_db_pool, init_database

DEFAULT_VENDORS = "Shopify,Salesforce,Jira"
VALID_REPORT_TYPES = ("weekly_churn_feed", "vendor_scorecard", "displacement_report", "category_overview")


async def _run_intelligence(vendors: str) -> dict:
    from atlas_brain.autonomous.tasks.b2b_churn_intelligence import run
    task = SimpleNamespace(metadata={"test_vendors": vendors})
    print(f"\n[1/2] Running intelligence for: {vendors}")
    print(f"      LLM backend: {settings.b2b_churn.intelligence_llm_backend}")
    result = await run(task)
    print(f"      Done: {json.dumps({k: v for k, v in result.items() if k != '_skip_synthesis'}, indent=2)}")
    return result


async def _fetch_scorecard(vendor: str, pool) -> dict | None:
    row = await pool.fetchrow(
        """
        SELECT intelligence_data, report_date, llm_model
        FROM b2b_intelligence
        WHERE report_type = 'vendor_scorecard'
          AND LOWER(vendor_filter) = LOWER($1)
        ORDER BY report_date DESC
        LIMIT 1
        """,
        vendor,
    )
    return dict(row) if row else None


async def _fetch_weekly_feed(pool) -> dict | None:
    row = await pool.fetchrow(
        """
        SELECT intelligence_data, report_date
        FROM b2b_intelligence
        WHERE report_type = 'weekly_churn_feed'
          AND vendor_filter IS NULL
        ORDER BY report_date DESC
        LIMIT 1
        """,
    )
    return dict(row) if row else None


async def _run_reports(vendors: list[str]) -> None:
    from atlas_brain.autonomous.tasks.b2b_churn_reports import run as reports_run
    task = SimpleNamespace(metadata={})
    print("\n[2/2] Running reports task...")
    result = await reports_run(task)
    print(f"      Done: {result.get('_skip_synthesis', result)}")


async def _main(args: argparse.Namespace) -> None:
    await init_database()
    pool = get_db_pool()
    cfg = settings.b2b_churn

    if not cfg.enabled:
        print("ERROR: B2B churn pipeline disabled. Set ATLAS_B2B_CHURN_ENABLED=true")
        await close_database()
        sys.exit(1)

    vendor_list = [v.strip() for v in args.vendors.split(",") if v.strip()]
    print(f"\nVendor subset : {vendor_list}")
    print(f"Report type   : {args.report}")
    print(f"LLM backend   : {cfg.intelligence_llm_backend}")

    # Step 1: run intelligence scoped to vendor subset
    await _run_intelligence(args.vendors)

    # Step 2: run reports to build scorecards from the fresh signals
    if args.report != "weekly_churn_feed":
        await _run_reports(vendor_list)

    # Step 3: fetch and print signals directly from b2b_churn_signals (bypasses limit=15 in scorecard builder)
    print(f"\n{'='*60}")
    print(f"SIGNALS: {vendor_list}")
    print('='*60)

    rows = await pool.fetch(
        """
        SELECT
            vendor_name, product_category, total_reviews, churn_intent_count,
            avg_urgency_score, archetype, archetype_confidence,
            reasoning_mode, reasoning_risk_level, reasoning_executive_summary,
            reasoning_key_signals, falsification_conditions, reasoning_uncertainty_sources,
            top_pain_categories, top_competitors, price_complaint_rate,
            decision_maker_churn_rate, confidence_score, last_computed_at
        FROM b2b_churn_signals
        WHERE LOWER(vendor_name) = ANY($1::text[])
        ORDER BY vendor_name, total_reviews DESC
        """,
        [v.lower() for v in vendor_list],
    )

    if not rows:
        print(f"No signals found for {vendor_list}. Intelligence may not have run yet.")
    else:
        for r in rows:
            d = dict(r)
            print(f"\n--- {d['vendor_name']} ({d.get('archetype', 'unknown')}) ---")
            print(f"Reviews: {d['total_reviews']} | Churn intents: {d['churn_intent_count']} | Avg urgency: {d.get('avg_urgency_score')}")
            print(f"Confidence: {d.get('confidence_score')} | DM churn rate: {d.get('decision_maker_churn_rate')}")
            print(f"Risk level: {d.get('reasoning_risk_level')} | Mode: {d.get('reasoning_mode')}")
            print(f"\nExecutive summary:\n  {d.get('reasoning_executive_summary', 'N/A')}")
            key_signals = d.get("reasoning_key_signals")
            if key_signals:
                signals = json.loads(key_signals) if isinstance(key_signals, str) else key_signals
                print(f"\nKey signals:")
                for s in (signals if isinstance(signals, list) else [signals]):
                    print(f"  - {s}")
            falsification = d.get("falsification_conditions")
            if falsification:
                fc = json.loads(falsification) if isinstance(falsification, str) else falsification
                print(f"\nFalsification conditions:")
                for f_item in (fc if isinstance(fc, list) else [fc]):
                    print(f"  - {f_item}")
            pain_cats = d.get("top_pain_categories")
            if pain_cats:
                pc = json.loads(pain_cats) if isinstance(pain_cats, str) else pain_cats
                print(f"\nTop pain categories: {pc}")
            competitors = d.get("top_competitors")
            if competitors:
                comp = json.loads(competitors) if isinstance(competitors, str) else competitors
                print(f"Top competitors: {comp}")
            print(f"\nLast computed: {d.get('last_computed_at')}")

    await close_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendors", default=DEFAULT_VENDORS, help="Comma-separated vendor names")
    parser.add_argument("--report", default="vendor_scorecard", choices=VALID_REPORT_TYPES)
    asyncio.run(_main(parser.parse_args()))
