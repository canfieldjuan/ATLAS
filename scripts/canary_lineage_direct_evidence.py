#!/usr/bin/env python3
"""Side-by-side canary: row-level vs lineage direct_evidence_count.

Phase 10 Patch 2d helper. Flips ClaimGatePolicy.use_claim_lineage_for_
direct_evidence in-process for one or more vendors, runs both rate
aggregators twice (default + lineage), and prints the diff. Restores
the default policy at the end so global state is unchanged.

The canary answers: does flipping the lineage flag on this dataset
escalate or suppress any rate card vs the v1 row-level grounding
approximation? It does NOT mutate the DB; it only re-runs read-side
aggregators with two different policy configurations.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate

  # default canary set (vendors with seeded b2b_evidence_claims data)
  python scripts/canary_lineage_direct_evidence.py

  # specific vendors / window
  python scripts/canary_lineage_direct_evidence.py \\
      --vendors ClickUp Pipedrive --window 30

If the row-level and lineage rows agree on all four gate fields
(supporting_count, direct_evidence_count, posture, render_allowed),
the lineage flag is safe to flip in production. If they diverge, read
the diff carefully -- it should always be in the direction of stricter
suppression, never looser.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)


_DEFAULT_VENDORS = ["ClickUp", "Pipedrive", "Monday.com"]


def _format_claim(claim, label: str) -> str:
    if claim is None:
        return f"  {label}: None (no eligible reviews)"
    sup = f"{claim.supporting_count}/{claim.denominator}"
    reason = claim.suppression_reason.value if claim.suppression_reason else "-"
    return (
        f"  {label}: sup={sup} direct={claim.direct_evidence_count} "
        f"posture={claim.evidence_posture.value} "
        f"render={claim.render_allowed} report={claim.report_allowed} "
        f"reason={reason}"
    )


def _diff_marker(a, b) -> str:
    if a is None and b is None:
        return ""
    if a is None or b is None:
        return "  *** divergence: one side is None ***"
    fields = (
        ("direct_evidence_count", a.direct_evidence_count, b.direct_evidence_count),
        ("evidence_posture", a.evidence_posture.value, b.evidence_posture.value),
        ("render_allowed", a.render_allowed, b.render_allowed),
        ("report_allowed", a.report_allowed, b.report_allowed),
    )
    diffs = [name for (name, av, bv) in fields if av != bv]
    if not diffs:
        return "  (paths agree)"
    return f"  *** divergence on: {', '.join(diffs)} ***"


async def _compare(pool, vendor: str, window: int) -> None:
    from atlas_brain.services.b2b.product_claim import ClaimScope, register_policy
    from atlas_brain.services.b2b.vendor_dashboard_claims import (
        _DM_CHURN_RATE_POLICY,
        _PRICE_COMPLAINT_RATE_POLICY,
        aggregate_dm_churn_rate_claim,
        aggregate_price_complaint_rate_claim,
    )

    today = date.today()
    print(f"\n=== {vendor} (window={window}) ===")

    register_policy(ClaimScope.VENDOR, "decision_maker_churn_rate", _DM_CHURN_RATE_POLICY)
    register_policy(ClaimScope.VENDOR, "price_complaint_rate", _PRICE_COMPLAINT_RATE_POLICY)
    dm_baseline = await aggregate_dm_churn_rate_claim(
        pool, vendor_name=vendor, as_of_date=today, analysis_window_days=window
    )
    price_baseline = await aggregate_price_complaint_rate_claim(
        pool, vendor_name=vendor, as_of_date=today, analysis_window_days=window
    )

    register_policy(
        ClaimScope.VENDOR,
        "decision_maker_churn_rate",
        replace(_DM_CHURN_RATE_POLICY, use_claim_lineage_for_direct_evidence=True),
    )
    register_policy(
        ClaimScope.VENDOR,
        "price_complaint_rate",
        replace(_PRICE_COMPLAINT_RATE_POLICY, use_claim_lineage_for_direct_evidence=True),
    )
    dm_lineage = await aggregate_dm_churn_rate_claim(
        pool, vendor_name=vendor, as_of_date=today, analysis_window_days=window
    )
    price_lineage = await aggregate_price_complaint_rate_claim(
        pool, vendor_name=vendor, as_of_date=today, analysis_window_days=window
    )

    register_policy(ClaimScope.VENDOR, "decision_maker_churn_rate", _DM_CHURN_RATE_POLICY)
    register_policy(ClaimScope.VENDOR, "price_complaint_rate", _PRICE_COMPLAINT_RATE_POLICY)

    print("DM churn:")
    print(_format_claim(dm_baseline, "row-level"))
    print(_format_claim(dm_lineage, "lineage  "))
    print(_diff_marker(dm_baseline, dm_lineage))
    print("Price complaint:")
    print(_format_claim(price_baseline, "row-level"))
    print(_format_claim(price_lineage, "lineage  "))
    print(_diff_marker(price_baseline, price_lineage))


async def _main(vendors: list[str], window: int) -> None:
    from atlas_brain.storage.database import close_database, get_db_pool, init_database

    await init_database()
    try:
        pool = get_db_pool()
        for vendor in vendors:
            await _compare(pool, vendor, window)
    finally:
        await close_database()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendors", nargs="+", default=_DEFAULT_VENDORS)
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(_main(args.vendors, args.window))
    return 0


if __name__ == "__main__":
    sys.exit(main())
