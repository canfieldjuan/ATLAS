#!/usr/bin/env python3
"""Audit witness selection across vendors.

Surfaces the gap that the Slack canary exposed: Phase 5/6 wiring can be
correct yet still invisible if the selector keeps preferring stale v3
rows over corrected v4 rows. This script does NOT change anything; it
only reports what the selector picked, alongside diagnostic fields.

Per-vendor it prints, for the latest snapshot:
  - witness_id (truncated), witness_type, selection_reason
  - source review's enrichment_schema_version
  - salience_score, specificity_score, generic_reason
  - pain_category, pain_confidence
  - phrase_subject, phrase_polarity, phrase_role, phrase_verbatim
  - grounding_status (and derived quote_grade)

Per-vendor summary lines tally:
  - v3-backed witnesses vs v4-backed
  - witnesses with at least one phrase_* tag set
  - witnesses where grounding_status='grounded'

The summary lets us spot vendors where the candidate pool has v4 rows
available but the witness pack came out predominantly v3 (the Slack
pattern). Use --vendor to drill into a single vendor.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  # Summary across vendors with at least 6 witnesses (default)
  python scripts/audit_witness_selection.py
  # Drilldown on one vendor (full table)
  python scripts/audit_witness_selection.py --vendor "Slack"
  # Restrict to vendors with v4 candidates available but stale-looking packs
  python scripts/audit_witness_selection.py --suspicious-only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)

from atlas_brain.storage.database import close_database, get_db_pool, init_database

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


_PER_VENDOR_HEADER = (
    f"{'witness_id':<32} {'type':<18} {'reason':<28} {'sv':>2} "
    f"{'sal':>5} {'spec':>5} {'pain_cat':<22} {'conf':<6} "
    f"{'subj':<14} {'pol':<8} {'role':<18} {'vrb':<5} {'grnd':<14}"
)


def _fmt_witness_row(r: dict) -> str:
    wid = (r["witness_id"] or "")[:30]
    wtype = str(r["witness_type"] or "")[:18]
    reason = str(r["selection_reason"] or "")[:28]
    sv = r["sv"] if r["sv"] is not None else "?"
    sal = r["salience_score"] if r["salience_score"] is not None else 0.0
    spec = r["specificity_score"] if r["specificity_score"] is not None else 0.0
    pain = str(r["pain_category"] or "")[:22]
    conf = str(r["pain_confidence"] or "")[:6]
    subj = str(r["phrase_subject"] or "")[:14]
    pol = str(r["phrase_polarity"] or "")[:8]
    role = str(r["phrase_role"] or "")[:18]
    if r["phrase_verbatim"] is True:
        vrb = "true"
    elif r["phrase_verbatim"] is False:
        vrb = "false"
    else:
        vrb = ""
    grnd = str(r["grounding_status"] or "")[:14]
    return (
        f"{wid:<32} {wtype:<18} {reason:<28} {str(sv):>2} "
        f"{float(sal):>5.2f} {float(spec):>5.2f} {pain:<22} {conf:<6} "
        f"{subj:<14} {pol:<8} {role:<18} {vrb:<5} {grnd:<14}"
    )


_LATEST_WITNESSES_QUERY = """
WITH latest AS (
    SELECT vendor_name, analysis_window_days, schema_version,
           MAX(as_of_date) AS as_of_date
    FROM b2b_vendor_witnesses
    {vendor_clause}
    GROUP BY 1, 2, 3
)
SELECT w.vendor_name, w.witness_id, w.witness_type, w.selection_reason,
       w.salience_score, w.specificity_score, w.generic_reason,
       w.pain_category, w.pain_confidence,
       w.phrase_subject, w.phrase_polarity, w.phrase_role, w.phrase_verbatim,
       w.grounding_status, w.review_id,
       COALESCE((r.enrichment->>'enrichment_schema_version')::int, 0) AS sv
FROM b2b_vendor_witnesses w
JOIN latest USING (vendor_name, analysis_window_days, schema_version, as_of_date)
LEFT JOIN b2b_reviews r ON r.id = w.review_id::uuid
{post_filter}
ORDER BY w.vendor_name, w.witness_id
"""


_VENDOR_POOL_QUERY = """
SELECT vendor_name,
       COUNT(*) FILTER (WHERE COALESCE((enrichment->>'enrichment_schema_version')::int, 0) >= 4) AS v4_pool,
       COUNT(*) FILTER (WHERE COALESCE((enrichment->>'enrichment_schema_version')::int, 0) < 4) AS v3_pool
FROM b2b_reviews
WHERE enrichment_status = 'enriched'
  AND duplicate_of_review_id IS NULL
  AND COALESCE(reviewed_at, imported_at) >= NOW() - INTERVAL '30 days'
GROUP BY 1
"""


async def _audit(pool, *, vendor: str | None, suspicious_only: bool) -> int:
    if vendor:
        clause = "WHERE vendor_name = $1"
        args: list = [vendor]
    else:
        clause = ""
        args = []
    sql = _LATEST_WITNESSES_QUERY.format(
        vendor_clause=clause,
        post_filter="WHERE w.vendor_name = $1" if vendor else "",
    )
    rows = await pool.fetch(sql, *args)
    if not rows:
        print(f"no witnesses found{' for ' + vendor if vendor else ''}")
        return 0

    pool_rows = await pool.fetch(_VENDOR_POOL_QUERY)
    pool_map = {p["vendor_name"]: (p["v4_pool"], p["v3_pool"]) for p in pool_rows}

    by_vendor: dict[str, list] = {}
    for row in rows:
        by_vendor.setdefault(row["vendor_name"], []).append(dict(row))

    suspicious_rows: list[tuple[str, int, int, int, int]] = []
    for v, vrows in by_vendor.items():
        v4_pool, v3_pool = pool_map.get(v, (0, 0))
        v4_backed = sum(1 for r in vrows if r["sv"] >= 4)
        v3_backed = sum(1 for r in vrows if r["sv"] < 4)
        # Suspicious: vendor has v4 candidates available, but witness pack
        # is dominated by v3-backed rows (>=80% v3 when v4 was available).
        susp = (
            v4_pool >= 5
            and len(vrows) >= 4
            and v3_backed / len(vrows) >= 0.8
        )
        if suspicious_only and not susp:
            continue
        suspicious_rows.append((v, v4_pool, v3_pool, v4_backed, v3_backed))

    if vendor:
        # Drilldown table
        print(_PER_VENDOR_HEADER)
        print("-" * len(_PER_VENDOR_HEADER))
        for r in by_vendor.get(vendor, []):
            print(_fmt_witness_row(r))
        v4p, v3p = pool_map.get(vendor, (0, 0))
        v4b = sum(1 for r in by_vendor[vendor] if r["sv"] >= 4)
        v3b = sum(1 for r in by_vendor[vendor] if r["sv"] < 4)
        print()
        print(
            f"summary: vendor={vendor} "
            f"v4_pool_30d={v4p} v3_pool_30d={v3p} "
            f"witnesses_v4_backed={v4b} witnesses_v3_backed={v3b}"
        )
        return 0

    # Aggregate summary across vendors
    print(
        f"{'vendor':<28} {'v4_pool':>7} {'v3_pool':>7} "
        f"{'v4_wit':>6} {'v3_wit':>6} {'tagged':>6} {'gnd':>4} {'flag':<10}"
    )
    print("-" * 80)
    for v, v4_pool, v3_pool, v4_backed, v3_backed in sorted(suspicious_rows):
        vrows = by_vendor[v]
        tagged = sum(1 for r in vrows if r["phrase_polarity"])
        grounded = sum(1 for r in vrows if r["grounding_status"] == "grounded")
        flag = ""
        if v4_pool >= 5 and v3_backed / max(len(vrows), 1) >= 0.8:
            flag = "STALE_PACK"
        print(
            f"{v[:28]:<28} {v4_pool:>7} {v3_pool:>7} "
            f"{v4_backed:>6} {v3_backed:>6} {tagged:>6} {grounded:>4} {flag:<10}"
        )
    return 0


async def _main_async(args: argparse.Namespace) -> int:
    await init_database()
    pool = get_db_pool()
    try:
        return await _audit(
            pool,
            vendor=args.vendor,
            suspicious_only=args.suspicious_only,
        )
    finally:
        await close_database()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor",
        type=str,
        default=None,
        help="Drill down to a single vendor; prints the per-witness table.",
    )
    parser.add_argument(
        "--suspicious-only",
        action="store_true",
        help="Aggregate view: only show vendors where the witness pack is "
             ">=80%% v3-backed despite >=5 v4 candidates in the 30-day pool.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
