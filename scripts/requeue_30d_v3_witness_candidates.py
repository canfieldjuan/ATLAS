#!/usr/bin/env python3
"""Phase 7 Step B: re-queue 30-day v3 witness candidates for v4 re-enrichment.

Targets reviews that:
  - have enrichment_status = 'enriched'
  - are NOT duplicates (duplicate_of_review_id IS NULL)
  - currently have schema_version < 4 (i.e. extracted under the v3 prompt)
  - were reviewed (or imported, whichever is more recent) within the last
    30 days (the active synthesis window)
  - have at least one row in b2b_review_vendor_mentions (i.e. would actually
    be a witness candidate for some vendor)

For each match, sets enrichment_status='pending' with the canonical reset
fields used by _queue_parser_upgrades / _queue_model_upgrades:
  - enrichment_attempts = 0
  - requeue_reason = 'phase_7_v2_phrase_metadata'
  - low_fidelity = false / low_fidelity_reasons = '[]'
  - enrichment_repair* fields cleared

This script does NOT call the LLM. After it runs, drain the queue with:

  ATLAS_B2B_CHURN_ENABLED=true \\
    python scripts/run_b2b_enrichment_until_exhausted.py

Once the queue drains, run vendor synthesis so witness rows rebuild from
the new v4 enrichments and Phase 5/6 fields populate.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  # Inspect scope without writing
  python scripts/requeue_30d_v3_witness_candidates.py --dry-run
  # Real run
  python scripts/requeue_30d_v3_witness_candidates.py
  # Constrain to a specific vendor for surgical testing
  python scripts/requeue_30d_v3_witness_candidates.py --vendor "Wrike" --dry-run

Exit codes:
  0 = success (or dry-run completed)
  1 = DB / setup error
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
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("requeue_30d_v3")


_REQUEUE_REASON = "phase_7_v2_phrase_metadata"
_WINDOW_DAYS_DEFAULT = 30


# Scope CTE shared by audit + UPDATE so they identify exactly the same rows.
# The vendor filter is appended via {vendor_clause} when --vendor is set.
_SCOPE_CTE = """
WITH scope AS (
    SELECT DISTINCT r.id, r.vendor_name,
                    r.reviewed_at, r.imported_at,
                    COALESCE((r.enrichment->>'enrichment_schema_version')::int, 0)
                        AS schema_version
      FROM b2b_reviews r
      JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
     WHERE r.enrichment_status = 'enriched'
       AND r.duplicate_of_review_id IS NULL
       AND COALESCE((r.enrichment->>'enrichment_schema_version')::int, 0) < 4
       AND COALESCE(r.reviewed_at, r.imported_at)
            >= NOW() - ($1::int * INTERVAL '1 day')
       {vendor_clause}
)
"""


async def _audit(pool, *, window_days: int, vendor: str | None) -> dict:
    vendor_clause = "AND r.vendor_name = $2" if vendor else ""
    cte = _SCOPE_CTE.format(vendor_clause=vendor_clause)

    args: list = [window_days]
    if vendor:
        args.append(vendor)

    total = await pool.fetchval(cte + " SELECT COUNT(*) FROM scope", *args)
    by_vendor = await pool.fetch(
        cte + " SELECT vendor_name, COUNT(*) AS n FROM scope GROUP BY 1 "
              "ORDER BY 2 DESC, 1 LIMIT 20",
        *args,
    )
    by_version = await pool.fetch(
        cte + " SELECT schema_version, COUNT(*) AS n FROM scope GROUP BY 1 "
              "ORDER BY 1",
        *args,
    )
    bounds = await pool.fetchrow(
        cte + " SELECT MIN(COALESCE(reviewed_at, imported_at)) AS oldest, "
              "MAX(COALESCE(reviewed_at, imported_at)) AS newest FROM scope",
        *args,
    )
    return {
        "total": total or 0,
        "by_vendor": [(r["vendor_name"], r["n"]) for r in by_vendor],
        "by_version": [(r["schema_version"], r["n"]) for r in by_version],
        "oldest": bounds["oldest"] if bounds else None,
        "newest": bounds["newest"] if bounds else None,
    }


async def _apply_requeue(pool, *, window_days: int, vendor: str | None) -> int:
    vendor_clause = "AND r.vendor_name = $2" if vendor else ""
    cte = _SCOPE_CTE.format(vendor_clause=vendor_clause)

    args: list = [window_days]
    if vendor:
        args.append(vendor)

    # Reset the same fields that _queue_parser_upgrades / _queue_model_upgrades
    # touch so we hit the same canonical "ready for re-extraction" state. Do
    # NOT clear enrichment itself -- the re-extraction path overwrites it
    # whole, and keeping the prior payload during the in-flight window means
    # synthesis (if it runs concurrently) still has SOMETHING to read.
    sql = cte + """
        UPDATE b2b_reviews
           SET enrichment_status = 'pending',
               enrichment_attempts = 0,
               requeue_reason = $%(reason_idx)d,
               low_fidelity = false,
               low_fidelity_reasons = '[]'::jsonb,
               low_fidelity_detected_at = NULL,
               enrichment_repair = NULL,
               enrichment_repair_status = NULL,
               enrichment_repair_attempts = 0,
               enrichment_repair_model = NULL,
               enrichment_repaired_at = NULL,
               enrichment_repair_applied_fields = '[]'::jsonb
         WHERE id IN (SELECT id FROM scope)
        RETURNING 1
    """
    # Position the requeue_reason placeholder after the scope args.
    reason_idx = len(args) + 1
    sql = sql % {"reason_idx": reason_idx}
    args.append(_REQUEUE_REASON)

    rows = await pool.fetch(sql, *args)
    return len(rows)


async def _main_async(args: argparse.Namespace) -> int:
    await init_database()
    pool = get_db_pool()
    try:
        audit = await _audit(
            pool,
            window_days=args.window_days,
            vendor=args.vendor,
        )

        scope_label = (
            f"vendor={args.vendor}, "
            if args.vendor
            else ""
        )
        scope_label += (
            f"window={args.window_days}d, enriched, non-dup, schema<4, "
            "has vendor_mention"
        )

        logger.info("scope: %s", scope_label)
        logger.info("rows in scope: %d", audit["total"])
        if audit["total"] == 0:
            logger.info("nothing to re-queue")
            return 0

        logger.info(
            "reviewed_at bounds: oldest=%s newest=%s",
            audit["oldest"],
            audit["newest"],
        )
        logger.info("schema version mix: %s", audit["by_version"])
        logger.info("top vendors in scope:")
        for vendor_name, n in audit["by_vendor"]:
            logger.info("  %s -> %d", vendor_name, n)

        if args.dry_run:
            logger.info("[DRY RUN] no rows updated")
            return 0

        n = await _apply_requeue(
            pool,
            window_days=args.window_days,
            vendor=args.vendor,
        )
        logger.info("re-queued %d reviews to enrichment_status='pending'", n)
        logger.info(
            "next: drain the pending queue with "
            "scripts/run_b2b_enrichment_until_exhausted.py"
        )
        return 0
    finally:
        await close_database()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-days",
        type=int,
        default=_WINDOW_DAYS_DEFAULT,
        help=f"Recency window (default {_WINDOW_DAYS_DEFAULT}).",
    )
    parser.add_argument(
        "--vendor",
        type=str,
        default=None,
        help="Restrict scope to a single vendor_name (exact match).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scope audit without UPDATEing any rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
