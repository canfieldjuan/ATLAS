#!/usr/bin/env python3
"""Reset enrichment baseline for clean three-layer re-enrichment.

Wipes all enrichment-derived data and resets reviews to pending.
Raw review text, ratings, and scrape metadata are untouched.

WARNING: This is destructive. Run with --dry-run first.

Usage:
    # See what would happen (no changes)
    python scripts/reset_enrichment_baseline.py --dry-run

    # Execute the reset
    python scripts/reset_enrichment_baseline.py --confirm

    # Reset only enrichment (skip derived table truncation)
    python scripts/reset_enrichment_baseline.py --confirm --enrichment-only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("reset_enrichment_baseline")

# Tables derived from enrichment, ordered by dependency (leaf first).
# Every table here is rebuilt by b2b_churn_intelligence or downstream tasks.
_DERIVED_TABLES = [
    "b2b_intelligence",
    "b2b_cross_vendor_conclusions",
    "b2b_reasoning_synthesis",
    "b2b_account_intelligence",
    "b2b_category_dynamics",
    "b2b_displacement_dynamics",
    "b2b_temporal_intelligence",
    "b2b_segment_intelligence",
    "b2b_evidence_vault",
    "b2b_article_correlations",
    "b2b_change_events",
    "b2b_vendor_snapshots",
    "b2b_vendor_buyer_profiles",
    "b2b_vendor_integrations",
    "b2b_vendor_use_cases",
    "b2b_vendor_pain_points",
    "b2b_company_signals",
    "b2b_displacement_edges",
    "b2b_churn_signals",
]

# Tables that must NEVER be touched.
_PROTECTED_TABLES = [
    "b2b_reviews",
    "b2b_vendors",
    "b2b_scrape_targets",
    "b2b_product_profiles",
]


async def _get_table_counts(pool, tables: list[str]) -> dict[str, int]:
    counts = {}
    for table in tables:
        try:
            n = await pool.fetchval(f"SELECT count(*) FROM {table}")
            counts[table] = n
        except Exception:
            counts[table] = -1
    return counts


async def _get_review_status_counts(pool) -> dict[str, int]:
    rows = await pool.fetch(
        "SELECT enrichment_status, count(*) as n FROM b2b_reviews GROUP BY enrichment_status"
    )
    return {r["enrichment_status"]: r["n"] for r in rows}


async def _run(args):
    from atlas_brain.storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        await pool.initialize()

    # Pre-flight: show current state
    logger.info("=== PRE-FLIGHT CHECK ===")

    status_counts = await _get_review_status_counts(pool)
    total_reviews = sum(status_counts.values())
    logger.info("Review status breakdown:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-15s %6d", status, count)
    logger.info("  %-15s %6d", "TOTAL", total_reviews)

    if not args.enrichment_only:
        table_counts = await _get_table_counts(pool, _DERIVED_TABLES)
        total_derived_rows = sum(v for v in table_counts.values() if v > 0)
        logger.info("Derived table row counts:")
        for table, count in table_counts.items():
            if count > 0:
                logger.info("  %-40s %6d", table, count)
            elif count == -1:
                logger.info("  %-40s  (table not found)", table)
        logger.info("  %-40s %6d", "TOTAL DERIVED ROWS", total_derived_rows)

    # Reviews that will be reset to pending (all non-pending statuses)
    resettable = sum(
        status_counts.get(s, 0)
        for s in ("enriched", "no_signal", "failed", "quarantined", "filtered", "not_applicable")
    )
    already_pending = status_counts.get("pending", 0)
    logger.info(
        "Reviews to reset: %d (already pending: %d)",
        resettable, already_pending,
    )

    if args.dry_run:
        logger.info("=== DRY RUN === No changes made.")
        return

    if not args.confirm:
        logger.error("Must pass --confirm to execute. Use --dry-run to preview.")
        sys.exit(1)

    # Safety: require explicit confirmation token
    logger.info("")
    logger.info("=== EXECUTING RESET ===")
    logger.info("This will:")
    logger.info("  1. Reset %d reviews to pending (wipe enrichment JSONB)", resettable)
    if not args.enrichment_only:
        logger.info("  2. Truncate %d derived tables", len(_DERIVED_TABLES))
    logger.info("")

    # Step 1: Truncate derived tables (leaf-first order)
    if not args.enrichment_only:
        for table in _DERIVED_TABLES:
            try:
                count = await pool.fetchval(f"SELECT count(*) FROM {table}")
                if count == 0:
                    continue
                await pool.execute(f"TRUNCATE {table} CASCADE")
                logger.info("  TRUNCATED %-40s (%d rows)", table, count)
            except Exception as exc:
                logger.warning("  SKIP %-40s (%s)", table, exc)

    # Step 2: Reset reviews
    reset_count = await pool.fetchval(
        """
        WITH updated AS (
            UPDATE b2b_reviews
            SET enrichment = NULL,
                enrichment_status = 'pending',
                enrichment_attempts = 0,
                enrichment_model = NULL,
                enriched_at = NULL,
                sentiment_direction = NULL,
                sentiment_tenure = NULL,
                sentiment_turning_point = NULL,
                low_fidelity = false,
                low_fidelity_reasons = '[]'::jsonb,
                low_fidelity_detected_at = NULL,
                enrichment_repair = NULL,
                enrichment_repair_status = NULL,
                enrichment_repair_attempts = 0,
                enrichment_repair_model = NULL,
                enrichment_repaired_at = NULL,
                enrichment_repair_applied_fields = '[]'::jsonb,
                requeue_reason = 'baseline_reset'
            WHERE enrichment_status IN ('enriched', 'no_signal', 'failed', 'quarantined', 'filtered', 'not_applicable')
            RETURNING 1
        )
        SELECT count(*) FROM updated
        """
    )
    logger.info("  RESET %d reviews to pending", reset_count)

    # Post-flight: verify
    logger.info("")
    logger.info("=== POST-FLIGHT VERIFICATION ===")
    post_status = await _get_review_status_counts(pool)
    for status, count in sorted(post_status.items(), key=lambda x: -x[1]):
        logger.info("  %-15s %6d", status, count)

    if not args.enrichment_only:
        post_counts = await _get_table_counts(pool, _DERIVED_TABLES)
        non_empty = {t: c for t, c in post_counts.items() if c > 0}
        if non_empty:
            logger.warning("  Non-empty derived tables after truncation:")
            for t, c in non_empty.items():
                logger.warning("    %-40s %6d", t, c)
        else:
            logger.info("  All derived tables empty.")

    logger.info("")
    logger.info("Reset complete. Next steps:")
    logger.info("  1. Start the server to begin re-enrichment")
    logger.info("  2. Monitor: SELECT enrichment_status, count(*) FROM b2b_reviews GROUP BY enrichment_status;")
    logger.info("  3. After enrichment completes, run b2b_churn_intelligence to rebuild derived tables")


def main():
    parser = argparse.ArgumentParser(
        description="Reset enrichment baseline for clean three-layer re-enrichment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show current state and what would change, no modifications",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required to execute the reset (safety gate)",
    )
    parser.add_argument(
        "--enrichment-only",
        action="store_true",
        help="Only reset review enrichment status, skip derived table truncation",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
