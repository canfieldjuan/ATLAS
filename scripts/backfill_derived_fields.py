#!/usr/bin/env python3
"""Backfill derived enrichment fields on existing reviews without calling LLM.

Reads existing enrichment JSONB, runs _compute_derived_fields() from the
Evidence Engine, and writes back the updated JSONB. Fixes urgency compression,
pain "other" overrides, and recommend derivation on existing data at zero LLM
cost.

Usage:
    # Dry run -- show what would change for 100 reviews
    python scripts/backfill_derived_fields.py --dry-run --limit 100

    # Live run -- backfill all enriched v1 reviews
    python scripts/backfill_derived_fields.py

    # Backfill only reviews with compressed urgency
    python scripts/backfill_derived_fields.py --filter urgency_compressed

    # Backfill only "other" pain categories
    python scripts/backfill_derived_fields.py --filter pain_other
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas_brain.storage.database import get_db_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_derived_fields")

BATCH_SIZE = 200

_FILTER_QUERIES = {
    "all": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type,
               summary, review_text, pros, cons, reviewer_title, reviewer_company
        FROM b2b_reviews
        WHERE enrichment_status IN ('enriched', 'quarantined')
        ORDER BY enriched_at DESC
    """,
    "urgency_compressed": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type,
               summary, review_text, pros, cons, reviewer_title, reviewer_company
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (enrichment->>'enrichment_schema_version') IS NULL
          AND (enrichment->>'urgency_score')::numeric BETWEEN 3.0 AND 4.5
        ORDER BY enriched_at DESC
    """,
    "pain_other": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type,
               summary, review_text, pros, cons, reviewer_title, reviewer_company
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (enrichment->>'enrichment_schema_version') IS NULL
          AND enrichment->>'pain_category' = 'other'
        ORDER BY enriched_at DESC
    """,
    "no_recommend": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type,
               summary, review_text, pros, cons, reviewer_title, reviewer_company
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (enrichment->>'enrichment_schema_version') IS NULL
          AND enrichment->>'would_recommend' IS NULL
          AND rating IS NOT NULL
        ORDER BY enriched_at DESC
    """,
    "competitor_mentions": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type,
               summary, review_text, pros, cons, reviewer_title, reviewer_company,
               vendor_name, source, enrichment_status
        FROM b2b_reviews
        WHERE enrichment_status IN ('enriched', 'quarantined')
          AND COALESCE(jsonb_array_length(enrichment->'competitors_mentioned'), 0) > 0
        ORDER BY enriched_at DESC NULLS LAST, imported_at DESC NULLS LAST
    """,
}


def _compute_for_row(row) -> tuple[dict, dict | None]:
    """Run canonical finalization on a single row. Returns (old_values, new_enrichment)."""
    from atlas_brain.autonomous.tasks.b2b_enrichment import _finalize_enrichment_for_persist

    enrichment = row["enrichment"]
    if isinstance(enrichment, str):
        enrichment = json.loads(enrichment)
    enrichment = dict(enrichment)

    old_values = {
        "urgency_score": enrichment.get("urgency_score"),
        "pain_category": enrichment.get("pain_category"),
        "would_recommend": enrichment.get("would_recommend"),
        "replacement_mode": enrichment.get("replacement_mode"),
        "evidence_spans": enrichment.get("evidence_spans"),
        "competitors_mentioned": enrichment.get("competitors_mentioned"),
    }
    finalized, _ = _finalize_enrichment_for_persist(enrichment, dict(row))
    return old_values, finalized


async def _run(args):
    pool = get_db_pool()
    if not pool.is_initialized:
        await pool.initialize()

    query = _FILTER_QUERIES.get(args.filter, _FILTER_QUERIES["all"])
    if args.limit:
        query += f" LIMIT {int(args.limit)}"

    rows = await pool.fetch(query)
    logger.info("Found %d reviews to backfill (filter=%s)", len(rows), args.filter)

    updated = 0
    skipped = 0
    changes = {"urgency_score": 0, "pain_category": 0, "would_recommend": 0, "competitors_mentioned": 0}

    for row in rows:
        try:
            old, new_enrichment = _compute_for_row(row)
        except Exception:
            logger.warning("Failed to compute for review %s", row["id"], exc_info=True)
            skipped += 1
            continue

        if not new_enrichment:
            skipped += 1
            continue

        # Track what changed
        changed = False
        if old["urgency_score"] != new_enrichment.get("urgency_score"):
            changes["urgency_score"] += 1
            changed = True
        if old["pain_category"] != new_enrichment.get("pain_category"):
            changes["pain_category"] += 1
            changed = True
        if old["would_recommend"] != new_enrichment.get("would_recommend"):
            changes["would_recommend"] += 1
            changed = True
        if old["replacement_mode"] != new_enrichment.get("replacement_mode"):
            changed = True
        if old["evidence_spans"] != new_enrichment.get("evidence_spans"):
            changed = True
        if old["competitors_mentioned"] != new_enrichment.get("competitors_mentioned"):
            changes["competitors_mentioned"] += 1
            changed = True

        if not changed:
            skipped += 1
            continue

        if args.dry_run:
            logger.info(
                "DRY RUN %s: urgency %.1f->%.1f, pain %s->%s, recommend %s->%s, competitors %s->%s",
                row["id"],
                old["urgency_score"] or 0,
                new_enrichment.get("urgency_score", 0),
                old["pain_category"],
                new_enrichment.get("pain_category"),
                old["would_recommend"],
                new_enrichment.get("would_recommend"),
                len(old.get("competitors_mentioned") or []),
                len(new_enrichment.get("competitors_mentioned") or []),
            )
            updated += 1
            continue

        await pool.execute(
            "UPDATE b2b_reviews SET enrichment = $1 WHERE id = $2",
            json.dumps(new_enrichment, default=str),
            row["id"],
        )
        updated += 1

    prefix = "DRY RUN: Would update" if args.dry_run else "Updated"
    logger.info(
        "%s %d reviews, skipped %d unchanged. Changes: urgency=%d, pain=%d, recommend=%d, competitors=%d",
        prefix, updated, skipped,
        changes["urgency_score"], changes["pain_category"], changes["would_recommend"], changes["competitors_mentioned"],
    )


def main():
    parser = argparse.ArgumentParser(description="Backfill derived enrichment fields")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument("--limit", type=int, default=0, help="Max reviews to process (0=all)")
    parser.add_argument(
        "--filter",
        choices=list(_FILTER_QUERIES.keys()),
        default="all",
        help="Which reviews to backfill",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
