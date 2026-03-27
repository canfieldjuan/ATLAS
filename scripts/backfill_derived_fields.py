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

from atlas_brain.reasoning.evidence_engine import get_evidence_engine
from atlas_brain.storage.database import get_db_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_derived_fields")

BATCH_SIZE = 200

_FILTER_QUERIES = {
    "all": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (enrichment->>'enrichment_schema_version') IS NULL
        ORDER BY enriched_at DESC
    """,
    "urgency_compressed": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (enrichment->>'enrichment_schema_version') IS NULL
          AND (enrichment->>'urgency_score')::numeric BETWEEN 3.0 AND 4.5
        ORDER BY enriched_at DESC
    """,
    "pain_other": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (enrichment->>'enrichment_schema_version') IS NULL
          AND enrichment->>'pain_category' = 'other'
        ORDER BY enriched_at DESC
    """,
    "no_recommend": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (enrichment->>'enrichment_schema_version') IS NULL
          AND enrichment->>'would_recommend' IS NULL
          AND rating IS NOT NULL
        ORDER BY enriched_at DESC
    """,
}


def _compute_for_row(row, engine) -> tuple[dict, dict]:
    """Run evidence engine compute on a single row. Returns (old_values, new_enrichment)."""
    enrichment = row["enrichment"]
    if isinstance(enrichment, str):
        enrichment = json.loads(enrichment)
    enrichment = dict(enrichment)

    raw_meta = row.get("raw_metadata") or {}
    if isinstance(raw_meta, str):
        raw_meta = json.loads(raw_meta)
    source_weight = float(raw_meta.get("source_weight", 0.7))
    content_type = row.get("content_type") or enrichment.get("content_classification") or "review"
    rating = float(row["rating"]) if row.get("rating") is not None else None
    rating_max = float(row.get("rating_max") or 5)

    indicators = enrichment.get("urgency_indicators", {})
    complaints = enrichment.get("specific_complaints", [])
    quotable = enrichment.get("quotable_phrases", [])
    pricing_phrases = enrichment.get("pricing_phrases", [])
    rec_lang = enrichment.get("recommendation_language", [])
    events = enrichment.get("event_mentions", [])
    pain_cats = enrichment.get("pain_categories", [])

    old_values = {
        "urgency_score": enrichment.get("urgency_score"),
        "pain_category": enrichment.get("pain_category"),
        "would_recommend": enrichment.get("would_recommend"),
    }

    # Compute urgency -- only if we have indicators (v2 reviews)
    # For v1 reviews without urgency_indicators, use a simplified version
    if indicators:
        enrichment["urgency_score"] = engine.compute_urgency(
            indicators, rating, rating_max, content_type, source_weight,
        )

    # Override pain "other" using keyword scan (works on v1 reviews too)
    pain_cat = enrichment.get("pain_category", "other")
    if pain_cat == "other":
        enrichment["pain_category"] = engine.override_pain(pain_cat, complaints, quotable)

    # Derive would_recommend (works on v1 reviews with rating fallback)
    if rec_lang:
        enrichment["would_recommend"] = engine.derive_recommend(rec_lang, rating, rating_max)
    elif enrichment.get("would_recommend") is None and rating is not None:
        enrichment["would_recommend"] = engine.derive_recommend([], rating, rating_max)

    # Derive price_complaint from existing data
    enrichment.setdefault("contract_context", {})
    cc = enrichment["contract_context"]
    if isinstance(cc, dict):
        cc["price_complaint"] = engine.derive_price_complaint(enrichment)
        if pricing_phrases:
            cc["price_context"] = pricing_phrases[0]

    # Derive has_budget_authority
    enrichment.setdefault("buyer_authority", {})
    ba = enrichment["buyer_authority"]
    if isinstance(ba, dict):
        ba["has_budget_authority"] = engine.derive_budget_authority(enrichment)

    # Turning point from event_mentions
    st = enrichment.get("sentiment_trajectory")
    if isinstance(st, dict) and events:
        first = events[0] if isinstance(events[0], dict) else {}
        event_text = str(first.get("event", "")).strip()
        timeframe = str(first.get("timeframe", "")).strip()
        if event_text:
            st["turning_point"] = f"{event_text} ({timeframe})" if timeframe and timeframe.lower() != "null" else event_text

    enrichment["enrichment_schema_version"] = 2
    return old_values, enrichment


async def _run(args):
    pool = get_db_pool()
    if not pool.is_initialized:
        await pool.initialize()

    engine = get_evidence_engine()

    query = _FILTER_QUERIES.get(args.filter, _FILTER_QUERIES["all"])
    if args.limit:
        query += f" LIMIT {int(args.limit)}"

    rows = await pool.fetch(query)
    logger.info("Found %d reviews to backfill (filter=%s)", len(rows), args.filter)

    updated = 0
    skipped = 0
    changes = {"urgency_score": 0, "pain_category": 0, "would_recommend": 0}

    for row in rows:
        try:
            old, new_enrichment = _compute_for_row(row, engine)
        except Exception:
            logger.warning("Failed to compute for review %s", row["id"], exc_info=True)
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

        if not changed:
            skipped += 1
            continue

        if args.dry_run:
            logger.info(
                "DRY RUN %s: urgency %.1f->%.1f, pain %s->%s, recommend %s->%s",
                row["id"],
                old["urgency_score"] or 0,
                new_enrichment.get("urgency_score", 0),
                old["pain_category"],
                new_enrichment.get("pain_category"),
                old["would_recommend"],
                new_enrichment.get("would_recommend"),
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
        "%s %d reviews, skipped %d unchanged. Changes: urgency=%d, pain=%d, recommend=%d",
        prefix, updated, skipped,
        changes["urgency_score"], changes["pain_category"], changes["would_recommend"],
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
