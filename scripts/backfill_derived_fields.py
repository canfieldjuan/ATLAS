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

    # Backfill one specific review
    python scripts/backfill_derived_fields.py --review-id 972312c9-4a60-47c7-b305-387226ccaa95

    # Backfill reviews where positive pricing language was misclassified as pricing backlash
    python scripts/backfill_derived_fields.py --dry-run --filter positive_pricing_false_positive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas_brain.storage.database import get_db_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_derived_fields")

BATCH_SIZE = 200

# APPROVED-ENRICHMENT-READ: enrichment_schema_version, urgency_score, pain_category, would_recommend, competitors_mentioned
# Reason: backfill/migration script - direct enrichment access required
_FILTER_QUERIES = {
    "all": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type,
               summary, review_text, pros, cons, reviewer_title, reviewer_company,
               vendor_name, source, enrichment_status
        FROM b2b_reviews
        WHERE enrichment_status IN ('enriched', 'quarantined')
        ORDER BY enriched_at DESC
    """,
    "urgency_compressed": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type,
               summary, review_text, pros, cons, reviewer_title, reviewer_company,
               vendor_name, source, enrichment_status
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (enrichment->>'enrichment_schema_version') IS NULL
          AND (enrichment->>'urgency_score')::numeric BETWEEN 3.0 AND 4.5
        ORDER BY enriched_at DESC
    """,
    "pain_other": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type,
               summary, review_text, pros, cons, reviewer_title, reviewer_company,
               vendor_name, source, enrichment_status
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (enrichment->>'enrichment_schema_version') IS NULL
          AND enrichment->>'pain_category' = 'other'
        ORDER BY enriched_at DESC
    """,
    "no_recommend": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type,
               summary, review_text, pros, cons, reviewer_title, reviewer_company,
               vendor_name, source, enrichment_status
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
    # APPROVED-ENRICHMENT-READ: pricing_phrases, contract_context.price_complaint, urgency_indicators.price_pressure_language, evidence_spans.signal_type
    # Reason: backfill review selection for pricing false-positive cleanup
    "positive_pricing_false_positive": """
        SELECT id, enrichment, rating, rating_max, raw_metadata, content_type,
               summary, review_text, pros, cons, reviewer_title, reviewer_company,
               vendor_name, source, enrichment_status
        FROM b2b_reviews
        WHERE enrichment_status IN ('enriched', 'quarantined')
          AND COALESCE(jsonb_array_length(enrichment->'pricing_phrases'), 0) > 0
          AND (
                COALESCE((enrichment->'contract_context'->>'price_complaint')::boolean, false)
             OR COALESCE((enrichment->'urgency_indicators'->>'price_pressure_language')::boolean, false)
             OR jsonb_path_exists(
                    COALESCE(enrichment->'evidence_spans', '[]'::jsonb),
                    '$[*] ? (@.signal_type == "pricing_backlash")'
                )
          )
        ORDER BY enriched_at DESC NULLS LAST, imported_at DESC NULLS LAST
    """,
}

_REVIEW_ID_QUERY = """
    SELECT id, enrichment, rating, rating_max, raw_metadata, content_type,
           summary, review_text, pros, cons, reviewer_title, reviewer_company,
           vendor_name, source, enrichment_status
    FROM b2b_reviews
    WHERE id = $1::uuid
"""


def _coerce_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _has_signal_type(spans: Any, signal_type: str) -> bool:
    target = str(signal_type or "").strip().lower()
    if not target:
        return False
    for span in spans or []:
        if not isinstance(span, dict):
            continue
        if str(span.get("signal_type") or "").strip().lower() == target:
            return True
    return False


def _price_state(enrichment: dict[str, Any] | None) -> dict[str, bool]:
    payload = _coerce_json_dict(enrichment)
    contract = _coerce_json_dict(payload.get("contract_context"))
    indicators = _coerce_json_dict(payload.get("urgency_indicators"))
    return {
        "price_complaint": bool(contract.get("price_complaint")),
        "price_pressure_language": bool(indicators.get("price_pressure_language")),
        "has_pricing_backlash": _has_signal_type(payload.get("evidence_spans"), "pricing_backlash"),
    }


def _matches_filter(
    filter_name: str,
    old_values: dict[str, Any],
    new_enrichment: dict[str, Any],
) -> bool:
    if filter_name != "positive_pricing_false_positive":
        return True
    new_state = _price_state(new_enrichment)
    had_old_positive_pricing = any(
        bool(old_values.get(key))
        for key in ("price_complaint", "price_pressure_language", "has_pricing_backlash")
    )
    return had_old_positive_pricing and not any(new_state.values())


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
    old_values.update(_price_state(enrichment))
    finalized, _ = _finalize_enrichment_for_persist(enrichment, dict(row))
    return old_values, finalized


async def _run(args):
    pool = get_db_pool()
    if not pool.is_initialized:
        await pool.initialize()

    if args.review_id:
        rows = await pool.fetch(_REVIEW_ID_QUERY, str(args.review_id))
    else:
        query = _FILTER_QUERIES.get(args.filter, _FILTER_QUERIES["all"])
        if args.limit:
            query += f" LIMIT {int(args.limit)}"
        rows = await pool.fetch(query)
    logger.info("Found %d reviews to backfill (filter=%s)", len(rows), args.filter)

    updated = 0
    skipped = 0
    changes = {
        "urgency_score": 0,
        "pain_category": 0,
        "would_recommend": 0,
        "competitors_mentioned": 0,
        "price_complaint": 0,
        "price_pressure_language": 0,
        "pricing_backlash": 0,
    }

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
        if not _matches_filter(args.filter, old, new_enrichment):
            skipped += 1
            continue

        new_price_state = _price_state(new_enrichment)

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
        if old["price_complaint"] != new_price_state["price_complaint"]:
            changes["price_complaint"] += 1
            changed = True
        if old["price_pressure_language"] != new_price_state["price_pressure_language"]:
            changes["price_pressure_language"] += 1
            changed = True
        if old["has_pricing_backlash"] != new_price_state["has_pricing_backlash"]:
            changes["pricing_backlash"] += 1
            changed = True

        if not changed:
            skipped += 1
            continue

        if args.dry_run:
            logger.info(
                "DRY RUN %s [%s]: urgency %.1f->%.1f, pain %s->%s, recommend %s->%s, price_complaint %s->%s, price_pressure %s->%s, pricing_backlash %s->%s, competitors %s->%s",
                row["id"],
                row.get("vendor_name") or "unknown_vendor",
                old["urgency_score"] or 0,
                new_enrichment.get("urgency_score", 0),
                old["pain_category"],
                new_enrichment.get("pain_category"),
                old["would_recommend"],
                new_enrichment.get("would_recommend"),
                old["price_complaint"],
                new_price_state["price_complaint"],
                old["price_pressure_language"],
                new_price_state["price_pressure_language"],
                old["has_pricing_backlash"],
                new_price_state["has_pricing_backlash"],
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
        "%s %d reviews, skipped %d unchanged. Changes: urgency=%d, pain=%d, recommend=%d, price_complaint=%d, price_pressure=%d, pricing_backlash=%d, competitors=%d",
        prefix, updated, skipped,
        changes["urgency_score"], changes["pain_category"], changes["would_recommend"],
        changes["price_complaint"], changes["price_pressure_language"], changes["pricing_backlash"],
        changes["competitors_mentioned"],
    )


def main():
    parser = argparse.ArgumentParser(description="Backfill derived enrichment fields")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument("--limit", type=int, default=0, help="Max reviews to process (0=all)")
    parser.add_argument("--review-id", default="", help="Single review UUID to backfill deterministically")
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
