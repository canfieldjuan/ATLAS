#!/usr/bin/env python3
"""Backfill cross-source duplicate review links across historical b2b_reviews.

Usage:
  python scripts/backfill_cross_source_review_dedup.py
  python scripts/backfill_cross_source_review_dedup.py --apply
  python scripts/backfill_cross_source_review_dedup.py --apply --vendors "ActiveCampaign,HubSpot"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import asyncpg

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)

from atlas_brain.autonomous.visibility import record_dedup
from atlas_brain.config import settings
from atlas_brain.services.b2b.review_dedup import cluster_cross_source_duplicates
from atlas_brain.storage.config import db_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_cross_source_review_dedup")


_CANDIDATE_VENDORS_SQL = """
SELECT vendor_name
FROM b2b_reviews
WHERE duplicate_of_review_id IS NULL
  AND ($1::text[] IS NULL OR LOWER(vendor_name) = ANY($1::text[]))
GROUP BY vendor_name
HAVING COUNT(*) > 1
   AND COUNT(DISTINCT LOWER(source)) > 1
ORDER BY vendor_name
LIMIT COALESCE($2::int, 1000000)
"""


_VENDOR_REVIEW_ROWS_SQL = """
SELECT id, vendor_name, source, source_review_id, reviewer_name, reviewed_at,
       rating, imported_at, enrichment_status, source_weight,
       summary, review_text, pros, cons,
       raw_metadata,
       cross_source_content_hash, cross_source_identity_key,
       duplicate_of_review_id
FROM b2b_reviews
WHERE vendor_name = $1
  AND duplicate_of_review_id IS NULL
  AND (
        cross_source_content_hash IS NOT NULL
     OR cross_source_identity_key IS NOT NULL
  )
ORDER BY imported_at ASC NULLS LAST, id ASC
"""


def _parse_vendor_filter(raw: str | None) -> list[str] | None:
    names = [part.strip().lower() for part in str(raw or "").split(",") if part.strip()]
    return names or None


async def _fetch_candidate_vendor_names(
    conn: asyncpg.Connection,
    *,
    vendors: list[str] | None,
    limit_vendors: int | None,
) -> list[str]:
    rows = await conn.fetch(
        _CANDIDATE_VENDORS_SQL,
        vendors,
        limit_vendors,
    )
    return [str(row["vendor_name"]) for row in rows]


async def _fetch_vendor_review_rows(
    conn: asyncpg.Connection,
    *,
    vendor_name: str,
) -> list[dict[str, Any]]:
    rows = await conn.fetch(_VENDOR_REVIEW_ROWS_SQL, vendor_name)
    return [dict(row) for row in rows]


def _metadata_patch(row: dict[str, Any], *, duplicate_of_review_id: str, duplicate_reason: str, duplicate_detail: dict[str, Any]) -> dict[str, Any]:
    raw_metadata = row.get("raw_metadata")
    if isinstance(raw_metadata, dict):
        metadata = dict(raw_metadata)
    elif isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
        except json.JSONDecodeError:
            parsed = {}
        metadata = dict(parsed) if isinstance(parsed, dict) else {}
    else:
        metadata = {}
    metadata["duplicate_of_review_id"] = duplicate_of_review_id
    metadata["duplicate_reason"] = duplicate_reason
    metadata["duplicate_detail"] = duplicate_detail
    metadata["dedup_backfill_scope"] = "cross_source_review_dedup"
    prior_status = str(row.get("enrichment_status") or "").strip()
    if prior_status:
        metadata["prior_enrichment_status"] = prior_status
    return metadata


def _plan_vendor_duplicate_updates(
    rows: list[dict[str, Any]],
    *,
    similarity_threshold: float,
    loose_similarity_threshold: float,
    reviewer_stem_length: int,
    review_date_tolerance_days: int,
    rating_tolerance: float,
) -> list[dict[str, Any]]:
    decisions = cluster_cross_source_duplicates(
        rows,
        similarity_threshold=similarity_threshold,
        loose_similarity_threshold=loose_similarity_threshold,
        reviewer_stem_length=reviewer_stem_length,
        review_date_tolerance_days=review_date_tolerance_days,
        rating_tolerance=rating_tolerance,
    )
    planned: list[dict[str, Any]] = []
    row_by_id = {str(row.get("id") or ""): row for row in rows}
    for row_id, decision in decisions.items():
        row = row_by_id.get(row_id)
        if row is None:
            continue
        survivor_review_id = str(decision["survivor_review_id"])
        duplicate_reason = str(decision["duplicate_reason"])
        duplicate_detail = dict(decision.get("duplicate_detail") or {})
        planned.append(
            {
                "review_id": row_id,
                "vendor_name": str(row.get("vendor_name") or ""),
                "source": str(row.get("source") or ""),
                "prior_enrichment_status": str(row.get("enrichment_status") or ""),
                "survivor_review_id": survivor_review_id,
                "duplicate_reason": duplicate_reason,
                "duplicate_detail": duplicate_detail,
                "metadata": _metadata_patch(
                    row,
                    duplicate_of_review_id=survivor_review_id,
                    duplicate_reason=duplicate_reason,
                    duplicate_detail=duplicate_detail,
                ),
            }
        )
    planned.sort(key=lambda item: (item["vendor_name"], item["review_id"]))
    return planned


async def _apply_vendor_duplicate_updates(
    conn: asyncpg.Connection,
    *,
    updates: list[dict[str, Any]],
) -> dict[str, int]:
    applied = 0
    by_reason: dict[str, int] = defaultdict(int)
    by_status: dict[str, int] = defaultdict(int)
    for update in updates:
        status = await conn.execute(
            """
            UPDATE b2b_reviews
            SET duplicate_of_review_id = $2::uuid,
                duplicate_reason = $3,
                deduped_at = NOW(),
                enrichment_status = 'duplicate',
                raw_metadata = $4::jsonb
            WHERE id = $1::uuid
              AND duplicate_of_review_id IS NULL
            """,
            update["review_id"],
            update["survivor_review_id"],
            update["duplicate_reason"],
            json.dumps(update["metadata"], default=str),
        )
        count = int(str(status).split()[-1])
        if count <= 0:
            continue
        applied += count
        by_reason[update["duplicate_reason"]] += count
        by_status[update["prior_enrichment_status"] or ""] += count
        await record_dedup(
            conn,
            stage="cross_source_review_dedup_backfill",
            entity_type="review",
            entity_id=update["review_id"],
            reason=update["duplicate_reason"],
            survivor_entity_id=update["survivor_review_id"],
            detail=update["duplicate_detail"],
        )
    return {
        "applied": applied,
        "exact_content": by_reason.get("cross_source_exact_content", 0),
        "identity_similarity": by_reason.get("cross_source_identity_similarity", 0),
        "reviewer_date_similarity": by_reason.get("cross_source_reviewer_date_similarity", 0),
        "prior_enriched": by_status.get("enriched", 0),
        "prior_pending": by_status.get("pending", 0),
        "prior_no_signal": by_status.get("no_signal", 0),
        "prior_quarantined": by_status.get("quarantined", 0),
    }


async def main(*, apply: bool, vendors: list[str] | None, limit_vendors: int | None) -> int:
    conn = await asyncpg.connect(db_settings.dsn)
    try:
        vendor_names = await _fetch_candidate_vendor_names(
            conn,
            vendors=vendors,
            limit_vendors=limit_vendors,
        )
        logger.info("Candidate vendors: %d", len(vendor_names))
        planned_updates: list[dict[str, Any]] = []
        impacted_vendors: list[str] = []
        for vendor_name in vendor_names:
            rows = await _fetch_vendor_review_rows(conn, vendor_name=vendor_name)
            updates = _plan_vendor_duplicate_updates(
                rows,
                similarity_threshold=float(settings.b2b_scrape.cross_source_dedup_similarity_threshold),
                loose_similarity_threshold=float(settings.b2b_scrape.cross_source_dedup_loose_similarity_threshold),
                reviewer_stem_length=int(settings.b2b_scrape.cross_source_dedup_reviewer_stem_length),
                review_date_tolerance_days=int(settings.b2b_scrape.cross_source_dedup_review_date_tolerance_days),
                rating_tolerance=float(settings.b2b_scrape.cross_source_dedup_rating_tolerance),
            )
            if not updates:
                continue
            planned_updates.extend(updates)
            impacted_vendors.append(vendor_name)
        logger.info("Planned duplicate links: %d", len(planned_updates))
        if impacted_vendors:
            logger.info("Impacted vendors: %s", ", ".join(impacted_vendors[:25]))
        if not apply or not planned_updates:
            return 0
        async with conn.transaction():
            stats = await _apply_vendor_duplicate_updates(conn, updates=planned_updates)
        logger.info("Applied duplicate links: %d", stats["applied"])
        logger.info(
            "Reason split: exact_content=%d identity_similarity=%d reviewer_date_similarity=%d",
            stats["exact_content"],
            stats["identity_similarity"],
            stats["reviewer_date_similarity"],
        )
        logger.info(
            "Prior status split: enriched=%d pending=%d no_signal=%d quarantined=%d",
            stats["prior_enriched"],
            stats["prior_pending"],
            stats["prior_no_signal"],
            stats["prior_quarantined"],
        )
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical cross-source duplicate review links")
    parser.add_argument("--apply", action="store_true", help="Apply updates. Default is dry run.")
    parser.add_argument("--vendors", help="Comma-separated vendor filter")
    parser.add_argument("--limit-vendors", type=int, default=None, help="Cap vendor scan count")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                apply=bool(args.apply),
                vendors=_parse_vendor_filter(args.vendors),
                limit_vendors=args.limit_vendors,
            )
        )
    )
