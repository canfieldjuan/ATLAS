#!/usr/bin/env python3
"""Collapse historical same-source multi-vendor review fanout.

This backfill is intentionally conservative:
- it only targets rows that share the same ``source`` and ``source_review_id``
- it only acts when that source item was stored under multiple vendor names
- it preserves one survivor row and links all vendor mentions to that survivor

Usage:
  python scripts/backfill_same_source_review_vendor_fanout.py
  python scripts/backfill_same_source_review_vendor_fanout.py --apply
  python scripts/backfill_same_source_review_vendor_fanout.py --apply --sources "reddit,twitter,hackernews"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
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
from atlas_brain.storage.config import db_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_same_source_review_vendor_fanout")


_CANDIDATE_GROUPS_SQL = """
SELECT source,
       source_review_id,
       COUNT(*) AS row_count,
       COUNT(DISTINCT LOWER(vendor_name)) AS vendor_count
FROM b2b_reviews
WHERE duplicate_of_review_id IS NULL
  AND COALESCE(BTRIM(source_review_id), '') <> ''
  AND ($1::text[] IS NULL OR LOWER(source) = ANY($1::text[]))
GROUP BY source, source_review_id
HAVING COUNT(*) > 1
   AND COUNT(DISTINCT LOWER(vendor_name)) > 1
ORDER BY row_count DESC, source, source_review_id
LIMIT COALESCE($2::int, 1000000)
"""


_GROUP_ROWS_SQL = """
SELECT id,
       source,
       source_review_id,
       vendor_name,
       imported_at,
       reviewed_at,
       enrichment_status,
       summary,
       review_text,
       pros,
       cons,
       raw_metadata
FROM b2b_reviews
WHERE duplicate_of_review_id IS NULL
  AND source = $1
  AND source_review_id = $2
ORDER BY imported_at ASC NULLS LAST, id ASC
"""


_UPSERT_VENDOR_MENTION_SQL = """
INSERT INTO b2b_review_vendor_mentions (
    review_id, vendor_name, is_primary, match_metadata
) VALUES (
    $1::uuid, $2, $3, $4::jsonb
)
ON CONFLICT (review_id, vendor_name) DO UPDATE
SET is_primary = b2b_review_vendor_mentions.is_primary OR EXCLUDED.is_primary,
    match_metadata = COALESCE(b2b_review_vendor_mentions.match_metadata, '{}'::jsonb) || EXCLUDED.match_metadata,
    updated_at = NOW()
"""


def _parse_csv(raw: str | None) -> list[str] | None:
    values = [part.strip().lower() for part in str(raw or "").split(",") if part.strip()]
    return values or None


def _status_rank(value: Any) -> int:
    status = str(value or "").strip().lower()
    order = {
        "enriched": 0,
        "pending": 1,
        "no_signal": 2,
        "quarantined": 3,
        "raw_only": 4,
        "not_applicable": 5,
        "filtered": 6,
        "failed": 7,
        "duplicate": 8,
    }
    return order.get(status, 9)


def _content_length(row: dict[str, Any]) -> int:
    return sum(len(str(row.get(field) or "").strip()) for field in ("summary", "review_text", "pros", "cons"))


def _imported_rank(row: dict[str, Any]) -> float:
    raw = row.get("imported_at")
    if isinstance(raw, datetime):
        value = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return float("inf")
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            return float("inf")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _choose_survivor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (
            _status_rank(row.get("enrichment_status")),
            -_content_length(row),
            _imported_rank(row),
            str(row.get("id") or ""),
        ),
    )
    return ranked[0]


def _metadata_patch(row: dict[str, Any], *, survivor_review_id: str, survivor_vendor: str, vendor_names: list[str]) -> dict[str, Any]:
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
    metadata["duplicate_of_review_id"] = survivor_review_id
    metadata["duplicate_reason"] = "same_source_multi_vendor_duplicate"
    metadata["duplicate_detail"] = {
        "scope": "same_source_vendor_fanout_backfill",
        "survivor_vendor": survivor_vendor,
        "vendor_names": vendor_names,
    }
    prior_status = str(row.get("enrichment_status") or "").strip()
    if prior_status:
        metadata["prior_enrichment_status"] = prior_status
    return metadata


async def _fetch_candidate_groups(
    conn: asyncpg.Connection,
    *,
    sources: list[str] | None,
    limit_groups: int | None,
) -> list[dict[str, Any]]:
    rows = await conn.fetch(_CANDIDATE_GROUPS_SQL, sources, limit_groups)
    return [dict(row) for row in rows]


async def _fetch_group_rows(
    conn: asyncpg.Connection,
    *,
    source: str,
    source_review_id: str,
) -> list[dict[str, Any]]:
    rows = await conn.fetch(_GROUP_ROWS_SQL, source, source_review_id)
    return [dict(row) for row in rows]


async def _apply_group(
    conn: asyncpg.Connection,
    *,
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    survivor = _choose_survivor(rows)
    survivor_id = str(survivor["id"])
    survivor_vendor = str(survivor.get("vendor_name") or "")
    vendor_names = sorted({str(row.get("vendor_name") or "") for row in rows if str(row.get("vendor_name") or "").strip()})

    mention_count = 0
    for vendor_name in vendor_names:
        await conn.execute(
            _UPSERT_VENDOR_MENTION_SQL,
            survivor_id,
            vendor_name,
            vendor_name == survivor_vendor,
            json.dumps(
                {
                    "association_source": "same_source_vendor_fanout_backfill",
                    "review_source": survivor.get("source"),
                    "source_review_id": survivor.get("source_review_id"),
                    "vendor_names": vendor_names,
                }
            ),
        )
        mention_count += 1

    duplicate_count = 0
    for row in rows:
        row_id = str(row["id"])
        if row_id == survivor_id:
            continue
        metadata = _metadata_patch(
            row,
            survivor_review_id=survivor_id,
            survivor_vendor=survivor_vendor,
            vendor_names=vendor_names,
        )
        status = await conn.execute(
            """
            UPDATE b2b_reviews
            SET duplicate_of_review_id = $2::uuid,
                duplicate_reason = 'same_source_multi_vendor_duplicate',
                deduped_at = NOW(),
                enrichment_status = 'duplicate',
                raw_metadata = $3::jsonb
            WHERE id = $1::uuid
              AND duplicate_of_review_id IS NULL
            """,
            row_id,
            survivor_id,
            json.dumps(metadata, default=str),
        )
        count = int(str(status).split()[-1])
        if count <= 0:
            continue
        duplicate_count += count
        await record_dedup(
            conn,
            stage="same_source_vendor_fanout_backfill",
            entity_type="review",
            entity_id=row_id,
            reason="same_source_multi_vendor_duplicate",
            survivor_entity_id=survivor_id,
            detail={
                "source": row.get("source"),
                "source_review_id": row.get("source_review_id"),
                "survivor_vendor": survivor_vendor,
                "vendor_names": vendor_names,
            },
        )

    return {
        "survivor_mentions_upserted": mention_count,
        "duplicates_applied": duplicate_count,
    }


async def main(*, apply: bool, sources: list[str] | None, limit_groups: int | None) -> int:
    conn = await asyncpg.connect(db_settings.dsn)
    try:
        groups = await _fetch_candidate_groups(
            conn,
            sources=sources,
            limit_groups=limit_groups,
        )
        logger.info("Candidate same-source multi-vendor groups: %d", len(groups))
        if not groups or not apply:
            if groups:
                logger.info(
                    "Preview groups: %s",
                    ", ".join(
                        f"{group['source']}:{group['source_review_id']}" for group in groups[:20]
                    ),
                )
            return 0

        totals: dict[str, int] = defaultdict(int)
        async with conn.transaction():
            for group in groups:
                rows = await _fetch_group_rows(
                    conn,
                    source=str(group["source"]),
                    source_review_id=str(group["source_review_id"]),
                )
                if len(rows) < 2:
                    continue
                stats = await _apply_group(conn, rows=rows)
                for key, value in stats.items():
                    totals[key] += int(value or 0)
        logger.info(
            "Applied same-source vendor-fanout backfill: groups=%d duplicates=%d survivor_mentions=%d",
            len(groups),
            totals["duplicates_applied"],
            totals["survivor_mentions_upserted"],
        )
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply updates instead of previewing them")
    parser.add_argument("--sources", help="Optional comma-separated source filter")
    parser.add_argument("--limit-groups", type=int, help="Optional max candidate groups to process")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                apply=bool(args.apply),
                sources=_parse_csv(args.sources),
                limit_groups=args.limit_groups,
            )
        )
    )
