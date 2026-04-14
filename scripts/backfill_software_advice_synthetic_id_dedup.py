#!/usr/bin/env python3
"""Collapse Software Advice parser-drift duplicates onto stable source IDs.

This pass is intentionally narrow:
- only canonical software_advice rows
- only rows with synthetic 16-char fallback source_review_id values
- only when a same-vendor canonical row with the same content hash and a
  non-synthetic source_review_id already exists
- only low-value hashed rows are deduped (raw_only, not_applicable, pending)

Usage:
  python scripts/backfill_software_advice_synthetic_id_dedup.py
  python scripts/backfill_software_advice_synthetic_id_dedup.py --vendors ClickUp,Zendesk --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from collections import Counter, defaultdict
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

from atlas_brain.storage.config import db_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_software_advice_synthetic_id_dedup")


_CANDIDATE_ROWS_SQL = """
SELECT id,
       vendor_name,
       source_review_id,
       enrichment_status,
       review_text,
       summary,
       pros,
       cons,
       reviewer_title,
       reviewer_company,
       reviewer_company_norm,
       company_size_raw,
       reviewer_industry,
       parser_version,
       raw_metadata,
       imported_at
FROM b2b_reviews
WHERE source = 'software_advice'
  AND duplicate_of_review_id IS NULL
  AND NULLIF(BTRIM(source_review_id), '') IS NOT NULL
  AND NULLIF(BTRIM(review_text), '') IS NOT NULL
  AND ($1::text[] IS NULL OR vendor_name = ANY($1::text[]))
ORDER BY imported_at DESC NULLS LAST, id DESC
LIMIT COALESCE($2::int, 1000000)
"""


def _parse_vendors(raw: str | None) -> list[str] | None:
    values = sorted({part.strip() for part in str(raw or "").split(",") if part.strip()})
    return values or None


def _coerce_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _looks_like_synthetic_source_review_id(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{16}", str(value or "").strip().lower()))


def _normalize_review_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _cleanup_metadata(raw_metadata: Any, *, canonical_review_id: str, canonical_source_review_id: str) -> dict[str, Any]:
    metadata = _coerce_json_dict(raw_metadata)
    metadata["software_advice_synthetic_id_cleanup"] = {
        "scope": "software_advice_same_source_parser_drift",
        "canonical_review_id": canonical_review_id,
        "canonical_source_review_id": canonical_source_review_id,
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }
    return metadata


def _plan_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stable_rows = [row for row in rows if not _looks_like_synthetic_source_review_id(row.get("source_review_id"))]
    if not stable_rows:
        return []
    canonical = stable_rows[0]
    planned: list[dict[str, Any]] = []
    for row in rows:
        if not _looks_like_synthetic_source_review_id(row.get("source_review_id")):
            continue
        if str(row.get("enrichment_status") or "").strip() not in {"raw_only", "not_applicable", "pending"}:
            continue
        planned.append({
            "duplicate_review_id": str(row["id"]),
            "duplicate_source_review_id": str(row.get("source_review_id") or "").strip(),
            "duplicate_status": str(row.get("enrichment_status") or "").strip(),
            "canonical_review_id": str(canonical["id"]),
            "canonical_source_review_id": str(canonical.get("source_review_id") or "").strip(),
            "vendor_name": str(row.get("vendor_name") or "").strip(),
            "summary": row.get("summary"),
            "pros": row.get("pros"),
            "cons": row.get("cons"),
            "reviewer_title": row.get("reviewer_title"),
            "reviewer_company": row.get("reviewer_company"),
            "reviewer_company_norm": row.get("reviewer_company_norm"),
            "company_size_raw": row.get("company_size_raw"),
            "reviewer_industry": row.get("reviewer_industry"),
            "parser_version": row.get("parser_version"),
            "metadata": _cleanup_metadata(
                row.get("raw_metadata"),
                canonical_review_id=str(canonical["id"]),
                canonical_source_review_id=str(canonical.get("source_review_id") or "").strip(),
            ),
        })
    return planned


def _plan_updates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        body_norm = _normalize_review_text(row.get("review_text"))
        if not body_norm:
            continue
        grouped[(str(row.get("vendor_name") or "").strip(), body_norm)].append(row)
    planned: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        planned.extend(_plan_group(group_rows))
    return planned


async def _fetch_candidate_rows(
    conn: asyncpg.Connection,
    *,
    vendors: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows = await conn.fetch(_CANDIDATE_ROWS_SQL, vendors, limit)
    return [dict(row) for row in rows]


async def _apply_updates(
    conn: asyncpg.Connection,
    *,
    updates: list[dict[str, Any]],
) -> dict[str, Any]:
    applied = 0
    by_vendor: Counter[str] = Counter()
    for update in updates:
        await conn.execute(
            """
            UPDATE b2b_reviews
            SET summary = COALESCE(summary, $2::text),
                pros = COALESCE(pros, $3::text),
                cons = COALESCE(cons, $4::text),
                reviewer_title = COALESCE(reviewer_title, $5::text),
                reviewer_company = COALESCE(reviewer_company, $6::text),
                reviewer_company_norm = COALESCE(reviewer_company_norm, $7::text),
                company_size_raw = COALESCE(company_size_raw, $8::text),
                reviewer_industry = COALESCE(reviewer_industry, $9::text),
                parser_version = COALESCE(parser_version, $10::text)
            WHERE id = $1::uuid
            """,
            update["canonical_review_id"],
            update["summary"],
            update["pros"],
            update["cons"],
            update["reviewer_title"],
            update["reviewer_company"],
            update["reviewer_company_norm"],
            update["company_size_raw"],
            update["reviewer_industry"],
            update["parser_version"],
        )
        await conn.execute(
            """
            UPDATE b2b_reviews
            SET duplicate_of_review_id = $2::uuid,
                duplicate_reason = 'same_source_parser_drift_duplicate',
                deduped_at = NOW(),
                enrichment_status = 'duplicate',
                raw_metadata = $3::jsonb
            WHERE id = $1::uuid
              AND duplicate_of_review_id IS NULL
            """,
            update["duplicate_review_id"],
            update["canonical_review_id"],
            json.dumps(update["metadata"], default=str),
        )
        await conn.execute(
            """
            INSERT INTO b2b_review_vendor_mentions (review_id, vendor_name, is_primary, match_metadata)
            VALUES (
                $1::uuid,
                $2::text,
                false,
                jsonb_build_object(
                    'match_reason', 'same_source_parser_drift_merge',
                    'replaced_review_id', $3::uuid,
                    'replaced_source_review_id', $4::text
                )
            )
            ON CONFLICT (review_id, vendor_name) DO UPDATE
            SET match_metadata = COALESCE(b2b_review_vendor_mentions.match_metadata, '{}'::jsonb) || EXCLUDED.match_metadata,
                updated_at = NOW()
            """,
            update["canonical_review_id"],
            update["vendor_name"],
            update["duplicate_review_id"],
            update["duplicate_source_review_id"],
        )
        applied += 1
        by_vendor[update["vendor_name"]] += 1
    return {"applied": applied, "by_vendor": dict(by_vendor)}


async def _run(args: argparse.Namespace) -> int:
    vendors = _parse_vendors(args.vendors)
    conn = await asyncpg.connect(db_settings.dsn)
    try:
        candidate_rows = await _fetch_candidate_rows(conn, vendors=vendors, limit=args.limit)
        planned = _plan_updates(candidate_rows)
        logger.info("Scanned %d software_advice rows; planned cleanups=%d", len(candidate_rows), len(planned))
        if planned:
            logger.info("Preview rows: %s", json.dumps(planned[: min(5, len(planned))], default=str))
        if not args.apply:
            logger.info("Dry run only. Re-run with --apply to persist updates.")
            return 0
        result = await _apply_updates(conn, updates=planned)
        logger.info("Applied duplicate cleanups=%d", result["applied"])
        logger.info("By vendor: %s", json.dumps(result["by_vendor"], default=str))
        return 0
    finally:
        await conn.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=1000000,
        help="Maximum number of canonical software_advice rows to inspect.",
    )
    parser.add_argument(
        "--vendors",
        type=str,
        default=None,
        help="Optional comma-separated vendor filter.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist updates instead of running a dry run.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
