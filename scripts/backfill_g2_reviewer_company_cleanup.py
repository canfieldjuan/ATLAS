#!/usr/bin/env python3
"""Clear historical G2 reviewer-company pollution caused by size or industry labels.

This pass is intentionally narrow:
- only canonical G2 rows
- only rows with a non-empty reviewer_company
- clears reviewer_company when it exactly matches company_size_raw
- clears reviewer_company when it exactly matches reviewer_industry and the label
  reads like an industry, not a company
- clears enrichment reviewer_context.company_name only when it matches the same
  polluted raw value

Usage:
  python scripts/backfill_g2_reviewer_company_cleanup.py
  python scripts/backfill_g2_reviewer_company_cleanup.py --limit 100 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
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
logger = logging.getLogger("backfill_g2_reviewer_company_cleanup")


_COMPANY_SIZE_RE = re.compile(
    r"(?:\bsmall-business\b|\bmid-market\b|\benterprise\b|\b\d[\d,\- ]*\s*employees?\b|\bemp\.)",
    re.I,
)
_INDUSTRY_HINT_RE = re.compile(
    r"\b("
    r"software|technology|services|marketing|advertising|insurance|health|healthcare|"
    r"financial|finance|banking|retail|education|government|manufacturing|media|"
    r"telecommunications|telecom|hospital|hospitality|construction|legal|accounting|"
    r"real estate|internet|computer|human resources|hr|pharma|pharmaceutical|"
    r"logistics|transportation|consumer goods|biotech|automotive|newspapers?|publishing"
    r")\b",
    re.I,
)

_CANDIDATE_ROWS_SQL = """
SELECT r.id,
       r.source,
       r.vendor_name,
       r.source_review_id,
       r.reviewer_company,
       r.reviewer_company_norm,
       r.reviewer_industry,
       r.company_size_raw,
       r.raw_metadata,
       r.enrichment,
       ar.id AS account_resolution_id,
       ar.resolution_status,
       ar.reviewer_company_raw,
       ar.resolved_company_name
FROM b2b_reviews r
LEFT JOIN b2b_account_resolution ar ON ar.review_id = r.id
WHERE r.source = 'g2'
  AND r.duplicate_of_review_id IS NULL
  AND NULLIF(BTRIM(r.reviewer_company), '') IS NOT NULL
ORDER BY r.enriched_at DESC NULLS LAST, r.imported_at DESC NULLS LAST, r.id DESC
LIMIT COALESCE($1::int, 1000000)
"""


def _trim_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _same_text(left: Any, right: Any) -> bool:
    left_text = _trim_text(left)
    right_text = _trim_text(right)
    if not left_text or not right_text:
        return False
    return left_text.casefold() == right_text.casefold()


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


def _looks_like_company_size_label(value: Any) -> bool:
    return bool(_COMPANY_SIZE_RE.search(str(value or "").strip()))


def _looks_like_industry_label(value: Any) -> bool:
    return bool(_INDUSTRY_HINT_RE.search(str(value or "").strip()))


def _metadata_patch(
    raw_metadata: Any,
    *,
    polluted_company: str,
    cleanup_reason: str,
    detected_at: datetime,
) -> dict[str, Any]:
    metadata = _coerce_json_dict(raw_metadata)
    metadata["g2_reviewer_company_cleanup"] = {
        "scope": "g2_reviewer_company_cleanup",
        "polluted_company": polluted_company,
        "cleanup_reason": cleanup_reason,
        "detected_at": detected_at.astimezone(timezone.utc).isoformat(),
    }
    return metadata


def _plan_row_cleanup(row: dict[str, Any]) -> dict[str, Any] | None:
    polluted_company = _trim_text(row.get("reviewer_company"))
    if not polluted_company:
        return None
    if str(row.get("source") or "").strip().lower() != "g2":
        return None

    cleanup_reason = None
    if _same_text(polluted_company, row.get("company_size_raw")) and _looks_like_company_size_label(
        polluted_company,
    ):
        cleanup_reason = "company_size_label"
    elif _same_text(polluted_company, row.get("reviewer_industry")) and _looks_like_industry_label(
        polluted_company,
    ):
        cleanup_reason = "industry_label"
    if not cleanup_reason:
        return None

    enrichment = _coerce_json_dict(row.get("enrichment"))
    reviewer_context = enrichment.get("reviewer_context")
    enrichment_company = reviewer_context.get("company_name") if isinstance(reviewer_context, dict) else None
    detected_at = datetime.now(timezone.utc)
    return {
        "review_id": str(row["id"]),
        "vendor_name": str(row.get("vendor_name") or "").strip(),
        "source_review_id": str(row.get("source_review_id") or "").strip(),
        "polluted_company": polluted_company,
        "cleanup_reason": cleanup_reason,
        "clear_enrichment_company": _same_text(enrichment_company, polluted_company),
        "metadata": _metadata_patch(
            row.get("raw_metadata"),
            polluted_company=polluted_company,
            cleanup_reason=cleanup_reason,
            detected_at=detected_at,
        ),
        "account_resolution_id": str(row["account_resolution_id"]) if row.get("account_resolution_id") else None,
    }


async def _fetch_candidate_rows(
    conn: asyncpg.Connection,
    *,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows = await conn.fetch(_CANDIDATE_ROWS_SQL, limit)
    return [dict(row) for row in rows]


async def _apply_updates(
    conn: asyncpg.Connection,
    *,
    updates: list[dict[str, Any]],
) -> dict[str, int]:
    applied_reviews = 0
    cleared_enrichment = 0
    touched_account_resolution = 0
    for update in updates:
        status = await conn.execute(
            """
            UPDATE b2b_reviews
            SET reviewer_company = NULL,
                reviewer_company_norm = NULL,
                enrichment = CASE
                    WHEN $2::boolean THEN enrichment #- '{reviewer_context,company_name}'
                    ELSE enrichment
                END,
                raw_metadata = $3::jsonb
            WHERE id = $1::uuid
              AND source = 'g2'
              AND duplicate_of_review_id IS NULL
              AND NULLIF(BTRIM(reviewer_company), '') IS NOT NULL
            """,
            update["review_id"],
            update["clear_enrichment_company"],
            json.dumps(update["metadata"], default=str),
        )
        count = int(str(status).split()[-1])
        if count <= 0:
            continue
        applied_reviews += count
        if update["clear_enrichment_company"]:
            cleared_enrichment += count

        if update["account_resolution_id"]:
            result = await conn.execute(
                """
                UPDATE b2b_account_resolution
                SET reviewer_company_raw = CASE
                        WHEN reviewer_company_raw IS NOT NULL
                         AND reviewer_company_raw = $2::text
                        THEN NULL
                        ELSE reviewer_company_raw
                    END,
                    resolved_company_name = CASE
                        WHEN resolved_company_name IS NOT NULL
                         AND resolved_company_name = $2::text
                        THEN NULL
                        ELSE resolved_company_name
                    END
                WHERE review_id = $1::uuid
                """,
                update["review_id"],
                update["polluted_company"],
            )
            touched_account_resolution += int(str(result).split()[-1])

    return {
        "applied_reviews": applied_reviews,
        "cleared_enrichment": cleared_enrichment,
        "touched_account_resolution": touched_account_resolution,
    }


async def _run(args: argparse.Namespace) -> int:
    conn = await asyncpg.connect(db_settings.dsn)
    try:
        candidate_rows = await _fetch_candidate_rows(conn, limit=args.limit)
        planned = [item for item in (_plan_row_cleanup(row) for row in candidate_rows) if item is not None]
        logger.info("Scanned %d G2 rows; planned cleanups=%d", len(candidate_rows), len(planned))
        if planned:
            logger.info("Preview rows: %s", json.dumps(planned[: min(10, len(planned))], default=str))
        if not args.apply:
            logger.info("Dry run only. Re-run with --apply to persist updates.")
            return 0
        result = await _apply_updates(conn, updates=planned)
        logger.info("Applied review cleanups=%d", result["applied_reviews"])
        logger.info("Cleared enrichment reviewer_context.company_name=%d", result["cleared_enrichment"])
        logger.info("Touched account-resolution rows=%d", result["touched_account_resolution"])
        return 0
    finally:
        await conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
