#!/usr/bin/env python3
"""Clear historical Trustpilot JSON-LD reviewer-company pollution.

This pass is intentionally narrow:
- only canonical Trustpilot rows
- only JSON-LD extracted rows
- only rows with a non-empty reviewer_company
- clears enrichment company only when it exactly matches the polluted raw value
- retracts account-resolution rows for the same reviews back to unsupported_source

Usage:
  python scripts/backfill_trustpilot_jsonld_company_cleanup.py
  python scripts/backfill_trustpilot_jsonld_company_cleanup.py --limit 100 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
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
logger = logging.getLogger("backfill_trustpilot_jsonld_company_cleanup")


_CANDIDATE_ROWS_SQL = """
SELECT r.id,
       r.source,
       r.vendor_name,
       r.reviewer_company,
       r.reviewer_company_norm,
       r.raw_metadata,
       r.enrichment,
       r.parser_version,
       ar.id AS account_resolution_id,
       ar.resolution_status,
       ar.resolution_method,
       ar.reviewer_company_raw,
       ar.resolved_company_name
FROM b2b_reviews r
LEFT JOIN b2b_account_resolution ar ON ar.review_id = r.id
WHERE r.source = 'trustpilot'
  AND r.duplicate_of_review_id IS NULL
  AND COALESCE(r.raw_metadata->>'extraction_method', '') = 'json_ld'
  AND NULLIF(BTRIM(r.reviewer_company), '') IS NOT NULL
ORDER BY r.enriched_at DESC NULLS LAST, r.imported_at DESC NULLS LAST, r.id DESC
LIMIT COALESCE($1::int, 1000000)
"""


def _trim_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


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


def _same_text(left: Any, right: Any) -> bool:
    left_text = _trim_text(left)
    right_text = _trim_text(right)
    if not left_text or not right_text:
        return False
    return left_text.casefold() == right_text.casefold()


def _metadata_patch(raw_metadata: Any, *, polluted_company: str, detected_at: datetime) -> dict[str, Any]:
    metadata = _coerce_json_dict(raw_metadata)
    metadata["trustpilot_jsonld_company_cleanup"] = {
        "scope": "trustpilot_jsonld_reviewer_company_cleanup",
        "polluted_company": polluted_company,
        "detected_at": detected_at.astimezone(timezone.utc).isoformat(),
    }
    return metadata


def _resolution_evidence_patch(*, polluted_company: str) -> dict[str, Any]:
    return {
        "reason": "trustpilot_jsonld_reviewer_company_cleanup",
        "source": "trustpilot",
        "polluted_company": polluted_company,
    }


def _plan_row_cleanup(row: dict[str, Any]) -> dict[str, Any] | None:
    polluted_company = _trim_text(row.get("reviewer_company"))
    if not polluted_company:
        return None
    raw_metadata = _coerce_json_dict(row.get("raw_metadata"))
    if str(row.get("source") or "").strip().lower() != "trustpilot":
        return None
    if str(raw_metadata.get("extraction_method") or "").strip().lower() != "json_ld":
        return None
    enrichment = _coerce_json_dict(row.get("enrichment"))
    enrichment_company = (
        enrichment.get("reviewer_context", {}).get("company_name")
        if isinstance(enrichment.get("reviewer_context"), dict)
        else None
    )
    detected_at = datetime.now(timezone.utc)
    return {
        "review_id": str(row["id"]),
        "vendor_name": str(row.get("vendor_name") or "").strip(),
        "polluted_company": polluted_company,
        "clear_enrichment_company": _same_text(enrichment_company, polluted_company),
        "metadata": _metadata_patch(
            row.get("raw_metadata"),
            polluted_company=polluted_company,
            detected_at=detected_at,
        ),
        "account_resolution_id": str(row["account_resolution_id"]) if row.get("account_resolution_id") else None,
        "detected_at": detected_at,
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
) -> dict[str, Any]:
    applied_reviews = 0
    cleared_enrichment = 0
    retracted_account_resolution = 0
    by_method: Counter[str] = Counter()
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
              AND source = 'trustpilot'
              AND duplicate_of_review_id IS NULL
              AND COALESCE(raw_metadata->>'extraction_method', '') = 'json_ld'
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
            resolution_status = await conn.execute(
                """
                UPDATE b2b_account_resolution
                SET reviewer_company_raw = NULL,
                    resolved_company_name = NULL,
                    normalized_company_name = NULL,
                    confidence_score = 0.0,
                    confidence_label = 'unresolved',
                    resolution_method = 'unsupported_source',
                    resolution_evidence = $2::jsonb,
                    resolution_status = 'excluded',
                    resolved_at = NOW()
                WHERE review_id = $1::uuid
                """,
                update["review_id"],
                json.dumps(
                    _resolution_evidence_patch(
                        polluted_company=update["polluted_company"],
                    ),
                ),
            )
            resolution_count = int(str(resolution_status).split()[-1])
            if resolution_count > 0:
                retracted_account_resolution += resolution_count
                by_method["unsupported_source"] += resolution_count
    return {
        "applied_reviews": applied_reviews,
        "cleared_enrichment": cleared_enrichment,
        "retracted_account_resolution": retracted_account_resolution,
        "by_method": dict(by_method),
    }


async def _run(args: argparse.Namespace) -> int:
    conn = await asyncpg.connect(db_settings.dsn)
    try:
        candidate_rows = await _fetch_candidate_rows(conn, limit=args.limit)
        planned = [item for item in (_plan_row_cleanup(row) for row in candidate_rows) if item is not None]
        logger.info("Scanned %d Trustpilot JSON-LD rows; planned cleanups=%d", len(candidate_rows), len(planned))
        if planned:
            logger.info(
                "Preview rows: %s",
                json.dumps(planned[: min(5, len(planned))], default=str),
            )
        if not args.apply:
            logger.info("Dry run only. Re-run with --apply to persist updates.")
            return 0
        result = await _apply_updates(conn, updates=planned)
        logger.info("Applied review cleanups=%d", result["applied_reviews"])
        logger.info("Cleared enrichment companies=%d", result["cleared_enrichment"])
        logger.info("Retracted account-resolution rows=%d", result["retracted_account_resolution"])
        return 0
    finally:
        await conn.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=1000000,
        help="Maximum number of canonical Trustpilot JSON-LD rows to inspect.",
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
