#!/usr/bin/env python3
"""Backfill low-fidelity quarantine for historical enriched B2B reviews.

This pass is intentionally conservative:
- only evaluates rows currently marked ``enriched``
- only targets configured noisy/community sources plus Trustpilot
- only changes rows when the live low-fidelity detector returns reasons
- preserves the existing enrichment payload and only downgrades status

Usage:
  python scripts/backfill_low_substance_enriched_reviews.py
  python scripts/backfill_low_substance_enriched_reviews.py --limit 500
  python scripts/backfill_low_substance_enriched_reviews.py --sources "reddit,software_advice" --apply
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

from atlas_brain.autonomous.tasks import b2b_enrichment
from atlas_brain.autonomous.visibility import record_quarantine
from atlas_brain.config import B2BChurnConfig
from atlas_brain.storage.config import db_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_low_substance_enriched_reviews")


_CANDIDATE_ROWS_SQL = """
SELECT id,
       source,
       vendor_name,
       product_name,
       summary,
       review_text,
       pros,
       cons,
       raw_metadata,
       enrichment,
       enriched_at,
       imported_at
FROM b2b_reviews
WHERE enrichment_status = 'enriched'
  AND duplicate_of_review_id IS NULL
  AND COALESCE(low_fidelity, false) = false
  AND enrichment IS NOT NULL
  AND ($1::text[] IS NULL OR LOWER(source) = ANY($1::text[]))
ORDER BY enriched_at DESC NULLS LAST, imported_at DESC NULLS LAST, id DESC
LIMIT COALESCE($2::int, 1000000)
"""


def _parse_sources(raw: str | None) -> list[str] | None:
    values = sorted(
        {
            part.strip().lower()
            for part in str(raw or "").split(",")
            if part.strip()
        }
    )
    return values or None


def _default_sources() -> list[str]:
    raw_default = B2BChurnConfig.model_fields["enrichment_low_fidelity_noisy_sources"].default
    sources = set(_parse_sources(str(raw_default or "")) or [])
    sources.add("trustpilot")
    return sorted(sources)


def _coerce_metadata(raw_metadata: Any) -> dict[str, Any]:
    if isinstance(raw_metadata, dict):
        return dict(raw_metadata)
    if isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _coerce_enrichment(raw_enrichment: Any) -> dict[str, Any]:
    enrichment = b2b_enrichment._coerce_json_dict(raw_enrichment)
    return enrichment if enrichment else {}


def _metadata_patch(
    row: dict[str, Any],
    *,
    low_fidelity_reasons: list[str],
    detected_at: datetime,
) -> dict[str, Any]:
    metadata = _coerce_metadata(row.get("raw_metadata"))
    metadata["low_fidelity_backfill"] = {
        "scope": "historical_low_substance_enriched_reviews",
        "reason_count": len(low_fidelity_reasons),
        "reasons": list(low_fidelity_reasons),
        "detected_at": detected_at.astimezone(timezone.utc).isoformat(),
    }
    prior_status = str(row.get("enrichment_status") or "enriched").strip()
    if prior_status:
        metadata["prior_enrichment_status"] = prior_status
    return metadata


def _plan_row_backfill(row: dict[str, Any]) -> dict[str, Any] | None:
    enrichment = _coerce_enrichment(row.get("enrichment"))
    if not enrichment:
        return None
    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, enrichment)
    if not reasons:
        return None
    detected_at = datetime.now(timezone.utc)
    return {
        "review_id": str(row["id"]),
        "source": str(row.get("source") or "").strip().lower(),
        "vendor_name": str(row.get("vendor_name") or "").strip(),
        "low_fidelity_reasons": reasons,
        "detected_at": detected_at,
        "metadata": _metadata_patch(
            row,
            low_fidelity_reasons=reasons,
            detected_at=detected_at,
        ),
    }


async def _fetch_candidate_rows(
    conn: asyncpg.Connection,
    *,
    sources: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows = await conn.fetch(_CANDIDATE_ROWS_SQL, sources, limit)
    return [dict(row) for row in rows]


async def _apply_updates(
    conn: asyncpg.Connection,
    *,
    updates: list[dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    applied = 0
    by_source: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    for update in updates:
        status = await conn.execute(
            """
            UPDATE b2b_reviews
            SET enrichment_status = 'quarantined',
                low_fidelity = true,
                low_fidelity_reasons = $2::jsonb,
                low_fidelity_detected_at = $3,
                raw_metadata = $4::jsonb
            WHERE id = $1::uuid
              AND enrichment_status = 'enriched'
              AND duplicate_of_review_id IS NULL
              AND COALESCE(low_fidelity, false) = false
            """,
            update["review_id"],
            json.dumps(update["low_fidelity_reasons"]),
            update["detected_at"],
            json.dumps(update["metadata"], default=str),
        )
        count = int(str(status).split()[-1])
        if count <= 0:
            continue
        applied += count
        by_source[update["source"]] += count
        for reason in update["low_fidelity_reasons"]:
            by_reason[reason] += 1
        await record_quarantine(
            conn,
            review_id=update["review_id"],
            vendor_name=update["vendor_name"],
            source=update["source"],
            reason_code=update["low_fidelity_reasons"][0],
            severity="warning",
            actionable=False,
            summary=f"Historical low-fidelity backfill: {', '.join(update['low_fidelity_reasons'][:3])}",
            evidence={
                "scope": "historical_low_substance_enriched_reviews",
                "reasons": update["low_fidelity_reasons"],
                "source": update["source"],
            },
            run_id=run_id,
        )
    return {
        "applied": applied,
        "by_source": dict(by_source),
        "by_reason": dict(by_reason),
    }


async def _run(args: argparse.Namespace) -> int:
    parsed_sources = _parse_sources(args.sources)
    sources = parsed_sources or _default_sources()
    run_id = f"backfill_low_substance_enriched_reviews:{datetime.now(timezone.utc).isoformat()}"
    logger.info("Connecting to %s", db_settings.dsn)
    conn = await asyncpg.connect(db_settings.dsn)
    try:
        candidate_rows = await _fetch_candidate_rows(
            conn,
            sources=sources,
            limit=args.limit,
        )
        planned = [item for item in (_plan_row_backfill(row) for row in candidate_rows) if item is not None]
        by_source = Counter(item["source"] for item in planned)
        by_reason = Counter(
            reason
            for item in planned
            for reason in item["low_fidelity_reasons"]
        )
        logger.info(
            "Scanned %d enriched rows across %s; planned quarantines=%d",
            len(candidate_rows),
            ",".join(sources),
            len(planned),
        )
        if planned:
            logger.info("Planned by source: %s", dict(by_source))
            logger.info("Planned by reason: %s", dict(by_reason))
            preview = planned[: min(5, len(planned))]
            logger.info("Preview rows: %s", json.dumps(preview, default=str))
        if not args.apply:
            logger.info("Dry run only. Re-run with --apply to persist updates.")
            return 0
        result = await _apply_updates(conn, updates=planned, run_id=run_id)
        logger.info("Applied %d low-fidelity quarantines", result["applied"])
        logger.info("Applied by source: %s", result["by_source"])
        logger.info("Applied by reason: %s", result["by_reason"])
        return 0
    finally:
        await conn.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated source allowlist. Default: configured noisy sources plus trustpilot.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max enriched rows to scan.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist quarantines. Omit for dry-run preview.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
