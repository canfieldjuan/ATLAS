#!/usr/bin/env python3
"""Phase 7 Step A: deterministic finalize pass on existing v4 enrichments.

These reviews were extracted under the v4 Tier 1 prompt (phrase_metadata
is populated) but pre-date Phase 4, so their enrichment JSONB is missing
`pain_confidence` and may have stale `pain_category` / `pain_categories`
values from before the Layer-3 causality gate was wired.

We call `_finalize_enrichment_for_persist()` on each row, which runs
`_compute_derived_fields()` on the existing payload -- no LLM call, no
new extraction. That helper:
  - Reruns `_derive_pain_categories` (Phase 2 subject + Phase 3 polarity)
  - Reruns `engine.override_pain`
  - Applies the Phase 4 causality gate + demotion
  - Stamps `pain_confidence`
  - Rebuilds `evidence_spans` so each span carries subject/polarity/role/
    verbatim (Phase 5a)

The pass is idempotent: if a row is already consistent, we skip the
UPDATE. It only touches rows that match the synthesis candidate filter
(enrichment_status = 'enriched', duplicate_of_review_id IS NULL) so we
don't finalize rows that synthesis will never use anyway.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  # Dry-run (prints what would change, does not UPDATE)
  python scripts/finalize_v4_enrichments.py --dry-run
  # Real run
  python scripts/finalize_v4_enrichments.py

Exit codes:
  0 = success (or dry-run completed)
  1 = DB / setup error
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)

from atlas_brain.autonomous.tasks.b2b_enrichment import (
    _finalize_enrichment_for_persist,
)
from atlas_brain.storage.database import close_database, get_db_pool, init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("finalize_v4_enrichments")


_V4_CANDIDATE_QUERY = """
SELECT id, vendor_name, rating, rating_max, summary, review_text, pros,
       cons, raw_metadata, content_type, reviewer_company, reviewer_title,
       source, reviewed_at, enrichment
FROM b2b_reviews
WHERE COALESCE((enrichment->>'enrichment_schema_version')::int, 0) >= 4
  AND enrichment_status = 'enriched'
  AND duplicate_of_review_id IS NULL
ORDER BY id
"""


def _coerce_enrichment(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _coerce_raw_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _source_row_for_finalize(row: dict[str, Any]) -> dict[str, Any]:
    """Slim source_row dict matching what _compute_derived_fields reads."""
    return {
        "id": str(row["id"]),
        "vendor_name": row.get("vendor_name"),
        "rating": row.get("rating"),
        "rating_max": row.get("rating_max"),
        "summary": row.get("summary") or "",
        "review_text": row.get("review_text") or "",
        "pros": row.get("pros") or "",
        "cons": row.get("cons") or "",
        "raw_metadata": _coerce_raw_metadata(row.get("raw_metadata")),
        "content_type": row.get("content_type") or "review",
        "reviewer_company": row.get("reviewer_company"),
        "reviewer_title": row.get("reviewer_title"),
        "source": row.get("source"),
        "reviewed_at": row.get("reviewed_at"),
    }


def _summarize_delta(before: dict, after: dict) -> dict[str, tuple]:
    """Identify which top-level fields changed between before/after enrichment.

    Used for logging only. Returns {field: (before_value, after_value)} for
    a small set of Phase 4/5-relevant fields.
    """
    watched = (
        "pain_category",
        "pain_confidence",
        "pain_categories",
    )
    delta: dict[str, tuple] = {}
    for key in watched:
        before_v = before.get(key)
        after_v = after.get(key)
        if before_v != after_v:
            delta[key] = (before_v, after_v)
    return delta


async def _main_async(args: argparse.Namespace) -> int:
    await init_database()
    pool = get_db_pool()
    try:
        rows = await pool.fetch(_V4_CANDIDATE_QUERY)
        total = len(rows)
        logger.info("v4 enriched non-duplicate witness candidates: %d", total)
        if total == 0:
            logger.info("nothing to finalize")
            return 0

        processed = 0
        skipped_no_change = 0
        updated = 0
        failed = 0
        failure_reasons: dict[str, int] = {}

        # One transaction per row keeps blast radius small at the cost of
        # extra round-trips. Scope is tiny (~dozens) so this is fine.
        for row in rows:
            processed += 1
            row_dict = dict(row)
            enrichment = _coerce_enrichment(row_dict.get("enrichment"))
            if not enrichment:
                failed += 1
                failure_reasons["missing_or_invalid_enrichment"] = (
                    failure_reasons.get("missing_or_invalid_enrichment", 0) + 1
                )
                continue

            # Preserve a copy of the current payload so we can diff for logs
            # and so skip-on-equal works without false positives from
            # _finalize serialising through json.dumps.
            before = json.loads(json.dumps(enrichment, default=str))

            source_row = _source_row_for_finalize(row_dict)
            finalized, error = _finalize_enrichment_for_persist(
                enrichment, source_row
            )
            if finalized is None:
                failed += 1
                reason = error or "unknown"
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                logger.info(
                    "finalize_failed id=%s reason=%s", row_dict["id"], reason,
                )
                continue

            # Normalise both sides through the same serializer to avoid
            # spurious diffs from key ordering or datetime formatting.
            after_json = json.dumps(finalized, sort_keys=True, default=str)
            before_json = json.dumps(before, sort_keys=True, default=str)
            if after_json == before_json:
                skipped_no_change += 1
                continue

            delta = _summarize_delta(before, finalized)

            if args.dry_run:
                logger.info(
                    "would_update id=%s vendor=%s delta=%s",
                    row_dict["id"],
                    row_dict.get("vendor_name"),
                    delta,
                )
                continue

            await pool.execute(
                """
                UPDATE b2b_reviews
                   SET enrichment = $1::jsonb
                 WHERE id = $2
                """,
                after_json,
                row_dict["id"],
            )
            updated += 1
            logger.info(
                "updated id=%s vendor=%s delta=%s",
                row_dict["id"],
                row_dict.get("vendor_name"),
                delta,
            )

        suffix = " [DRY RUN -- no rows updated]" if args.dry_run else ""
        logger.info(
            "complete: processed=%d updated=%d skipped_no_change=%d "
            "failed=%d failure_reasons=%s%s",
            processed,
            updated if not args.dry_run else 0,
            skipped_no_change,
            failed,
            failure_reasons,
            suffix,
        )
        return 0
    finally:
        await close_database()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the finalize result but do NOT UPDATE any rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
