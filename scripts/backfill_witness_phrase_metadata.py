#!/usr/bin/env python3
"""Batch-populate `pain_confidence` on b2b_vendor_witnesses (Phase 5b).

For each row whose `pain_confidence` is NULL, joins b2b_reviews to read the
review's enrichment JSONB, recomputes the per-review pain_confidence using
the same Layer-3 helper that Phase 4 wires into _compute_derived_fields,
and stamps the result on the witness row.

Phase 5b deliberately does NOT backfill phrase_polarity / phrase_subject /
phrase_role / phrase_verbatim. Those are span-level tags whose mapping
back to a witness requires re-running witness selection -- the existing
synthesis cron does that on each vendor-as_of_date pair, so those columns
will fill in naturally on the next synthesis run. Backfilling them here
would require fragile substring matching from witness.excerpt_text back
to enrichment.phrase_metadata phrases and could mis-attribute when
multiple phrases share a sentence.

Safe to re-run: only touches rows with NULL pain_confidence. Idempotent.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  # Sample check on 200 rows without writing
  python scripts/backfill_witness_phrase_metadata.py --dry-run --max-rows 200
  # Full run
  python scripts/backfill_witness_phrase_metadata.py
  # Throttled run for production-like databases
  python scripts/backfill_witness_phrase_metadata.py --batch-size 200

Exit codes:
  0 = success (or dry-run completed)
  1 = no pool / DB error before any work was done
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

from atlas_brain.autonomous.tasks.b2b_enrichment import _compute_pain_confidence
from atlas_brain.storage.database import close_database, get_db_pool, init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_witness_phrase_metadata")

_DEFAULT_BATCH_SIZE = 500
_DEFAULT_DRY_RUN_CAP = 200


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


async def _fetch_null_batch(
    conn,
    *,
    batch_size: int,
    last_key: tuple[str, Any, int, str, str] | None,
) -> list[dict[str, Any]]:
    """Fetch one deterministic batch of witnesses missing pain_confidence.

    Keyset pagination is used for both dry-run and write mode. In write mode
    we intentionally skip UPDATEs for rows whose enrichment cannot be graded,
    so repeatedly selecting the first NULL page could count the same rows as
    progress and starve later rows. Advancing by the composite row key avoids
    duplicate processing while still making the script safe to re-run.
    """
    order_clause = """
            ORDER BY w.vendor_name, w.as_of_date, w.analysis_window_days,
                     w.schema_version, w.witness_id
    """
    if last_key is None:
        rows = await conn.fetch(
            f"""
            SELECT w.vendor_name, w.as_of_date, w.analysis_window_days,
                   w.schema_version, w.witness_id, w.pain_category,
                   r.enrichment
            FROM b2b_vendor_witnesses w
            LEFT JOIN b2b_reviews r ON r.id = w.review_id::uuid
            WHERE w.pain_confidence IS NULL
            {order_clause}
            LIMIT $1
            """,
            batch_size,
        )
    else:
        rows = await conn.fetch(
            f"""
            SELECT w.vendor_name, w.as_of_date, w.analysis_window_days,
                   w.schema_version, w.witness_id, w.pain_category,
                   r.enrichment
            FROM b2b_vendor_witnesses w
            LEFT JOIN b2b_reviews r ON r.id = w.review_id::uuid
            WHERE w.pain_confidence IS NULL
              AND (
                    w.vendor_name,
                    w.as_of_date,
                    w.analysis_window_days,
                    w.schema_version,
                    w.witness_id
                  ) > ($2, $3, $4, $5, $6)
            {order_clause}
            LIMIT $1
            """,
            batch_size,
            *last_key,
        )
    return [dict(r) for r in rows]


def _row_key(row: dict[str, Any]) -> tuple[str, Any, int, str, str]:
    """Return the composite key used by _fetch_null_batch ordering."""
    return (
        str(row["vendor_name"]),
        row["as_of_date"],
        int(row["analysis_window_days"]),
        str(row["schema_version"]),
        str(row["witness_id"]),
    )


def _classify_row(row: dict[str, Any]) -> str | None:
    """Return the pain_confidence value for a single witness row.

    Prefers `enrichment.pain_confidence` when the enrichment was already
    stamped by Phase 4 (post-2026-04-24 enrichments). Falls back to
    re-running the rubric on the enrichment JSONB so we can grade older
    enrichments that pre-date the column.

    Returns None when there's no enrichment to grade against -- the row
    stays NULL until next synthesis.
    """
    enrichment = _coerce_enrichment(row.get("enrichment"))
    if not enrichment:
        return None
    stamped = enrichment.get("pain_confidence")
    if isinstance(stamped, str) and stamped in {"strong", "weak", "none"}:
        return stamped
    pain_category = (
        str(row.get("pain_category") or "").strip()
        or str(enrichment.get("pain_category") or "").strip()
        or "overall_dissatisfaction"
    )
    return _compute_pain_confidence(enrichment, pain_category)


async def _apply_updates(
    conn,
    classified: list[tuple[str | None, str, Any, int, str, str]],
) -> None:
    if not classified:
        return
    await conn.executemany(
        """
        UPDATE b2b_vendor_witnesses
           SET pain_confidence = $1
         WHERE vendor_name = $2
           AND as_of_date = $3
           AND analysis_window_days = $4
           AND schema_version = $5
           AND witness_id = $6
        """,
        classified,
    )


async def _main_async(args: argparse.Namespace) -> int:
    await init_database()
    pool = get_db_pool()
    try:
        total_null = await pool.fetchval(
            "SELECT count(*) FROM b2b_vendor_witnesses WHERE pain_confidence IS NULL"
        )
        logger.info("witnesses with NULL pain_confidence: %d", total_null)
        if total_null == 0:
            logger.info("nothing to do")
            return 0

        cap = min(total_null, args.max_rows) if args.max_rows else total_null
        if args.dry_run and not args.max_rows:
            cap = min(cap, _DEFAULT_DRY_RUN_CAP)
            logger.info(
                "dry-run defaulting to first %d rows; pass --max-rows to override",
                cap,
            )

        processed = 0
        graded_counts = {"strong": 0, "weak": 0, "none": 0, "null": 0}
        last_key: tuple[str, Any, int, str, str] | None = None

        while processed < cap:
            batch_n = min(args.batch_size, cap - processed)
            conn = await pool.acquire()
            try:
                rows = await _fetch_null_batch(
                    conn,
                    batch_size=batch_n,
                    last_key=last_key,
                )
                if not rows:
                    break
                last_key = _row_key(rows[-1])

                classified: list[tuple[str | None, str, Any, int, str, str]] = []
                batch_counts = {"strong": 0, "weak": 0, "none": 0, "null": 0}
                for row in rows:
                    status = _classify_row(row)
                    classified.append((
                        status,
                        row["vendor_name"],
                        row["as_of_date"],
                        row["analysis_window_days"],
                        row["schema_version"],
                        row["witness_id"],
                    ))
                    bucket = status if status in batch_counts else "null"
                    batch_counts[bucket] = batch_counts.get(bucket, 0) + 1

                if not args.dry_run:
                    # Skip the UPDATE for rows whose status is None unless
                    # we explicitly want to leave them at NULL (which is
                    # already the column default). UPDATE-to-NULL is a
                    # no-op anyway for these rows but skipping saves
                    # write amplification.
                    writable = [c for c in classified if c[0] is not None]
                    async with conn.transaction():
                        await _apply_updates(conn, writable)

                processed += len(rows)
                for k, v in batch_counts.items():
                    graded_counts[k] = graded_counts.get(k, 0) + v
                logger.info(
                    "batch: %d processed, strong=%d weak=%d none=%d "
                    "null=%d (running total %d/%d)",
                    len(rows),
                    batch_counts.get("strong", 0),
                    batch_counts.get("weak", 0),
                    batch_counts.get("none", 0),
                    batch_counts.get("null", 0),
                    processed,
                    cap,
                )
            finally:
                await pool.release(conn)

        if processed == 0:
            logger.warning(
                "no rows fetched even though %d had NULL pain_confidence; aborting",
                total_null,
            )
            return 0

        suffix = " [DRY RUN -- no rows updated]" if args.dry_run else ""
        logger.info(
            "complete: processed=%d, strong=%d weak=%d none=%d "
            "kept_null=%d%s",
            processed,
            graded_counts.get("strong", 0),
            graded_counts.get("weak", 0),
            graded_counts.get("none", 0),
            graded_counts.get("null", 0),
            suffix,
        )

        if not args.dry_run:
            remaining = await pool.fetchval(
                "SELECT count(*) FROM b2b_vendor_witnesses "
                "WHERE pain_confidence IS NULL"
            )
            logger.info(
                "remaining NULL after backfill: %d "
                "(rows whose enrichment was empty / unparseable -- expected to "
                "fill on next synthesis)",
                remaining,
            )
        return 0
    finally:
        await close_database()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=f"Rows per UPDATE round-trip (default {_DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap total processed rows (default: drain everything).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Compute statuses but skip UPDATE; defaults --max-rows to "
             f"{_DEFAULT_DRY_RUN_CAP}.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
