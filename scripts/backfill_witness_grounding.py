#!/usr/bin/env python3
"""Batch-populate `grounding_status` on b2b_vendor_witnesses (Phase 1b).

For each row where grounding_status = 'pending', joins b2b_reviews to fetch
summary + review_text, runs the same normalized grounding helper used at
write time, and updates grounding_status to 'grounded' or 'not_grounded'
plus stamps grounding_checked_at = NOW().

Safe to re-run: only touches pending rows. After successful completion the
release sign-off audit query

  SELECT grounding_status, count(*) FROM b2b_vendor_witnesses GROUP BY 1;

should return zero rows in 'pending'.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  # Sample check on 200 rows without writing
  python scripts/backfill_witness_grounding.py --dry-run --max-rows 200
  # Full run
  python scripts/backfill_witness_grounding.py
  # Throttled run for production-like databases
  python scripts/backfill_witness_grounding.py --batch-size 200

Exit codes:
  0 = success (or dry-run completed)
  1 = no pool / DB error before any work was done
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)

from atlas_brain.autonomous.tasks._b2b_grounding import check_phrase_grounded
from atlas_brain.storage.database import close_database, get_db_pool, init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_witness_grounding")

_DEFAULT_BATCH_SIZE = 500
_DEFAULT_DRY_RUN_CAP = 200


async def _fetch_pending_batch(
    conn,
    *,
    batch_size: int,
    offset: int,
    use_offset: bool,
) -> list[dict[str, Any]]:
    """Fetch one batch of pending witnesses with their source text.

    When `use_offset=True` (dry-run), pages through the result set
    deterministically so we don't loop forever on the same rows. When
    `use_offset=False` (real run), each iteration's SELECT naturally
    returns the next pending rows because previously-processed rows
    leave the partial index after being UPDATEd.
    """
    if use_offset:
        rows = await conn.fetch(
            """
            SELECT w.vendor_name, w.as_of_date, w.analysis_window_days,
                   w.schema_version, w.witness_id, w.excerpt_text,
                   r.summary, r.review_text
            FROM b2b_vendor_witnesses w
            LEFT JOIN b2b_reviews r ON r.id = w.review_id::uuid
            WHERE w.grounding_status = 'pending'
            ORDER BY w.witness_id
            LIMIT $1 OFFSET $2
            """,
            batch_size,
            offset,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT w.vendor_name, w.as_of_date, w.analysis_window_days,
                   w.schema_version, w.witness_id, w.excerpt_text,
                   r.summary, r.review_text
            FROM b2b_vendor_witnesses w
            LEFT JOIN b2b_reviews r ON r.id = w.review_id::uuid
            WHERE w.grounding_status = 'pending'
            ORDER BY w.witness_id
            LIMIT $1
            """,
            batch_size,
        )
    return [dict(r) for r in rows]


def _classify_row(row: dict[str, Any]) -> str:
    """Return the grounding_status value for a single witness row."""
    is_grounded = check_phrase_grounded(
        row.get("excerpt_text"),
        summary=row.get("summary"),
        review_text=row.get("review_text"),
    )
    return "grounded" if is_grounded else "not_grounded"


async def _apply_updates(
    conn,
    classified: list[tuple[str, str, Any, int, str, str]],
) -> None:
    """UPDATE the grounding columns for a batch in one round-trip.

    Each tuple is (status, vendor_name, as_of_date, analysis_window_days,
    schema_version, witness_id) -- matches the placeholder order in the
    UPDATE.
    """
    if not classified:
        return
    await conn.executemany(
        """
        UPDATE b2b_vendor_witnesses
           SET grounding_status = $1,
               grounding_checked_at = NOW()
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
        total_pending = await pool.fetchval(
            "SELECT count(*) FROM b2b_vendor_witnesses WHERE grounding_status = 'pending'"
        )
        logger.info("pending witnesses: %d", total_pending)
        if total_pending == 0:
            logger.info("nothing to do")
            return 0

        cap = min(total_pending, args.max_rows) if args.max_rows else total_pending
        if args.dry_run and not args.max_rows:
            cap = min(cap, _DEFAULT_DRY_RUN_CAP)
            logger.info(
                "dry-run defaulting to first %d rows; pass --max-rows to override",
                cap,
            )

        processed = 0
        grounded_total = 0
        not_grounded_total = 0

        while processed < cap:
            batch_n = min(args.batch_size, cap - processed)
            conn = await pool.acquire()
            try:
                rows = await _fetch_pending_batch(
                    conn,
                    batch_size=batch_n,
                    offset=processed if args.dry_run else 0,
                    use_offset=args.dry_run,
                )
                if not rows:
                    break

                classified: list[tuple[str, str, Any, int, str, str]] = []
                grounded_in_batch = 0
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
                    if status == "grounded":
                        grounded_in_batch += 1

                if not args.dry_run:
                    async with conn.transaction():
                        await _apply_updates(conn, classified)

                processed += len(rows)
                grounded_total += grounded_in_batch
                not_grounded_total += len(rows) - grounded_in_batch
                logger.info(
                    "batch: %d processed, grounded=%d not_grounded=%d "
                    "(running total %d/%d)",
                    len(rows),
                    grounded_in_batch,
                    len(rows) - grounded_in_batch,
                    processed,
                    cap,
                )
            finally:
                await pool.release(conn)

        if processed == 0:
            logger.warning("no rows fetched even though %d pending; aborting",
                           total_pending)
            return 0

        grounded_pct = 100 * grounded_total / processed
        not_grounded_pct = 100 * not_grounded_total / processed
        suffix = " [DRY RUN -- no rows updated]" if args.dry_run else ""
        logger.info(
            "complete: processed=%d, grounded=%d (%.1f%%), "
            "not_grounded=%d (%.1f%%)%s",
            processed, grounded_total, grounded_pct,
            not_grounded_total, not_grounded_pct, suffix,
        )

        if not args.dry_run:
            remaining = await pool.fetchval(
                "SELECT count(*) FROM b2b_vendor_witnesses "
                "WHERE grounding_status = 'pending'"
            )
            if remaining:
                logger.warning(
                    "remaining pending rows: %d "
                    "(may be newly inserted; re-run if expected to be 0)",
                    remaining,
                )
            else:
                logger.info("audit: 0 rows in 'pending' state")
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
