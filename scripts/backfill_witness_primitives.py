#!/usr/bin/env python3
"""Deterministically backfill witness-ready enrichment primitives.

Reuses the persisted enrichment JSON plus raw review context and runs the same
finalization helper used by canonical enrichment. No LLM calls are made.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_witness_primitives")


def _candidate_clause() -> str:
    return """
      AND (
        COALESCE(enrichment->>'replacement_mode', '') = ''
        OR COALESCE(enrichment->>'operating_model_shift', '') = ''
        OR COALESCE(enrichment->>'productivity_delta_claim', '') = ''
        OR COALESCE(enrichment->>'org_pressure_type', '') = ''
        OR enrichment->'salience_flags' IS NULL
        OR enrichment->'evidence_spans' IS NULL
        OR COALESCE(enrichment->>'evidence_map_hash', '') = ''
        OR jsonb_typeof(enrichment->'salience_flags') != 'array'
        OR jsonb_typeof(enrichment->'evidence_spans') != 'array'
        OR CASE
             WHEN jsonb_typeof(enrichment->'salience_flags') = 'array'
             THEN jsonb_array_length(enrichment->'salience_flags')
             ELSE 0
           END = 0
        OR CASE
             WHEN jsonb_typeof(enrichment->'evidence_spans') = 'array'
             THEN jsonb_array_length(enrichment->'evidence_spans')
             ELSE 0
           END = 0
      )
    """


async def _fetch_rows(pool, args) -> list[dict[str, Any]]:
    clauses = [_candidate_clause()]
    params: list[Any] = []
    idx = 1
    if args.vendor:
        clauses.append(f"AND vendor_name = ${idx}")
        params.append(args.vendor)
        idx += 1
    params.append(args.limit)
    return await pool.fetch(
        f"""
        SELECT id, vendor_name, source, content_type, summary, review_text,
               pros, cons, reviewer_title, reviewer_company, company_size_raw,
               reviewer_industry, rating, rating_max, raw_metadata, enrichment,
               enriched_at
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
        {' '.join(clauses)}
        ORDER BY enriched_at DESC NULLS LAST, id DESC
        LIMIT ${idx}
        """,
        *params,
    )


async def run(args) -> None:
    from atlas_brain.autonomous.tasks.b2b_enrichment import _finalize_enrichment_for_persist
    from atlas_brain.storage.database import close_database, get_db_pool, init_database

    await init_database()
    pool = get_db_pool()
    rows = await _fetch_rows(pool, args)
    logger.info("Scanned %d candidate reviews", len(rows))

    updated = 0
    skipped = 0

    for row in rows:
        raw_enrichment = row.get("enrichment")
        if isinstance(raw_enrichment, str):
            try:
                enrichment = json.loads(raw_enrichment)
            except json.JSONDecodeError:
                skipped += 1
                continue
        elif isinstance(raw_enrichment, dict):
            enrichment = dict(raw_enrichment)
        else:
            skipped += 1
            continue

        finalized, _ = _finalize_enrichment_for_persist(enrichment, dict(row))
        if not finalized or finalized == enrichment:
            skipped += 1
            continue

        if args.dry_run:
            updated += 1
            continue

        await pool.execute(
            """
            UPDATE b2b_reviews
            SET enrichment = $1::jsonb
            WHERE id = $2
            """,
            json.dumps(finalized, default=str),
            row["id"],
        )
        updated += 1

    logger.info(
        "%s %d reviews, skipped %d unchanged/invalid",
        "Would update" if args.dry_run else "Updated",
        updated,
        skipped,
    )
    await close_database()


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill witness-ready enrichment primitives deterministically")
    ap.add_argument("--vendor", default=None, help="Target one vendor_name")
    ap.add_argument("--limit", type=int, default=5000, help="Max reviews to scan")
    ap.add_argument("--dry-run", action="store_true", help="Show scope without updating rows")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
