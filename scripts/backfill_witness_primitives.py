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
        OR (
          COALESCE(enrichment->>'productivity_delta_claim', '') IN ('unknown', 'none')
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(save time|saves time|time savings|sped up|streamlined workflow|reduced manual work|less manual work|slowed us down|time consuming|takes too long|waste of time|more manual work|too many manual steps)'
        )
        OR (
          COALESCE(enrichment->>'org_pressure_type', '') IN ('none', '')
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(procurement|approved software|approved list|security review|legal review|vendor review|standardized on|standardisation|standardization|preferred vendor|company standard|it standard|standard tool|budget freeze|spend freeze|cost cutting|cost-cutting|budget cuts|spend reduction)'
        )
        OR (
          COALESCE(enrichment#>>'{timeline,decision_timeline}', '') IN ('', 'unknown', 'none')
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(renewal|contract|evaluate|evaluation|considering|switch|switching|migration|migrate|deadline|cutover|go live|go-live|budget|pricing|price|cost|expensive|replace|replaced)'
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(renewal|deadline|contract end|contract expires|next quarter|q1|q2|q3|q4|30 days|60 days|90 days|this month|this week)'
        )
        OR (
          COALESCE(enrichment#>>'{timeline,contract_end}', '') = ''
          AND COALESCE(enrichment#>>'{timeline,evaluation_deadline}', '') = ''
          AND COALESCE(enrichment#>>'{churn_signals,renewal_timing}', '') = ''
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(renewal|contract|evaluate|evaluation|considering|switch|switching|migration|migrate|cancel|notice|deadline|cutover|go live|go-live)'
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(next quarter|this quarter|next month|this month|end of month|month end|end of quarter|quarter end|30 days|60 days|90 days|few days|few weeks|notice period|before renewal|before the contract ends|before the contract expires|term ends|term expires)'
        )
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
    field_changes = {
        "replacement_mode": 0,
        "operating_model_shift": 0,
        "productivity_delta_claim": 0,
        "org_pressure_type": 0,
        "decision_timeline": 0,
        "contract_end": 0,
        "evaluation_deadline": 0,
        "salience_flags": 0,
        "evidence_spans": 0,
        "evidence_map_hash": 0,
    }

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

        changed = False
        for field in field_changes:
            if field == "decision_timeline":
                before = (
                    ((enrichment.get("timeline") or {}) if isinstance(enrichment.get("timeline"), dict) else {})
                    .get("decision_timeline")
                )
                after = (
                    ((finalized.get("timeline") or {}) if isinstance(finalized.get("timeline"), dict) else {})
                    .get("decision_timeline")
                )
                if before != after:
                    field_changes[field] += 1
                    changed = True
                continue
            if field in {"contract_end", "evaluation_deadline"}:
                before = (
                    ((enrichment.get("timeline") or {}) if isinstance(enrichment.get("timeline"), dict) else {})
                    .get(field)
                )
                after = (
                    ((finalized.get("timeline") or {}) if isinstance(finalized.get("timeline"), dict) else {})
                    .get(field)
                )
                if before != after:
                    field_changes[field] += 1
                    changed = True
                continue
            if enrichment.get(field) != finalized.get(field):
                field_changes[field] += 1
                changed = True
        if not changed:
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
        "%s %d reviews, skipped %d unchanged/invalid; field changes=%s",
        "Would update" if args.dry_run else "Updated",
        updated,
        skipped,
        ", ".join(f"{field}:{count}" for field, count in field_changes.items()),
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
