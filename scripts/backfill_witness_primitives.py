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


# APPROVED-ENRICHMENT-READ: replacement_mode, operating_model_shift, productivity_delta_claim, org_pressure_type, salience_flags, evidence_spans, evidence_map_hash, timeline, reviewer_context, churn_signals, buyer_authority, budget_signals
# Reason: backfill/migration script — direct enrichment access required
def _candidate_clause() -> str:
    return """
      AND (
        LOWER(COALESCE(reviewer_title, '')) LIKE 'repeat churn signal%'
        OR
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
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(save time|saves time|time savings|sped up|streamlin(e|ed|es|ing)|reduced manual work|less manual work|less labor-intensive|less labour-intensive|simplif(y|ies|ied)|administrative complexity|manual data entry|response time|incident response|slowed us down|time consuming|takes too long|waste of time|more manual work|too many manual steps)'
        )
        OR (
          COALESCE(enrichment->>'operating_model_shift', 'none') = 'none'
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(all-in-one|one platform|single system|one place|everything in one place|source of truth)'
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
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(renewal|contract|evaluate|evaluation|considering|switch|switching|migration|migrate|cancel|notice|deadline|cutover|go live|go-live|current contract)'
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(next quarter|this quarter|next month|this month|end of month|month end|end of quarter|quarter end|30 days|60 days|90 days|few days|few weeks|notice period|before renewal|at renewal|upon renewal|auto[- ]?renew|annual renewal|next renewal|before the contract ends|before the contract expires|term ends|term expires|final month)'
        )
        OR (
          COALESCE(enrichment#>>'{reviewer_context,company_name}', '') = ''
          AND NULLIF(trim(reviewer_company), '') IS NOT NULL
        )
        OR (
          NULLIF(trim(reviewer_title), '') IS NOT NULL
          AND COALESCE(enrichment#>>'{reviewer_context,role_level}', 'unknown') = 'unknown'
        )
        OR (
          NULLIF(trim(reviewer_title), '') IS NOT NULL
          AND COALESCE(enrichment#>>'{reviewer_context,decision_maker}', 'false') IN ('false', '')
          AND (
            COALESCE(enrichment#>>'{buyer_authority,has_budget_authority}', 'false') = 'true'
            OR COALESCE(enrichment#>>'{churn_signals,actively_evaluating}', 'false') = 'true'
            OR COALESCE(enrichment#>>'{churn_signals,migration_in_progress}', 'false') = 'true'
            OR COALESCE(enrichment#>>'{churn_signals,contract_renewal_mentioned}', 'false') = 'true'
            OR COALESCE(enrichment#>>'{timeline,contract_end}', '') <> ''
            OR COALESCE(enrichment#>>'{timeline,evaluation_deadline}', '') <> ''
            OR COALESCE(enrichment#>>'{budget_signals,annual_spend_estimate}', '') <> ''
            OR COALESCE(enrichment#>>'{budget_signals,price_per_seat}', '') <> ''
            OR COALESCE(enrichment#>>'{budget_signals,price_increase_detail}', '') <> ''
            OR COALESCE(enrichment#>>'{budget_signals,price_increase_mentioned}', 'false') = 'true'
          )
        )
        OR (
          COALESCE(enrichment#>>'{budget_signals,annual_spend_estimate}', '') ~* 'ayear|ayr'
        )
        OR (
          COALESCE(enrichment#>>'{budget_signals,annual_spend_estimate}', '') = ''
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(\\$\\s?\\d|usd\\s?\\d)'
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(renewal|quote|quoted|contract|subscription|license|annual|annually|yearly|a year|a yr|per year|/\\s*year|/\\s*yr)'
        )
        OR (
          COALESCE(enrichment#>>'{budget_signals,price_per_seat}', '') = ''
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(\\$\\s?\\d|usd\\s?\\d)'
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(per seat|/\\s*seat|per user|/\\s*user|per license|/\\s*license)'
        )
        OR (
          COALESCE(enrichment#>>'{budget_signals,seat_count}', '') = ''
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '\\m\\d[\\d,]*\\M\\s+(seat|seats|user|users|license|licenses|licence|licences)'
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(pricing|price|budget|renewal|quote|quoted|contract|subscription|license|seat|user)'
        )
        OR (
          COALESCE(enrichment#>>'{budget_signals,annual_spend_estimate}', '') = ''
          AND COALESCE(enrichment#>>'{budget_signals,price_per_seat}', '') <> ''
          AND COALESCE(enrichment#>>'{budget_signals,seat_count}', '') <> ''
          AND COALESCE(enrichment#>>'{budget_signals,price_per_seat}', '') ~* '(monthly|per\\s+month|/\\s*month|/\\s*mo|a\\s+month|annual|annually|yearly|per\\s+year|/\\s*year|/\\s*yr|a\\s+year|a\\s+yr)'
        )
        OR (
          COALESCE(enrichment#>>'{budget_signals,price_increase_mentioned}', '') IN ('', 'false')
          AND concat_ws(' ', summary, review_text, pros, cons) ~* '(\\d+(?:\\.\\d+)?%\\s+(price\\s+)?(increase|higher|more|jump|hike)|price\\s+increase|pricing\\s+increase|renewal\\s+increase|raised\\s+(our\\s+)?(price|pricing|renewal|invoice)|increased\\s+(our\\s+)?(price|pricing|renewal|invoice))'
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
    from atlas_brain.services.b2b.reviewer_identity import sanitize_reviewer_title
    from atlas_brain.storage.database import close_database, get_db_pool, init_database

    await init_database()
    pool = get_db_pool()
    rows = await _fetch_rows(pool, args)
    logger.info("Scanned %d candidate reviews", len(rows))

    updated = 0
    skipped = 0
    field_changes = {
        "reviewer_title_cleanup": 0,
        "replacement_mode": 0,
        "operating_model_shift": 0,
        "productivity_delta_claim": 0,
        "org_pressure_type": 0,
        "decision_timeline": 0,
        "contract_end": 0,
        "evaluation_deadline": 0,
        "reviewer_company_name": 0,
        "role_level": 0,
        "decision_maker": 0,
        "buyer_role_type": 0,
        "annual_spend_estimate": 0,
        "price_per_seat": 0,
        "seat_count": 0,
        "price_increase_mentioned": 0,
        "price_increase_detail": 0,
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
        cleaned_reviewer_title = sanitize_reviewer_title(row.get("reviewer_title"))
        if not finalized:
            skipped += 1
            continue

        changed = False
        if row.get("reviewer_title") != cleaned_reviewer_title:
            field_changes["reviewer_title_cleanup"] += 1
            changed = True
        for field in field_changes:
            if field == "reviewer_title_cleanup":
                continue
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
            if field == "reviewer_company_name":
                before = (
                    ((enrichment.get("reviewer_context") or {}) if isinstance(enrichment.get("reviewer_context"), dict) else {})
                    .get("company_name")
                )
                after = (
                    ((finalized.get("reviewer_context") or {}) if isinstance(finalized.get("reviewer_context"), dict) else {})
                    .get("company_name")
                )
                if before != after:
                    field_changes[field] += 1
                    changed = True
                continue
            if field in {"role_level", "decision_maker"}:
                before = (
                    ((enrichment.get("reviewer_context") or {}) if isinstance(enrichment.get("reviewer_context"), dict) else {})
                    .get(field)
                )
                after = (
                    ((finalized.get("reviewer_context") or {}) if isinstance(finalized.get("reviewer_context"), dict) else {})
                    .get(field)
                )
                if before != after:
                    field_changes[field] += 1
                    changed = True
                continue
            if field == "buyer_role_type":
                before = (
                    ((enrichment.get("buyer_authority") or {}) if isinstance(enrichment.get("buyer_authority"), dict) else {})
                    .get("role_type")
                )
                after = (
                    ((finalized.get("buyer_authority") or {}) if isinstance(finalized.get("buyer_authority"), dict) else {})
                    .get("role_type")
                )
                if before != after:
                    field_changes[field] += 1
                    changed = True
                continue
            if field in {
                "annual_spend_estimate",
                "price_per_seat",
                "seat_count",
                "price_increase_mentioned",
                "price_increase_detail",
            }:
                before = (
                    ((enrichment.get("budget_signals") or {}) if isinstance(enrichment.get("budget_signals"), dict) else {})
                    .get(field)
                )
                after = (
                    ((finalized.get("budget_signals") or {}) if isinstance(finalized.get("budget_signals"), dict) else {})
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
            SET enrichment = $1::jsonb,
                reviewer_title = $2
            WHERE id = $3
            """,
            json.dumps(finalized, default=str),
            cleaned_reviewer_title,
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
