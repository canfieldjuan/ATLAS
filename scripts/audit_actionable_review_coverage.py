#!/usr/bin/env python3
"""Audit review identity and actionability coverage across canonical reviews."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for env_name in (".env", ".env.local"):
    env_path = REPO_ROOT / env_name
    if not env_path.exists():
        continue
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

from atlas_brain.storage.database import close_database, get_db_pool, init_database


WINDOW_FILTER = """
    AND ($1::int IS NULL OR reviewed_at >= NOW() - make_interval(days => $1))
"""


async def _fetch_scalar_report(days: int | None) -> dict[str, int]:
    pool = get_db_pool()
    row = await pool.fetchrow(
        f"""
        SELECT
            COUNT(*)::int AS canonical_reviews,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched')::int AS enriched_reviews,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND NULLIF(BTRIM(reviewer_company), '') IS NOT NULL
            )::int AS enriched_with_company,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND NULLIF(BTRIM(reviewer_title), '') IS NOT NULL
            )::int AS enriched_with_title,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND NULLIF(BTRIM(company_size_raw), '') IS NOT NULL
            )::int AS enriched_with_company_size,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND NULLIF(BTRIM(reviewer_industry), '') IS NOT NULL
            )::int AS enriched_with_industry,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND COALESCE(NULLIF(BTRIM(enrichment #>> '{{reviewer_context,company_name}}'), ''), NULL) IS NOT NULL
            )::int AS enriched_with_enrichment_company,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND COALESCE((enrichment #>> '{{reviewer_context,decision_maker}}')::boolean, false) = true
            )::int AS enriched_decision_maker,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND COALESCE(NULLIF(BTRIM(enrichment #>> '{{timeline,decision_timeline}}'), ''), 'unknown') <> 'unknown'
            )::int AS enriched_with_known_timeline
        FROM b2b_reviews
        WHERE duplicate_of_review_id IS NULL
        {WINDOW_FILTER}
        """,
        days,
    )
    return dict(row or {})


async def _fetch_source_report(days: int | None, min_enriched: int) -> list[dict[str, object]]:
    pool = get_db_pool()
    rows = await pool.fetch(
        f"""
        SELECT
            source,
            COALESCE(raw_metadata->>'extraction_method', '') AS extraction_method,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched')::int AS enriched_reviews,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND NULLIF(BTRIM(reviewer_company), '') IS NOT NULL
            )::int AS with_company,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND NULLIF(BTRIM(reviewer_title), '') IS NOT NULL
            )::int AS with_title,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND NULLIF(BTRIM(company_size_raw), '') IS NOT NULL
            )::int AS with_company_size,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND NULLIF(BTRIM(reviewer_industry), '') IS NOT NULL
            )::int AS with_industry,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND COALESCE(NULLIF(BTRIM(enrichment #>> '{{reviewer_context,company_name}}'), ''), NULL) IS NOT NULL
            )::int AS with_enrichment_company
            ,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND COALESCE((enrichment #>> '{{reviewer_context,decision_maker}}')::boolean, false) = true
            )::int AS with_decision_maker,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND COALESCE(NULLIF(BTRIM(enrichment #>> '{{timeline,decision_timeline}}'), ''), 'unknown') <> 'unknown'
            )::int AS with_known_timeline,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND (
                        NULLIF(BTRIM(reviewer_company), '') IS NOT NULL
                     OR COALESCE(NULLIF(BTRIM(enrichment #>> '{{reviewer_context,company_name}}'), ''), NULL) IS NOT NULL
                     OR NULLIF(BTRIM(reviewer_title), '') IS NOT NULL
                     OR NULLIF(BTRIM(company_size_raw), '') IS NOT NULL
                     OR NULLIF(BTRIM(reviewer_industry), '') IS NOT NULL
                     OR COALESCE((enrichment #>> '{{reviewer_context,decision_maker}}')::boolean, false) = true
                     OR COALESCE(NULLIF(BTRIM(enrichment #>> '{{timeline,decision_timeline}}'), ''), 'unknown') <> 'unknown'
                  )
            )::int AS with_actionable_context
        FROM b2b_reviews
        WHERE duplicate_of_review_id IS NULL
        {WINDOW_FILTER}
        GROUP BY source, COALESCE(raw_metadata->>'extraction_method', '')
        HAVING COUNT(*) FILTER (WHERE enrichment_status = 'enriched') >= $2
        ORDER BY enriched_reviews DESC, source ASC, extraction_method ASC
        """,
        days,
        min_enriched,
    )
    enriched_rows: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        total = int(item.get("enriched_reviews") or 0)
        if total <= 0:
            continue
        item["company_rate"] = round(int(item.get("with_company") or 0) / total, 4)
        item["enrichment_company_rate"] = round(
            int(item.get("with_enrichment_company") or 0) / total, 4
        )
        item["title_rate"] = round(int(item.get("with_title") or 0) / total, 4)
        item["firmographic_rate"] = round(
            max(
                int(item.get("with_company_size") or 0),
                int(item.get("with_industry") or 0),
            )
            / total,
            4,
        )
        item["decision_maker_rate"] = round(
            int(item.get("with_decision_maker") or 0) / total, 4
        )
        item["known_timeline_rate"] = round(
            int(item.get("with_known_timeline") or 0) / total, 4
        )
        item["actionable_context_rate"] = round(
            int(item.get("with_actionable_context") or 0) / total, 4
        )
        enriched_rows.append(item)
    return enriched_rows


async def _fetch_buying_stage_report(days: int | None) -> list[dict[str, object]]:
    pool = get_db_pool()
    rows = await pool.fetch(
        f"""
        SELECT
            COALESCE(NULLIF(BTRIM(enrichment #>> '{{buyer_authority,buying_stage}}'), ''), 'unknown') AS buying_stage,
            COUNT(*)::int AS review_count
        FROM b2b_reviews
        WHERE duplicate_of_review_id IS NULL
          AND enrichment_status = 'enriched'
        {WINDOW_FILTER}
        GROUP BY 1
        ORDER BY review_count DESC, buying_stage ASC
        """,
        days,
    )
    return [dict(r) for r in rows]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=None, help="Limit audit to reviews in the last N days")
    parser.add_argument("--top-sources", type=int, default=15, help="How many source rows to print")
    parser.add_argument(
        "--min-enriched",
        type=int,
        default=25,
        help="Minimum enriched reviews required for source-level ranking output",
    )
    args = parser.parse_args()

    await init_database()
    try:
        scalars = await _fetch_scalar_report(args.days)
        source_rows = await _fetch_source_report(args.days, args.min_enriched)
        buying_stages = await _fetch_buying_stage_report(args.days)
    finally:
        await close_database()

    identity_ranked = sorted(
        source_rows,
        key=lambda row: (
            float(row.get("enrichment_company_rate") or 0.0),
            float(row.get("company_rate") or 0.0),
            int(row.get("enriched_reviews") or 0),
        ),
        reverse=True,
    )
    actionable_ranked = sorted(
        source_rows,
        key=lambda row: (
            float(row.get("actionable_context_rate") or 0.0),
            int(row.get("enriched_reviews") or 0),
        ),
        reverse=True,
    )

    print(json.dumps({
        "window_days": args.days,
        "summary": scalars,
        "buying_stages": buying_stages,
        "top_sources": source_rows[: args.top_sources],
        "top_identity_yield_sources": identity_ranked[: args.top_sources],
        "top_actionable_context_sources": actionable_ranked[: args.top_sources],
    }, indent=2, sort_keys=False))


if __name__ == "__main__":
    asyncio.run(main())
