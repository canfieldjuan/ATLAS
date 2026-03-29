#!/usr/bin/env python3
"""Backfill missing firmographic fields in prospect_org_cache via Apollo org enrich.

Uses orgs that already have a domain but are missing short_description,
founded_year, total_funding, latest_funding_stage, or headcount_growth.

Calls GET /api/v1/organizations/enrich?domain=... (1 credit per org).

Usage:
    python scripts/backfill_firmographics_from_apollo.py --dry-run
    python scripts/backfill_firmographics_from_apollo.py --limit 50
    python scripts/backfill_firmographics_from_apollo.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_firmographics")


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


async def _run(args):
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.services.apollo_provider import get_apollo_provider

    pool = get_db_pool()
    if not pool.is_initialized:
        await pool.initialize()

    apollo = get_apollo_provider()

    rows = await pool.fetch("""
        SELECT id, domain, company_name_raw
        FROM prospect_org_cache
        WHERE domain IS NOT NULL AND domain != ''
          AND (
            short_description IS NULL
            OR founded_year IS NULL
            OR total_funding IS NULL
            OR headcount_growth_6m IS NULL
          )
        ORDER BY company_name_raw
    """)

    logger.info("Found %d orgs needing firmographic enrichment", len(rows))

    if args.limit:
        rows = rows[:args.limit]
        logger.info("Limited to %d", args.limit)

    if args.dry_run:
        for r in rows:
            logger.info("  Would enrich: %s (%s)", r["company_name_raw"], r["domain"])
        logger.info("DRY RUN: %d orgs would use %d credits", len(rows), len(rows))
        return

    enriched = 0
    skipped = 0
    errors = 0
    credits_used = 0

    for row in rows:
        org_id = row["id"]
        domain = row["domain"]
        name = row["company_name_raw"]

        try:
            data = await apollo._request("GET", "/api/v1/organizations/enrich", params={"domain": domain})
            credits_used += 1

            if not data:
                logger.debug("No data for %s (%s)", name, domain)
                skipped += 1
                continue

            org = data.get("organization") or {}
            if not org:
                logger.debug("Empty org for %s (%s)", name, domain)
                skipped += 1
                continue

            hg6 = _safe_float(org.get("organization_headcount_six_month_growth"))
            hg12 = _safe_float(org.get("organization_headcount_twelve_month_growth"))
            hg24 = _safe_float(org.get("organization_headcount_twenty_four_month_growth"))
            total_funding = str(org.get("total_funding_printed") or org.get("total_funding") or "") or None
            short_desc = org.get("short_description") or None

            await pool.execute("""
                UPDATE prospect_org_cache SET
                    short_description = CASE
                        WHEN short_description IS NULL AND $2::text IS NOT NULL THEN $2::text
                        ELSE short_description END,
                    founded_year = CASE
                        WHEN founded_year IS NULL THEN $3::smallint
                        ELSE founded_year END,
                    total_funding = CASE
                        WHEN total_funding IS NULL AND $4::text IS NOT NULL THEN $4::text
                        ELSE total_funding END,
                    latest_funding_stage = CASE
                        WHEN latest_funding_stage IS NULL AND $5::text IS NOT NULL THEN $5::text
                        ELSE latest_funding_stage END,
                    headcount_growth_6m = CASE
                        WHEN headcount_growth_6m IS NULL AND $6::double precision IS NOT NULL THEN $6::double precision
                        ELSE headcount_growth_6m END,
                    headcount_growth_12m = CASE
                        WHEN headcount_growth_12m IS NULL AND $7::double precision IS NOT NULL THEN $7::double precision
                        ELSE headcount_growth_12m END,
                    headcount_growth_24m = CASE
                        WHEN headcount_growth_24m IS NULL AND $8::double precision IS NOT NULL THEN $8::double precision
                        ELSE headcount_growth_24m END,
                    publicly_traded_exchange = CASE
                        WHEN publicly_traded_exchange IS NULL AND $9::text IS NOT NULL THEN $9::text
                        ELSE publicly_traded_exchange END,
                    publicly_traded_symbol = CASE
                        WHEN publicly_traded_symbol IS NULL AND $10::text IS NOT NULL THEN $10::text
                        ELSE publicly_traded_symbol END,
                    updated_at = NOW()
                WHERE id = $1
            """,
                org_id,
                short_desc,
                org.get("founded_year"),
                total_funding,
                org.get("latest_funding_stage") or None,
                hg6, hg12, hg24,
                org.get("publicly_traded_exchange") or None,
                org.get("publicly_traded_symbol") or None,
            )
            enriched += 1
            logger.debug("Enriched %s (%s): desc=%s, founded=%s, funding=%s, hg6m=%s",
                         name, domain, bool(short_desc), org.get("founded_year"), total_funding, hg6)

        except Exception as exc:
            logger.warning("Error enriching %s (%s): %s", name, domain, exc)
            errors += 1

        # Rate limit: ~2 req/s
        await asyncio.sleep(0.5)

        if credits_used % 25 == 0:
            logger.info("Progress: %d/%d credits used, %d enriched, %d skipped, %d errors",
                        credits_used, len(rows), enriched, skipped, errors)

    logger.info("Done: credits=%d, enriched=%d, skipped=%d, errors=%d",
                credits_used, enriched, skipped, errors)

    # Final fill-rate summary
    stats = await pool.fetchrow("""
        SELECT
          COUNT(*) FILTER (WHERE short_description IS NOT NULL) AS has_desc,
          COUNT(*) FILTER (WHERE founded_year IS NOT NULL) AS has_founded,
          COUNT(*) FILTER (WHERE total_funding IS NOT NULL) AS has_funding,
          COUNT(*) FILTER (WHERE headcount_growth_6m IS NOT NULL) AS has_hc_growth,
          COUNT(*) AS total
        FROM prospect_org_cache
    """)
    logger.info("Final fill rates: desc=%d/%d, founded=%d/%d, funding=%d/%d, hc_growth=%d/%d",
                stats["has_desc"], stats["total"],
                stats["has_founded"], stats["total"],
                stats["has_funding"], stats["total"],
                stats["has_hc_growth"], stats["total"])


def main():
    parser = argparse.ArgumentParser(description="Backfill firmographics from Apollo org enrich")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run, no API calls")
    parser.add_argument("--limit", type=int, default=0, help="Max orgs to process (0=all)")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
