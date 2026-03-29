#!/usr/bin/env python3
"""Backfill prospect_org_cache gaps using Apollo org search + enrichment.

Finds orgs with missing domain/phone/industry/employees, searches Apollo
by company name to find the domain, then enriches to fill gaps.

Usage:
    python scripts/backfill_org_cache_from_apollo.py --dry-run
    python scripts/backfill_org_cache_from_apollo.py --limit 50
    python scripts/backfill_org_cache_from_apollo.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_org_cache")

_JUNK_PATTERNS = [
    re.compile(r"^\d+ employees$", re.I),
    re.compile(r"^(a |the |our |my )", re.I),
    re.compile(r"(company|organization|agency|firm|client|facility|corporation)$", re.I),
    re.compile(r"^(large|small|mid|big|fortune|series|top)\b", re.I),
    re.compile(r"^(B2B|B2C|SaaS|BPO|LTC|FMCG|IT)\b", re.I),
    re.compile(r"(based|sector|industry|system|department)$", re.I),
    re.compile(r"^(county|community|local|restaurant|nightclub|bookstore|charity)", re.I),
    re.compile(r"^(var/partner|UK LTDs|WV agencies|Puerto Rico|Costa Rica)", re.I),
]


def _is_junk_name(name: str) -> bool:
    if len(name.strip()) < 4:
        return True
    for pat in _JUNK_PATTERNS:
        if pat.search(name):
            return True
    return False


async def _run(args):
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.services.apollo_provider import get_apollo_provider

    pool = get_db_pool()
    if not pool.is_initialized:
        await pool.initialize()

    apollo = get_apollo_provider()

    rows = await pool.fetch("""
        SELECT id, company_name_raw, domain FROM prospect_org_cache
        WHERE (domain IS NULL OR domain = '')
          AND (COALESCE(phone, '') = '' OR COALESCE(industry, '') = '' OR employee_count IS NULL)
          AND company_name_raw IS NOT NULL AND company_name_raw != ''
        ORDER BY company_name_raw
    """)

    candidates = [(r["id"], r["company_name_raw"]) for r in rows if not _is_junk_name(r["company_name_raw"])]
    logger.info("Found %d searchable orgs (filtered %d junk)", len(candidates), len(rows) - len(candidates))

    if args.limit:
        candidates = candidates[:args.limit]
        logger.info("Limited to %d", args.limit)

    if args.dry_run:
        for _, name in candidates:
            logger.info("  Would search: %s", name)
        logger.info("DRY RUN: %d orgs would be searched", len(candidates))
        return

    searched = 0
    enriched = 0
    updated = 0
    credits_used = 0
    errors = 0

    for org_id, name in candidates:
        try:
            # Search Apollo for company by name
            result = await apollo.search_organizations(name=name, per_page=1)
            credits_used += 1
            searched += 1

            orgs = result.get("organizations") or []
            if not orgs:
                continue

            org = orgs[0]
            domain = org.get("primary_domain") or ""
            phone = org.get("sanitized_phone") or org.get("phone") or ""
            industry = org.get("industry") or ""
            emp = org.get("estimated_num_employees")
            city = org.get("city") or ""
            state = org.get("state") or ""
            country = org.get("country") or ""
            apollo_org_id = org.get("id") or ""

            await pool.execute("""
                UPDATE prospect_org_cache SET
                    domain = CASE WHEN COALESCE(domain, '') = '' THEN $2 ELSE domain END,
                    phone = CASE WHEN COALESCE(phone, '') = '' AND $3 != '' THEN $3 ELSE phone END,
                    sanitized_phone = CASE WHEN COALESCE(sanitized_phone, '') = '' AND $3 != '' THEN $3 ELSE sanitized_phone END,
                    industry = CASE WHEN COALESCE(industry, '') = '' AND $4 != '' THEN $4 ELSE industry END,
                    employee_count = CASE WHEN employee_count IS NULL THEN $5 ELSE employee_count END,
                    city = CASE WHEN COALESCE(city, '') = '' AND $6 != '' THEN $6 ELSE city END,
                    state = CASE WHEN COALESCE(state, '') = '' AND $7 != '' THEN $7 ELSE state END,
                    country = CASE WHEN COALESCE(country, '') = '' AND $8 != '' THEN $8 ELSE country END,
                    apollo_org_id = CASE WHEN COALESCE(apollo_org_id, '') = '' THEN $9 ELSE apollo_org_id END,
                    updated_at = NOW()
                WHERE id = $1
            """, org_id, domain, phone, industry, emp, city, state, country, apollo_org_id)
            updated += 1
            enriched += 1

            if searched % 25 == 0:
                logger.info("Progress: %d/%d searched, %d updated, %d credits", searched, len(candidates), updated, credits_used)

        except Exception as exc:
            logger.warning("Error searching %s: %s", name, exc)
            errors += 1

        # Rate limit
        await asyncio.sleep(0.5)

    # Final stats
    no_phone = await pool.fetchval("SELECT count(*) FROM prospect_org_cache WHERE COALESCE(phone, '') = ''")
    no_industry = await pool.fetchval("SELECT count(*) FROM prospect_org_cache WHERE COALESCE(industry, '') = ''")
    no_emp = await pool.fetchval("SELECT count(*) FROM prospect_org_cache WHERE employee_count IS NULL")

    logger.info("Done: searched=%d, enriched=%d, updated=%d, errors=%d, credits=%d",
                searched, enriched, updated, errors, credits_used)
    logger.info("Remaining gaps: phone=%d, industry=%d, employees=%d", no_phone, no_industry, no_emp)


def main():
    parser = argparse.ArgumentParser(description="Backfill org cache from Apollo")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
