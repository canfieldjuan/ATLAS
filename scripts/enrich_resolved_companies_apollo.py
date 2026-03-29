#!/usr/bin/env python3
"""Enrich newly resolved reviewer companies via Apollo org search.

Finds company names from b2b_account_resolution that don't yet have a row
in prospect_org_cache, searches Apollo for each, and inserts cache rows.

Usage:
    python scripts/enrich_resolved_companies_apollo.py --dry-run
    python scripts/enrich_resolved_companies_apollo.py --limit 50
    python scripts/enrich_resolved_companies_apollo.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("enrich_resolved_companies")

# Patterns that identify names that won't return useful Apollo results
_JUNK_PATTERNS = [
    re.compile(r"^\d", re.I),                                          # starts with digit
    re.compile(r"^(a |an |the |our |my )", re.I),                      # articles
    re.compile(r"\b(company|organization|agency|firm|startup|corp)\s*$", re.I),
    re.compile(r"^(large|small|mid|big|fortune|series|top|b2b|b2c|saas|var)\b", re.I),
    re.compile(r"\b(based|sector|industry|department|division)\s*$", re.I),
    re.compile(r"[.!?]$"),                                              # sentence punctuation
    re.compile(r"\b(is|are|was|were|has|had|have|uses|using|use)\b", re.I),  # verb → fragment
    re.compile(r"^(currently|recently|earlier|mostly|mainly)\b", re.I),
]


def _is_junk(name: str) -> bool:
    if not name or len(name.strip()) < 4:
        return True
    for pat in _JUNK_PATTERNS:
        if pat.search(name.strip()):
            return True
    return False


async def _run(args):
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.services.apollo_provider import get_apollo_provider

    pool = get_db_pool()
    if not pool.is_initialized:
        await pool.initialize()

    apollo = get_apollo_provider()

    # Companies resolved but not yet in cache
    rows = await pool.fetch("""
        SELECT DISTINCT ON (ar.normalized_company_name)
               ar.resolved_company_name,
               ar.normalized_company_name,
               ar.confidence_label,
               COUNT(*) OVER (PARTITION BY ar.normalized_company_name) AS review_count
        FROM b2b_account_resolution ar
        WHERE ar.resolution_status = 'resolved'
          AND ar.normalized_company_name IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM prospect_org_cache poc
              WHERE poc.company_name_norm = ar.normalized_company_name
          )
        ORDER BY ar.normalized_company_name, ar.confidence_label DESC
    """)

    candidates = [r for r in rows if not _is_junk(r["resolved_company_name"])]
    junk_count = len(rows) - len(candidates)

    # Sort: high confidence + most reviews first
    label_order = {"high": 0, "medium": 1, "low": 2}
    candidates.sort(key=lambda r: (label_order.get(r["confidence_label"], 9), -r["review_count"]))

    logger.info(
        "Found %d companies to enrich (%d filtered as junk)",
        len(candidates), junk_count,
    )

    if args.limit:
        candidates = candidates[:args.limit]
        logger.info("Limited to %d", args.limit)

    if args.dry_run:
        for r in candidates:
            logger.info("  Would search: %r (%s, %d reviews)",
                        r["resolved_company_name"], r["confidence_label"], r["review_count"])
        logger.info("DRY RUN: %d companies would be searched", len(candidates))
        return

    inserted = 0
    not_found = 0
    errors = 0
    credits_used = 0

    for r in candidates:
        name = r["resolved_company_name"]
        norm = r["normalized_company_name"]
        try:
            result = await apollo._request(
                "POST", "/v1/mixed_companies/search",
                json={"q_organization_name": name, "per_page": 1},
            )
            credits_used += 1

            orgs = (result or {}).get("organizations") or []
            if not orgs:
                # Insert a stub so we don't re-query next run
                await pool.execute("""
                    INSERT INTO prospect_org_cache (
                        id, company_name_raw, company_name_norm,
                        status, error_detail, created_at, updated_at
                    ) VALUES ($1, $2, $3, 'not_found', 'apollo_no_results', NOW(), NOW())
                    ON CONFLICT DO NOTHING
                """, uuid.uuid4(), name, norm)
                not_found += 1
                continue

            org = orgs[0]
            await pool.execute("""
                INSERT INTO prospect_org_cache (
                    id, company_name_raw, company_name_norm,
                    apollo_org_id, domain,
                    industry, employee_count, annual_revenue_range, annual_revenue,
                    city, state, country, founded_year,
                    total_funding, latest_funding_stage,
                    linkedin_url, website_url, short_description,
                    headcount_growth_6m, headcount_growth_12m, headcount_growth_24m,
                    publicly_traded_exchange, publicly_traded_symbol,
                    phone, sanitized_phone,
                    status, enriched_at, created_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9,
                    $10, $11, $12, $13, $14, $15, $16, $17, $18,
                    $19, $20, $21, $22, $23, $24, $25,
                    'enriched', NOW(), NOW(), NOW()
                )
                ON CONFLICT DO NOTHING
            """,
                uuid.uuid4(), name, norm,
                org.get("id") or "",
                org.get("primary_domain") or "",
                org.get("industry") or "",
                org.get("estimated_num_employees"),
                org.get("annual_revenue_range") or "",
                org.get("annual_revenue"),
                org.get("city") or "",
                org.get("state") or "",
                org.get("country") or "",
                org.get("founded_year"),
                org.get("total_funding") or "",
                org.get("latest_funding_stage") or "",
                org.get("linkedin_url") or "",
                org.get("website_url") or "",
                org.get("short_description") or "",
                org.get("headcount_6_month_growth"),
                org.get("headcount_12_month_growth"),
                org.get("headcount_24_month_growth"),
                org.get("publicly_traded_exchange") or "",
                org.get("publicly_traded_symbol") or "",
                org.get("phone") or "",
                org.get("sanitized_phone") or "",
            )
            inserted += 1
            logger.info(
                "[%d/%d] %r → %s (%s, %s)",
                credits_used, len(candidates), name,
                org.get("primary_domain") or "no domain",
                org.get("industry") or "unknown industry",
                org.get("country") or "?",
            )

            if credits_used % 25 == 0:
                logger.info(
                    "Progress: %d/%d — inserted=%d not_found=%d errors=%d credits=%d",
                    credits_used, len(candidates), inserted, not_found, errors, credits_used,
                )

        except Exception as exc:
            logger.warning("Error searching %r: %s", name, exc)
            errors += 1

        await asyncio.sleep(0.4)

    logger.info(
        "Done: inserted=%d not_found=%d errors=%d credits_used=%d",
        inserted, not_found, errors, credits_used,
    )


def main():
    parser = argparse.ArgumentParser(description="Enrich resolved reviewer companies via Apollo")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be searched, no DB writes")
    parser.add_argument("--limit", type=int, default=0, help="Max companies to process (0 = all)")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
