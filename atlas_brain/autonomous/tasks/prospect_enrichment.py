"""
Autonomous task: prospect enrichment via Apollo.io.

Runs daily at 8 PM (before campaign generation at 10 PM).
Discovers companies from two sources:
  1. Proactive: high-urgency companies from b2b_reviews
  2. Reactive: campaign_sequences with NULL recipient_email

For each company: org enrich -> people search -> bulk enrich -> upsert prospects.
Respects credit budget (max_credits_per_run) and org cache TTL (org_cache_days).
"""

import json
import logging
import re
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.prospect_enrichment")

# Legal suffixes to strip for normalization
_LEGAL_SUFFIXES = re.compile(
    r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|plc|gmbh|ag|sa|srl|pty|nv|bv)\b\.?",
    re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r"\s+")


_TRAILING_PUNCT = re.compile(r"[,.\-;:]+$")


def _normalize_company(name: str) -> str:
    """Lowercase, strip legal suffixes, collapse whitespace, strip trailing punctuation."""
    n = name.lower().strip()
    n = _LEGAL_SUFFIXES.sub("", n)
    n = _MULTI_SPACE.sub(" ", n).strip()
    n = _TRAILING_PUNCT.sub("", n).strip()
    return n


async def _discover_companies(pool, cfg) -> list[dict[str, str]]:
    """Find companies that need prospect enrichment.

    Returns list of {"raw": original_name, "norm": normalized_name}.
    """
    companies: dict[str, str] = {}  # norm -> raw

    # 1. Proactive: high-urgency from enriched b2b_reviews
    rows = await pool.fetch(
        """
        SELECT DISTINCT reviewer_company
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (enrichment->>'urgency_score')::numeric >= $1
          AND reviewer_company IS NOT NULL
          AND reviewer_company != ''
          AND LOWER(TRIM(reviewer_company)) NOT IN (
              SELECT company_name_norm FROM prospect_org_cache
              WHERE status = 'enriched'
                AND enriched_at > NOW() - make_interval(days => $2)
          )
        LIMIT 100
        """,
        cfg.min_urgency_score,
        cfg.org_cache_days,
    )
    for r in rows:
        raw = r["reviewer_company"]
        norm = _normalize_company(raw)
        if norm and norm not in companies:
            companies[norm] = raw

    # 2. Reactive: sequences with NULL recipient_email
    seq_rows = await pool.fetch(
        """
        SELECT DISTINCT cs.company_name
        FROM campaign_sequences cs
        WHERE cs.status = 'active'
          AND cs.recipient_email IS NULL
          AND LOWER(TRIM(cs.company_name)) NOT IN (
              SELECT company_name_norm FROM prospect_org_cache
              WHERE status IN ('enriched', 'not_found')
                AND enriched_at > NOW() - make_interval(days => $1)
          )
        LIMIT 50
        """,
        cfg.org_cache_days,
    )
    for r in seq_rows:
        raw = r["company_name"]
        norm = _normalize_company(raw)
        if norm and norm not in companies:
            companies[norm] = raw

    return [{"raw": raw, "norm": norm} for norm, raw in companies.items()]


async def _enrich_company(pool, apollo, cfg, company: dict[str, str], credits_used: int) -> tuple[int, dict[str, Any]]:
    """Enrich a single company. Returns (credits_spent, stats_dict).

    Flow: search people by company name (FREE) -> reveal top N (1 credit each).
    Org data comes back in the reveal response, cached for future runs.
    """
    from .campaign_suppression import is_suppressed

    norm = company["norm"]
    raw = company["raw"]
    credits = 0
    stats: dict[str, Any] = {"company": raw, "prospects_created": 0}

    # Check org cache — skip if recently processed (even if no prospects found)
    cached = await pool.fetchrow(
        "SELECT id, status, enriched_at FROM prospect_org_cache WHERE company_name_norm = $1",
        norm,
    )
    org_cache_id = cached["id"] if cached else None

    if cached and cached["status"] == "not_found":
        stats["skipped"] = "previously_not_found"
        return credits, stats

    # People search by company name (FREE — no credits)
    people = await apollo.search_people(company_name=raw, seniorities=cfg.target_seniorities)
    if not people:
        # Widen to manager level and retry once
        people = await apollo.search_people(company_name=raw, seniorities=["manager"])

    if not people:
        stats["people_found"] = 0
        # Cache as not_found to avoid re-searching
        await pool.execute(
            """
            INSERT INTO prospect_org_cache
                (company_name_raw, company_name_norm, status, enriched_at, updated_at)
            VALUES ($1, $2, 'not_found', NOW(), NOW())
            ON CONFLICT (company_name_norm) DO UPDATE SET
                status = 'not_found', enriched_at = NOW(), updated_at = NOW()
            """,
            raw, norm,
        )
        return credits, stats

    # Filter to people with email available
    with_email = [p for p in people if p.has_email]
    stats["people_found"] = len(people)
    stats["people_with_email"] = len(with_email)

    if not with_email:
        return credits, stats

    # Pick top N for reveal (1 credit each)
    max_reveal = min(cfg.max_prospects_per_company, len(with_email))
    remaining_budget = cfg.max_credits_per_run - (credits_used + credits)
    max_reveal = min(max_reveal, remaining_budget)

    if max_reveal <= 0:
        stats["skipped"] = "budget_exhausted"
        return credits, stats

    accepted = cfg.accepted_email_statuses

    for stub in with_email[:max_reveal]:
        person = await apollo.reveal_person(stub.apollo_person_id)
        credits += 1

        if not person or not person.email:
            continue
        if person.email_status not in accepted:
            continue

        # Skip suppressed emails
        suppression = await is_suppressed(pool, email=person.email)
        if suppression:
            logger.debug("Skipping suppressed email: %s", person.email)
            continue

        # Upsert org cache from reveal response (free enrichment data)
        if not org_cache_id and person.company_domain:
            org_cache_id = await pool.fetchval(
                """
                INSERT INTO prospect_org_cache
                    (company_name_raw, company_name_norm, domain,
                     status, enriched_at, updated_at)
                VALUES ($1, $2, $3, 'enriched', NOW(), NOW())
                ON CONFLICT (company_name_norm) DO UPDATE SET
                    domain = COALESCE(EXCLUDED.domain, prospect_org_cache.domain),
                    status = 'enriched',
                    enriched_at = NOW(),
                    updated_at = NOW()
                RETURNING id
                """,
                raw, norm, person.company_domain,
            )

        await pool.execute(
            """
            INSERT INTO prospects
                (apollo_person_id, first_name, last_name, email, email_status,
                 title, seniority, department, linkedin_url,
                 city, state, country,
                 company_name, company_domain, company_name_norm,
                 org_cache_id, status, raw_person_response, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, 'active', $17::jsonb, NOW())
            ON CONFLICT (apollo_person_id) WHERE apollo_person_id IS NOT NULL
            DO UPDATE SET
                email = COALESCE(EXCLUDED.email, prospects.email),
                email_status = COALESCE(EXCLUDED.email_status, prospects.email_status),
                title = COALESCE(EXCLUDED.title, prospects.title),
                seniority = COALESCE(EXCLUDED.seniority, prospects.seniority),
                raw_person_response = EXCLUDED.raw_person_response,
                updated_at = NOW()
            """,
            person.apollo_person_id, person.first_name, person.last_name,
            person.email, person.email_status,
            person.title, person.seniority, person.department, person.linkedin_url,
            person.city, person.state, person.country,
            person.company_name, person.company_domain, norm,
            org_cache_id, json.dumps(person.raw),
        )
        stats["prospects_created"] += 1

    return credits, stats


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: enrich companies with Apollo.io prospects."""
    cfg = settings.apollo
    if not cfg.enabled:
        return {"_skip_synthesis": "Apollo integration disabled"}
    if not cfg.api_key:
        return {"_skip_synthesis": "Apollo API key not configured"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    from ...services.apollo_provider import get_apollo_provider
    apollo = get_apollo_provider()

    companies = await _discover_companies(pool, cfg)
    if not companies:
        return {"_skip_synthesis": "No companies need prospect enrichment"}

    total_credits = 0
    total_prospects = 0
    enriched_companies = 0
    errors = 0
    results = []

    for company in companies:
        if total_credits >= cfg.max_credits_per_run:
            logger.warning("Credit budget exhausted (%d/%d), stopping", total_credits, cfg.max_credits_per_run)
            break

        try:
            spent, stats = await _enrich_company(pool, apollo, cfg, company, total_credits)
            total_credits += spent
            if stats.get("org_found"):
                enriched_companies += 1
            total_prospects += stats.get("prospects_created", 0)
            results.append(stats)
        except Exception:
            logger.exception("Error enriching %s", company["raw"])
            errors += 1

    return {
        "companies_discovered": len(companies),
        "companies_enriched": enriched_companies,
        "prospects_created": total_prospects,
        "credits_used": total_credits,
        "credit_budget": cfg.max_credits_per_run,
        "errors": errors,
        "details": results[:10],  # cap detail output
    }
