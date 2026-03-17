"""
Autonomous task: prospect enrichment via Apollo.io.

Runs daily at 8 PM (before campaign generation at 10 PM).
Discovers companies from three sources:
  1. Proactive: named customer companies showing complaint/churn signals
  2. Proactive: vendors with high churn signal volume from b2b_reviews
  3. Reactive: campaign_sequences with NULL recipient_email

For each company: search_people by name (FREE) -> reveal_person (1 credit each)
-> upsert prospects. Org data comes from person reveal responses, not a
separate enrich_organization call. Respects credit budget
(max_credits_per_run) and org cache TTL (org_cache_days).
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from ...config import settings
from ...services.apollo_company_overrides import fetch_company_override_map
from ...services.company_normalization import normalize_company_name as _normalize_company
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.prospect_enrichment")

# Known vendor -> canonical domain mapping.
# Loaded once from DB at task start; used to validate Apollo search results
# before spending reveal credits.
_vendor_domains: dict[str, str] = {}  # norm vendor name -> domain
_company_override_cache: dict[str, dict[str, Any]] = {}

_GENERIC_COMPANY_PATTERNS = (
    re.compile(r"^(msp|saas|fintech|non-?profit|university|school|fortune 500|startup)$", re.I),
    re.compile(r"^(fortune \d+|government|automation|pharma|recruitment|techfirm|software house|video studio)$", re.I),
    re.compile(r"(^| )(start up|startup)( |$)", re.I),
    re.compile(r"(^| )(software company|erp software company|tech company|tech start up|b2b tech|manufacturer)( |$)", re.I),
    re.compile(r"(^| )(saas provider|saas company|b2b saas|b2b e-?commerce)( |$)", re.I),
    re.compile(r"(^| )(email marketing agency|design agency)( |$)", re.I),
    re.compile(r"(^| )(erp software|security [& ]+business intelligence manufacturer)( |$)", re.I),
    re.compile(r"(^| )(small|mid.?sized|midsized|series [a-z]|b2b|toronto|swiss|finnish|vietnam|germany'?s biggest)( .*)?(software|tech|manufacturer|agency|provider)( |$)", re.I),
    re.compile(r"(^| )\d+\s+employees?( |$)", re.I),
    re.compile(r"(^| )(it\s*-\s*\d+\s+employees?|big 4 firm|consulting company|company i'?m currently employed with)( |$)", re.I),
    re.compile(r"(^| )([a-z]+-based|costa rica)( |$)", re.I),
    re.compile(r"(^| )(drop shipping|education nonprofit|nonprofit|consulting|firm)( |$)", re.I),
)

def _manual_review_block_sources() -> set[str]:
    """Return discovery sources blocked by manual-review cache rows."""
    return set(settings.apollo.manual_review_block_sources or [])


async def _load_vendor_domains(pool) -> None:
    """Build vendor domain lookup from prospect_org_cache + scrape targets."""
    global _vendor_domains
    rows = await pool.fetch(
        """
        SELECT company_name_norm, domain
        FROM prospect_org_cache
        WHERE status = 'enriched' AND domain IS NOT NULL AND domain <> ''
        """,
    )
    _vendor_domains = {r["company_name_norm"]: r["domain"] for r in rows}


async def _load_company_enrichment_overrides(pool) -> None:
    """Cache company overrides for this task run."""
    global _company_override_cache
    _company_override_cache = await fetch_company_override_map(pool)


def _is_generic_company_descriptor(name: str) -> bool:
    """Return True for reviewer-company labels that are descriptors, not orgs."""
    norm = _normalize_company(name)
    if not norm:
        return True
    return any(pattern.search(norm) for pattern in _GENERIC_COMPANY_PATTERNS)


def _company_enrichment_override(name: str) -> dict[str, Any]:
    """Return alias/domain overrides for hard-to-match customer companies."""
    return _company_override_cache.get(_normalize_company(name), {})


def _override_domains(name: str) -> list[str]:
    """Return normalized expected domains for a company override."""
    override = _company_enrichment_override(name)
    raw_domains = override.get("domains") or ([] if not override.get("domain") else [override["domain"]])
    domains: list[str] = []
    for domain in raw_domains:
        d = str(domain).strip().lower().lstrip("www.")
        if d and d not in domains:
            domains.append(d)
    return domains


def _candidate_company_search_names(name: str) -> list[str]:
    """Return deduplicated search names for Apollo company search."""
    raw = name.strip()
    norm = _normalize_company(raw)
    override = _company_enrichment_override(raw)
    candidates: list[str] = []

    def _add(value: str) -> None:
        value = value.strip()
        if not value:
            return
        if value.lower() not in {c.lower() for c in candidates}:
            candidates.append(value)

    for alias in override.get("search_names", []):
        _add(alias)
    _add(raw)

    if "." in raw:
        stripped = re.sub(r"\.(com|co|io|ai|net|org|uk)$", "", raw, flags=re.I)
        stripped = re.sub(r"\.(co|com)$", "", stripped, flags=re.I)
        _add(stripped)
    if " " in raw and not any(ch in raw for ch in ".&"):
        _add("".join(part for part in raw.split()))
    if "&" in raw:
        _add(raw.replace("&", "and"))
    if norm and norm != raw.lower():
        _add(norm)

    return candidates


def _org_matches_vendor(
    org_name: str,
    org_domain: str,
    vendor_name: str,
) -> bool:
    """Check if an Apollo search result actually belongs to the target vendor.

    Uses domain matching (primary) with strict name fallback.
    Prevents wasting reveal credits on resellers, consultancies, and
    similarly-named companies.
    """
    if not org_name:
        return False
    vendor_norm = _normalize_company(vendor_name)

    # 1. Domain check (most reliable)
    known_domains: list[str] = []
    vd = _vendor_domains.get(vendor_norm, "")
    if vd:
        known_domains.append(vd.lower().lstrip("www."))
    for domain in _override_domains(vendor_name):
        if domain not in known_domains:
            known_domains.append(domain)
    if known_domains and org_domain:
        # Strip www. and compare
        od = org_domain.lower().lstrip("www.")
        for kd in known_domains:
            if od == kd or od.endswith(f".{kd}") or kd.endswith(f".{od}"):
                return True

    # 2. No known domain -- exact name match only (zero tolerance for junk).
    #    Once a vendor is enriched with a real domain, future runs use path 1.
    strip = {
        "inc", "llc", "ltd", "corp", "co", "company",
        "the", "a", "an", "and",
    }

    def _clean(n: str) -> list[str]:
        n = re.sub(r"\s*\(.*?\)\s*", " ", n.lower())
        n = re.sub(r"[^\w\s]", " ", n)
        return [t for t in n.split() if t and t not in strip]

    org_tokens = _clean(org_name)
    if not org_tokens:
        return False
    org_joined = "".join(org_tokens)
    for candidate in _candidate_company_search_names(vendor_name):
        vendor_tokens = _clean(candidate)
        if not vendor_tokens:
            continue
        if org_tokens == vendor_tokens or org_joined == "".join(vendor_tokens):
            return True
    return False


async def _route_manual_domain_assist(
    pool,
    *,
    raw: str,
    norm: str,
    reason: str,
    search_attempts: list[dict[str, Any]] | None = None,
) -> None:
    """Queue a hard org-match case for manual/domain-assisted enrichment."""
    payload = {
        "route": "manual_domain_assist",
        "reason": reason,
        "search_names": _candidate_company_search_names(raw),
        "suggested_domains": _override_domains(raw),
        "search_attempts": search_attempts or [],
    }
    await pool.execute(
        """
        INSERT INTO prospect_org_cache
            (company_name_raw, company_name_norm, domain, status, error_detail, enriched_at, updated_at)
        VALUES ($1, $2, NULLIF($3, ''), 'manual_review', $4, NOW(), NOW())
        ON CONFLICT (company_name_norm) DO UPDATE SET
            company_name_raw = EXCLUDED.company_name_raw,
            domain = COALESCE(prospect_org_cache.domain, EXCLUDED.domain),
            status = 'manual_review',
            error_detail = EXCLUDED.error_detail,
            enriched_at = NOW(),
            updated_at = NOW()
        """,
        raw,
        norm,
        (_override_domains(raw) or [""])[0],
        json.dumps(payload),
    )


async def _discover_companies(pool, cfg) -> list[dict[str, str]]:
    """Find companies worth Apollo enrichment.

    Priority order:
    1. Named customer companies with explicit churn/complaint signals
    2. Vendors with high churn signal volume
    3. Active campaign sequences still missing recipients

    Returns list of {"raw": original_name, "norm": normalized_name, "source": source_name}.
    """
    companies: dict[str, str] = {}  # norm -> raw
    company_sources: dict[str, str] = {}
    manual_review_block_sources = set(cfg.manual_review_block_sources or [])
    manual_review_blocked_norms = {
        r["company_name_norm"]
        for r in await pool.fetch(
            """
            SELECT company_name_norm
            FROM prospect_org_cache
            WHERE status = 'manual_review'
            """,
        )
    } if manual_review_block_sources else set()
    covered_company_norms = {
        r["company_name_norm"]
        for r in await pool.fetch(
            """
            SELECT DISTINCT company_name_norm
            FROM prospects
            WHERE company_name_norm IS NOT NULL
              AND company_name_norm != ''
              AND status = 'active'
              AND email IS NOT NULL
              AND TRIM(email) != ''
            """,
        )
    }

    def _add_company(raw: str, source: str) -> None:
        norm = _normalize_company(raw)
        if not norm or norm in companies:
            return
        if norm in covered_company_norms:
            return
        if source in manual_review_block_sources and norm in manual_review_blocked_norms:
            return
        if source == "reviewer_company" and _is_generic_company_descriptor(raw):
            return
        companies[norm] = raw
        company_sources[norm] = source

    # 1. Proactive: named customer companies showing strong complaint signals.
    complaint_rows = await pool.fetch(
        """
        SELECT MIN(reviewer_company) AS reviewer_company,
               reviewer_company_norm,
               COUNT(*) AS signal_count,
               COUNT(*) FILTER (
                   WHERE COALESCE((enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
               ) AS leave_count,
               AVG(COALESCE((enrichment->>'urgency_score')::numeric, 0)) AS avg_urgency
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND reviewer_company_norm IS NOT NULL
          AND reviewer_company_norm != ''
          AND (
              COALESCE((enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
              OR COALESCE((enrichment->>'urgency_score')::numeric, 0) >= $1
          )
          AND reviewer_company_norm NOT IN (
              SELECT company_name_norm FROM prospect_org_cache
              WHERE status IN ('enriched', 'not_found')
                AND enriched_at > NOW() - make_interval(days => $2)
          )
        GROUP BY reviewer_company_norm
        ORDER BY
            COUNT(*) FILTER (
                WHERE COALESCE((enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
            ) DESC,
            COUNT(*) DESC,
            AVG(COALESCE((enrichment->>'urgency_score')::numeric, 0)) DESC
        LIMIT 200
        """,
        cfg.min_urgency_score,
        cfg.org_cache_days,
    )
    for r in complaint_rows:
        _add_company(r["reviewer_company"], "reviewer_company")

    # 2. Proactive: vendors with significant churn signal volume
    rows = await pool.fetch(
        """
        SELECT vendor_name, COUNT(*) AS signal_count
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (enrichment->>'urgency_score')::numeric >= $1
          AND vendor_name IS NOT NULL
          AND vendor_name != ''
          AND LOWER(TRIM(vendor_name)) NOT IN (
              SELECT company_name_norm FROM prospect_org_cache
              WHERE status IN ('enriched', 'not_found')
                AND enriched_at > NOW() - make_interval(days => $2)
          )
        GROUP BY vendor_name
        HAVING COUNT(*) >= $3
        ORDER BY COUNT(*) DESC
        LIMIT 100
        """,
        cfg.min_urgency_score,
        cfg.org_cache_days,
        cfg.min_churn_signals,
    )
    for r in rows:
        _add_company(r["vendor_name"], "vendor_name")

    # 3. Reactive: sequences with NULL recipient_email
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
        _add_company(r["company_name"], "campaign_sequence")

    return [
        {"raw": raw, "norm": norm, "source": company_sources.get(norm, "unknown")}
        for norm, raw in companies.items()
    ]


async def _enrich_company(pool, apollo, cfg, company: dict[str, str], credits_used: int) -> tuple[int, dict[str, Any]]:
    """Enrich a single company. Returns (credits_spent, stats_dict).

    Flow: search people by company name (FREE) -> reveal top N (1 credit each).
    Org data comes back in the reveal response, cached for future runs.
    """
    from .campaign_suppression import is_suppressed

    norm = company["norm"]
    raw = company["raw"]
    credits = 0
    stats: dict[str, Any] = {
        "company": raw,
        "discovery_source": company.get("source", "unknown"),
        "prospects_created": 0,
    }

    # Check org cache -- skip if recently processed (even if no prospects found)
    cached = await pool.fetchrow(
        "SELECT id, status, enriched_at FROM prospect_org_cache WHERE company_name_norm = $1",
        norm,
    )
    org_cache_id = cached["id"] if cached else None
    manual_review_blocks = company.get("source") in _manual_review_block_sources()

    if cached and cached["status"] == "manual_review" and manual_review_blocks:
        stats["skipped"] = "manual_review_queued"
        return credits, stats

    if cached and cached["status"] == "not_found":
        retry_after = datetime.now(timezone.utc) - timedelta(days=cfg.org_cache_days)
        enriched_at = cached["enriched_at"]
        has_override = bool(_company_enrichment_override(raw).get("search_names") or _override_domains(raw))
        if enriched_at and enriched_at.replace(tzinfo=timezone.utc) > retry_after and not has_override:
            stats["skipped"] = "previously_not_found"
            return credits, stats

    # People search by company name (FREE -- no credits). Retry aliases before spending credits.
    search_attempts: list[dict[str, int | str]] = []
    search_names = _candidate_company_search_names(raw)
    people = []
    verified: list[Any] = []
    chosen_search_name = raw
    for seniorities in (cfg.target_seniorities,):
        for search_name in search_names:
            people = await apollo.search_people(company_name=search_name, seniorities=seniorities)
            verified = [
                p for p in people
                if _org_matches_vendor(p.organization_name, p.organization_domain, raw)
            ]
            search_attempts.append({
                "search_name": search_name,
                "people_found": len(people),
                "verified": len(verified),
                "seniority_mode": "primary" if seniorities == cfg.target_seniorities else "fallback",
            })
            if verified:
                chosen_search_name = search_name
                break
        if verified:
            break

    if not people:
        stats["people_found"] = 0
        stats["search_attempts"] = search_attempts[:5]
        if company.get("source") == "reviewer_company":
            await _route_manual_domain_assist(
                pool, raw=raw, norm=norm, reason="no_people_found", search_attempts=search_attempts[:5],
            )
            stats["queued_manual_review"] = True
        else:
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

    rejected = len(people) - len(verified)
    if rejected:
        logger.info(
            "Org filter for %s: %d/%d people rejected (wrong org)",
            raw, rejected, len(people),
        )
    stats["search_name"] = chosen_search_name
    stats["people_found"] = len(people)
    stats["org_rejected"] = rejected
    stats["search_attempts"] = search_attempts[:5]

    # Filter to people with email available
    with_email = [p for p in verified if p.has_email]
    stats["people_with_email"] = len(with_email)

    if not with_email:
        if company.get("source") == "reviewer_company":
            await _route_manual_domain_assist(
                pool, raw=raw, norm=norm, reason="no_verified_email", search_attempts=search_attempts[:5],
            )
            stats["queued_manual_review"] = True
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

        # Upsert org cache from reveal response (free firmographic data)
        if not org_cache_id and person.company_domain:
            org = person.raw.get("organization") or {}
            org_cache_id = await pool.fetchval(
                """
                INSERT INTO prospect_org_cache
                    (company_name_raw, company_name_norm, domain,
                     apollo_org_id, industry, employee_count,
                     annual_revenue_range, tech_stack,
                     status, enriched_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb,
                        'enriched', NOW(), NOW())
                ON CONFLICT (company_name_norm) DO UPDATE SET
                    domain = COALESCE(EXCLUDED.domain, prospect_org_cache.domain),
                    apollo_org_id = COALESCE(EXCLUDED.apollo_org_id, prospect_org_cache.apollo_org_id),
                    industry = COALESCE(EXCLUDED.industry, prospect_org_cache.industry),
                    employee_count = COALESCE(EXCLUDED.employee_count, prospect_org_cache.employee_count),
                    annual_revenue_range = COALESCE(EXCLUDED.annual_revenue_range, prospect_org_cache.annual_revenue_range),
                    tech_stack = CASE WHEN EXCLUDED.tech_stack != '[]'::jsonb
                                      THEN EXCLUDED.tech_stack
                                      ELSE prospect_org_cache.tech_stack END,
                    status = 'enriched',
                    enriched_at = NOW(),
                    updated_at = NOW()
                RETURNING id
                """,
                raw,
                norm,
                person.company_domain,
                org.get("id") or None,
                org.get("industry") or None,
                org.get("estimated_num_employees"),
                org.get("annual_revenue_printed") or None,
                json.dumps([
                    t.get("name", t) if isinstance(t, dict) else str(t)
                    for t in (org.get("current_technologies") or [])
                ]),
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

    if stats["prospects_created"] == 0 and company.get("source") == "reviewer_company":
        await _route_manual_domain_assist(
            pool, raw=raw, norm=norm, reason="reveals_no_usable_contact", search_attempts=search_attempts[:5],
        )
        stats["queued_manual_review"] = True

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

    # Load known vendor domains for org validation (prevents wasting reveal credits)
    await _load_vendor_domains(pool)
    await _load_company_enrichment_overrides(pool)

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
            if stats.get("prospects_created", 0) > 0:
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
