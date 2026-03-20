"""
Autonomous task: enrich vendor/challenger targets with Apollo.io contacts.

Runs daily at 19:30 (before prospect_enrichment at 20:00, before campaign
generation at 22:00).  Discovers vendor_targets with NULL or stale
contact_email, finds the best-fit person via Apollo, upserts a prospect
record, and wires it back to the vendor target.

Paths B (vendor_retention) and C (challenger_intel) depend on this task
to populate contact_email so b2b_campaign_generation can create campaigns
with a valid recipient.
"""

import json
import logging
import re
from typing import Any

from ...config import settings
from ...services.company_normalization import normalize_company_name as _normalize_company
from ...services.vendor_target_selection import dedupe_vendor_target_rows
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.vendor_target_enrichment")


# ---------------------------------------------------------------------------
# Title relevance scoring by target mode
# ---------------------------------------------------------------------------

_VENDOR_RETENTION_TITLES: list[tuple[re.Pattern, int]] = [
    (re.compile(r"vp.*customer\s*success|vice\s*president.*customer\s*success", re.I), 100),
    (re.compile(r"head.*customer\s*success|director.*customer\s*success", re.I), 90),
    (re.compile(r"head.*product|vp.*product|vice\s*president.*product", re.I), 80),
    (re.compile(r"\bcoo\b|chief\s*operating", re.I), 70),
    (re.compile(r"\bcto\b|chief\s*technology", re.I), 60),
    (re.compile(r"\bceo\b|chief\s*executive|founder|co-founder", re.I), 50),
    (re.compile(r"director|head", re.I), 30),
    (re.compile(r"vp|vice\s*president", re.I), 25),
    (re.compile(r"manager", re.I), 10),
]

_CHALLENGER_INTEL_TITLES: list[tuple[re.Pattern, int]] = [
    (re.compile(r"vp.*sales|vice\s*president.*sales", re.I), 100),
    (re.compile(r"\bcro\b|chief\s*revenue", re.I), 100),
    (re.compile(r"vp.*business\s*dev|vice\s*president.*business\s*dev", re.I), 95),
    (re.compile(r"head.*sales|director.*sales", re.I), 90),
    (re.compile(r"vp.*marketing|vice\s*president.*marketing|head.*marketing", re.I), 80),
    (re.compile(r"\bcmo\b|chief\s*marketing", re.I), 75),
    (re.compile(r"\bceo\b|chief\s*executive|founder|co-founder", re.I), 50),
    (re.compile(r"director|head", re.I), 30),
    (re.compile(r"vp|vice\s*president", re.I), 25),
    (re.compile(r"manager", re.I), 10),
]

_SENIORITY_RANK = {
    "c_suite": 6, "owner": 5, "founder": 5, "vp": 4, "head": 3,
    "director": 3, "manager": 2, "senior": 1,
}


def _title_relevance_score(title: str, target_mode: str) -> int:
    """Score a job title for relevance to the target mode."""
    if not title:
        return 0
    patterns = _VENDOR_RETENTION_TITLES if target_mode == "vendor_retention" else _CHALLENGER_INTEL_TITLES
    for pattern, score in patterns:
        if pattern.search(title):
            return score
    return 0


# ---------------------------------------------------------------------------
# Discovery: find vendor_targets that need contact enrichment
# ---------------------------------------------------------------------------

async def _discover_targets(pool, cfg) -> list[dict]:
    """Find vendor_targets with NULL or stale contact_email.

    Skips manually-set contacts (contact_email set but contact_enriched_at IS NULL).
    """
    rows = await pool.fetch(
        """
        SELECT id, company_name, target_mode, prospect_id, contact_email,
               contact_enriched_at, account_id, created_at, updated_at
        FROM vendor_targets
        WHERE status = 'active'
          AND (
            contact_email IS NULL
            OR (
              contact_enriched_at IS NOT NULL
              AND (
                contact_enriched_at < NOW() - make_interval(days => $1)
                OR prospect_id IN (
                  SELECT id FROM prospects WHERE status IN ('bounced', 'suppressed')
                )
              )
            )
          )
        ORDER BY CASE WHEN contact_email IS NULL THEN 0 ELSE 1 END, created_at ASC
        """,
        cfg.org_cache_days,
    )
    deduped = dedupe_vendor_target_rows(rows)
    return deduped[: cfg.max_vendor_credits_per_run]


# ---------------------------------------------------------------------------
# Enrich a single vendor target
# ---------------------------------------------------------------------------

async def _enrich_target(pool, apollo, cfg, target: dict, credits_used: int) -> tuple[int, dict[str, Any]]:
    """Enrich a single vendor target with Apollo contact data.

    Returns (credits_spent, stats_dict).
    """
    from .campaign_suppression import is_suppressed

    company = target["company_name"]
    target_mode = target["target_mode"]
    norm = _normalize_company(company)
    credits = 0
    stats: dict[str, Any] = {"company": company, "target_mode": target_mode}

    # 1. Check if we already have a usable prospect in the DB for this company
    accepted = cfg.accepted_email_statuses
    existing = await pool.fetch(
        """
        SELECT id, first_name, last_name, email, email_status, title, seniority
        FROM prospects
        WHERE company_name_norm = $1
          AND status = 'active'
          AND email IS NOT NULL
          AND email_status = ANY($2)
        ORDER BY updated_at DESC
        LIMIT 10
        """,
        norm, accepted,
    )
    if existing:
        best = _pick_best_prospect([dict(r) for r in existing], target_mode)
        if best:
            suppression = await is_suppressed(pool, email=best["email"])
            if not suppression:
                await _update_vendor_target(pool, target["id"], best)
                stats["source"] = "existing_prospect"
                stats["contact"] = best["email"]
                return 0, stats

    # 2. Check org cache before Apollo spend.
    cached = await pool.fetchrow(
        "SELECT id, status, enriched_at FROM prospect_org_cache WHERE company_name_norm = $1",
        norm,
    )
    if cached and cached["status"] == "manual_review" and "vendor_target" in set(cfg.manual_review_block_sources or []):
        stats["skipped"] = "manual_review_queued"
        return 0, stats
    if cached and cached["status"] == "not_found":
        stats["skipped"] = "org_not_found_cached"
        return 0, stats

    org_cache_id = cached["id"] if cached else None

    # 3. Apollo search_people (FREE)
    people = await apollo.search_people(company_name=company, seniorities=cfg.target_seniorities)
    if not people:
        people = await apollo.search_people(company_name=company, seniorities=["senior", "entry"])

    if not people:
        stats["people_found"] = 0
        await pool.execute(
            """
            INSERT INTO prospect_org_cache
                (company_name_raw, company_name_norm, status, enriched_at, updated_at)
            VALUES ($1, $2, 'not_found', NOW(), NOW())
            ON CONFLICT (company_name_norm) DO UPDATE SET
                status = 'not_found', enriched_at = NOW(), updated_at = NOW()
            """,
            company, norm,
        )
        return 0, stats

    with_email = [p for p in people if p.has_email]
    stats["people_found"] = len(people)
    stats["people_with_email"] = len(with_email)

    if not with_email:
        return 0, stats

    # Sort stubs by title relevance so we reveal the best candidates first
    scored_stubs = sorted(
        with_email,
        key=lambda s: _title_relevance_score(s.title, target_mode),
        reverse=True,
    )

    # 4. Reveal candidates (1 credit each) until we find a usable one
    best_person = None
    best_score = -1

    remaining_budget = cfg.max_vendor_credits_per_run - credits_used
    max_reveals = min(cfg.max_prospects_per_company, len(scored_stubs), remaining_budget)
    if max_reveals <= 0:
        stats["skipped"] = "budget_exhausted"
        return 0, stats

    for stub in scored_stubs[:max_reveals]:
        person = await apollo.reveal_person(stub.apollo_person_id)
        credits += 1

        if not person or not person.email:
            continue
        if person.email_status not in accepted:
            continue

        suppression = await is_suppressed(pool, email=person.email)
        if suppression:
            logger.debug("Skipping suppressed email: %s", person.email)
            continue

        # Upsert org cache
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
                company, norm, person.company_domain,
            )

        # Upsert prospect
        prospect_id = await pool.fetchval(
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
            RETURNING id
            """,
            person.apollo_person_id, person.first_name, person.last_name,
            person.email, person.email_status,
            person.title, person.seniority, person.department, person.linkedin_url,
            person.city, person.state, person.country,
            person.company_name, person.company_domain, norm,
            org_cache_id, json.dumps(person.raw),
        )

        title_score = _title_relevance_score(person.title, target_mode)
        seniority_score = _SENIORITY_RANK.get(person.seniority, 0)
        combined = title_score * 10 + seniority_score

        if combined > best_score:
            best_score = combined
            best_person = {
                "prospect_id": prospect_id,
                "first_name": person.first_name,
                "last_name": person.last_name,
                "email": person.email,
                "title": person.title,
            }

    # 5. Update vendor target with best match
    if best_person:
        await _update_vendor_target(pool, target["id"], best_person)
        stats["contact"] = best_person["email"]
        stats["source"] = "apollo_reveal"
    else:
        stats["no_usable_contact"] = True

    return credits, stats


def _pick_best_prospect(prospects: list[dict], target_mode: str) -> dict | None:
    """Pick the best prospect from existing DB rows by title + seniority."""
    best = None
    best_score = -1
    for p in prospects:
        title_score = _title_relevance_score(p.get("title", ""), target_mode)
        seniority_score = _SENIORITY_RANK.get(p.get("seniority", ""), 0)
        combined = title_score * 10 + seniority_score
        if combined > best_score:
            best_score = combined
            best = p
    return best if best_score > 0 else (prospects[0] if prospects else None)


async def _update_vendor_target(pool, target_id, prospect: dict) -> None:
    """Write enriched contact info back to vendor_targets."""
    first = prospect.get("first_name", "")
    last = prospect.get("last_name", "")
    name = f"{first} {last}".strip() if first or last else None

    await pool.execute(
        """
        UPDATE vendor_targets
        SET contact_name = $2,
            contact_email = $3,
            contact_role = $4,
            prospect_id = $5,
            contact_enriched_at = NOW(),
            updated_at = NOW()
        WHERE id = $1
        """,
        target_id,
        name,
        prospect.get("email"),
        prospect.get("title"),
        prospect.get("prospect_id") or prospect.get("id"),
    )


# ---------------------------------------------------------------------------
# Task handler
# ---------------------------------------------------------------------------

async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: enrich vendor/challenger targets with Apollo.io contacts."""
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

    targets = await _discover_targets(pool, cfg)
    if not targets:
        return {"_skip_synthesis": "No vendor targets need contact enrichment"}

    total_credits = 0
    enriched = 0
    errors = 0
    results = []

    for target in targets:
        if total_credits >= cfg.max_vendor_credits_per_run:
            logger.warning(
                "Vendor credit budget exhausted (%d/%d), stopping",
                total_credits, cfg.max_vendor_credits_per_run,
            )
            break

        try:
            spent, stats = await _enrich_target(pool, apollo, cfg, target, total_credits)
            total_credits += spent
            if stats.get("contact"):
                enriched += 1
            results.append(stats)
        except Exception:
            logger.exception("Error enriching vendor target %s", target["company_name"])
            errors += 1

    return {
        "targets_discovered": len(targets),
        "targets_enriched": enriched,
        "credits_used": total_credits,
        "credit_budget": cfg.max_vendor_credits_per_run,
        "errors": errors,
        "details": results[:10],
    }
