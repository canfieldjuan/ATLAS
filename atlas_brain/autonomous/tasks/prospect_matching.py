"""
Autonomous task: match prospects to unmatched campaign sequences.

Runs hourly. Pure database reads/writes -- no external API calls.
Matches by normalized company name + role relevance based on pain categories.
"""

import json
import logging
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.prospect_matching")

# Pain category -> preferred title keywords (ordered by preference)
_PAIN_TITLE_MAP: dict[str, list[str]] = {
    "pricing": ["cfo", "vp finance", "head of finance", "finance director", "controller"],
    "cost": ["cfo", "vp finance", "head of finance", "finance director"],
    "features": ["cto", "vp product", "vp engineering", "head of product", "product director"],
    "ux": ["vp product", "head of design", "cto", "vp engineering"],
    "usability": ["vp product", "head of product", "cto"],
    "reliability": ["vp engineering", "cto", "vp operations", "head of engineering"],
    "performance": ["vp engineering", "cto", "vp operations"],
    "support": ["vp customer success", "coo", "head of customer success", "vp operations"],
    "service": ["vp customer success", "coo", "vp operations"],
    "integration": ["cto", "vp engineering", "head of it", "vp technology"],
    "security": ["ciso", "cto", "vp engineering", "head of security"],
    "scalability": ["cto", "vp engineering", "coo"],
}

# Persona -> preferred title keywords (takes priority over pain-based matching)
# Includes both abbreviated and full forms to handle title format variations
_PERSONA_TITLE_MAP: dict[str, list[str]] = {
    "executive": [
        "ceo", "cfo", "coo", "cro", "vp finance", "president",
        "chief executive", "chief financial", "chief operating",
        "chief revenue", "vp of finance",
    ],
    "technical": [
        "cto", "vp engineering", "vp product", "head of engineering",
        "head of product", "head of it", "vp technology",
        "chief technology", "vp of engineering", "vp of product",
    ],
    "operations": [
        "vp customer success", "vp operations", "vp sales",
        "head of customer success", "head of operations",
        "director of operations", "director of customer success",
        "vp of customer success", "vp of operations", "vp of sales",
    ],
    "evaluator": [
        "procurement", "purchasing", "sourcing", "vendor management",
        "it manager", "solutions architect", "systems administrator",
        "project manager", "program manager", "business analyst",
        "director of it", "manager of it", "head of procurement",
        "revops", "revenue operations", "sales operations", "sales ops",
        "marketing ops", "marketing operations", "business systems",
        "gtm systems", "go-to-market", "crm admin", "crm manager",
        "salesforce admin", "hubspot admin", "systems manager",
    ],
    "champion": [
        "team lead", "manager", "senior manager", "group manager",
        "department head", "supervisor", "coordinator",
        "account manager", "customer success manager",
        "implementation manager", "adoption manager",
        "crm specialist", "salesforce specialist", "hubspot specialist",
        "sales enablement", "revenue enablement",
    ],
}

# Seniority ranking for tiebreaking
_SENIORITY_RANK: dict[str, int] = {
    "c_suite": 7,
    "owner": 6,
    "founder": 6,
    "vp": 5,
    "head": 4,
    "director": 3,
    "manager": 2,
    "senior": 1,
}


def _extract_pain_categories(context: dict | str | None) -> list[str]:
    """Extract pain category names from company_context JSONB."""
    if not context:
        return []
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(context, dict):
        return []

    # company_context may have pain_categories as list of strings or list of dicts
    cats = context.get("pain_categories") or context.get("pain_points") or []
    result = []
    for cat in cats:
        if isinstance(cat, str):
            result.append(cat.lower())
        elif isinstance(cat, dict):
            name = cat.get("category") or cat.get("name") or ""
            if name:
                result.append(name.lower())
    return result


def _preferred_title_keywords(pain_categories: list[str]) -> list[str]:
    """Get ordered list of preferred title keywords based on pain categories."""
    seen: set[str] = set()
    keywords: list[str] = []
    for cat in pain_categories:
        for key, titles in _PAIN_TITLE_MAP.items():
            if key in cat:
                for t in titles:
                    if t not in seen:
                        seen.add(t)
                        keywords.append(t)
    return keywords


def _score_prospect(
    prospect: dict,
    preferred_keywords: list[str],
    accepted_statuses: list[str],
) -> tuple[int, int]:
    """Score a prospect for matching. Returns (title_score, seniority_score).

    Higher is better. title_score: 100 - keyword index (or 0 if no match).
    seniority_score: from _SENIORITY_RANK.
    """
    title = (prospect.get("title") or "").lower()
    seniority = (prospect.get("seniority") or "").lower()
    email_status = prospect.get("email_status") or ""

    # Must have usable email
    if not prospect.get("email") or email_status not in accepted_statuses:
        return (-1, -1)

    # Title keyword match score
    title_score = 0
    for i, kw in enumerate(preferred_keywords):
        if kw in title:
            title_score = 100 - i
            break

    seniority_score = _SENIORITY_RANK.get(seniority, 0)

    return (title_score, seniority_score)


def _extract_target_persona(context: dict | str | None) -> str | None:
    """Extract target_persona from company_context JSONB."""
    if not context:
        return None
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(context, dict):
        return context.get("target_persona")
    return None


async def _fetch_already_matched_prospect_ids(
    pool, company_name: str,
) -> set[str]:
    """Get prospect IDs already matched to other persona sequences for this company."""
    rows = await pool.fetch(
        """
        SELECT metadata->>'prospect_id' AS pid
        FROM campaign_audit_log
        WHERE event_type = 'prospect_matched'
          AND metadata->>'company' = $1
          AND created_at > NOW() - INTERVAL '30 days'
        """,
        company_name,
    )
    return {r["pid"] for r in rows if r["pid"]}


async def _match_sequence(pool, seq: dict, accepted_statuses: list[str]) -> dict[str, Any] | None:
    """Try to match a single sequence to a prospect. Returns match info or None."""
    company_name = seq["company_name"]
    company_norm = company_name.lower().strip()
    # Also try a stripped version
    from .prospect_enrichment import _normalize_company
    company_norm_stripped = _normalize_company(company_name)

    # Find active prospects for this company
    prospects = await pool.fetch(
        """
        SELECT id, first_name, last_name, email, email_status,
               title, seniority, department
        FROM prospects
        WHERE company_name_norm IN ($1, $2)
          AND status = 'active'
          AND email IS NOT NULL
        ORDER BY created_at DESC
        """,
        company_norm, company_norm_stripped,
    )

    if not prospects:
        return None

    # Extract context for matching
    context_raw = seq.get("company_context")
    target_persona = _extract_target_persona(context_raw)

    # Choose title keywords: persona-based (primary) or pain-based (fallback)
    if target_persona and target_persona in _PERSONA_TITLE_MAP:
        preferred_kw = _PERSONA_TITLE_MAP[target_persona]
    else:
        pain_cats = _extract_pain_categories(context_raw)
        preferred_kw = _preferred_title_keywords(pain_cats)

    # Dedup: deprioritize prospects already matched to other persona sequences
    already_matched = await _fetch_already_matched_prospect_ids(pool, company_name)

    # Score and rank prospects
    scored = []
    for p in prospects:
        title_sc, seniority_sc = _score_prospect(dict(p), preferred_kw, accepted_statuses)
        if title_sc < 0:
            continue  # no usable email
        # Deprioritize already-matched prospects (still allow if no alternative)
        dedup_penalty = -1000 if str(p["id"]) in already_matched else 0
        scored.append((dedup_penalty, title_sc, seniority_sc, p))

    if not scored:
        return None

    # Sort: dedup penalty first, then title match, then seniority
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    best = scored[0][3]

    return {
        "prospect_id": best["id"],
        "email": best["email"],
        "first_name": best["first_name"],
        "last_name": best["last_name"],
        "title": best["title"],
        "title_score": scored[0][1],
        "seniority_score": scored[0][2],
    }


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: match prospects to unmatched campaign sequences."""
    cfg = settings.apollo
    if not cfg.enabled:
        return {"_skip_synthesis": "Apollo integration disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    # Find unmatched sequences
    sequences = await pool.fetch(
        """
        SELECT cs.id, cs.company_name, cs.company_context, cs.batch_id
        FROM campaign_sequences cs
        WHERE cs.status = 'active'
          AND cs.recipient_email IS NULL
        ORDER BY cs.created_at DESC
        LIMIT 50
        """,
    )

    if not sequences:
        return {"_skip_synthesis": "No unmatched sequences"}

    from .campaign_audit import log_campaign_event

    matched = 0
    skipped = 0
    accepted = cfg.accepted_email_statuses

    for seq in sequences:
        try:
            match = await _match_sequence(pool, dict(seq), accepted)
            if not match:
                skipped += 1
                continue

            email = match["email"]
            seq_id = seq["id"]

            # Assign to sequence
            await pool.execute(
                "UPDATE campaign_sequences SET recipient_email = $1, updated_at = NOW() WHERE id = $2",
                email, seq_id,
            )

            # Update draft campaigns in this sequence
            await pool.execute(
                """
                UPDATE b2b_campaigns
                SET recipient_email = $1
                WHERE sequence_id = $2 AND status = 'draft'
                """,
                email, seq_id,
            )

            # Mark prospect as contacted
            await pool.execute(
                "UPDATE prospects SET status = 'contacted', updated_at = NOW() WHERE id = $1",
                match["prospect_id"],
            )

            # Audit log
            await log_campaign_event(
                pool,
                event_type="prospect_matched",
                source="prospect_matching",
                sequence_id=seq_id,
                recipient_email=email,
                metadata={
                    "prospect_id": str(match["prospect_id"]),
                    "prospect_name": f"{match['first_name']} {match['last_name']}",
                    "prospect_title": match["title"],
                    "title_score": match["title_score"],
                    "seniority_score": match["seniority_score"],
                    "company": seq["company_name"],
                },
            )

            matched += 1
            logger.info(
                "Matched %s (%s) -> sequence %s (%s)",
                email, match["title"], seq_id, seq["company_name"],
            )

        except Exception:
            logger.exception("Error matching sequence %s", seq["id"])

    result = {
        "sequences_checked": len(sequences),
        "matched": matched,
        "no_prospect_available": skipped,
    }

    if not matched:
        result["_skip_synthesis"] = "No new matches found"

    return result
