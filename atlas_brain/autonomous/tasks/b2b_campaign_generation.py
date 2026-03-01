"""
B2B ABM Campaign Generation: uses Claude (draft LLM) to produce personalized
outreach content -- cold emails, LinkedIn messages, follow-up emails -- from
the highest-scoring churn intelligence opportunities.

Runs daily after b2b_churn_intelligence. Reads enriched b2b_reviews, scores
opportunities, groups by company, and generates multi-channel campaigns.

Returns _skip_synthesis.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.b2b_campaign_generation")

# Reuse scoring constants from b2b_affiliates
_ROLE_SCORES = {
    "decision_maker": 20,
    "economic_buyer": 15,
    "champion": 15,
    "evaluator": 10,
}

_STAGE_SCORES = {
    "active_purchase": 25,
    "evaluation": 20,
    "renewal_decision": 15,
    "post_purchase": 5,
}

_CONTEXT_SCORES = {
    "considering": 10,
    "switched_to": 8,
    "compared": 6,
    "switched_from": 2,
}


def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _compute_score(row: dict) -> int:
    """Compute opportunity score (0-100) from enrichment signals."""
    score = 0.0
    urgency = _safe_float(row.get("urgency"), 0)
    score += max(0, min(30, (urgency - 5) * 6))

    if row.get("is_dm"):
        score += 20
    elif row.get("role_type") in _ROLE_SCORES:
        score += _ROLE_SCORES[row["role_type"]]

    buying_stage = row.get("buying_stage") or ""
    score += _STAGE_SCORES.get(buying_stage, 0)

    seat_count = row.get("seat_count")
    if seat_count is not None:
        if seat_count >= 500:
            score += 15
        elif seat_count >= 100:
            score += 10
        elif seat_count >= 20:
            score += 5

    mention_context = (row.get("mention_context") or "").lower()
    for keyword, pts in _CONTEXT_SCORES.items():
        if keyword in mention_context:
            score += pts
            break

    return int(min(100, max(0, score)))


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: generate ABM campaigns from churn intelligence."""
    cfg = settings.b2b_campaign
    if not cfg.enabled:
        return {"_skip_synthesis": "B2B campaign generation disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    result = await generate_campaigns(
        pool=pool,
        min_score=cfg.min_opportunity_score,
        limit=cfg.max_campaigns_per_run,
    )

    # Send notification
    if result.get("generated", 0) > 0:
        from ...pipelines.notify import send_pipeline_notification

        msg = (
            f"Generated {result['generated']} campaign(s) for "
            f"{result['companies']} company/companies. "
            f"Review drafts in the Leads dashboard."
        )
        await send_pipeline_notification(
            msg, task, title="Atlas: ABM Campaigns",
            default_tags="briefcase,campaign",
        )

    return {"_skip_synthesis": "Campaign generation complete", **result}


async def generate_campaigns(
    pool,
    min_score: int = 70,
    limit: int = 20,
    vendor_filter: str | None = None,
    company_filter: str | None = None,
) -> dict[str, Any]:
    """Core generation logic, shared by autonomous task and manual API trigger."""
    cfg = settings.b2b_campaign

    # 1. Fetch top opportunities from enriched reviews
    opportunities = await _fetch_opportunities(
        pool, min_score, limit,
        vendor_filter=vendor_filter,
        company_filter=company_filter,
        dm_only=cfg.require_decision_maker,
    )
    if not opportunities:
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0}

    # 2. Group by company (one campaign set per company)
    #    Skip opportunities with no reviewer_company -- we need a real target
    by_company: dict[str, list[dict]] = {}
    skipped_no_company = 0
    for opp in opportunities:
        company = opp.get("reviewer_company")
        if not company:
            skipped_no_company += 1
            continue
        key = company.lower()
        if key not in by_company:
            by_company[key] = []
        by_company[key].append(opp)

    # 3. Dedup: skip companies with recent campaigns
    companies_to_process = []
    for company_key, opps in by_company.items():
        company_name = opps[0].get("reviewer_company") or opps[0]["vendor_name"]
        existing = await pool.fetchval(
            """
            SELECT COUNT(*) FROM b2b_campaigns
            WHERE LOWER(company_name) = $1
              AND created_at > NOW() - make_interval(days => $2)
            """,
            company_key, cfg.dedup_days,
        )
        if existing == 0:
            companies_to_process.append((company_name, opps))

    if not companies_to_process:
        return {"generated": 0, "skipped": len(by_company), "failed": 0, "companies": 0}

    # 4. Get LLM
    from ...services.llm_router import get_llm
    llm = get_llm("campaign")
    if llm is None:
        from ...services import llm_registry
        llm = llm_registry.get_active()
    if llm is None:
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0, "error": "No LLM available"}

    # 5. Load skill prompt
    from ...skills import get_skill_registry
    from ...services.protocols import Message

    skill = get_skill_registry().get("digest/b2b_campaign_generation")
    if not skill:
        logger.warning("Skill 'digest/b2b_campaign_generation' not found")
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0, "error": "Skill not found"}

    llm_model_name = getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
    batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    generated = 0
    failed = 0

    for company_name, opps in companies_to_process:
        # Build context from best opportunity in the group
        best = max(opps, key=lambda o: o["opportunity_score"])
        context = _build_company_context(best, opps)

        for channel in cfg.channels:
            payload = {**context, "channel": channel}

            content = await _generate_content(
                llm, skill.content, payload, cfg.max_tokens, cfg.temperature,
            )

            if content:
                try:
                    review_ids = [o["review_id"] for o in opps if o.get("review_id")]
                    await pool.execute(
                        """
                        INSERT INTO b2b_campaigns (
                            company_name, vendor_name, product_category,
                            opportunity_score, urgency_score, pain_categories,
                            competitors_considering, seat_count, contract_end,
                            decision_timeline, buying_stage, role_type,
                            key_quotes, source_review_ids,
                            channel, subject, body, cta,
                            status, batch_id, llm_model
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                            $11, $12, $13, $14, $15, $16, $17, $18,
                            $19, $20, $21
                        )
                        """,
                        company_name,
                        best["vendor_name"],
                        best.get("product_category"),
                        best["opportunity_score"],
                        best.get("urgency"),
                        json.dumps(context.get("pain_categories", [])),
                        json.dumps(context.get("competitors_considering", [])),
                        best.get("seat_count"),
                        best.get("contract_end"),
                        best.get("decision_timeline"),
                        best.get("buying_stage"),
                        best.get("role_type"),
                        json.dumps(context.get("key_quotes", [])),
                        review_ids or None,
                        channel,
                        content.get("subject", ""),
                        content.get("body", ""),
                        content.get("cta", ""),
                        "draft",
                        batch_id,
                        llm_model_name,
                    )
                    generated += 1
                except Exception:
                    logger.exception(
                        "Failed to store campaign for %s/%s", company_name, channel
                    )
                    failed += 1
            else:
                failed += 1

    logger.info(
        "Campaign generation: %d pieces generated, %d failed (from %d companies)",
        generated, failed, len(companies_to_process),
    )

    return {
        "generated": generated,
        "failed": failed,
        "skipped": len(by_company) - len(companies_to_process),
        "companies": len(companies_to_process),
        "batch_id": batch_id,
    }


# ------------------------------------------------------------------
# Data fetchers
# ------------------------------------------------------------------


async def _fetch_opportunities(
    pool,
    min_score: int,
    limit: int,
    vendor_filter: str | None = None,
    company_filter: str | None = None,
    dm_only: bool = True,
) -> list[dict[str, Any]]:
    """Fetch and score top opportunities from enriched b2b_reviews."""
    extra_conditions = ""
    params: list[Any] = [90, 5.0, min(limit * 3, 500)]  # window_days, min_urgency, fetch_limit
    idx = 4

    if vendor_filter:
        extra_conditions += f" AND r.vendor_name ILIKE '%' || ${idx} || '%'"
        params.append(vendor_filter)
        idx += 1

    if company_filter:
        extra_conditions += f" AND r.reviewer_company ILIKE '%' || ${idx} || '%'"
        params.append(company_filter)
        idx += 1

    if dm_only:
        extra_conditions += " AND (r.enrichment->'reviewer_context'->>'decision_maker')::boolean = true"

    rows = await pool.fetch(
        f"""
        SELECT r.id AS review_id, r.vendor_name, r.reviewer_company, r.product_category,
               (r.enrichment->>'urgency_score')::numeric AS urgency,
               (r.enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
               r.enrichment->'buyer_authority'->>'role_type' AS role_type,
               r.enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
               CASE WHEN r.enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                    THEN (r.enrichment->'budget_signals'->>'seat_count')::int END AS seat_count,
               r.enrichment->'timeline'->>'contract_end' AS contract_end,
               r.enrichment->'timeline'->>'decision_timeline' AS decision_timeline,
               r.enrichment->'competitors_mentioned' AS competitors_json,
               r.enrichment->'pain_categories' AS pain_json,
               r.review_text
        FROM b2b_reviews r
        WHERE r.enrichment_status = 'enriched'
          AND r.enriched_at > NOW() - make_interval(days => $1)
          AND (r.enrichment->>'urgency_score')::numeric >= $2
          {extra_conditions}
        ORDER BY (r.enrichment->>'urgency_score')::numeric DESC
        LIMIT $3
        """,
        *params,
    )

    opportunities = []
    for r in rows:
        row_dict = dict(r)
        # Parse competitors for context scoring
        competitors = row_dict.get("competitors_json")
        if isinstance(competitors, str):
            try:
                competitors = json.loads(competitors)
            except (json.JSONDecodeError, TypeError):
                competitors = []
        if not isinstance(competitors, list):
            competitors = []

        mention_context = ""
        if competitors:
            mention_context = competitors[0].get("context", "")

        row_dict["mention_context"] = mention_context
        row_dict["urgency"] = _safe_float(row_dict.get("urgency"), 0)
        opp_score = _compute_score(row_dict)

        if opp_score < min_score:
            continue

        row_dict["opportunity_score"] = opp_score
        row_dict["competitors"] = competitors
        row_dict["review_id"] = str(r["review_id"])
        opportunities.append(row_dict)

    opportunities.sort(key=lambda o: o["opportunity_score"], reverse=True)
    return opportunities[:limit]


def _build_company_context(best: dict, all_opps: list[dict]) -> dict[str, Any]:
    """Build rich context dict for LLM from grouped opportunities."""
    # Aggregate pain categories across all reviews for this company
    pain_cats: dict[str, str] = {}
    competitors_considering: list[dict] = []
    key_quotes: list[str] = []

    for opp in all_opps:
        # Pain categories
        pain = opp.get("pain_json")
        if isinstance(pain, str):
            try:
                pain = json.loads(pain)
            except (json.JSONDecodeError, TypeError):
                pain = []
        if isinstance(pain, list):
            for p in pain:
                if isinstance(p, dict) and p.get("category"):
                    pain_cats[p["category"]] = p.get("severity", "mentioned")

        # Competitors
        comps = opp.get("competitors", [])
        for c in comps:
            if isinstance(c, dict) and c.get("name"):
                if not any(x["name"].lower() == c["name"].lower() for x in competitors_considering):
                    competitors_considering.append({
                        "name": c["name"],
                        "reason": c.get("reason", ""),
                    })

        # Quotes (snippets from review text)
        review_text = opp.get("review_text", "")
        if review_text and len(review_text) > 20:
            # Take first 200 chars as a quote snippet
            snippet = review_text[:200].strip()
            if snippet and snippet not in key_quotes:
                key_quotes.append(snippet)

    return {
        "company": best.get("reviewer_company") or best["vendor_name"],
        "churning_from": best["vendor_name"],
        "category": best.get("product_category", ""),
        "pain_categories": [
            {"category": k, "severity": v} for k, v in pain_cats.items()
        ],
        "competitors_considering": competitors_considering[:5],
        "urgency": best.get("urgency", 0),
        "seat_count": best.get("seat_count"),
        "contract_end": best.get("contract_end"),
        "decision_timeline": best.get("decision_timeline"),
        "role_type": best.get("role_type"),
        "industry": None,
        "key_quotes": key_quotes[:5],
    }


# ------------------------------------------------------------------
# LLM generation
# ------------------------------------------------------------------


async def _generate_content(
    llm,
    system_prompt: str,
    payload: dict[str, Any],
    max_tokens: int,
    temperature: float,
) -> dict[str, Any] | None:
    """Call LLM with campaign generation skill and parse response."""
    from ...pipelines.llm import clean_llm_output
    from ...services.protocols import Message

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=json.dumps(payload, indent=2, default=str)),
    ]

    text = ""
    try:
        result = llm.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = result.get("response", "").strip()
        if not text:
            return None

        text = clean_llm_output(text)
        parsed = json.loads(text)

        if not isinstance(parsed, dict) or "body" not in parsed:
            logger.debug("Campaign generation missing 'body' field")
            return None

        return parsed

    except json.JSONDecodeError:
        logger.debug("Failed to parse campaign generation JSON: %.200s", text)
        return None
    except Exception:
        logger.exception("Campaign generation LLM call failed")
        return None
