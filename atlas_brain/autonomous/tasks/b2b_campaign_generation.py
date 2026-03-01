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
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from .campaign_audit import log_campaign_event

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
        target_mode=cfg.target_mode,
    )

    # Send notification
    if result.get("generated", 0) > 0:
        from ...pipelines.notify import send_pipeline_notification

        mode_label = result.get("target_mode", cfg.target_mode).replace("_", " ").title()
        msg = (
            f"[{mode_label}] Generated {result['generated']} campaign(s) for "
            f"{result['companies']} company/companies. "
            f"Review drafts in the Leads dashboard."
        )
        await send_pipeline_notification(
            msg, task, title="Atlas: ABM Campaigns",
            default_tags="briefcase,campaign",
        )

    return {"_skip_synthesis": "Campaign generation complete", **result}


async def _create_sequence_for_cold_email(
    pool,
    *,
    company_name: str,
    batch_id: str,
    partner_id: str | None,
    context: dict[str, Any],
    cold_email_subject: str,
    cold_email_body: str,
) -> _uuid.UUID | None:
    """Create a campaign_sequences row and link the cold email to it.

    Returns the sequence ID if created, None on conflict (already exists).
    """
    cfg = settings.campaign_sequence

    seq_id = await pool.fetchval(
        """
        INSERT INTO campaign_sequences (
            company_name, batch_id, partner_id,
            company_context, selling_context, max_steps
        ) VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT ((LOWER(company_name)), batch_id) DO NOTHING
        RETURNING id
        """,
        company_name,
        batch_id,
        _uuid.UUID(partner_id) if partner_id else None,
        json.dumps(context, default=str),
        json.dumps(context.get("selling", {}), default=str),
        cfg.max_steps,
    )

    if not seq_id:
        logger.debug("Sequence already exists for %s / %s", company_name, batch_id)
        return None

    # Link the cold email campaign row to this sequence
    await pool.execute(
        """
        UPDATE b2b_campaigns
        SET sequence_id = $1, step_number = 1
        WHERE company_name = $2 AND batch_id = $3 AND channel = 'email_cold'
        """,
        seq_id,
        company_name,
        batch_id,
    )

    # Audit log
    await log_campaign_event(
        pool,
        event_type="generated",
        sequence_id=seq_id,
        step_number=1,
        source="system",
        subject=cold_email_subject,
        body=cold_email_body,
    )

    # Best-effort CRM recipient lookup
    try:
        contact_email = await pool.fetchval(
            """
            SELECT email FROM contacts
            WHERE LOWER(full_name) LIKE '%' || LOWER($1) || '%'
              AND email IS NOT NULL
            ORDER BY created_at DESC LIMIT 1
            """,
            company_name,
        )
        if contact_email:
            await pool.execute(
                "UPDATE campaign_sequences SET recipient_email = $1 WHERE id = $2",
                contact_email,
                seq_id,
            )
            logger.info(
                "Auto-populated recipient %s for sequence %s (%s)",
                contact_email, seq_id, company_name,
            )
    except Exception:
        logger.debug("CRM recipient lookup failed for %s, skipping", company_name)

    logger.info("Created campaign sequence %s for %s (batch %s)", seq_id, company_name, batch_id)
    return seq_id


async def generate_campaigns(
    pool,
    min_score: int = 70,
    limit: int = 20,
    vendor_filter: str | None = None,
    company_filter: str | None = None,
    target_mode: str | None = None,
) -> dict[str, Any]:
    """Core generation logic, shared by autonomous task and manual API trigger.

    Dispatches to the appropriate generation path based on target_mode:
      - churning_company: original behavior (outreach to the churning company)
      - vendor_retention: sell churn intelligence to the vendor losing customers
      - challenger_intel: sell intent leads to the challenger gaining customers
    """
    cfg = settings.b2b_campaign
    mode = target_mode or cfg.target_mode

    if mode == "vendor_retention":
        return await _generate_vendor_campaigns(pool, min_score, limit, vendor_filter)
    elif mode == "challenger_intel":
        return await _generate_challenger_campaigns(pool, min_score, limit, vendor_filter)

    # Default: churning_company (original behavior)
    return await _generate_churning_company_campaigns(
        pool, min_score, limit, vendor_filter, company_filter,
    )


async def _generate_churning_company_campaigns(
    pool,
    min_score: int,
    limit: int,
    vendor_filter: str | None,
    company_filter: str | None,
) -> dict[str, Any]:
    """Original generation path: outreach to the churning company."""
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
    skipped_no_partner = 0
    sequences_created = 0

    # Fetch affiliate partners for sender identity matching
    partner_index = await _fetch_affiliate_partners(pool)

    for company_name, opps in companies_to_process:
        # Build context from best opportunity in the group
        best = max(opps, key=lambda o: o["opportunity_score"])
        context = _build_company_context(best, opps)

        # Match to an affiliate partner (Gap 4: sender identity)
        partner = _match_partner(context, partner_index)
        if not partner:
            logger.debug("No partner match for %s, skipping", company_name)
            skipped_no_partner += 1
            continue

        # Inject selling context
        context["selling"] = {
            "product_name": partner["product_name"],
            "affiliate_url": partner["affiliate_url"],
            "sender_name": cfg.default_sender_name,
            "sender_company": cfg.default_sender_company,
        }
        partner_id = partner["id"]

        # Channel chaining: track cold email output for follow-up context (Gap 1)
        cold_email_content: dict[str, str] | None = None

        for channel in cfg.channels:
            payload = {**context, "channel": channel}

            # Inject cold email context for follow-up (Gap 1)
            if channel == "email_followup" and cold_email_content:
                payload["cold_email_context"] = cold_email_content

            content = await _generate_content(
                llm, skill.content, payload, cfg.max_tokens, cfg.temperature,
            )

            if content:
                # Capture cold email output for chaining
                if channel == "email_cold":
                    cold_email_content = {
                        "subject": content.get("subject", ""),
                        "body": content.get("body", ""),
                    }

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
                            status, batch_id, llm_model,
                            partner_id, industry, target_mode
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                            $11, $12, $13, $14, $15, $16, $17, $18,
                            $19, $20, $21, $22, $23, $24
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
                        _uuid.UUID(partner_id),
                        context.get("industry"),
                        "churning_company",
                    )
                    generated += 1
                except Exception:
                    logger.exception(
                        "Failed to store campaign for %s/%s", company_name, channel
                    )
                    failed += 1
            else:
                failed += 1

        # Create campaign sequence for the cold email (if sequences enabled)
        if cold_email_content and settings.campaign_sequence.enabled:
            try:
                seq_id = await _create_sequence_for_cold_email(
                    pool,
                    company_name=company_name,
                    batch_id=batch_id,
                    partner_id=partner_id,
                    context=context,
                    cold_email_subject=cold_email_content.get("subject", ""),
                    cold_email_body=cold_email_content.get("body", ""),
                )
                if seq_id:
                    sequences_created += 1
            except Exception as exc:
                logger.warning(
                    "Failed to create sequence for %s: %s", company_name, exc
                )

    logger.info(
        "Campaign generation (churning_company): %d generated, %d failed, %d skipped (no partner), %d sequences from %d companies",
        generated, failed, skipped_no_partner, sequences_created, len(companies_to_process),
    )

    return {
        "generated": generated,
        "failed": failed,
        "skipped": len(by_company) - len(companies_to_process),
        "skipped_no_partner": skipped_no_partner,
        "sequences_created": sequences_created,
        "companies": len(companies_to_process),
        "batch_id": batch_id,
        "target_mode": "churning_company",
    }


# ------------------------------------------------------------------
# Vendor retention campaign generation (P1)
# ------------------------------------------------------------------


async def _fetch_vendor_targets(pool, vendor_name: str | None = None) -> list[dict]:
    """Fetch active vendor targets, optionally filtered by vendor name."""
    if vendor_name:
        rows = await pool.fetch(
            """
            SELECT id, company_name, target_mode, contact_name, contact_email,
                   contact_role, products_tracked, competitors_tracked, tier, status, notes
            FROM vendor_targets
            WHERE status = 'active' AND target_mode = 'vendor_retention'
              AND company_name ILIKE '%' || $1 || '%'
            """,
            vendor_name,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, company_name, target_mode, contact_name, contact_email,
                   contact_role, products_tracked, competitors_tracked, tier, status, notes
            FROM vendor_targets
            WHERE status = 'active' AND target_mode = 'vendor_retention'
            """
        )
    return [dict(r) for r in rows]


def _build_vendor_context(vendor_name: str, signals: list[dict]) -> dict[str, Any]:
    """Aggregate churn signals into a vendor-scoped intelligence summary."""
    total = len(signals)
    high_urgency = sum(1 for s in signals if _safe_float(s.get("urgency"), 0) >= 8)
    medium_urgency = sum(1 for s in signals if 5 <= _safe_float(s.get("urgency"), 0) < 8)

    # Pain distribution
    pain_counts: dict[str, int] = {}
    for s in signals:
        pain = _parse_json_field(s.get("pain_json"))
        for p in pain:
            if isinstance(p, dict) and p.get("category"):
                pain_counts[p["category"]] = pain_counts.get(p["category"], 0) + 1

    # Competitor distribution (who they're losing to)
    comp_counts: dict[str, int] = {}
    for s in signals:
        comps = s.get("competitors", [])
        for c in comps:
            if isinstance(c, dict) and c.get("name"):
                name = c["name"]
                comp_counts[name] = comp_counts.get(name, 0) + 1

    # Feature gaps
    gap_counts: dict[str, int] = {}
    for s in signals:
        gaps = _parse_json_field(s.get("feature_gaps"))
        for g in gaps:
            label = g if isinstance(g, str) else (g.get("feature", "") if isinstance(g, dict) else "")
            if label:
                gap_counts[label] = gap_counts.get(label, 0) + 1

    # Timeline signals
    timeline_count = sum(1 for s in signals if s.get("contract_end"))

    return {
        "vendor_name": vendor_name,
        "signal_summary": {
            "total_signals": total,
            "high_urgency_count": high_urgency,
            "medium_urgency_count": medium_urgency,
            "pain_distribution": sorted(
                [{"category": k, "count": v} for k, v in pain_counts.items()],
                key=lambda x: x["count"], reverse=True,
            )[:10],
            "competitor_distribution": sorted(
                [{"name": k, "count": v} for k, v in comp_counts.items()],
                key=lambda x: x["count"], reverse=True,
            )[:10],
            "feature_gaps": sorted(
                gap_counts.keys(), key=lambda k: gap_counts[k], reverse=True,
            )[:10],
            "timeline_signals": timeline_count,
            "trend_vs_last_month": None,  # TODO: compute from historical data
        },
    }


async def _generate_vendor_campaigns(
    pool,
    min_score: int,
    limit: int,
    vendor_filter: str | None,
) -> dict[str, Any]:
    """Generate campaigns targeting vendor CS/Product leaders with churn intelligence."""
    cfg = settings.b2b_campaign

    # 1. Fetch vendor targets (our customers)
    targets = await _fetch_vendor_targets(pool, vendor_filter)
    if not targets:
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0,
                "target_mode": "vendor_retention", "error": "No active vendor targets"}

    # 2. Fetch all enriched opportunities
    opportunities = await _fetch_opportunities(pool, min_score, limit * 5, dm_only=False)

    # 3. Get LLM + skill
    from ...services.llm_router import get_llm
    llm = get_llm("campaign")
    if llm is None:
        from ...services import llm_registry
        llm = llm_registry.get_active()
    if llm is None:
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0,
                "target_mode": "vendor_retention", "error": "No LLM available"}

    from ...skills import get_skill_registry
    skill = get_skill_registry().get("digest/b2b_vendor_outreach")
    if not skill:
        logger.warning("Skill 'digest/b2b_vendor_outreach' not found")
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0,
                "target_mode": "vendor_retention", "error": "Skill not found"}

    llm_model_name = getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
    batch_id = f"batch_vr_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    generated = 0
    failed = 0
    skipped = 0

    for target in targets[:limit]:
        vendor_name = target["company_name"]

        # Dedup: skip vendor if campaign sent within dedup_days
        existing = await pool.fetchval(
            """
            SELECT COUNT(*) FROM b2b_campaigns
            WHERE LOWER(company_name) = $1
              AND target_mode = 'vendor_retention'
              AND created_at > NOW() - make_interval(days => $2)
            """,
            vendor_name.lower(), cfg.dedup_days,
        )
        if existing > 0:
            skipped += 1
            continue

        # Group signals: opportunities where vendor_name matches this target
        products = target.get("products_tracked") or []
        vendor_signals = [
            opp for opp in opportunities
            if opp["vendor_name"].lower() == vendor_name.lower()
            or (products and opp["vendor_name"].lower() in [p.lower() for p in products])
        ]

        if not vendor_signals:
            logger.debug("No churn signals found for vendor %s, skipping", vendor_name)
            skipped += 1
            continue

        # Build vendor-scoped context
        vendor_ctx = _build_vendor_context(vendor_name, vendor_signals)
        best = max(vendor_signals, key=lambda o: o["opportunity_score"])
        review_ids = [o["review_id"] for o in vendor_signals if o.get("review_id")]

        # Generate for email_cold and email_followup
        cold_email_content: dict[str, str] | None = None
        for channel in ["email_cold", "email_followup"]:
            payload = {
                **vendor_ctx,
                "contact_name": target.get("contact_name"),
                "contact_role": target.get("contact_role"),
                "tier": target.get("tier", "report"),
                "selling": {
                    "sender_name": cfg.default_sender_name,
                    "sender_company": cfg.default_sender_company,
                    "booking_url": cfg.default_booking_url,
                },
                "channel": channel,
            }
            if channel == "email_followup" and cold_email_content:
                payload["cold_email_context"] = cold_email_content

            content = await _generate_content(
                llm, skill.content, payload, cfg.max_tokens, cfg.temperature,
            )

            if content:
                if channel == "email_cold":
                    cold_email_content = {
                        "subject": content.get("subject", ""),
                        "body": content.get("body", ""),
                    }
                try:
                    await pool.execute(
                        """
                        INSERT INTO b2b_campaigns (
                            company_name, vendor_name, product_category,
                            opportunity_score, urgency_score, pain_categories,
                            competitors_considering, seat_count, contract_end,
                            decision_timeline, buying_stage, role_type,
                            key_quotes, source_review_ids,
                            channel, subject, body, cta,
                            status, batch_id, llm_model, industry, target_mode
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                            $11, $12, $13, $14, $15, $16, $17, $18,
                            $19, $20, $21, $22, $23
                        )
                        """,
                        vendor_name,  # company_name = the vendor we're targeting
                        vendor_name,  # vendor_name = same (they're the vendor)
                        best.get("product_category"),
                        best["opportunity_score"],
                        best.get("urgency"),
                        json.dumps(vendor_ctx["signal_summary"]["pain_distribution"]),
                        json.dumps(vendor_ctx["signal_summary"]["competitor_distribution"]),
                        best.get("seat_count"),
                        best.get("contract_end"),
                        best.get("decision_timeline"),
                        best.get("buying_stage"),
                        target.get("contact_role"),
                        json.dumps([]),
                        review_ids[:20] or None,
                        channel,
                        content.get("subject", ""),
                        content.get("body", ""),
                        content.get("cta", ""),
                        "draft",
                        batch_id,
                        llm_model_name,
                        best.get("industry"),
                        "vendor_retention",
                    )
                    generated += 1
                except Exception:
                    logger.exception("Failed to store vendor campaign for %s/%s", vendor_name, channel)
                    failed += 1
            else:
                failed += 1

    logger.info(
        "Campaign generation (vendor_retention): %d generated, %d failed, %d skipped from %d targets",
        generated, failed, skipped, len(targets),
    )

    return {
        "generated": generated,
        "failed": failed,
        "skipped": skipped,
        "companies": len(targets) - skipped,
        "batch_id": batch_id,
        "target_mode": "vendor_retention",
    }


# ------------------------------------------------------------------
# Challenger intel campaign generation (P2)
# ------------------------------------------------------------------


async def _fetch_challenger_targets(pool, vendor_filter: str | None = None) -> list[dict]:
    """Fetch active challenger targets."""
    if vendor_filter:
        rows = await pool.fetch(
            """
            SELECT id, company_name, target_mode, contact_name, contact_email,
                   contact_role, products_tracked, competitors_tracked, tier, status, notes
            FROM vendor_targets
            WHERE status = 'active' AND target_mode = 'challenger_intel'
              AND company_name ILIKE '%' || $1 || '%'
            """,
            vendor_filter,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, company_name, target_mode, contact_name, contact_email,
                   contact_role, products_tracked, competitors_tracked, tier, status, notes
            FROM vendor_targets
            WHERE status = 'active' AND target_mode = 'challenger_intel'
            """
        )
    return [dict(r) for r in rows]


def _build_challenger_context(challenger_name: str, signals: list[dict]) -> dict[str, Any]:
    """Aggregate signals where a specific product is mentioned as the alternative."""
    total = len(signals)

    # Buying stage distribution
    by_stage: dict[str, int] = {}
    for s in signals:
        stage = s.get("buying_stage") or "unknown"
        by_stage[stage] = by_stage.get(stage, 0) + 1

    # Role distribution
    role_counts: dict[str, int] = {}
    for s in signals:
        role = s.get("role_type") or "unknown"
        role_counts[role] = role_counts.get(role, 0) + 1

    # Pain categories driving the switch (from incumbent)
    pain_counts: dict[str, int] = {}
    for s in signals:
        pain = _parse_json_field(s.get("pain_json"))
        for p in pain:
            if isinstance(p, dict) and p.get("category"):
                pain_counts[p["category"]] = pain_counts.get(p["category"], 0) + 1

    # Incumbents losing customers
    incumbent_counts: dict[str, int] = {}
    for s in signals:
        vendor = s.get("vendor_name", "")
        if vendor:
            incumbent_counts[vendor] = incumbent_counts.get(vendor, 0) + 1

    # Seat count buckets
    large = sum(1 for s in signals if (s.get("seat_count") or 0) >= 500)
    mid = sum(1 for s in signals if 100 <= (s.get("seat_count") or 0) < 500)
    small = sum(1 for s in signals if 0 < (s.get("seat_count") or 0) < 100)

    # Feature mentions (positive mentions of challenger from competitor context)
    feature_mentions: list[str] = []
    for s in signals:
        comps = s.get("competitors", [])
        for c in comps:
            if isinstance(c, dict) and c.get("name", "").lower() == challenger_name.lower():
                reason = c.get("reason", "")
                if reason and reason not in feature_mentions:
                    feature_mentions.append(reason)

    return {
        "challenger_name": challenger_name,
        "signal_summary": {
            "total_leads": total,
            "by_buying_stage": {
                "active_purchase": by_stage.get("active_purchase", 0),
                "evaluation": by_stage.get("evaluation", 0),
                "renewal_decision": by_stage.get("renewal_decision", 0),
            },
            "role_distribution": sorted(
                [{"role": k, "count": v} for k, v in role_counts.items()],
                key=lambda x: x["count"], reverse=True,
            )[:5],
            "pain_driving_switch": sorted(
                [{"category": k, "count": v} for k, v in pain_counts.items()],
                key=lambda x: x["count"], reverse=True,
            )[:10],
            "incumbents_losing": sorted(
                [{"name": k, "count": v} for k, v in incumbent_counts.items()],
                key=lambda x: x["count"], reverse=True,
            )[:10],
            "seat_count_signals": {
                "large_500plus": large,
                "mid_100_499": mid,
                "small_under_100": small,
            },
            "feature_mentions": feature_mentions[:10],
        },
    }


async def _generate_challenger_campaigns(
    pool,
    min_score: int,
    limit: int,
    vendor_filter: str | None,
) -> dict[str, Any]:
    """Generate campaigns targeting challenger Sales/Competitive Intel leaders."""
    cfg = settings.b2b_campaign

    # 1. Fetch challenger targets
    targets = await _fetch_challenger_targets(pool, vendor_filter)
    if not targets:
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0,
                "target_mode": "challenger_intel", "error": "No active challenger targets"}

    # 2. Fetch all enriched opportunities
    opportunities = await _fetch_opportunities(pool, min_score, limit * 5, dm_only=False)

    # 3. Get LLM + skill
    from ...services.llm_router import get_llm
    llm = get_llm("campaign")
    if llm is None:
        from ...services import llm_registry
        llm = llm_registry.get_active()
    if llm is None:
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0,
                "target_mode": "challenger_intel", "error": "No LLM available"}

    from ...skills import get_skill_registry
    skill = get_skill_registry().get("digest/b2b_challenger_outreach")
    if not skill:
        logger.warning("Skill 'digest/b2b_challenger_outreach' not found")
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0,
                "target_mode": "challenger_intel", "error": "Skill not found"}

    llm_model_name = getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
    batch_id = f"batch_ci_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    generated = 0
    failed = 0
    skipped = 0

    for target in targets[:limit]:
        challenger_name = target["company_name"]

        # Dedup
        existing = await pool.fetchval(
            """
            SELECT COUNT(*) FROM b2b_campaigns
            WHERE LOWER(company_name) = $1
              AND target_mode = 'challenger_intel'
              AND created_at > NOW() - make_interval(days => $2)
            """,
            challenger_name.lower(), cfg.dedup_days,
        )
        if existing > 0:
            skipped += 1
            continue

        # Find signals where this challenger is mentioned as a competitor being considered
        challenger_signals = []
        for opp in opportunities:
            comps = opp.get("competitors", [])
            for c in comps:
                if isinstance(c, dict) and c.get("name", "").lower() == challenger_name.lower():
                    challenger_signals.append(opp)
                    break
            # Also match against products_tracked / competitors_tracked
            if not any(opp is s for s in challenger_signals):
                products = target.get("competitors_tracked") or []
                if opp["vendor_name"].lower() in [p.lower() for p in products]:
                    challenger_signals.append(opp)

        if not challenger_signals:
            logger.debug("No intent signals found for challenger %s, skipping", challenger_name)
            skipped += 1
            continue

        # Build challenger-scoped context
        challenger_ctx = _build_challenger_context(challenger_name, challenger_signals)
        best = max(challenger_signals, key=lambda o: o["opportunity_score"])
        review_ids = [o["review_id"] for o in challenger_signals if o.get("review_id")]

        cold_email_content: dict[str, str] | None = None
        for channel in ["email_cold", "email_followup"]:
            payload = {
                **challenger_ctx,
                "contact_name": target.get("contact_name"),
                "contact_role": target.get("contact_role"),
                "tier": target.get("tier", "report"),
                "selling": {
                    "sender_name": cfg.default_sender_name,
                    "sender_company": cfg.default_sender_company,
                    "booking_url": cfg.default_booking_url,
                },
                "channel": channel,
            }
            if channel == "email_followup" and cold_email_content:
                payload["cold_email_context"] = cold_email_content

            content = await _generate_content(
                llm, skill.content, payload, cfg.max_tokens, cfg.temperature,
            )

            if content:
                if channel == "email_cold":
                    cold_email_content = {
                        "subject": content.get("subject", ""),
                        "body": content.get("body", ""),
                    }
                try:
                    await pool.execute(
                        """
                        INSERT INTO b2b_campaigns (
                            company_name, vendor_name, product_category,
                            opportunity_score, urgency_score, pain_categories,
                            competitors_considering, seat_count, contract_end,
                            decision_timeline, buying_stage, role_type,
                            key_quotes, source_review_ids,
                            channel, subject, body, cta,
                            status, batch_id, llm_model, industry, target_mode
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                            $11, $12, $13, $14, $15, $16, $17, $18,
                            $19, $20, $21, $22, $23
                        )
                        """,
                        challenger_name,  # company_name = the challenger we're targeting
                        best["vendor_name"],  # vendor_name = the incumbent losing
                        best.get("product_category"),
                        best["opportunity_score"],
                        best.get("urgency"),
                        json.dumps(challenger_ctx["signal_summary"]["pain_driving_switch"]),
                        json.dumps(challenger_ctx["signal_summary"]["incumbents_losing"]),
                        best.get("seat_count"),
                        best.get("contract_end"),
                        best.get("decision_timeline"),
                        best.get("buying_stage"),
                        target.get("contact_role"),
                        json.dumps([]),
                        review_ids[:20] or None,
                        channel,
                        content.get("subject", ""),
                        content.get("body", ""),
                        content.get("cta", ""),
                        "draft",
                        batch_id,
                        llm_model_name,
                        best.get("industry"),
                        "challenger_intel",
                    )
                    generated += 1
                except Exception:
                    logger.exception("Failed to store challenger campaign for %s/%s", challenger_name, channel)
                    failed += 1
            else:
                failed += 1

    logger.info(
        "Campaign generation (challenger_intel): %d generated, %d failed, %d skipped from %d targets",
        generated, failed, skipped, len(targets),
    )

    return {
        "generated": generated,
        "failed": failed,
        "skipped": skipped,
        "companies": len(targets) - skipped,
        "batch_id": batch_id,
        "target_mode": "challenger_intel",
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
               r.enrichment->'quotable_phrases' AS quotable_phrases,
               r.enrichment->'feature_gaps' AS feature_gaps,
               r.enrichment->'use_case'->>'primary_workflow' AS primary_workflow,
               r.enrichment->'use_case'->'integration_stack' AS integration_stack,
               r.enrichment->'sentiment_trajectory'->>'direction' AS sentiment_direction,
               COALESCE(r.reviewer_industry, r.enrichment->'reviewer_context'->>'industry') AS industry
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


def _parse_json_field(val) -> list:
    """Safely parse a JSONB field that may be a str, list, or None."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _build_company_context(best: dict, all_opps: list[dict]) -> dict[str, Any]:
    """Build rich context dict for LLM from grouped opportunities."""
    pain_cats: dict[str, str] = {}
    competitors_considering: list[dict] = []
    key_quotes: list[str] = []
    all_feature_gaps: list[str] = []
    all_integrations: list[str] = []

    for opp in all_opps:
        # Pain categories
        pain = _parse_json_field(opp.get("pain_json"))
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

        # Curated quotes from enrichment (replaces raw review_text truncation)
        phrases = _parse_json_field(opp.get("quotable_phrases"))
        for phrase in phrases:
            text = phrase if isinstance(phrase, str) else (phrase.get("text", "") if isinstance(phrase, dict) else "")
            if text and text not in key_quotes:
                key_quotes.append(text)

        # Feature gaps
        gaps = _parse_json_field(opp.get("feature_gaps"))
        for g in gaps:
            label = g if isinstance(g, str) else (g.get("feature", "") if isinstance(g, dict) else "")
            if label and label not in all_feature_gaps:
                all_feature_gaps.append(label)

        # Integration stack
        stacks = _parse_json_field(opp.get("integration_stack"))
        for s in stacks:
            if isinstance(s, str) and s not in all_integrations:
                all_integrations.append(s)

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
        "industry": best.get("industry"),
        "key_quotes": key_quotes[:5],
        "feature_gaps": all_feature_gaps[:5],
        "primary_workflow": best.get("primary_workflow"),
        "integration_stack": all_integrations[:5],
        "sentiment_direction": best.get("sentiment_direction"),
    }


# ------------------------------------------------------------------
# Partner matching
# ------------------------------------------------------------------


async def _fetch_affiliate_partners(pool) -> dict[str, Any]:
    """Fetch enabled affiliate partners, indexed by product name and category."""
    rows = await pool.fetch(
        "SELECT id, name, product_name, product_aliases, category, affiliate_url "
        "FROM affiliate_partners WHERE enabled = true"
    )
    by_product: dict[str, dict] = {}
    by_category: dict[str, list[dict]] = {}

    for r in rows:
        partner = dict(r)
        partner["id"] = str(partner["id"])
        # Index by lowercase product name + aliases
        by_product[partner["product_name"].lower()] = partner
        for alias in (partner.get("product_aliases") or []):
            by_product[alias.lower()] = partner
        # Index by lowercase category
        cat = (partner.get("category") or "").lower()
        if cat:
            by_category.setdefault(cat, []).append(partner)

    return {"by_product": by_product, "by_category": by_category}


def _match_partner(
    context: dict,
    partner_index: dict[str, Any],
) -> dict | None:
    """Match a company context to the best affiliate partner.

    Priority: (1) exact product match against competitors, (2) category fallback.
    """
    by_product = partner_index["by_product"]
    by_category = partner_index["by_category"]

    # Try exact match on competitor names
    for comp in context.get("competitors_considering", []):
        name = (comp.get("name") or "").lower()
        if name and name in by_product:
            return by_product[name]

    # Fallback: match by product category
    category = (context.get("category") or "").lower()
    if category and category in by_category:
        return by_category[category][0]

    return None


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
