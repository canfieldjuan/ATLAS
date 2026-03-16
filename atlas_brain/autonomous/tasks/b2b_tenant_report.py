"""Weekly per-tenant B2B intelligence report generation.

Runs before the global intelligence task (Sunday 8 PM). For each B2B
account with tracked vendors, gathers vendor-scoped intelligence data,
calls the LLM with the existing b2b_churn_intelligence skill, persists
to b2b_intelligence with account_id, and emails the report summary.
"""

import asyncio
import json
import logging
from datetime import date, timezone
from typing import Any

import httpx

from ...config import settings
from ...services.tracing import (
    build_business_trace_context,
    build_reasoning_trace_context,
    tracer,
)
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.b2b_tenant_report")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Generate scoped intelligence reports for each B2B tenant."""
    cfg = settings.b2b_churn
    if not cfg.enabled or not cfg.intelligence_enabled:
        return {"_skip_synthesis": "B2B churn intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    # Only generate for accounts with b2b_starter+ plans (reports require b2b_starter)
    accounts = await pool.fetch(
        """
        SELECT sa.id AS account_id, sa.name AS account_name,
               sa.plan,
               su.email AS owner_email,
               array_agg(tv.vendor_name) AS vendors
        FROM saas_accounts sa
        JOIN saas_users su ON su.account_id = sa.id AND su.role = 'owner'
        JOIN tracked_vendors tv ON tv.account_id = sa.id
        WHERE sa.product IN ('b2b_retention', 'b2b_challenger')
          AND sa.plan IN ('b2b_starter', 'b2b_growth', 'b2b_pro')
        GROUP BY sa.id, sa.name, sa.plan, su.email
        """
    )

    if not accounts:
        return {"_skip_synthesis": "No eligible B2B accounts for reports"}

    from .b2b_churn_intelligence import gather_intelligence_data

    reports_generated = 0
    today = date.today()

    for acct in accounts:
        account_id = acct["account_id"]
        vendor_names = [v for v in (acct["vendors"] or []) if v]

        if not vendor_names:
            continue

        span = tracer.start_span(
            span_name="b2b.tenant_report",
            operation_type="intelligence",
            session_id=str(account_id),
            metadata={
                "business": build_business_trace_context(
                    account_id=str(account_id),
                    workflow="tenant_report",
                    report_type="weekly_b2b_intelligence",
                    vendor_name=", ".join(vendor_names[:5]),
                ),
            },
        )

        try:
            result = await gather_intelligence_data(
                pool,
                window_days=cfg.intelligence_window_days,
                min_reviews=cfg.intelligence_min_reviews,
                vendor_names=vendor_names,
            )
        except Exception as exc:
            tracer.end_span(
                span,
                status="failed",
                error_message=str(exc),
                error_type=type(exc).__name__,
            )
            logger.error("Data gathering failed for account %s: %s", account_id, exc)
            continue

        payload = result["payload"]

        # Skip if no data
        if not payload.get("vendor_churn_scores") and not payload.get("high_intent_companies"):
            tracer.end_span(span, status="completed", output_data={"skipped": "no scoped intelligence data"})
            continue

        # --- Narrative engine: structured evidence chains + rule evaluation ---
        try:
            from atlas_brain.reasoning.narrative import NarrativeEngine

            narrative_engine = NarrativeEngine(pool)
            narrative_payloads = []
            all_triggered_rules = []

            for vs in (payload.get("vendor_churn_scores") or []):
                vname = vs.get("vendor_name") or vs.get("vendor") or ""
                if not vname:
                    continue

                # Build structured narrative from available evidence
                narrative = narrative_engine.build_vendor_narrative(
                    vendor_name=vname,
                    snapshot=vs,
                    archetype_match={"archetype": vs.get("archetype", ""),
                                     "signal_score": vs.get("signal_score", 0)},
                )

                # Evaluate threshold rules (P5-005)
                triggered = narrative_engine.evaluate_rules(vname, vs, narrative.archetype)
                all_triggered_rules.extend(triggered)

                # Build explainability audit trail
                explain = narrative_engine.build_explainability(narrative)

                # Add structured evidence to payload
                intel_payload = NarrativeEngine.to_intelligence_payload(narrative)
                intel_payload["explainability"] = explain
                narrative_payloads.append(intel_payload)

            if narrative_payloads:
                payload["narrative_evidence"] = narrative_payloads

            # P5-005: Send ntfy alerts for critical/high priority rule triggers
            critical_rules = [t for t in all_triggered_rules
                              if t.rule.priority in ("critical", "high")]
            if critical_rules:
                await _send_rule_alerts(critical_rules)

        except Exception:
            logger.debug("Narrative engine enrichment skipped", exc_info=True)

        # --- Stratified reasoning: recall/reconstitute/reason per vendor ---
        try:
            from atlas_brain.reasoning import get_stratified_reasoner
            from atlas_brain.reasoning.tiers import Tier, gather_tier_context

            reasoner = get_stratified_reasoner()
            if reasoner is not None:
                stratified_results = []
                for vs in (payload.get("vendor_churn_scores") or []):
                    vname = vs.get("vendor_name") or vs.get("vendor") or ""
                    if not vname:
                        continue
                    category = vs.get("product_category") or vs.get("category") or ""
                    try:
                        tier_ctx = await gather_tier_context(
                            reasoner._cache, Tier.VENDOR_ARCHETYPE,
                            vendor_name=vname, product_category=category,
                        )
                        sr = await reasoner.analyze(
                            vendor_name=vname,
                            evidence=vs,
                            product_category=category,
                            tier_context=tier_ctx,
                        )
                        stratified_results.append({
                            "vendor_name": vname,
                            "mode": sr.mode,
                            "archetype": sr.conclusion.get("archetype", ""),
                            "confidence": sr.confidence,
                            "tokens_used": sr.tokens_used,
                            "conclusion": sr.conclusion,
                        })
                    except Exception:
                        logger.debug("Stratified reasoning failed for %s", vname, exc_info=True)

                if stratified_results:
                    # Fetch prior archetypes for temporal context
                    from .b2b_churn_intelligence import _fetch_prior_archetypes

                    sr_vendor_names = [r["vendor_name"] for r in stratified_results if r.get("vendor_name")]
                    prior_archetypes = await _fetch_prior_archetypes(pool, sr_vendor_names)

                    for sr in stratified_results:
                        vname = sr.get("vendor_name", "")
                        prior = prior_archetypes.get(vname, {})
                        sr["archetype_was"] = prior.get("archetype")
                        sr["confidence_was"] = prior.get("confidence")
                        sr["archetype_changed"] = (
                            prior.get("archetype") != sr.get("archetype")
                            if prior.get("archetype") and sr.get("archetype")
                            else None
                        )

                    payload["stratified_reasoning"] = stratified_results
                    logger.info(
                        "Stratified reasoning: %d vendors (%s)",
                        len(stratified_results),
                        ", ".join(f"{r['vendor_name']}={r['mode']}" for r in stratified_results),
                    )
        except Exception:
            logger.debug("Stratified reasoning integration skipped", exc_info=True)

        # LLM synthesis with existing skill
        from ...pipelines.llm import call_llm_with_skill, parse_json_response

        llm_usage: dict[str, Any] = {}
        try:
            analysis = await asyncio.wait_for(
                asyncio.to_thread(
                    call_llm_with_skill,
                    "digest/b2b_churn_intelligence",
                    payload,
                    max_tokens=cfg.intelligence_max_tokens,
                    temperature=0.4,
                    workload="synthesis",
                    response_format={"type": "json_object"},
                    usage_out=llm_usage,
                ),
                timeout=300,
            )
        except asyncio.TimeoutError:
            tracer.end_span(
                span,
                status="failed",
                error_message="tenant report llm timed out",
                error_type="TimeoutError",
            )
            logger.error("LLM timed out for tenant report account=%s", account_id)
            continue

        if llm_usage.get("input_tokens"):
            logger.info("b2b_tenant_report LLM tokens: in=%d out=%d model=%s account=%s",
                         llm_usage["input_tokens"], llm_usage["output_tokens"],
                         llm_usage.get("model", ""), account_id)

        if not analysis:
            tracer.end_span(span, status="failed", error_message="tenant report llm returned no analysis")
            continue

        parsed = parse_json_response(analysis, recover_truncated=True)
        exec_summary = parsed.get("executive_summary", "")

        # Persist reports with account_id
        report_types = [
            ("weekly_churn_feed", parsed.get("weekly_churn_feed", [])),
            ("vendor_scorecard", parsed.get("vendor_scorecards", [])),
            ("displacement_report", parsed.get("displacement_map", [])),
            ("category_overview", parsed.get("category_insights", [])),
        ]

        data_density = json.dumps({
            "vendors_analyzed": result["vendors_analyzed"],
            "high_intent_companies": result["high_intent_companies"],
            "competitive_flows": result["competitive_flows"],
            "pain_categories": result.get("pain_categories", 0),
            "feature_gaps": result.get("feature_gaps", 0),
        })

        try:
            async with pool.transaction() as conn:
                for report_type, data in report_types:
                    await conn.execute(
                        """
                        INSERT INTO b2b_intelligence (
                            report_date, report_type, intelligence_data,
                            executive_summary, data_density, status,
                            llm_model, account_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        today,
                        report_type,
                        json.dumps(data, default=str),
                        exec_summary,
                        data_density,
                        "published",
                        "pipeline_default",
                        account_id,
                    )
            reports_generated += 1
        except Exception as exc:
            tracer.end_span(
                span,
                status="failed",
                error_message=str(exc),
                error_type=type(exc).__name__,
            )
            logger.error("Failed to persist tenant report for %s: %s", account_id, exc)
            continue

        # Send email summary via Resend if configured
        await _send_report_email(
            owner_email=acct["owner_email"],
            account_name=acct["account_name"],
            vendor_names=vendor_names,
            summary=exec_summary,
        )

        # ntfy notification
        await _send_ntfy(acct["account_name"], len(vendor_names))
        tracer.end_span(
            span,
            status="completed",
            input_data={"vendor_names": vendor_names, "window_days": cfg.intelligence_window_days},
            output_data={"reports_generated": len(report_types), "account_name": acct["account_name"]},
            input_tokens=llm_usage.get("input_tokens"),
            output_tokens=llm_usage.get("output_tokens"),
            metadata={
                "reasoning": build_reasoning_trace_context(
                    decision={"report_types": [name for name, _ in report_types]},
                    evidence={
                        "vendors_analyzed": result["vendors_analyzed"],
                        "high_intent_companies": result["high_intent_companies"],
                        "competitive_flows": result["competitive_flows"],
                    },
                    rationale=exec_summary,
                ),
            },
        )

    return {
        "_skip_synthesis": "B2B tenant reports complete",
        "accounts_processed": len(accounts),
        "reports_generated": reports_generated,
    }


async def _send_report_email(
    *,
    owner_email: str,
    account_name: str,
    vendor_names: list[str],
    summary: str,
) -> None:
    """Send weekly report email via Resend."""
    cfg = settings.campaign_sequence
    if not cfg.resend_api_key or not cfg.resend_from_email:
        return

    vendors_str = ", ".join(vendor_names[:5])
    subject = f"Weekly Intelligence Report: {vendors_str}"
    body = (
        f"<h2>Weekly B2B Intelligence Report</h2>"
        f"<p><strong>Account:</strong> {account_name}</p>"
        f"<p><strong>Tracked Vendors:</strong> {vendors_str}</p>"
        f"<hr>"
        f"<h3>Executive Summary</h3>"
        f"<p>{summary or 'No significant changes this week.'}</p>"
        f"<hr>"
        f"<p><em>View full details in your churn intelligence feed.</em></p>"
    )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {cfg.resend_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": cfg.resend_from_email,
                    "to": [owner_email],
                    "subject": subject,
                    "html": body,
                },
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to send report email to %s: %s", owner_email, exc)


async def _send_ntfy(account_name: str, vendor_count: int) -> None:
    """Send ntfy notification about completed tenant report."""
    if not settings.alerts.ntfy_enabled:
        return

    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
    message = f"Weekly report generated for {account_name} ({vendor_count} vendors)"
    headers = {
        "Title": f"Tenant Report: {account_name}",
        "Priority": "default",
        "Tags": "chart_with_upwards_trend,b2b,report",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(ntfy_url, content=message, headers=headers)
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("ntfy failed for tenant report %s: %s", account_name, exc)


async def _send_rule_alerts(triggered_rules: list) -> None:
    """Send ntfy alerts for critical/high priority threshold rule triggers (P5-005)."""
    if not settings.alerts.ntfy_enabled:
        return

    # Group by vendor
    by_vendor: dict[str, list] = {}
    for t in triggered_rules:
        by_vendor.setdefault(t.vendor_name, []).append(t)

    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"

    for vendor, rules in by_vendor.items():
        lines = [f"- {t.rule.description} (actual: {t.actual_value:.1f})" for t in rules]
        message = f"{vendor}: {len(rules)} threshold alert(s)\n" + "\n".join(lines)
        headers = {
            "Title": f"B2B Alert: {vendor}",
            "Priority": "high",
            "Tags": "warning,b2b,threshold",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(ntfy_url, content=message, headers=headers)
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("ntfy rule alert failed for %s: %s", vendor, exc)
