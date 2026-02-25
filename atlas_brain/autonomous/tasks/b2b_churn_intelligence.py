"""
B2B churn intelligence: aggregate enriched review data, feed to LLM
for synthesis, persist intelligence products, and notify.

Runs weekly (default Sunday 9 PM). Produces 4 report types:
  - weekly_churn_feed: ranked companies showing churn intent
  - vendor_scorecard: per-vendor health metrics
  - displacement_report: competitive flow map
  - category_overview: cross-vendor trends

Handles its own LLM call, report persistence, churn_signals upserts,
and ntfy notification -- returns _skip_synthesis so the runner does
not double-synthesize.
"""

import asyncio
import json
import logging
from datetime import date, datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.b2b_churn_intelligence")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: weekly B2B churn intelligence."""
    cfg = settings.b2b_churn
    if not cfg.enabled or not cfg.intelligence_enabled:
        return {"_skip_synthesis": "B2B churn intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    window_days = cfg.intelligence_window_days
    min_reviews = cfg.intelligence_min_reviews
    urgency_threshold = cfg.high_churn_urgency_threshold
    today = date.today()

    # Gather all 5 data sources in parallel
    (
        vendor_scores, high_intent, competitive_disp,
        pain_dist, feature_gaps,
    ) = await asyncio.gather(
        _fetch_vendor_churn_scores(pool, window_days, min_reviews),
        _fetch_high_intent_companies(pool, urgency_threshold, window_days),
        _fetch_competitive_displacement(pool, window_days),
        _fetch_pain_distribution(pool, window_days),
        _fetch_feature_gaps(pool, window_days),
        return_exceptions=True,
    )

    # Convert exceptions to empty values
    if isinstance(vendor_scores, Exception):
        logger.warning("Vendor scores fetch failed: %s", vendor_scores)
        vendor_scores = []
    if isinstance(high_intent, Exception):
        logger.warning("High intent fetch failed: %s", high_intent)
        high_intent = []
    if isinstance(competitive_disp, Exception):
        logger.warning("Competitive displacement fetch failed: %s", competitive_disp)
        competitive_disp = []
    if isinstance(pain_dist, Exception):
        logger.warning("Pain distribution fetch failed: %s", pain_dist)
        pain_dist = []
    if isinstance(feature_gaps, Exception):
        logger.warning("Feature gaps fetch failed: %s", feature_gaps)
        feature_gaps = []

    # Check if there's enough data
    if not vendor_scores and not high_intent:
        return {"_skip_synthesis": "No enriched B2B reviews to analyze"}

    # Fetch prior reports for trend comparison
    prior_reports = await _fetch_prior_reports(pool)

    # Build payload
    payload = {
        "date": str(today),
        "analysis_window_days": window_days,
        "vendor_churn_scores": vendor_scores,
        "high_intent_companies": high_intent,
        "competitive_displacement": competitive_disp,
        "pain_distribution": pain_dist,
        "feature_gaps": feature_gaps,
        "prior_reports": prior_reports,
    }

    # Load skill and call LLM
    from ...pipelines.llm import call_llm_with_skill, parse_json_response

    analysis = call_llm_with_skill(
        "digest/b2b_churn_intelligence", payload,
        max_tokens=cfg.intelligence_max_tokens, temperature=0.4,
    )
    if not analysis:
        return {"_skip_synthesis": "LLM analysis failed"}

    parsed = parse_json_response(analysis, recover_truncated=True)

    # Persist intelligence reports
    report_types = [
        ("weekly_churn_feed", parsed.get("weekly_churn_feed", [])),
        ("vendor_scorecard", parsed.get("vendor_scorecards", [])),
        ("displacement_report", parsed.get("displacement_map", [])),
        ("category_overview", parsed.get("category_insights", [])),
    ]

    for report_type, data in report_types:
        try:
            await pool.execute(
                """
                INSERT INTO b2b_intelligence (
                    report_date, report_type, intelligence_data,
                    executive_summary, data_density, status, llm_model
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                today,
                report_type,
                json.dumps(data, default=str),
                parsed.get("executive_summary", ""),
                json.dumps({
                    "vendors_analyzed": len(vendor_scores),
                    "high_intent_companies": len(high_intent),
                    "competitive_flows": len(competitive_disp),
                    "pain_categories": len(pain_dist),
                    "feature_gaps": len(feature_gaps),
                }),
                "published",
                "pipeline_default",
            )
        except Exception:
            logger.exception("Failed to store %s report", report_type)

    # Upsert per-vendor churn signals
    await _upsert_churn_signals(pool, vendor_scores, parsed)

    # Send ntfy notification
    await _send_notification(task, parsed, high_intent)

    return {
        "_skip_synthesis": "B2B churn intelligence complete",
        "date": str(today),
        "vendors_analyzed": len(vendor_scores),
        "high_intent_companies": len(high_intent),
        "competitive_flows": len(competitive_disp),
        "report_types": len(report_types),
    }


# ------------------------------------------------------------------
# Data fetchers
# ------------------------------------------------------------------


async def _fetch_vendor_churn_scores(pool, window_days: int, min_reviews: int) -> list[dict[str, Any]]:
    """Per-vendor health metrics from enriched reviews."""
    rows = await pool.fetch(
        """
        SELECT vendor_name, product_category,
            count(*) AS total_reviews,
            count(*) FILTER (
                WHERE (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
            ) AS churn_intent,
            avg((enrichment->>'urgency_score')::numeric) AS avg_urgency,
            avg(rating / NULLIF(rating_max, 0)) AS avg_rating_normalized,
            count(*) FILTER (
                WHERE (enrichment->>'would_recommend')::boolean = true
            ) AS recommend_yes,
            count(*) FILTER (
                WHERE (enrichment->>'would_recommend')::boolean = false
            ) AS recommend_no
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name, product_category
        HAVING count(*) >= $2
        ORDER BY avg((enrichment->>'urgency_score')::numeric) DESC
        """,
        window_days,
        min_reviews,
    )
    return [
        {
            "vendor_name": r["vendor_name"],
            "product_category": r["product_category"],
            "total_reviews": r["total_reviews"],
            "churn_intent": r["churn_intent"],
            "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
            "avg_rating_normalized": float(r["avg_rating_normalized"]) if r["avg_rating_normalized"] else None,
            "recommend_yes": r["recommend_yes"],
            "recommend_no": r["recommend_no"],
        }
        for r in rows
    ]


async def _fetch_high_intent_companies(pool, urgency_threshold: int, window_days: int) -> list[dict[str, Any]]:
    """Companies showing high churn intent -- the money feed."""
    rows = await pool.fetch(
        """
        SELECT reviewer_company, vendor_name, product_category,
            enrichment->'reviewer_context'->>'role_level' AS role_level,
            (enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
            (enrichment->>'urgency_score')::numeric AS urgency,
            enrichment->>'pain_category' AS pain,
            enrichment->'competitors_mentioned' AS alternatives,
            enrichment->'quotable_phrases' AS quotes,
            enrichment->'contract_context'->>'contract_value_signal' AS value_signal
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (enrichment->>'urgency_score')::numeric >= $1
          AND reviewer_company IS NOT NULL AND reviewer_company != ''
          AND enriched_at > NOW() - make_interval(days => $2)
        ORDER BY (enrichment->>'urgency_score')::numeric DESC
        """,
        urgency_threshold,
        window_days,
    )
    return [
        {
            "company": r["reviewer_company"],
            "vendor": r["vendor_name"],
            "category": r["product_category"],
            "role_level": r["role_level"],
            "decision_maker": r["is_dm"],
            "urgency": float(r["urgency"]) if r["urgency"] else 0,
            "pain": r["pain"],
            "alternatives": json.loads(r["alternatives"]) if isinstance(r["alternatives"], str) else (r["alternatives"] or []),
            "quotes": json.loads(r["quotes"]) if isinstance(r["quotes"], str) else (r["quotes"] or []),
            "contract_signal": r["value_signal"],
        }
        for r in rows
    ]


async def _fetch_competitive_displacement(pool, window_days: int) -> list[dict[str, Any]]:
    """Who's winning from whom -- competitive flows."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            comp.value->>'name' AS competitor,
            comp.value->>'context' AS direction,
            count(*) AS mention_count
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements(enrichment->'competitors_mentioned') AS comp(value)
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name, comp.value->>'name', comp.value->>'context'
        ORDER BY mention_count DESC
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "competitor": r["competitor"],
            "direction": r["direction"],
            "mention_count": r["mention_count"],
        }
        for r in rows
    ]


async def _fetch_pain_distribution(pool, window_days: int) -> list[dict[str, Any]]:
    """What's driving churn per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            enrichment->>'pain_category' AS pain,
            count(*) AS complaint_count,
            avg((enrichment->>'urgency_score')::numeric) AS avg_urgency
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name, enrichment->>'pain_category'
        ORDER BY complaint_count DESC
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "pain": r["pain"],
            "complaint_count": r["complaint_count"],
            "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
        }
        for r in rows
    ]


async def _fetch_feature_gaps(pool, window_days: int) -> list[dict[str, Any]]:
    """Most-mentioned missing features per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            gap.value #>> '{}' AS feature_gap,
            count(*) AS mentions
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements(enrichment->'feature_gaps') AS gap(value)
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name, gap.value #>> '{}'
        HAVING count(*) >= 2
        ORDER BY mentions DESC
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "feature_gap": r["feature_gap"],
            "mentions": r["mentions"],
        }
        for r in rows
    ]


async def _fetch_prior_reports(pool) -> list[dict[str, Any]]:
    """Fetch most recent prior intelligence reports for trend comparison."""
    rows = await pool.fetch(
        """
        SELECT report_type, intelligence_data, executive_summary, report_date
        FROM b2b_intelligence
        WHERE report_type = 'weekly_churn_feed'
        ORDER BY report_date DESC
        LIMIT 2
        """,
    )
    return [
        {
            "report_type": r["report_type"],
            "report_date": str(r["report_date"]),
            "executive_summary": r["executive_summary"],
        }
        for r in rows
    ]


# ------------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------------


async def _upsert_churn_signals(pool, vendor_scores: list[dict], parsed: dict) -> None:
    """Upsert b2b_churn_signals with aggregated per-vendor metrics."""
    now = datetime.now(timezone.utc)

    # Extract pain and competitor data from parsed output
    scorecards = {
        (sc.get("vendor"), sc.get("category")): sc
        for sc in parsed.get("vendor_scorecards", [])
        if isinstance(sc, dict)
    }

    for vs in vendor_scores:
        vendor = vs["vendor_name"]
        category = vs.get("product_category")
        sc = scorecards.get((vendor, category), {})

        total = vs["total_reviews"]
        recommend_yes = vs.get("recommend_yes", 0)
        recommend_no = vs.get("recommend_no", 0)
        nps = ((recommend_yes - recommend_no) / total * 100) if total > 0 else None

        try:
            await pool.execute(
                """
                INSERT INTO b2b_churn_signals (
                    vendor_name, product_category,
                    total_reviews, negative_reviews, churn_intent_count,
                    avg_urgency_score, avg_rating_normalized, nps_proxy,
                    top_pain_categories, top_competitors, top_feature_gaps,
                    last_computed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (vendor_name, COALESCE(product_category, '')) DO UPDATE SET
                    total_reviews = EXCLUDED.total_reviews,
                    negative_reviews = EXCLUDED.negative_reviews,
                    churn_intent_count = EXCLUDED.churn_intent_count,
                    avg_urgency_score = EXCLUDED.avg_urgency_score,
                    avg_rating_normalized = EXCLUDED.avg_rating_normalized,
                    nps_proxy = EXCLUDED.nps_proxy,
                    top_pain_categories = EXCLUDED.top_pain_categories,
                    top_competitors = EXCLUDED.top_competitors,
                    top_feature_gaps = EXCLUDED.top_feature_gaps,
                    last_computed_at = EXCLUDED.last_computed_at
                """,
                vendor,
                category,
                total,
                0,  # negative_reviews computed separately if needed
                vs.get("churn_intent", 0),
                vs.get("avg_urgency", 0),
                vs.get("avg_rating_normalized"),
                nps,
                json.dumps(sc.get("top_pain", [])) if isinstance(sc.get("top_pain"), list) else "[]",
                json.dumps(sc.get("top_competitor_threat", [])) if isinstance(sc.get("top_competitor_threat"), list) else "[]",
                "[]",
                now,
            )
        except Exception:
            logger.exception("Failed to upsert churn signal for %s", vendor)


# ------------------------------------------------------------------
# Notification
# ------------------------------------------------------------------


async def _send_notification(task: ScheduledTask, parsed: dict, high_intent: list) -> None:
    """Send ntfy push notification with executive summary."""
    from ...pipelines.notify import send_pipeline_notification

    # Build a custom notification body for churn intelligence
    parts: list[str] = []

    summary = parsed.get("executive_summary", "")
    if summary:
        parts.append(summary.strip())

    # Top high-intent companies
    feed = parsed.get("weekly_churn_feed", [])
    if feed and isinstance(feed, list):
        items = []
        for entry in feed[:5]:
            if isinstance(entry, dict):
                company = entry.get("company", "Unknown")
                vendor = entry.get("vendor", "")
                urgency = entry.get("urgency", "?")
                pain = entry.get("pain", "")
                role = entry.get("reviewer_role", "")
                quote = entry.get("key_quote", "")
                line = f"- **{company}** ({role}) -- {vendor}, urgency {urgency}/10"
                if pain:
                    line += f"\n  Pain: {pain}"
                if quote:
                    line += f'\n  "{quote}"'
                items.append(line)
        if items:
            parts.append("\n**High-Intent Companies**\n" + "\n".join(items))

    message = "\n\n".join(parts) if parts else "Weekly churn intelligence report generated."

    high_count = len(high_intent)
    title = f"Atlas: Weekly Churn Feed ({high_count} high-intent compan{'y' if high_count == 1 else 'ies'})"

    await send_pipeline_notification(
        message, task,
        title=title,
        default_tags="brain,chart_with_downwards_trend",
    )
