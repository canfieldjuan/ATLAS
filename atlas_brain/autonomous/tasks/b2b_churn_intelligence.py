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


def _safe_json(value: Any, default: Any = None) -> Any:
    """Safely deserialize a JSON value, returning *default* on failure."""
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Malformed JSON in aggregation data: %.100r", value)
            return default
    return default


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
    neg_threshold = cfg.negative_review_threshold
    fg_min_mentions = cfg.feature_gap_min_mentions
    quote_min_urgency = cfg.quotable_phrase_min_urgency
    tl_limit = cfg.timeline_signals_limit
    prior_limit = cfg.prior_reports_limit
    today = date.today()

    # Gather all 16 data sources in parallel
    (
        vendor_scores, high_intent, competitive_disp,
        pain_dist, feature_gaps,
        negative_counts, price_rates, dm_rates,
        churning_companies, quotable_evidence,
        budget_signals, use_case_dist, sentiment_traj,
        buyer_auth, timeline_signals, competitor_reasons,
    ) = await asyncio.gather(
        _fetch_vendor_churn_scores(pool, window_days, min_reviews),
        _fetch_high_intent_companies(pool, urgency_threshold, window_days),
        _fetch_competitive_displacement(pool, window_days),
        _fetch_pain_distribution(pool, window_days),
        _fetch_feature_gaps(pool, window_days, min_mentions=fg_min_mentions),
        _fetch_negative_review_counts(pool, window_days, threshold=neg_threshold),
        _fetch_price_complaint_rates(pool, window_days),
        _fetch_dm_churn_rates(pool, window_days),
        _fetch_churning_companies(pool, window_days),
        _fetch_quotable_evidence(pool, window_days, min_urgency=quote_min_urgency),
        _fetch_budget_signals(pool, window_days),
        _fetch_use_case_distribution(pool, window_days),
        _fetch_sentiment_trajectory(pool, window_days),
        _fetch_buyer_authority_summary(pool, window_days),
        _fetch_timeline_signals(pool, window_days, limit=tl_limit),
        _fetch_competitor_reasons(pool, window_days),
        return_exceptions=True,
    )

    # Convert exceptions to empty values, track failures
    fetcher_failures = 0

    def _safe(val: Any, name: str) -> list:
        nonlocal fetcher_failures
        if isinstance(val, Exception):
            fetcher_failures += 1
            logger.error("%s fetch failed: %s", name, val, exc_info=val)
            return []
        return val

    vendor_scores = _safe(vendor_scores, "vendor_scores")
    high_intent = _safe(high_intent, "high_intent")
    competitive_disp = _safe(competitive_disp, "competitive_disp")
    pain_dist = _safe(pain_dist, "pain_dist")
    feature_gaps = _safe(feature_gaps, "feature_gaps")
    negative_counts = _safe(negative_counts, "negative_counts")
    price_rates = _safe(price_rates, "price_rates")
    dm_rates = _safe(dm_rates, "dm_rates")
    churning_companies = _safe(churning_companies, "churning_companies")
    quotable_evidence = _safe(quotable_evidence, "quotable_evidence")
    budget_signals = _safe(budget_signals, "budget_signals")
    use_case_dist = _safe(use_case_dist, "use_case_dist")
    sentiment_traj = _safe(sentiment_traj, "sentiment_traj")
    buyer_auth = _safe(buyer_auth, "buyer_auth")
    timeline_signals = _safe(timeline_signals, "timeline_signals")
    competitor_reasons = _safe(competitor_reasons, "competitor_reasons")

    # Check if there's enough data
    if not vendor_scores and not high_intent:
        return {"_skip_synthesis": "No enriched B2B reviews to analyze"}

    # Fetch prior reports for trend comparison
    prior_reports = await _fetch_prior_reports(pool, limit=prior_limit)

    # Build payload
    payload = {
        "date": str(today),
        "analysis_window_days": window_days,
        "vendor_churn_scores": vendor_scores,
        "high_intent_companies": high_intent,
        "competitive_displacement": competitive_disp,
        "pain_distribution": pain_dist,
        "feature_gaps": feature_gaps,
        "negative_review_counts": negative_counts,
        "price_complaint_rates": price_rates,
        "decision_maker_churn_rates": dm_rates,
        "budget_signal_summary": budget_signals,
        "use_case_distribution": use_case_dist,
        "sentiment_trajectory_distribution": sentiment_traj,
        "buyer_authority_summary": buyer_auth,
        "timeline_signals": timeline_signals,
        "competitor_reasons": competitor_reasons,
        "prior_reports": prior_reports,
    }

    # Load skill and call LLM (synchronous -- run in thread to avoid blocking)
    from ...pipelines.llm import call_llm_with_skill, parse_json_response

    try:
        analysis = await asyncio.wait_for(
            asyncio.to_thread(
                call_llm_with_skill,
                "digest/b2b_churn_intelligence", payload,
                max_tokens=cfg.intelligence_max_tokens, temperature=0.4,
            ),
            timeout=300,
        )
    except asyncio.TimeoutError:
        logger.error("LLM call timed out after 300s for b2b_churn_intelligence")
        return {"_skip_synthesis": "LLM analysis timed out"}
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

    data_density = json.dumps({
        "vendors_analyzed": len(vendor_scores),
        "high_intent_companies": len(high_intent),
        "competitive_flows": len(competitive_disp),
        "pain_categories": len(pain_dist),
        "feature_gaps": len(feature_gaps),
    })
    exec_summary = parsed.get("executive_summary", "")

    try:
        async with pool.transaction() as conn:
            for report_type, data in report_types:
                await conn.execute(
                    """
                    INSERT INTO b2b_intelligence (
                        report_date, report_type, intelligence_data,
                        executive_summary, data_density, status, llm_model
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    today,
                    report_type,
                    json.dumps(data, default=str),
                    exec_summary,
                    data_density,
                    "published",
                    "pipeline_default",
                )
    except Exception:
        logger.exception("Failed to store intelligence reports (rolled back)")

    # Build lookups for upsert
    pain_lookup = _build_pain_lookup(pain_dist)
    competitor_lookup = _build_competitor_lookup(competitive_disp)
    feature_gap_lookup = _build_feature_gap_lookup(feature_gaps)
    neg_lookup = {r["vendor"]: r["negative_count"] for r in negative_counts}
    price_lookup = {r["vendor"]: r["price_complaint_rate"] for r in price_rates}
    dm_lookup = {r["vendor"]: r["dm_churn_rate"] for r in dm_rates}
    company_lookup = {r["vendor"]: r["companies"] for r in churning_companies}
    quote_lookup = {r["vendor"]: r["quotes"] for r in quotable_evidence}
    budget_lookup = {r["vendor"]: {k: v for k, v in r.items() if k != "vendor"} for r in budget_signals}
    use_case_lookup = _build_use_case_lookup(use_case_dist)
    integration_lookup = _build_integration_lookup(use_case_dist)
    sentiment_lookup = _build_sentiment_lookup(sentiment_traj)
    buyer_auth_lookup = _build_buyer_auth_lookup(buyer_auth)
    timeline_lookup = _build_timeline_lookup(timeline_signals)

    # Upsert per-vendor churn signals
    upsert_failures = await _upsert_churn_signals(
        pool, vendor_scores,
        neg_lookup, pain_lookup, competitor_lookup, feature_gap_lookup,
        price_lookup, dm_lookup, company_lookup, quote_lookup,
        budget_lookup, use_case_lookup, integration_lookup,
        sentiment_lookup, buyer_auth_lookup, timeline_lookup,
    )

    # Send ntfy notification
    await _send_notification(task, parsed, high_intent)

    # Emit reasoning events (no-op when reasoning disabled)
    await _emit_reasoning_events(parsed, high_intent, vendor_scores)

    return {
        "_skip_synthesis": "B2B churn intelligence complete",
        "date": str(today),
        "vendors_analyzed": len(vendor_scores),
        "high_intent_companies": len(high_intent),
        "competitive_flows": len(competitive_disp),
        "report_types": len(report_types),
        "fetcher_failures": fetcher_failures,
        "upsert_failures": upsert_failures,
    }


# ------------------------------------------------------------------
# Reasoning events
# ------------------------------------------------------------------


async def _emit_reasoning_events(
    parsed: dict[str, Any],
    high_intent: list[dict[str, Any]],
    vendor_scores: list[dict[str, Any]],
) -> None:
    """Emit B2B events for the reasoning agent (no-op when disabled)."""
    from ...reasoning.producers import emit_if_enabled
    from ...reasoning.events import EventType

    # One report-level event per run
    await emit_if_enabled(
        EventType.B2B_INTELLIGENCE_GENERATED,
        source="b2b_churn_intelligence",
        payload={
            "vendors_analyzed": len(vendor_scores),
            "high_intent_count": len(high_intent),
            "executive_summary": parsed.get("executive_summary", ""),
        },
    )

    # One event per high-intent company (cap at 10)
    for company in high_intent[:10]:
        await emit_if_enabled(
            EventType.B2B_HIGH_INTENT_DETECTED,
            source="b2b_churn_intelligence",
            payload={
                "company": company.get("company", ""),
                "vendor": company.get("vendor", ""),
                "urgency": company.get("urgency", 0),
                "pain": company.get("pain", ""),
                "alternatives": company.get("alternatives", []),
            },
            entity_type="company",
            entity_id=company.get("company", ""),
        )


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
            -- Source-weighted urgency: weighted avg preserving 0-10 scale
            -- Falls back to 0.7 for pre-existing reviews without source_weight
            avg(
                (enrichment->>'urgency_score')::numeric
                * COALESCE((raw_metadata->>'source_weight')::numeric, 0.7)
            ) / NULLIF(avg(COALESCE((raw_metadata->>'source_weight')::numeric, 0.7)), 0)
            AS avg_urgency,
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
    results = []
    for r in rows:
        try:
            urgency = float(r["urgency"]) if r["urgency"] is not None else 0
        except (ValueError, TypeError):
            urgency = 0
        results.append({
            "company": r["reviewer_company"],
            "vendor": r["vendor_name"],
            "category": r["product_category"],
            "role_level": r["role_level"],
            "decision_maker": r["is_dm"],
            "urgency": urgency,
            "pain": r["pain"],
            "alternatives": _safe_json(r["alternatives"]),
            "quotes": _safe_json(r["quotes"]),
            "contract_signal": r["value_signal"],
        })
    return results


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


async def _fetch_feature_gaps(pool, window_days: int, *, min_mentions: int = 2) -> list[dict[str, Any]]:
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
        HAVING count(*) >= $2
        ORDER BY mentions DESC
        """,
        window_days,
        min_mentions,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "feature_gap": r["feature_gap"],
            "mentions": r["mentions"],
        }
        for r in rows
    ]


async def _fetch_negative_review_counts(pool, window_days: int, *, threshold: float = 0.5) -> list[dict[str, Any]]:
    """Count reviews with below-threshold ratings per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name, count(*) AS negative_count
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
          AND rating IS NOT NULL AND rating_max > 0
          AND (rating / rating_max) < $2
        GROUP BY vendor_name
        """,
        window_days,
        threshold,
    )
    return [{"vendor": r["vendor_name"], "negative_count": r["negative_count"]} for r in rows]


async def _fetch_price_complaint_rates(pool, window_days: int) -> list[dict[str, Any]]:
    """Fraction of reviews with pain_category='pricing' per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            count(*) FILTER (WHERE enrichment->>'pain_category' = 'pricing') AS pricing_count,
            count(*) AS total
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name
        HAVING count(*) > 0
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "price_complaint_rate": r["pricing_count"] / r["total"] if r["total"] else 0,
        }
        for r in rows
    ]


async def _fetch_dm_churn_rates(pool, window_days: int) -> list[dict[str, Any]]:
    """Decision-maker churn rate: DMs with intent_to_leave / total DMs, per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            count(*) FILTER (
                WHERE (enrichment->'reviewer_context'->>'decision_maker')::boolean = true
                  AND (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
            ) AS dm_churning,
            count(*) FILTER (
                WHERE (enrichment->'reviewer_context'->>'decision_maker')::boolean = true
            ) AS dm_total
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name
        HAVING count(*) FILTER (
            WHERE (enrichment->'reviewer_context'->>'decision_maker')::boolean = true
        ) > 0
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "dm_churn_rate": r["dm_churning"] / r["dm_total"] if r["dm_total"] else 0,
        }
        for r in rows
    ]


async def _fetch_churning_companies(pool, window_days: int) -> list[dict[str, Any]]:
    """Companies with high churn intent, aggregated per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            jsonb_agg(jsonb_build_object(
                'company', reviewer_company,
                'urgency', (enrichment->>'urgency_score')::numeric,
                'role', enrichment->'reviewer_context'->>'role_level',
                'pain', enrichment->>'pain_category'
            ) ORDER BY (enrichment->>'urgency_score')::numeric DESC)
            AS companies
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
          AND (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
          AND reviewer_company IS NOT NULL AND reviewer_company != ''
        GROUP BY vendor_name
        """,
        window_days,
    )
    results = []
    for r in rows:
        companies = _safe_json(r["companies"])
        results.append({"vendor": r["vendor_name"], "companies": companies})
    return results


async def _fetch_quotable_evidence(pool, window_days: int, *, min_urgency: float = 6) -> list[dict[str, Any]]:
    """High-urgency quotable phrases per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            jsonb_agg(phrase.value ORDER BY (enrichment->>'urgency_score')::numeric DESC)
            AS quotes
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements_text(
            COALESCE(enrichment->'quotable_phrases', '[]'::jsonb)
        ) AS phrase(value)
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
          AND (enrichment->>'urgency_score')::numeric >= $2
        GROUP BY vendor_name
        """,
        window_days,
        min_urgency,
    )
    results = []
    for r in rows:
        quotes = _safe_json(r["quotes"])
        results.append({"vendor": r["vendor_name"], "quotes": quotes})
    return results


async def _fetch_budget_signals(pool, window_days: int) -> list[dict[str, Any]]:
    """Aggregate budget signals: seat_count stats and price-increase mentions per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            avg(NULLIF(
                CASE WHEN enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                     THEN (enrichment->'budget_signals'->>'seat_count')::numeric END,
                0)) AS avg_seat_count,
            percentile_cont(0.5) WITHIN GROUP (
                ORDER BY NULLIF(
                    CASE WHEN enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                         THEN (enrichment->'budget_signals'->>'seat_count')::numeric END,
                    0)
            ) AS median_seat_count,
            max(NULLIF(
                CASE WHEN enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                     THEN (enrichment->'budget_signals'->>'seat_count')::numeric END,
                0)) AS max_seat_count,
            count(*) FILTER (
                WHERE (enrichment->'budget_signals'->>'price_increase_mentioned')::boolean = true
            ) AS price_increase_count,
            count(*) AS total
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
          AND enrichment->'budget_signals' IS NOT NULL
          AND enrichment->'budget_signals' != 'null'::jsonb
        GROUP BY vendor_name
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "avg_seat_count": float(r["avg_seat_count"]) if r["avg_seat_count"] else None,
            "median_seat_count": float(r["median_seat_count"]) if r["median_seat_count"] else None,
            "max_seat_count": float(r["max_seat_count"]) if r["max_seat_count"] else None,
            "price_increase_count": r["price_increase_count"],
            "price_increase_rate": r["price_increase_count"] / r["total"] if r["total"] else 0,
        }
        for r in rows
    ]


async def _fetch_use_case_distribution(pool, window_days: int) -> list[dict[str, Any]]:
    """Explode use_case modules and integration stacks, count per vendor."""
    module_rows = await pool.fetch(
        """
        SELECT vendor_name,
            mod.value #>> '{}' AS module_name,
            count(*) AS mentions
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements(
            COALESCE(enrichment->'use_case'->'modules_mentioned', '[]'::jsonb)
        ) AS mod(value)
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name, mod.value #>> '{}'
        HAVING count(*) >= 2
        ORDER BY mentions DESC
        """,
        window_days,
    )
    stack_rows = await pool.fetch(
        """
        SELECT vendor_name,
            tool.value #>> '{}' AS tool_name,
            count(*) AS mentions
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements(
            COALESCE(enrichment->'use_case'->'integration_stack', '[]'::jsonb)
        ) AS tool(value)
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name, tool.value #>> '{}'
        HAVING count(*) >= 2
        ORDER BY mentions DESC
        """,
        window_days,
    )
    lock_rows = await pool.fetch(
        """
        SELECT vendor_name,
            enrichment->'use_case'->>'lock_in_level' AS lock_in_level,
            count(*) AS cnt
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
          AND enrichment->'use_case'->>'lock_in_level' IS NOT NULL
        GROUP BY vendor_name, enrichment->'use_case'->>'lock_in_level'
        ORDER BY cnt DESC
        """,
        window_days,
    )
    return [
        {"type": "modules", "data": [dict(r) for r in module_rows]},
        {"type": "stacks", "data": [dict(r) for r in stack_rows]},
        {"type": "lock_in", "data": [dict(r) for r in lock_rows]},
    ]


async def _fetch_sentiment_trajectory(pool, window_days: int) -> list[dict[str, Any]]:
    """Count reviews per sentiment direction per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            enrichment->'sentiment_trajectory'->>'direction' AS direction,
            count(*) AS cnt
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
          AND enrichment->'sentiment_trajectory'->>'direction' IS NOT NULL
        GROUP BY vendor_name, enrichment->'sentiment_trajectory'->>'direction'
        ORDER BY cnt DESC
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "direction": r["direction"],
            "count": r["cnt"],
        }
        for r in rows
    ]


async def _fetch_buyer_authority_summary(pool, window_days: int) -> list[dict[str, Any]]:
    """Count reviews per role_type and buying_stage per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            enrichment->'buyer_authority'->>'role_type' AS role_type,
            enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
            count(*) AS cnt
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
          AND enrichment->'buyer_authority' IS NOT NULL
          AND enrichment->'buyer_authority' != 'null'::jsonb
        GROUP BY vendor_name,
            enrichment->'buyer_authority'->>'role_type',
            enrichment->'buyer_authority'->>'buying_stage'
        ORDER BY cnt DESC
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "role_type": r["role_type"],
            "buying_stage": r["buying_stage"],
            "count": r["cnt"],
        }
        for r in rows
    ]


async def _fetch_timeline_signals(pool, window_days: int, *, limit: int = 50) -> list[dict[str, Any]]:
    """Extract reviews with non-null contract_end or evaluation_deadline -- hottest leads."""
    rows = await pool.fetch(
        """
        SELECT reviewer_company, vendor_name,
            enrichment->'timeline'->>'contract_end' AS contract_end,
            enrichment->'timeline'->>'evaluation_deadline' AS evaluation_deadline,
            enrichment->'timeline'->>'decision_timeline' AS decision_timeline,
            (enrichment->>'urgency_score')::numeric AS urgency
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
          AND (
              enrichment->'timeline'->>'contract_end' IS NOT NULL
              OR enrichment->'timeline'->>'evaluation_deadline' IS NOT NULL
          )
        ORDER BY (enrichment->>'urgency_score')::numeric DESC
        LIMIT $2
        """,
        window_days,
        limit,
    )
    return [
        {
            "company": r["reviewer_company"],
            "vendor": r["vendor_name"],
            "contract_end": r["contract_end"],
            "evaluation_deadline": r["evaluation_deadline"],
            "decision_timeline": r["decision_timeline"],
            "urgency": float(r["urgency"]) if r["urgency"] else 0,
        }
        for r in rows
    ]


async def _fetch_competitor_reasons(pool, window_days: int) -> list[dict[str, Any]]:
    """Explode competitors_mentioned and extract reason alongside name/context."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            comp.value->>'name' AS competitor,
            comp.value->>'context' AS direction,
            comp.value->>'reason' AS reason,
            count(*) AS mention_count
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements(enrichment->'competitors_mentioned') AS comp(value)
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
          AND comp.value->>'reason' IS NOT NULL
        GROUP BY vendor_name, comp.value->>'name', comp.value->>'context', comp.value->>'reason'
        ORDER BY mention_count DESC
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "competitor": r["competitor"],
            "direction": r["direction"],
            "reason": r["reason"],
            "mention_count": r["mention_count"],
        }
        for r in rows
    ]


async def _fetch_prior_reports(pool, *, limit: int = 4) -> list[dict[str, Any]]:
    """Fetch most recent prior intelligence reports for trend comparison.

    Includes both weekly_churn_feed and vendor_scorecard, with full
    intelligence_data so the LLM can compute trends from actual numbers
    instead of guessing from prose.
    """
    rows = await pool.fetch(
        """
        SELECT report_type, intelligence_data, executive_summary, report_date
        FROM b2b_intelligence
        WHERE report_type IN ('weekly_churn_feed', 'vendor_scorecard')
        ORDER BY report_date DESC
        LIMIT $1
        """,
        limit,
    )
    results = []
    for r in rows:
        intel_data = r["intelligence_data"]
        # asyncpg auto-deserializes JSONB to dict/list, but handle string fallback
        if isinstance(intel_data, str):
            try:
                intel_data = json.loads(intel_data)
            except (json.JSONDecodeError, TypeError):
                intel_data = {}
        results.append({
            "report_type": r["report_type"],
            "report_date": str(r["report_date"]),
            "executive_summary": r["executive_summary"],
            "intelligence_data": intel_data,
        })
    return results


# ------------------------------------------------------------------
# Public aggregation entry point (reused by b2b_tenant_report)
# ------------------------------------------------------------------


def _vendor_match(value: str, vendor_set: set[str]) -> bool:
    """Case-insensitive ILIKE-style match: vendor_set entry is contained in value."""
    vl = value.lower()
    return any(name in vl for name in vendor_set)


def _filter_by_vendors(data: list[dict], vendor_names: list[str]) -> list[dict]:
    """Post-filter fetcher results to only include rows matching vendor_names.

    Handles two structures:
    - Flat dicts with a ``vendor`` / ``vendor_name`` key (most fetchers)
    - Nested dicts like ``use_case_distribution`` with ``{"type": ..., "data": [...]}``
      where vendor data lives inside ``data[*]["vendor_name"]``
    """
    lowered = {v.lower() for v in vendor_names}
    filtered = []
    for row in data:
        vn = row.get("vendor") or row.get("vendor_name") or ""
        if vn:
            # Standard flat row
            if _vendor_match(vn, lowered):
                filtered.append(row)
        elif "data" in row and isinstance(row["data"], list):
            # Nested structure (use_case_distribution): filter inner data
            inner = [r for r in row["data"] if _vendor_match(
                r.get("vendor_name") or r.get("vendor") or "", lowered
            )]
            if inner:
                filtered.append({**row, "data": inner})
        # else: no vendor key at all, skip row
    return filtered


async def gather_intelligence_data(
    pool,
    window_days: int = 30,
    min_reviews: int = 3,
    vendor_names: list[str] | None = None,
) -> dict[str, Any]:
    """Gather all 16 intelligence data sources, optionally scoped to vendors.

    Returns the same payload dict that the LLM expects. Used by both
    the global ``run()`` handler and per-tenant report generation.
    """
    cfg = settings.b2b_churn
    urgency_threshold = cfg.high_churn_urgency_threshold
    neg_threshold = cfg.negative_review_threshold
    fg_min_mentions = cfg.feature_gap_min_mentions
    quote_min_urgency = cfg.quotable_phrase_min_urgency
    tl_limit = cfg.timeline_signals_limit
    prior_limit = cfg.prior_reports_limit

    results = await asyncio.gather(
        _fetch_vendor_churn_scores(pool, window_days, min_reviews),
        _fetch_high_intent_companies(pool, urgency_threshold, window_days),
        _fetch_competitive_displacement(pool, window_days),
        _fetch_pain_distribution(pool, window_days),
        _fetch_feature_gaps(pool, window_days, min_mentions=fg_min_mentions),
        _fetch_negative_review_counts(pool, window_days, threshold=neg_threshold),
        _fetch_price_complaint_rates(pool, window_days),
        _fetch_dm_churn_rates(pool, window_days),
        _fetch_churning_companies(pool, window_days),
        _fetch_quotable_evidence(pool, window_days, min_urgency=quote_min_urgency),
        _fetch_budget_signals(pool, window_days),
        _fetch_use_case_distribution(pool, window_days),
        _fetch_sentiment_trajectory(pool, window_days),
        _fetch_buyer_authority_summary(pool, window_days),
        _fetch_timeline_signals(pool, window_days, limit=tl_limit),
        _fetch_competitor_reasons(pool, window_days),
        return_exceptions=True,
    )

    names = [
        "vendor_scores", "high_intent", "competitive_disp", "pain_dist",
        "feature_gaps", "negative_counts", "price_rates", "dm_rates",
        "churning_companies", "quotable_evidence", "budget_signals",
        "use_case_dist", "sentiment_traj", "buyer_auth",
        "timeline_signals", "competitor_reasons",
    ]

    fetcher_failures = 0
    data = {}
    for name, val in zip(names, results):
        if isinstance(val, Exception):
            fetcher_failures += 1
            logger.error("%s fetch failed: %s", name, val, exc_info=val)
            data[name] = []
        else:
            data[name] = val

    # Post-filter by vendor names if scoped
    if vendor_names:
        for key in data:
            if isinstance(data[key], list) and data[key] and isinstance(data[key][0], dict):
                data[key] = _filter_by_vendors(data[key], vendor_names)

    prior_reports = await _fetch_prior_reports(pool, limit=prior_limit)

    payload = {
        "date": str(date.today()),
        "analysis_window_days": window_days,
        "vendor_churn_scores": data["vendor_scores"],
        "high_intent_companies": data["high_intent"],
        "competitive_displacement": data["competitive_disp"],
        "pain_distribution": data["pain_dist"],
        "feature_gaps": data["feature_gaps"],
        "negative_review_counts": data["negative_counts"],
        "price_complaint_rates": data["price_rates"],
        "decision_maker_churn_rates": data["dm_rates"],
        "churning_companies": data["churning_companies"],
        "quotable_evidence": data["quotable_evidence"],
        "budget_signal_summary": data["budget_signals"],
        "use_case_distribution": data["use_case_dist"],
        "sentiment_trajectory_distribution": data["sentiment_traj"],
        "buyer_authority_summary": data["buyer_auth"],
        "timeline_signals": data["timeline_signals"],
        "competitor_reasons": data["competitor_reasons"],
        "prior_reports": prior_reports,
    }

    return {
        "payload": payload,
        "fetcher_failures": fetcher_failures,
        "vendors_analyzed": len(data["vendor_scores"]),
        "high_intent_companies": len(data["high_intent"]),
        "competitive_flows": len(data["competitive_disp"]),
        "pain_categories": len(data["pain_dist"]),
        "feature_gaps": len(data["feature_gaps"]),
    }


# ------------------------------------------------------------------
# Lookup builders (pure Python, no DB)
# ------------------------------------------------------------------


def _build_pain_lookup(pain_dist: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {category, count, avg_urgency}."""
    lookup: dict[str, list[dict]] = {}
    for row in pain_dist:
        vendor = row.get("vendor", "")
        lookup.setdefault(vendor, []).append({
            "category": row.get("pain", "other"),
            "count": row.get("complaint_count", 0),
            "avg_urgency": round(row.get("avg_urgency", 0), 1),
        })
    for v in lookup:
        lookup[v].sort(key=lambda x: x["count"], reverse=True)
    return lookup


def _build_competitor_lookup(competitive_disp: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {name, direction, mentions}."""
    lookup: dict[str, list[dict]] = {}
    for row in competitive_disp:
        vendor = row.get("vendor", "")
        lookup.setdefault(vendor, []).append({
            "name": row.get("competitor", ""),
            "direction": row.get("direction", ""),
            "mentions": row.get("mention_count", 0),
        })
    for v in lookup:
        lookup[v].sort(key=lambda x: x["mentions"], reverse=True)
    return lookup


def _build_feature_gap_lookup(feature_gaps: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {feature, mentions}."""
    lookup: dict[str, list[dict]] = {}
    for row in feature_gaps:
        vendor = row.get("vendor", "")
        lookup.setdefault(vendor, []).append({
            "feature": row.get("feature_gap", ""),
            "mentions": row.get("mentions", 0),
        })
    for v in lookup:
        lookup[v].sort(key=lambda x: x["mentions"], reverse=True)
    return lookup


def _build_use_case_lookup(use_case_dist: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {module, mentions}."""
    lookup: dict[str, list[dict]] = {}
    for entry in use_case_dist:
        if entry.get("type") != "modules":
            continue
        for row in entry.get("data", []):
            vendor = row.get("vendor_name", "")
            lookup.setdefault(vendor, []).append({
                "module": row.get("module_name", ""),
                "mentions": row.get("mentions", 0),
            })
    for v in lookup:
        lookup[v].sort(key=lambda x: x["mentions"], reverse=True)
    return lookup


def _build_integration_lookup(use_case_dist: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {tool, mentions}."""
    lookup: dict[str, list[dict]] = {}
    for entry in use_case_dist:
        if entry.get("type") != "stacks":
            continue
        for row in entry.get("data", []):
            vendor = row.get("vendor_name", "")
            lookup.setdefault(vendor, []).append({
                "tool": row.get("tool_name", ""),
                "mentions": row.get("mentions", 0),
            })
    for v in lookup:
        lookup[v].sort(key=lambda x: x["mentions"], reverse=True)
    return lookup


def _build_sentiment_lookup(sentiment_traj: list[dict]) -> dict[str, dict[str, int]]:
    """vendor -> {direction: count}."""
    lookup: dict[str, dict[str, int]] = {}
    for row in sentiment_traj:
        vendor = row.get("vendor", "")
        direction = row.get("direction", "unknown")
        lookup.setdefault(vendor, {})[direction] = row.get("count", 0)
    return lookup


def _build_buyer_auth_lookup(buyer_auth: list[dict]) -> dict[str, dict]:
    """vendor -> {role_types: {type: count}, buying_stages: {stage: count}}."""
    lookup: dict[str, dict] = {}
    for row in buyer_auth:
        vendor = row.get("vendor", "")
        if vendor not in lookup:
            lookup[vendor] = {"role_types": {}, "buying_stages": {}}
        rt = row.get("role_type", "unknown")
        bs = row.get("buying_stage", "unknown")
        cnt = row.get("count", 0)
        lookup[vendor]["role_types"][rt] = lookup[vendor]["role_types"].get(rt, 0) + cnt
        lookup[vendor]["buying_stages"][bs] = lookup[vendor]["buying_stages"].get(bs, 0) + cnt
    return lookup


def _build_timeline_lookup(timeline_signals: list[dict]) -> dict[str, list[dict]]:
    """vendor -> list of timeline entries."""
    lookup: dict[str, list[dict]] = {}
    for row in timeline_signals:
        vendor = row.get("vendor", "")
        lookup.setdefault(vendor, []).append({
            "company": row.get("company"),
            "contract_end": row.get("contract_end"),
            "evaluation_deadline": row.get("evaluation_deadline"),
            "decision_timeline": row.get("decision_timeline"),
            "urgency": row.get("urgency", 0),
        })
    return lookup


# ------------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------------


async def _upsert_churn_signals(
    pool,
    vendor_scores: list[dict],
    neg_lookup: dict[str, int],
    pain_lookup: dict[str, list[dict]],
    competitor_lookup: dict[str, list[dict]],
    feature_gap_lookup: dict[str, list[dict]],
    price_lookup: dict[str, float],
    dm_lookup: dict[str, float],
    company_lookup: dict[str, list[dict]],
    quote_lookup: dict[str, list[str]],
    budget_lookup: dict[str, dict] | None = None,
    use_case_lookup: dict[str, list[dict]] | None = None,
    integration_lookup: dict[str, list[dict]] | None = None,
    sentiment_lookup: dict[str, dict[str, int]] | None = None,
    buyer_auth_lookup: dict[str, dict] | None = None,
    timeline_lookup: dict[str, list[dict]] | None = None,
) -> int:
    """Upsert b2b_churn_signals with all 21 columns. Returns failure count."""
    now = datetime.now(timezone.utc)
    budget_lookup = budget_lookup or {}
    use_case_lookup = use_case_lookup or {}
    integration_lookup = integration_lookup or {}
    sentiment_lookup = sentiment_lookup or {}
    buyer_auth_lookup = buyer_auth_lookup or {}
    timeline_lookup = timeline_lookup or {}
    failures = 0

    for vs in vendor_scores:
        vendor = vs["vendor_name"]
        category = vs.get("product_category")

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
                    price_complaint_rate, decision_maker_churn_rate,
                    company_churn_list, quotable_evidence,
                    top_use_cases, top_integration_stacks,
                    budget_signal_summary, sentiment_distribution,
                    buyer_authority_summary, timeline_summary,
                    last_computed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                          $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
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
                    price_complaint_rate = EXCLUDED.price_complaint_rate,
                    decision_maker_churn_rate = EXCLUDED.decision_maker_churn_rate,
                    company_churn_list = EXCLUDED.company_churn_list,
                    quotable_evidence = EXCLUDED.quotable_evidence,
                    top_use_cases = EXCLUDED.top_use_cases,
                    top_integration_stacks = EXCLUDED.top_integration_stacks,
                    budget_signal_summary = EXCLUDED.budget_signal_summary,
                    sentiment_distribution = EXCLUDED.sentiment_distribution,
                    buyer_authority_summary = EXCLUDED.buyer_authority_summary,
                    timeline_summary = EXCLUDED.timeline_summary,
                    last_computed_at = EXCLUDED.last_computed_at
                """,
                vendor,
                category,
                total,
                neg_lookup.get(vendor, 0),
                vs.get("churn_intent", 0),
                vs.get("avg_urgency", 0),
                vs.get("avg_rating_normalized"),
                nps,
                json.dumps(pain_lookup.get(vendor, [])[:5]),
                json.dumps(competitor_lookup.get(vendor, [])[:5]),
                json.dumps(feature_gap_lookup.get(vendor, [])[:5]),
                price_lookup.get(vendor),
                dm_lookup.get(vendor),
                json.dumps(company_lookup.get(vendor, [])[:20]),
                json.dumps(quote_lookup.get(vendor, [])[:10]),
                json.dumps(use_case_lookup.get(vendor, [])[:10]),
                json.dumps(integration_lookup.get(vendor, [])[:10]),
                json.dumps(budget_lookup.get(vendor, {})),
                json.dumps(sentiment_lookup.get(vendor, {})),
                json.dumps(buyer_auth_lookup.get(vendor, {})),
                json.dumps(timeline_lookup.get(vendor, [])[:10]),
                now,
            )
        except Exception:
            failures += 1
            logger.exception("Failed to upsert churn signal for %s", vendor)

    return failures


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


# ------------------------------------------------------------------
# Vendor-scoped intelligence report (P1: Vendor Retention)
# ------------------------------------------------------------------


async def generate_vendor_report(
    pool,
    vendor_name: str,
    window_days: int = 90,
) -> dict[str, Any] | None:
    """Generate a structured intelligence report for a specific vendor.

    Returns the report dict (also stored in b2b_intelligence) or None on failure.
    Called by the vendor_targets API or campaign generation pipeline.
    """
    today = date.today()

    # Fetch signals for this vendor
    rows = await pool.fetch(
        """
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
               r.enrichment->'sentiment_trajectory'->>'direction' AS sentiment_direction
        FROM b2b_reviews r
        WHERE r.enrichment_status = 'enriched'
          AND r.enriched_at > NOW() - make_interval(days => $1)
          AND r.vendor_name ILIKE '%' || $2 || '%'
          AND (r.enrichment->>'urgency_score')::numeric >= 3
        ORDER BY (r.enrichment->>'urgency_score')::numeric DESC
        LIMIT 500
        """,
        window_days,
        vendor_name,
    )

    if not rows:
        return None

    signals = []
    for r in rows:
        d = dict(r)
        d["urgency"] = float(d.get("urgency") or 0)
        comps = d.get("competitors_json")
        if isinstance(comps, str):
            try:
                comps = json.loads(comps)
            except (json.JSONDecodeError, TypeError):
                comps = []
        d["competitors"] = comps if isinstance(comps, list) else []
        signals.append(d)

    total = len(signals)
    high_urgency = [s for s in signals if s["urgency"] >= 8]
    medium_urgency = [s for s in signals if 5 <= s["urgency"] < 8]

    # Pain distribution
    pain_counts: dict[str, int] = {}
    for s in signals:
        pain = _safe_json(s.get("pain_json"))
        for p in pain:
            if isinstance(p, dict) and p.get("category"):
                pain_counts[p["category"]] = pain_counts.get(p["category"], 0) + 1

    # Competitive displacement
    comp_counts: dict[str, int] = {}
    for s in signals:
        for c in s["competitors"]:
            if isinstance(c, dict) and c.get("name"):
                comp_counts[c["name"]] = comp_counts.get(c["name"], 0) + 1

    # Feature gaps
    gap_counts: dict[str, int] = {}
    for s in signals:
        gaps = _safe_json(s.get("feature_gaps"))
        for g in gaps:
            label = g if isinstance(g, str) else (g.get("feature", "") if isinstance(g, dict) else "")
            if label:
                gap_counts[label] = gap_counts.get(label, 0) + 1

    # Anonymized quotes (high-urgency only)
    anon_quotes: list[str] = []
    for s in high_urgency[:20]:
        phrases = _safe_json(s.get("quotable_phrases"))
        for phrase in phrases:
            text = phrase if isinstance(phrase, str) else (phrase.get("text", "") if isinstance(phrase, dict) else "")
            if text and text not in anon_quotes:
                anon_quotes.append(text)
            if len(anon_quotes) >= 10:
                break

    report_data = {
        "vendor_name": vendor_name,
        "report_date": str(today),
        "window_days": window_days,
        "signal_count": total,
        "high_urgency_count": len(high_urgency),
        "medium_urgency_count": len(medium_urgency),
        "pain_categories": sorted(
            [{"category": k, "count": v} for k, v in pain_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:10],
        "competitive_displacement": sorted(
            [{"competitor": k, "count": v} for k, v in comp_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:10],
        "top_feature_gaps": sorted(
            [{"feature": k, "count": v} for k, v in gap_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:10],
        "anonymized_quotes": anon_quotes[:10],
    }

    # Persist to b2b_intelligence
    try:
        await pool.execute(
            """
            INSERT INTO b2b_intelligence (
                report_date, report_type, vendor_filter,
                intelligence_data, executive_summary, data_density, status, llm_model
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            today,
            "vendor_retention",
            vendor_name,
            json.dumps(report_data, default=str),
            f"{total} accounts showing churn signals for {vendor_name}. "
            f"{len(high_urgency)} at critical urgency.",
            json.dumps({
                "signal_count": total,
                "pain_categories": len(pain_counts),
                "competitors": len(comp_counts),
                "feature_gaps": len(gap_counts),
            }),
            "published",
            "pipeline_aggregation",
        )
    except Exception:
        logger.exception("Failed to store vendor report for %s", vendor_name)

    return report_data


# ------------------------------------------------------------------
# Challenger-scoped intelligence report (P2: Challenger Intel)
# ------------------------------------------------------------------


async def generate_challenger_report(
    pool,
    challenger_name: str,
    window_days: int = 90,
) -> dict[str, Any] | None:
    """Generate a structured intelligence report for a challenger target.

    Queries reviews where *challenger_name* appears in the enrichment
    ``competitors_mentioned`` array (i.e. reviewers of *other* vendors
    who are considering switching to this challenger).

    Returns the report dict (also stored in b2b_intelligence) or None
    when no matching signals exist.
    """
    today = date.today()

    rows = await pool.fetch(
        """
        SELECT r.id AS review_id, r.vendor_name, r.reviewer_company, r.product_category,
               (r.enrichment->>'urgency_score')::numeric AS urgency,
               (r.enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
               r.enrichment->'buyer_authority'->>'role_type' AS role_type,
               r.enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
               CASE WHEN r.enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                    THEN (r.enrichment->'budget_signals'->>'seat_count')::int END AS seat_count,
               r.enrichment->'competitors_mentioned' AS competitors_json,
               r.enrichment->'pain_categories' AS pain_json,
               r.enrichment->'quotable_phrases' AS quotable_phrases,
               r.enrichment->'feature_gaps' AS feature_gaps
        FROM b2b_reviews r
        WHERE r.enrichment_status = 'enriched'
          AND r.enriched_at > NOW() - make_interval(days => $1)
          AND (r.enrichment->>'urgency_score')::numeric >= 3
          AND EXISTS (
                SELECT 1 FROM jsonb_array_elements(r.enrichment->'competitors_mentioned') AS comp(value)
                WHERE comp.value->>'name' ILIKE '%' || $2 || '%'
              )
        ORDER BY (r.enrichment->>'urgency_score')::numeric DESC
        LIMIT 500
        """,
        window_days,
        challenger_name,
    )

    if not rows:
        return None

    signals = []
    for r in rows:
        d = dict(r)
        d["urgency"] = float(d.get("urgency") or 0)
        comps = d.get("competitors_json")
        if isinstance(comps, str):
            try:
                comps = json.loads(comps)
            except (json.JSONDecodeError, TypeError):
                comps = []
        d["competitors"] = comps if isinstance(comps, list) else []
        signals.append(d)

    total = len(signals)
    high_urgency = [s for s in signals if s["urgency"] >= 8]
    medium_urgency = [s for s in signals if 5 <= s["urgency"] < 8]

    # Buying stage distribution
    stage_counts: dict[str, int] = {}
    for s in signals:
        stage = s.get("buying_stage")
        if stage:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

    by_buying_stage = {
        "active_purchase": stage_counts.get("active_purchase", 0),
        "evaluation": stage_counts.get("evaluation", 0),
        "renewal_decision": stage_counts.get("renewal_decision", 0),
    }

    # Role distribution
    role_counts: dict[str, int] = {}
    for s in signals:
        role = s.get("role_type")
        if role:
            role_counts[role] = role_counts.get(role, 0) + 1

    # Pain driving switch
    pain_counts: dict[str, int] = {}
    for s in signals:
        pain = _safe_json(s.get("pain_json"))
        for p in pain:
            if isinstance(p, dict) and p.get("category"):
                pain_counts[p["category"]] = pain_counts.get(p["category"], 0) + 1

    # Incumbents losing (the vendor_name on each review is the incumbent)
    incumbent_counts: dict[str, int] = {}
    for s in signals:
        vname = s.get("vendor_name")
        if vname:
            incumbent_counts[vname] = incumbent_counts.get(vname, 0) + 1

    # Seat count distribution
    large = mid = small = 0
    for s in signals:
        sc = s.get("seat_count")
        if sc is not None:
            if sc >= 500:
                large += 1
            elif sc >= 100:
                mid += 1
            else:
                small += 1

    # Incumbent feature gaps (what incumbents are missing)
    gap_counts: dict[str, int] = {}
    for s in signals:
        gaps = _safe_json(s.get("feature_gaps"))
        for g in gaps:
            label = g if isinstance(g, str) else (g.get("feature", "") if isinstance(g, dict) else "")
            if label:
                gap_counts[label] = gap_counts.get(label, 0) + 1

    # Feature mentions (challenger features reviewers cite)
    feature_set: list[str] = []
    for s in signals:
        for c in s["competitors"]:
            if isinstance(c, dict):
                cname = (c.get("name") or "").lower()
                if cname and challenger_name.lower() in cname:
                    for feat in c.get("features", []):
                        if isinstance(feat, str) and feat not in feature_set:
                            feature_set.append(feat)

    # Anonymized quotes (high-urgency only)
    anon_quotes: list[str] = []
    for s in high_urgency[:20]:
        phrases = _safe_json(s.get("quotable_phrases"))
        for phrase in phrases:
            text = phrase if isinstance(phrase, str) else (phrase.get("text", "") if isinstance(phrase, dict) else "")
            if text and text not in anon_quotes:
                anon_quotes.append(text)
            if len(anon_quotes) >= 10:
                break

    report_data = {
        "challenger_name": challenger_name,
        "report_date": str(today),
        "window_days": window_days,
        "signal_count": total,
        "high_urgency_count": len(high_urgency),
        "medium_urgency_count": len(medium_urgency),
        "by_buying_stage": by_buying_stage,
        "role_distribution": sorted(
            [{"role": k, "count": v} for k, v in role_counts.items()],
            key=lambda x: x["count"], reverse=True,
        ),
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
        "incumbent_feature_gaps": sorted(
            [{"feature": k, "count": v} for k, v in gap_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:10],
        "feature_mentions": feature_set[:20],
        "anonymized_quotes": anon_quotes[:10],
    }

    # Persist to b2b_intelligence
    try:
        await pool.execute(
            """
            INSERT INTO b2b_intelligence (
                report_date, report_type, vendor_filter,
                intelligence_data, executive_summary, data_density, status, llm_model
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            today,
            "challenger_intel",
            challenger_name,
            json.dumps(report_data, default=str),
            f"{total} accounts mentioning {challenger_name} as alternative. "
            f"{len(high_urgency)} at critical urgency.",
            json.dumps({
                "signal_count": total,
                "buying_stages": len(stage_counts),
                "incumbents": len(incumbent_counts),
                "feature_gaps": len(gap_counts),
            }),
            "published",
            "pipeline_aggregation",
        )
    except Exception:
        logger.exception("Failed to store challenger report for %s", challenger_name)

    return report_data
