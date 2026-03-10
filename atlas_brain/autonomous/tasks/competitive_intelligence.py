"""
Complaint vulnerability intelligence: cross-brand analysis from deep-extracted reviews.

Aggregates deep_extraction JSONB fields across brands to produce competitive
flow maps, feature gap rankings, buyer persona clusters, and brand vulnerability
scores. Source data is complaint/negative reviews only -- scores measure
vulnerability and dissatisfaction, not overall brand health.

Runs daily (default 9:30 PM, after complaint_analysis at 9 PM). Handles its
own LLM call, report persistence, brand_intelligence upserts, and ntfy
notification -- returns _skip_synthesis so the runner does not double-synthesize.
"""

import asyncio
import json
import logging
import math
import re
import uuid as _uuid
from datetime import date, datetime, timezone
from typing import Any

from ...config import settings
from ...services.brand_registry import resolve_brand_name_cached, _ensure_cache as _ensure_brand_cache
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.competitive_intelligence")

# Category filter expression matching blog topic selection (categories->>2 granular)
_CAT_EXPR = (
    "COALESCE(REPLACE(pm.categories->>2, '&amp;', '&'),"
    " REPLACE(pm.categories->>1, '&amp;', '&'), pr.source_category)"
)


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: daily competitive intelligence."""
    cfg = settings.external_data
    if not cfg.complaint_mining_enabled or not cfg.competitive_intelligence_enabled:
        return {"_skip_synthesis": "Competitive intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    today = date.today()

    # Skip if we already have a report for today
    existing = await pool.fetchrow(
        "SELECT id FROM market_intelligence_reports "
        "WHERE report_date = $1 AND report_type = 'daily_competitive' LIMIT 1",
        today,
    )
    if existing:
        return {"_skip_synthesis": f"Report already exists for {today}"}

    # Check minimum deep-enriched count
    count_row = await pool.fetchrow(
        "SELECT count(*) AS cnt FROM product_reviews "
        "WHERE deep_enrichment_status = 'enriched'"
    )
    total_deep = count_row["cnt"] if count_row else 0
    if total_deep < cfg.competitive_intelligence_min_deep_enriched:
        return {
            "_skip_synthesis": f"Only {total_deep} deep-enriched reviews "
            f"(need {cfg.competitive_intelligence_min_deep_enriched})"
        }

    # Verify product_metadata table exists (populated by match_product_metadata script)
    has_metadata = await pool.fetchrow(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.tables "
        "  WHERE table_name = 'product_metadata'"
        ") AS ok"
    )
    if not has_metadata or not has_metadata["ok"]:
        return {"_skip_synthesis": "product_metadata table not found (run match_product_metadata first)"}

    # Gather 9 data sources in parallel
    (
        brand_health,
        competitive_flows,
        feature_gaps,
        buyer_personas,
        sentiment_landscape,
        safety_signals,
        loyalty_churn,
        prior_reports,
        data_context,
    ) = await asyncio.gather(
        _fetch_brand_health(pool),
        _fetch_competitive_flows(pool),
        _fetch_feature_gaps(pool),
        _fetch_buyer_personas(pool),
        _fetch_sentiment_landscape(pool),
        _fetch_safety_signals(pool),
        _fetch_loyalty_churn(pool),
        _fetch_prior_reports(pool),
        _fetch_data_context(pool),
        return_exceptions=True,
    )

    # Convert exceptions to empty values
    fetchers = {
        "brand_health": brand_health,
        "competitive_flows": competitive_flows,
        "feature_gaps": feature_gaps,
        "buyer_personas": buyer_personas,
        "sentiment_landscape": sentiment_landscape,
        "safety_signals": safety_signals,
        "loyalty_churn": loyalty_churn,
        "prior_reports": prior_reports,
    }
    for key, val in fetchers.items():
        if isinstance(val, Exception):
            logger.warning("%s fetch failed: %s", key, val)
            fetchers[key] = []
    if isinstance(data_context, Exception):
        logger.warning("Data context fetch failed: %s", data_context)
        data_context = {}

    if not fetchers["brand_health"]:
        return {"_skip_synthesis": "No brand data to analyze"}

    # Post-process: normalize feature requests
    # (competitive_flows already normalized by fetch_competitive_flows)
    if fetchers["feature_gaps"]:
        fetchers["feature_gaps"] = _normalize_feature_requests(fetchers["feature_gaps"])

    # Build LLM payload -- trim to fit ~4k token input budget (8k context - 4k output).
    # Full fetchers data used below for upserts.
    llm_brands = [
        {
            "brand": b["brand"],
            "reviews": b["total_reviews"],
            "period": b.get("review_period", ""),
            "rating": b["avg_rating"],
            "pain": b["avg_pain_score"],
            "repurchase": f"{b['repurchase_yes']}/{b['repurchase_yes'] + b['repurchase_no']}",
            "safety": b["safety_flagged_count"],
        }
        for b in fetchers["brand_health"][:15]
    ]
    payload = {
        "date": str(today),
        "data_context": data_context,
        "total_brands": len(fetchers["brand_health"]),
        "brand_health": llm_brands,
        "competitive_flows": fetchers["competitive_flows"][:10],
        "feature_gaps": fetchers["feature_gaps"][:8],
        "buyer_personas": fetchers["buyer_personas"][:6],
        "safety_signals": fetchers["safety_signals"][:6],
        "loyalty_churn": fetchers["loyalty_churn"][:8],
    }

    # Load skill and call LLM
    from ...pipelines.llm import call_llm_with_skill, parse_json_response

    try:
        analysis = await asyncio.wait_for(
            asyncio.to_thread(
                call_llm_with_skill,
                "digest/competitive_intelligence",
                payload,
                max_tokens=cfg.competitive_intelligence_max_tokens,
                temperature=0.4,
                workload="synthesis",
                response_format={"type": "json_object"},
            ),
            timeout=300,
        )
    except asyncio.TimeoutError:
        logger.error("LLM call timed out after 300s for competitive_intelligence")
        return {"_skip_synthesis": "LLM analysis timed out"}
    if not analysis:
        return {"_skip_synthesis": "LLM analysis failed"}

    parsed = parse_json_response(analysis, recover_truncated=True)

    # Persist to market_intelligence_reports
    report_stored = False
    try:
        await pool.execute(
            """
            INSERT INTO market_intelligence_reports (
                report_date, report_type, analysis_text,
                competitive_flows, feature_gaps, buyer_personas,
                brand_scorecards, insights, recommendations
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            today,
            "daily_competitive",
            parsed.get("analysis_text", analysis),
            json.dumps(parsed.get("competitive_flows", [])),
            json.dumps(parsed.get("feature_gaps", [])),
            json.dumps(parsed.get("buyer_personas", [])),
            json.dumps(parsed.get("brand_vulnerability", parsed.get("brand_scorecards", []))),
            json.dumps(parsed.get("insights", [])),
            json.dumps(parsed.get("recommendations", [])),
        )
        report_stored = True
        logger.info("Stored competitive intelligence report for %s", today)
    except Exception:
        logger.exception("Failed to store competitive intelligence report")

    # Upsert brand_intelligence scorecards (only if report stored successfully)
    snapshots_persisted = 0
    change_events_detected = 0
    displacement_edges_persisted = 0
    if report_stored:
        await _upsert_brand_intelligence(
            pool, fetchers["brand_health"], parsed,
            safety_signals=fetchers["safety_signals"],
            loyalty_churn=fetchers["loyalty_churn"],
        )
        # Persist daily brand snapshots + detect change events
        snapshots_persisted = await _persist_brand_snapshots(
            pool, fetchers["brand_health"], parsed,
        )
        change_events_detected = await _detect_change_events(
            pool, fetchers["brand_health"],
        )
        # Detect market-level concurrent shifts (3+ brands with same event today)
        concurrent_shifts = await _detect_concurrent_shifts(pool)
        displacement_edges_persisted = await _persist_displacement_edges(pool)

        # Dispatch consumer webhooks for change events + report
        await _dispatch_consumer_events(
            pool, fetchers["brand_health"], change_events_detected, concurrent_shifts,
        )

    # Send ntfy notification
    from ...pipelines.notify import send_pipeline_notification

    await send_pipeline_notification(
        parsed.get("analysis_text", analysis),
        task,
        title="Atlas: Complaint Vulnerability Intelligence",
        default_tags="brain,bar_chart",
        parsed=parsed,
    )

    return {
        "_skip_synthesis": "Competitive intelligence complete",
        "date": str(today),
        "brands_analyzed": len(fetchers["brand_health"]),
        "competitive_flows": len(fetchers["competitive_flows"]),
        "feature_gaps": len(fetchers["feature_gaps"]),
        "insights": len(parsed.get("insights", [])),
        "snapshots_persisted": snapshots_persisted,
        "change_events_detected": change_events_detected,
        "concurrent_shifts_detected": concurrent_shifts,
        "displacement_edges_persisted": displacement_edges_persisted,
    }


# ------------------------------------------------------------------
# Data fetchers
# ------------------------------------------------------------------


async def _fetch_data_context(pool) -> dict[str, Any]:
    """Compute temporal metadata so the LLM can anchor claims with timeframes."""
    row = await pool.fetchrow(
        """
        SELECT
            count(*) AS total,
            count(*) FILTER (WHERE reviewed_at IS NOT NULL) AS with_date,
            min(reviewed_at) AS earliest,
            max(reviewed_at) AS latest,
            count(*) FILTER (WHERE reviewed_at >= NOW() - INTERVAL '1 year') AS last_1y,
            count(*) FILTER (WHERE reviewed_at >= NOW() - INTERVAL '3 years') AS last_3y,
            count(*) FILTER (WHERE reviewed_at < NOW() - INTERVAL '3 years') AS older_3y
        FROM product_reviews
        WHERE deep_enrichment_status = 'enriched'
        """
    )
    return {
        "total_deep_enriched": row["total"],
        "reviews_with_dates": row["with_date"],
        "review_period": {
            "earliest": str(row["earliest"].date()) if row["earliest"] else None,
            "latest": str(row["latest"].date()) if row["latest"] else None,
        },
        "recency": {
            "last_1_year": row["last_1y"],
            "last_3_years": row["last_3y"],
            "older_than_3_years": row["older_3y"],
        },
        "note": "IMPORTANT: Always anchor statistics with timeframes. Say 'between 2012 and 2023' or 'over the review period' instead of unqualified claims. Use the date range per brand when available.",
    }


async def _fetch_brand_health(pool, *, category: str | None = None) -> list[dict[str, Any]]:
    """Per-brand health: reviews, rating, pain, severity, repurchase, safety."""
    # Warm brand registry cache for sync resolution below
    await _ensure_brand_cache()

    cat_filter = ""
    params: list = []
    if category:
        cat_filter = f"AND {_CAT_EXPR} = $1"
        params = [category]
    rows = await pool.fetch(
        f"""
        SELECT
            pm.brand,
            count(*) AS total_reviews,
            avg(pr.rating) AS avg_rating,
            avg(pr.pain_score) AS avg_pain_score,
            count(*) FILTER (WHERE pr.severity = 'critical') AS critical_count,
            count(*) FILTER (WHERE pr.severity = 'major') AS major_count,
            count(*) FILTER (WHERE pr.severity = 'minor') AS minor_count,
            count(*) FILTER (
                WHERE (pr.deep_extraction->>'would_repurchase')::boolean IS TRUE
            ) AS repurchase_yes,
            count(*) FILTER (
                WHERE (pr.deep_extraction->>'would_repurchase')::boolean IS FALSE
            ) AS repurchase_no,
            count(*) FILTER (
                WHERE (pr.deep_extraction->'safety_flag'->>'flagged')::boolean IS TRUE
            ) AS safety_flagged_count,
            min(pr.reviewed_at)::date AS earliest_review,
            max(pr.reviewed_at)::date AS latest_review
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.deep_enrichment_status = 'enriched'
          AND pm.brand IS NOT NULL AND pm.brand != ''
          {cat_filter}
        GROUP BY pm.brand
        HAVING count(*) >= 5
        ORDER BY count(*) DESC
        """,
        *params,
    )
    return [
        {
            "brand": resolve_brand_name_cached(r["brand"]),
            "total_reviews": r["total_reviews"],
            "review_period": f"{r['earliest_review']} to {r['latest_review']}" if r["earliest_review"] else "dates unavailable",
            "avg_rating": round(float(r["avg_rating"]), 2) if r["avg_rating"] else 0.0,
            "avg_pain_score": round(float(r["avg_pain_score"]), 1) if r["avg_pain_score"] else 0.0,
            "severity_distribution": {
                "critical": r["critical_count"],
                "major": r["major_count"],
                "minor": r["minor_count"],
            },
            "repurchase_yes": r["repurchase_yes"],
            "repurchase_no": r["repurchase_no"],
            "safety_flagged_count": r["safety_flagged_count"],
        }
        for r in rows
    ]


async def _fetch_competitive_flows(pool, *, category: str | None = None) -> list[dict[str, Any]]:
    """Brand-to-brand customer migration from product_comparisons."""
    from ...pipelines.comparisons import fetch_competitive_flows

    where = "pm.brand IS NOT NULL AND pm.brand != ''"
    params: list = []
    if category:
        where += f" AND {_CAT_EXPR} = $1"
        params = [category]
    return await fetch_competitive_flows(
        pool,
        where_clause=where,
        params=params,
        min_mentions=2,
        limit=500,
    )


async def _fetch_feature_gaps(pool, *, category: str | None = None) -> list[dict[str, Any]]:
    """Most-requested features across all products."""
    cat_filter = ""
    params: list = []
    if category:
        cat_filter = f"AND {_CAT_EXPR} = $1"
        params = [category]
    rows = await pool.fetch(
        f"""
        SELECT
            {_CAT_EXPR} AS category,
            feat AS feature,
            count(*) AS mentions,
            avg(pr.pain_score) AS avg_pain_score
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        CROSS JOIN jsonb_array_elements_text(pr.deep_extraction->'feature_requests') AS feat
        WHERE pr.deep_enrichment_status = 'enriched'
          AND jsonb_array_length(pr.deep_extraction->'feature_requests') > 0
          {cat_filter}
        GROUP BY category, feat
        HAVING count(*) >= 2
        ORDER BY count(*) DESC
        LIMIT 500
        """,
        *params,
    )
    return [
        {
            "category": r["category"],
            "feature": r["feature"],
            "mentions": r["mentions"],
            "avg_pain_score": round(float(r["avg_pain_score"]), 1) if r["avg_pain_score"] else 0.0,
        }
        for r in rows
    ]


async def _fetch_buyer_personas(pool, *, category: str | None = None) -> list[dict[str, Any]]:
    """Buyer segment clusters from buyer_context + expertise/budget."""
    cat_filter = ""
    params: list = []
    if category:
        cat_filter = f"AND {_CAT_EXPR} = $1"
        params = [category]
    rows = await pool.fetch(
        f"""
        SELECT
            {_CAT_EXPR} AS category,
            pr.deep_extraction->'buyer_context'->>'buyer_type' AS buyer_type,
            pr.deep_extraction->'buyer_context'->>'use_case' AS use_case,
            pr.deep_extraction->'buyer_context'->>'price_sentiment' AS price_sentiment,
            pr.deep_extraction->>'expertise_level' AS expertise_level,
            pr.deep_extraction->>'budget_type' AS budget_type,
            count(*) AS review_count,
            avg(pr.rating) AS avg_rating,
            avg(pr.pain_score) AS avg_pain
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.deep_enrichment_status = 'enriched'
          AND pr.deep_extraction->'buyer_context' IS NOT NULL
          {cat_filter}
        GROUP BY
            category,
            pr.deep_extraction->'buyer_context'->>'buyer_type',
            pr.deep_extraction->'buyer_context'->>'use_case',
            pr.deep_extraction->'buyer_context'->>'price_sentiment',
            pr.deep_extraction->>'expertise_level',
            pr.deep_extraction->>'budget_type'
        HAVING count(*) >= 3
        ORDER BY count(*) DESC
        LIMIT 500
        """,
        *params,
    )
    return [
        {
            "category": r["category"],
            "buyer_type": r["buyer_type"],
            "use_case": r["use_case"],
            "price_sentiment": r["price_sentiment"],
            "expertise_level": r["expertise_level"],
            "budget_type": r["budget_type"],
            "review_count": r["review_count"],
            "avg_rating": round(float(r["avg_rating"]), 2) if r["avg_rating"] else 0.0,
            "avg_pain": round(float(r["avg_pain"]), 1) if r["avg_pain"] else 0.0,
        }
        for r in rows
    ]


async def _fetch_sentiment_landscape(pool, *, category: str | None = None) -> list[dict[str, Any]]:
    """Per-brand sentiment on specific aspects."""
    cat_filter = ""
    params: list = []
    if category:
        cat_filter = f"AND {_CAT_EXPR} = $1"
        params = [category]
    rows = await pool.fetch(
        f"""
        SELECT
            pm.brand,
            asp->>'aspect' AS aspect,
            asp->>'sentiment' AS sentiment,
            count(*) AS cnt
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        CROSS JOIN jsonb_array_elements(pr.deep_extraction->'sentiment_aspects') AS asp
        WHERE pr.deep_enrichment_status = 'enriched'
          AND pm.brand IS NOT NULL AND pm.brand != ''
          AND jsonb_array_length(pr.deep_extraction->'sentiment_aspects') > 0
          {cat_filter}
        GROUP BY pm.brand, asp->>'aspect', asp->>'sentiment'
        ORDER BY count(*) DESC
        LIMIT 500
        """,
        *params,
    )
    return [
        {
            "brand": r["brand"],
            "aspect": r["aspect"],
            "sentiment": r["sentiment"],
            "count": r["cnt"],
        }
        for r in rows
    ]


async def _fetch_safety_signals(pool, *, category: str | None = None) -> list[dict[str, Any]]:
    """Per-brand safety-flagged reviews with consequence_severity breakdown."""
    cat_filter = ""
    params: list = []
    if category:
        cat_filter = f"AND {_CAT_EXPR} = $1"
        params = [category]
    rows = await pool.fetch(
        f"""
        SELECT
            pm.brand,
            pr.deep_extraction->>'consequence_severity' AS consequence,
            count(*) AS cnt
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.deep_enrichment_status = 'enriched'
          AND (pr.deep_extraction->'safety_flag'->>'flagged')::boolean IS TRUE
          AND pm.brand IS NOT NULL AND pm.brand != ''
          {cat_filter}
        GROUP BY pm.brand, pr.deep_extraction->>'consequence_severity'
        ORDER BY count(*) DESC
        LIMIT 500
        """,
        *params,
    )
    return [
        {
            "brand": r["brand"],
            "consequence": r["consequence"],
            "count": r["cnt"],
        }
        for r in rows
    ]


async def _fetch_safety_products(pool, *, category: str | None = None) -> list[dict[str, Any]]:
    """Top products (ASINs) by safety-flag count with product title."""
    cat_filter = ""
    params: list = []
    if category:
        cat_filter = f"AND {_CAT_EXPR} = $1"
        params = [category]
    rows = await pool.fetch(
        f"""
        SELECT
            pm.asin,
            pm.title AS product_title,
            pm.brand,
            count(*) AS safety_flags,
            round(avg(pr.pain_score), 1) AS avg_pain
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.deep_enrichment_status = 'enriched'
          AND (pr.deep_extraction->'safety_flag'->>'flagged')::boolean IS TRUE
          AND pm.brand IS NOT NULL AND pm.brand != ''
          {cat_filter}
        GROUP BY pm.asin, pm.title, pm.brand
        ORDER BY count(*) DESC
        LIMIT 15
        """,
        *params,
    )
    return [
        {
            "asin": r["asin"],
            "product_title": r["product_title"],
            "brand": r["brand"],
            "safety_flags": r["safety_flags"],
            "avg_pain": float(r["avg_pain"]) if r["avg_pain"] else None,
        }
        for r in rows
    ]


async def _fetch_loyalty_churn(pool, *, category: str | None = None) -> list[dict[str, Any]]:
    """Per-brand loyalty depth x replacement behavior cross-tab."""
    cat_filter = ""
    params: list = []
    if category:
        cat_filter = f"AND {_CAT_EXPR} = $1"
        params = [category]
    rows = await pool.fetch(
        f"""
        SELECT
            pm.brand,
            pr.deep_extraction->>'brand_loyalty_depth' AS loyalty,
            pr.deep_extraction->>'replacement_behavior' AS replacement,
            count(*) AS cnt
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.deep_enrichment_status = 'enriched'
          AND pm.brand IS NOT NULL AND pm.brand != ''
          {cat_filter}
        GROUP BY pm.brand,
            pr.deep_extraction->>'brand_loyalty_depth',
            pr.deep_extraction->>'replacement_behavior'
        HAVING count(*) >= 2
        ORDER BY count(*) DESC
        LIMIT 500
        """,
        *params,
    )
    return [
        {
            "brand": r["brand"],
            "loyalty": r["loyalty"],
            "replacement": r["replacement"],
            "count": r["cnt"],
        }
        for r in rows
    ]


async def _fetch_prior_reports(pool, limit: int = 3) -> list[dict[str, Any]]:
    """Fetch prior market_intelligence_reports for trend context."""
    rows = await pool.fetch(
        """
        SELECT report_date, analysis_text,
               competitive_flows, feature_gaps, buyer_personas,
               brand_scorecards, insights, recommendations
        FROM market_intelligence_reports
        WHERE report_type = 'daily_competitive'
        ORDER BY report_date DESC
        LIMIT $1
        """,
        limit,
    )
    result = []
    for r in rows:
        entry: dict[str, Any] = {
            "report_date": str(r["report_date"]),
            "analysis_text": (r["analysis_text"] or "")[:1000],
        }
        for field in (
            "competitive_flows", "feature_gaps", "buyer_personas",
            "brand_scorecards", "insights", "recommendations",
        ):
            val = r[field]
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    val = []
            entry[field] = val if isinstance(val, list) else []
        result.append(entry)
    return result


# ------------------------------------------------------------------
# Post-processing normalization
# ------------------------------------------------------------------


# _normalize_competitors and _COMPETITOR_NOISE moved to atlas_brain.pipelines.comparisons


def _normalize_feature_requests(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lowercase dedup + merge counts for near-identical feature requests."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for gap in gaps:
        feat_raw = (gap.get("feature") or "").strip()
        if not feat_raw:
            continue

        # Normalize: lowercase, collapse whitespace
        feat_key = re.sub(r"\s+", " ", feat_raw.lower())
        category = gap.get("category", "")
        key = (category, feat_key)

        if key in merged:
            merged[key]["mentions"] += gap.get("mentions", 0)
            # Keep the higher pain score
            existing_pain = merged[key].get("avg_pain_score", 0.0)
            new_pain = gap.get("avg_pain_score", 0.0)
            merged[key]["avg_pain_score"] = max(existing_pain, new_pain)
        else:
            merged[key] = {
                "category": category,
                "feature": feat_raw.strip(),  # Keep original casing for display
                "mentions": gap.get("mentions", 0),
                "avg_pain_score": gap.get("avg_pain_score", 0.0),
            }

    result = sorted(merged.values(), key=lambda x: x["mentions"], reverse=True)
    return result


# ------------------------------------------------------------------
# Confidence scoring
# ------------------------------------------------------------------


def _compute_consumer_confidence(
    mention_count: int,
    deep_enriched_count: int,
    total_count: int,
    severity_counts: dict[str, int] | None = None,
) -> float:
    """Evidence-based confidence score for consumer entities (0.0-1.0).

    Three equally-weighted signals (each 0-1, averaged):
      1. mention_weight:     log-scaled mention count (caps at 50)
      2. enrichment_weight:  proportion of reviews that are deep-enriched
      3. severity_weight:    consistency of severity signals (entropy-based)

    Adapted from B2B ``_compute_evidence_confidence()`` which uses
    mention weight + source diversity + verified source proportion.
    Consumer has a single source (Amazon), so source signals are replaced
    with enrichment depth and severity consistency.
    """
    mention_weight = min(
        math.log2(max(mention_count, 1)) / math.log2(50), 1.0
    )

    enrichment_weight = (
        deep_enriched_count / total_count if total_count > 0 else 0.0
    )

    severity_weight = 0.5
    if severity_counts:
        vals = [severity_counts.get(s, 0) for s in ("critical", "major", "minor")]
        total_sev = sum(vals)
        if total_sev > 0:
            probs = [v / total_sev for v in vals if v > 0]
            entropy = -sum(p * math.log2(p) for p in probs)
            max_entropy = math.log2(3)
            severity_weight = 1.0 - (entropy / max_entropy)

    score = (mention_weight + enrichment_weight + severity_weight) / 3.0
    return round(max(0.0, min(1.0, score)), 2)


# ------------------------------------------------------------------
# Brand intelligence upserts
# ------------------------------------------------------------------


def _compute_vulnerability_score(brand_data: dict) -> float:
    """Composite vulnerability score 0-100. Higher = more vulnerable.

    Formula: (1 - repurchase_rate) * 30 + pain/10 * 30 + (5 - rating)/5 * 15
             + churn_signal * 10 + safety_rate * 15
    Where churn_signal = proportion of reviews with would_repurchase=false out of
    total with any signal, and safety_rate = safety_flagged_count / total_reviews.
    Weights: 30 + 30 + 15 + 10 + 15 = 100.
    """
    yes = brand_data.get("repurchase_yes", 0)
    no = brand_data.get("repurchase_no", 0)
    total_signal = yes + no
    repurchase_rate = yes / total_signal if total_signal > 0 else 0.5
    churn_signal = no / total_signal if total_signal > 0 else 0.5

    pain = brand_data.get("avg_pain_score", 5.0)
    rating = brand_data.get("avg_rating", 3.0)

    total_reviews = brand_data.get("total_reviews", 0)
    safety_flagged = brand_data.get("safety_flagged_count", 0)
    safety_rate = safety_flagged / total_reviews if total_reviews > 0 else 0.0

    score = (
        (1 - repurchase_rate) * 30
        + pain / 10 * 30
        + (5 - rating) / 5 * 15
        + churn_signal * 10
        + safety_rate * 15
    )
    return max(0.0, min(100.0, round(score, 2)))


async def _upsert_brand_intelligence(
    pool,
    brand_health: list[dict[str, Any]],
    parsed: dict[str, Any],
    safety_signals: list[dict[str, Any]] | None = None,
    loyalty_churn: list[dict[str, Any]] | None = None,
) -> None:
    """Upsert brand_intelligence from aggregated brand stats + LLM scorecards."""
    now = datetime.now(timezone.utc)
    upserted = 0

    # Build lookup from LLM-generated scorecards (new key: brand_vulnerability)
    raw_scorecards = parsed.get("brand_vulnerability", []) or parsed.get("brand_scorecards", [])
    scorecards = {
        sc["brand"]: sc
        for sc in raw_scorecards
        if isinstance(sc, dict) and sc.get("brand")
    }

    # Build competitive flows per brand
    flows_by_brand: dict[str, list] = {}
    for flow in parsed.get("competitive_flows", []):
        if isinstance(flow, dict):
            brand = flow.get("from_brand", "")
            if brand:
                flows_by_brand.setdefault(brand, []).append(flow)

    # Build brand-keyed safety signal lookups
    safety_by_brand: dict[str, dict[str, int]] = {}
    for sig in (safety_signals or []):
        b = sig.get("brand", "")
        if b:
            safety_by_brand.setdefault(b, {})[sig.get("consequence") or "unknown"] = sig.get("count", 0)

    # Build brand-keyed loyalty/replacement lookups
    loyalty_by_brand: dict[str, dict[str, int]] = {}
    replacement_by_brand: dict[str, dict[str, int]] = {}
    for row in (loyalty_churn or []):
        b = row.get("brand", "")
        if not b:
            continue
        cnt = row.get("count", 0)
        loy = row.get("loyalty") or "unknown"
        rep = row.get("replacement") or "unknown"
        loyalty_by_brand.setdefault(b, {})
        loyalty_by_brand[b][loy] = loyalty_by_brand[b].get(loy, 0) + cnt
        replacement_by_brand.setdefault(b, {})
        replacement_by_brand[b][rep] = replacement_by_brand[b].get(rep, 0) + cnt

    for brand_data in brand_health:
        brand = brand_data.get("brand")
        if not brand:
            continue

        scorecard = scorecards.get(brand, {})
        health = _compute_vulnerability_score(brand_data)

        # Merge safety signals into sentiment_breakdown
        sentiment = dict(scorecard.get("sentiment_breakdown", {}))
        brand_safety = safety_by_brand.get(brand)
        if brand_safety:
            sentiment["safety_signals"] = brand_safety

        # Merge loyalty/replacement into buyer_profile
        buyer = dict(scorecard.get("buyer_profile", {}))
        brand_loyalty = loyalty_by_brand.get(brand)
        if brand_loyalty:
            buyer["loyalty_distribution"] = brand_loyalty
        brand_replacement = replacement_by_brand.get(brand)
        if brand_replacement:
            buyer["replacement_distribution"] = brand_replacement

        source_review_count = brand_data.get("total_reviews", 0)
        source_dist = json.dumps({"amazon": source_review_count})
        brand_confidence = _compute_consumer_confidence(
            mention_count=brand_data.get("total_reviews", 0),
            deep_enriched_count=brand_data.get("total_reviews", 0),
            total_count=brand_data.get("total_reviews", 0),
            severity_counts=brand_data.get("severity_distribution"),
        )

        try:
            await pool.execute(
                """
                INSERT INTO brand_intelligence (
                    brand, source, total_reviews, avg_rating, avg_pain_score,
                    repurchase_yes, repurchase_no,
                    sentiment_breakdown, top_feature_requests, top_complaints,
                    competitive_flows, buyer_profile, positive_aspects,
                    health_score, last_computed_at,
                    source_review_count, source_distribution, confidence_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17::jsonb, $18)
                ON CONFLICT (brand, source) DO UPDATE SET
                    total_reviews = EXCLUDED.total_reviews,
                    avg_rating = EXCLUDED.avg_rating,
                    avg_pain_score = EXCLUDED.avg_pain_score,
                    repurchase_yes = EXCLUDED.repurchase_yes,
                    repurchase_no = EXCLUDED.repurchase_no,
                    sentiment_breakdown = EXCLUDED.sentiment_breakdown,
                    top_feature_requests = EXCLUDED.top_feature_requests,
                    top_complaints = EXCLUDED.top_complaints,
                    competitive_flows = EXCLUDED.competitive_flows,
                    buyer_profile = EXCLUDED.buyer_profile,
                    positive_aspects = EXCLUDED.positive_aspects,
                    health_score = EXCLUDED.health_score,
                    last_computed_at = EXCLUDED.last_computed_at,
                    source_review_count = EXCLUDED.source_review_count,
                    source_distribution = EXCLUDED.source_distribution,
                    confidence_score = EXCLUDED.confidence_score
                """,
                brand,
                "all",
                brand_data.get("total_reviews", 0),
                brand_data.get("avg_rating"),
                brand_data.get("avg_pain_score"),
                brand_data.get("repurchase_yes", 0),
                brand_data.get("repurchase_no", 0),
                json.dumps(sentiment),
                json.dumps(scorecard.get("top_feature_requests", [])),
                json.dumps(scorecard.get("top_complaints", [])),
                json.dumps(flows_by_brand.get(brand, [])),
                json.dumps(buyer),
                json.dumps(scorecard.get("positive_aspects", [])),
                health,
                now,
                source_review_count,
                source_dist,
                brand_confidence,
            )
            upserted += 1
        except Exception:
            logger.warning("Failed to upsert brand intelligence for %s", brand, exc_info=True)

    if upserted:
        logger.info("Upserted %d brand intelligence scorecards", upserted)


async def _persist_brand_snapshots(
    pool,
    brand_health: list[dict[str, Any]],
    parsed: dict[str, Any],
) -> int:
    """Persist daily brand health snapshots (append-only)."""
    today = date.today()
    persisted = 0

    scorecards = {
        sc["brand"]: sc
        for sc in (parsed.get("brand_vulnerability", []) or parsed.get("brand_scorecards", []))
        if isinstance(sc, dict) and sc.get("brand")
    }

    # Fetch trajectory data from materialized view (if it exists)
    trajectory_by_brand: dict[str, tuple[int, int]] = {}
    try:
        traj_rows = await pool.fetch(
            "SELECT brand, trajectory_positive, trajectory_negative "
            "FROM mv_brand_summary WHERE brand IS NOT NULL"
        )
        for tr in traj_rows:
            trajectory_by_brand[tr["brand"]] = (
                tr["trajectory_positive"] or 0,
                tr["trajectory_negative"] or 0,
            )
    except Exception:
        logger.debug("mv_brand_summary not available for trajectory data", exc_info=True)

    for bd in brand_health:
        brand = bd.get("brand")
        if not brand:
            continue
        sc = scorecards.get(brand, {})
        top_complaints = sc.get("top_complaints", [])
        top_features = sc.get("top_feature_requests", [])
        flows = sc.get("competitive_flows") or parsed.get("competitive_flows", [])
        flow_count = sum(1 for f in flows if isinstance(f, dict) and f.get("from_brand") == brand)

        try:
            await pool.execute(
                """
                INSERT INTO brand_intelligence_snapshots (
                    brand, snapshot_date, total_reviews, avg_rating,
                    avg_pain_score, health_score,
                    repurchase_yes, repurchase_no,
                    complaint_count, safety_count,
                    top_complaint, top_feature_request,
                    competitive_flow_count,
                    trajectory_positive, trajectory_negative
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                ON CONFLICT (brand, snapshot_date) DO UPDATE SET
                    total_reviews = EXCLUDED.total_reviews,
                    avg_rating = EXCLUDED.avg_rating,
                    avg_pain_score = EXCLUDED.avg_pain_score,
                    health_score = EXCLUDED.health_score,
                    repurchase_yes = EXCLUDED.repurchase_yes,
                    repurchase_no = EXCLUDED.repurchase_no,
                    complaint_count = EXCLUDED.complaint_count,
                    safety_count = EXCLUDED.safety_count,
                    top_complaint = EXCLUDED.top_complaint,
                    top_feature_request = EXCLUDED.top_feature_request,
                    competitive_flow_count = EXCLUDED.competitive_flow_count,
                    trajectory_positive = EXCLUDED.trajectory_positive,
                    trajectory_negative = EXCLUDED.trajectory_negative
                """,
                brand, today,
                bd.get("total_reviews", 0),
                bd.get("avg_rating"),
                bd.get("avg_pain_score"),
                _compute_vulnerability_score(bd),
                bd.get("repurchase_yes", 0),
                bd.get("repurchase_no", 0),
                sum(bd.get("severity_distribution", {}).values()),
                bd.get("safety_flagged_count", 0),
                top_complaints[0].get("complaint", "") if top_complaints else None,
                top_features[0].get("feature", "") if top_features else None,
                flow_count,
                trajectory_by_brand.get(brand, (0, 0))[0],
                trajectory_by_brand.get(brand, (0, 0))[1],
            )
            persisted += 1
        except Exception:
            logger.warning("Failed to persist brand snapshot for %s", brand, exc_info=True)

    if persisted:
        logger.info("Persisted %d brand intelligence snapshots for %s", persisted, today)
    return persisted


async def _detect_change_events(
    pool,
    brand_health: list[dict[str, Any]],
) -> int:
    """Compare today's brand metrics against prior snapshot; log anomalies."""
    today = date.today()
    detected = 0

    for bd in brand_health:
        brand = bd.get("brand")
        if not brand:
            continue

        # Fetch most recent prior snapshot
        prior = await pool.fetchrow(
            """
            SELECT avg_pain_score, health_score, safety_count,
                   repurchase_yes, repurchase_no, avg_rating
            FROM brand_intelligence_snapshots
            WHERE brand = $1 AND snapshot_date < $2
            ORDER BY snapshot_date DESC LIMIT 1
            """,
            brand, today,
        )
        if not prior:
            continue

        events: list[tuple[str, str, float | None, float | None, float | None]] = []
        cur_pain = float(bd.get("avg_pain_score") or 0)
        old_pain = float(prior["avg_pain_score"] or 0)

        # Pain score spike (>= 1.5 points)
        if old_pain > 0 and cur_pain - old_pain >= 1.5:
            events.append((
                "pain_score_spike",
                f"{brand} pain score spiked from {old_pain:.1f} to {cur_pain:.1f}",
                old_pain, cur_pain, cur_pain - old_pain,
            ))

        # Health score spike (vulnerability increase >= 10 pts on 0-100)
        cur_health = _compute_vulnerability_score(bd)
        old_health = float(prior["health_score"] or 0)
        if old_health > 0 and cur_health - old_health >= 10:
            events.append((
                "vulnerability_spike",
                f"{brand} vulnerability score rose from {old_health:.0f} to {cur_health:.0f}",
                old_health, cur_health, cur_health - old_health,
            ))

        # Safety signal emergence (new safety count > 0 when prior was 0)
        cur_safety = bd.get("safety_flagged_count", 0)
        old_safety = prior["safety_count"] or 0
        if cur_safety > 0 and old_safety == 0:
            events.append((
                "safety_flag_emergence",
                f"{brand} gained {cur_safety} safety-flagged reviews (was 0)",
                float(old_safety), float(cur_safety), float(cur_safety),
            ))

        # Repurchase rate decline (>= 15 percentage points)
        cur_yes = bd.get("repurchase_yes", 0)
        cur_no = bd.get("repurchase_no", 0)
        old_yes = prior["repurchase_yes"] or 0
        old_no = prior["repurchase_no"] or 0
        cur_rate = cur_yes / (cur_yes + cur_no) * 100 if (cur_yes + cur_no) > 0 else 0
        old_rate = old_yes / (old_yes + old_no) * 100 if (old_yes + old_no) > 0 else 0
        if old_rate > 0 and old_rate - cur_rate >= 15:
            events.append((
                "repurchase_decline",
                f"{brand} repurchase rate dropped from {old_rate:.0f}% to {cur_rate:.0f}%",
                old_rate, cur_rate, cur_rate - old_rate,
            ))

        # Rating drop (>= 0.5 stars)
        cur_rating = float(bd.get("avg_rating") or 0)
        old_rating = float(prior["avg_rating"] or 0)
        if old_rating > 0 and old_rating - cur_rating >= 0.5:
            events.append((
                "rating_drop",
                f"{brand} avg rating dropped from {old_rating:.2f} to {cur_rating:.2f}",
                old_rating, cur_rating, cur_rating - old_rating,
            ))

        for event_type, description, old_val, new_val, delta in events:
            try:
                await pool.execute(
                    """
                    INSERT INTO product_change_events
                        (brand, event_date, event_type, description,
                         old_value, new_value, delta)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    brand, today, event_type, description,
                    old_val, new_val, delta,
                )
                detected += 1
            except Exception:
                logger.warning("Failed to log change event for %s", brand, exc_info=True)

    if detected:
        logger.info("Detected %d consumer change events", detected)
    return detected


# ------------------------------------------------------------------
# Concurrent shift detection (market-level)
# ------------------------------------------------------------------


async def _detect_concurrent_shifts(pool) -> int:
    """Detect dates where 3+ brands had the same event type -- signals market trend."""
    today = date.today()
    detected = 0
    try:
        rows = await pool.fetch(
            """
            SELECT event_type, COUNT(DISTINCT brand) AS brand_count,
                   ARRAY_AGG(DISTINCT brand ORDER BY brand) AS brands,
                   AVG(delta) AS avg_delta
            FROM product_change_events
            WHERE event_date = $1
              AND brand != '__market__'
            GROUP BY event_type
            HAVING COUNT(DISTINCT brand) >= 3
            """,
            today,
        )
        for row in rows:
            event_type = row["event_type"]
            brand_count = row["brand_count"]
            brands = row["brands"]
            avg_delta = round(float(row["avg_delta"] or 0), 2)
            brand_list = ", ".join(brands[:5])
            suffix = f" +{brand_count - 5} more" if brand_count > 5 else ""
            description = (
                f"Concurrent {event_type} across {brand_count} brands: "
                f"{brand_list}{suffix} (avg delta: {avg_delta})"
            )
            try:
                await pool.execute(
                    """
                    INSERT INTO product_change_events
                        (brand, event_date, event_type, description, delta, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    """,
                    "__market__",
                    today,
                    "concurrent_shift",
                    description,
                    avg_delta,
                    json.dumps({
                        "original_event_type": event_type,
                        "brand_count": brand_count,
                        "brands": brands,
                    }),
                )
                detected += 1
            except Exception:
                logger.debug("Failed to persist concurrent_shift for %s", event_type)
    except Exception:
        logger.debug("Concurrent shift detection skipped", exc_info=True)
    if detected:
        logger.info("Detected %d concurrent shifts for %s", detected, today)
    return detected


# ------------------------------------------------------------------
# Webhook dispatch for consumer events
# ------------------------------------------------------------------


async def _dispatch_consumer_events(
    pool, brand_health: list, change_events: int, concurrent_shifts: int,
) -> None:
    """Dispatch consumer webhook events for change events and report generation."""
    try:
        from ...services.b2b.webhook_dispatcher import dispatch_consumer_webhooks

        _METRIC_KEYS = {
            "avg_rating", "total_reviews", "avg_pain_score",
            "repurchase_yes", "repurchase_no", "safety_flagged_count",
        }

        # Dispatch report_generated for each brand analyzed
        # brand_health is a list[dict] from _fetch_brand_health()
        for bd in brand_health:
            brand_name = bd.get("brand", "")
            if not brand_name:
                continue
            await dispatch_consumer_webhooks(
                pool, "consumer_report_generated", brand_name, {
                    "event": "report_generated",
                    "brand": brand_name,
                    "metrics": {
                        k: v for k, v in bd.items() if k in _METRIC_KEYS
                    },
                },
            )

        # Dispatch change events (already persisted, fetch today's)
        if change_events > 0:
            today = date.today()
            rows = await pool.fetch(
                """
                SELECT brand, event_type, description, delta
                FROM product_change_events
                WHERE event_date = $1 AND brand != '__market__'
                """,
                today,
            )
            for r in rows:
                await dispatch_consumer_webhooks(
                    pool, "consumer_change_event", r["brand"], {
                        "event_type": r["event_type"],
                        "brand": r["brand"],
                        "description": r["description"],
                        "delta": float(r["delta"]) if r["delta"] else None,
                    },
                )

        # Dispatch concurrent shifts
        if concurrent_shifts > 0:
            today = date.today()
            shifts = await pool.fetch(
                """
                SELECT description, delta, metadata
                FROM product_change_events
                WHERE event_date = $1 AND brand = '__market__'
                  AND event_type = 'concurrent_shift'
                """,
                today,
            )
            for s in shifts:
                meta = s["metadata"]
                if isinstance(meta, str):
                    meta = json.loads(meta)
                brands = (meta or {}).get("brands", [])
                for brand in brands:
                    await dispatch_consumer_webhooks(
                        pool, "consumer_concurrent_shift", brand, {
                            "description": s["description"],
                            "delta": float(s["delta"]) if s["delta"] else None,
                            "metadata": meta,
                        },
                    )
    except Exception:
        logger.debug("Consumer webhook dispatch skipped", exc_info=True)


# ------------------------------------------------------------------
# Displacement edge persistence
# ------------------------------------------------------------------


async def _persist_displacement_edges(pool) -> int:
    """Extract competitive flows from deep-enriched reviews and persist as canonical edges."""
    from ...pipelines.comparisons import (
        ADJACENCY_DIRECTIONS,
        load_known_brands,
        normalize_brand,
        normalize_canonical_brand,
    )

    today = date.today()
    known_brands = await load_known_brands(pool)

    rows = await pool.fetch(
        f"""
        SELECT
            pm.brand AS reviewed_brand,
            comp->>'product_name' AS product_name,
            COALESCE(comp->>'direction', 'compared') AS direction,
            {_CAT_EXPR} AS category,
            pr.rating,
            pr.severity,
            pr.deep_enrichment_status,
            pr.id AS review_id
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin,
             jsonb_array_elements(
                 CASE jsonb_typeof(pr.deep_extraction->'product_comparisons')
                      WHEN 'array' THEN pr.deep_extraction->'product_comparisons'
                      ELSE '[]'::jsonb
                 END
             ) AS comp
        WHERE pr.deep_enrichment_status = 'enriched'
          AND pr.deep_extraction->'product_comparisons' IS NOT NULL
          AND pm.brand IS NOT NULL AND pm.brand != ''
        """,
    )

    # Aggregate per (from_brand, to_brand, direction)
    edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in rows:
        direction = row["direction"]
        if direction in ADJACENCY_DIRECTIONS:
            continue

        raw_other = (row["product_name"] or "").strip()
        reviewed_brand = (row["reviewed_brand"] or "").strip()

        normalized_other = normalize_brand(raw_other, known_brands)
        if not normalized_other:
            continue
        normalized_reviewed = normalize_canonical_brand(reviewed_brand, known_brands)
        if not normalized_reviewed:
            continue
        if normalized_other.lower() == normalized_reviewed.lower():
            continue

        if direction == "switched_from":
            from_brand, to_brand = normalized_other, normalized_reviewed
        elif direction == "switched_to":
            from_brand, to_brand = normalized_reviewed, normalized_other
        else:
            from_brand, to_brand = normalized_reviewed, normalized_other

        # Resolve through brand registry
        from_brand = resolve_brand_name_cached(from_brand)
        to_brand = resolve_brand_name_cached(to_brand)

        key = (from_brand, to_brand, direction)
        if key not in edge_map:
            edge_map[key] = {
                "mention_count": 0,
                "ratings": [],
                "categories": {},
                "sample_ids": [],
                "severity": {"critical": 0, "major": 0, "minor": 0},
                "deep_count": 0,
                "total": 0,
            }

        e = edge_map[key]
        e["mention_count"] += 1
        e["total"] += 1
        if row["rating"] is not None:
            e["ratings"].append(float(row["rating"]))
        cat = row.get("category") or "unknown"
        e["categories"][cat] = e["categories"].get(cat, 0) + 1
        if row["review_id"]:
            e["sample_ids"].append(row["review_id"])
        sev = row.get("severity")
        if sev in e["severity"]:
            e["severity"][sev] += 1
        if row["deep_enrichment_status"] == "enriched":
            e["deep_count"] += 1

    # Persist edges with >= 2 mentions
    persisted = 0
    for (from_b, to_b, dirn), e in edge_map.items():
        mc = e["mention_count"]
        if mc < 2:
            continue

        if mc >= 10:
            strength = "strong"
        elif mc >= 4:
            strength = "moderate"
        else:
            strength = "emerging"

        avg_r = round(sum(e["ratings"]) / len(e["ratings"]), 2) if e["ratings"] else None
        conf = _compute_consumer_confidence(mc, e["deep_count"], e["total"], e["severity"])
        sample = [sid for sid in e["sample_ids"][:20]]

        try:
            await pool.execute(
                """
                INSERT INTO product_displacement_edges (
                    from_brand, to_brand, direction, mention_count,
                    signal_strength, avg_rating, category_distribution,
                    sample_review_ids, confidence_score, computed_date
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::uuid[], $9, $10)
                ON CONFLICT (from_brand, to_brand, direction, computed_date)
                DO UPDATE SET
                    mention_count = EXCLUDED.mention_count,
                    signal_strength = EXCLUDED.signal_strength,
                    avg_rating = EXCLUDED.avg_rating,
                    category_distribution = EXCLUDED.category_distribution,
                    sample_review_ids = EXCLUDED.sample_review_ids,
                    confidence_score = EXCLUDED.confidence_score
                """,
                from_b, to_b, dirn, mc, strength, avg_r,
                json.dumps(e["categories"]), sample, conf, today,
            )
            persisted += 1
        except Exception:
            logger.warning("Failed to persist edge %s -> %s", from_b, to_b, exc_info=True)

    if persisted:
        logger.info("Persisted %d product displacement edges for %s", persisted, today)
    return persisted
