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
import re
from datetime import date, datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.competitive_intelligence")


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

    # Gather 8 data sources in parallel
    (
        brand_health,
        competitive_flows,
        feature_gaps,
        buyer_personas,
        sentiment_landscape,
        safety_signals,
        loyalty_churn,
        prior_reports,
    ) = await asyncio.gather(
        _fetch_brand_health(pool),
        _fetch_competitive_flows(pool),
        _fetch_feature_gaps(pool),
        _fetch_buyer_personas(pool),
        _fetch_sentiment_landscape(pool),
        _fetch_safety_signals(pool),
        _fetch_loyalty_churn(pool),
        _fetch_prior_reports(pool),
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

    if not fetchers["brand_health"]:
        return {"_skip_synthesis": "No brand data to analyze"}

    # Post-process: normalize competitor names and feature requests
    if fetchers["competitive_flows"]:
        fetchers["competitive_flows"] = await _normalize_competitors(
            fetchers["competitive_flows"], pool
        )
    if fetchers["feature_gaps"]:
        fetchers["feature_gaps"] = _normalize_feature_requests(fetchers["feature_gaps"])

    # Build payload
    payload = {"date": str(today), **fetchers}

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
    if report_stored:
        await _upsert_brand_intelligence(
            pool, fetchers["brand_health"], parsed,
            safety_signals=fetchers["safety_signals"],
            loyalty_churn=fetchers["loyalty_churn"],
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
    }


# ------------------------------------------------------------------
# Data fetchers
# ------------------------------------------------------------------


async def _fetch_brand_health(pool) -> list[dict[str, Any]]:
    """Per-brand health: reviews, rating, pain, severity, repurchase, safety."""
    rows = await pool.fetch(
        """
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
            ) AS safety_flagged_count
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.deep_enrichment_status = 'enriched'
          AND pm.brand IS NOT NULL AND pm.brand != ''
        GROUP BY pm.brand
        HAVING count(*) >= 5
        ORDER BY count(*) DESC
        LIMIT 50
        """
    )
    return [
        {
            "brand": r["brand"],
            "total_reviews": r["total_reviews"],
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


async def _fetch_competitive_flows(pool) -> list[dict[str, Any]]:
    """Brand-to-brand customer migration from product_comparisons."""
    rows = await pool.fetch(
        """
        SELECT
            pm.brand AS source_brand,
            comp->>'product_name' AS competitor,
            comp->>'direction' AS direction,
            count(*) AS mentions
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        CROSS JOIN jsonb_array_elements(pr.deep_extraction->'product_comparisons') AS comp
        WHERE pr.deep_enrichment_status = 'enriched'
          AND pm.brand IS NOT NULL AND pm.brand != ''
          AND jsonb_array_length(pr.deep_extraction->'product_comparisons') > 0
        GROUP BY pm.brand, comp->>'product_name', comp->>'direction'
        HAVING count(*) >= 2
        ORDER BY count(*) DESC
        LIMIT 100
        """
    )
    return [
        {
            "source_brand": r["source_brand"],
            "competitor": r["competitor"],
            "direction": r["direction"],
            "mentions": r["mentions"],
        }
        for r in rows
    ]


async def _fetch_feature_gaps(pool) -> list[dict[str, Any]]:
    """Most-requested features across all products."""
    rows = await pool.fetch(
        """
        SELECT
            COALESCE(
                REPLACE(pm.categories->>2, '&amp;', '&'),
                REPLACE(pm.categories->>1, '&amp;', '&'),
                pr.source_category
            ) AS category,
            feat AS feature,
            count(*) AS mentions,
            avg(pr.pain_score) AS avg_pain_score
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        CROSS JOIN jsonb_array_elements_text(pr.deep_extraction->'feature_requests') AS feat
        WHERE pr.deep_enrichment_status = 'enriched'
          AND jsonb_array_length(pr.deep_extraction->'feature_requests') > 0
        GROUP BY category, feat
        HAVING count(*) >= 2
        ORDER BY count(*) DESC
        LIMIT 100
        """
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


async def _fetch_buyer_personas(pool) -> list[dict[str, Any]]:
    """Buyer segment clusters from buyer_context + expertise/budget."""
    rows = await pool.fetch(
        """
        SELECT
            COALESCE(
                REPLACE(pm.categories->>2, '&amp;', '&'),
                REPLACE(pm.categories->>1, '&amp;', '&'),
                pr.source_category
            ) AS category,
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
        GROUP BY
            category,
            pr.deep_extraction->'buyer_context'->>'buyer_type',
            pr.deep_extraction->'buyer_context'->>'use_case',
            pr.deep_extraction->'buyer_context'->>'price_sentiment',
            pr.deep_extraction->>'expertise_level',
            pr.deep_extraction->>'budget_type'
        HAVING count(*) >= 3
        ORDER BY count(*) DESC
        LIMIT 100
        """
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


async def _fetch_sentiment_landscape(pool) -> list[dict[str, Any]]:
    """Per-brand sentiment on specific aspects."""
    rows = await pool.fetch(
        """
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
        GROUP BY pm.brand, asp->>'aspect', asp->>'sentiment'
        ORDER BY count(*) DESC
        LIMIT 200
        """
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


async def _fetch_safety_signals(pool) -> list[dict[str, Any]]:
    """Per-brand safety-flagged reviews with consequence_severity breakdown."""
    rows = await pool.fetch(
        """
        SELECT
            pm.brand,
            pr.deep_extraction->>'consequence_severity' AS consequence,
            count(*) AS cnt
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.deep_enrichment_status = 'enriched'
          AND (pr.deep_extraction->'safety_flag'->>'flagged')::boolean IS TRUE
          AND pm.brand IS NOT NULL AND pm.brand != ''
        GROUP BY pm.brand, pr.deep_extraction->>'consequence_severity'
        ORDER BY count(*) DESC
        LIMIT 100
        """
    )
    return [
        {
            "brand": r["brand"],
            "consequence": r["consequence"],
            "count": r["cnt"],
        }
        for r in rows
    ]


async def _fetch_loyalty_churn(pool) -> list[dict[str, Any]]:
    """Per-brand loyalty depth x replacement behavior cross-tab."""
    rows = await pool.fetch(
        """
        SELECT
            pm.brand,
            pr.deep_extraction->>'brand_loyalty_depth' AS loyalty,
            pr.deep_extraction->>'replacement_behavior' AS replacement,
            count(*) AS cnt
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.deep_enrichment_status = 'enriched'
          AND pm.brand IS NOT NULL AND pm.brand != ''
        GROUP BY pm.brand,
            pr.deep_extraction->>'brand_loyalty_depth',
            pr.deep_extraction->>'replacement_behavior'
        HAVING count(*) >= 2
        ORDER BY count(*) DESC
        LIMIT 200
        """
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

_COMPETITOR_NOISE = {
    "competitor", "brand", "other", "another", "generic", "cheap",
    "remote", "device", "product", "unit", "model", "version",
    "itunes", "youtube", "amazon", "forum", "ebay", "walmart",
    "microsoft vista", "windows", "mac os", "other brand",
    "another product", "cheap one", "other one",
    "2 button version", "button", "cable", "adapter", "charger",
}


async def _normalize_competitors(flows: list[dict[str, Any]], pool) -> list[dict[str, Any]]:
    """Clean up competitor names: noise filter, case dedup, brand-aware collapse."""
    # Fetch known brands from product_metadata (lowercase key -> original casing)
    brand_rows = await pool.fetch(
        "SELECT DISTINCT brand FROM product_metadata "
        "WHERE brand IS NOT NULL AND brand != ''"
    )
    known_brands = {r["brand"].lower(): r["brand"] for r in brand_rows}

    merged: dict[tuple[str, str, str], dict[str, Any]] = {}

    for flow in flows:
        raw = (flow.get("competitor") or "").strip()
        if not raw:
            continue

        lowered = raw.lower()

        # Filter noise
        if lowered in _COMPETITOR_NOISE:
            continue
        if raw.isdigit():
            continue

        # Brand-aware collapse: if first word matches a known brand, normalize
        # to just the brand name (drop model number / spec details)
        words = raw.split()
        first_lower = words[0].lower()

        # Filter short names unless they match a known brand (e.g. "LG")
        if len(raw) < 3 and first_lower not in known_brands:
            continue
        if first_lower in known_brands:
            normalized = known_brands[first_lower]  # Preserve original DB casing
        else:
            # Simple case normalization
            normalized = raw.title()

        key = (flow.get("source_brand", ""), normalized, flow.get("direction", ""))
        if key in merged:
            merged[key]["mentions"] += flow.get("mentions", 0)
        else:
            merged[key] = {
                "source_brand": flow.get("source_brand", ""),
                "competitor": normalized,
                "direction": flow.get("direction", ""),
                "mentions": flow.get("mentions", 0),
            }

    result = sorted(merged.values(), key=lambda x: x["mentions"], reverse=True)
    return result


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
            brand = flow.get("source_brand", "")
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

        try:
            await pool.execute(
                """
                INSERT INTO brand_intelligence (
                    brand, source, total_reviews, avg_rating, avg_pain_score,
                    repurchase_yes, repurchase_no,
                    sentiment_breakdown, top_feature_requests, top_complaints,
                    competitive_flows, buyer_profile, positive_aspects,
                    health_score, last_computed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
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
                    last_computed_at = EXCLUDED.last_computed_at
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
            )
            upserted += 1
        except Exception:
            logger.warning("Failed to upsert brand intelligence for %s", brand, exc_info=True)

    if upserted:
        logger.info("Upserted %d brand intelligence scorecards", upserted)
