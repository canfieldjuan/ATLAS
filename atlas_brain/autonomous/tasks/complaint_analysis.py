"""
Complaint analysis: aggregate enriched product reviews by category and ASIN,
feed to LLM with prior reports, and persist structured conclusions.

Runs daily (default 9 PM). Handles its own LLM call, report persistence,
product_pain_points upserts, and ntfy notification -- returns _skip_synthesis
so the runner does not double-synthesize.
"""

import asyncio
import json
import logging
from datetime import date, datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.complaint_analysis")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: daily complaint analysis."""
    cfg = settings.external_data
    if not cfg.complaint_mining_enabled or not cfg.complaint_analysis_enabled:
        return {"_skip_synthesis": "Complaint analysis disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    today = date.today()

    # Skip if we already have a report for today
    existing = await pool.fetchrow(
        "SELECT id FROM complaint_reports WHERE report_date = $1 LIMIT 1",
        today,
    )
    if existing:
        return {"_skip_synthesis": f"Report already exists for {today}"}

    # Gather data sources in parallel
    category_stats, product_stats, prior_reports, data_context = await asyncio.gather(
        _fetch_category_stats(pool),
        _fetch_product_stats(pool),
        _fetch_prior_reports(pool),
        _fetch_data_context(pool),
        return_exceptions=True,
    )

    # Convert exceptions to empty values
    if isinstance(category_stats, Exception):
        logger.warning("Category stats fetch failed: %s", category_stats)
        category_stats = []
    if isinstance(product_stats, Exception):
        logger.warning("Product stats fetch failed: %s", product_stats)
        product_stats = []
    if isinstance(prior_reports, Exception):
        logger.warning("Prior reports fetch failed: %s", prior_reports)
        prior_reports = []
    if isinstance(data_context, Exception):
        logger.warning("Data context fetch failed: %s", data_context)
        data_context = {}

    # Check if there's enough data
    total_enriched = sum(c.get("total_enriched", 0) for c in category_stats)
    if total_enriched == 0 and not product_stats:
        return {"_skip_synthesis": "No enriched reviews to analyze"}

    # Build payload -- trim to fit ~4k token input budget (8k context - 4k output).
    # Full product_stats used below for upserts.
    llm_product_stats = [
        {
            "asin": p["asin"],
            "category": p["category"],
            "complaints": p["complaint_count"],
            "pain": p["avg_pain_score"],
            "rating": p["avg_rating"],
            "top_complaints": p["top_complaints"][:2],
            "root_causes": p["root_causes"],
        }
        for p in product_stats[:15]
    ]
    # Compact category stats for LLM
    llm_categories = [
        {
            "category": c["category"],
            "count": c["total_enriched"],
            "period": c.get("review_period", ""),
            "pain": c["avg_pain_score"],
            "top_cause": c["top_root_cause"],
        }
        for c in category_stats
    ]
    payload = {
        "date": str(today),
        "data_context": data_context,
        "total_products_with_complaints": len(product_stats),
        "category_stats": llm_categories,
        "product_stats": llm_product_stats,
    }

    # Load skill and call LLM (in thread to avoid blocking event loop)
    from ...pipelines.llm import call_llm_with_skill, parse_json_response

    try:
        analysis = await asyncio.wait_for(
            asyncio.to_thread(
                call_llm_with_skill,
                "digest/complaint_analysis", payload,
                max_tokens=cfg.complaint_analysis_max_tokens, temperature=0.4,
                response_format={"type": "json_object"},
            ),
            timeout=300,
        )
    except asyncio.TimeoutError:
        logger.error("LLM call timed out after 300s for complaint_analysis")
        # Still upsert pain points even if LLM times out
        await _upsert_pain_points(pool, product_stats, {})
        return {"_skip_synthesis": "LLM analysis timed out", "products_upserted": len(product_stats)}
    if not analysis:
        return {"_skip_synthesis": "LLM analysis failed"}

    # Parse structured output
    parsed = parse_json_response(analysis, recover_truncated=True)

    # Persist to complaint_reports
    try:
        await pool.execute(
            """
            INSERT INTO complaint_reports (
                report_date, report_type, category_filter,
                analysis_output, top_pain_points, opportunities,
                recommendations, product_highlights
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            today,
            "daily",
            None,
            parsed.get("analysis_text", analysis),
            json.dumps(parsed.get("top_pain_points", [])),
            json.dumps(parsed.get("opportunities", [])),
            json.dumps(parsed.get("recommendations", [])),
            json.dumps(parsed.get("product_highlights", [])),
        )
        logger.info("Stored complaint report for %s", today)
    except Exception:
        logger.exception("Failed to store complaint report")

    # Upsert product_pain_points from product_stats
    await _upsert_pain_points(pool, product_stats, parsed)

    # Send ntfy notification
    from ...pipelines.notify import send_pipeline_notification

    await send_pipeline_notification(
        parsed.get("analysis_text", analysis), task,
        title="Atlas: Complaint Analysis",
        default_tags="brain,shopping_cart",
        parsed=parsed,
    )

    return {
        "_skip_synthesis": "Complaint analysis complete",
        "date": str(today),
        "categories": len(category_stats),
        "products_analyzed": len(product_stats),
        "total_enriched": total_enriched,
        "pain_points": len(parsed.get("top_pain_points", [])),
        "opportunities": len(parsed.get("opportunities", [])),
    }


# ------------------------------------------------------------------
# Data fetchers
# ------------------------------------------------------------------


async def _fetch_data_context(pool) -> dict[str, Any]:
    """Compute temporal metadata for the dataset so the LLM can anchor claims."""
    row = await pool.fetchrow(
        """
        SELECT
            count(*) AS total_enriched,
            count(*) FILTER (WHERE reviewed_at IS NOT NULL) AS with_date,
            min(reviewed_at) AS earliest_review,
            max(reviewed_at) AS latest_review,
            count(*) FILTER (WHERE reviewed_at >= NOW() - INTERVAL '1 year') AS last_1y,
            count(*) FILTER (WHERE reviewed_at >= NOW() - INTERVAL '3 years') AS last_3y,
            count(*) FILTER (WHERE reviewed_at < NOW() - INTERVAL '3 years') AS older_3y
        FROM product_reviews
        WHERE enrichment_status = 'enriched'
        """
    )
    return {
        "total_reviews_analyzed": row["total_enriched"],
        "reviews_with_dates": row["with_date"],
        "review_period": {
            "earliest": str(row["earliest_review"].date()) if row["earliest_review"] else None,
            "latest": str(row["latest_review"].date()) if row["latest_review"] else None,
        },
        "recency": {
            "last_1_year": row["last_1y"],
            "last_3_years": row["last_3y"],
            "older_than_3_years": row["older_3y"],
        },
        "note": "Use these date ranges when citing statistics. Say 'over the past N years' or 'between YYYY and YYYY' instead of unanchored claims.",
    }


async def _fetch_category_stats(pool) -> list[dict[str, Any]]:
    """Aggregate enriched reviews by source_category (all enriched, no time window).

    Uses a single bulk query for root cause distributions instead of N+1.
    """
    rows, rc_rows = await asyncio.gather(
        pool.fetch(
            """
            SELECT
                source_category AS category,
                count(*) AS total_enriched,
                count(*) FILTER (WHERE severity = 'critical') AS critical_count,
                count(*) FILTER (WHERE severity = 'major') AS major_count,
                count(*) FILTER (WHERE severity = 'minor') AS minor_count,
                avg(pain_score) AS avg_pain_score,
                mode() WITHIN GROUP (ORDER BY root_cause) AS top_root_cause,
                min(reviewed_at)::date AS earliest_review,
                max(reviewed_at)::date AS latest_review
            FROM product_reviews
            WHERE enrichment_status = 'enriched'
            GROUP BY source_category
            ORDER BY count(*) DESC
            """,
        ),
        pool.fetch(
            """
            SELECT source_category AS category, root_cause, count(*) AS cnt
            FROM product_reviews
            WHERE enrichment_status = 'enriched'
              AND root_cause IS NOT NULL
            GROUP BY source_category, root_cause
            ORDER BY source_category, cnt DESC
            """,
        ),
    )

    # Build root cause lookup by category
    rc_by_cat: dict[str, dict[str, int]] = {}
    for row in rc_rows:
        rc_by_cat.setdefault(row["category"], {})[row["root_cause"]] = row["cnt"]

    result = []
    for r in rows:
        result.append({
            "category": r["category"],
            "total_enriched": r["total_enriched"],
            "review_period": f"{r['earliest_review']} to {r['latest_review']}" if r["earliest_review"] else "dates unavailable",
            "severity_distribution": {
                "critical": r["critical_count"],
                "major": r["major_count"],
                "minor": r["minor_count"],
            },
            "root_cause_distribution": rc_by_cat.get(r["category"], {}),
            "avg_pain_score": round(float(r["avg_pain_score"]), 2) if r["avg_pain_score"] else 0.0,
            "top_root_cause": r["top_root_cause"],
        })
    return result


async def _fetch_product_stats(pool) -> list[dict[str, Any]]:
    """Aggregate by ASIN for products with 5+ complaints (all enriched, no time window).

    Uses bulk queries instead of per-ASIN sub-queries to avoid N+1 pattern.
    4 queries total instead of 4*N (was ~14k queries for 3,710 ASINs).
    """
    rows = await pool.fetch(
        """
        SELECT
            asin,
            source_category AS category,
            count(*) AS complaint_count,
            avg(pain_score) AS avg_pain_score,
            avg(rating) AS avg_rating
        FROM product_reviews
        WHERE enrichment_status = 'enriched'
        GROUP BY asin, source_category
        HAVING count(*) >= 5
        ORDER BY avg(pain_score) DESC
        """,
    )
    if not rows:
        return []

    # Bulk fetch all sub-data in 4 queries (replaces 4 queries per ASIN)
    complaint_rows, rc_rows, mfg_rows, alt_rows = await asyncio.gather(
        pool.fetch(
            """
            SELECT asin, specific_complaint, count(*) AS cnt
            FROM product_reviews
            WHERE enrichment_status = 'enriched'
              AND specific_complaint IS NOT NULL
            GROUP BY asin, specific_complaint
            ORDER BY asin, cnt DESC
            """,
        ),
        pool.fetch(
            """
            SELECT asin, root_cause, count(*) AS cnt
            FROM product_reviews
            WHERE enrichment_status = 'enriched'
              AND root_cause IS NOT NULL
            GROUP BY asin, root_cause
            ORDER BY asin, cnt DESC
            """,
        ),
        pool.fetch(
            """
            SELECT DISTINCT ON (asin, manufacturing_suggestion)
                asin, manufacturing_suggestion
            FROM product_reviews
            WHERE enrichment_status = 'enriched'
              AND actionable_for_manufacturing = true
              AND manufacturing_suggestion IS NOT NULL
            ORDER BY asin, manufacturing_suggestion
            """,
        ),
        pool.fetch(
            """
            SELECT asin, alternative_name, count(*) AS cnt
            FROM product_reviews
            WHERE enrichment_status = 'enriched'
              AND alternative_mentioned = true
              AND alternative_name IS NOT NULL
            GROUP BY asin, alternative_name
            ORDER BY asin, cnt DESC
            """,
        ),
    )

    # Build lookup dicts keyed by ASIN
    complaints_by_asin: dict[str, list[str]] = {}
    for row in complaint_rows:
        complaints_by_asin.setdefault(row["asin"], []).append(row["specific_complaint"])

    rc_by_asin: dict[str, dict[str, int]] = {}
    for row in rc_rows:
        rc_by_asin.setdefault(row["asin"], {})[row["root_cause"]] = row["cnt"]

    mfg_by_asin: dict[str, list[str]] = {}
    for row in mfg_rows:
        mfg_by_asin.setdefault(row["asin"], []).append(row["manufacturing_suggestion"])

    alt_by_asin: dict[str, list[dict]] = {}
    for row in alt_rows:
        alt_by_asin.setdefault(row["asin"], []).append(
            {"name": row["alternative_name"], "mentions": row["cnt"]}
        )

    # Assemble results
    result = []
    for r in rows:
        asin = r["asin"]
        result.append({
            "asin": asin,
            "category": r["category"],
            "complaint_count": r["complaint_count"],
            "avg_pain_score": round(float(r["avg_pain_score"]), 2) if r["avg_pain_score"] else 0.0,
            "avg_rating": round(float(r["avg_rating"]), 2) if r["avg_rating"] else 0.0,
            "top_complaints": complaints_by_asin.get(asin, [])[:5],
            "root_causes": rc_by_asin.get(asin, {}),
            "manufacturing_suggestions": mfg_by_asin.get(asin, [])[:5],
            "alternatives": alt_by_asin.get(asin, [])[:5],
        })

    return result


async def _fetch_prior_reports(pool, limit: int = 5) -> list[dict[str, Any]]:
    """Fetch prior complaint_reports (most recent first)."""
    rows = await pool.fetch(
        """
        SELECT report_date, report_type, analysis_output,
               top_pain_points, opportunities, recommendations,
               product_highlights
        FROM complaint_reports
        ORDER BY report_date DESC
        LIMIT $1
        """,
        limit,
    )
    result = []
    for r in rows:
        entry: dict[str, Any] = {
            "report_date": str(r["report_date"]),
            "report_type": r["report_type"],
            "analysis_output": (r["analysis_output"] or "")[:1000],
        }
        for field in ("top_pain_points", "opportunities", "recommendations", "product_highlights"):
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
# Pain point upserts
# ------------------------------------------------------------------


async def _upsert_pain_points(
    pool, product_stats: list[dict[str, Any]], parsed: dict[str, Any]
) -> None:
    """Upsert product_pain_points from aggregated product stats."""
    now = datetime.now(timezone.utc)
    upserted = 0

    # Build a lookup from parsed highlights for product_name
    highlights = {
        h["asin"]: h
        for h in parsed.get("product_highlights", [])
        if isinstance(h, dict) and h.get("asin")
    }

    for prod in product_stats:
        asin = prod.get("asin")
        if not asin:
            continue

        highlight = highlights.get(asin, {})
        product_name = highlight.get("product_name", "")

        try:
            await pool.execute(
                """
                INSERT INTO product_pain_points (
                    asin, product_name, category,
                    total_reviews, complaint_reviews, complaint_rate,
                    top_complaints, root_cause_distribution, severity_distribution,
                    differentiation_opportunities, alternative_products,
                    pain_score, last_computed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (asin) DO UPDATE SET
                    product_name = COALESCE(NULLIF(EXCLUDED.product_name, ''), product_pain_points.product_name),
                    category = EXCLUDED.category,
                    complaint_reviews = EXCLUDED.complaint_reviews,
                    complaint_rate = EXCLUDED.complaint_rate,
                    top_complaints = EXCLUDED.top_complaints,
                    root_cause_distribution = EXCLUDED.root_cause_distribution,
                    differentiation_opportunities = EXCLUDED.differentiation_opportunities,
                    alternative_products = EXCLUDED.alternative_products,
                    pain_score = EXCLUDED.pain_score,
                    last_computed_at = EXCLUDED.last_computed_at
                """,
                asin,
                product_name,
                prod.get("category", ""),
                prod.get("complaint_count", 0),  # total_reviews approximation
                prod.get("complaint_count", 0),
                1.0,  # all reviews in our dataset are complaints
                json.dumps(prod.get("top_complaints", [])),
                json.dumps(prod.get("root_causes", {})),
                json.dumps({}),  # severity_distribution computed at category level
                json.dumps([]),  # filled by analysis
                json.dumps(prod.get("alternatives", [])),
                prod.get("avg_pain_score", 0.0),
                now,
            )
            upserted += 1
        except Exception:
            logger.warning("Failed to upsert pain point for %s", asin, exc_info=True)

    if upserted:
        logger.info("Upserted %d product pain points", upserted)


