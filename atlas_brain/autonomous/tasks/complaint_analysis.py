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

    window_days = cfg.complaint_analysis_window_days
    today = date.today()

    # Skip if we already have a report for today
    existing = await pool.fetchrow(
        "SELECT id FROM complaint_reports WHERE report_date = $1 LIMIT 1",
        today,
    )
    if existing:
        return {"_skip_synthesis": f"Report already exists for {today}"}

    # Gather data sources in parallel
    category_stats, product_stats, prior_reports = await asyncio.gather(
        _fetch_category_stats(pool, window_days),
        _fetch_product_stats(pool, window_days),
        _fetch_prior_reports(pool),
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

    # Check if there's enough data
    total_enriched = sum(c.get("total_enriched", 0) for c in category_stats)
    if total_enriched == 0 and not product_stats:
        return {"_skip_synthesis": "No enriched reviews to analyze"}

    # Build payload
    payload = {
        "date": str(today),
        "category_stats": category_stats,
        "product_stats": product_stats,
        "prior_reports": prior_reports,
    }

    # Load skill and call LLM
    from ...pipelines.llm import call_llm_with_skill, parse_json_response

    analysis = call_llm_with_skill(
        "digest/complaint_analysis", payload,
        max_tokens=cfg.complaint_analysis_max_tokens, temperature=0.4,
    )
    if not analysis:
        return {"_skip_synthesis": "LLM analysis failed"}

    # Parse structured output
    parsed = parse_json_response(analysis)

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


async def _fetch_category_stats(pool, window_days: int) -> list[dict[str, Any]]:
    """Aggregate enriched reviews by source_category."""
    rows = await pool.fetch(
        """
        SELECT
            source_category AS category,
            count(*) AS total_enriched,
            count(*) FILTER (WHERE severity = 'critical') AS critical_count,
            count(*) FILTER (WHERE severity = 'major') AS major_count,
            count(*) FILTER (WHERE severity = 'minor') AS minor_count,
            avg(pain_score) AS avg_pain_score,
            mode() WITHIN GROUP (ORDER BY root_cause) AS top_root_cause
        FROM product_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY source_category
        ORDER BY count(*) DESC
        """,
        window_days,
    )
    result = []
    for r in rows:
        # Fetch root cause distribution for this category
        rc_rows = await pool.fetch(
            """
            SELECT root_cause, count(*) AS cnt
            FROM product_reviews
            WHERE enrichment_status = 'enriched'
              AND source_category = $1
              AND enriched_at > NOW() - make_interval(days => $2)
            GROUP BY root_cause
            ORDER BY cnt DESC
            """,
            r["category"],
            window_days,
        )
        rc_dist = {row["root_cause"]: row["cnt"] for row in rc_rows if row["root_cause"]}

        result.append({
            "category": r["category"],
            "total_enriched": r["total_enriched"],
            "severity_distribution": {
                "critical": r["critical_count"],
                "major": r["major_count"],
                "minor": r["minor_count"],
            },
            "root_cause_distribution": rc_dist,
            "avg_pain_score": round(float(r["avg_pain_score"]), 2) if r["avg_pain_score"] else 0.0,
            "top_root_cause": r["top_root_cause"],
        })
    return result


async def _fetch_product_stats(pool, window_days: int) -> list[dict[str, Any]]:
    """Aggregate by ASIN for products with 3+ complaints."""
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
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY asin, source_category
        HAVING count(*) >= 3
        ORDER BY avg(pain_score) DESC
        LIMIT 50
        """,
        window_days,
    )

    result = []
    for r in rows:
        asin = r["asin"]

        # Top specific complaints for this ASIN
        complaint_rows = await pool.fetch(
            """
            SELECT specific_complaint, count(*) AS cnt
            FROM product_reviews
            WHERE asin = $1 AND enrichment_status = 'enriched'
              AND specific_complaint IS NOT NULL
            GROUP BY specific_complaint
            ORDER BY cnt DESC
            LIMIT 5
            """,
            asin,
        )
        top_complaints = [row["specific_complaint"] for row in complaint_rows]

        # Root cause distribution
        rc_rows = await pool.fetch(
            """
            SELECT root_cause, count(*) AS cnt
            FROM product_reviews
            WHERE asin = $1 AND enrichment_status = 'enriched'
            GROUP BY root_cause
            ORDER BY cnt DESC
            """,
            asin,
        )
        root_causes = {row["root_cause"]: row["cnt"] for row in rc_rows if row["root_cause"]}

        # Manufacturing suggestions
        mfg_rows = await pool.fetch(
            """
            SELECT manufacturing_suggestion
            FROM product_reviews
            WHERE asin = $1 AND enrichment_status = 'enriched'
              AND actionable_for_manufacturing = true
              AND manufacturing_suggestion IS NOT NULL
            LIMIT 5
            """,
            asin,
        )
        mfg_suggestions = [row["manufacturing_suggestion"] for row in mfg_rows]

        # Alternatives mentioned
        alt_rows = await pool.fetch(
            """
            SELECT alternative_name, count(*) AS cnt
            FROM product_reviews
            WHERE asin = $1 AND enrichment_status = 'enriched'
              AND alternative_mentioned = true
              AND alternative_name IS NOT NULL
            GROUP BY alternative_name
            ORDER BY cnt DESC
            LIMIT 5
            """,
            asin,
        )
        alternatives = [
            {"name": row["alternative_name"], "mentions": row["cnt"]}
            for row in alt_rows
        ]

        result.append({
            "asin": asin,
            "category": r["category"],
            "complaint_count": r["complaint_count"],
            "avg_pain_score": round(float(r["avg_pain_score"]), 2) if r["avg_pain_score"] else 0.0,
            "avg_rating": round(float(r["avg_rating"]), 2) if r["avg_rating"] else 0.0,
            "top_complaints": top_complaints,
            "root_causes": root_causes,
            "manufacturing_suggestions": mfg_suggestions,
            "alternatives": alternatives,
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


