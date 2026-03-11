"""
Subcategory-level intelligence reports for Amazon product categories.

Auto-discovers qualifying subcategories from product_metadata.categories JSONB,
aggregates enriched review data (same 8-query pattern as seller campaigns),
then generates structured reports targeting three buyer audiences:
existing sellers, dropshippers, and new brand entrants.

Returns _skip_synthesis.
"""

import asyncio
import json
import logging
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.subcategory_intelligence")


# ------------------------------------------------------------------
# Subcategory discovery
# ------------------------------------------------------------------


async def _discover_subcategories(pool, cfg) -> list[dict[str, Any]]:
    """Find subcategories with enough products and enriched reviews.

    Returns list of dicts with subcategory, product_count, review_count.
    """
    if cfg.target_subcategories:
        # Validate explicit targets
        rows = await pool.fetch(
            """
            WITH cat_elements AS (
                SELECT pm.asin, elem.value #>> '{}' AS subcategory
                FROM product_metadata pm,
                     jsonb_array_elements(pm.categories) AS elem
                WHERE jsonb_typeof(pm.categories) = 'array'
            )
            SELECT ce.subcategory,
                   COUNT(DISTINCT ce.asin) AS product_count,
                   COUNT(DISTINCT pr.id) AS review_count
            FROM cat_elements ce
            JOIN product_reviews pr ON pr.asin = ce.asin
                 AND pr.enrichment_status = 'enriched'
            WHERE ce.subcategory = ANY($1)
            GROUP BY ce.subcategory
            HAVING COUNT(DISTINCT ce.asin) >= $2
               AND COUNT(DISTINCT pr.id) >= $3
            ORDER BY COUNT(DISTINCT pr.id) DESC
            """,
            cfg.target_subcategories,
            cfg.min_products,
            cfg.min_reviews,
        )
    else:
        # Auto-discover: find qualifying subcategories, exclude overly broad ones
        rows = await pool.fetch(
            """
            WITH cat_elements AS (
                SELECT pm.asin, elem.value #>> '{}' AS subcategory
                FROM product_metadata pm,
                     jsonb_array_elements(pm.categories) AS elem
                WHERE jsonb_typeof(pm.categories) = 'array'
            ),
            total_products AS (
                SELECT COUNT(DISTINCT asin) AS cnt FROM product_metadata
            ),
            subcats AS (
                SELECT ce.subcategory,
                       COUNT(DISTINCT ce.asin) AS product_count,
                       COUNT(DISTINCT pr.id) AS review_count
                FROM cat_elements ce
                JOIN product_reviews pr ON pr.asin = ce.asin
                     AND pr.enrichment_status = 'enriched'
                GROUP BY ce.subcategory
                HAVING COUNT(DISTINCT ce.asin) >= $1
                   AND COUNT(DISTINCT pr.id) >= $2
            )
            SELECT s.subcategory, s.product_count, s.review_count
            FROM subcats s, total_products tp
            WHERE s.product_count < tp.cnt * 0.3
            ORDER BY s.review_count DESC
            LIMIT $3
            """,
            cfg.min_products,
            cfg.min_reviews,
            cfg.max_subcategories_per_run,
        )

    return [dict(r) for r in rows]


async def _filter_dedup(pool, subcategories: list[dict], dedup_days: int) -> list[dict]:
    """Remove subcategories that already have a recent report."""
    if not subcategories or dedup_days <= 0:
        return subcategories

    names = [s["subcategory"] for s in subcategories]
    recent = await pool.fetch(
        """
        SELECT DISTINCT entity_name
        FROM intelligence_reports
        WHERE entity_type = 'subcategory'
          AND report_type = 'subcategory_intelligence'
          AND entity_name = ANY($1)
          AND created_at >= NOW() - MAKE_INTERVAL(days => $2)
        """,
        names,
        dedup_days,
    )
    recent_set = {r["entity_name"] for r in recent}
    return [s for s in subcategories if s["subcategory"] not in recent_set]


# ------------------------------------------------------------------
# Subcategory intelligence aggregation (8-query pattern)
# ------------------------------------------------------------------


async def _aggregate_subcategory_intelligence(
    pool, subcategory: str,
) -> dict[str, Any] | None:
    """Build intelligence snapshot for a subcategory using JSONB containment.

    Same 8-query structure as amazon_seller_campaign_generation but uses
    pm.categories @> $1::jsonb instead of pr.source_category = $1.
    """
    cat_filter = json.dumps([subcategory])

    # 1. Stats
    stats = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total_reviews,
            COUNT(DISTINCT pr.asin) AS total_products,
            COUNT(DISTINCT pm.brand) AS total_brands
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.categories @> $1::jsonb
        """,
        cat_filter,
    )
    if not stats or stats["total_reviews"] < settings.subcategory_intelligence.min_reviews:
        return None

    # 2. Brand health rows
    brand_rows = await pool.fetch(
        """
        SELECT
            pm.brand,
            COUNT(*) AS total_reviews,
            ROUND(AVG(pr.rating), 2) AS avg_rating,
            COUNT(*) FILTER (
                WHERE pr.deep_extraction->>'would_repurchase' = 'true'
            ) AS repurchase_yes,
            COUNT(*) FILTER (
                WHERE pr.deep_extraction->>'would_repurchase' = 'false'
            ) AS repurchase_no,
            COUNT(*) FILTER (
                WHERE pr.deep_extraction->'safety_flag'->>'flagged' = 'true'
            ) AS safety_count,
            COUNT(*) FILTER (
                WHERE pr.deep_extraction IS NOT NULL
                  AND pr.deep_extraction != '{}'::jsonb
            ) AS deep_count
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.categories @> $1::jsonb
          AND pm.brand IS NOT NULL AND pm.brand != ''
        GROUP BY pm.brand
        HAVING COUNT(*) >= 5
        ORDER BY COUNT(*) DESC
        LIMIT 20
        """,
        cat_filter,
    )

    # 3. Top pain points
    pain_rows = await pool.fetch(
        """
        SELECT pr.root_cause AS complaint,
               COUNT(*) AS count,
               MAX(pr.severity) AS severity,
               COUNT(DISTINCT pm.brand) AS affected_brands
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.categories @> $1::jsonb
          AND pr.enrichment_status = 'enriched'
          AND pr.root_cause IS NOT NULL AND pr.root_cause != ''
          AND pr.rating <= 3
        GROUP BY pr.root_cause
        ORDER BY COUNT(*) DESC
        LIMIT 10
        """,
        cat_filter,
    )
    top_pain_points = [
        {
            "complaint": r["complaint"],
            "count": r["count"],
            "severity": r["severity"] or "medium",
            "affected_brands": r["affected_brands"] or 0,
        }
        for r in pain_rows
    ]

    # 4. Feature gaps
    feature_rows = await pool.fetch(
        """
        SELECT req AS request,
               COUNT(*) AS count,
               COUNT(DISTINCT brand) AS brand_count,
               ROUND(AVG(rating), 1) AS avg_rating
        FROM (
            SELECT pr.asin, pr.rating, pm.brand,
                   CASE jsonb_typeof(elem)
                        WHEN 'string' THEN elem #>> '{}'
                        WHEN 'object' THEN elem ->> 'request'
                        ELSE elem #>> '{}'
                   END AS req
            FROM product_reviews pr
            JOIN product_metadata pm ON pm.asin = pr.asin,
                 jsonb_array_elements(
                     CASE jsonb_typeof(pr.deep_extraction->'feature_requests')
                          WHEN 'array' THEN pr.deep_extraction->'feature_requests'
                          ELSE '[]'::jsonb
                     END
                 ) AS elem
            WHERE pm.categories @> $1::jsonb
              AND pr.deep_enrichment_status = 'enriched'
              AND pr.deep_extraction->'feature_requests' IS NOT NULL
        ) sub
        WHERE req IS NOT NULL AND req != '' AND req != 'null'
        GROUP BY req
        HAVING COUNT(*) >= 2
        ORDER BY COUNT(*) DESC
        LIMIT 15
        """,
        cat_filter,
    )
    feature_gaps = [
        {
            "request": r["request"],
            "count": r["count"],
            "brand_count": r["brand_count"],
            "avg_rating": float(r["avg_rating"]) if r["avg_rating"] else 0,
        }
        for r in feature_rows
    ]

    # 5. Competitive flows
    from ...pipelines.comparisons import fetch_competitive_flows

    competitive_flows = await fetch_competitive_flows(
        pool,
        where_clause="pm.categories @> $1::jsonb",
        params=[cat_filter],
        min_mentions=2,
        limit=15,
    )

    # 6. Brand health scores
    brand_health = []
    for br in (brand_rows or [])[:10]:
        yes = br["repurchase_yes"] or 0
        no = br["repurchase_no"] or 0
        repurchase_total = yes + no
        repurchase_rate = yes / repurchase_total if repurchase_total > 0 else 0.5

        deep = br["deep_count"] or 0
        safety = br["safety_count"] or 0
        safety_rate = max(0, 1.0 - (safety / deep) * 10) if deep >= 5 else 1.0

        hs = round((repurchase_rate + safety_rate) / 2 * 100, 1)
        brand_health.append({
            "brand": br["brand"],
            "health_score": hs,
            "trend": "rising" if hs >= 70 else ("falling" if hs < 40 else "stable"),
            "review_count": br["total_reviews"] or 0,
        })

    # 7. Safety signals
    safety_rows = await pool.fetch(
        """
        SELECT
            COALESCE(pm.brand, pr.asin) AS brand,
            pr.deep_extraction->'safety_flag'->>'category' AS category,
            pr.deep_extraction->'safety_flag'->>'description' AS description,
            COUNT(*) AS flagged_count
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.categories @> $1::jsonb
          AND pr.deep_enrichment_status = 'enriched'
          AND pr.deep_extraction->'safety_flag'->>'flagged' = 'true'
        GROUP BY COALESCE(pm.brand, pr.asin),
                 pr.deep_extraction->'safety_flag'->>'category',
                 pr.deep_extraction->'safety_flag'->>'description'
        ORDER BY COUNT(*) DESC
        LIMIT 10
        """,
        cat_filter,
    )
    safety_signals = [
        {
            "brand": r["brand"],
            "category": r["category"] or "",
            "description": r["description"] or "",
            "flagged_count": r["flagged_count"],
        }
        for r in safety_rows
    ]

    # 8. Manufacturing insights + top root causes
    mfg_rows = await pool.fetch(
        """
        SELECT pr.manufacturing_suggestion AS suggestion,
               COUNT(*) AS count,
               ARRAY_AGG(DISTINCT pr.asin) AS affected_asins
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.categories @> $1::jsonb
          AND pr.enrichment_status = 'enriched'
          AND pr.actionable_for_manufacturing = TRUE
          AND pr.manufacturing_suggestion IS NOT NULL
          AND pr.manufacturing_suggestion != ''
        GROUP BY pr.manufacturing_suggestion
        ORDER BY COUNT(*) DESC
        LIMIT 10
        """,
        cat_filter,
    )
    manufacturing_insights = [
        {
            "suggestion": r["suggestion"],
            "count": r["count"],
            "affected_asins": (r["affected_asins"] or [])[:5],
        }
        for r in mfg_rows
    ]

    cause_rows = await pool.fetch(
        """
        SELECT pr.root_cause AS cause, COUNT(*) AS count
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.categories @> $1::jsonb
          AND pr.enrichment_status = 'enriched'
          AND pr.root_cause IS NOT NULL AND pr.root_cause != ''
        GROUP BY pr.root_cause
        ORDER BY COUNT(*) DESC
        LIMIT 10
        """,
        cat_filter,
    )
    top_root_causes = [
        {"cause": r["cause"], "count": r["count"]}
        for r in cause_rows
    ]

    # Build category path from one representative product
    path_row = await pool.fetchrow(
        """
        SELECT categories FROM product_metadata
        WHERE categories @> $1::jsonb
        LIMIT 1
        """,
        cat_filter,
    )
    category_path = path_row["categories"] if path_row else [subcategory]

    return {
        "subcategory": subcategory,
        "category_path": category_path,
        "category": subcategory,
        "category_stats": {
            "total_reviews": stats["total_reviews"],
            "total_brands": stats["total_brands"],
            "total_products": stats["total_products"],
            "date_range": "all available data",
        },
        "top_pain_points": top_pain_points,
        "feature_gaps": feature_gaps,
        "competitive_flows": competitive_flows,
        "brand_health": brand_health,
        "safety_signals": safety_signals,
        "manufacturing_insights": manufacturing_insights,
        "top_root_causes": top_root_causes,
    }


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------


async def _save_intelligence_snapshot(pool, intel: dict[str, Any]) -> None:
    """Cache subcategory intelligence snapshot."""
    cat_path = intel.get("category_path")
    cat_path_json = json.dumps(cat_path) if cat_path else None

    await pool.execute(
        """
        INSERT INTO category_intelligence_snapshots (
            category, subcategory, category_path,
            total_reviews, total_brands, total_products,
            top_pain_points, feature_gaps, competitive_flows,
            brand_health, safety_signals, manufacturing_insights,
            top_root_causes
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        ON CONFLICT (category, COALESCE(subcategory, ''), snapshot_date) DO UPDATE SET
            category_path = EXCLUDED.category_path,
            total_reviews = EXCLUDED.total_reviews,
            total_brands = EXCLUDED.total_brands,
            total_products = EXCLUDED.total_products,
            top_pain_points = EXCLUDED.top_pain_points,
            feature_gaps = EXCLUDED.feature_gaps,
            competitive_flows = EXCLUDED.competitive_flows,
            brand_health = EXCLUDED.brand_health,
            safety_signals = EXCLUDED.safety_signals,
            manufacturing_insights = EXCLUDED.manufacturing_insights,
            top_root_causes = EXCLUDED.top_root_causes
        """,
        intel["category"],
        intel.get("subcategory"),
        cat_path_json,
        intel["category_stats"]["total_reviews"],
        intel["category_stats"]["total_brands"],
        intel["category_stats"]["total_products"],
        json.dumps(intel["top_pain_points"]),
        json.dumps(intel["feature_gaps"]),
        json.dumps(intel["competitive_flows"]),
        json.dumps(intel["brand_health"]),
        json.dumps(intel["safety_signals"]),
        json.dumps(intel["manufacturing_insights"]),
        json.dumps(intel["top_root_causes"]),
    )


async def _save_report(pool, subcategory: str, report: dict[str, Any]) -> None:
    """Persist the generated report to intelligence_reports."""
    await pool.execute(
        """
        INSERT INTO intelligence_reports (
            entity_name, entity_type, report_type,
            report_text, structured_data
        ) VALUES ($1, 'subcategory', 'subcategory_intelligence', $2, $3::jsonb)
        """,
        subcategory,
        report.get("analysis_text", ""),
        json.dumps(report, default=str),
    )


# ------------------------------------------------------------------
# LLM report generation
# ------------------------------------------------------------------


async def _generate_report(
    llm, system_prompt: str, intel: dict[str, Any],
    max_tokens: int, temperature: float,
) -> dict[str, Any] | None:
    """Call LLM with subcategory intelligence skill and parse JSON response."""
    from ...pipelines.llm import clean_llm_output
    from ...services.protocols import Message

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=json.dumps(intel, separators=(",", ":"), default=str)),
    ]

    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: llm.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
            ),
            timeout=120,
        )
        _usage = result.get("usage", {})
        if _usage.get("input_tokens"):
            logger.info("subcategory_intelligence LLM tokens: in=%d out=%d",
                         _usage["input_tokens"], _usage.get("output_tokens", 0))
            from ...pipelines.llm import trace_llm_call
            trace_llm_call("task.subcategory_intelligence", input_tokens=_usage["input_tokens"],
                           output_tokens=_usage.get("output_tokens", 0),
                           model=getattr(llm, "model", ""), provider=getattr(llm, "name", ""))
        text = result.get("response", "").strip()
        if not text:
            return None

        text = clean_llm_output(text)

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: extract outermost { ... } block
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    start = -1

        logger.debug("Failed to parse report JSON: %.200s", text)
        return None
    except asyncio.TimeoutError:
        logger.warning("Subcategory report LLM call timed out (120s)")
        return None
    except Exception:
        logger.exception("Subcategory report LLM call failed")
        return None


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: generate subcategory intelligence reports."""
    cfg = settings.subcategory_intelligence
    if not cfg.enabled:
        return {"_skip_synthesis": "Subcategory intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    # Discover qualifying subcategories
    candidates = await _discover_subcategories(pool, cfg)
    if not candidates:
        return {"_skip_synthesis": "No qualifying subcategories found"}

    # Dedup against recent reports
    subcategories = await _filter_dedup(pool, candidates, cfg.dedup_days)
    if not subcategories:
        return {"_skip_synthesis": "All qualifying subcategories have recent reports"}

    # Load LLM + skill
    from ...pipelines.llm import get_pipeline_llm
    llm = get_pipeline_llm(prefer_cloud=False)
    if llm is None:
        return {"_skip_synthesis": "No LLM available"}

    from ...skills import get_skill_registry
    skill = get_skill_registry().get("digest/subcategory_intelligence")
    if not skill:
        return {"_skip_synthesis": "Skill digest/subcategory_intelligence not found"}

    generated = 0
    errors = 0
    processed_subcategories = []

    for sc in subcategories:
        name = sc["subcategory"]
        try:
            intel = await _aggregate_subcategory_intelligence(pool, name)
            if not intel:
                logger.info("Skipping %s: insufficient data after aggregation", name)
                continue

            report = await _generate_report(
                llm, skill.content, intel, cfg.max_tokens, cfg.temperature,
            )
            if not report:
                logger.warning("LLM returned no report for %s", name)
                errors += 1
                continue

            await _save_intelligence_snapshot(pool, intel)
            await _save_report(pool, name, report)

            generated += 1
            processed_subcategories.append(name)
            logger.info("Generated report for subcategory: %s", name)

        except Exception:
            logger.exception("Failed to process subcategory %s", name)
            errors += 1

    result = {
        "generated": generated,
        "errors": errors,
        "candidates": len(candidates),
        "after_dedup": len(subcategories),
        "subcategories": processed_subcategories,
    }

    if generated > 0:
        from ...pipelines.notify import send_pipeline_notification
        msg = (
            f"Generated {generated} subcategory intelligence report(s) "
            f"for: {', '.join(processed_subcategories)}."
        )
        await send_pipeline_notification(
            msg, task, title="Atlas: Subcategory Intelligence",
            default_tags="brain,bar_chart",
        )

    return {"_skip_synthesis": "Subcategory intelligence complete", **result}
