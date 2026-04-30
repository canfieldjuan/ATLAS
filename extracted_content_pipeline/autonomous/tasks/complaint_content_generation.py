"""
Content generation from complaint analysis: uses Claude (triage LLM) to produce
sellable content -- forum posts, comparison articles, email copy -- from
the highest-pain-score products identified by complaint_analysis.

Runs daily after complaint_analysis (default 10 PM). Reads product_pain_points
and recent complaint_reports, picks the top N products with alternatives, and
generates content for each.

Forces triage LLM (Claude) -- local models do not produce publication-quality
copy. Falls back gracefully if Claude is unavailable.

Returns _skip_synthesis.
"""

import json
import logging
from datetime import date, datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.complaint_content_generation")

# Content types to generate per qualifying product
_CONTENT_TYPES = ["comparison_article", "forum_post", "email_copy"]


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: generate sellable content from pain points."""
    cfg = settings.external_data
    if not cfg.complaint_mining_enabled or not cfg.complaint_content_enabled:
        return {"_skip_synthesis": "Complaint content generation disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    max_products = cfg.complaint_content_max_per_run

    # Find top pain-point products that have alternatives
    candidates = await _fetch_candidates(pool, max_products)
    if not candidates:
        return {"_skip_synthesis": "No products with alternatives to generate content for"}

    # Get category context for enriching prompts
    category_ctx = await _fetch_category_context(pool)

    from ...pipelines.llm import get_pipeline_llm

    llm = get_pipeline_llm(workload="synthesis", try_openrouter=True, auto_activate_ollama=False)
    if llm is None:
        # Last resort: try active LLM
        from ...services import llm_registry
        llm = llm_registry.get_active()
    if llm is None:
        return {"_skip_synthesis": "Claude LLM not available for content generation"}

    generated = 0
    failed = 0
    today = date.today()

    for product in candidates:
        asin = product["asin"]
        # Check what content we've already generated for this ASIN recently
        existing = await _get_existing_content_types(pool, asin)

        for content_type in _CONTENT_TYPES:
            if content_type in existing:
                continue  # Already generated this type

            payload = _build_payload(content_type, product, category_ctx)
            content = await _generate_content(llm, payload, cfg.complaint_content_max_tokens)

            if content:
                try:
                    await pool.execute(
                        """
                        INSERT INTO complaint_content (
                            content_type, category, target_asin, competitor_asin,
                            title, body, pain_point_summary, source_report_date,
                            status, llm_model
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        """,
                        content_type,
                        product.get("category"),
                        asin,
                        (product.get("alternatives") or [{}])[0].get("name"),
                        content.get("title", ""),
                        content.get("body", ""),
                        product.get("top_complaint", ""),
                        today,
                        "draft",
                        "claude",
                    )
                    generated += 1
                except Exception:
                    logger.exception("Failed to store content for %s/%s", asin, content_type)
                    failed += 1
            else:
                failed += 1

    logger.info(
        "Content generation: %d pieces generated, %d failed (from %d products)",
        generated, failed, len(candidates),
    )

    # Send notification
    if generated > 0:
        from ...pipelines.notify import send_pipeline_notification

        msg = f"Generated {generated} content pieces from {len(candidates)} high-pain products. Review drafts in complaint_content table."
        await send_pipeline_notification(
            msg, task, title="Atlas: Complaint Content",
            default_tags="brain,memo",
        )

    return {
        "_skip_synthesis": "Content generation complete",
        "products": len(candidates),
        "generated": generated,
        "failed": failed,
    }


# ------------------------------------------------------------------
# Data fetchers
# ------------------------------------------------------------------


async def _fetch_candidates(pool, limit: int) -> list[dict[str, Any]]:
    """Fetch top pain-point products that have reviewer-mentioned alternatives."""
    rows = await pool.fetch(
        """
        SELECT asin, product_name, category,
               complaint_reviews, pain_score,
               top_complaints, root_cause_distribution,
               alternative_products
        FROM product_pain_points
        WHERE pain_score >= 4.0
          AND alternative_products != '[]'::jsonb
        ORDER BY pain_score DESC, complaint_reviews DESC
        LIMIT $1
        """,
        limit,
    )

    result = []
    for r in rows:
        alternatives = r["alternative_products"]
        if isinstance(alternatives, str):
            try:
                alternatives = json.loads(alternatives)
            except (json.JSONDecodeError, TypeError):
                alternatives = []

        if not alternatives:
            continue

        top_complaints = r["top_complaints"]
        if isinstance(top_complaints, str):
            try:
                top_complaints = json.loads(top_complaints)
            except (json.JSONDecodeError, TypeError):
                top_complaints = []

        root_causes = r["root_cause_distribution"]
        if isinstance(root_causes, str):
            try:
                root_causes = json.loads(root_causes)
            except (json.JSONDecodeError, TypeError):
                root_causes = {}

        # Get avg rating from reviews
        rating_row = await pool.fetchrow(
            "SELECT avg(rating) AS avg_rating FROM product_reviews WHERE asin = $1",
            r["asin"],
        )
        avg_rating = round(float(rating_row["avg_rating"]), 1) if rating_row and rating_row["avg_rating"] else 0.0

        result.append({
            "asin": r["asin"],
            "product_name": r["product_name"] or r["asin"],
            "category": r["category"],
            "complaint_count": r["complaint_reviews"],
            "avg_pain_score": float(r["pain_score"]),
            "avg_rating": avg_rating,
            "top_complaints": top_complaints[:5] if isinstance(top_complaints, list) else [],
            "root_causes": root_causes if isinstance(root_causes, dict) else {},
            "alternatives": alternatives[:3] if isinstance(alternatives, list) else [],
            "top_complaint": top_complaints[0] if isinstance(top_complaints, list) and top_complaints else "",
        })

    return result


async def _fetch_category_context(pool) -> dict[str, dict[str, Any]]:
    """Fetch per-category aggregate context for content enrichment."""
    rows = await pool.fetch(
        """
        SELECT source_category AS category,
               count(*) AS total_complaints,
               avg(pain_score) AS avg_pain_score,
               mode() WITHIN GROUP (ORDER BY root_cause) AS top_root_cause
        FROM product_reviews
        WHERE enrichment_status = 'enriched'
        GROUP BY source_category
        """,
    )
    return {
        r["category"]: {
            "category": r["category"],
            "total_complaints": r["total_complaints"],
            "avg_pain_score": round(float(r["avg_pain_score"]), 1) if r["avg_pain_score"] else 0.0,
            "top_root_cause": r["top_root_cause"],
        }
        for r in rows
    }


async def _get_existing_content_types(pool, asin: str) -> set[str]:
    """Check which content types already exist for this ASIN (within last 30 days)."""
    rows = await pool.fetch(
        """
        SELECT DISTINCT content_type
        FROM complaint_content
        WHERE target_asin = $1
          AND created_at > NOW() - INTERVAL '30 days'
        """,
        asin,
    )
    return {r["content_type"] for r in rows}


# ------------------------------------------------------------------
# Payload & LLM
# ------------------------------------------------------------------


def _build_payload(
    content_type: str,
    product: dict[str, Any],
    category_ctx: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the LLM input payload for content generation."""
    # Get manufacturing suggestions from reviews if available
    mfg = []
    for complaint in product.get("top_complaints", []):
        if isinstance(complaint, str) and complaint:
            mfg.append(complaint)

    return {
        "content_type": content_type,
        "target_product": {
            "asin": product["asin"],
            "product_name": product.get("product_name", product["asin"]),
            "category": product.get("category", ""),
            "complaint_count": product.get("complaint_count", 0),
            "avg_pain_score": product.get("avg_pain_score", 0),
            "avg_rating": product.get("avg_rating", 0),
            "top_complaints": product.get("top_complaints", []),
            "root_causes": product.get("root_causes", {}),
            "manufacturing_suggestions": mfg[:3],
        },
        "alternatives": product.get("alternatives", []),
        "category_context": category_ctx.get(product.get("category", ""), {}),
    }


async def _generate_content(
    llm, payload: dict[str, Any], max_tokens: int
) -> dict[str, Any] | None:
    """Call LLM with content generation skill and parse response."""
    from ...pipelines.llm import clean_llm_output
    from ...skills import get_skill_registry
    from ...services.protocols import Message

    skill = get_skill_registry().get("digest/complaint_content_generation")
    if not skill:
        logger.warning("Skill 'digest/complaint_content_generation' not found")
        return None

    messages = [
        Message(role="system", content=skill.content),
        Message(role="user", content=json.dumps(payload, separators=(",", ":"), default=str)),
    ]

    try:
        result = llm.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,  # Higher temp for creative writing
        )
        _usage = result.get("usage", {})
        if _usage.get("input_tokens"):
            logger.info("complaint_content_generation LLM tokens: in=%d out=%d",
                         _usage["input_tokens"], _usage.get("output_tokens", 0))
            from ...pipelines.llm import trace_llm_call
            _trace = result.get("_trace_meta", {})
            _target = payload.get("target_product", {})
            trace_llm_call("task.complaint_content_generation", input_tokens=_usage["input_tokens"],
                           output_tokens=_usage.get("output_tokens", 0),
                           cached_tokens=_trace.get("cached_tokens") or _trace.get("cache_read_tokens"),
                           cache_write_tokens=_trace.get("cache_write_tokens") or _trace.get("cache_creation_tokens"),
                           billable_input_tokens=_trace.get("billable_input_tokens"),
                           model=getattr(llm, "model", ""), provider=getattr(llm, "name", ""),
                           input_data={
                               "messages": [
                                   {"role": m.role, "content": m.content[:500]} for m in messages
                               ],
                           },
                           output_data={
                               "response": result.get("response", "")[:2000],
                           },
                           api_endpoint=_trace.get("api_endpoint"),
                           provider_request_id=_trace.get("provider_request_id"),
                           ttft_ms=_trace.get("ttft_ms"),
                           inference_time_ms=_trace.get("inference_time_ms"),
                           queue_time_ms=_trace.get("queue_time_ms"),
                           metadata={
                               "workflow": "complaint_content_generation",
                               "content_type": payload.get("content_type", ""),
                               "asin": _target.get("asin", ""),
                               "category": _target.get("category", ""),
                               "avg_pain_score": _target.get("avg_pain_score", 0),
                               "avg_rating": _target.get("avg_rating", 0),
                           })
        text = result.get("response", "").strip()
        if not text:
            return None

        text = clean_llm_output(text)

        parsed = json.loads(text)
        if not isinstance(parsed, dict) or "body" not in parsed:
            logger.debug("Content generation missing 'body' field")
            return None

        return parsed

    except json.JSONDecodeError:
        logger.debug("Failed to parse content generation JSON: %.200s", text)
        return None
    except Exception:
        logger.exception("Content generation LLM call failed")
        return None
