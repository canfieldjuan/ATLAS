"""B2B Churn MCP -- review tools."""
import json
import uuid as _uuid
from typing import Optional

from ._shared import (
    _apply_field_overrides,
    _is_uuid,
    _safe_json,
    _suppress_predicate,
    get_pool,
    logger,
)
from .server import mcp


@mcp.tool()
async def search_reviews(
    vendor_name: Optional[str] = None,
    pain_category: Optional[str] = None,
    min_urgency: Optional[float] = None,
    min_relevance: Optional[float] = None,
    company: Optional[str] = None,
    has_churn_intent: Optional[bool] = None,
    content_type: Optional[str] = None,
    exclude_low_fidelity: bool = False,
    window_days: int = 30,
    limit: int = 20,
) -> str:
    """
    Search enriched reviews with flexible filters.

    vendor_name: Filter by vendor (partial match, case-insensitive)
    pain_category: Filter by pain category (exact match)
    min_urgency: Minimum urgency score (0-10)
    min_relevance: Minimum relevance score (0-1); filters low-signal social posts
    company: Filter by reviewer company (partial match)
    has_churn_intent: Filter by churn intent flag
    content_type: Filter by content type -- one of: review, community_discussion, comment, insider_account
    exclude_low_fidelity: When true, exclude reviews flagged as low quality by the enrichment pipeline
    window_days: How far back to look in days (default 30)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    window_days = max(1, min(window_days, 3650))
    if min_urgency is not None:
        min_urgency = max(0.0, min(min_urgency, 10.0))
    if min_relevance is not None:
        min_relevance = max(0.0, min(min_relevance, 1.0))
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})
        conditions = [
            "enrichment_status = 'enriched'",
            "enriched_at > NOW() - make_interval(days => $1)",
        ]
        params: list = [window_days]
        idx = 2

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        if pain_category:
            conditions.append(f"enrichment->>'pain_category' = ${idx}")
            params.append(pain_category)
            idx += 1

        if min_urgency is not None:
            conditions.append(f"(enrichment->>'urgency_score')::numeric >= ${idx}")
            params.append(min_urgency)
            idx += 1

        if company:
            conditions.append(f"reviewer_company ILIKE '%' || ${idx} || '%'")
            params.append(company)
            idx += 1

        if has_churn_intent is not None:
            conditions.append(
                f"(enrichment->'churn_signals'->>'intent_to_leave')::boolean = ${idx}"
            )
            params.append(has_churn_intent)
            idx += 1

        if min_relevance is not None:
            conditions.append(f"COALESCE(relevance_score, 0.5) >= ${idx}")
            params.append(min_relevance)
            idx += 1

        if exclude_low_fidelity:
            conditions.append("(low_fidelity IS NULL OR low_fidelity = false)")

        if content_type:
            conditions.append(f"content_type = ${idx}")
            params.append(content_type)
            idx += 1
        else:
            conditions.append(_suppress_predicate('review'))

        capped = min(limit, 100)
        params.append(capped)
        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT id, vendor_name, product_category, reviewer_company,
                   rating,
                   (enrichment->>'urgency_score')::numeric AS urgency_score,
                   enrichment->>'pain_category' AS pain_category,
                   (enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
                   (enrichment->'reviewer_context'->>'decision_maker')::boolean AS decision_maker,
                   enriched_at, reviewer_title, company_size_raw,
                   COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry,
                   content_type, thread_id,
                   relevance_score, author_churn_score,
                   low_fidelity, low_fidelity_reasons
            FROM b2b_reviews
            WHERE {where}
            ORDER BY (enrichment->>'urgency_score')::numeric DESC
            LIMIT ${idx}
            """,
            *params,
        )

        reviews = [
            {
                "id": str(r["id"]),
                "vendor_name": r["vendor_name"],
                "product_category": r["product_category"],
                "reviewer_company": r["reviewer_company"],
                "rating": float(r["rating"]) if r["rating"] is not None else None,
                "urgency_score": float(r["urgency_score"]) if r["urgency_score"] is not None else None,
                "pain_category": r["pain_category"],
                "intent_to_leave": r["intent_to_leave"],
                "decision_maker": r["decision_maker"],
                "enriched_at": r["enriched_at"],
                "reviewer_title": r["reviewer_title"],
                "company_size": r["company_size_raw"],
                "industry": r["industry"],
                "content_type": r["content_type"],
                "thread_id": r["thread_id"],
                "relevance_score": float(r["relevance_score"]) if r["relevance_score"] is not None else None,
                "author_churn_score": float(r["author_churn_score"]) if r["author_churn_score"] is not None else None,
                "low_fidelity": bool(r["low_fidelity"]) if r["low_fidelity"] is not None else False,
                "low_fidelity_reasons": _safe_json(r["low_fidelity_reasons"]) or [],
            }
            for r in rows
        ]

        return json.dumps({"reviews": reviews, "count": len(reviews)}, default=str)
    except Exception as exc:
        logger.exception("search_reviews error")
        return json.dumps({"error": "Internal error", "reviews": [], "count": 0})


@mcp.tool()
async def get_review(review_id: str) -> str:
    """
    Fetch a single review with full enrichment data.

    review_id: UUID of the review to retrieve
    """
    if not _is_uuid(review_id):
        return json.dumps({"success": False, "error": "Invalid review_id (must be UUID)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})
        rid = _uuid.UUID(review_id)
        row = await pool.fetchrow(
            "SELECT * FROM b2b_reviews WHERE id = $1",
            rid,
        )

        if not row:
            return json.dumps({"success": False, "error": "Review not found"})

        suppressed = await pool.fetchval(
            """SELECT 1 FROM data_corrections
               WHERE entity_type = 'review' AND entity_id = $1
                 AND correction_type = 'suppress' AND status = 'applied'""",
            row["id"],
        )
        if suppressed:
            return json.dumps({"success": False, "error": "Review not found"})

        review = {
            "id": str(row["id"]),
            "source": row["source"],
            "source_url": row["source_url"],
            "vendor_name": row["vendor_name"],
            "product_name": row["product_name"],
            "product_category": row["product_category"],
            "rating": float(row["rating"]) if row["rating"] is not None else None,
            "summary": row["summary"],
            "review_text": row["review_text"],
            "pros": row["pros"],
            "cons": row["cons"],
            "reviewer_name": row["reviewer_name"],
            "reviewer_title": row["reviewer_title"],
            "reviewer_company": row["reviewer_company"],
            "company_size_raw": row["company_size_raw"],
            "reviewer_industry": row["reviewer_industry"],
            "reviewed_at": row["reviewed_at"],
            "imported_at": row["imported_at"],
            "enrichment": _safe_json(row["enrichment"]),
            "enrichment_status": row["enrichment_status"],
            "enriched_at": row["enriched_at"],
            "relevance_score": float(row["relevance_score"]) if row["relevance_score"] is not None else None,
            "author_churn_score": float(row["author_churn_score"]) if row["author_churn_score"] is not None else None,
            "low_fidelity": bool(row["low_fidelity"]) if row["low_fidelity"] is not None else False,
            "low_fidelity_reasons": _safe_json(row["low_fidelity_reasons"]) or [],
        }
        review = await _apply_field_overrides(pool, "review", str(row["id"]), review)

        return json.dumps({"success": True, "review": review}, default=str)
    except Exception as exc:
        logger.exception("get_review error")
        return json.dumps({"success": False, "error": "Internal error"})
