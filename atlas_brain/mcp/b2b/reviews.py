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
        from atlas_brain.autonomous.tasks._b2b_shared import read_review_details
        results = await read_review_details(
            pool,
            window_days=window_days,
            vendor_name=vendor_name,
            pain_category=pain_category,
            min_urgency=min_urgency,
            company=company,
            has_churn_intent=has_churn_intent,
            min_relevance=min_relevance,
            exclude_low_fidelity=exclude_low_fidelity,
            content_type=content_type,
            limit=limit,
        )
        reviews = [
            {
                "id": r.get("id"),
                "vendor_name": r.get("vendor_name"),
                "product_category": r.get("product_category"),
                "reviewer_company": r.get("reviewer_company"),
                "rating": r.get("rating"),
                "urgency_score": r.get("urgency_score"),
                "pain_category": r.get("pain_category"),
                "intent_to_leave": r.get("intent_to_leave"),
                "decision_maker": r.get("decision_maker"),
                "enriched_at": r.get("enriched_at"),
                "reviewer_title": r.get("reviewer_title"),
                "company_size": r.get("company_size"),
                "industry": r.get("industry"),
                "content_type": r.get("content_type"),
                "thread_id": r.get("thread_id"),
                "relevance_score": r.get("relevance_score"),
                "author_churn_score": r.get("author_churn_score"),
                "low_fidelity": r.get("low_fidelity", False),
                "low_fidelity_reasons": r.get("low_fidelity_reasons") or [],
            }
            for r in results
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
            """
            SELECT
                r.*,
                COALESCE(primary_vm.vendor_name, r.vendor_name) AS matched_vendor_name
            FROM b2b_reviews r
            LEFT JOIN LATERAL (
                SELECT vm.vendor_name
                FROM b2b_review_vendor_mentions vm
                WHERE vm.review_id = r.id
                ORDER BY vm.is_primary DESC, vm.id ASC
                LIMIT 1
            ) AS primary_vm ON TRUE
            WHERE r.id = $1
            """,
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
            "vendor_name": row["matched_vendor_name"],
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
