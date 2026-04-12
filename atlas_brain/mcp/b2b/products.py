"""B2B Churn MCP -- product intelligence tools."""
import json
from typing import Optional

from ._shared import _safe_json, logger, get_pool
from .server import mcp


def _clean_optional_text(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _clean_required_text(value: Optional[str]) -> str | None:
    return _clean_optional_text(value)


@mcp.tool()
async def get_product_profile(vendor_name: str) -> str:
    """
    Fetch a pre-computed product profile knowledge card for a vendor.

    vendor_name: Vendor name (fuzzy match, case-insensitive)

    Returns strengths, weaknesses, pain addressed scores, competitive
    positioning, use cases, company size fit, and LLM-generated summary.
    """
    clean_vendor_name = _clean_required_text(vendor_name)
    if clean_vendor_name is None:
        return json.dumps({"success": False, "error": "vendor_name is required"})

    try:
        pool = get_pool()
        row = await pool.fetchrow(
            """
            SELECT id, vendor_name, product_category,
                   strengths, weaknesses, pain_addressed,
                   total_reviews_analyzed, avg_rating, recommend_rate, avg_urgency,
                   primary_use_cases, typical_company_size, typical_industries,
                   top_integrations, commonly_compared_to, commonly_switched_from,
                   profile_summary, confidence_score, last_computed_at, created_at
            FROM b2b_product_profiles
            WHERE vendor_name ILIKE '%' || $1 || '%'
            ORDER BY total_reviews_analyzed DESC
            LIMIT 1
            """,
            clean_vendor_name,
        )

        if not row:
            return json.dumps({"success": False, "error": f"No product profile found for '{clean_vendor_name}'"})

        profile = {
            "id": str(row["id"]),
            "vendor_name": row["vendor_name"],
            "product_category": row["product_category"],
            "strengths": _safe_json(row["strengths"]),
            "weaknesses": _safe_json(row["weaknesses"]),
            "pain_addressed": _safe_json(row["pain_addressed"]),
            "total_reviews_analyzed": row["total_reviews_analyzed"],
            "avg_rating": float(row["avg_rating"]) if row["avg_rating"] is not None else None,
            "recommend_rate": float(row["recommend_rate"]) if row["recommend_rate"] is not None else None,
            "avg_urgency": float(row["avg_urgency"]) if row["avg_urgency"] is not None else None,
            "primary_use_cases": _safe_json(row["primary_use_cases"]),
            "typical_company_size": _safe_json(row["typical_company_size"]),
            "typical_industries": _safe_json(row["typical_industries"]),
            "top_integrations": _safe_json(row["top_integrations"]),
            "commonly_compared_to": _safe_json(row["commonly_compared_to"]),
            "commonly_switched_from": _safe_json(row["commonly_switched_from"]),
            "profile_summary": row["profile_summary"],
            "confidence_score": float(row["confidence_score"]) if row["confidence_score"] is not None else 0,
            "last_computed_at": row["last_computed_at"],
            "created_at": row["created_at"],
        }

        return json.dumps({"success": True, "profile": profile}, default=str)
    except Exception:
        logger.exception("get_product_profile error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def get_product_profile_history(
    vendor_name: str,
    days: int = 90,
    limit: int = 90,
) -> str:
    """
    Get daily product profile snapshots for a vendor over time.

    vendor_name: Vendor name (case-insensitive partial match)
    days: How many days back to look (default 90)
    limit: Max snapshots to return (default 90, max 365)
    """
    clean_vendor_name = _clean_required_text(vendor_name)
    if clean_vendor_name is None:
        return json.dumps({"error": "vendor_name is required"})

    limit = min(max(limit, 1), 365)
    days = min(max(days, 1), 365)
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        rows = await pool.fetch(
            """
            SELECT vendor_name, snapshot_date,
                   total_reviews_analyzed, avg_rating, recommend_rate, avg_urgency,
                   strength_count, weakness_count, top_strength, top_weakness,
                   top_use_case, top_integration,
                   compared_to_count, switched_from_count,
                   pain_categories_covered, profile_summary_len
            FROM b2b_product_profile_snapshots
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND snapshot_date >= CURRENT_DATE - $2::int
            ORDER BY snapshot_date DESC
            LIMIT $3
            """,
            clean_vendor_name, days, limit,
        )

        if not rows:
            return json.dumps({
                "vendor_name": clean_vendor_name,
                "snapshots": [],
                "count": 0,
                "message": f"No product profile snapshots found for '{clean_vendor_name}'",
            })

        resolved = rows[0]["vendor_name"]
        snapshots = []
        for r in rows:
            snapshots.append({
                "snapshot_date": str(r["snapshot_date"]),
                "total_reviews_analyzed": r["total_reviews_analyzed"],
                "avg_rating": float(r["avg_rating"]) if r["avg_rating"] is not None else None,
                "recommend_rate": float(r["recommend_rate"]) if r["recommend_rate"] is not None else None,
                "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] is not None else None,
                "strength_count": r["strength_count"],
                "weakness_count": r["weakness_count"],
                "top_strength": r["top_strength"],
                "top_weakness": r["top_weakness"],
                "top_use_case": r["top_use_case"],
                "top_integration": r["top_integration"],
                "compared_to_count": r["compared_to_count"],
                "switched_from_count": r["switched_from_count"],
                "pain_categories_covered": r["pain_categories_covered"],
                "profile_summary_len": r["profile_summary_len"],
            })

        return json.dumps({
            "vendor_name": resolved,
            "snapshots": snapshots,
            "count": len(snapshots),
        }, default=str)
    except Exception:
        logger.exception("get_product_profile_history error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def match_products_tool(
    churning_from: str,
    pain_categories: Optional[str] = None,
    company_size: Optional[int] = None,
    industry: Optional[str] = None,
    limit: int = 3,
) -> str:
    """
    Find the best product alternatives for a company churning from a vendor.

    Scores all product profiles against the company's pain points using a
    weighted algorithm (pain alignment, displacement evidence, company size
    fit, satisfaction delta, recommend rate).

    churning_from: Vendor name the company is leaving (required)
    pain_categories: JSON array of pain objects, e.g. [{"category": "pricing", "severity": "primary"}]
    company_size: Number of employees (optional)
    industry: Company industry (optional)
    limit: Max results (default 3, cap 10)
    """
    clean_churning_from = _clean_required_text(churning_from)
    if clean_churning_from is None:
        return json.dumps({"success": False, "error": "churning_from is required"})
    clean_industry = _clean_optional_text(industry)
    clean_pain_categories = _clean_optional_text(pain_categories)

    limit = max(1, min(limit, 10))

    # Parse pain_categories from JSON string
    pains: list[dict] = []
    if clean_pain_categories:
        try:
            parsed = json.loads(clean_pain_categories)
            if isinstance(parsed, list):
                pains = parsed
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"success": False, "error": "pain_categories must be a valid JSON array"})

    try:
        from atlas_brain.services.b2b.product_matching import match_products

        pool = get_pool()
        matches = await match_products(
            churning_from=clean_churning_from,
            pain_categories=pains,
            company_size=company_size,
            industry=clean_industry,
            pool=pool,
            limit=limit,
        )

        return json.dumps({"success": True, "matches": matches, "count": len(matches)}, default=str)
    except Exception:
        logger.exception("match_products error")
        return json.dumps({"success": False, "error": "Internal error", "matches": [], "count": 0})
