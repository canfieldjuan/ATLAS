"""
Read-only REST endpoints for the Consumer Competitive Intelligence Dashboard.

Queries ``product_reviews`` (deep_extraction JSONB) and ``product_metadata``
tables to surface brand analysis, competitive flow mapping, feature gaps,
safety signals, and buyer psychology.
"""

import json
import logging
import uuid as _uuid
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.consumer_dashboard")

router = APIRouter(prefix="/consumer/dashboard", tags=["consumer-dashboard"])


def _safe_json(val):
    """Return val if already a list/dict, else try json.loads, else as-is."""
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return val


def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _pool_or_503():
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


# ---------------------------------------------------------------------------
# GET /pipeline
# ---------------------------------------------------------------------------


@router.get("/pipeline")
async def get_pipeline_status():
    pool = _pool_or_503()

    enrichment_rows = await pool.fetch(
        """
        SELECT enrichment_status, COUNT(*) AS cnt
        FROM product_reviews
        GROUP BY enrichment_status
        """
    )
    enrichment_counts = {r["enrichment_status"]: r["cnt"] for r in enrichment_rows}

    deep_rows = await pool.fetch(
        """
        SELECT deep_enrichment_status, COUNT(*) AS cnt
        FROM product_reviews
        GROUP BY deep_enrichment_status
        """
    )
    deep_counts = {r["deep_enrichment_status"]: r["cnt"] for r in deep_rows}

    category_rows = await pool.fetch(
        """
        SELECT source_category, COUNT(*) AS cnt
        FROM product_reviews
        GROUP BY source_category
        """
    )
    category_counts = {r["source_category"]: r["cnt"] for r in category_rows}

    totals = await pool.fetchrow(
        """
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched,
               COUNT(*) FILTER (WHERE deep_enrichment_status = 'enriched') AS deep_enriched,
               MAX(enriched_at) AS last_enrichment_at,
               MAX(deep_enriched_at) AS last_deep_enrichment_at
        FROM product_reviews
        """
    )

    return {
        "enrichment_counts": enrichment_counts,
        "deep_enrichment_counts": deep_counts,
        "category_counts": category_counts,
        "total_reviews": totals["total"] if totals else 0,
        "enriched": totals["enriched"] if totals else 0,
        "deep_enriched": totals["deep_enriched"] if totals else 0,
        "last_enrichment_at": str(totals["last_enrichment_at"]) if totals and totals["last_enrichment_at"] else None,
        "last_deep_enrichment_at": str(totals["last_deep_enrichment_at"]) if totals and totals["last_deep_enrichment_at"] else None,
    }


# ---------------------------------------------------------------------------
# GET /brands
# ---------------------------------------------------------------------------


@router.get("/brands")
async def list_brands(
    source_category: Optional[str] = Query(None),
    min_reviews: int = Query(0),
    search: Optional[str] = Query(None),
    sort_by: str = Query("review_count"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    pool = _pool_or_503()
    conditions: list[str] = ["pm.brand IS NOT NULL", "pm.brand != ''"]
    params: list = []
    idx = 1

    if source_category:
        conditions.append(f"pr.source_category = ${idx}")
        params.append(source_category)
        idx += 1

    if search:
        conditions.append(f"pm.brand ILIKE '%' || ${idx} || '%'")
        params.append(search)
        idx += 1

    where = " AND ".join(conditions)

    sort_map = {
        "review_count": "review_count DESC",
        "avg_complaint_score": "avg_complaint_score DESC NULLS LAST",
        "avg_praise_score": "avg_praise_score DESC NULLS LAST",
        "avg_rating": "pm_avg_rating DESC NULLS LAST",
        "safety_count": "safety_count DESC",
        "brand": "brand ASC",
    }
    order = sort_map.get(sort_by, "review_count DESC")

    if min_reviews > 0:
        having = f"HAVING COUNT(pr.id) >= ${idx}"
        params.append(min_reviews)
        idx += 1
    else:
        having = ""

    params.extend([limit, offset])
    limit_idx = idx
    offset_idx = idx + 1

    rows = await pool.fetch(
        f"""
        SELECT pm.brand,
               COUNT(DISTINCT pm.asin) AS product_count,
               COUNT(pr.id) AS review_count,
               AVG(pm.average_rating) AS pm_avg_rating,
               SUM(pm.rating_number) AS total_ratings,
               AVG(pr.pain_score) FILTER (WHERE pr.rating <= 3) AS avg_complaint_score,
               AVG(pr.pain_score) FILTER (WHERE pr.rating > 3)  AS avg_praise_score,
               COUNT(*) FILTER (WHERE pr.rating <= 3) AS complaint_count,
               COUNT(*) FILTER (WHERE pr.rating > 3)  AS praise_count,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction IS NOT NULL
                     AND pr.deep_extraction != '{{}}'::jsonb
                     AND pr.deep_extraction->'safety_flag'->>'flagged' = 'true'
               ) AS safety_count
        FROM product_metadata pm
        JOIN product_reviews pr ON pr.asin = pm.asin
        WHERE {where}
        GROUP BY pm.brand
        {having}
        ORDER BY {order}
        LIMIT ${limit_idx} OFFSET ${offset_idx}
        """,
        *params,
    )

    brands = [
        {
            "brand": r["brand"],
            "product_count": r["product_count"],
            "review_count": r["review_count"],
            "avg_rating": _safe_float(r["pm_avg_rating"]),
            "total_ratings": r["total_ratings"],
            "avg_complaint_score": _safe_float(r["avg_complaint_score"]),
            "avg_praise_score": _safe_float(r["avg_praise_score"]),
            "complaint_count": r["complaint_count"],
            "praise_count": r["praise_count"],
            "safety_count": r["safety_count"],
        }
        for r in rows
    ]

    return {"brands": brands, "count": len(brands)}


# ---------------------------------------------------------------------------
# GET /brands/{brand_name}
# ---------------------------------------------------------------------------


@router.get("/brands/{brand_name}")
async def get_brand_detail(brand_name: str):
    pool = _pool_or_503()
    bname = brand_name.strip()

    # Products for this brand
    products = await pool.fetch(
        """
        SELECT pm.asin, pm.title, pm.average_rating, pm.rating_number, pm.price,
               COUNT(pr.id) AS review_count,
               AVG(pr.pain_score) FILTER (WHERE pr.rating <= 3) AS avg_complaint_score,
               AVG(pr.pain_score) FILTER (WHERE pr.rating > 3)  AS avg_praise_score,
               COUNT(*) FILTER (WHERE pr.rating <= 3) AS complaint_count,
               COUNT(*) FILTER (WHERE pr.rating > 3)  AS praise_count
        FROM product_metadata pm
        LEFT JOIN product_reviews pr ON pr.asin = pm.asin
        WHERE pm.brand ILIKE $1
        GROUP BY pm.asin, pm.title, pm.average_rating, pm.rating_number, pm.price
        ORDER BY review_count DESC
        """,
        bname,
    )

    if not products:
        raise HTTPException(status_code=404, detail="Brand not found")

    # Aggregate sentiment aspects
    aspect_rows = await pool.fetch(
        """
        SELECT deep_extraction->'sentiment_aspects' AS aspects
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand ILIKE $1
          AND pr.deep_extraction IS NOT NULL
          AND pr.deep_extraction != '{}'::jsonb
          AND pr.deep_extraction->'sentiment_aspects' IS NOT NULL
        """,
        bname,
    )

    sentiment: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0, "mixed": 0, "neutral": 0})
    for row in aspect_rows:
        aspects = _safe_json(row["aspects"])
        if isinstance(aspects, list):
            for asp in aspects:
                if isinstance(asp, dict):
                    name = asp.get("aspect", "unknown")
                    s = asp.get("sentiment", "neutral")
                    if s in ("positive", "negative", "mixed", "neutral"):
                        sentiment[name][s] += 1

    sentiment_list = [
        {"aspect": k, **v}
        for k, v in sorted(sentiment.items(), key=lambda x: sum(x[1].values()), reverse=True)[:20]
    ]

    # Feature requests
    feat_rows = await pool.fetch(
        """
        SELECT deep_extraction->'feature_requests' AS requests
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand ILIKE $1
          AND pr.deep_extraction IS NOT NULL
          AND pr.deep_extraction != '{}'::jsonb
          AND jsonb_array_length(COALESCE(pr.deep_extraction->'feature_requests', '[]'::jsonb)) > 0
        """,
        bname,
    )

    feature_counter: dict[str, int] = defaultdict(int)
    for row in feat_rows:
        reqs = _safe_json(row["requests"])
        if isinstance(reqs, list):
            for req in reqs:
                text = req if isinstance(req, str) else (req.get("request", str(req)) if isinstance(req, dict) else str(req))
                feature_counter[text.strip().lower()] += 1

    top_features = [
        {"request": k, "count": v}
        for k, v in sorted(feature_counter.items(), key=lambda x: x[1], reverse=True)[:15]
    ]

    # Competitive flows
    flow_rows = await pool.fetch(
        """
        SELECT deep_extraction->'product_comparisons' AS comparisons, pr.rating
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand ILIKE $1
          AND pr.deep_extraction IS NOT NULL
          AND pr.deep_extraction != '{}'::jsonb
          AND jsonb_array_length(COALESCE(pr.deep_extraction->'product_comparisons', '[]'::jsonb)) > 0
        """,
        bname,
    )

    comp_counter: dict[str, dict] = {}
    for row in flow_rows:
        comps = _safe_json(row["comparisons"])
        if isinstance(comps, list):
            for comp in comps:
                if isinstance(comp, dict):
                    other = comp.get("product_name") or comp.get("product", "Unknown")
                    direction = comp.get("direction", "compared")
                    key = f"{other}|{direction}"
                    if key not in comp_counter:
                        comp_counter[key] = {"brand": other, "direction": direction, "count": 0, "ratings": []}
                    comp_counter[key]["count"] += 1
                    if row["rating"] is not None:
                        comp_counter[key]["ratings"].append(float(row["rating"]))

    flows = sorted(
        [
            {
                "brand": v["brand"],
                "direction": v["direction"],
                "count": v["count"],
                "avg_rating": round(sum(v["ratings"]) / len(v["ratings"]), 2) if v["ratings"] else None,
            }
            for v in comp_counter.values()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )[:15]

    # Loyalty breakdown
    loyalty_rows = await pool.fetch(
        """
        SELECT deep_extraction->>'brand_loyalty_depth' AS loyalty
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand ILIKE $1
          AND pr.deep_extraction IS NOT NULL
          AND pr.deep_extraction != '{}'::jsonb
          AND pr.deep_extraction->>'brand_loyalty_depth' IS NOT NULL
        """,
        bname,
    )

    loyalty_counter: dict[str, int] = defaultdict(int)
    for row in loyalty_rows:
        loyalty_counter[row["loyalty"]] += 1

    loyalty_list = [{"level": k, "count": v} for k, v in loyalty_counter.items()]

    # Buyer profile distributions
    buyer_rows = await pool.fetch(
        """
        SELECT deep_extraction->>'expertise_level' AS expertise,
               deep_extraction->>'budget_type' AS budget,
               deep_extraction->>'discovery_channel' AS channel
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand ILIKE $1
          AND pr.deep_extraction IS NOT NULL
          AND pr.deep_extraction != '{}'::jsonb
        """,
        bname,
    )

    expertise_counter: dict[str, int] = defaultdict(int)
    budget_counter: dict[str, int] = defaultdict(int)
    channel_counter: dict[str, int] = defaultdict(int)
    for row in buyer_rows:
        if row["expertise"]:
            expertise_counter[row["expertise"]] += 1
        if row["budget"]:
            budget_counter[row["budget"]] += 1
        if row["channel"]:
            channel_counter[row["channel"]] += 1

    # Totals
    total_reviews = sum(r["review_count"] for r in products)
    avg_rating_all = await pool.fetchval(
        "SELECT AVG(average_rating) FROM product_metadata WHERE brand ILIKE $1", bname
    )

    return {
        "brand": bname,
        "product_count": len(products),
        "total_reviews": total_reviews,
        "avg_rating": _safe_float(avg_rating_all),
        "products": [
            {
                "asin": r["asin"],
                "title": r["title"],
                "average_rating": _safe_float(r["average_rating"]),
                "rating_number": r["rating_number"],
                "price": r["price"],
                "review_count": r["review_count"],
                "avg_complaint_score": _safe_float(r["avg_complaint_score"]),
                "avg_praise_score": _safe_float(r["avg_praise_score"]),
                "complaint_count": r["complaint_count"],
                "praise_count": r["praise_count"],
            }
            for r in products
        ],
        "sentiment_aspects": sentiment_list,
        "top_features": top_features,
        "competitive_flows": flows,
        "loyalty_breakdown": loyalty_list,
        "buyer_profile": {
            "expertise": [{"level": k, "count": v} for k, v in expertise_counter.items()],
            "budget": [{"type": k, "count": v} for k, v in budget_counter.items()],
            "discovery_channel": [{"channel": k, "count": v} for k, v in channel_counter.items()],
        },
    }


# ---------------------------------------------------------------------------
# GET /flows
# ---------------------------------------------------------------------------


@router.get("/flows")
async def get_competitive_flows(
    source_category: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    min_count: int = Query(2),
    limit: int = Query(100, le=500),
):
    pool = _pool_or_503()
    conditions = [
        "pr.deep_extraction IS NOT NULL",
        "pr.deep_extraction != '{}'::jsonb",
        "jsonb_array_length(COALESCE(pr.deep_extraction->'product_comparisons', '[]'::jsonb)) > 0",
    ]
    params: list = []
    idx = 1

    if source_category:
        conditions.append(f"pr.source_category = ${idx}")
        params.append(source_category)
        idx += 1

    if brand:
        conditions.append(f"pm.brand ILIKE '%' || ${idx} || '%'")
        params.append(brand)
        idx += 1

    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT deep_extraction->'product_comparisons' AS comparisons,
               pm.brand, pr.asin, pr.rating
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE {where}
        """,
        *params,
    )

    # Build flow graph: {from_brand -> to_brand -> direction -> stats}
    flow_map: dict[str, dict] = {}
    for row in rows:
        comps = _safe_json(row["comparisons"])
        from_brand = row["brand"] or "Unknown"
        if isinstance(comps, list):
            for comp in comps:
                if isinstance(comp, dict):
                    to_brand = comp.get("product_name") or comp.get("product", "Unknown")
                    direction = comp.get("direction", "compared")
                    key = f"{from_brand}|{to_brand}|{direction}"
                    if key not in flow_map:
                        flow_map[key] = {
                            "from_brand": from_brand,
                            "to_brand": to_brand,
                            "direction": direction,
                            "count": 0,
                            "ratings": [],
                        }
                    flow_map[key]["count"] += 1
                    if row["rating"] is not None:
                        flow_map[key]["ratings"].append(float(row["rating"]))

    flows = sorted(
        [
            {
                "from_brand": v["from_brand"],
                "to_brand": v["to_brand"],
                "direction": v["direction"],
                "count": v["count"],
                "avg_rating": round(sum(v["ratings"]) / len(v["ratings"]), 2) if v["ratings"] else None,
            }
            for v in flow_map.values()
            if v["count"] >= min_count
        ],
        key=lambda x: x["count"],
        reverse=True,
    )[:limit]

    return {"flows": flows, "count": len(flows)}


# ---------------------------------------------------------------------------
# GET /features
# ---------------------------------------------------------------------------


@router.get("/features")
async def get_feature_gaps(
    source_category: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    pool = _pool_or_503()
    conditions = [
        "pr.deep_extraction IS NOT NULL",
        "pr.deep_extraction != '{}'::jsonb",
    ]
    params: list = []
    idx = 1

    if source_category:
        conditions.append(f"pr.source_category = ${idx}")
        params.append(source_category)
        idx += 1

    if brand:
        conditions.append(f"pm.brand ILIKE '%' || ${idx} || '%'")
        params.append(brand)
        idx += 1

    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT deep_extraction->'feature_requests' AS requests,
               deep_extraction->'sentiment_aspects' AS aspects,
               pm.brand, pr.rating
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE {where}
        """,
        *params,
    )

    # Feature requests aggregation
    feature_map: dict[str, dict] = {}
    for row in rows:
        reqs = _safe_json(row["requests"])
        if isinstance(reqs, list):
            for req in reqs:
                text = req if isinstance(req, str) else (req.get("request", str(req)) if isinstance(req, dict) else str(req))
                text = text.strip().lower()
                if not text:
                    continue
                if text not in feature_map:
                    feature_map[text] = {"request": text, "count": 0, "brands": set(), "ratings": []}
                feature_map[text]["count"] += 1
                if row["brand"]:
                    feature_map[text]["brands"].add(row["brand"])
                if row["rating"] is not None:
                    feature_map[text]["ratings"].append(float(row["rating"]))

    top_features = sorted(
        [
            {
                "request": v["request"],
                "count": v["count"],
                "brands_affected": len(v["brands"]),
                "brand_list": sorted(v["brands"])[:5],
                "avg_rating": round(sum(v["ratings"]) / len(v["ratings"]), 2) if v["ratings"] else None,
            }
            for v in feature_map.values()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )[:limit]

    # Negative sentiment aspects aggregation
    aspect_map: dict[str, dict] = {}
    for row in rows:
        aspects = _safe_json(row["aspects"])
        if isinstance(aspects, list):
            for asp in aspects:
                if isinstance(asp, dict):
                    name = asp.get("aspect", "")
                    if not name:
                        continue
                    s = asp.get("sentiment", "neutral")
                    if name not in aspect_map:
                        aspect_map[name] = {"aspect": name, "positive": 0, "negative": 0, "mixed": 0, "neutral": 0, "brands": set()}
                    if s in ("positive", "negative", "mixed", "neutral"):
                        aspect_map[name][s] += 1
                    if row["brand"]:
                        aspect_map[name]["brands"].add(row["brand"])

    negative_aspects = sorted(
        [
            {
                "aspect": v["aspect"],
                "negative": v["negative"],
                "total": v["positive"] + v["negative"] + v["mixed"] + v["neutral"],
                "pct_negative": round(v["negative"] / max(v["positive"] + v["negative"] + v["mixed"] + v["neutral"], 1) * 100, 1),
                "top_brands": sorted(v["brands"])[:5],
            }
            for v in aspect_map.values()
            if v["negative"] > 0
        ],
        key=lambda x: x["negative"],
        reverse=True,
    )[:limit]

    return {
        "feature_requests": top_features,
        "negative_aspects": negative_aspects,
    }


# ---------------------------------------------------------------------------
# GET /safety
# ---------------------------------------------------------------------------


@router.get("/safety")
async def get_safety_signals(
    source_category: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    pool = _pool_or_503()
    conditions = [
        "pr.deep_extraction->'safety_flag'->>'flagged' = 'true'",
    ]
    params: list = []
    idx = 1

    if source_category:
        conditions.append(f"pr.source_category = ${idx}")
        params.append(source_category)
        idx += 1

    if brand:
        conditions.append(f"pm.brand ILIKE '%' || ${idx} || '%'")
        params.append(brand)
        idx += 1

    where = " AND ".join(conditions)
    params.append(min(limit, 200))

    rows = await pool.fetch(
        f"""
        SELECT pr.id, pr.asin, pr.rating, pr.summary,
               LEFT(pr.review_text, 300) AS review_excerpt,
               pr.deep_extraction->'safety_flag' AS safety_flag,
               pm.brand, pm.title, pr.imported_at
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE {where}
        ORDER BY pr.imported_at DESC
        LIMIT ${idx}
        """,
        *params,
    )

    signals = [
        {
            "id": str(r["id"]),
            "asin": r["asin"],
            "rating": _safe_float(r["rating"]),
            "summary": r["summary"],
            "review_excerpt": r["review_excerpt"],
            "safety_flag": _safe_json(r["safety_flag"]),
            "brand": r["brand"],
            "title": r["title"],
            "imported_at": str(r["imported_at"]) if r["imported_at"] else None,
        }
        for r in rows
    ]

    total_flagged = await pool.fetchval(
        "SELECT COUNT(*) FROM product_reviews WHERE deep_extraction->'safety_flag'->>'flagged' = 'true'"
    )

    return {
        "signals": signals,
        "count": len(signals),
        "total_flagged": total_flagged or 0,
    }


# ---------------------------------------------------------------------------
# GET /reviews
# ---------------------------------------------------------------------------


@router.get("/reviews")
async def search_reviews(
    source_category: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    asin: Optional[str] = Query(None),
    min_rating: Optional[float] = Query(None),
    max_rating: Optional[float] = Query(None),
    root_cause: Optional[str] = Query(None),
    has_comparisons: Optional[bool] = Query(None),
    has_feature_requests: Optional[bool] = Query(None),
    search: Optional[str] = Query(None),
    sort_by: str = Query("imported_at"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    pool = _pool_or_503()
    conditions: list[str] = []
    params: list = []
    idx = 1

    if source_category:
        conditions.append(f"pr.source_category = ${idx}")
        params.append(source_category)
        idx += 1

    if brand:
        conditions.append(f"pm.brand ILIKE '%' || ${idx} || '%'")
        params.append(brand)
        idx += 1

    if asin:
        conditions.append(f"pr.asin = ${idx}")
        params.append(asin)
        idx += 1

    if min_rating is not None:
        conditions.append(f"pr.rating >= ${idx}")
        params.append(min_rating)
        idx += 1

    if max_rating is not None:
        conditions.append(f"pr.rating <= ${idx}")
        params.append(max_rating)
        idx += 1

    if root_cause:
        conditions.append(f"pr.root_cause = ${idx}")
        params.append(root_cause)
        idx += 1

    if has_comparisons is True:
        conditions.append(
            "pr.deep_extraction IS NOT NULL AND pr.deep_extraction != '{}'::jsonb "
            "AND jsonb_array_length(COALESCE(pr.deep_extraction->'product_comparisons', '[]'::jsonb)) > 0"
        )
    elif has_comparisons is False:
        conditions.append(
            "(pr.deep_extraction IS NULL OR pr.deep_extraction = '{}'::jsonb "
            "OR jsonb_array_length(COALESCE(pr.deep_extraction->'product_comparisons', '[]'::jsonb)) = 0)"
        )

    if has_feature_requests is True:
        conditions.append(
            "pr.deep_extraction IS NOT NULL AND pr.deep_extraction != '{}'::jsonb "
            "AND jsonb_array_length(COALESCE(pr.deep_extraction->'feature_requests', '[]'::jsonb)) > 0"
        )
    elif has_feature_requests is False:
        conditions.append(
            "(pr.deep_extraction IS NULL OR pr.deep_extraction = '{}'::jsonb "
            "OR jsonb_array_length(COALESCE(pr.deep_extraction->'feature_requests', '[]'::jsonb)) = 0)"
        )

    if search:
        conditions.append(f"pr.review_text ILIKE '%' || ${idx} || '%'")
        params.append(search)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    sort_map = {
        "imported_at": "pr.imported_at DESC NULLS LAST",
        "rating": "pr.rating DESC NULLS LAST",
        "pain_score": "pr.pain_score DESC NULLS LAST",
    }
    order = sort_map.get(sort_by, "pr.imported_at DESC NULLS LAST")

    params.extend([limit, offset])
    limit_idx = idx
    offset_idx = idx + 1

    rows = await pool.fetch(
        f"""
        SELECT pr.id, pr.asin, pr.rating, pr.root_cause, pr.pain_score,
               pr.severity, LEFT(pr.summary, 200) AS summary,
               pr.source_category, pr.enrichment_status, pr.deep_enrichment_status,
               pm.brand, pm.title, pr.imported_at
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        {where}
        ORDER BY {order}
        LIMIT ${limit_idx} OFFSET ${offset_idx}
        """,
        *params,
    )

    reviews = [
        {
            "id": str(r["id"]),
            "asin": r["asin"],
            "brand": r["brand"],
            "title": r["title"],
            "rating": _safe_float(r["rating"]),
            "root_cause": r["root_cause"],
            "pain_score": _safe_float(r["pain_score"]),
            "severity": r["severity"],
            "summary": r["summary"],
            "source_category": r["source_category"],
            "enrichment_status": r["enrichment_status"],
            "deep_enrichment_status": r["deep_enrichment_status"],
            "imported_at": str(r["imported_at"]) if r["imported_at"] else None,
        }
        for r in rows
    ]

    return {"reviews": reviews, "count": len(reviews)}


# ---------------------------------------------------------------------------
# GET /reviews/{review_id}
# ---------------------------------------------------------------------------


@router.get("/reviews/{review_id}")
async def get_review(review_id: str):
    try:
        rid = _uuid.UUID(review_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid review_id (must be UUID)")

    pool = _pool_or_503()
    row = await pool.fetchrow(
        """
        SELECT pr.*, pm.brand, pm.title AS product_title,
               pm.average_rating AS product_avg_rating,
               pm.rating_number AS product_total_ratings,
               pm.price AS product_price
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.id = $1
        """,
        rid,
    )

    if not row:
        raise HTTPException(status_code=404, detail="Review not found")

    return {
        "id": str(row["id"]),
        "asin": row["asin"],
        "source_category": row["source_category"],
        "rating": _safe_float(row["rating"]),
        "summary": row["summary"],
        "review_text": row["review_text"],
        "reviewer_id": row.get("reviewer_id"),
        "imported_at": str(row["imported_at"]) if row["imported_at"] else None,
        # First-pass enrichment
        "enrichment_status": row["enrichment_status"],
        "root_cause": row.get("root_cause"),
        "severity": row.get("severity"),
        "pain_score": _safe_float(row.get("pain_score")),
        "time_to_failure": row.get("time_to_failure"),
        "workaround_found": row.get("workaround_found"),
        "workaround_text": row.get("workaround_text"),
        "alternative_mentioned": row.get("alternative_mentioned"),
        "alternative_name": row.get("alternative_name"),
        "alternative_asin": row.get("alternative_asin"),
        # Deep enrichment (stored in deep_extraction JSONB column)
        "deep_enrichment_status": row.get("deep_enrichment_status"),
        "deep_enrichment": _safe_json(row.get("deep_extraction")),
        "deep_enriched_at": str(row["deep_enriched_at"]) if row.get("deep_enriched_at") else None,
        # Product metadata
        "brand": row["brand"],
        "product_title": row["product_title"],
        "product_avg_rating": _safe_float(row["product_avg_rating"]),
        "product_total_ratings": row["product_total_ratings"],
        "product_price": row["product_price"],
    }
