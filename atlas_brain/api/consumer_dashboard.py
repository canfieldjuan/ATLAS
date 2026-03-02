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

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..auth.dependencies import AuthUser, require_auth, require_plan
from ..config import settings
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


def _dist(counter: dict[str, int]) -> list[dict]:
    """Convert a {label: count} dict to sorted [{label, count}] list."""
    return sorted(
        [{"label": k, "count": v} for k, v in counter.items()],
        key=lambda x: x["count"], reverse=True,
    )


def _compute_brand_health(r) -> int | None:
    """Composite brand health score (0-100) from deep enrichment signals.

    Components (equal weight, 25 pts each):
      1. Repurchase rate: % of reviewers who would repurchase
      2. Retention rate: positive replacement / (positive + negative)
      3. Trajectory rate: positive trajectory / (positive + negative)
      4. Safety rate: inverted -- fewer safety flags = higher score

    Returns None if fewer than 5 deep-enriched reviews.
    """
    deep = r["deep_count"]
    if deep < 5:
        return None

    scores: list[float] = []

    # 1. Repurchase
    rp_total = r["repurchase_total"]
    if rp_total > 0:
        scores.append(r["repurchase_yes"] / rp_total)
    else:
        scores.append(0.5)  # neutral if no data

    # 2. Retention
    ret_pos = r["retention_pos"]
    ret_neg = r["retention_neg"]
    ret_total = ret_pos + ret_neg
    if ret_total > 0:
        scores.append(ret_pos / ret_total)
    else:
        scores.append(0.5)

    # 3. Trajectory
    traj_pos = r["trajectory_pos"]
    traj_neg = r["trajectory_neg"]
    traj_total = traj_pos + traj_neg
    if traj_total > 0:
        scores.append(traj_pos / traj_total)
    else:
        scores.append(0.5)

    # 4. Safety (inverted: 0 flags = 1.0, lots of flags relative to deep count = 0.0)
    safety = r["safety_count"]
    safety_rate = max(0.0, 1.0 - (safety / deep) * 10)  # 10% flagged = score 0
    scores.append(safety_rate)

    return round(sum(scores) / len(scores) * 100)


# ---------------------------------------------------------------------------
# Tenant scoping helper
# ---------------------------------------------------------------------------


def _tenant_cond(alias: str, param_idx: int) -> str:
    """Return a SQL condition that restricts to tracked ASINs for an account.

    When SaaS auth is disabled, returns 'TRUE' so queries are unscoped.
    """
    if not settings.saas_auth.enabled:
        return "TRUE"
    return f"{alias}.asin IN (SELECT asin FROM tracked_asins WHERE account_id = ${param_idx})"


def _tenant_params(user: AuthUser) -> list:
    """Return [account_id] when auth is enabled, else empty (no param needed)."""
    if not settings.saas_auth.enabled:
        return []
    return [_uuid.UUID(user.account_id)]


# ---------------------------------------------------------------------------
# ASIN management
# ---------------------------------------------------------------------------


class AddAsinRequest(BaseModel):
    asin: str
    label: str | None = None


@router.get("/asins")
async def list_tracked_asins(user: AuthUser = Depends(require_auth)):
    """List ASINs tracked by the current account."""
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)
    rows = await pool.fetch(
        """
        SELECT ta.asin, ta.label, ta.added_at,
               pm.title, pm.brand, pm.average_rating, pm.rating_number, pm.price
        FROM tracked_asins ta
        LEFT JOIN product_metadata pm ON pm.asin = ta.asin
        WHERE ta.account_id = $1
        ORDER BY ta.added_at DESC
        """,
        acct,
    )
    return {
        "asins": [
            {
                "asin": r["asin"],
                "label": r["label"],
                "added_at": r["added_at"].isoformat() if r["added_at"] else None,
                "title": r["title"],
                "brand": r["brand"],
                "average_rating": _safe_float(r["average_rating"]),
                "rating_number": r["rating_number"],
                "price": r["price"],
            }
            for r in rows
        ],
        "count": len(rows),
    }


@router.post("/asins")
async def add_tracked_asin(req: AddAsinRequest, user: AuthUser = Depends(require_auth)):
    """Add an ASIN to track. Enforces plan limit."""
    pool = _pool_or_503()

    acct = _uuid.UUID(user.account_id)

    # Check limit
    current_count = await pool.fetchval(
        "SELECT COUNT(*) FROM tracked_asins WHERE account_id = $1",
        acct,
    )
    asin_limit = await pool.fetchval(
        "SELECT asin_limit FROM saas_accounts WHERE id = $1",
        acct,
    )
    if current_count >= (asin_limit or 5):
        raise HTTPException(status_code=403, detail="ASIN limit reached. Upgrade your plan for more.")

    # Upsert
    await pool.execute(
        """
        INSERT INTO tracked_asins (account_id, asin, label)
        VALUES ($1, $2, $3)
        ON CONFLICT (account_id, asin) DO UPDATE SET label = EXCLUDED.label
        """,
        acct,
        req.asin.strip().upper(),
        req.label,
    )

    return {"status": "ok", "asin": req.asin.strip().upper()}


@router.delete("/asins/{asin}")
async def remove_tracked_asin(asin: str, user: AuthUser = Depends(require_auth)):
    """Remove a tracked ASIN."""
    pool = _pool_or_503()
    result = await pool.execute(
        "DELETE FROM tracked_asins WHERE account_id = $1 AND asin = $2",
        _uuid.UUID(user.account_id),
        asin.strip().upper(),
    )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="ASIN not tracked")
    return {"status": "ok"}


@router.get("/asins/search")
async def search_available_asins(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, le=50),
    user: AuthUser = Depends(require_auth),
):
    """Search available ASINs from product_metadata."""
    pool = _pool_or_503()
    rows = await pool.fetch(
        """
        SELECT asin, title, brand, average_rating, rating_number, price
        FROM product_metadata
        WHERE asin ILIKE '%' || $1 || '%'
           OR title ILIKE '%' || $1 || '%'
           OR brand ILIKE '%' || $1 || '%'
        ORDER BY rating_number DESC NULLS LAST
        LIMIT $2
        """,
        q,
        limit,
    )
    return {
        "results": [
            {
                "asin": r["asin"],
                "title": r["title"],
                "brand": r["brand"],
                "average_rating": _safe_float(r["average_rating"]),
                "rating_number": r["rating_number"],
                "price": r["price"],
            }
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# GET /pipeline
# ---------------------------------------------------------------------------


@router.get("/pipeline")
async def get_pipeline_status(
    source_category: Optional[str] = Query(None),
    user: AuthUser = Depends(require_auth),
):
    pool = _pool_or_503()

    conditions: list[str] = []
    params: list = []
    idx = 1

    # Tenant scoping
    t_cond = _tenant_cond("product_reviews", idx)
    if t_cond != "TRUE":
        conditions.append(t_cond)
        params.extend(_tenant_params(user))
        idx += 1

    if source_category:
        conditions.append(f"source_category = ${idx}")
        params.append(source_category)
        idx += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    enrichment_rows = await pool.fetch(
        f"""
        SELECT enrichment_status, COUNT(*) AS cnt
        FROM product_reviews
        {where}
        GROUP BY enrichment_status
        """,
        *params,
    )
    enrichment_counts = {r["enrichment_status"]: r["cnt"] for r in enrichment_rows}

    deep_rows = await pool.fetch(
        f"""
        SELECT deep_enrichment_status, COUNT(*) AS cnt
        FROM product_reviews
        {where}
        GROUP BY deep_enrichment_status
        """,
        *params,
    )
    deep_counts = {r["deep_enrichment_status"]: r["cnt"] for r in deep_rows}

    category_rows = await pool.fetch(
        f"""
        SELECT source_category, COUNT(*) AS cnt
        FROM product_reviews
        {where}
        GROUP BY source_category
        """,
        *params,
    )
    category_counts = {r["source_category"]: r["cnt"] for r in category_rows}

    totals = await pool.fetchrow(
        f"""
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched,
               COUNT(*) FILTER (WHERE deep_enrichment_status = 'enriched') AS deep_enriched,
               COUNT(*) FILTER (WHERE deep_enrichment_status IS NOT NULL AND deep_enrichment_status != 'not_applicable') AS targeted_for_deep,
               MAX(enriched_at) AS last_enrichment_at,
               MAX(deep_enriched_at) AS last_deep_enrichment_at
        FROM product_reviews
        {where}
        """,
        *params,
    )

    # Meta totals - scope by tracked ASINs
    meta_conditions: list[str] = []
    meta_params: list = []
    meta_idx = 1

    t_meta = _tenant_cond("pm", meta_idx)
    if t_meta != "TRUE":
        meta_conditions.append(t_meta)
        meta_params.extend(_tenant_params(user))
        meta_idx += 1

    if source_category:
        meta_conditions.append(f"pr.source_category = ${meta_idx}")
        meta_params.append(source_category)
        meta_idx += 1

    if meta_conditions:
        meta_where = "WHERE " + " AND ".join(meta_conditions)
        meta_totals = await pool.fetchrow(
            f"""
            SELECT COUNT(DISTINCT pm.brand) FILTER (WHERE pm.brand IS NOT NULL AND pm.brand != '') AS total_brands,
                   COUNT(DISTINCT pm.asin) AS total_asins
            FROM product_metadata pm
            JOIN product_reviews pr ON pr.asin = pm.asin
            {meta_where}
            """,
            *meta_params,
        )
    else:
        meta_totals = await pool.fetchrow(
            """
            SELECT COUNT(DISTINCT brand) FILTER (WHERE brand IS NOT NULL AND brand != '') AS total_brands,
                   COUNT(DISTINCT asin) AS total_asins
            FROM product_metadata
            """
        )

    return {
        "enrichment_counts": enrichment_counts,
        "deep_enrichment_counts": deep_counts,
        "category_counts": category_counts,
        "total_reviews": totals["total"] if totals else 0,
        "enriched": totals["enriched"] if totals else 0,
        "deep_enriched": totals["deep_enriched"] if totals else 0,
        "targeted_for_deep": totals["targeted_for_deep"] if totals else 0,
        "total_brands": meta_totals["total_brands"] if meta_totals else 0,
        "total_asins": meta_totals["total_asins"] if meta_totals else 0,
        "last_enrichment_at": str(totals["last_enrichment_at"]) if totals and totals["last_enrichment_at"] else None,
        "last_deep_enrichment_at": str(totals["last_deep_enrichment_at"]) if totals and totals["last_deep_enrichment_at"] else None,
    }


# ---------------------------------------------------------------------------
# GET /categories
# ---------------------------------------------------------------------------


@router.get("/categories")
async def list_categories(user: AuthUser = Depends(require_auth)):
    pool = _pool_or_503()
    t_cond = _tenant_cond("product_reviews", 1)
    t_params = _tenant_params(user)
    where = f"WHERE source_category IS NOT NULL AND source_category != '' AND {t_cond}" if t_cond != "TRUE" else "WHERE source_category IS NOT NULL AND source_category != ''"
    rows = await pool.fetch(
        f"SELECT DISTINCT source_category FROM product_reviews {where} ORDER BY source_category",
        *t_params,
    )
    return {"categories": [r["source_category"] for r in rows]}


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
    user: AuthUser = Depends(require_auth),
):
    pool = _pool_or_503()
    conditions: list[str] = ["pm.brand IS NOT NULL", "pm.brand != ''"]
    params: list = []
    idx = 1

    # Tenant scoping
    t_cond = _tenant_cond("pr", idx)
    if t_cond != "TRUE":
        conditions.append(t_cond)
        params.extend(_tenant_params(user))
        idx += 1

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
    health_sort = sort_by == "brand_health"
    order = sort_map.get(sort_by, "review_count DESC")

    if min_reviews > 0:
        having = f"HAVING COUNT(pr.id) >= ${idx}"
        params.append(min_reviews)
        idx += 1
    else:
        having = ""

    # For brand_health sort, fetch all rows and paginate in Python
    # (health is computed after SQL, can't ORDER BY it in SQL)
    if health_sort:
        sql_limit_clause = "LIMIT 1000"
    else:
        params.extend([limit, offset])
        limit_idx = idx
        offset_idx = idx + 1
        sql_limit_clause = f"LIMIT ${limit_idx} OFFSET ${offset_idx}"

    rows = await pool.fetch(
        f"""
        WITH brand_products AS (
            SELECT pm.brand, AVG(pm.average_rating) AS pm_avg_rating,
                   SUM(pm.rating_number) AS total_ratings
            FROM product_metadata pm
            WHERE pm.brand IS NOT NULL AND pm.brand != ''
            GROUP BY pm.brand
        )
        SELECT pm.brand,
               COUNT(DISTINCT pm.asin) AS product_count,
               COUNT(pr.id) AS review_count,
               bp.pm_avg_rating,
               bp.total_ratings,
               AVG(pr.pain_score) FILTER (WHERE pr.rating <= 3) AS avg_complaint_score,
               AVG(pr.pain_score) FILTER (WHERE pr.rating > 3)  AS avg_praise_score,
               COUNT(*) FILTER (WHERE pr.rating <= 3) AS complaint_count,
               COUNT(*) FILTER (WHERE pr.rating > 3)  AS praise_count,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction IS NOT NULL
                     AND pr.deep_extraction != '{{}}'::jsonb
                     AND pr.deep_extraction->'safety_flag'->>'flagged' = 'true'
               ) AS safety_count,
               -- Brand health score components
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction IS NOT NULL
                     AND pr.deep_extraction != '{{}}'::jsonb
               ) AS deep_count,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction->>'would_repurchase' = 'true'
               ) AS repurchase_yes,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction->>'would_repurchase' IN ('true','false')
               ) AS repurchase_total,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction->>'replacement_behavior'
                         IN ('kept_using','repurchased','replaced_same')
               ) AS retention_pos,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction->>'replacement_behavior'
                         IN ('switched_to','switched_brand','returned','avoided')
               ) AS retention_neg,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction->>'sentiment_trajectory'
                         IN ('always_positive','improved','mixed_then_positive')
               ) AS trajectory_pos,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction->>'sentiment_trajectory'
                         IN ('always_negative','degraded','mixed_then_negative','mixed_then_bad','always_bad')
               ) AS trajectory_neg
        FROM product_metadata pm
        JOIN product_reviews pr ON pr.asin = pm.asin
        JOIN brand_products bp ON bp.brand = pm.brand
        WHERE {where}
        GROUP BY pm.brand, bp.pm_avg_rating, bp.total_ratings
        {having}
        ORDER BY {order}
        {sql_limit_clause}
        """,
        *params,
    )

    brands = []
    for r in rows:
        health = _compute_brand_health(r)
        brands.append({
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
            "brand_health": health,
        })

    if health_sort:
        brands.sort(key=lambda b: (b["brand_health"] is not None, b["brand_health"] or 0), reverse=True)
        total_count = len(brands)
        brands = brands[offset : offset + limit]
    else:
        # Total count (without LIMIT/OFFSET) for pagination
        count_params = params[:-2]  # strip limit & offset
        total_count = await pool.fetchval(
            f"""
            SELECT COUNT(*) FROM (
                SELECT pm.brand
                FROM product_metadata pm
                JOIN product_reviews pr ON pr.asin = pm.asin
                WHERE {where}
                GROUP BY pm.brand
                {having}
            ) sub
            """,
            *count_params,
        )
        total_count = total_count or 0

    return {"brands": brands, "count": len(brands), "total_count": total_count}


# ---------------------------------------------------------------------------
# GET /brands/compare
# ---------------------------------------------------------------------------


@router.get("/brands/compare")
async def compare_brands(
    brands: str = Query(..., description="Comma-separated brand names (2-4)"),
    user: AuthUser = require_plan("growth"),
):
    """Side-by-side comparison of 2-4 brands on core signals + cross-brand intelligence."""
    pool = _pool_or_503()

    brand_list = [b.strip() for b in brands.split(",") if b.strip()]
    if len(brand_list) < 2 or len(brand_list) > 4:
        raise HTTPException(status_code=400, detail="Provide 2-4 comma-separated brand names")

    # Build parameterized brand filter: (pm.brand ILIKE $1 OR pm.brand ILIKE $2 ...)
    brand_clauses = " OR ".join(f"pm.brand ILIKE ${i+1}" for i in range(len(brand_list)))
    brand_params = list(brand_list)

    # Tenant scoping
    t_idx = len(brand_list) + 1
    t_cond = _tenant_cond("pr", t_idx)
    t_extra = _tenant_params(user)
    t_sql = f"AND {t_cond}" if t_cond != "TRUE" else ""

    # -- Query 1: Summary stats per brand --
    summary_rows = await pool.fetch(
        f"""
        WITH brand_products AS (
            SELECT pm.brand, AVG(pm.average_rating) AS pm_avg_rating
            FROM product_metadata pm
            WHERE pm.brand IS NOT NULL AND pm.brand != ''
              AND ({brand_clauses})
            GROUP BY pm.brand
        )
        SELECT pm.brand,
               COUNT(DISTINCT pm.asin) AS product_count,
               COUNT(pr.id) AS review_count,
               bp.pm_avg_rating,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction IS NOT NULL
                     AND pr.deep_extraction != '{{}}'::jsonb
               ) AS deep_count,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction->>'would_repurchase' = 'true'
               ) AS repurchase_yes,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction->>'would_repurchase' IN ('true','false')
               ) AS repurchase_total,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction IS NOT NULL
                     AND pr.deep_extraction != '{{}}'::jsonb
                     AND pr.deep_extraction->'safety_flag'->>'flagged' = 'true'
               ) AS safety_count,
               -- Health components
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction->>'replacement_behavior'
                         IN ('kept_using','repurchased','replaced_same')
               ) AS retention_pos,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction->>'replacement_behavior'
                         IN ('switched_to','switched_brand','returned','avoided')
               ) AS retention_neg,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction->>'sentiment_trajectory'
                         IN ('always_positive','improved','mixed_then_positive')
               ) AS trajectory_pos,
               COUNT(*) FILTER (
                   WHERE pr.deep_extraction->>'sentiment_trajectory'
                         IN ('always_negative','degraded','mixed_then_negative','mixed_then_bad','always_bad')
               ) AS trajectory_neg
        FROM product_metadata pm
        JOIN product_reviews pr ON pr.asin = pm.asin
        JOIN brand_products bp ON bp.brand = pm.brand
        WHERE pm.brand IS NOT NULL AND pm.brand != ''
          AND ({brand_clauses})
          {t_sql}
        GROUP BY pm.brand, bp.pm_avg_rating
        """,
        *brand_params, *t_extra,
    )

    # Map brand name (lowered) -> summary row
    summary_map: dict[str, dict] = {}
    for r in summary_rows:
        bkey = r["brand"]
        deep = r["deep_count"] or 0
        # Compute brand health
        health: int | None = None
        if deep >= 5:
            scores: list[float] = []
            rp_total = r["repurchase_total"] or 0
            scores.append((r["repurchase_yes"] or 0) / rp_total if rp_total > 0 else 0.5)
            ret_total = (r["retention_pos"] or 0) + (r["retention_neg"] or 0)
            scores.append((r["retention_pos"] or 0) / ret_total if ret_total > 0 else 0.5)
            traj_total = (r["trajectory_pos"] or 0) + (r["trajectory_neg"] or 0)
            scores.append((r["trajectory_pos"] or 0) / traj_total if traj_total > 0 else 0.5)
            safety_cnt = r["safety_count"] or 0
            scores.append(max(0.0, 1.0 - (safety_cnt / deep) * 10))
            health = round(sum(scores) / len(scores) * 100)

        summary_map[bkey] = {
            "product_count": r["product_count"],
            "total_reviews": r["review_count"],
            "deep_review_count": deep,
            "avg_rating": _safe_float(r["pm_avg_rating"]),
            "brand_health": health,
            "safety_flagged_count": r["safety_count"] or 0,
        }

    # -- Query 2: Deep enum fields (single scan for all brands) --
    enum_rows = await pool.fetch(
        f"""
        SELECT pm.brand,
               deep_extraction->>'would_repurchase'      AS repurchase,
               deep_extraction->>'replacement_behavior'   AS replacement,
               deep_extraction->>'sentiment_trajectory'   AS trajectory,
               deep_extraction->>'consequence_severity'   AS consequence,
               deep_extraction->'switching_barrier'       AS barrier,
               deep_extraction->'failure_details'         AS failure,
               deep_extraction->'safety_flag'             AS safety,
               deep_extraction->'feature_requests'        AS feature_requests,
               deep_extraction->'product_comparisons'     AS comparisons,
               deep_extraction->'consideration_set'       AS consideration,
               pr.rating
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand IS NOT NULL AND pm.brand != ''
          AND ({brand_clauses})
          {t_sql}
          AND pr.deep_extraction IS NOT NULL
          AND pr.deep_extraction != '{{}}'::jsonb
        LIMIT 5000
        """,
        *brand_params, *t_extra,
    )

    # Per-brand counters
    brand_counters: dict[str, dict] = {}
    # Cross-brand data accumulators
    all_feature_requests: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # request -> {brand: count}
    all_comparisons: list[dict] = []
    all_considerations: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # product -> {brand: count}

    for row in enum_rows:
        brand = row["brand"]
        if brand not in brand_counters:
            brand_counters[brand] = {
                "repurchase": defaultdict(int),
                "replacement": defaultdict(int),
                "trajectory": defaultdict(int),
                "consequence": defaultdict(int),
                "barrier": defaultdict(int),
                "failure_count": 0,
                "dollar_lost_total": 0.0,
                "dollar_lost_n": 0,
                "safety_flagged": 0,
            }
        bc = brand_counters[brand]

        # Enums
        for field in ("repurchase", "replacement", "trajectory", "consequence"):
            val = row[field]
            if val and val not in ("none", "unknown", "not_mentioned"):
                bc[field][val] += 1

        # Switching barrier
        barrier = _safe_json(row["barrier"])
        if isinstance(barrier, dict) and barrier.get("level"):
            bc["barrier"][barrier["level"]] += 1

        # Failure details
        fail = _safe_json(row["failure"])
        if isinstance(fail, dict) and fail.get("failure_mode"):
            bc["failure_count"] += 1
            dl = fail.get("dollar_amount_lost")
            if dl is not None:
                try:
                    bc["dollar_lost_total"] += float(dl)
                    bc["dollar_lost_n"] += 1
                except (ValueError, TypeError):
                    pass

        # Safety flag
        sf = _safe_json(row["safety"])
        if isinstance(sf, dict) and sf.get("flagged"):
            bc["safety_flagged"] += 1

        # Feature requests (for cross-brand)
        reqs = _safe_json(row["feature_requests"])
        if isinstance(reqs, list):
            for req in reqs:
                text = req if isinstance(req, str) else (req.get("request", str(req)) if isinstance(req, dict) else str(req))
                all_feature_requests[text.strip().lower()][brand] += 1

        # Product comparisons (for cross-brand flows)
        comps = _safe_json(row["comparisons"])
        if isinstance(comps, list):
            for comp in comps:
                if isinstance(comp, dict):
                    other = comp.get("product_name") or comp.get("product", "")
                    direction = comp.get("direction", "compared")
                    all_comparisons.append({
                        "from_brand": brand,
                        "to_brand": other,
                        "direction": direction,
                        "rating": float(row["rating"]) if row["rating"] is not None else None,
                    })

        # Consideration set (for cross-brand overlap)
        cset = _safe_json(row["consideration"])
        if isinstance(cset, list):
            for item in cset:
                if isinstance(item, dict):
                    prod = item.get("product", "Unknown")
                    all_considerations[prod.strip().lower()][brand] += 1

    # -- Query 3: First-pass enrichment (severity, workaround) per brand --
    fp_rows = await pool.fetch(
        f"""
        SELECT pm.brand,
               severity,
               workaround_found
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand IS NOT NULL AND pm.brand != ''
          AND ({brand_clauses})
          {t_sql}
          AND pr.enrichment_status = 'enriched'
        LIMIT 5000
        """,
        *brand_params, *t_extra,
    )

    fp_counters: dict[str, dict] = {}
    for row in fp_rows:
        brand = row["brand"]
        if brand not in fp_counters:
            fp_counters[brand] = {"severity": defaultdict(int), "workaround_count": 0, "total": 0}
        fpc = fp_counters[brand]
        fpc["total"] += 1
        sev = row["severity"]
        if sev and sev not in ("", "none"):
            fpc["severity"][sev] += 1
        if row["workaround_found"]:
            fpc["workaround_count"] += 1

    # -- Assemble per-brand metrics --
    per_brand: dict[str, dict] = {}
    for bname in brand_list:
        # Find the actual brand key (case-insensitive match)
        actual_key = None
        for k in summary_map:
            if k.lower() == bname.lower():
                actual_key = k
                break
        if not actual_key:
            continue

        sm = summary_map.get(actual_key, {})
        bc = brand_counters.get(actual_key, {})
        fpc = fp_counters.get(actual_key, {})

        # Repurchase pct
        rp_yes = bc.get("repurchase", {}).get("true", 0) if bc else 0
        rp_total = rp_yes + (bc.get("repurchase", {}).get("false", 0) if bc else 0)
        repurchase_pct = round(rp_yes / rp_total * 100) if rp_total > 0 else None

        fp_total = fpc.get("total", 0) if fpc else 0

        per_brand[actual_key] = {
            "product_count": sm.get("product_count", 0),
            "total_reviews": sm.get("total_reviews", 0),
            "deep_review_count": sm.get("deep_review_count", 0),
            "avg_rating": sm.get("avg_rating"),
            "brand_health": sm.get("brand_health"),
            "repurchase_pct": repurchase_pct,
            "safety_flagged_count": sm.get("safety_flagged_count", 0),
            "failure_count": bc.get("failure_count", 0) if bc else 0,
            "avg_dollar_lost": round(bc["dollar_lost_total"] / bc["dollar_lost_n"], 2) if bc and bc.get("dollar_lost_n") else None,
            "replacement_breakdown": _dist(bc.get("replacement", {})) if bc else [],
            "trajectory_breakdown": _dist(bc.get("trajectory", {})) if bc else [],
            "switching_barrier": _dist(bc.get("barrier", {})) if bc else [],
            "consequence_breakdown": _dist(bc.get("consequence", {})) if bc else [],
            "severity_breakdown": _dist(fpc.get("severity", {})) if fpc else [],
            "workaround_rate": round(fpc["workaround_count"] / fp_total * 100) if fp_total > 0 else None,
        }

    # -- Cross-brand computations --

    # Competitive flows: only between compared brands
    flow_agg: dict[str, dict] = {}
    for c in all_comparisons:
        to_brand_lower = c["to_brand"].lower()
        from_brand_lower = c["from_brand"].lower()
        # Check if to_brand matches any of our compared brands (and not same brand)
        matched_to = None
        for bname in per_brand:
            if bname.lower() == to_brand_lower and bname.lower() != from_brand_lower:
                matched_to = bname
                break
        if not matched_to:
            continue
        key = f"{c['from_brand']}|{matched_to}|{c['direction']}"
        if key not in flow_agg:
            flow_agg[key] = {"from_brand": c["from_brand"], "to_brand": matched_to, "direction": c["direction"], "count": 0, "ratings": []}
        flow_agg[key]["count"] += 1
        if c["rating"] is not None:
            flow_agg[key]["ratings"].append(c["rating"])

    competitive_flows = sorted(
        [
            {
                "from_brand": v["from_brand"],
                "to_brand": v["to_brand"],
                "direction": v["direction"],
                "count": v["count"],
                "avg_rating": round(sum(v["ratings"]) / len(v["ratings"]), 2) if v["ratings"] else None,
            }
            for v in flow_agg.values()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )

    # Shared feature requests: appearing in 2+ brands
    shared_features = []
    for req, brand_counts in all_feature_requests.items():
        if len(brand_counts) >= 2:
            shared_features.append({
                "request": req,
                "brands": list(brand_counts.keys()),
                "total_count": sum(brand_counts.values()),
            })
    shared_features.sort(key=lambda x: x["total_count"], reverse=True)
    shared_features = shared_features[:20]

    # Consideration overlap: products mentioned by 2+ brands' reviewers
    consideration_overlap = []
    for prod, brand_counts in all_considerations.items():
        if len(brand_counts) >= 2:
            consideration_overlap.append({
                "product": prod,
                "mentioned_by_brands": list(brand_counts.keys()),
                "total_count": sum(brand_counts.values()),
            })
    consideration_overlap.sort(key=lambda x: x["total_count"], reverse=True)
    consideration_overlap = consideration_overlap[:20]

    return {
        "brands": list(per_brand.keys()),
        "per_brand": per_brand,
        "cross_brand": {
            "competitive_flows": competitive_flows,
            "shared_feature_requests": shared_features,
            "consideration_overlap": consideration_overlap,
        },
    }


# ---------------------------------------------------------------------------
# GET /brands/{brand_name}
# ---------------------------------------------------------------------------


@router.get("/brands/{brand_name}")
async def get_brand_detail(brand_name: str, user: AuthUser = Depends(require_auth)):
    pool = _pool_or_503()
    bname = brand_name.strip()

    # Tenant scoping for brand detail queries
    t_cond2 = _tenant_cond("pr", 2)
    t_extra = _tenant_params(user)
    t_and = f"AND {t_cond2}" if t_cond2 != "TRUE" else ""

    # Products for this brand
    # Move tenant condition into JOIN ON clause to preserve LEFT JOIN semantics
    # (products with zero reviews should still appear)
    t_join_on = f"AND {t_cond2}" if t_cond2 != "TRUE" else ""
    products = await pool.fetch(
        f"""
        SELECT pm.asin, pm.title, pm.average_rating, pm.rating_number, pm.price,
               COUNT(pr.id) AS review_count,
               AVG(pr.pain_score) FILTER (WHERE pr.rating <= 3) AS avg_complaint_score,
               AVG(pr.pain_score) FILTER (WHERE pr.rating > 3)  AS avg_praise_score,
               COUNT(*) FILTER (WHERE pr.rating <= 3) AS complaint_count,
               COUNT(*) FILTER (WHERE pr.rating > 3)  AS praise_count
        FROM product_metadata pm
        LEFT JOIN product_reviews pr ON pr.asin = pm.asin {t_join_on}
        WHERE pm.brand ILIKE $1
        GROUP BY pm.asin, pm.title, pm.average_rating, pm.rating_number, pm.price
        ORDER BY review_count DESC
        """,
        bname, *t_extra,
    )

    if not products:
        raise HTTPException(status_code=404, detail="Brand not found")

    # Aggregate sentiment aspects
    aspect_rows = await pool.fetch(
        f"""
        SELECT deep_extraction->'sentiment_aspects' AS aspects
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand ILIKE $1
          {t_and}
          AND pr.deep_extraction IS NOT NULL
          AND pr.deep_extraction != '{{}}'::jsonb
          AND pr.deep_extraction->'sentiment_aspects' IS NOT NULL
        LIMIT 5000
        """,
        bname, *t_extra,
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
        f"""
        SELECT deep_extraction->'feature_requests' AS requests
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand ILIKE $1
          {t_and}
          AND pr.deep_extraction IS NOT NULL
          AND pr.deep_extraction != '{{}}'::jsonb
          AND jsonb_array_length(COALESCE(pr.deep_extraction->'feature_requests', '[]'::jsonb)) > 0
        LIMIT 5000
        """,
        bname, *t_extra,
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
        f"""
        SELECT deep_extraction->'product_comparisons' AS comparisons, pr.rating
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand ILIKE $1
          {t_and}
          AND pr.deep_extraction IS NOT NULL
          AND pr.deep_extraction != '{{}}'::jsonb
          AND jsonb_array_length(COALESCE(pr.deep_extraction->'product_comparisons', '[]'::jsonb)) > 0
        LIMIT 5000
        """,
        bname, *t_extra,
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

    # ------------------------------------------------------------------
    # All enum/scalar deep fields in a single pass
    # ------------------------------------------------------------------
    enum_rows = await pool.fetch(
        f"""
        SELECT deep_extraction->>'brand_loyalty_depth'   AS loyalty,
               deep_extraction->>'expertise_level'       AS expertise,
               deep_extraction->>'budget_type'           AS budget,
               deep_extraction->>'discovery_channel'     AS channel,
               deep_extraction->>'would_repurchase'      AS repurchase,
               deep_extraction->>'replacement_behavior'  AS replacement,
               deep_extraction->>'sentiment_trajectory'  AS trajectory,
               deep_extraction->>'consequence_severity'  AS consequence,
               deep_extraction->>'frustration_threshold' AS frustration,
               deep_extraction->>'use_intensity'         AS intensity,
               deep_extraction->>'research_depth'        AS research,
               deep_extraction->>'occasion_context'      AS occasion,
               deep_extraction->>'review_delay_signal'   AS delay,
               deep_extraction->>'buyer_household'       AS household,
               deep_extraction->>'profession_hint'       AS profession,
               deep_extraction->'switching_barrier'      AS barrier,
               deep_extraction->'ecosystem_lock_in'      AS ecosystem,
               deep_extraction->'amplification_intent'   AS amplification,
               deep_extraction->'review_sentiment_openness' AS openness,
               deep_extraction->'failure_details'        AS failure,
               deep_extraction->'safety_flag'            AS safety,
               deep_extraction->'bulk_purchase_signal'   AS bulk,
               deep_extraction->'buyer_context'          AS buyer_ctx
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand ILIKE $1
          {t_and}
          AND pr.deep_extraction IS NOT NULL
          AND pr.deep_extraction != '{{}}'::jsonb
        LIMIT 5000
        """,
        bname, *t_extra,
    )

    # Enum distribution counters
    _enum_fields = [
        "loyalty", "expertise", "budget", "channel", "repurchase",
        "replacement", "trajectory", "consequence", "frustration",
        "intensity", "research", "occasion", "delay", "household",
    ]
    counters: dict[str, dict[str, int]] = {f: defaultdict(int) for f in _enum_fields}

    # Structured object counters
    barrier_counter: dict[str, int] = defaultdict(int)
    barrier_reasons: dict[str, list[str]] = defaultdict(list)
    ecosystem_counter: dict[str, int] = defaultdict(int)
    amplification_counter: dict[str, int] = defaultdict(int)
    openness_counter: dict[str, int] = defaultdict(int)
    buyer_type_counter: dict[str, int] = defaultdict(int)
    price_sentiment_counter: dict[str, int] = defaultdict(int)

    # Failure aggregation
    failure_modes: dict[str, int] = defaultdict(int)
    failed_components: dict[str, int] = defaultdict(int)
    total_dollar_lost = 0.0
    dollar_lost_count = 0
    failure_count = 0

    # Safety aggregation
    safety_flagged = 0

    # Profession collection
    professions: dict[str, int] = defaultdict(int)

    for row in enum_rows:
        # Enum fields
        for f in _enum_fields:
            val = row[f]
            if val and val != "none" and val != "unknown" and val != "not_mentioned":
                counters[f][val] += 1

        # Switching barrier
        barrier = _safe_json(row["barrier"])
        if isinstance(barrier, dict) and barrier.get("level"):
            lvl = barrier["level"]
            barrier_counter[lvl] += 1
            reason = barrier.get("reason")
            if reason and lvl not in ("none",):
                barrier_reasons[lvl].append(reason)

        # Ecosystem lock-in
        eco = _safe_json(row["ecosystem"])
        if isinstance(eco, dict) and eco.get("level"):
            ecosystem_counter[eco["level"]] += 1

        # Amplification intent
        amp = _safe_json(row["amplification"])
        if isinstance(amp, dict) and amp.get("intent"):
            amplification_counter[amp["intent"]] += 1

        # Sentiment openness
        opn = _safe_json(row["openness"])
        if isinstance(opn, dict) and opn.get("open") is not None:
            openness_counter["open" if opn["open"] else "closed"] += 1

        # Buyer context
        ctx = _safe_json(row["buyer_ctx"])
        if isinstance(ctx, dict):
            bt = ctx.get("buyer_type")
            if bt:
                buyer_type_counter[bt] += 1
            ps = ctx.get("price_sentiment")
            if ps and ps != "not_mentioned":
                price_sentiment_counter[ps] += 1

        # Failure details
        fail = _safe_json(row["failure"])
        if isinstance(fail, dict) and fail.get("failure_mode"):
            failure_count += 1
            fm = fail["failure_mode"].strip().lower()
            failure_modes[fm] += 1
            fc = fail.get("failed_component")
            if fc:
                failed_components[fc.strip().lower()] += 1
            dl = fail.get("dollar_amount_lost")
            if dl is not None:
                try:
                    total_dollar_lost += float(dl)
                    dollar_lost_count += 1
                except (ValueError, TypeError):
                    pass

        # Safety flag
        sf = _safe_json(row["safety"])
        if isinstance(sf, dict) and sf.get("flagged"):
            safety_flagged += 1

        # Profession hints
        prof = row["profession"]
        if prof:
            professions[prof.strip().lower()] += 1

    # ------------------------------------------------------------------
    # Positive aspects (top terms across reviews)
    # ------------------------------------------------------------------
    pos_rows = await pool.fetch(
        f"""
        SELECT deep_extraction->'positive_aspects' AS aspects
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand ILIKE $1
          {t_and}
          AND pr.deep_extraction IS NOT NULL
          AND pr.deep_extraction != '{{}}'::jsonb
          AND jsonb_array_length(COALESCE(pr.deep_extraction->'positive_aspects', '[]'::jsonb)) > 0
        """,
        bname, *t_extra,
    )
    positive_counter: dict[str, int] = defaultdict(int)
    for row in pos_rows:
        items = _safe_json(row["aspects"])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str) and item.strip():
                    positive_counter[item.strip().lower()] += 1

    top_positives = [
        {"aspect": k, "count": v}
        for k, v in sorted(positive_counter.items(), key=lambda x: x[1], reverse=True)[:20]
    ]

    # ------------------------------------------------------------------
    # Consideration set (what buyers considered and rejected)
    # ------------------------------------------------------------------
    cons_rows = await pool.fetch(
        f"""
        SELECT deep_extraction->'consideration_set' AS cset
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand ILIKE $1
          {t_and}
          AND pr.deep_extraction IS NOT NULL
          AND pr.deep_extraction != '{{}}'::jsonb
          AND jsonb_array_length(COALESCE(pr.deep_extraction->'consideration_set', '[]'::jsonb)) > 0
        """,
        bname, *t_extra,
    )
    consideration_counter: dict[str, dict] = {}
    for row in cons_rows:
        items = _safe_json(row["cset"])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    prod = item.get("product", "Unknown")
                    key = prod.strip().lower()
                    if key not in consideration_counter:
                        consideration_counter[key] = {"product": prod, "count": 0, "reasons": []}
                    consideration_counter[key]["count"] += 1
                    why = item.get("why_not")
                    if why:
                        consideration_counter[key]["reasons"].append(why)

    top_considerations = sorted(
        [
            {"product": v["product"], "count": v["count"],
             "top_reason": max(set(v["reasons"]), key=v["reasons"].count) if v["reasons"] else None}
            for v in consideration_counter.values()
        ],
        key=lambda x: x["count"], reverse=True,
    )[:15]

    # ------------------------------------------------------------------
    # Totals & brand health score
    # ------------------------------------------------------------------
    total_reviews = sum(r["review_count"] for r in products)
    deep_review_count = len(enum_rows)
    if t_cond2 != "TRUE":
        # Query product_metadata directly with subquery filter — avoids JOIN fanout
        avg_rating_all = await pool.fetchval(
            """
            SELECT AVG(average_rating) FROM product_metadata
            WHERE brand ILIKE $1
              AND asin IN (SELECT asin FROM tracked_asins WHERE account_id = $2)
            """,
            bname, *t_extra,
        )
    else:
        avg_rating_all = await pool.fetchval(
            "SELECT AVG(average_rating) FROM product_metadata WHERE brand ILIKE $1", bname
        )

    # Compute brand health from already-aggregated counters
    detail_health: int | None = None
    if deep_review_count >= 5:
        _scores: list[float] = []
        # Repurchase
        rp_yes = counters["repurchase"].get("true", 0)
        rp_total = rp_yes + counters["repurchase"].get("false", 0)
        _scores.append(rp_yes / rp_total if rp_total > 0 else 0.5)
        # Retention
        ret_pos = sum(counters["replacement"].get(k, 0) for k in ("kept_using", "repurchased", "replaced_same"))
        ret_neg = sum(counters["replacement"].get(k, 0) for k in ("switched_to", "switched_brand", "returned", "avoided"))
        ret_total = ret_pos + ret_neg
        _scores.append(ret_pos / ret_total if ret_total > 0 else 0.5)
        # Trajectory
        traj_pos = sum(counters["trajectory"].get(k, 0) for k in ("always_positive", "improved", "mixed_then_positive"))
        traj_neg = sum(counters["trajectory"].get(k, 0) for k in ("always_negative", "degraded", "mixed_then_negative", "mixed_then_bad", "always_bad"))
        traj_total = traj_pos + traj_neg
        _scores.append(traj_pos / traj_total if traj_total > 0 else 0.5)
        # Safety (inverted)
        _scores.append(max(0.0, 1.0 - (safety_flagged / deep_review_count) * 10))
        detail_health = round(sum(_scores) / len(_scores) * 100)

    # ------------------------------------------------------------------
    # First-pass enrichment fields (severity, time_to_failure, etc.)
    # ------------------------------------------------------------------
    fp_rows = await pool.fetch(
        f"""
        SELECT severity,
               time_to_failure,
               root_cause,
               workaround_found,
               manufacturing_suggestion,
               alternative_name
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.brand ILIKE $1
          {t_and}
          AND pr.enrichment_status = 'enriched'
        LIMIT 5000
        """,
        bname, *t_extra,
    )

    severity_counter: dict[str, int] = defaultdict(int)
    ttf_counter: dict[str, int] = defaultdict(int)
    root_cause_counter: dict[str, int] = defaultdict(int)
    mfg_counter: dict[str, int] = defaultdict(int)
    alt_counter: dict[str, int] = defaultdict(int)
    workaround_count = 0
    fp_total = len(fp_rows)

    for row in fp_rows:
        sev = row["severity"]
        if sev and sev not in ("", "none"):
            severity_counter[sev] += 1
        ttf = row["time_to_failure"]
        if ttf and ttf not in ("", "not_mentioned"):
            ttf_counter[ttf] += 1
        rc = row["root_cause"]
        if rc and rc.strip():
            root_cause_counter[rc.strip().lower()] += 1
        if row["workaround_found"]:
            workaround_count += 1
        mfg = row["manufacturing_suggestion"]
        if mfg and mfg.strip():
            mfg_counter[mfg.strip().lower()] += 1
        alt = row["alternative_name"]
        if alt and alt.strip():
            alt_counter[alt.strip()] += 1

    return {
        "brand": bname,
        "product_count": len(products),
        "total_reviews": total_reviews,
        "deep_review_count": deep_review_count,
        "avg_rating": _safe_float(avg_rating_all),
        "brand_health": detail_health,
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
        "top_positives": top_positives,
        "consideration_set": top_considerations,
        # Churn signals
        "loyalty_breakdown": _dist(counters["loyalty"]),
        "repurchase_breakdown": _dist(counters["repurchase"]),
        "replacement_breakdown": _dist(counters["replacement"]),
        "trajectory_breakdown": _dist(counters["trajectory"]),
        "switching_barrier": _dist(barrier_counter),
        # Failure analysis
        "failure_analysis": {
            "failure_count": failure_count,
            "top_failure_modes": [
                {"mode": k, "count": v}
                for k, v in sorted(failure_modes.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
            "top_failed_components": [
                {"component": k, "count": v}
                for k, v in sorted(failed_components.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
            "avg_dollar_lost": round(total_dollar_lost / dollar_lost_count, 2) if dollar_lost_count else None,
            "total_dollar_lost": round(total_dollar_lost, 2) if dollar_lost_count else None,
        },
        # Buyer psychology
        "buyer_profile": {
            "expertise": _dist(counters["expertise"]),
            "budget": _dist(counters["budget"]),
            "discovery_channel": _dist(counters["channel"]),
            "frustration": _dist(counters["frustration"]),
            "intensity": _dist(counters["intensity"]),
            "research_depth": _dist(counters["research"]),
            "occasion": _dist(counters["occasion"]),
            "household": _dist(counters["household"]),
            "buyer_type": _dist(buyer_type_counter),
            "price_sentiment": _dist(price_sentiment_counter),
            "professions": [
                {"profession": k, "count": v}
                for k, v in sorted(professions.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
        },
        # Engagement signals
        "consequence_breakdown": _dist(counters["consequence"]),
        "delay_breakdown": _dist(counters["delay"]),
        "ecosystem_lock_in": _dist(ecosystem_counter),
        "amplification_intent": _dist(amplification_counter),
        "openness_breakdown": _dist(openness_counter),
        "safety_flagged_count": safety_flagged,
        # First-pass enrichment
        "first_pass": {
            "enriched_count": fp_total,
            "severity_breakdown": _dist(severity_counter),
            "time_to_failure": _dist(ttf_counter),
            "workaround_rate": round(workaround_count / fp_total * 100) if fp_total else None,
            "workaround_count": workaround_count,
            "top_root_causes": [
                {"cause": k, "count": v}
                for k, v in sorted(root_cause_counter.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
            "top_manufacturing_suggestions": [
                {"suggestion": k, "count": v}
                for k, v in sorted(mfg_counter.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
            "top_alternatives_mentioned": [
                {"product": k, "count": v}
                for k, v in sorted(alt_counter.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
        },
    }


# ---------------------------------------------------------------------------
# GET /flows
# ---------------------------------------------------------------------------


@router.get("/flows")
async def get_competitive_flows(
    source_category: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    direction: Optional[str] = Query(None),
    min_count: int = Query(2),
    limit: int = Query(100, le=500),
    user: AuthUser = Depends(require_auth),
):
    pool = _pool_or_503()
    base_conditions = [
        "pr.deep_extraction IS NOT NULL",
        "pr.deep_extraction != '{}'::jsonb",
    ]
    params: list = []
    idx = 1

    # Tenant scoping
    t_cond = _tenant_cond("pr", idx)
    if t_cond != "TRUE":
        base_conditions.append(t_cond)
        params.extend(_tenant_params(user))
        idx += 1

    if source_category:
        base_conditions.append(f"pr.source_category = ${idx}")
        params.append(source_category)
        idx += 1

    if brand:
        base_conditions.append(f"pm.brand ILIKE '%' || ${idx} || '%'")
        params.append(brand)
        idx += 1

    base_where = " AND ".join(base_conditions)

    # Fetch both product_comparisons and consideration_set in one query
    rows = await pool.fetch(
        f"""
        SELECT deep_extraction->'product_comparisons' AS comparisons,
               deep_extraction->'consideration_set' AS considerations,
               pm.brand, pr.asin, pr.rating
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE {base_where}
          AND (
            jsonb_array_length(COALESCE(pr.deep_extraction->'product_comparisons', '[]'::jsonb)) > 0
            OR jsonb_array_length(COALESCE(pr.deep_extraction->'consideration_set', '[]'::jsonb)) > 0
          )
        """,
        *params,
    )

    # Build flow graph: {from_brand -> to_brand -> direction -> stats}
    flow_map: dict[str, dict] = {}

    def _add_flow(from_b: str, to_b: str, flow_direction: str, rating):
        key = f"{from_b}|{to_b}|{flow_direction}"
        if key not in flow_map:
            flow_map[key] = {
                "from_brand": from_b,
                "to_brand": to_b,
                "direction": flow_direction,
                "count": 0,
                "ratings": [],
            }
        flow_map[key]["count"] += 1
        if rating is not None:
            flow_map[key]["ratings"].append(float(rating))

    for row in rows:
        from_brand = row["brand"] or "Unknown"

        # product_comparisons
        comps = _safe_json(row["comparisons"])
        if isinstance(comps, list):
            for comp in comps:
                if isinstance(comp, dict):
                    to_brand = comp.get("product_name") or comp.get("product", "Unknown")
                    comp_direction = comp.get("direction", "compared")
                    _add_flow(from_brand, to_brand, comp_direction, row["rating"])

        # consideration_set (rejected alternatives)
        cset = _safe_json(row["considerations"])
        if isinstance(cset, list):
            for item in cset:
                if isinstance(item, dict):
                    to_brand = item.get("product", "Unknown")
                    _add_flow(from_brand, to_brand, "considered", row["rating"])

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
            and (not direction or v["direction"] == direction)
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
    min_count: int = Query(1),
    limit: int = Query(50, le=200),
    user: AuthUser = Depends(require_auth),
):
    pool = _pool_or_503()
    conditions = [
        "pr.deep_extraction IS NOT NULL",
        "pr.deep_extraction != '{}'::jsonb",
    ]
    params: list = []
    idx = 1

    # Tenant scoping
    t_cond = _tenant_cond("pr", idx)
    if t_cond != "TRUE":
        conditions.append(t_cond)
        params.extend(_tenant_params(user))
        idx += 1

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
            if v["count"] >= min_count
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
    min_rating: Optional[float] = Query(None),
    max_rating: Optional[float] = Query(None),
    limit: int = Query(50, le=200),
    user: AuthUser = Depends(require_auth),
):
    pool = _pool_or_503()
    conditions = [
        "pr.deep_extraction->'safety_flag'->>'flagged' = 'true'",
    ]
    params: list = []
    idx = 1

    # Tenant scoping
    t_cond = _tenant_cond("pr", idx)
    if t_cond != "TRUE":
        conditions.append(t_cond)
        params.extend(_tenant_params(user))
        idx += 1

    if source_category:
        conditions.append(f"pr.source_category = ${idx}")
        params.append(source_category)
        idx += 1

    if brand:
        conditions.append(f"pm.brand ILIKE '%' || ${idx} || '%'")
        params.append(brand)
        idx += 1

    if min_rating is not None:
        conditions.append(f"pr.rating >= ${idx}")
        params.append(min_rating)
        idx += 1

    if max_rating is not None:
        conditions.append(f"pr.rating <= ${idx}")
        params.append(max_rating)
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

    # Count uses same filters (minus LIMIT) so total_flagged reflects active filters
    count_params = params[:-1]  # strip the limit param
    total_flagged = await pool.fetchval(
        f"""
        SELECT COUNT(*)
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE {where}
        """,
        *count_params,
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
    severity: Optional[str] = Query(None),
    enrichment_status: Optional[str] = Query(None),
    imported_after: Optional[str] = Query(None),
    imported_before: Optional[str] = Query(None),
    sort_by: str = Query("imported_at"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    user: AuthUser = Depends(require_auth),
):
    pool = _pool_or_503()
    conditions: list[str] = []
    params: list = []
    idx = 1

    # Tenant scoping
    t_cond = _tenant_cond("pr", idx)
    if t_cond != "TRUE":
        conditions.append(t_cond)
        params.extend(_tenant_params(user))
        idx += 1

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

    if severity:
        conditions.append(f"pr.severity = ${idx}")
        params.append(severity)
        idx += 1

    if enrichment_status:
        conditions.append(f"pr.enrichment_status = ${idx}")
        params.append(enrichment_status)
        idx += 1

    if imported_after:
        conditions.append(f"pr.imported_at >= ${idx}::timestamptz")
        params.append(imported_after)
        idx += 1

    if imported_before:
        conditions.append(f"pr.imported_at < (${idx}::date + interval '1 day')")
        params.append(imported_before)
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

    # Total count (without LIMIT/OFFSET) for pagination
    count_params = params[:-2]  # strip limit & offset
    total_count = await pool.fetchval(
        f"""
        SELECT COUNT(*)
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        {where}
        """,
        *count_params,
    )

    return {"reviews": reviews, "count": len(reviews), "total_count": total_count or 0}


# ---------------------------------------------------------------------------
# GET /reviews/{review_id}
# ---------------------------------------------------------------------------


@router.get("/reviews/{review_id}")
async def get_review(review_id: str, user: AuthUser = Depends(require_auth)):
    try:
        rid = _uuid.UUID(review_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid review_id (must be UUID)")

    pool = _pool_or_503()
    t_cond = _tenant_cond("pr", 2)
    t_params = _tenant_params(user)
    t_and = f"AND {t_cond}" if t_cond != "TRUE" else ""

    row = await pool.fetchrow(
        f"""
        SELECT pr.*, pm.brand, pm.title AS product_title,
               pm.average_rating AS product_avg_rating,
               pm.rating_number AS product_total_ratings,
               pm.price AS product_price
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.id = $1 {t_and}
        """,
        rid, *t_params,
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
