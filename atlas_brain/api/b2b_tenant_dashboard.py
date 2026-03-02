"""
Tenant-scoped B2B dashboard endpoints for P5 (Vendor Retention) and P6 (Challenger Lead Gen).

All endpoints require authentication and scope data to the tenant's tracked vendors.
"""

import json
import logging
import uuid as _uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth.dependencies import AuthUser, require_auth, require_b2b_plan
from ..config import settings
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.b2b_tenant")

router = APIRouter(prefix="/b2b/tenant", tags=["b2b-tenant"])


# ---------------------------------------------------------------------------
# Helpers (mirrors b2b_dashboard.py)
# ---------------------------------------------------------------------------

def _safe_json(val):
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


def _require_b2b_product(user: AuthUser):
    """Raise 403 if user is not on a B2B product."""
    if user.product not in ("b2b_retention", "b2b_challenger"):
        raise HTTPException(status_code=403, detail="B2B product required")


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class AddVendorRequest(BaseModel):
    vendor_name: str = Field(..., max_length=256)
    track_mode: str = Field(default="own", description="own (P5) | competitor (P6)")
    label: str = Field(default="", max_length=256)


class GenerateCampaignRequest(BaseModel):
    vendor_name: str = Field(..., max_length=256)
    company_filter: str = Field(default="", max_length=256)


class UpdateCampaignRequest(BaseModel):
    status: str = Field(..., description="approved | cancelled")


# ---------------------------------------------------------------------------
# Vendor tracking (4 endpoints)
# ---------------------------------------------------------------------------

@router.get("/vendors")
async def list_tracked_vendors(user: AuthUser = Depends(require_auth)):
    """List tracked vendors with inline signal summary."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)

    rows = await pool.fetch(
        """
        SELECT tv.id, tv.vendor_name, tv.track_mode, tv.label, tv.added_at,
               cs.avg_urgency_score, cs.churn_intent_count, cs.total_reviews,
               cs.nps_proxy
        FROM tracked_vendors tv
        LEFT JOIN b2b_churn_signals cs ON cs.vendor_name = tv.vendor_name
        WHERE tv.account_id = $1
        ORDER BY tv.added_at
        """,
        acct,
    )

    vendors = [
        {
            "id": str(r["id"]),
            "vendor_name": r["vendor_name"],
            "track_mode": r["track_mode"],
            "label": r["label"],
            "added_at": str(r["added_at"]) if r["added_at"] else None,
            "avg_urgency": _safe_float(r["avg_urgency_score"]),
            "churn_intent_count": r["churn_intent_count"],
            "total_reviews": r["total_reviews"],
            "nps_proxy": _safe_float(r["nps_proxy"]),
        }
        for r in rows
    ]

    return {"vendors": vendors, "count": len(vendors)}


@router.post("/vendors")
async def add_tracked_vendor(req: AddVendorRequest, user: AuthUser = Depends(require_auth)):
    """Add a vendor to track. Enforces vendor_limit in a transaction."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)

    if req.track_mode not in ("own", "competitor"):
        raise HTTPException(status_code=400, detail="track_mode must be 'own' or 'competitor'")

    async with pool.transaction() as conn:
        current_count = await conn.fetchval(
            "SELECT COUNT(*) FROM tracked_vendors WHERE account_id = $1",
            acct,
        )
        vendor_limit = await conn.fetchval(
            "SELECT vendor_limit FROM saas_accounts WHERE id = $1",
            acct,
        )
        if current_count >= (vendor_limit or 1):
            raise HTTPException(
                status_code=403,
                detail="Vendor limit reached. Upgrade your plan for more.",
            )

        row = await conn.fetchrow(
            """
            INSERT INTO tracked_vendors (account_id, vendor_name, track_mode, label)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (account_id, vendor_name) DO NOTHING
            RETURNING id, vendor_name, track_mode, label, added_at
            """,
            acct,
            req.vendor_name.strip(),
            req.track_mode,
            req.label.strip() if req.label else None,
        )

    if not row:
        raise HTTPException(status_code=409, detail="Vendor already tracked")

    return {
        "id": str(row["id"]),
        "vendor_name": row["vendor_name"],
        "track_mode": row["track_mode"],
        "label": row["label"],
        "added_at": str(row["added_at"]) if row["added_at"] else None,
    }


@router.delete("/vendors/{vendor_name}")
async def remove_tracked_vendor(vendor_name: str, user: AuthUser = Depends(require_auth)):
    """Remove a tracked vendor."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)

    result = await pool.execute(
        "DELETE FROM tracked_vendors WHERE account_id = $1 AND vendor_name = $2",
        acct,
        vendor_name.strip(),
    )

    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Vendor not found in tracked list")

    return {"status": "ok"}


@router.get("/vendors/search")
async def search_available_vendors(
    q: str = Query(..., min_length=1, max_length=256),
    limit: int = Query(20, ge=1, le=50),
    user: AuthUser = Depends(require_auth),
):
    """Search b2b_churn_signals for vendors matching query (for onboarding)."""
    _require_b2b_product(user)
    pool = _pool_or_503()

    rows = await pool.fetch(
        """
        SELECT DISTINCT vendor_name, product_category,
               total_reviews, avg_urgency_score
        FROM b2b_churn_signals
        WHERE vendor_name ILIKE '%' || $1 || '%'
        ORDER BY total_reviews DESC
        LIMIT $2
        """,
        q.strip(),
        min(limit, 50),
    )

    return {
        "vendors": [
            {
                "vendor_name": r["vendor_name"],
                "product_category": r["product_category"],
                "total_reviews": r["total_reviews"],
                "avg_urgency": _safe_float(r["avg_urgency_score"]),
            }
            for r in rows
        ],
        "count": len(rows),
    }


# ---------------------------------------------------------------------------
# Tenant scope helper
# ---------------------------------------------------------------------------

def _vendor_scope_sql(param_idx: int) -> str:
    """SQL clause restricting to tracked vendors for the account."""
    if not settings.saas_auth.enabled:
        return "TRUE"
    return f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${param_idx})"


def _tenant_params(user: AuthUser) -> list:
    if not settings.saas_auth.enabled:
        return []
    return [_uuid.UUID(user.account_id)]


# ---------------------------------------------------------------------------
# Health & signals (5 endpoints)
# ---------------------------------------------------------------------------

@router.get("/overview")
async def dashboard_overview(user: AuthUser = Depends(require_auth)):
    """Dashboard overview stats for tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)

    # Vendor count
    vendor_count = await pool.fetchval(
        "SELECT COUNT(*) FROM tracked_vendors WHERE account_id = $1",
        acct,
    )

    # Signal summary for tracked vendors
    signal_stats = await pool.fetchrow(
        f"""
        SELECT COALESCE(AVG(avg_urgency_score), 0) AS avg_urgency,
               COALESCE(SUM(churn_intent_count), 0) AS total_churn_signals,
               COALESCE(SUM(total_reviews), 0) AS total_reviews
        FROM b2b_churn_signals
        WHERE {_vendor_scope_sql(1)}
        """,
        *_tenant_params(user),
    )

    # Recent high-intent leads
    lead_rows = await pool.fetch(
        f"""
        SELECT reviewer_company, vendor_name,
               (enrichment->>'urgency_score')::numeric AS urgency,
               enrichment->>'pain_category' AS pain
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND reviewer_company IS NOT NULL AND reviewer_company != ''
          AND (enrichment->>'urgency_score')::numeric >= 7
          AND {_vendor_scope_sql(1)}
        ORDER BY (enrichment->>'urgency_score')::numeric DESC
        LIMIT 5
        """,
        *_tenant_params(user),
    )

    return {
        "tracked_vendors": vendor_count,
        "avg_urgency": _safe_float(signal_stats["avg_urgency"] if signal_stats else 0, 0),
        "total_churn_signals": signal_stats["total_churn_signals"] if signal_stats else 0,
        "total_reviews": signal_stats["total_reviews"] if signal_stats else 0,
        "recent_leads": [
            {
                "company": r["reviewer_company"],
                "vendor": r["vendor_name"],
                "urgency": _safe_float(r["urgency"], 0),
                "pain": r["pain"],
            }
            for r in lead_rows
        ],
    }


@router.get("/signals")
async def list_tenant_signals(
    min_urgency: float = Query(0, ge=0, le=10),
    category: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser = Depends(require_auth),
):
    """Churn signals for tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()

    conditions: list[str] = []
    params: list = []
    idx = 1

    # Tenant scope
    t_params = _tenant_params(user)
    scope = _vendor_scope_sql(idx)
    if scope != "TRUE":
        conditions.append(scope)
        params.extend(t_params)
        idx += 1

    if min_urgency > 0:
        conditions.append(f"avg_urgency_score >= ${idx}")
        params.append(min_urgency)
        idx += 1

    if category:
        conditions.append(f"product_category = ${idx}")
        params.append(category)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    capped = min(limit, 100)
    params.append(capped)

    rows = await pool.fetch(
        f"""
        SELECT vendor_name, product_category, total_reviews,
               churn_intent_count, avg_urgency_score, avg_rating_normalized,
               nps_proxy, price_complaint_rate, decision_maker_churn_rate,
               last_computed_at
        FROM b2b_churn_signals
        {where}
        ORDER BY avg_urgency_score DESC
        LIMIT ${idx}
        """,
        *params,
    )

    signals = [
        {
            "vendor_name": r["vendor_name"],
            "product_category": r["product_category"],
            "total_reviews": r["total_reviews"],
            "churn_intent_count": r["churn_intent_count"],
            "avg_urgency_score": _safe_float(r["avg_urgency_score"], 0.0),
            "avg_rating_normalized": _safe_float(r["avg_rating_normalized"]),
            "nps_proxy": _safe_float(r["nps_proxy"]),
            "price_complaint_rate": _safe_float(r["price_complaint_rate"]),
            "decision_maker_churn_rate": _safe_float(r["decision_maker_churn_rate"]),
            "last_computed_at": str(r["last_computed_at"]) if r["last_computed_at"] else None,
        }
        for r in rows
    ]

    return {"signals": signals, "count": len(signals)}


@router.get("/signals/{vendor_name}")
async def get_vendor_detail(vendor_name: str, user: AuthUser = Depends(require_auth)):
    """Full vendor detail: signal + pain + competitors + evidence."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    vname = vendor_name.strip()

    # Verify vendor is tracked by this account
    if settings.saas_auth.enabled:
        acct = _uuid.UUID(user.account_id)
        tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1 AND vendor_name = $2",
            acct,
            vname,
        )
        if not tracked:
            raise HTTPException(status_code=403, detail="Vendor not in your tracked list")

    signal_row = await pool.fetchrow(
        """
        SELECT * FROM b2b_churn_signals
        WHERE vendor_name ILIKE '%' || $1 || '%'
        ORDER BY avg_urgency_score DESC LIMIT 1
        """,
        vname,
    )

    counts = await pool.fetchrow(
        """
        SELECT COUNT(*) AS total_reviews,
               COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched
        FROM b2b_reviews
        WHERE vendor_name ILIKE '%' || $1 || '%'
        """,
        vname,
    )

    pain_rows = await pool.fetch(
        """
        SELECT enrichment->>'pain_category' AS pain, COUNT(*) AS cnt
        FROM b2b_reviews
        WHERE vendor_name ILIKE '%' || $1 || '%'
          AND enrichment_status = 'enriched'
          AND enrichment->>'pain_category' IS NOT NULL
        GROUP BY enrichment->>'pain_category'
        ORDER BY cnt DESC LIMIT 50
        """,
        vname,
    )

    hi_rows = await pool.fetch(
        """
        SELECT reviewer_company,
               (enrichment->>'urgency_score')::numeric AS urgency,
               enrichment->>'pain_category' AS pain
        FROM b2b_reviews
        WHERE vendor_name ILIKE '%' || $1 || '%'
          AND enrichment_status = 'enriched'
          AND (enrichment->>'urgency_score')::numeric >= 7
          AND reviewer_company IS NOT NULL AND reviewer_company != ''
        ORDER BY (enrichment->>'urgency_score')::numeric DESC LIMIT 10
        """,
        vname,
    )

    profile: dict = {"vendor_name": vname}

    if signal_row:
        profile["churn_signal"] = {
            "avg_urgency_score": _safe_float(signal_row["avg_urgency_score"], 0.0),
            "churn_intent_count": signal_row["churn_intent_count"],
            "total_reviews": signal_row["total_reviews"],
            "nps_proxy": _safe_float(signal_row["nps_proxy"]),
            "price_complaint_rate": _safe_float(signal_row["price_complaint_rate"]),
            "decision_maker_churn_rate": _safe_float(signal_row["decision_maker_churn_rate"]),
            "top_pain_categories": _safe_json(signal_row["top_pain_categories"]),
            "top_competitors": _safe_json(signal_row["top_competitors"]),
            "top_feature_gaps": _safe_json(signal_row["top_feature_gaps"]),
            "quotable_evidence": _safe_json(signal_row["quotable_evidence"]),
            "top_use_cases": _safe_json(signal_row["top_use_cases"]),
            "top_integration_stacks": _safe_json(signal_row["top_integration_stacks"]),
            "budget_signal_summary": _safe_json(signal_row["budget_signal_summary"]),
            "sentiment_distribution": _safe_json(signal_row["sentiment_distribution"]),
            "buyer_authority_summary": _safe_json(signal_row["buyer_authority_summary"]),
            "timeline_summary": _safe_json(signal_row["timeline_summary"]),
            "last_computed_at": str(signal_row["last_computed_at"]) if signal_row["last_computed_at"] else None,
        }
    else:
        profile["churn_signal"] = None

    profile["review_counts"] = {
        "total": counts["total_reviews"] if counts else 0,
        "enriched": counts["enriched"] if counts else 0,
    }

    profile["high_intent_companies"] = [
        {
            "company": r["reviewer_company"],
            "urgency": _safe_float(r["urgency"], 0),
            "pain": r["pain"],
        }
        for r in hi_rows
    ]

    profile["pain_distribution"] = [
        {"pain_category": r["pain"], "count": r["cnt"]}
        for r in pain_rows
    ]

    return profile


@router.get("/pain-trends")
async def pain_trends(
    window_days: int = Query(90, ge=1, le=3650),
    user: AuthUser = Depends(require_auth),
):
    """Pain categories over time for tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()

    t_params = _tenant_params(user)
    idx = 1
    conditions = []
    params: list = []

    scope = _vendor_scope_sql(idx)
    if scope != "TRUE":
        conditions.append(scope)
        params.extend(t_params)
        idx += 1

    conditions.append("enrichment_status = 'enriched'")
    conditions.append(f"enriched_at > NOW() - make_interval(days => ${idx})")
    params.append(window_days)
    idx += 1
    conditions.append("enrichment->>'pain_category' IS NOT NULL")

    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT DATE_TRUNC('week', enriched_at) AS week,
               enrichment->>'pain_category' AS pain,
               COUNT(*) AS cnt
        FROM b2b_reviews
        WHERE {where}
        GROUP BY week, pain
        ORDER BY week, cnt DESC
        """,
        *params,
    )

    trends = [
        {
            "week": str(r["week"]),
            "pain_category": r["pain"],
            "count": r["cnt"],
        }
        for r in rows
    ]

    return {"trends": trends, "count": len(trends)}


@router.get("/displacement")
async def competitor_displacement(
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser = Depends(require_auth),
):
    """Competitor displacement flows for tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()

    t_params = _tenant_params(user)
    idx = 1
    conditions = ["enrichment_status = 'enriched'"]
    params: list = []

    scope = _vendor_scope_sql(idx)
    if scope != "TRUE":
        conditions.append(scope)
        params.extend(t_params)
        idx += 1

    conditions.append("enrichment->'competitors_mentioned' IS NOT NULL")
    conditions.append("jsonb_array_length(enrichment->'competitors_mentioned') > 0")

    where = " AND ".join(conditions)
    capped = min(limit, 100)
    params.append(capped)

    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
               enrichment->'competitors_mentioned' AS competitors,
               (enrichment->'churn_signals'->>'intent_to_leave')::boolean AS leaving,
               COUNT(*) AS mention_count
        FROM b2b_reviews
        WHERE {where}
        GROUP BY vendor_name, competitors, leaving
        ORDER BY mention_count DESC
        LIMIT ${idx}
        """,
        *params,
    )

    flows = [
        {
            "vendor_name": r["vendor_name"],
            "competitors": _safe_json(r["competitors"]),
            "leaving": r["leaving"],
            "mention_count": r["mention_count"],
        }
        for r in rows
    ]

    return {"displacement": flows, "count": len(flows)}


# ---------------------------------------------------------------------------
# Leads (2 endpoints -- P6 focused)
# ---------------------------------------------------------------------------

@router.get("/leads")
async def list_leads(
    min_urgency: float = Query(7, ge=0, le=10),
    window_days: int = Query(30, ge=1, le=3650),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser = Depends(require_auth),
):
    """High-intent companies leaving tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()

    t_params = _tenant_params(user)
    idx = 1
    conditions = [
        "enrichment_status = 'enriched'",
        "reviewer_company IS NOT NULL AND reviewer_company != ''",
    ]
    params: list = []

    scope = _vendor_scope_sql(idx)
    if scope != "TRUE":
        conditions.append(scope)
        params.extend(t_params)
        idx += 1

    conditions.append(f"(enrichment->>'urgency_score')::numeric >= ${idx}")
    params.append(min_urgency)
    idx += 1

    conditions.append(f"enriched_at > NOW() - make_interval(days => ${idx})")
    params.append(window_days)
    idx += 1

    where = " AND ".join(conditions)
    capped = min(limit, 100)
    params.append(capped)

    rows = await pool.fetch(
        f"""
        SELECT reviewer_company, vendor_name, product_category,
               enrichment->'reviewer_context'->>'role_level' AS role_level,
               (enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
               (enrichment->>'urgency_score')::numeric AS urgency,
               enrichment->>'pain_category' AS pain,
               enrichment->'competitors_mentioned' AS alternatives,
               enrichment->'contract_context'->>'contract_value_signal' AS value_signal,
               CASE WHEN enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                    THEN (enrichment->'budget_signals'->>'seat_count')::int END AS seat_count,
               enrichment->'timeline'->>'contract_end' AS contract_end,
               enrichment->'buyer_authority'->>'buying_stage' AS buying_stage
        FROM b2b_reviews
        WHERE {where}
        ORDER BY (enrichment->>'urgency_score')::numeric DESC
        LIMIT ${idx}
        """,
        *params,
    )

    companies = [
        {
            "company": r["reviewer_company"],
            "vendor": r["vendor_name"],
            "category": r["product_category"],
            "role_level": r["role_level"],
            "decision_maker": r["is_dm"],
            "urgency": _safe_float(r["urgency"], 0),
            "pain": r["pain"],
            "alternatives": _safe_json(r["alternatives"]),
            "contract_signal": r["value_signal"],
            "seat_count": r["seat_count"],
            "contract_end": r["contract_end"],
            "buying_stage": r["buying_stage"],
        }
        for r in rows
    ]

    return {"leads": companies, "count": len(companies)}


@router.get("/leads/{company}")
async def get_lead_detail(company: str, user: AuthUser = Depends(require_auth)):
    """Company drill-down: all reviews, signals, buying stage."""
    _require_b2b_product(user)
    pool = _pool_or_503()

    t_params = _tenant_params(user)
    idx = 1
    conditions = ["enrichment_status = 'enriched'"]
    params: list = []

    scope = _vendor_scope_sql(idx)
    if scope != "TRUE":
        conditions.append(scope)
        params.extend(t_params)
        idx += 1

    conditions.append(f"reviewer_company ILIKE ${idx}")
    params.append(company.strip())
    idx += 1

    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT id, vendor_name, product_category, rating,
               (enrichment->>'urgency_score')::numeric AS urgency,
               enrichment->>'pain_category' AS pain,
               (enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
               (enrichment->'reviewer_context'->>'decision_maker')::boolean AS decision_maker,
               enrichment->'reviewer_context'->>'role_level' AS role_level,
               enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
               enrichment->'competitors_mentioned' AS alternatives,
               enrichment->'timeline'->>'contract_end' AS contract_end,
               enriched_at
        FROM b2b_reviews
        WHERE {where}
        ORDER BY (enrichment->>'urgency_score')::numeric DESC
        LIMIT 50
        """,
        *params,
    )

    if not rows:
        raise HTTPException(status_code=404, detail="No reviews found for this company")

    reviews = [
        {
            "id": str(r["id"]),
            "vendor_name": r["vendor_name"],
            "category": r["product_category"],
            "rating": _safe_float(r["rating"]),
            "urgency": _safe_float(r["urgency"], 0),
            "pain": r["pain"],
            "intent_to_leave": r["intent_to_leave"],
            "decision_maker": r["decision_maker"],
            "role_level": r["role_level"],
            "buying_stage": r["buying_stage"],
            "alternatives": _safe_json(r["alternatives"]),
            "contract_end": r["contract_end"],
            "enriched_at": str(r["enriched_at"]) if r["enriched_at"] else None,
        }
        for r in rows
    ]

    return {
        "company": company.strip(),
        "reviews": reviews,
        "count": len(reviews),
    }


# ---------------------------------------------------------------------------
# Reports + Reviews (4 endpoints)
# ---------------------------------------------------------------------------

@router.get("/reports")
async def list_tenant_reports(
    report_type: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    user: AuthUser = Depends(require_auth),
):
    """Reports scoped to tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()

    t_params = _tenant_params(user)
    idx = 1
    conditions: list[str] = []
    params: list = []

    # Scope by vendor_filter matching tracked vendors
    scope = _vendor_scope_sql(idx)
    if scope != "TRUE":
        conditions.append(
            f"vendor_filter IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx})"
        )
        params.extend(t_params)
        idx += 1

    if report_type:
        conditions.append(f"report_type = ${idx}")
        params.append(report_type)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    capped = min(limit, 50)
    params.append(capped)

    rows = await pool.fetch(
        f"""
        SELECT id, report_date, report_type, executive_summary,
               vendor_filter, status, created_at
        FROM b2b_intelligence
        {where}
        ORDER BY report_date DESC
        LIMIT ${idx}
        """,
        *params,
    )

    reports = [
        {
            "id": str(r["id"]),
            "report_date": str(r["report_date"]) if r["report_date"] else None,
            "report_type": r["report_type"],
            "executive_summary": r["executive_summary"],
            "vendor_filter": r["vendor_filter"],
            "status": r["status"],
            "created_at": str(r["created_at"]) if r["created_at"] else None,
        }
        for r in rows
    ]

    return {"reports": reports, "count": len(reports)}


@router.get("/reports/{report_id}")
async def get_tenant_report(report_id: str, user: AuthUser = Depends(require_auth)):
    """Report detail (verify vendor in tracked)."""
    _require_b2b_product(user)
    try:
        rid = _uuid.UUID(report_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid report_id (must be UUID)")

    pool = _pool_or_503()
    row = await pool.fetchrow("SELECT * FROM b2b_intelligence WHERE id = $1", rid)
    if not row:
        raise HTTPException(status_code=404, detail="Report not found")

    # Verify vendor is tracked
    if settings.saas_auth.enabled and row["vendor_filter"]:
        acct = _uuid.UUID(user.account_id)
        tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1 AND vendor_name = $2",
            acct,
            row["vendor_filter"],
        )
        if not tracked:
            raise HTTPException(status_code=403, detail="Report vendor not in your tracked list")

    return {
        "id": str(row["id"]),
        "report_date": str(row["report_date"]) if row["report_date"] else None,
        "report_type": row["report_type"],
        "vendor_filter": row["vendor_filter"],
        "category_filter": row["category_filter"],
        "executive_summary": row["executive_summary"],
        "intelligence_data": _safe_json(row["intelligence_data"]),
        "data_density": _safe_json(row["data_density"]),
        "status": row["status"],
        "llm_model": row["llm_model"],
        "created_at": str(row["created_at"]) if row["created_at"] else None,
    }


@router.get("/reviews")
async def list_tenant_reviews(
    pain_category: Optional[str] = Query(None),
    min_urgency: Optional[float] = Query(None, ge=0, le=10),
    company: Optional[str] = Query(None),
    has_churn_intent: Optional[bool] = Query(None),
    window_days: int = Query(90, ge=1, le=3650),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser = Depends(require_auth),
):
    """Reviews scoped to tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()

    t_params = _tenant_params(user)
    idx = 1
    conditions = ["enrichment_status = 'enriched'"]
    params: list = []

    scope = _vendor_scope_sql(idx)
    if scope != "TRUE":
        conditions.append(scope)
        params.extend(t_params)
        idx += 1

    conditions.append(f"enriched_at > NOW() - make_interval(days => ${idx})")
    params.append(window_days)
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

    where = " AND ".join(conditions)
    capped = min(limit, 100)
    params.append(capped)

    rows = await pool.fetch(
        f"""
        SELECT id, vendor_name, product_category, reviewer_company,
               rating,
               (enrichment->>'urgency_score')::numeric AS urgency_score,
               enrichment->>'pain_category' AS pain_category,
               (enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
               (enrichment->'reviewer_context'->>'decision_maker')::boolean AS decision_maker,
               enriched_at
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
            "rating": _safe_float(r["rating"]),
            "urgency_score": _safe_float(r["urgency_score"]),
            "pain_category": r["pain_category"],
            "intent_to_leave": r["intent_to_leave"],
            "decision_maker": r["decision_maker"],
            "enriched_at": str(r["enriched_at"]) if r["enriched_at"] else None,
        }
        for r in rows
    ]

    return {"reviews": reviews, "count": len(reviews)}


@router.get("/reviews/{review_id}")
async def get_tenant_review(review_id: str, user: AuthUser = Depends(require_auth)):
    """Review detail (verify vendor in tracked)."""
    _require_b2b_product(user)
    try:
        rid = _uuid.UUID(review_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid review_id (must be UUID)")

    pool = _pool_or_503()
    row = await pool.fetchrow("SELECT * FROM b2b_reviews WHERE id = $1", rid)
    if not row:
        raise HTTPException(status_code=404, detail="Review not found")

    # Verify vendor is tracked
    if settings.saas_auth.enabled:
        acct = _uuid.UUID(user.account_id)
        tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1 AND vendor_name = $2",
            acct,
            row["vendor_name"],
        )
        if not tracked:
            raise HTTPException(status_code=403, detail="Review vendor not in your tracked list")

    return {
        "id": str(row["id"]),
        "source": row["source"],
        "source_url": row["source_url"],
        "vendor_name": row["vendor_name"],
        "product_name": row["product_name"],
        "product_category": row["product_category"],
        "rating": _safe_float(row["rating"]),
        "summary": row["summary"],
        "review_text": row["review_text"],
        "pros": row["pros"],
        "cons": row["cons"],
        "reviewer_name": row["reviewer_name"],
        "reviewer_title": row["reviewer_title"],
        "reviewer_company": row["reviewer_company"],
        "company_size_raw": row["company_size_raw"],
        "reviewer_industry": row["reviewer_industry"],
        "reviewed_at": str(row["reviewed_at"]) if row["reviewed_at"] else None,
        "enrichment": _safe_json(row["enrichment"]),
        "enrichment_status": row["enrichment_status"],
        "enriched_at": str(row["enriched_at"]) if row["enriched_at"] else None,
    }


# ---------------------------------------------------------------------------
# Campaigns (3 endpoints -- b2b_growth+ only)
# ---------------------------------------------------------------------------

@router.get("/campaigns")
async def list_tenant_campaigns(
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser = Depends(require_b2b_plan("b2b_growth")),
):
    """Campaigns scoped to tracked vendors. Requires b2b_growth+ plan."""
    pool = _pool_or_503()

    t_params = _tenant_params(user)
    idx = 1
    conditions: list[str] = []
    params: list = []

    # Scope campaigns by vendor_name matching tracked vendors
    scope = _vendor_scope_sql(idx)
    if scope != "TRUE":
        # b2b_campaigns uses company_name, but we scope via metadata.vendor_name
        conditions.append(
            f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx})"
        )
        params.extend(t_params)
        idx += 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    capped = min(limit, 100)
    params.append(capped)

    rows = await pool.fetch(
        f"""
        SELECT id, company_name, vendor_name, channel, subject,
               status, approved_at, sent_at, created_at
        FROM b2b_campaigns
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx}
        """,
        *params,
    )

    campaigns = [
        {
            "id": str(r["id"]),
            "company_name": r["company_name"],
            "vendor_name": r["vendor_name"],
            "channel": r["channel"],
            "subject": r["subject"],
            "status": r["status"],
            "approved_at": str(r["approved_at"]) if r["approved_at"] else None,
            "sent_at": str(r["sent_at"]) if r["sent_at"] else None,
            "created_at": str(r["created_at"]) if r["created_at"] else None,
        }
        for r in rows
    ]

    return {"campaigns": campaigns, "count": len(campaigns)}


@router.post("/campaigns/generate")
async def generate_campaigns(
    req: GenerateCampaignRequest,
    user: AuthUser = Depends(require_b2b_plan("b2b_growth")),
):
    """Generate campaign drafts for a tracked vendor's high-intent leads."""
    pool = _pool_or_503()
    vname = req.vendor_name.strip()

    # Verify vendor is tracked
    if settings.saas_auth.enabled:
        acct = _uuid.UUID(user.account_id)
        tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1 AND vendor_name = $2",
            acct,
            vname,
        )
        if not tracked:
            raise HTTPException(status_code=403, detail="Vendor not in your tracked list")

    # Find high-intent leads for this vendor
    conditions = [
        "enrichment_status = 'enriched'",
        "reviewer_company IS NOT NULL AND reviewer_company != ''",
        "(enrichment->>'urgency_score')::numeric >= 7",
        "vendor_name ILIKE '%' || $1 || '%'",
    ]
    params: list = [vname]
    idx = 2

    if req.company_filter:
        conditions.append(f"reviewer_company ILIKE '%' || ${idx} || '%'")
        params.append(req.company_filter.strip())
        idx += 1

    where = " AND ".join(conditions)

    leads = await pool.fetch(
        f"""
        SELECT DISTINCT ON (reviewer_company) reviewer_company,
               (enrichment->>'urgency_score')::numeric AS urgency,
               enrichment->>'pain_category' AS pain,
               enrichment->'competitors_mentioned' AS competitors,
               enrichment->'buyer_authority'->>'buying_stage' AS buying_stage
        FROM b2b_reviews
        WHERE {where}
        ORDER BY reviewer_company, (enrichment->>'urgency_score')::numeric DESC
        LIMIT 20
        """,
        *params,
    )

    if not leads:
        return {"campaigns_created": 0, "message": "No high-intent leads found for this vendor"}

    # Create draft campaigns for each lead
    created = 0
    for lead in leads:
        pain = lead["pain"] or "general dissatisfaction"
        company = lead["reviewer_company"]

        await pool.execute(
            """
            INSERT INTO b2b_campaigns (company_name, vendor_name, channel, subject, body, status, batch_id)
            VALUES ($1, $2, 'email_cold', $3, $4, 'draft', $5)
            ON CONFLICT DO NOTHING
            """,
            company,
            vname,
            f"Re: {pain} challenges with {vname}",
            f"Draft campaign targeting {company} (pain: {pain}, urgency: {_safe_float(lead['urgency'], 0)}, stage: {lead['buying_stage'] or 'unknown'})",
            f"tenant_{user.account_id}_{vname}",
        )
        created += 1

    return {"campaigns_created": created}


@router.patch("/campaigns/{campaign_id}")
async def update_campaign(
    campaign_id: str,
    req: UpdateCampaignRequest,
    user: AuthUser = Depends(require_b2b_plan("b2b_growth")),
):
    """Approve or cancel a campaign draft."""
    try:
        cid = _uuid.UUID(campaign_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid campaign_id (must be UUID)")

    if req.status not in ("approved", "cancelled"):
        raise HTTPException(status_code=400, detail="Status must be 'approved' or 'cancelled'")

    pool = _pool_or_503()

    # Fetch campaign and verify vendor is tracked
    row = await pool.fetchrow("SELECT vendor_name, status FROM b2b_campaigns WHERE id = $1", cid)
    if not row:
        raise HTTPException(status_code=404, detail="Campaign not found")

    if settings.saas_auth.enabled:
        acct = _uuid.UUID(user.account_id)
        tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1 AND vendor_name = $2",
            acct,
            row["vendor_name"],
        )
        if not tracked:
            raise HTTPException(status_code=403, detail="Campaign vendor not in your tracked list")

    if row["status"] != "draft":
        raise HTTPException(status_code=400, detail=f"Campaign is '{row['status']}', can only update drafts")

    update_fields = "status = $1, updated_at = NOW()"
    if req.status == "approved":
        update_fields += ", approved_at = NOW()"

    await pool.execute(
        f"UPDATE b2b_campaigns SET {update_fields} WHERE id = $2",
        req.status,
        cid,
    )

    return {"status": "ok", "campaign_id": str(cid), "new_status": req.status}
