"""
Read-only REST endpoints for the B2B Churn Intelligence Dashboard.

Mirrors the SQL from ``atlas_brain.mcp.b2b_churn_server`` so the frontend
can query data directly over HTTP instead of going through MCP stdio.
"""

import csv
import io
import json
import logging
import uuid as _uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from ..auth.dependencies import AuthUser, optional_auth
from ..services.scraping.capabilities import get_capability
from ..services.scraping.sources import ALL_SOURCES, ReviewSource, display_name as source_display_name
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.b2b_dashboard")

EXPORT_ROW_LIMIT = 10_000

router = APIRouter(prefix="/b2b/dashboard", tags=["b2b-dashboard"])


class VendorComparisonRequest(BaseModel):
    primary_vendor: str = Field(..., min_length=1, max_length=200)
    comparison_vendor: str = Field(..., min_length=1, max_length=200)
    window_days: int = Field(90, ge=1, le=3650)
    persist: bool = True


class AccountComparisonRequest(BaseModel):
    primary_company: str = Field(..., min_length=1, max_length=200)
    comparison_company: str = Field(..., min_length=1, max_length=200)
    window_days: int = Field(90, ge=1, le=3650)
    persist: bool = True


class AccountDeepDiveRequest(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=200)
    window_days: int = Field(90, ge=1, le=3650)
    persist: bool = True


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
# GET /signals
# ---------------------------------------------------------------------------


@router.get("/signals")
async def list_signals(
    vendor_name: Optional[str] = Query(None),
    min_urgency: float = Query(0, ge=0, le=10),
    category: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    conditions: list[str] = []
    params: list = []
    idx = 1

    if user:
        conditions.append(f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)")
        params.append(user.account_id)
        idx += 1

    if vendor_name:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
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


# ---------------------------------------------------------------------------
# GET /signals/{vendor_name}
# ---------------------------------------------------------------------------


@router.get("/signals/{vendor_name}")
async def get_signal(
    vendor_name: str,
    product_category: Optional[str] = Query(None),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    vname = vendor_name.strip()

    # Build optional tenant scope clause
    scope = ""
    scope_params: list = [vname]
    if user:
        scope = f" AND vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${ len(scope_params) + 1 }::uuid)"
        scope_params.append(user.account_id)

    if product_category:
        pidx = len(scope_params) + 1
        row = await pool.fetchrow(
            f"""
            SELECT * FROM b2b_churn_signals
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND product_category = ${pidx}
              {scope}
            ORDER BY avg_urgency_score DESC
            LIMIT 1
            """,
            *scope_params,
            product_category,
        )
    else:
        row = await pool.fetchrow(
            f"""
            SELECT * FROM b2b_churn_signals
            WHERE vendor_name ILIKE '%' || $1 || '%'
              {scope}
            ORDER BY avg_urgency_score DESC
            LIMIT 1
            """,
            *scope_params,
        )

    if not row:
        raise HTTPException(status_code=404, detail="No churn signal found for that vendor")

    return {
        "vendor_name": row["vendor_name"],
        "product_category": row["product_category"],
        "total_reviews": row["total_reviews"],
        "negative_reviews": row["negative_reviews"],
        "churn_intent_count": row["churn_intent_count"],
        "avg_urgency_score": _safe_float(row["avg_urgency_score"], 0.0),
        "avg_rating_normalized": _safe_float(row["avg_rating_normalized"]),
        "nps_proxy": _safe_float(row["nps_proxy"]),
        "price_complaint_rate": _safe_float(row["price_complaint_rate"]),
        "decision_maker_churn_rate": _safe_float(row["decision_maker_churn_rate"]),
        "top_pain_categories": _safe_json(row["top_pain_categories"]),
        "top_competitors": _safe_json(row["top_competitors"]),
        "top_feature_gaps": _safe_json(row["top_feature_gaps"]),
        "company_churn_list": _safe_json(row["company_churn_list"]),
        "quotable_evidence": _safe_json(row["quotable_evidence"]),
        "top_use_cases": _safe_json(row["top_use_cases"]),
        "top_integration_stacks": _safe_json(row["top_integration_stacks"]),
        "budget_signal_summary": _safe_json(row["budget_signal_summary"]),
        "sentiment_distribution": _safe_json(row["sentiment_distribution"]),
        "buyer_authority_summary": _safe_json(row["buyer_authority_summary"]),
        "timeline_summary": _safe_json(row["timeline_summary"]),
        "last_computed_at": str(row["last_computed_at"]) if row["last_computed_at"] else None,
        "created_at": str(row["created_at"]) if row["created_at"] else None,
    }


# ---------------------------------------------------------------------------
# GET /high-intent
# ---------------------------------------------------------------------------


@router.get("/high-intent")
async def list_high_intent(
    vendor_name: Optional[str] = Query(None),
    min_urgency: float = Query(7, ge=0, le=10),
    window_days: int = Query(30, ge=1, le=3650),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    conditions = [
        "enrichment_status = 'enriched'",
        "(enrichment->>'urgency_score')::numeric >= $1",
        "reviewer_company IS NOT NULL AND reviewer_company != ''",
        "enriched_at > NOW() - make_interval(days => $2)",
    ]
    params: list = [min_urgency, window_days]
    idx = 3

    if user:
        conditions.append(f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)")
        params.append(user.account_id)
        idx += 1

    if vendor_name:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1

    capped = min(limit, 100)
    params.append(capped)
    where = " AND ".join(conditions)

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
               enrichment->'use_case'->>'lock_in_level' AS lock_in_level,
               enrichment->'timeline'->>'contract_end' AS contract_end,
               enrichment->'buyer_authority'->>'buying_stage' AS buying_stage
        FROM b2b_reviews
        WHERE {where}
        ORDER BY (enrichment->>'urgency_score')::numeric DESC
        LIMIT ${idx}
        """,
        *params,
    )

    companies = []
    for r in rows:
        companies.append({
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
            "lock_in_level": r["lock_in_level"],
            "contract_end": r["contract_end"],
            "buying_stage": r["buying_stage"],
        })

    return {"companies": companies, "count": len(companies)}


# ---------------------------------------------------------------------------
# GET /vendors/{vendor_name}
# ---------------------------------------------------------------------------


@router.get("/vendors/{vendor_name}")
async def get_vendor_profile(vendor_name: str, user: AuthUser | None = Depends(optional_auth)):
    pool = _pool_or_503()
    vname = vendor_name.strip()

    # When authenticated, verify vendor is tracked by the user's account
    if user:
        is_tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1::uuid AND vendor_name ILIKE $2",
            user.account_id, vname,
        )
        if not is_tracked:
            raise HTTPException(status_code=403, detail="Vendor not in your tracked list")

    signal_row = await pool.fetchrow(
        """
        SELECT * FROM b2b_churn_signals
        WHERE vendor_name ILIKE '%' || $1 || '%'
        ORDER BY avg_urgency_score DESC
        LIMIT 1
        """,
        vname,
    )

    counts = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total_reviews,
            COUNT(*) FILTER (WHERE enrichment_status = 'pending') AS pending_enrichment,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched
        FROM b2b_reviews
        WHERE vendor_name ILIKE '%' || $1 || '%'
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
        ORDER BY (enrichment->>'urgency_score')::numeric DESC
        LIMIT 5
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
        ORDER BY cnt DESC
        LIMIT 50
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
        "pending_enrichment": counts["pending_enrichment"] if counts else 0,
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


# ---------------------------------------------------------------------------
# GET /reports
# ---------------------------------------------------------------------------

VALID_REPORT_TYPES = (
    "weekly_churn_feed",
    "vendor_scorecard",
    "displacement_report",
    "category_overview",
    "exploratory_overview",
    "vendor_comparison",
    "account_comparison",
    "account_deep_dive",
    "vendor_retention",
    "challenger_intel",
)


@router.post("/reports/compare")
async def generate_comparison_report(
    body: VendorComparisonRequest,
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    primary_vendor = body.primary_vendor.strip()
    comparison_vendor = body.comparison_vendor.strip()
    if not primary_vendor or not comparison_vendor:
        raise HTTPException(status_code=400, detail="Both vendors are required")
    if primary_vendor.lower() == comparison_vendor.lower():
        raise HTTPException(status_code=400, detail="Choose two different vendors")
    if user:
        tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1::uuid AND vendor_name ILIKE $2 LIMIT 1",
            user.account_id,
            primary_vendor,
        )
        if not tracked:
            raise HTTPException(status_code=403, detail="Primary vendor must be in your tracked vendor list")
    from ..autonomous.tasks.b2b_churn_intelligence import generate_vendor_comparison_report

    report = await generate_vendor_comparison_report(
        pool,
        primary_vendor,
        comparison_vendor,
        window_days=body.window_days,
        persist=body.persist,
    )
    if not report:
        raise HTTPException(status_code=404, detail="Insufficient comparison data for the selected vendors")
    return report


@router.post("/reports/compare-companies")
async def generate_account_comparison_report(
    body: AccountComparisonRequest,
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    primary_company = body.primary_company.strip()
    comparison_company = body.comparison_company.strip()
    if not primary_company or not comparison_company:
        raise HTTPException(status_code=400, detail="Both companies are required")
    if primary_company.lower() == comparison_company.lower():
        raise HTTPException(status_code=400, detail="Choose two different companies")
    from ..autonomous.tasks.b2b_churn_intelligence import generate_company_comparison_report

    report = await generate_company_comparison_report(
        pool,
        primary_company,
        comparison_company,
        window_days=body.window_days,
        persist=body.persist,
        account_id=(user.account_id if user else None),
    )
    if not report:
        raise HTTPException(status_code=404, detail="Insufficient company comparison data for the selected accounts")
    return report


@router.post("/reports/company-deep-dive")
async def generate_account_deep_dive_report(
    body: AccountDeepDiveRequest,
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    company_name = body.company_name.strip()
    if not company_name:
        raise HTTPException(status_code=400, detail="Company name is required")
    from ..autonomous.tasks.b2b_churn_intelligence import generate_company_deep_dive_report

    report = await generate_company_deep_dive_report(
        pool,
        company_name,
        window_days=body.window_days,
        persist=body.persist,
        account_id=(user.account_id if user else None),
    )
    if not report:
        raise HTTPException(status_code=404, detail="No company-level churn data found for the selected account")
    return report


@router.get("/reports")
async def list_reports(
    report_type: Optional[str] = Query(None),
    vendor_filter: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    user: AuthUser | None = Depends(optional_auth),
):
    if report_type and report_type not in VALID_REPORT_TYPES:
        raise HTTPException(status_code=400, detail=f"report_type must be one of {VALID_REPORT_TYPES}")

    pool = _pool_or_503()
    conditions: list[str] = []
    params: list = []
    idx = 1

    if user:
        conditions.append(
            f"(vendor_filter IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid) OR account_id = ${idx}::uuid)"
        )
        params.append(user.account_id)
        idx += 1

    if report_type:
        conditions.append(f"report_type = ${idx}")
        params.append(report_type)
        idx += 1

    if vendor_filter:
        conditions.append(f"vendor_filter ILIKE '%' || ${idx} || '%'")
        params.append(vendor_filter)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    capped = min(limit, 50)
    params.append(capped)

    rows = await pool.fetch(
        f"""
        SELECT id, report_date, report_type, executive_summary,
             vendor_filter, category_filter, status, created_at
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
            "category_filter": r["category_filter"],
            "status": r["status"],
            "created_at": str(r["created_at"]) if r["created_at"] else None,
        }
        for r in rows
    ]

    return {"reports": reports, "count": len(reports)}


# ---------------------------------------------------------------------------
# GET /reports/{report_id}
# ---------------------------------------------------------------------------


@router.get("/reports/{report_id}")
async def get_report(report_id: str, user: AuthUser | None = Depends(optional_auth)):
    try:
        rid = _uuid.UUID(report_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid report_id (must be UUID)")

    pool = _pool_or_503()
    row = await pool.fetchrow("SELECT * FROM b2b_intelligence WHERE id = $1", rid)

    if not row:
        raise HTTPException(status_code=404, detail="Report not found")

    if user and row["account_id"] == user.account_id:
        pass
    elif user and row["vendor_filter"]:
        is_tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1::uuid AND vendor_name = $2",
            user.account_id, row["vendor_filter"],
        )
        if not is_tracked:
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


# ---------------------------------------------------------------------------
# GET /reviews
# ---------------------------------------------------------------------------


@router.get("/reviews")
async def search_reviews(
    vendor_name: Optional[str] = Query(None),
    pain_category: Optional[str] = Query(None),
    min_urgency: Optional[float] = Query(None, ge=0, le=10),
    company: Optional[str] = Query(None),
    has_churn_intent: Optional[bool] = Query(None),
    window_days: int = Query(30, ge=1, le=3650),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    conditions = [
        "enrichment_status = 'enriched'",
        "enriched_at > NOW() - make_interval(days => $1)",
    ]
    params: list = [window_days]
    idx = 2

    if user:
        conditions.append(f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)")
        params.append(user.account_id)
        idx += 1

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


# ---------------------------------------------------------------------------
# GET /reviews/{review_id}
# ---------------------------------------------------------------------------


@router.get("/reviews/{review_id}")
async def get_review(review_id: str, user: AuthUser | None = Depends(optional_auth)):
    try:
        rid = _uuid.UUID(review_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid review_id (must be UUID)")

    pool = _pool_or_503()
    row = await pool.fetchrow("SELECT * FROM b2b_reviews WHERE id = $1", rid)

    if not row:
        raise HTTPException(status_code=404, detail="Review not found")

    if user:
        is_tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1::uuid AND vendor_name ILIKE $2",
            user.account_id, row["vendor_name"],
        )
        if not is_tracked:
            raise HTTPException(status_code=403, detail="Vendor not in your tracked list")

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
        "imported_at": str(row["imported_at"]) if row["imported_at"] else None,
        "enrichment": _safe_json(row["enrichment"]),
        "enrichment_status": row["enrichment_status"],
        "enriched_at": str(row["enriched_at"]) if row["enriched_at"] else None,
    }


# ---------------------------------------------------------------------------
# GET /pipeline
# ---------------------------------------------------------------------------


@router.get("/pipeline")
async def get_pipeline_status(user: AuthUser | None = Depends(optional_auth)):
    pool = _pool_or_503()

    # Build vendor scope clause for authenticated users
    vendor_scope = ""
    scrape_scope = ""
    scope_params: list = []
    if user:
        vendor_scope = " WHERE vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = $1::uuid)"
        scrape_scope = " WHERE vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = $1::uuid)"
        scope_params = [user.account_id]

    status_rows = await pool.fetch(
        f"""
        SELECT enrichment_status, COUNT(*) AS cnt
        FROM b2b_reviews
        {vendor_scope}
        GROUP BY enrichment_status
        """,
        *scope_params,
    )
    enrichment_counts = {r["enrichment_status"]: r["cnt"] for r in status_rows}

    stats = await pool.fetchrow(
        f"""
        SELECT
            COUNT(*) FILTER (WHERE imported_at > NOW() - INTERVAL '24 hours') AS recent_imports_24h,
            MAX(enriched_at) AS last_enrichment_at
        FROM b2b_reviews
        {vendor_scope}
        """,
        *scope_params,
    )

    scrape_stats = await pool.fetchrow(
        f"""
        SELECT
            COUNT(*) FILTER (WHERE enabled) AS active_scrape_targets,
            MAX(last_scraped_at) AS last_scrape_at
        FROM b2b_scrape_targets
        {scrape_scope}
        """,
        *scope_params,
    )

    return {
        "enrichment_counts": enrichment_counts,
        "recent_imports_24h": stats["recent_imports_24h"] if stats else 0,
        "last_enrichment_at": str(stats["last_enrichment_at"]) if stats and stats["last_enrichment_at"] else None,
        "active_scrape_targets": scrape_stats["active_scrape_targets"] if scrape_stats else 0,
        "last_scrape_at": str(scrape_stats["last_scrape_at"]) if scrape_stats and scrape_stats["last_scrape_at"] else None,
    }


# ---------------------------------------------------------------------------
# GET /source-health
# ---------------------------------------------------------------------------

_SOURCE_HEALTH_SQL = """
WITH current_window AS (
    SELECT
        source,
        COUNT(*)                                            AS total_scrapes,
        COUNT(*) FILTER (WHERE status = 'success')          AS success_count,
        COUNT(*) FILTER (WHERE status = 'partial')          AS partial_count,
        COUNT(*) FILTER (WHERE status = 'failed')           AS failed_count,
        COUNT(*) FILTER (WHERE status = 'blocked')          AS blocked_count,
        AVG(reviews_found)                                  AS avg_reviews_found,
        AVG(reviews_inserted)                               AS avg_reviews_inserted,
        AVG(duration_ms)                                    AS avg_duration_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_duration_ms,
        MAX(started_at) FILTER (WHERE status = 'success')   AS last_success_at,
        MAX(started_at)                                     AS last_scrape_at
    FROM b2b_scrape_log
    WHERE started_at >= NOW() - make_interval(days => $1)
    {source_filter_current}
    GROUP BY source
),
prev_window AS (
    SELECT
        source,
        COUNT(*)                                            AS total_scrapes,
        COUNT(*) FILTER (WHERE status = 'success')          AS success_count,
        COUNT(*) FILTER (WHERE status = 'blocked')          AS blocked_count,
        AVG(reviews_found)                                  AS avg_reviews_found
    FROM b2b_scrape_log
    WHERE started_at >= NOW() - make_interval(days => $1 * 2)
      AND started_at <  NOW() - make_interval(days => $1)
    {source_filter_prev}
    GROUP BY source
),
target_counts AS (
    SELECT source, COUNT(*) FILTER (WHERE enabled) AS active_targets
    FROM b2b_scrape_targets
    {target_filter}
    GROUP BY source
)
SELECT
    c.source, c.total_scrapes, c.success_count, c.partial_count,
    c.failed_count, c.blocked_count, c.avg_reviews_found,
    c.avg_reviews_inserted, c.avg_duration_ms, c.p95_duration_ms,
    c.last_success_at, c.last_scrape_at,
    COALESCE(t.active_targets, 0)  AS active_targets,
    p.total_scrapes                AS prev_total_scrapes,
    p.success_count                AS prev_success_count,
    p.blocked_count                AS prev_blocked_count,
    p.avg_reviews_found            AS prev_avg_reviews_found
FROM current_window c
LEFT JOIN prev_window p USING (source)
LEFT JOIN target_counts t USING (source)
ORDER BY c.total_scrapes DESC
"""


def _build_source_health_query(source: str | None):
    """Return (sql, params) for the source-health CTE query."""
    if source:
        sql = _SOURCE_HEALTH_SQL.format(
            source_filter_current="AND source = $2",
            source_filter_prev="AND source = $2",
            target_filter="WHERE source = $2",
        )
        return sql, [source]
    sql = _SOURCE_HEALTH_SQL.format(
        source_filter_current="",
        source_filter_prev="",
        target_filter="",
    )
    return sql, []


def _row_to_source_dict(r) -> dict:
    """Convert a DB row to a source-health dict with computed rates."""
    total = r["total_scrapes"] or 1
    success_rate = round(r["success_count"] / total, 3)
    block_rate = round(r["blocked_count"] / total, 3)

    prev_total = r["prev_total_scrapes"] or 0
    prev_success_rate = round(r["prev_success_count"] / max(prev_total, 1), 3) if prev_total else None
    prev_block_rate = round(r["prev_blocked_count"] / max(prev_total, 1), 3) if prev_total else None
    prev_avg = _safe_float(r["prev_avg_reviews_found"])

    trend = {
        "prev_window_scrapes": prev_total,
        "prev_success_rate": prev_success_rate,
        "prev_block_rate": prev_block_rate,
        "prev_avg_reviews_found": prev_avg,
        "success_rate_delta": round(success_rate - prev_success_rate, 3) if prev_success_rate is not None else None,
        "block_rate_delta": round(block_rate - prev_block_rate, 3) if prev_block_rate is not None else None,
    }

    cap = get_capability(r["source"])
    capabilities = cap.to_dict() if cap else None

    return {
        "source": r["source"],
        "display_name": source_display_name(r["source"]),
        "total_scrapes": r["total_scrapes"],
        "success_count": r["success_count"],
        "partial_count": r["partial_count"],
        "failed_count": r["failed_count"],
        "blocked_count": r["blocked_count"],
        "success_rate": success_rate,
        "block_rate": block_rate,
        "avg_reviews_found": _safe_float(r["avg_reviews_found"]),
        "avg_reviews_inserted": _safe_float(r["avg_reviews_inserted"]),
        "avg_duration_ms": _safe_float(r["avg_duration_ms"]),
        "p95_duration_ms": _safe_float(r["p95_duration_ms"]),
        "last_success_at": str(r["last_success_at"]) if r["last_success_at"] else None,
        "last_scrape_at": str(r["last_scrape_at"]) if r["last_scrape_at"] else None,
        "active_targets": r["active_targets"],
        "capabilities": capabilities,
        "trend": trend,
    }


@router.get("/source-health")
async def get_source_health(
    window_days: int = Query(7, ge=1, le=30),
    source: Optional[str] = Query(None),
):
    pool = _pool_or_503()

    if source:
        source = source.strip().lower()
        if source not in ALL_SOURCES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source. Must be one of: {sorted(s.value for s in ALL_SOURCES)}",
            )

    sql, extra_params = _build_source_health_query(source)
    rows = await pool.fetch(sql, window_days, *extra_params)

    sources_list = [_row_to_source_dict(r) for r in rows]

    total_scrapes = sum(s["total_scrapes"] for s in sources_list)
    total_success = sum(s["success_count"] for s in sources_list)
    total_blocked = sum(s["blocked_count"] for s in sources_list)

    summary = {
        "total_sources": len(sources_list),
        "total_scrapes": total_scrapes,
        "overall_success_rate": round(total_success / max(total_scrapes, 1), 3),
        "overall_block_rate": round(total_blocked / max(total_scrapes, 1), 3),
        "worst_source": min(sources_list, key=lambda s: s["success_rate"])["source"] if sources_list else None,
        "best_source": max(sources_list, key=lambda s: s["success_rate"])["source"] if sources_list else None,
    }

    return {
        "window_days": window_days,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources_list,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _csv_response(rows: list[dict], filename: str) -> StreamingResponse:
    """Build a StreamingResponse from a list of dicts."""
    if not rows:
        buf = io.StringIO()
        buf.write("")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /displacement-edges
# ---------------------------------------------------------------------------


@router.get("/displacement-edges")
async def list_displacement_edges(
    from_vendor: Optional[str] = Query(None),
    to_vendor: Optional[str] = Query(None),
    min_strength: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    window_days: int = Query(90, ge=1, le=3650),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    strength_order = {"strong": 3, "moderate": 2, "emerging": 1}
    if min_strength and min_strength not in strength_order:
        raise HTTPException(400, f"Invalid min_strength: {min_strength}")

    conditions: list[str] = ["computed_date > NOW() - make_interval(days => $1)"]
    params: list = [window_days]
    idx = 2

    if user:
        conditions.append(
            f"(from_vendor IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)"
            f" OR to_vendor IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid))"
        )
        params.append(user.account_id)
        idx += 1

    if from_vendor:
        conditions.append(f"from_vendor ILIKE '%' || ${idx} || '%'")
        params.append(from_vendor)
        idx += 1

    if to_vendor:
        conditions.append(f"to_vendor ILIKE '%' || ${idx} || '%'")
        params.append(to_vendor)
        idx += 1

    if min_strength:
        min_val = strength_order[min_strength]
        allowed = [k for k, v in strength_order.items() if v >= min_val]
        conditions.append(f"signal_strength = ANY(${idx}::text[])")
        params.append(allowed)
        idx += 1

    if min_confidence is not None:
        conditions.append(f"confidence_score >= ${idx}")
        params.append(min_confidence)
        idx += 1

    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT id, from_vendor, to_vendor, mention_count,
               primary_driver, signal_strength, key_quote,
               source_distribution, confidence_score,
               computed_date, report_id, created_at
        FROM b2b_displacement_edges
        WHERE {where}
        ORDER BY confidence_score DESC, mention_count DESC
        LIMIT ${idx}
        """,
        *params,
        limit,
    )

    edges = []
    for r in rows:
        edges.append({
            "id": str(r["id"]),
            "from_vendor": r["from_vendor"],
            "to_vendor": r["to_vendor"],
            "mention_count": r["mention_count"],
            "primary_driver": r["primary_driver"],
            "signal_strength": r["signal_strength"],
            "key_quote": r["key_quote"],
            "source_distribution": _safe_json(r["source_distribution"]),
            "confidence_score": _safe_float(r["confidence_score"], 0),
            "computed_date": str(r["computed_date"]),
            "report_id": str(r["report_id"]) if r["report_id"] else None,
        })

    return {"edges": edges, "count": len(edges)}


# ---------------------------------------------------------------------------
# GET /company-signals
# ---------------------------------------------------------------------------


@router.get("/company-signals")
async def list_company_signals(
    vendor_name: Optional[str] = Query(None),
    company_name: Optional[str] = Query(None),
    min_urgency: float = Query(0, ge=0, le=10),
    decision_makers_only: bool = Query(False),
    window_days: int = Query(90, ge=1, le=3650),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    conditions: list[str] = ["last_seen_at > NOW() - make_interval(days => $1)"]
    params: list = [window_days]
    idx = 2

    if user:
        conditions.append(f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)")
        params.append(user.account_id)
        idx += 1

    if vendor_name:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1

    if company_name:
        conditions.append(f"company_name ILIKE '%' || ${idx} || '%'")
        params.append(company_name)
        idx += 1

    if min_urgency > 0:
        conditions.append(f"urgency_score >= ${idx}")
        params.append(min_urgency)
        idx += 1

    if decision_makers_only:
        conditions.append("decision_maker = true")

    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT id, company_name, vendor_name, urgency_score,
               pain_category, buyer_role, decision_maker,
               seat_count, contract_end, buying_stage,
               source, first_seen_at, last_seen_at
        FROM b2b_company_signals
        WHERE {where}
        ORDER BY urgency_score DESC
        LIMIT ${idx}
        """,
        *params,
        limit,
    )

    signals = []
    for r in rows:
        signals.append({
            "id": str(r["id"]),
            "company_name": r["company_name"],
            "vendor_name": r["vendor_name"],
            "urgency_score": _safe_float(r["urgency_score"], 0),
            "pain_category": r["pain_category"],
            "buyer_role": r["buyer_role"],
            "decision_maker": r["decision_maker"],
            "seat_count": r["seat_count"],
            "contract_end": r["contract_end"],
            "buying_stage": r["buying_stage"],
            "source": r["source"],
            "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
            "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
        })

    return {"signals": signals, "count": len(signals)}


# ---------------------------------------------------------------------------
# GET /export/signals  (CSV)
# ---------------------------------------------------------------------------


@router.get("/export/signals")
async def export_signals(
    vendor_name: Optional[str] = Query(None),
    min_urgency: float = Query(0, ge=0, le=10),
    category: Optional[str] = Query(None),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    conditions: list[str] = []
    params: list = []
    idx = 1

    if user:
        conditions.append(f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)")
        params.append(user.account_id)
        idx += 1

    if vendor_name:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
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

    rows = await pool.fetch(
        f"""
        SELECT vendor_name, product_category, total_reviews,
               churn_intent_count, avg_urgency_score, avg_rating_normalized,
               nps_proxy, price_complaint_rate, decision_maker_churn_rate,
               last_computed_at
        FROM b2b_churn_signals
        {where}
        ORDER BY avg_urgency_score DESC
        LIMIT {EXPORT_ROW_LIMIT}
        """,
        *params,
    )

    data = [
        {
            "vendor_name": r["vendor_name"],
            "product_category": r["product_category"] or "",
            "total_reviews": r["total_reviews"],
            "churn_intent_count": r["churn_intent_count"],
            "avg_urgency_score": _safe_float(r["avg_urgency_score"], ""),
            "avg_rating_normalized": _safe_float(r["avg_rating_normalized"], ""),
            "nps_proxy": _safe_float(r["nps_proxy"], ""),
            "price_complaint_rate": _safe_float(r["price_complaint_rate"], ""),
            "decision_maker_churn_rate": _safe_float(r["decision_maker_churn_rate"], ""),
            "last_computed_at": str(r["last_computed_at"]) if r["last_computed_at"] else "",
        }
        for r in rows
    ]

    return _csv_response(data, "churn_signals.csv")


# ---------------------------------------------------------------------------
# GET /export/reviews  (CSV)
# ---------------------------------------------------------------------------


@router.get("/export/reviews")
async def export_reviews(
    vendor_name: Optional[str] = Query(None),
    pain_category: Optional[str] = Query(None),
    min_urgency: Optional[float] = Query(None, ge=0, le=10),
    company: Optional[str] = Query(None),
    has_churn_intent: Optional[bool] = Query(None),
    window_days: int = Query(90, ge=1, le=3650),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    conditions = [
        "enrichment_status = 'enriched'",
        "enriched_at > NOW() - make_interval(days => $1)",
    ]
    params: list = [window_days]
    idx = 2

    if user:
        conditions.append(f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)")
        params.append(user.account_id)
        idx += 1

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

    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT vendor_name, product_category, reviewer_company,
               rating,
               (enrichment->>'urgency_score')::numeric AS urgency_score,
               enrichment->>'pain_category' AS pain_category,
               (enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
               (enrichment->'reviewer_context'->>'decision_maker')::boolean AS decision_maker,
               enriched_at
        FROM b2b_reviews
        WHERE {where}
        ORDER BY (enrichment->>'urgency_score')::numeric DESC
        LIMIT {EXPORT_ROW_LIMIT}
        """,
        *params,
    )

    data = [
        {
            "vendor_name": r["vendor_name"],
            "product_category": r["product_category"] or "",
            "reviewer_company": r["reviewer_company"] or "",
            "rating": _safe_float(r["rating"], ""),
            "urgency_score": _safe_float(r["urgency_score"], ""),
            "pain_category": r["pain_category"] or "",
            "intent_to_leave": r["intent_to_leave"] if r["intent_to_leave"] is not None else "",
            "decision_maker": r["decision_maker"] if r["decision_maker"] is not None else "",
            "enriched_at": str(r["enriched_at"]) if r["enriched_at"] else "",
        }
        for r in rows
    ]

    return _csv_response(data, "enriched_reviews.csv")


# ---------------------------------------------------------------------------
# GET /export/high-intent  (CSV)
# ---------------------------------------------------------------------------


@router.get("/export/high-intent")
async def export_high_intent(
    vendor_name: Optional[str] = Query(None),
    min_urgency: float = Query(7, ge=0, le=10),
    window_days: int = Query(90, ge=1, le=3650),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    conditions = [
        "enrichment_status = 'enriched'",
        "(enrichment->>'urgency_score')::numeric >= $1",
        "reviewer_company IS NOT NULL AND reviewer_company != ''",
        "enriched_at > NOW() - make_interval(days => $2)",
    ]
    params: list = [min_urgency, window_days]
    idx = 3

    if user:
        conditions.append(f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)")
        params.append(user.account_id)
        idx += 1

    if vendor_name:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1

    where = " AND ".join(conditions)

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
               enrichment->'use_case'->>'lock_in_level' AS lock_in_level,
               enrichment->'timeline'->>'contract_end' AS contract_end,
               enrichment->'buyer_authority'->>'buying_stage' AS buying_stage
        FROM b2b_reviews
        WHERE {where}
        ORDER BY (enrichment->>'urgency_score')::numeric DESC
        LIMIT {EXPORT_ROW_LIMIT}
        """,
        *params,
    )

    data = []
    for r in rows:
        alternatives = _safe_json(r["alternatives"])
        if isinstance(alternatives, list):
            alt_str = "; ".join(str(a) for a in alternatives)
        else:
            alt_str = str(alternatives) if alternatives else ""

        data.append({
            "company": r["reviewer_company"],
            "vendor": r["vendor_name"],
            "category": r["product_category"] or "",
            "role_level": r["role_level"] or "",
            "decision_maker": r["is_dm"] if r["is_dm"] is not None else "",
            "urgency": _safe_float(r["urgency"], ""),
            "pain": r["pain"] or "",
            "alternatives": alt_str,
            "contract_signal": r["value_signal"] or "",
            "seat_count": r["seat_count"] if r["seat_count"] is not None else "",
            "lock_in_level": r["lock_in_level"] or "",
            "contract_end": r["contract_end"] or "",
            "buying_stage": r["buying_stage"] or "",
        })

    return _csv_response(data, "high_intent_leads.csv")


# ---------------------------------------------------------------------------
# GET /export/source-health  (CSV)
# ---------------------------------------------------------------------------


@router.get("/export/source-health")
async def export_source_health(
    window_days: int = Query(7, ge=1, le=30),
    source: Optional[str] = Query(None),
):
    pool = _pool_or_503()

    if source:
        source = source.strip().lower()
        if source not in ALL_SOURCES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source. Must be one of: {sorted(s.value for s in ALL_SOURCES)}",
            )

    sql, extra_params = _build_source_health_query(source)
    rows = await pool.fetch(sql, window_days, *extra_params)

    data = []
    for r in rows:
        total = r["total_scrapes"] or 1
        prev_total = r["prev_total_scrapes"] or 0
        data.append({
            "source": r["source"],
            "display_name": source_display_name(r["source"]),
            "total_scrapes": r["total_scrapes"],
            "success_count": r["success_count"],
            "partial_count": r["partial_count"],
            "failed_count": r["failed_count"],
            "blocked_count": r["blocked_count"],
            "success_rate": round(r["success_count"] / total, 3),
            "block_rate": round(r["blocked_count"] / total, 3),
            "avg_reviews_found": _safe_float(r["avg_reviews_found"], ""),
            "avg_reviews_inserted": _safe_float(r["avg_reviews_inserted"], ""),
            "avg_duration_ms": _safe_float(r["avg_duration_ms"], ""),
            "p95_duration_ms": _safe_float(r["p95_duration_ms"], ""),
            "last_success_at": str(r["last_success_at"]) if r["last_success_at"] else "",
            "last_scrape_at": str(r["last_scrape_at"]) if r["last_scrape_at"] else "",
            "active_targets": r["active_targets"],
            "prev_window_scrapes": prev_total,
            "prev_success_rate": round(r["prev_success_count"] / max(prev_total, 1), 3) if prev_total else "",
            "prev_block_rate": round(r["prev_blocked_count"] / max(prev_total, 1), 3) if prev_total else "",
        })

    return _csv_response(data, "source_health.csv")
