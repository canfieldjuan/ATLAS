"""
Tenant-scoped B2B dashboard endpoints for P5 (Vendor Retention) and P6 (Challenger Lead Gen).

All endpoints require authentication and scope data to the tenant's tracked vendors.
"""

import json
import logging
import uuid as _uuid
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

from ..auth.dependencies import AuthUser, require_auth, require_b2b_plan
from ..autonomous.tasks.b2b_campaign_generation import (
    generate_campaigns as _generate_campaigns,
)
from ..config import settings
from ..services.scraping.target_provisioning import (
    provision_vendor_onboarding_targets,
)
from ..services.tracked_vendor_sources import (
    MANUAL_DIRECT_SOURCE_KEY,
    MANUAL_SOURCE_TYPE,
    purge_tracked_vendor_sources,
    upsert_tracked_vendor_source,
)
from ..services.vendor_registry import resolve_vendor_name
from ..storage.database import get_db_pool
from .b2b_dashboard import (
    _load_reasoning_views_for_vendors,
    _normalize_vendor_name,
    _overlay_reasoning_detail_from_view,
    _overlay_reasoning_summary_from_view,
)

logger = logging.getLogger("atlas.api.b2b_tenant")

router = APIRouter(prefix="/b2b/tenant", tags=["b2b-tenant"])
_LEGACY_ALIAS_REGISTRATION_DONE = False


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


def _is_admin_user(user: AuthUser | None) -> bool:
    if not user:
        return False
    if bool(getattr(user, "is_admin", False)):
        return True
    return str(getattr(user, "role", "")).lower() in {"owner", "admin"}


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class AddVendorRequest(BaseModel):
    vendor_name: str = Field(..., max_length=256)
    product_category: str | None = Field(None, max_length=200)
    scrape_target_slugs: dict[str, str] = Field(default_factory=dict)
    track_mode: str = Field(default="own", description="own (P5) | competitor (P6)")
    label: str = Field(default="", max_length=256)


class GenerateCampaignRequest(BaseModel):
    vendor_name: str = Field(..., max_length=256)
    company_filter: str = Field(default="", max_length=256)


class UpdateCampaignRequest(BaseModel):
    status: str = Field(..., description="approved | cancelled")


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

    canonical_vendor = await resolve_vendor_name(req.vendor_name)
    label = req.label.strip() if req.label else ""
    async with pool.transaction() as conn:
        existing_row = await conn.fetchrow(
            """
            SELECT id, vendor_name, track_mode, label, added_at
            FROM tracked_vendors
            WHERE account_id = $1 AND vendor_name = $2
            """,
            acct,
            canonical_vendor,
        )
        has_manual_source = await conn.fetchval(
            """
            SELECT 1
            FROM tracked_vendor_sources
            WHERE account_id = $1
              AND vendor_name = $2
              AND source_type = $3
            LIMIT 1
            """,
            acct,
            canonical_vendor,
            MANUAL_SOURCE_TYPE,
        )
        if existing_row and has_manual_source:
            raise HTTPException(status_code=409, detail="Vendor already tracked")
        if not existing_row:
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

        await upsert_tracked_vendor_source(
            conn,
            str(acct),
            canonical_vendor,
            source_type=MANUAL_SOURCE_TYPE,
            source_key=MANUAL_DIRECT_SOURCE_KEY,
            track_mode=req.track_mode,
        )
        if label:
            await conn.execute(
                """
                UPDATE tracked_vendors
                SET label = $3
                WHERE account_id = $1 AND vendor_name = $2
                """,
                acct,
                canonical_vendor,
                label,
            )
        row = await conn.fetchrow(
            """
            SELECT id, vendor_name, track_mode, label, added_at
            FROM tracked_vendors
            WHERE account_id = $1 AND vendor_name = $2
            """,
            acct,
            canonical_vendor,
        )

    try:
        scrape_provisioning = await provision_vendor_onboarding_targets(
            pool,
            canonical_vendor,
            product_category=req.product_category,
            source_slug_overrides=req.scrape_target_slugs,
            dry_run=False,
        )
    except Exception as exc:  # pragma: no cover - defensive operational guard
        logger.warning(
            "Auto scrape provisioning failed for tracked vendor %s: %s",
            canonical_vendor,
            exc,
        )
        scrape_provisioning = {
            "status": "error",
            "requested": 0,
            "applied": 0,
            "matched_vendors": [],
            "unmatched_vendors": [canonical_vendor.lower()],
            "actions": [],
        }

    return {
        "id": str(row["id"]),
        "vendor_name": row["vendor_name"],
        "track_mode": row["track_mode"],
        "label": row["label"],
        "added_at": str(row["added_at"]) if row["added_at"] else None,
        "scrape_provisioning": scrape_provisioning,
    }


@router.delete("/vendors/{vendor_name}")
async def remove_tracked_vendor(vendor_name: str, user: AuthUser = Depends(require_auth)):
    """Remove a tracked vendor."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)

    canonical_vendor = await resolve_vendor_name(vendor_name)
    was_tracked = await pool.fetchval(
        """
        SELECT 1
        FROM tracked_vendors
        WHERE account_id = $1 AND vendor_name = $2
        LIMIT 1
        """,
        acct,
        canonical_vendor,
    )
    cleanup = await purge_tracked_vendor_sources(
        pool,
        str(acct),
        canonical_vendor,
    )
    if not was_tracked and not cleanup["removed_sources"]:
        raise HTTPException(status_code=404, detail="Vendor not found in tracked list")

    return {
        "status": "ok",
        "vendor_name": canonical_vendor,
        "removed_sources": cleanup["removed_sources"],
        "still_tracked": cleanup["still_tracked"],
    }


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

def _vendor_scope_sql(param_idx: int, user: AuthUser | None = None) -> str:
    """SQL clause restricting to tracked vendors for the account."""
    if not settings.saas_auth.enabled or _is_admin_user(user):
        return "TRUE"
    return f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${param_idx})"


def _tenant_params(user: AuthUser) -> list:
    if not settings.saas_auth.enabled or _is_admin_user(user):
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
    t_params = _tenant_params(user)
    if t_params:
        vendor_count = await pool.fetchval(
            "SELECT COUNT(*) FROM tracked_vendors WHERE account_id = $1",
            t_params[0],
        )
    else:
        vendor_count = await pool.fetchval(
            "SELECT COUNT(DISTINCT vendor_name) FROM b2b_churn_signals",
        )

    # Signal summary for tracked vendors
    signal_stats = await pool.fetchrow(
        f"""
        SELECT COALESCE(AVG(avg_urgency_score), 0) AS avg_urgency,
               COALESCE(SUM(churn_intent_count), 0) AS total_churn_signals,
               COALESCE(SUM(total_reviews), 0) AS total_reviews
        FROM b2b_churn_signals
        WHERE {_vendor_scope_sql(1, user)}
        """,
        *t_params,
    )

    # Recent high-intent leads
    # DEPRECATED-ENRICHMENT-READ: urgency_score, pain_category, reviewer_context.industry
    # Migrate to: read_high_intent_companies() from _b2b_shared
    lead_rows = await pool.fetch(
        f"""
        SELECT reviewer_company, vendor_name,
               (enrichment->>'urgency_score')::numeric AS urgency,
               enrichment->>'pain_category' AS pain,
               reviewer_title, company_size_raw,
               COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND reviewer_company IS NOT NULL AND reviewer_company != ''
          AND (enrichment->>'urgency_score')::numeric >= 7
          AND {_vendor_scope_sql(1, user)}
        ORDER BY (enrichment->>'urgency_score')::numeric DESC
        LIMIT 5
        """,
        *t_params,
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
                "title": r["reviewer_title"],
                "company_size": r["company_size_raw"],
                "industry": r["industry"],
            }
            for r in lead_rows
        ],
    }


@router.get("/signals")
async def list_tenant_signals(
    vendor_name: Optional[str] = Query(None),
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
    scope = _vendor_scope_sql(idx, user)
    if scope != "TRUE":
        conditions.append(scope)
        params.extend(t_params)
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
    summary_params = list(params)
    capped = min(limit, 100)
    params.append(capped)

    rows = await pool.fetch(
        f"""
        SELECT sig.vendor_name, sig.product_category, sig.total_reviews,
               sig.churn_intent_count, sig.avg_urgency_score, sig.avg_rating_normalized,
               sig.nps_proxy, sig.price_complaint_rate, sig.decision_maker_churn_rate,
               snap.support_sentiment AS support_sentiment,
               snap.legacy_support_score AS legacy_support_score,
               snap.new_feature_velocity AS new_feature_velocity,
               snap.employee_growth_rate AS employee_growth_rate,
               sig.last_computed_at
        FROM b2b_churn_signals sig
        LEFT JOIN LATERAL (
            SELECT support_sentiment, legacy_support_score,
                   new_feature_velocity, employee_growth_rate
            FROM b2b_vendor_snapshots snap
            WHERE snap.vendor_name = sig.vendor_name
            ORDER BY snap.snapshot_date DESC
            LIMIT 1
        ) snap ON TRUE
        {where}
        ORDER BY avg_urgency_score DESC
        LIMIT ${idx}
        """,
        *params,
    )

    summary = await pool.fetchrow(
        f"""
        SELECT COUNT(DISTINCT vendor_name) AS total_vendors,
               COUNT(*) FILTER (WHERE avg_urgency_score >= 7) AS high_urgency_count,
               COALESCE(SUM(total_reviews), 0) AS total_signal_reviews
        FROM b2b_churn_signals
        {where}
        """,
        *summary_params,
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
            "support_sentiment": _safe_float(r["support_sentiment"]),
            "legacy_support_score": _safe_float(r["legacy_support_score"]),
            "new_feature_velocity": _safe_float(r["new_feature_velocity"]),
            "employee_growth_rate": _safe_float(r["employee_growth_rate"]),
            "last_computed_at": str(r["last_computed_at"]) if r["last_computed_at"] else None,
        }
        for r in rows
    ]

    return {
        "signals": signals,
        "count": len(signals),
        "total_vendors": int(summary["total_vendors"]) if summary and summary["total_vendors"] is not None else len(signals),
        "high_urgency_count": int(summary["high_urgency_count"]) if summary and summary["high_urgency_count"] is not None else 0,
        "total_signal_reviews": int(summary["total_signal_reviews"]) if summary and summary["total_signal_reviews"] is not None else 0,
    }


@router.get("/slow-burn-watchlist")
async def list_tenant_slow_burn_watchlist(
    vendor_name: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    user: AuthUser = Depends(require_auth),
):
    """Slow-burn watchlist for tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()

    conditions = [
        "("
        "snap.support_sentiment IS NOT NULL OR "
        "snap.legacy_support_score IS NOT NULL OR "
        "snap.new_feature_velocity IS NOT NULL OR "
        "snap.employee_growth_rate IS NOT NULL"
        ")",
    ]
    params: list = []
    idx = 1

    t_params = _tenant_params(user)
    scope = _vendor_scope_sql(idx, user)
    if scope != "TRUE":
        conditions.append(f"sig.{scope}")
        params.extend(t_params)
        idx += 1

    if vendor_name:
        conditions.append(f"sig.vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1

    if category:
        conditions.append(f"sig.product_category = ${idx}")
        params.append(category)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}"
    params.append(min(limit, 100))

    rows = await pool.fetch(
        f"""
        WITH ranked_signals AS (
            SELECT sig.vendor_name, sig.product_category, sig.total_reviews,
                   sig.churn_intent_count, sig.avg_urgency_score, sig.avg_rating_normalized,
                   sig.nps_proxy, sig.price_complaint_rate, sig.decision_maker_churn_rate,
                   snap.support_sentiment AS support_sentiment,
                   snap.legacy_support_score AS legacy_support_score,
                   snap.new_feature_velocity AS new_feature_velocity,
                   snap.employee_growth_rate AS employee_growth_rate,
                   sig.archetype, sig.archetype_confidence, sig.reasoning_mode,
                   sig.last_computed_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY sig.vendor_name
                       ORDER BY sig.avg_urgency_score DESC,
                                sig.total_reviews DESC,
                                sig.last_computed_at DESC NULLS LAST,
                                sig.product_category ASC NULLS LAST
                   ) AS vendor_row_rank
            FROM b2b_churn_signals sig
            LEFT JOIN LATERAL (
                SELECT support_sentiment, legacy_support_score,
                       new_feature_velocity, employee_growth_rate
                FROM b2b_vendor_snapshots snap
                WHERE snap.vendor_name = sig.vendor_name
                ORDER BY snap.snapshot_date DESC
                LIMIT 1
            ) snap ON TRUE
            {where}
        )
        SELECT vendor_name, product_category, total_reviews,
               churn_intent_count, avg_urgency_score, avg_rating_normalized,
               nps_proxy, price_complaint_rate, decision_maker_churn_rate,
               support_sentiment, legacy_support_score,
               new_feature_velocity, employee_growth_rate,
               archetype, archetype_confidence, reasoning_mode,
               last_computed_at
        FROM ranked_signals
        WHERE vendor_row_rank = 1
        ORDER BY employee_growth_rate DESC NULLS LAST,
                 support_sentiment ASC NULLS LAST,
                 legacy_support_score ASC NULLS LAST,
                 new_feature_velocity DESC NULLS LAST,
                 avg_urgency_score DESC,
                 last_computed_at DESC NULLS LAST
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
            "support_sentiment": _safe_float(r["support_sentiment"]),
            "legacy_support_score": _safe_float(r["legacy_support_score"]),
            "new_feature_velocity": _safe_float(r["new_feature_velocity"]),
            "employee_growth_rate": _safe_float(r["employee_growth_rate"]),
            "archetype": r["archetype"],
            "archetype_confidence": _safe_float(r["archetype_confidence"]),
            "reasoning_mode": r["reasoning_mode"],
            "last_computed_at": str(r["last_computed_at"]) if r["last_computed_at"] else None,
        }
        for r in rows
    ]
    reasoning_views = await _load_reasoning_views_for_vendors(
        pool,
        [signal.get("vendor_name", "") for signal in signals],
    )
    for signal in signals:
        view = reasoning_views.get(_normalize_vendor_name(signal.get("vendor_name")))
        if view is not None:
            _overlay_reasoning_summary_from_view(signal, view)

    return {
        "signals": signals,
        "count": len(rows),
    }


@router.get("/signals/{vendor_name}")
async def get_vendor_detail(vendor_name: str, user: AuthUser = Depends(require_auth)):
    """Full vendor detail: signal + pain + competitors + evidence."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    vname = vendor_name.strip()

    # Verify vendor is tracked by this account
    if settings.saas_auth.enabled and not _is_admin_user(user):
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
        SELECT sig.*, snap.support_sentiment AS support_sentiment,
               snap.legacy_support_score AS legacy_support_score,
               snap.new_feature_velocity AS new_feature_velocity,
               snap.employee_growth_rate AS employee_growth_rate
        FROM b2b_churn_signals sig
        LEFT JOIN LATERAL (
            SELECT support_sentiment, legacy_support_score,
                   new_feature_velocity, employee_growth_rate
            FROM b2b_vendor_snapshots snap
            WHERE snap.vendor_name = sig.vendor_name
            ORDER BY snap.snapshot_date DESC
            LIMIT 1
        ) snap ON TRUE
        WHERE sig.vendor_name ILIKE '%' || $1 || '%'
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

    # DEPRECATED-ENRICHMENT-READ: pain_category
    # Migrate to: read_vendor_evidence() from _b2b_shared
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

    # DEPRECATED-ENRICHMENT-READ: urgency_score, pain_category, reviewer_context.industry
    # Migrate to: read_high_intent_companies() from _b2b_shared
    hi_rows = await pool.fetch(
        """
        SELECT reviewer_company,
               (enrichment->>'urgency_score')::numeric AS urgency,
               enrichment->>'pain_category' AS pain,
               reviewer_title, company_size_raw,
               COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry
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
            "support_sentiment": _safe_float(signal_row["support_sentiment"]),
            "legacy_support_score": _safe_float(signal_row["legacy_support_score"]),
            "new_feature_velocity": _safe_float(signal_row["new_feature_velocity"]),
            "employee_growth_rate": _safe_float(signal_row["employee_growth_rate"]),
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
        reasoning_views = await _load_reasoning_views_for_vendors(
            pool,
            [signal_row["vendor_name"]],
        )
        view = reasoning_views.get(_normalize_vendor_name(signal_row["vendor_name"]))
        if view is not None:
            _overlay_reasoning_detail_from_view(
                profile["churn_signal"],
                view,
                requested_as_of=date.today(),
            )
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
            "title": r["reviewer_title"],
            "company_size": r["company_size_raw"],
            "industry": r["industry"],
        }
        for r in hi_rows
    ]

    profile["pain_distribution"] = [
        {"pain_category": r["pain"], "count": r["cnt"]}
        for r in pain_rows
    ]

    return profile


@router.get("/vendor-history")
async def get_tenant_vendor_history(
    vendor_name: str = Query(...),
    days: int = Query(90, ge=1, le=365),
    limit: int = Query(90, ge=1, le=365),
    user: AuthUser = Depends(require_auth),
):
    """Snapshot history for a tracked vendor."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    vname = vendor_name.strip()

    if settings.saas_auth.enabled and not _is_admin_user(user):
        acct = _uuid.UUID(user.account_id)
        tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1 AND vendor_name ILIKE $2 LIMIT 1",
            acct,
            vname,
        )
        if not tracked:
            raise HTTPException(status_code=403, detail="Vendor not in your tracked list")

    rows = await pool.fetch(
        """
        SELECT vendor_name, snapshot_date, total_reviews, churn_intent,
               churn_density, avg_urgency, positive_review_pct, recommend_ratio,
               support_sentiment, legacy_support_score,
               new_feature_velocity, employee_growth_rate,
               top_pain, top_competitor, pain_count, competitor_count,
               displacement_edge_count, high_intent_company_count
        FROM b2b_vendor_snapshots
        WHERE vendor_name ILIKE '%' || $1 || '%'
          AND snapshot_date >= CURRENT_DATE - $2::int
        ORDER BY snapshot_date DESC
        LIMIT $3
        """,
        vname, days, limit,
    )
    resolved = rows[0]["vendor_name"] if rows else vname
    snapshots = []
    for r in rows:
        snapshots.append({
            "snapshot_date": str(r["snapshot_date"]),
            "total_reviews": r["total_reviews"],
            "churn_intent": r["churn_intent"],
            "churn_density": _safe_float(r["churn_density"], 0),
            "avg_urgency": _safe_float(r["avg_urgency"], 0),
            "positive_review_pct": _safe_float(r["positive_review_pct"]),
            "recommend_ratio": _safe_float(r["recommend_ratio"]),
            "support_sentiment": _safe_float(r["support_sentiment"]),
            "legacy_support_score": _safe_float(r["legacy_support_score"]),
            "new_feature_velocity": _safe_float(r["new_feature_velocity"]),
            "employee_growth_rate": _safe_float(r["employee_growth_rate"]),
            "top_pain": r["top_pain"],
            "top_competitor": r["top_competitor"],
            "pain_count": r["pain_count"],
            "competitor_count": r["competitor_count"],
            "displacement_edge_count": r["displacement_edge_count"],
            "high_intent_company_count": r["high_intent_company_count"],
        })
    return {"vendor_name": resolved, "snapshots": snapshots, "count": len(snapshots)}


@router.get("/compare-vendor-periods")
async def compare_tenant_vendor_periods(
    vendor_name: str = Query(...),
    period_a_days_ago: int = Query(30, ge=0, le=365),
    period_b_days_ago: int = Query(0, ge=0, le=365),
    user: AuthUser = Depends(require_auth),
):
    """Compare two snapshot periods for a tracked vendor."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    vname = vendor_name.strip()

    if settings.saas_auth.enabled and not _is_admin_user(user):
        acct = _uuid.UUID(user.account_id)
        tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1 AND vendor_name ILIKE $2 LIMIT 1",
            acct,
            vname,
        )
        if not tracked:
            raise HTTPException(status_code=403, detail="Vendor not in your tracked list")

    async def _nearest_snapshot(target_days_ago: int):
        return await pool.fetchrow(
            """
            SELECT vendor_name, snapshot_date, total_reviews, churn_intent,
                   churn_density, avg_urgency, positive_review_pct, recommend_ratio,
                   support_sentiment, legacy_support_score,
                   new_feature_velocity, employee_growth_rate,
                   top_pain, top_competitor, pain_count, competitor_count,
                   displacement_edge_count, high_intent_company_count
            FROM b2b_vendor_snapshots
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND snapshot_date <= CURRENT_DATE - $2::int
            ORDER BY snapshot_date DESC
            LIMIT 1
            """,
            vname, target_days_ago,
        )

    snap_a = await _nearest_snapshot(period_a_days_ago)
    snap_b = await _nearest_snapshot(period_b_days_ago)

    if not snap_a and not snap_b:
        raise HTTPException(status_code=404, detail=f"No snapshots found for vendor matching '{vname}'")

    def _format(snap):
        if not snap:
            return None
        return {
            "snapshot_date": str(snap["snapshot_date"]),
            "total_reviews": snap["total_reviews"],
            "churn_intent": snap["churn_intent"],
            "churn_density": _safe_float(snap["churn_density"], 0),
            "avg_urgency": _safe_float(snap["avg_urgency"], 0),
            "positive_review_pct": _safe_float(snap["positive_review_pct"]),
            "recommend_ratio": _safe_float(snap["recommend_ratio"]),
            "support_sentiment": _safe_float(snap["support_sentiment"]),
            "legacy_support_score": _safe_float(snap["legacy_support_score"]),
            "new_feature_velocity": _safe_float(snap["new_feature_velocity"]),
            "employee_growth_rate": _safe_float(snap["employee_growth_rate"]),
            "top_pain": snap["top_pain"],
            "top_competitor": snap["top_competitor"],
            "pain_count": snap["pain_count"],
            "competitor_count": snap["competitor_count"],
            "displacement_edge_count": snap["displacement_edge_count"],
            "high_intent_company_count": snap["high_intent_company_count"],
        }

    a_fmt = _format(snap_a)
    b_fmt = _format(snap_b)
    deltas = {}
    if a_fmt and b_fmt:
        for key in (
            "churn_density", "avg_urgency", "recommend_ratio", "total_reviews",
            "churn_intent", "pain_count", "competitor_count",
            "displacement_edge_count", "high_intent_company_count",
            "support_sentiment", "legacy_support_score",
            "new_feature_velocity", "employee_growth_rate",
        ):
            a_val = a_fmt.get(key)
            b_val = b_fmt.get(key)
            if a_val is not None and b_val is not None:
                deltas[key] = round(b_val - a_val, 2)

    resolved = (snap_a or snap_b)["vendor_name"]
    return {"vendor_name": resolved, "period_a": a_fmt, "period_b": b_fmt, "deltas": deltas}


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

    scope = _vendor_scope_sql(idx, user)
    if scope != "TRUE":
        conditions.append(scope)
        params.extend(t_params)
        idx += 1

    conditions.append("enrichment_status = 'enriched'")
    conditions.append(f"enriched_at > NOW() - make_interval(days => ${idx})")
    params.append(window_days)
    idx += 1
    # DEPRECATED-ENRICHMENT-READ: pain_category
    # Migrate to: read_vendor_evidence() from _b2b_shared
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

    scope = _vendor_scope_sql(idx, user)
    if scope != "TRUE":
        conditions.append(scope)
        params.extend(t_params)
        idx += 1

    # DEPRECATED-ENRICHMENT-READ: competitors_mentioned, churn_signals.intent_to_leave
    # Migrate to: read_vendor_evidence() from _b2b_shared
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


@router.get("/pipeline")
async def get_tenant_pipeline_status(user: AuthUser = Depends(require_auth)):
    """Pipeline status for tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()

    t_params = _tenant_params(user)

    review_scope = ""
    scrape_scope = ""
    scope_params: list = []
    if t_params:
        review_scope = "WHERE vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = $1)"
        scrape_scope = "WHERE vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = $1)"
        scope_params = t_params

    status_rows = await pool.fetch(
        f"""
        SELECT enrichment_status, COUNT(*) AS cnt
        FROM b2b_reviews
        {review_scope}
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
        {review_scope}
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
    company_expr = (
        "COALESCE(NULLIF(BTRIM(reviewer_company), ''), "
        "NULLIF(BTRIM(reviewer_company_norm), ''))"
    )
    conditions = [
        "enrichment_status = 'enriched'",
        f"{company_expr} IS NOT NULL",
    ]
    params: list = []

    scope = _vendor_scope_sql(idx, user)
    if scope != "TRUE":
        conditions.append(scope)
        params.extend(t_params)
        idx += 1

    # DEPRECATED-ENRICHMENT-READ: urgency_score, reviewer_context.role_level, reviewer_context.decision_maker, pain_category, competitors_mentioned, contract_context.contract_value_signal, budget_signals.seat_count, use_case.lock_in_level, timeline.contract_end, buyer_authority.buying_stage
    # Migrate to: read_high_intent_companies() from _b2b_shared
    conditions.append(f"(enrichment->>'urgency_score')::numeric >= ${idx}")
    params.append(min_urgency)
    idx += 1

    conditions.append(
        f"COALESCE(reviewed_at, imported_at, enriched_at) > NOW() - make_interval(days => ${idx})"
    )
    params.append(window_days)
    idx += 1

    where = " AND ".join(conditions)
    capped = min(limit, 100)
    params.append(capped)

    # DEPRECATED-ENRICHMENT-READ: reviewer_context.role_level, reviewer_context.decision_maker, urgency_score, pain_category, competitors_mentioned, contract_context.contract_value_signal, budget_signals.seat_count, use_case.lock_in_level, timeline.contract_end, buyer_authority.buying_stage
    # Migrate to: read_high_intent_companies() from _b2b_shared
    rows = await pool.fetch(
        f"""
        SELECT {company_expr} AS company_name, vendor_name, product_category,
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

    companies = [
        {
            "company": r["company_name"],
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
        }
        for r in rows
    ]

    return {"leads": companies, "count": len(companies)}


@router.get("/high-intent")
async def list_tenant_high_intent(
    vendor_name: Optional[str] = Query(None),
    min_urgency: float = Query(7, ge=0, le=10),
    window_days: int = Query(30, ge=1, le=3650),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser = Depends(require_auth),
):
    """Compatibility alias for legacy high-intent payload shape."""
    _require_b2b_product(user)
    lead_payload = await list_leads(
        min_urgency=min_urgency,
        window_days=window_days,
        limit=limit,
        user=user,
    )
    companies = lead_payload["leads"]
    if vendor_name:
        needle = vendor_name.lower().strip()
        companies = [c for c in companies if needle in str(c.get("vendor") or "").lower()]
    return {"companies": companies, "count": len(companies)}


@router.get("/leads/{company}")
async def get_lead_detail(company: str, user: AuthUser = Depends(require_auth)):
    """Company drill-down: all reviews, signals, buying stage."""
    _require_b2b_product(user)
    pool = _pool_or_503()

    t_params = _tenant_params(user)
    idx = 1
    company_expr = (
        "COALESCE(NULLIF(BTRIM(reviewer_company), ''), "
        "NULLIF(BTRIM(reviewer_company_norm), ''))"
    )
    conditions = ["enrichment_status = 'enriched'"]
    params: list = []

    scope = _vendor_scope_sql(idx, user)
    if scope != "TRUE":
        conditions.append(scope)
        params.extend(t_params)
        idx += 1

    # DEPRECATED-ENRICHMENT-READ: urgency_score, pain_category, churn_signals.intent_to_leave, reviewer_context.decision_maker, reviewer_context.role_level, buyer_authority.buying_stage, competitors_mentioned, timeline.contract_end
    # Migrate to: read_review_details() from _b2b_shared
    conditions.append(f"LOWER({company_expr}) = LOWER(${idx})")
    params.append(company.strip())
    idx += 1

    where = " AND ".join(conditions)

    # DEPRECATED-ENRICHMENT-READ: urgency_score, pain_category, churn_signals.intent_to_leave, reviewer_context.decision_maker, reviewer_context.role_level, buyer_authority.buying_stage, competitors_mentioned, timeline.contract_end
    # Migrate to: read_review_details() from _b2b_shared
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

@router.post("/reports/compare")
async def generate_tenant_comparison_report(
    body: VendorComparisonRequest,
    user: AuthUser = Depends(require_auth),
):
    """Generate a vendor comparison report from tenant-scoped dashboard."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    primary_vendor = body.primary_vendor.strip()
    comparison_vendor = body.comparison_vendor.strip()
    if not primary_vendor or not comparison_vendor:
        raise HTTPException(status_code=400, detail="Both vendors are required")
    if primary_vendor.lower() == comparison_vendor.lower():
        raise HTTPException(status_code=400, detail="Choose two different vendors")

    if settings.saas_auth.enabled and not _is_admin_user(user):
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
async def generate_tenant_account_comparison_report(
    body: AccountComparisonRequest,
    user: AuthUser = Depends(require_auth),
):
    """Generate a company comparison report from tenant-scoped dashboard."""
    _require_b2b_product(user)
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
    )
    if not report:
        raise HTTPException(status_code=404, detail="Insufficient comparison data for the selected companies")
    return report


@router.post("/reports/company-deep-dive")
async def generate_tenant_account_deep_dive_report(
    body: AccountDeepDiveRequest,
    user: AuthUser = Depends(require_auth),
):
    """Generate a company deep-dive report from tenant-scoped dashboard."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    company_name = body.company_name.strip()
    if not company_name:
        raise HTTPException(status_code=400, detail="company_name is required")

    from ..autonomous.tasks.b2b_churn_intelligence import generate_company_deep_dive_report

    report = await generate_company_deep_dive_report(
        pool,
        company_name,
        window_days=body.window_days,
        persist=body.persist,
    )
    if not report:
        raise HTTPException(status_code=404, detail="Insufficient data for this company")
    return report


@router.get("/reports")
async def list_tenant_reports(
    report_type: Optional[str] = Query(None),
    vendor_filter: Optional[str] = Query(None),
    include_stale: bool = Query(False),
    limit: int = Query(10, ge=1, le=200),
    user: AuthUser = Depends(require_auth),
):
    """Reports scoped to tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()

    t_params = _tenant_params(user)
    idx = 1
    conditions: list[str] = []
    params: list = []

    # Scope by tracked vendors while allowing global and account-scoped rows.
    scope = _vendor_scope_sql(idx, user)
    if scope != "TRUE":
        conditions.append(
            f"(vendor_filter IS NULL OR vendor_filter = '' "
            f"OR LOWER(vendor_filter) IN (SELECT LOWER(vendor_name) FROM tracked_vendors WHERE account_id = ${idx}) "
            f"OR account_id = ${idx})"
        )
        params.extend(t_params)
        idx += 1

    if not include_stale:
        conditions.append(
            "COALESCE((intelligence_data->>'data_stale')::boolean, false) = false"
        )

    _REPORT_TYPE_ALIASES = {"challenger_intel": "challenger_brief"}
    if report_type:
        resolved_type = _REPORT_TYPE_ALIASES.get(report_type, report_type)
        conditions.append(f"report_type = ${idx}")
        params.append(resolved_type)
        idx += 1

    if vendor_filter:
        conditions.append(f"vendor_filter ILIKE '%' || ${idx} || '%'")
        params.append(vendor_filter)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    capped = min(limit, 200)
    params.append(capped)

    rows = await pool.fetch(
        f"""
        SELECT id, report_date, report_type, executive_summary,
               vendor_filter, category_filter, status, created_at,
               latest_failure_step, latest_error_code, latest_error_summary,
               blocker_count, warning_count,
               (
                 SELECT COUNT(*)
                 FROM pipeline_visibility_reviews r
                 JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
                 WHERE r.status = 'open'
                   AND e.entity_type = 'churn_report'
                   AND e.entity_id = b2b_intelligence.id::text
               ) AS unresolved_issue_count,
               CASE
                 WHEN report_type = 'battle_card'
                 THEN COALESCE(intelligence_data->>'quality_status', intelligence_data->'battle_card_quality'->>'status')
                 ELSE NULL
               END AS quality_status,
               CASE
                 WHEN report_type = 'battle_card'
                 THEN COALESCE((intelligence_data->'battle_card_quality'->>'score')::int, NULL)
                 ELSE NULL
               END AS quality_score
        FROM b2b_intelligence
        {where}
        ORDER BY report_date DESC NULLS LAST, created_at DESC NULLS LAST, id DESC
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
            "latest_failure_step": r["latest_failure_step"],
            "latest_error_code": r["latest_error_code"],
            "latest_error_summary": r["latest_error_summary"],
            "blocker_count": r["blocker_count"] or 0,
            "warning_count": r["warning_count"] or 0,
            "unresolved_issue_count": r["unresolved_issue_count"] or 0,
            "quality_status": r["quality_status"],
            "quality_score": r["quality_score"],
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
    if settings.saas_auth.enabled and not _is_admin_user(user) and row["vendor_filter"]:
        acct = _uuid.UUID(user.account_id)
        tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1 AND vendor_name = $2",
            acct,
            row["vendor_filter"],
        )
        if not tracked:
            raise HTTPException(status_code=403, detail="Report vendor not in your tracked list")

    intelligence_data = _safe_json(row["intelligence_data"])
    quality_status = None
    quality_score = None
    unresolved_issue_count = await pool.fetchval(
        """
        SELECT COUNT(*)
        FROM pipeline_visibility_reviews r
        JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
        WHERE r.status = 'open'
          AND e.entity_type = 'churn_report'
          AND e.entity_id = $1
        """,
        str(row["id"]),
    )
    if isinstance(intelligence_data, dict):
        quality = _safe_json(intelligence_data.get("battle_card_quality"))
        quality_status = intelligence_data.get("quality_status")
        if not quality_status and isinstance(quality, dict):
            quality_status = quality.get("status")
        if isinstance(quality, dict):
            quality_score = quality.get("score")

    return {
        "id": str(row["id"]),
        "report_date": str(row["report_date"]) if row["report_date"] else None,
        "report_type": row["report_type"],
        "vendor_filter": row["vendor_filter"],
        "category_filter": row["category_filter"],
        "executive_summary": row["executive_summary"],
        "intelligence_data": intelligence_data,
        "data_density": _safe_json(row["data_density"]),
        "status": row["status"],
        "latest_failure_step": row["latest_failure_step"],
        "latest_error_code": row["latest_error_code"],
        "latest_error_summary": row["latest_error_summary"],
        "blocker_count": row["blocker_count"] or 0,
        "warning_count": row["warning_count"] or 0,
        "unresolved_issue_count": unresolved_issue_count or 0,
        "quality_status": quality_status,
        "quality_score": quality_score,
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
    company_expr = (
        "COALESCE(NULLIF(BTRIM(reviewer_company), ''), "
        "NULLIF(BTRIM(reviewer_company_norm), ''))"
    )
    conditions = ["enrichment_status = 'enriched'"]
    params: list = []

    scope = _vendor_scope_sql(idx, user)
    if scope != "TRUE":
        conditions.append(scope)
        params.extend(t_params)
        idx += 1

    conditions.append(
        f"COALESCE(reviewed_at, imported_at, enriched_at) > NOW() - make_interval(days => ${idx})"
    )
    params.append(window_days)
    idx += 1

    # DEPRECATED-ENRICHMENT-READ: pain_category, urgency_score, churn_signals.intent_to_leave, reviewer_context.decision_maker, reviewer_context.industry
    # Migrate to: read_review_details() from _b2b_shared
    if pain_category:
        conditions.append(f"enrichment->>'pain_category' = ${idx}")
        params.append(pain_category)
        idx += 1

    if min_urgency is not None:
        conditions.append(f"(enrichment->>'urgency_score')::numeric >= ${idx}")
        params.append(min_urgency)
        idx += 1

    if company:
        conditions.append(f"{company_expr} ILIKE '%' || ${idx} || '%'")
        params.append(company)
        idx += 1

    # DEPRECATED-ENRICHMENT-READ: churn_signals.intent_to_leave
    # Migrate to: read_review_details() from _b2b_shared
    if has_churn_intent is not None:
        conditions.append(
            f"(enrichment->'churn_signals'->>'intent_to_leave')::boolean = ${idx}"
        )
        params.append(has_churn_intent)
        idx += 1

    where = " AND ".join(conditions)
    capped = min(limit, 100)
    params.append(capped)

    # DEPRECATED-ENRICHMENT-READ: urgency_score, pain_category, churn_signals.intent_to_leave, reviewer_context.decision_maker, reviewer_context.industry
    # Migrate to: read_review_details() from _b2b_shared
    rows = await pool.fetch(
        f"""
        SELECT id, vendor_name, product_category, {company_expr} AS reviewer_company,
               rating,
               (enrichment->>'urgency_score')::numeric AS urgency_score,
               enrichment->>'pain_category' AS pain_category,
               (enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
               (enrichment->'reviewer_context'->>'decision_maker')::boolean AS decision_maker,
               enriched_at, reviewer_title, company_size_raw,
               COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry
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
            "reviewer_title": r["reviewer_title"],
            "company_size": r["company_size_raw"],
            "industry": r["industry"],
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
    if settings.saas_auth.enabled and not _is_admin_user(user):
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
# CSV exports (tenant aliases)
# ---------------------------------------------------------------------------

@router.get("/export/signals")
async def export_tenant_signals(
    vendor_name: Optional[str] = Query(None),
    min_urgency: float = Query(0, ge=0, le=10),
    category: Optional[str] = Query(None),
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    from .b2b_dashboard import export_signals
    return await export_signals(
        vendor_name=vendor_name,
        min_urgency=min_urgency,
        category=category,
        user=user,
    )


@router.get("/export/reviews")
async def export_tenant_reviews(
    vendor_name: Optional[str] = Query(None),
    pain_category: Optional[str] = Query(None),
    min_urgency: Optional[float] = Query(None, ge=0, le=10),
    company: Optional[str] = Query(None),
    has_churn_intent: Optional[bool] = Query(None),
    window_days: int = Query(90, ge=1, le=3650),
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    from .b2b_dashboard import export_reviews
    return await export_reviews(
        vendor_name=vendor_name,
        pain_category=pain_category,
        min_urgency=min_urgency,
        company=company,
        has_churn_intent=has_churn_intent,
        window_days=window_days,
        user=user,
    )


@router.get("/export/high-intent")
async def export_tenant_high_intent(
    vendor_name: Optional[str] = Query(None),
    min_urgency: float = Query(7, ge=0, le=10),
    window_days: int = Query(90, ge=1, le=3650),
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    from .b2b_dashboard import export_high_intent
    return await export_high_intent(
        vendor_name=vendor_name,
        min_urgency=min_urgency,
        window_days=window_days,
        user=user,
    )


@router.get("/export/source-health")
async def export_tenant_source_health(
    window_days: int = Query(7, ge=1, le=30),
    source: Optional[str] = Query(None),
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    from .b2b_dashboard import export_source_health
    return await export_source_health(window_days=window_days, source=source)


# ---------------------------------------------------------------------------
# Campaigns (3 endpoints -- b2b_growth+ only)
# ---------------------------------------------------------------------------

@router.get("/campaigns")
async def list_tenant_campaigns(
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser = require_b2b_plan("b2b_growth"),
):
    """Campaigns scoped to tracked vendors. Requires b2b_growth+ plan."""
    pool = _pool_or_503()

    t_params = _tenant_params(user)
    idx = 1
    conditions: list[str] = []
    params: list = []

    # Scope campaigns by vendor_name matching tracked vendors
    scope = _vendor_scope_sql(idx, user)
    if scope != "TRUE":
        # b2b_campaigns uses company_name, but we scope via metadata.vendor_name
        conditions.append(
            f"bc.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx})"
        )
        params.extend(t_params)
        idx += 1

    if status:
        conditions.append(f"bc.status = ${idx}")
        params.append(status)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    capped = min(limit, 100)
    params.append(capped)

    rows = await pool.fetch(
        f"""
        SELECT bc.id, bc.company_name, bc.vendor_name, bc.channel, bc.subject,
               bc.status, bc.approved_at, bc.sent_at, bc.created_at,
               bc.recipient_email, cs.company_context
        FROM b2b_campaigns bc
        LEFT JOIN campaign_sequences cs ON cs.id = bc.sequence_id
        {where}
        ORDER BY bc.created_at DESC
        LIMIT ${idx}
        """,
        *params,
    )

    campaigns = []
    for r in rows:
        cc = r.get("company_context")
        persona = None
        if cc and isinstance(cc, dict):
            persona = cc.get("target_persona")
        campaigns.append({
            "id": str(r["id"]),
            "company_name": r["company_name"],
            "vendor_name": r["vendor_name"],
            "channel": r["channel"],
            "subject": r["subject"],
            "status": r["status"],
            "approved_at": str(r["approved_at"]) if r["approved_at"] else None,
            "sent_at": str(r["sent_at"]) if r["sent_at"] else None,
            "created_at": str(r["created_at"]) if r["created_at"] else None,
            "recipient_email": r["recipient_email"],
            "target_persona": persona,
        })

    return {"campaigns": campaigns, "count": len(campaigns)}


@router.post("/campaigns/generate")
async def generate_campaigns(
    req: GenerateCampaignRequest,
    user: AuthUser = require_b2b_plan("b2b_growth"),
):
    """Generate campaign drafts for a tracked vendor's high-intent leads."""
    pool = _pool_or_503()
    vname = req.vendor_name.strip()

    # Verify vendor is tracked
    if settings.saas_auth.enabled and not _is_admin_user(user):
        acct = _uuid.UUID(user.account_id)
        tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1 AND vendor_name = $2",
            acct,
            vname,
        )
        if not tracked:
            raise HTTPException(status_code=403, detail="Vendor not in your tracked list")

    # Use the full LLM campaign generation pipeline
    cfg = settings.b2b_campaign

    try:
        result = await _generate_campaigns(
            pool,
            vendor_filter=vname,
            company_filter=req.company_filter or None,
            target_mode=cfg.target_mode,
            min_score=cfg.min_opportunity_score,
            limit=cfg.max_campaigns_per_run,
        )
    except Exception as exc:
        logger.error("Campaign generation failed for vendor %s: %s", vname, exc)
        raise HTTPException(status_code=500, detail="Campaign generation failed")

    # Surface pipeline errors as 5xx instead of silent 200
    if result.get("error"):
        raise HTTPException(status_code=503, detail=result["error"])

    return {"campaigns_created": result.get("generated", 0), **result}


@router.patch("/campaigns/{campaign_id}")
async def update_campaign(
    campaign_id: str,
    req: UpdateCampaignRequest,
    user: AuthUser = require_b2b_plan("b2b_growth"),
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

    if settings.saas_auth.enabled and not _is_admin_user(user):
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


def _register_legacy_dashboard_aliases() -> None:
    """Register tenant aliases for any legacy dashboard route not yet migrated."""
    global _LEGACY_ALIAS_REGISTRATION_DONE
    if _LEGACY_ALIAS_REGISTRATION_DONE:
        return

    from .b2b_dashboard import router as legacy_router

    existing_methods: set[tuple[str, str]] = set()
    for route in router.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in (route.methods or set()):
            if method in {"HEAD", "OPTIONS"}:
                continue
            existing_methods.add((route.path, method))

    migrated_count = 0
    already_present = 0
    legacy_prefix = "/b2b/dashboard"
    tenant_prefix = "/b2b/tenant"

    for legacy_route in legacy_router.routes:
        if not isinstance(legacy_route, APIRoute):
            continue
        if not legacy_route.path.startswith(legacy_prefix):
            continue

        route_suffix = legacy_route.path[len(legacy_prefix):] or "/"
        tenant_path = tenant_prefix + route_suffix
        methods = sorted(
            m for m in (legacy_route.methods or set())
            if m not in {"HEAD", "OPTIONS"}
        )
        if not methods:
            continue
        if all((tenant_path, method) in existing_methods for method in methods):
            already_present += 1
            continue

        router.add_api_route(
            route_suffix,
            legacy_route.endpoint,
            methods=methods,
            response_model=legacy_route.response_model,
            status_code=legacy_route.status_code,
            tags=legacy_route.tags,
            dependencies=legacy_route.dependencies,
            summary=legacy_route.summary,
            description=legacy_route.description,
            response_description=legacy_route.response_description,
            responses=legacy_route.responses,
            deprecated=legacy_route.deprecated,
            include_in_schema=legacy_route.include_in_schema,
            name=legacy_route.name,
            callbacks=legacy_route.callbacks,
            openapi_extra=legacy_route.openapi_extra,
        )
        for method in methods:
            existing_methods.add((tenant_path, method))
        migrated_count += 1

    _LEGACY_ALIAS_REGISTRATION_DONE = True
    logger.info(
        "Tenant alias registration complete: %s migrated, %s already present",
        migrated_count,
        already_present,
    )


_register_legacy_dashboard_aliases()
