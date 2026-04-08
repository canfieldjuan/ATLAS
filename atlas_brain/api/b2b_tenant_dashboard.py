"""
Tenant-scoped B2B dashboard endpoints for P5 (Vendor Retention) and P6 (Challenger Lead Gen).

All endpoints require authentication and scope data to the tenant's tracked vendors.
"""

import asyncio
import json
import logging
import uuid as _uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.routing import APIRoute
from pydantic import BaseModel, EmailStr, Field

from ..auth.dependencies import AuthUser, require_auth, require_b2b_plan
from ..autonomous.tasks.b2b_campaign_generation import (
    generate_campaigns as _generate_campaigns,
)
from ..autonomous.scheduler import get_task_scheduler
from ..config import settings
from ..services.scraping.target_provisioning import (
    provision_vendor_onboarding_targets,
)
from ..services.b2b_competitive_sets import (
    build_competitive_set_plan,
    estimate_competitive_set_plan,
    load_vendor_category_map,
    plan_to_synthesis_metadata,
)
from ..services.tracked_vendor_sources import (
    MANUAL_DIRECT_SOURCE_KEY,
    MANUAL_SOURCE_TYPE,
    purge_tracked_vendor_sources,
    upsert_tracked_vendor_source,
)
from ..services.vendor_registry import resolve_vendor_name
from ..storage.repositories.competitive_set import get_competitive_set_repo
from ..storage.repositories.scheduled_task import get_scheduled_task_repo
from ..storage.database import get_db_pool
from .b2b_dashboard import (
    _list_accounts_in_motion_from_report,
    _load_reasoning_views_for_vendors,
    _normalize_vendor_name,
    _overlay_reasoning_detail_from_view,
    _overlay_reasoning_summary_from_view,
)

logger = logging.getLogger("atlas.api.b2b_tenant")

router = APIRouter(prefix="/b2b/tenant", tags=["b2b-tenant"])
_LEGACY_ALIAS_REGISTRATION_DONE = False
_REPORT_SUBSCRIPTION_SCOPE_TYPES = {"library", "report"}
_REPORT_SUBSCRIPTION_FREQUENCIES = {
    "weekly": 7,
    "monthly": 30,
    "quarterly": 90,
}


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


def _coerce_uuid(value: str | None) -> _uuid.UUID | None:
    if not value:
        return None
    try:
        return _uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


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


def _normalize_report_subscription_scope(scope_type: str, scope_key: str) -> tuple[str, str]:
    normalized_scope_type = scope_type.strip().lower()
    if normalized_scope_type not in _REPORT_SUBSCRIPTION_SCOPE_TYPES:
        raise HTTPException(status_code=400, detail="scope_type must be library or report")

    normalized_scope_key = scope_key.strip()
    if not normalized_scope_key:
        raise HTTPException(status_code=400, detail="scope_key is required")

    if normalized_scope_type == "library" and normalized_scope_key != "library":
        raise HTTPException(status_code=400, detail="Library scope_key must be 'library'")

    return normalized_scope_type, normalized_scope_key


def _normalize_report_subscription_recipients(values: list[EmailStr]) -> list[str]:
    seen: set[str] = set()
    recipients: list[str] = []
    for raw_value in values:
        value = str(raw_value).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        recipients.append(value)
    return recipients


def _next_report_subscription_delivery_at(
    delivery_frequency: str,
    enabled: bool,
) -> datetime | None:
    if not enabled:
        return None
    days = _REPORT_SUBSCRIPTION_FREQUENCIES[delivery_frequency]
    return datetime.now(timezone.utc) + timedelta(days=days)


def _resolve_report_subscription_next_delivery_at(
    *,
    existing_row,
    delivery_frequency: str,
    enabled: bool,
) -> datetime | None:
    if not enabled:
        return None
    if not existing_row:
        return _next_report_subscription_delivery_at(delivery_frequency, enabled)

    was_enabled = bool(existing_row.get("enabled"))
    previous_frequency = str(existing_row.get("delivery_frequency") or "").strip().lower()
    previous_next_delivery_at = existing_row.get("next_delivery_at")

    if not was_enabled:
        return _next_report_subscription_delivery_at(delivery_frequency, enabled)
    if previous_frequency != delivery_frequency:
        return _next_report_subscription_delivery_at(delivery_frequency, enabled)
    if previous_next_delivery_at:
        return previous_next_delivery_at
    return _next_report_subscription_delivery_at(delivery_frequency, enabled)


def _serialize_report_subscription(row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "scope_type": row["scope_type"],
        "scope_key": row["scope_key"],
        "scope_label": row["scope_label"],
        "report_id": str(row["report_id"]) if row["report_id"] else None,
        "delivery_frequency": row["delivery_frequency"],
        "deliverable_focus": row["deliverable_focus"],
        "freshness_policy": row["freshness_policy"],
        "recipient_emails": list(row["recipient_emails"] or []),
        "delivery_note": row["delivery_note"] or "",
        "enabled": bool(row["enabled"]),
        "next_delivery_at": row["next_delivery_at"].isoformat() if row["next_delivery_at"] else None,
        "last_delivery_status": row["last_delivery_status"] or None,
        "last_delivery_at": row["last_delivery_at"].isoformat() if row["last_delivery_at"] else None,
        "last_delivery_summary": row["last_delivery_summary"] or "",
        "last_delivery_error": row["last_delivery_error"] or "",
        "last_delivery_report_count": int(row["last_delivery_report_count"] or 0),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


async def _fetch_report_subscription_row(
    pool,
    *,
    account_id: _uuid.UUID | str,
    scope_type: str,
    scope_key: str,
):
    return await pool.fetchrow(
        """
        SELECT s.id, s.scope_type, s.scope_key, s.scope_label, s.report_id,
               s.delivery_frequency, s.deliverable_focus, s.freshness_policy,
               s.recipient_emails, s.delivery_note, s.enabled, s.next_delivery_at,
               s.created_at, s.updated_at,
               dl.status AS last_delivery_status,
               dl.delivered_at AS last_delivery_at,
               dl.summary AS last_delivery_summary,
               dl.error AS last_delivery_error,
               COALESCE(array_length(dl.delivered_report_ids, 1), 0) AS last_delivery_report_count
        FROM b2b_report_subscriptions s
        LEFT JOIN LATERAL (
            SELECT status, delivered_at, summary, error, delivered_report_ids
            FROM b2b_report_subscription_delivery_log
            WHERE subscription_id = s.id
              AND status <> 'processing'
            ORDER BY CASE WHEN delivery_mode = 'live' THEN 0 ELSE 1 END,
                     delivered_at DESC
            LIMIT 1
        ) dl ON TRUE
        WHERE s.account_id = $1::uuid
          AND s.scope_type = $2
          AND s.scope_key = $3
        """,
        account_id,
        scope_type,
        scope_key,
    )


async def _fetch_report_subscription_schedule_row(
    pool,
    *,
    account_id: _uuid.UUID | str,
    scope_type: str,
    scope_key: str,
):
    return await pool.fetchrow(
        """
        SELECT enabled, delivery_frequency, next_delivery_at
        FROM b2b_report_subscriptions
        WHERE account_id = $1::uuid
          AND scope_type = $2
          AND scope_key = $3
        """,
        account_id,
        scope_type,
        scope_key,
    )


async def _load_accessible_tenant_report(pool, report_id: _uuid.UUID, user: AuthUser):
    row = await pool.fetchrow("SELECT * FROM b2b_intelligence WHERE id = $1", report_id)
    if not row:
        raise HTTPException(status_code=404, detail="Report not found")

    if settings.saas_auth.enabled and not _is_admin_user(user) and row["vendor_filter"]:
        acct = _uuid.UUID(user.account_id)
        tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1 AND vendor_name = $2",
            acct,
            row["vendor_filter"],
        )
        if not tracked:
            raise HTTPException(status_code=403, detail="Report vendor not in your tracked list")

    return row


async def _tracked_vendor_map(pool, account_id: _uuid.UUID) -> dict[str, dict]:
    rows = await pool.fetch(
        """
        SELECT vendor_name, track_mode, label
        FROM tracked_vendors
        WHERE account_id = $1
        ORDER BY added_at ASC
        """,
        account_id,
    )
    return {
        str(row["vendor_name"]).strip().lower(): {
            "vendor_name": row["vendor_name"],
            "track_mode": row["track_mode"],
            "label": row["label"],
        }
        for row in rows
    }


async def _canonical_competitive_set_payload(
    pool,
    account_id: _uuid.UUID,
    *,
    name: str,
    focal_vendor_name: str,
    competitor_vendor_names: list[str],
    refresh_mode: str,
    refresh_interval_hours: int | None,
) -> dict[str, Any]:
    name = str(name or "").strip()
    focal_vendor_name = await resolve_vendor_name(focal_vendor_name)
    competitors = [await resolve_vendor_name(vendor_name) for vendor_name in (competitor_vendor_names or [])]
    deduped_competitors: list[str] = []
    seen: set[str] = {focal_vendor_name.lower()}
    for vendor_name in competitors:
        key = vendor_name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_competitors.append(vendor_name)
    if len(deduped_competitors) > settings.b2b_churn.competitive_set_max_competitors:
        raise HTTPException(
            status_code=400,
            detail=(
                "Competitive set exceeds max competitors "
                f"({settings.b2b_churn.competitive_set_max_competitors})"
            ),
        )
    tracked = await _tracked_vendor_map(pool, account_id)
    missing = [
        vendor_name
        for vendor_name in [focal_vendor_name, *deduped_competitors]
        if vendor_name.lower() not in tracked
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "All competitive-set vendors must already be tracked. Missing: "
                + ", ".join(missing)
            ),
        )
    if refresh_mode == "scheduled" and refresh_interval_hours is None:
        raise HTTPException(status_code=400, detail="refresh_interval_hours required for scheduled competitive sets")
    if refresh_mode == "manual":
        refresh_interval_hours = None
    return {
        "name": name,
        "focal_vendor_name": focal_vendor_name,
        "competitor_vendor_names": deduped_competitors,
        "refresh_mode": refresh_mode,
        "refresh_interval_hours": refresh_interval_hours,
    }


def _competitive_set_defaults_payload() -> dict[str, Any]:
    return {
        "default_refresh_interval_hours": max(
            1,
            int(settings.b2b_churn.competitive_set_refresh_interval_seconds // 3600),
        ),
        "max_competitors": int(settings.b2b_churn.competitive_set_max_competitors),
        "default_changed_vendors_only": bool(
            settings.b2b_churn.competitive_set_changed_vendors_only_default
        ),
    }


async def _competitive_set_plan_payload(pool, competitive_set) -> dict[str, Any]:
    vendor_names = [competitive_set.focal_vendor_name, *competitive_set.competitor_vendor_names]
    category_by_vendor = await load_vendor_category_map(pool, vendor_names)
    plan = build_competitive_set_plan(
        competitive_set,
        category_by_vendor=category_by_vendor,
    )
    payload = plan.to_dict()
    payload["category_by_vendor"] = {
        vendor_name: category_by_vendor.get(vendor_name.lower()) or None
        for vendor_name in vendor_names
    }
    payload["estimate"] = await estimate_competitive_set_plan(pool, plan)
    return payload


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


class ReportSubscriptionUpsertRequest(BaseModel):
    scope_label: str = Field(..., min_length=1, max_length=255)
    delivery_frequency: str = Field("weekly", pattern="^(weekly|monthly|quarterly)$")
    deliverable_focus: str = Field(
        "all",
        pattern="^(all|battle_cards|executive_reports|comparison_packs)$",
    )
    freshness_policy: str = Field(
        "fresh_or_monitor",
        pattern="^(fresh_only|fresh_or_monitor|any)$",
    )
    recipients: list[EmailStr] = Field(default_factory=list)
    delivery_note: str = Field(default="", max_length=2000)
    enabled: bool = True


class CompetitiveSetRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    focal_vendor_name: str = Field(..., min_length=1, max_length=256)
    competitor_vendor_names: list[str] = Field(default_factory=list)
    active: bool = True
    refresh_mode: str = Field(default="manual", pattern="^(manual|scheduled)$")
    refresh_interval_hours: int | None = Field(default=None, ge=1, le=720)
    vendor_synthesis_enabled: bool = True
    pairwise_enabled: bool = True
    category_council_enabled: bool = False
    asymmetry_enabled: bool = False


class CompetitiveSetUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    focal_vendor_name: str | None = Field(default=None, min_length=1, max_length=256)
    competitor_vendor_names: list[str] | None = None
    active: bool | None = None
    refresh_mode: str | None = Field(default=None, pattern="^(manual|scheduled)$")
    refresh_interval_hours: int | None = Field(default=None, ge=1, le=720)
    vendor_synthesis_enabled: bool | None = None
    pairwise_enabled: bool | None = None
    category_council_enabled: bool | None = None
    asymmetry_enabled: bool | None = None


class CompetitiveSetRunRequest(BaseModel):
    force: bool = False
    force_cross_vendor: bool = False
    changed_vendors_only: bool | None = None


class PushToCrmOpportunity(BaseModel):
    company: str = Field(..., min_length=1, max_length=200)
    vendor: str = Field(..., min_length=1, max_length=200)
    urgency: float = Field(..., ge=0, le=10)
    pain: str | None = None
    role_type: str | None = None
    buying_stage: str | None = None
    contract_end: str | None = None
    decision_timeline: str | None = None
    decision_maker: bool | None = None
    competitor_context: str | None = None
    primary_quote: str | None = None
    trust_tier: str | None = None
    source: str | None = None
    review_id: str | None = None
    seat_count: int | None = None
    industry: str | None = None
    company_size: str | None = None
    company_domain: str | None = None
    company_country: str | None = None
    revenue_range: str | None = None
    alternatives: list[str] | None = None


class PushToCrmBody(BaseModel):
    opportunities: list[PushToCrmOpportunity] = Field(..., min_length=1, max_length=50)


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
# CRM push (export opportunities to configured CRM webhooks)
# ---------------------------------------------------------------------------


@router.post("/push-to-crm")
async def push_to_crm(
    body: PushToCrmBody,
    user: AuthUser = Depends(require_auth),
):
    """Push selected high-intent opportunities to configured CRM webhooks."""
    _require_b2b_product(user)
    pool = _pool_or_503()

    subs = await pool.fetch(
        """
        SELECT id, url, secret, account_id,
               COALESCE(channel, 'generic') AS channel,
               auth_header
        FROM b2b_webhook_subscriptions
        WHERE enabled = true
          AND account_id = $1
          AND channel LIKE 'crm_%'
        """,
        user.account_id,
    )

    if not subs:
        raise HTTPException(
            status_code=422,
            detail="No CRM webhook subscriptions configured. Add one via the webhook settings.",
        )

    from ..services.b2b.webhook_dispatcher import (
        _build_envelope,
        _format_for_channel,
        _deliver_single,
    )

    cfg = settings.b2b_webhook
    pushed = 0
    failed: list[dict[str, str]] = []

    for opp in body.opportunities:
        opp_data = opp.model_dump(exclude_none=True)
        opp_data["company_name"] = opp_data.pop("company", "")
        envelope = _build_envelope("high_intent_push", opp.vendor, opp_data)

        opp_ok = False
        failure_reason = "delivery_failed"
        for sub in subs:
            payload_bytes = _format_for_channel(sub["channel"], envelope)
            if len(payload_bytes) > cfg.max_payload_bytes:
                logger.warning(
                    "CRM push payload too large (%d bytes, max %d) for account=%s vendor=%s company=%s channel=%s",
                    len(payload_bytes),
                    cfg.max_payload_bytes,
                    user.account_id,
                    opp.vendor,
                    opp.company,
                    sub["channel"],
                )
                failure_reason = "payload_too_large"
                continue
            ok = await _deliver_single(
                pool, sub, "high_intent_push", envelope, payload_bytes, cfg,
            )
            if ok:
                opp_ok = True

        if opp_ok:
            pushed += 1
        else:
            failed.append(
                {
                    "company": opp.company,
                    "vendor": opp.vendor,
                    "reason": failure_reason,
                }
            )

    return {"pushed": pushed, "failed": failed}


# ---------------------------------------------------------------------------
# Competitive sets (scoped synthesis control)
# ---------------------------------------------------------------------------


@router.get("/competitive-sets")
async def list_competitive_sets(
    include_inactive: bool = Query(False),
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    repo = get_competitive_set_repo()
    sets = await repo.list_for_account(
        _uuid.UUID(user.account_id),
        include_inactive=include_inactive,
    )
    return {
        "competitive_sets": [item.to_dict() for item in sets],
        "count": len(sets),
        "defaults": _competitive_set_defaults_payload(),
    }


@router.post("/competitive-sets", status_code=201)
async def create_competitive_set(
    req: CompetitiveSetRequest,
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    pool = _pool_or_503()
    repo = get_competitive_set_repo()
    requested_name = str(req.name or "").strip()
    existing_name = await repo.get_by_name_for_account(
        _uuid.UUID(user.account_id),
        requested_name,
    )
    if existing_name:
        raise HTTPException(status_code=409, detail="Competitive set name already exists")
    payload = await _canonical_competitive_set_payload(
        pool,
        _uuid.UUID(user.account_id),
        name=req.name,
        focal_vendor_name=req.focal_vendor_name,
        competitor_vendor_names=req.competitor_vendor_names,
        refresh_mode=req.refresh_mode,
        refresh_interval_hours=req.refresh_interval_hours,
    )
    created = await repo.create(
        account_id=_uuid.UUID(user.account_id),
        active=req.active,
        vendor_synthesis_enabled=req.vendor_synthesis_enabled,
        pairwise_enabled=req.pairwise_enabled,
        category_council_enabled=req.category_council_enabled,
        asymmetry_enabled=req.asymmetry_enabled,
        **payload,
    )
    return created.to_dict()


@router.put("/competitive-sets/{competitive_set_id}")
async def update_competitive_set(
    competitive_set_id: _uuid.UUID,
    req: CompetitiveSetUpdateRequest,
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    pool = _pool_or_503()
    repo = get_competitive_set_repo()
    existing = await repo.get_by_id_for_account(competitive_set_id, _uuid.UUID(user.account_id))
    if not existing:
        raise HTTPException(status_code=404, detail="Competitive set not found")
    next_name = str(req.name).strip() if req.name is not None else existing.name
    existing_name = await repo.get_by_name_for_account(_uuid.UUID(user.account_id), next_name)
    if existing_name and existing_name.id != competitive_set_id:
        raise HTTPException(status_code=409, detail="Competitive set name already exists")
    canonical = await _canonical_competitive_set_payload(
        pool,
        _uuid.UUID(user.account_id),
        name=next_name,
        focal_vendor_name=(
            req.focal_vendor_name
            if req.focal_vendor_name is not None
            else existing.focal_vendor_name
        ),
        competitor_vendor_names=(
            req.competitor_vendor_names
            if req.competitor_vendor_names is not None
            else existing.competitor_vendor_names
        ),
        refresh_mode=req.refresh_mode if req.refresh_mode is not None else existing.refresh_mode,
        refresh_interval_hours=(
            req.refresh_interval_hours
            if req.refresh_interval_hours is not None or (req.refresh_mode == "manual")
            else existing.refresh_interval_hours
        ),
    )
    updated = await repo.update(
        competitive_set_id,
        active=req.active,
        vendor_synthesis_enabled=req.vendor_synthesis_enabled,
        pairwise_enabled=req.pairwise_enabled,
        category_council_enabled=req.category_council_enabled,
        asymmetry_enabled=req.asymmetry_enabled,
        **canonical,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Competitive set not found")
    return updated.to_dict()


@router.delete("/competitive-sets/{competitive_set_id}")
async def delete_competitive_set(
    competitive_set_id: _uuid.UUID,
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    repo = get_competitive_set_repo()
    existing = await repo.get_by_id_for_account(competitive_set_id, _uuid.UUID(user.account_id))
    if not existing:
        raise HTTPException(status_code=404, detail="Competitive set not found")
    deleted = await repo.delete(competitive_set_id)
    return {"deleted": deleted, "competitive_set_id": str(competitive_set_id)}


@router.get("/competitive-sets/{competitive_set_id}/plan")
async def preview_competitive_set_plan(
    competitive_set_id: _uuid.UUID,
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    pool = _pool_or_503()
    repo = get_competitive_set_repo()
    existing = await repo.get_by_id_for_account(competitive_set_id, _uuid.UUID(user.account_id))
    if not existing:
        raise HTTPException(status_code=404, detail="Competitive set not found")
    plan = await _competitive_set_plan_payload(pool, existing)
    recent_runs = await repo.list_runs_for_account_set(
        competitive_set_id,
        _uuid.UUID(user.account_id),
        limit=5,
    )
    return {
        "competitive_set": existing.to_dict(),
        "plan": plan,
        "recent_runs": [item.to_dict() for item in recent_runs],
    }


@router.post("/competitive-sets/{competitive_set_id}/run", status_code=202)
async def run_competitive_set_now(
    competitive_set_id: _uuid.UUID,
    req: CompetitiveSetRunRequest,
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    pool = _pool_or_503()
    repo = get_competitive_set_repo()
    competitive_set = await repo.get_by_id_for_account(competitive_set_id, _uuid.UUID(user.account_id))
    if not competitive_set:
        raise HTTPException(status_code=404, detail="Competitive set not found")
    plan = build_competitive_set_plan(
        competitive_set,
        category_by_vendor=await load_vendor_category_map(
            pool,
            [competitive_set.focal_vendor_name, *competitive_set.competitor_vendor_names],
        ),
    )
    task_repo = get_scheduled_task_repo()
    synthesis_task = await task_repo.get_by_name("b2b_reasoning_synthesis")
    if not synthesis_task:
        raise HTTPException(status_code=503, detail="b2b_reasoning_synthesis task is not registered")
    changed_vendors_only = (
        req.changed_vendors_only
        if req.changed_vendors_only is not None
        else bool(settings.b2b_churn.competitive_set_changed_vendors_only_default)
    )
    synthesis_task.metadata = {
        **(synthesis_task.metadata or {}),
        **plan_to_synthesis_metadata(plan),
        "scope_name": competitive_set.name,
        "scope_trigger": "manual",
        "force": req.force,
        "force_cross_vendor": req.force_cross_vendor,
        "changed_vendors_only": changed_vendors_only,
    }
    scheduler = get_task_scheduler()
    result = await scheduler.run_now(synthesis_task)
    return {
        **result,
        "competitive_set_id": str(competitive_set_id),
        "plan": plan.to_dict(),
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

    # Recent high-intent leads via shared adapter
    from ..autonomous.tasks._b2b_shared import read_high_intent_companies

    scoped_vendors: list[str] | None = None
    if t_params:
        sv_rows = await pool.fetch(
            "SELECT vendor_name FROM tracked_vendors WHERE account_id = $1",
            t_params[0],
        )
        scoped_vendors = [r["vendor_name"] for r in sv_rows]

    lead_items = await read_high_intent_companies(
        pool,
        min_urgency=7.0,
        scoped_vendors=scoped_vendors,
        limit=5,
    )

    return {
        "tracked_vendors": vendor_count,
        "avg_urgency": _safe_float(signal_stats["avg_urgency"] if signal_stats else 0, 0),
        "total_churn_signals": signal_stats["total_churn_signals"] if signal_stats else 0,
        "total_reviews": signal_stats["total_reviews"] if signal_stats else 0,
        "recent_leads": [
            {
                "company": item["company"],
                "vendor": item["vendor"],
                "urgency": _safe_float(item.get("urgency"), 0),
                "pain": item.get("pain"),
                "title": item.get("title"),
                "company_size": item.get("company_size"),
                "industry": item.get("industry"),
            }
            for item in lead_items
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
            "archetype": None,
            "archetype_confidence": None,
            "reasoning_mode": None,
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
            "archetype": None,
            "archetype_confidence": None,
            "reasoning_mode": None,
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

    # APPROVED-ENRICHMENT-READ: pain_category
    # Reason: aggregation, GROUP BY + COUNT
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

    from ..autonomous.tasks._b2b_shared import read_high_intent_companies

    hi_items = await read_high_intent_companies(
        pool,
        min_urgency=7.0,
        vendor_name=vname,
        limit=10,
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
            "company": item["company"],
            "urgency": _safe_float(item.get("urgency"), 0),
            "pain": item.get("pain"),
            "title": item.get("title"),
            "company_size": item.get("company_size"),
            "industry": item.get("industry"),
        }
        for item in hi_items
    ]

    profile["pain_distribution"] = [
        {"pain_category": r["pain"], "count": r["cnt"]}
        for r in pain_rows
    ]

    return profile


@router.get("/accounts-in-motion-feed")
async def list_tenant_accounts_in_motion_feed(
    min_urgency: float = Query(settings.b2b_churn.accounts_in_motion_min_urgency, ge=0, le=10),
    per_vendor_limit: int = Query(settings.b2b_churn.accounts_in_motion_max_per_vendor, ge=1, le=100),
    limit: int = Query(settings.b2b_churn.accounts_in_motion_feed_max_total, ge=1, le=200),
    user: AuthUser = Depends(require_auth),
):
    """Aggregated persisted accounts-in-motion feed across tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)

    tracked_rows = await pool.fetch(
        """
        SELECT vendor_name, track_mode, label, added_at
        FROM tracked_vendors
        WHERE account_id = $1
        ORDER BY added_at, vendor_name
        """,
        acct,
    )
    if not tracked_rows:
        return {
            "accounts": [],
            "count": 0,
            "tracked_vendor_count": 0,
            "vendors_with_accounts": 0,
            "min_urgency": min_urgency,
            "per_vendor_limit": per_vendor_limit,
            "freshest_report_date": None,
        }

    vendor_reports = await asyncio.gather(
        *[
            _list_accounts_in_motion_from_report(
                pool,
                row["vendor_name"],
                min_urgency=min_urgency,
                limit=per_vendor_limit,
                user=user,
            )
            for row in tracked_rows
        ]
    )

    accounts: list[dict] = []
    freshest_report_date: str | None = None
    vendors_with_accounts = 0
    for tracked, report in zip(tracked_rows, vendor_reports):
        if not report:
            continue
        report_date = report.get("report_date")
        if report.get("accounts"):
            vendors_with_accounts += 1
        if isinstance(report_date, str) and (
            freshest_report_date is None or report_date > freshest_report_date
        ):
            freshest_report_date = report_date
        for account in report.get("accounts") or []:
            if not isinstance(account, dict):
                continue
            accounts.append(
                {
                    **account,
                    "watch_vendor": tracked["vendor_name"],
                    "track_mode": tracked["track_mode"],
                    "watchlist_label": tracked["label"],
                    "report_date": report_date,
                    "stale_days": report.get("stale_days"),
                    "is_stale": bool(report.get("is_stale")),
                    "data_source": report.get("data_source"),
                }
            )

    accounts.sort(
        key=lambda account: (
            bool(account.get("is_stale")),
            -(account.get("opportunity_score") or 0),
            -(account.get("urgency") or 0),
            str(account.get("vendor") or ""),
            str(account.get("company") or ""),
        )
    )
    limited_accounts = accounts[:limit]
    return {
        "accounts": limited_accounts,
        "count": len(limited_accounts),
        "tracked_vendor_count": len(tracked_rows),
        "vendors_with_accounts": vendors_with_accounts,
        "min_urgency": min_urgency,
        "per_vendor_limit": per_vendor_limit,
        "freshest_report_date": freshest_report_date,
    }


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
    # APPROVED-ENRICHMENT-READ: pain_category
    # Reason: aggregation, GROUP BY week + COUNT
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

    # APPROVED-ENRICHMENT-READ: competitors_mentioned, churn_signals.intent_to_leave
    # (aggregation -- GROUP BY vendor + competitors + leaving + COUNT, not row-level)
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
    vendor_name: Optional[str] = Query(None),
    min_urgency: float = Query(7, ge=0, le=10),
    window_days: int = Query(30, ge=1, le=3650),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser = Depends(require_auth),
):
    """High-intent companies leaving tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    from ..autonomous.tasks._b2b_shared import read_high_intent_companies

    t_params = _tenant_params(user)
    scoped_vendors: list[str] | None = None
    if t_params:
        sv_rows = await pool.fetch(
            "SELECT vendor_name FROM tracked_vendors WHERE account_id = $1",
            t_params[0],
        )
        scoped_vendors = [r["vendor_name"] for r in sv_rows]

    capped = min(limit, 100)
    items = await read_high_intent_companies(
        pool,
        min_urgency=min_urgency,
        window_days=window_days,
        vendor_name=vendor_name,
        scoped_vendors=scoped_vendors,
        limit=capped,
    )

    companies = [
        {
            "company": item["company"],
            "vendor": item["vendor"],
            "category": item.get("category"),
            "role_level": item.get("role_level"),
            "decision_maker": item.get("decision_maker"),
            "urgency": _safe_float(item.get("urgency"), 0),
            "pain": item.get("pain"),
            "alternatives": _safe_json(item.get("alternatives")),
            "contract_signal": item.get("contract_signal"),
            "seat_count": item.get("seat_count"),
            "lock_in_level": item.get("lock_in_level"),
            "contract_end": item.get("contract_end"),
            "buying_stage": item.get("buying_stage"),
            "reviewer_title": item.get("title"),
            "company_size": item.get("company_size"),
            "industry": item.get("industry"),
            "review_id": item.get("review_id"),
            "source": item.get("source"),
            "quotes": _safe_json(item.get("quotes")),
            "intent_signals": item.get("intent_signals"),
            "relevance_score": _safe_float(item.get("relevance_score")),
            "author_churn_score": _safe_float(item.get("author_churn_score")),
            "resolution_confidence": item.get("resolution_confidence"),
            "verified_employee_count": item.get("verified_employee_count"),
            "company_domain": item.get("company_domain"),
            "company_country": item.get("company_country"),
            "revenue_range": item.get("revenue_range"),
            "founded_year": item.get("founded_year"),
            "company_description": item.get("company_description"),
        }
        for item in items
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
        vendor_name=vendor_name,
        min_urgency=min_urgency,
        window_days=window_days,
        limit=limit,
        user=user,
    )
    companies = lead_payload["leads"]
    return {"companies": companies, "count": len(companies)}


@router.get("/leads/{company}")
async def get_lead_detail(company: str, user: AuthUser = Depends(require_auth)):
    """Company drill-down: all reviews, signals, buying stage."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    from ..autonomous.tasks._b2b_shared import read_review_details

    t_params = _tenant_params(user)
    scoped_vendors: list[str] | None = None
    if t_params:
        sv_rows = await pool.fetch(
            "SELECT vendor_name FROM tracked_vendors WHERE account_id = $1",
            t_params[0],
        )
        scoped_vendors = [r["vendor_name"] for r in sv_rows]

    details = await read_review_details(
        pool,
        window_days=3650,
        scoped_vendors=scoped_vendors,
        company=company.strip(),
        recency_column="coalesce",
        limit=50,
    )

    if not details:
        raise HTTPException(status_code=404, detail="No reviews found for this company")

    reviews = [
        {
            "id": item.get("id"),
            "vendor_name": item["vendor_name"],
            "category": item.get("product_category"),
            "rating": _safe_float(item.get("rating")),
            "urgency": _safe_float(item.get("urgency_score"), 0),
            "pain": item.get("pain_category"),
            "intent_to_leave": item.get("intent_to_leave"),
            "decision_maker": item.get("decision_maker"),
            "role_level": item.get("role_level"),
            "buying_stage": item.get("buying_stage"),
            "alternatives": _safe_json(item.get("competitors_mentioned")),
            "contract_end": None,
            "enriched_at": str(item["enriched_at"]) if item.get("enriched_at") else None,
        }
        for item in details
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
    row = await _load_accessible_tenant_report(pool, rid, user)

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


@router.get("/report-subscriptions/{scope_type}/{scope_key}")
async def get_report_subscription(
    scope_type: str,
    scope_key: str,
    user: AuthUser = Depends(require_auth),
):
    """Load the saved recurring-delivery policy for the library or a specific report."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    normalized_scope_type, normalized_scope_key = _normalize_report_subscription_scope(
        scope_type,
        scope_key,
    )

    if normalized_scope_type == "report":
        report_id = _coerce_uuid(normalized_scope_key)
        if not report_id:
            raise HTTPException(status_code=400, detail="scope_key must be a report UUID")
        await _load_accessible_tenant_report(pool, report_id, user)

    row = await _fetch_report_subscription_row(
        pool,
        account_id=user.account_id,
        scope_type=normalized_scope_type,
        scope_key=normalized_scope_key,
    )

    return {"subscription": _serialize_report_subscription(row) if row else None}


@router.put("/report-subscriptions/{scope_type}/{scope_key}")
async def upsert_report_subscription(
    scope_type: str,
    scope_key: str,
    body: ReportSubscriptionUpsertRequest,
    user: AuthUser = Depends(require_auth),
):
    """Create or update the saved recurring-delivery policy for the library or a specific report."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    normalized_scope_type, normalized_scope_key = _normalize_report_subscription_scope(
        scope_type,
        scope_key,
    )
    account_id = _uuid.UUID(user.account_id)
    user_id = _coerce_uuid(getattr(user, "user_id", None))

    report_id: _uuid.UUID | None = None
    if normalized_scope_type == "report":
        report_id = _coerce_uuid(normalized_scope_key)
        if not report_id:
            raise HTTPException(status_code=400, detail="scope_key must be a report UUID")
        await _load_accessible_tenant_report(pool, report_id, user)

    normalized_recipients = _normalize_report_subscription_recipients(body.recipients)
    if body.enabled and not normalized_recipients:
        raise HTTPException(
            status_code=400,
            detail="Add at least one recipient before enabling recurring delivery",
        )

    scope_label = body.scope_label.strip()
    if not scope_label:
        raise HTTPException(status_code=400, detail="scope_label is required")

    existing_subscription = await _fetch_report_subscription_schedule_row(
        pool,
        account_id=account_id,
        scope_type=normalized_scope_type,
        scope_key=normalized_scope_key,
    )

    next_delivery_at = _resolve_report_subscription_next_delivery_at(
        existing_row=existing_subscription,
        delivery_frequency=body.delivery_frequency,
        enabled=body.enabled,
    )

    await pool.execute(
        """
        INSERT INTO b2b_report_subscriptions (
            account_id,
            report_id,
            scope_type,
            scope_key,
            scope_label,
            delivery_frequency,
            deliverable_focus,
            freshness_policy,
            recipient_emails,
            delivery_note,
            enabled,
            next_delivery_at,
            created_by,
            updated_by
        )
        VALUES (
            $1::uuid,
            $2::uuid,
            $3,
            $4,
            $5,
            $6,
            $7,
            $8,
            $9::text[],
            $10,
            $11,
            $12,
            $13::uuid,
            $13::uuid
        )
        ON CONFLICT (account_id, scope_type, scope_key)
        DO UPDATE
        SET report_id = EXCLUDED.report_id,
            scope_label = EXCLUDED.scope_label,
            delivery_frequency = EXCLUDED.delivery_frequency,
            deliverable_focus = EXCLUDED.deliverable_focus,
            freshness_policy = EXCLUDED.freshness_policy,
            recipient_emails = EXCLUDED.recipient_emails,
            delivery_note = EXCLUDED.delivery_note,
            enabled = EXCLUDED.enabled,
            next_delivery_at = EXCLUDED.next_delivery_at,
            updated_by = EXCLUDED.updated_by,
            updated_at = NOW()
        """,
        account_id,
        report_id,
        normalized_scope_type,
        normalized_scope_key,
        scope_label,
        body.delivery_frequency,
        body.deliverable_focus,
        body.freshness_policy,
        normalized_recipients,
        body.delivery_note.strip(),
        body.enabled,
        next_delivery_at,
        user_id,
    )

    row = await _fetch_report_subscription_row(
        pool,
        account_id=account_id,
        scope_type=normalized_scope_type,
        scope_key=normalized_scope_key,
    )

    return {"subscription": _serialize_report_subscription(row)}


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
    from ..autonomous.tasks._b2b_shared import read_review_details

    t_params = _tenant_params(user)
    scoped_vendors: list[str] | None = None
    if t_params:
        sv_rows = await pool.fetch(
            "SELECT vendor_name FROM tracked_vendors WHERE account_id = $1",
            t_params[0],
        )
        scoped_vendors = [r["vendor_name"] for r in sv_rows]

    capped = min(limit, 100)
    details = await read_review_details(
        pool,
        window_days=window_days,
        scoped_vendors=scoped_vendors,
        pain_category=pain_category,
        min_urgency=min_urgency,
        company=company,
        has_churn_intent=has_churn_intent,
        recency_column="coalesce",
        limit=capped,
    )

    reviews = [
        {
            "id": item.get("id"),
            "vendor_name": item["vendor_name"],
            "product_category": item.get("product_category"),
            "reviewer_company": item.get("reviewer_company"),
            "rating": _safe_float(item.get("rating")),
            "urgency_score": _safe_float(item.get("urgency_score")),
            "pain_category": item.get("pain_category"),
            "intent_to_leave": item.get("intent_to_leave"),
            "decision_maker": item.get("decision_maker"),
            "enriched_at": str(item["enriched_at"]) if item.get("enriched_at") else None,
            "reviewer_title": item.get("reviewer_title"),
            "company_size": item.get("company_size"),
            "industry": item.get("industry"),
        }
        for item in details
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
    user: AuthUser = Depends(require_b2b_plan("b2b_growth")),
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
    user: AuthUser = Depends(require_b2b_plan("b2b_growth")),
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
