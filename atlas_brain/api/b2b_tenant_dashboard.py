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
from fastapi.params import Param
from fastapi.routing import APIRoute
from pydantic import BaseModel, EmailStr, Field, field_validator

from ..auth.dependencies import AuthUser, require_auth, require_b2b_plan
from ..autonomous.tasks.b2b_campaign_generation import (
    generate_campaigns as _generate_campaigns,
)
from ..autonomous.tasks.campaign_send import _unsub_headers, _wrap_with_footer
from ..autonomous.tasks.campaign_suppression import is_suppressed
from ..autonomous.scheduler import get_task_scheduler
from ..config import settings
from ..services.scraping.target_provisioning import (
    provision_vendor_onboarding_targets,
)
from ..services.campaign_sender import get_campaign_sender
from ..services.company_normalization import (
    normalize_company_name,
    normalized_company_name_sql,
)
from ..services.b2b.report_trust import (
    REPORT_QUALITY_STATUSES,
    report_evidence_snapshot_payload,
    report_freshness_payload,
    report_review_payload,
    report_section_evidence_payload,
    report_trust_payload,
)
from ..services.b2b import watchlist_alerts as watchlist_alert_service
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
from ..templates.email.watchlist_alert_delivery import (
    render_watchlist_alert_delivery_html,
)
from .b2b_dashboard import (
    _accounts_in_motion_alert_basis,
    _accounts_in_motion_preview_alert_policy,
    _extract_report_account_preview_fields,
    _list_accounts_in_motion_from_report,
    _load_reasoning_views_for_vendors,
    _normalize_vendor_name,
    _overlay_reasoning_detail_from_view,
    _overlay_reasoning_summary_from_view,
)

logger = logging.getLogger("atlas.api.b2b_tenant")

router = APIRouter(prefix="/b2b/tenant", tags=["b2b-tenant"])
_LEGACY_ALIAS_REGISTRATION_DONE = False
_REPORT_SUBSCRIPTION_SCOPE_TYPES = {"library", "library_view", "report"}
_REPORT_SUBSCRIPTION_FREQUENCIES = {
    "weekly": 7,
    "monthly": 30,
    "quarterly": 90,
}
_WATCHLIST_ALERT_DELIVERY_FREQUENCIES = {
    "daily": 1,
    "weekly": 7,
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


def _matches_text_filter(candidate: Any, needle: str | None) -> bool:
    if not needle:
        return True
    candidate_text = str(candidate or "").strip().lower()
    needle_text = str(needle).strip().lower()
    if not candidate_text or not needle_text:
        return False
    return needle_text in candidate_text


def _canonical_review_predicate(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return f"{prefix}duplicate_of_review_id IS NULL"


def _matches_source_filter(account: dict[str, Any], source: str | None) -> bool:
    if not source:
        return True
    source_name = str(source).strip().lower()
    if not source_name:
        return True
    distribution = account.get("source_distribution")
    if isinstance(distribution, dict):
        for key, count in distribution.items():
            if str(key).strip().lower() == source_name and int(count or 0) > 0:
                return True
    for review in account.get("source_reviews") or []:
        if not isinstance(review, dict):
            continue
        if str(review.get("source") or "").strip().lower() == source_name:
            return True
    return False


def _clean_optional_text(value: Any) -> str | None:
    if isinstance(value, Param):
        value = value.default
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_required_text(value: Any, field_name: str) -> str:
    text = _clean_optional_text(value)
    if text is None:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    return text


def _clean_required_watchlist_view_name(value: Any) -> str:
    text = _clean_optional_text(value)
    if text is None:
        raise HTTPException(status_code=422, detail="Saved view name is required")
    return text


def _clean_watchlist_alert_delivery_frequency(value: Any) -> str:
    try:
        return watchlist_alert_service.clean_watchlist_alert_delivery_frequency(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _clean_optional_text_list(value: Any) -> list[str] | None:
    if isinstance(value, Param):
        value = value.default
    if isinstance(value, (list, tuple)):
        cleaned = [_clean_optional_text(item) for item in value]
        values = [item for item in cleaned if item]
        return values or None
    text = _clean_optional_text(value)
    return [text] if text is not None else None


def _company_signal_lookup_key(company_name: Any, vendor_name: Any) -> tuple[str, str] | None:
    company_key = normalize_company_name(str(company_name or "").strip())
    vendor_key = _normalize_vendor_name(vendor_name)
    if not company_key or not vendor_key:
        return None
    return company_key, vendor_key


async def _load_company_signal_ids_for_opportunities(
    pool,
    opportunities: list["PushToCrmOpportunity"],
) -> dict[tuple[str, str], str]:
    requested_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for opportunity in opportunities:
        lookup_key = _company_signal_lookup_key(opportunity.company, opportunity.vendor)
        if lookup_key is None or lookup_key in seen_pairs:
            continue
        seen_pairs.add(lookup_key)
        requested_pairs.append(lookup_key)

    if not requested_pairs:
        return {}

    rows = await pool.fetch(
        f"""
        WITH requested AS (
            SELECT DISTINCT company_key, vendor_key
            FROM UNNEST($1::text[], $2::text[]) AS request(company_key, vendor_key)
        )
        SELECT cs.id, cs.company_name, cs.vendor_name
        FROM b2b_company_signals cs
        JOIN requested req
          ON {normalized_company_name_sql('cs.company_name')} = req.company_key
         AND LOWER(TRIM(COALESCE(cs.vendor_name, ''))) = req.vendor_key
        """,
        [company_key for company_key, _vendor_key in requested_pairs],
        [vendor_key for _company_key, vendor_key in requested_pairs],
    )

    lookup: dict[tuple[str, str], str] = {}
    for row in rows:
        lookup_key = _company_signal_lookup_key(row.get('company_name'), row.get('vendor_name'))
        signal_id = str(row.get('id') or "").strip()
        if lookup_key is None or not signal_id or lookup_key in lookup:
            continue
        lookup[lookup_key] = signal_id
    return lookup


def _threshold_hit_numeric(value: Any, threshold: float | None) -> bool:
    numeric_threshold = _safe_float(threshold)
    if numeric_threshold is None:
        return False
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return False
    return numeric_value >= numeric_threshold


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return None
    return None


def _coerce_int_with_default(value: Any, default: int) -> int:
    coerced = _coerce_optional_int(value)
    return coerced if coerced is not None else default


def _parse_timestamp_value(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _threshold_hit_stale(
    threshold_days: int | None,
    *,
    stale_days: Any = None,
    freshness_timestamp: Any = None,
    fallback_timestamp: Any = None,
) -> bool:
    if threshold_days is None:
        return False
    if stale_days is not None:
        try:
            return int(stale_days) > int(threshold_days)
        except (TypeError, ValueError):
            pass
    timestamp = _parse_timestamp_value(freshness_timestamp) or _parse_timestamp_value(fallback_timestamp)
    if timestamp is None:
        return False
    age_days = (datetime.now(timezone.utc) - timestamp).total_seconds() / 86400.0
    return age_days > int(threshold_days)


async def _resolve_tracked_vendor_for_view(
    pool,
    account_id: _uuid.UUID,
    vendor_name: str | None,
) -> str | None:
    cleaned = _clean_optional_text(vendor_name)
    if not cleaned:
        return None
    resolved = await pool.fetchval(
        """
        SELECT vendor_name
        FROM tracked_vendors
        WHERE account_id = $1
          AND vendor_name ILIKE $2
        LIMIT 1
        """,
        account_id,
        cleaned,
    )
    if not resolved:
        raise HTTPException(status_code=422, detail="Saved view vendor must be one of your tracked vendors")
    return str(resolved)


async def _resolve_tracked_vendors_for_view(
    pool,
    account_id: _uuid.UUID,
    vendor_names: list[str] | None,
) -> list[str] | None:
    cleaned = _clean_optional_text_list(vendor_names)
    if not cleaned:
        return None
    resolved = []
    for name in cleaned:
        row = await pool.fetchval(
            """
            SELECT vendor_name
            FROM tracked_vendors
            WHERE account_id = $1
              AND vendor_name ILIKE $2
            LIMIT 1
            """,
            account_id,
            name,
        )
        if not row:
            raise HTTPException(status_code=422, detail=f"Vendor '{name}' is not in your tracked vendors")
        resolved.append(str(row))
    return resolved or None


def _merge_vendor_names_from_request(req) -> list[str] | None:
    """Merge vendor_names and vendor_name fields from request for backward compat."""
    vendor_names = _clean_optional_text_list(getattr(req, "vendor_names", None))
    if vendor_names:
        return vendor_names
    vendor_name = _clean_optional_text(getattr(req, "vendor_name", None))
    return [vendor_name] if vendor_name is not None else None


def _watchlist_view_payload(row: Any) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "name": row["name"],
        "vendor_name": (row["vendor_names"] or [None])[0],
        "vendor_names": list(row["vendor_names"] or []),
        "category": row["category"],
        "source": row["source"],
        "min_urgency": _safe_float(row["min_urgency"]),
        "include_stale": bool(row["include_stale"]),
        "named_accounts_only": bool(row["named_accounts_only"]),
        "changed_wedges_only": bool(row["changed_wedges_only"]),
        "vendor_alert_threshold": _safe_float(row.get("vendor_alert_threshold")),
        "account_alert_threshold": _safe_float(row.get("account_alert_threshold")),
        "stale_days_threshold": int(row["stale_days_threshold"]) if row.get("stale_days_threshold") is not None else None,
        "alert_email_enabled": bool(row.get("alert_email_enabled")),
        "alert_delivery_frequency": str(row.get("alert_delivery_frequency") or "daily"),
        "next_alert_delivery_at": str(row["next_alert_delivery_at"]) if row.get("next_alert_delivery_at") else None,
        "last_alert_delivery_at": str(row["last_alert_delivery_at"]) if row.get("last_alert_delivery_at") else None,
        "last_alert_delivery_status": row.get("last_alert_delivery_status"),
        "last_alert_delivery_summary": row.get("last_alert_delivery_summary"),
        "created_at": str(row["created_at"]) if row["created_at"] else None,
        "updated_at": str(row["updated_at"]) if row["updated_at"] else None,
    }

def _watchlist_vendor_freshness_status(signal: dict[str, Any]) -> tuple[str, str | None, str | None]:
    last_computed_at = signal.get("last_computed_at")
    if not signal.get("reasoning_source"):
        return (
            "synthesis_pending",
            "Reasoning synthesis has not been materialized for this vendor yet",
            str(last_computed_at) if last_computed_at else None,
        )
    if bool(signal.get("data_stale")):
        timestamp = signal.get("data_as_of_date") or last_computed_at
        return (
            "stale",
            "Reasoning synthesis is older than the current watchlist data window",
            str(timestamp) if timestamp else None,
        )
    ts = last_computed_at
    if isinstance(ts, str):
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - parsed).total_seconds() / 3600
            if age_hours > 24:
                return "stale", "Vendor movement snapshot is older than 24 hours", ts
        except ValueError:
            pass
    if last_computed_at:
        return "fresh", None, str(last_computed_at)
    timestamp = signal.get("data_as_of_date")
    return (
        "artifact_missing",
        "Vendor movement row is missing a computed timestamp",
        str(timestamp) if timestamp else None,
    )


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        return default

async def _fetch_watchlist_view_row(
    pool,
    *,
    account_id: _uuid.UUID,
    view_id: _uuid.UUID,
):
    return await pool.fetchrow(
        """
        SELECT id, account_id, name, vendor_names, category, source, min_urgency,
               include_stale, named_accounts_only, changed_wedges_only,
               vendor_alert_threshold, account_alert_threshold, stale_days_threshold,
               alert_email_enabled, alert_delivery_frequency, next_alert_delivery_at,
               last_alert_delivery_at, last_alert_delivery_status, last_alert_delivery_summary,
               created_at, updated_at
        FROM b2b_watchlist_views
        WHERE id = $1
          AND account_id = $2
        """,
        view_id,
        account_id,
    )


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
        raise HTTPException(status_code=400, detail="scope_type must be library, library_view, or report")

    normalized_scope_key = scope_key.strip()
    if not normalized_scope_key:
        raise HTTPException(status_code=400, detail="scope_key is required")

    if normalized_scope_type == "library" and normalized_scope_key != "library":
        raise HTTPException(status_code=400, detail="Library scope_key must be 'library'")

    return normalized_scope_type, normalized_scope_key


def _normalize_report_subscription_filter_payload(
    scope_type: str,
    payload: dict[str, Any] | None,
) -> dict[str, str]:
    if scope_type != "library_view":
        return {}
    if not isinstance(payload, dict):
        payload = {}

    normalized: dict[str, str] = {}
    report_type = str(payload.get("report_type") or "").strip()
    vendor_filter = str(payload.get("vendor_filter") or "").strip()
    quality_status = str(payload.get("quality_status") or "").strip().lower()
    freshness_state = str(payload.get("freshness_state") or "").strip().lower()
    review_state = str(payload.get("review_state") or "").strip().lower()
    if report_type:
        normalized["report_type"] = report_type
    if vendor_filter:
        normalized["vendor_filter"] = vendor_filter
    if quality_status:
        if quality_status not in REPORT_QUALITY_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "quality_status must be sales_ready, needs_review, "
                    "thin_evidence, or deterministic_fallback"
                ),
            )
        normalized["quality_status"] = quality_status
    if freshness_state:
        if freshness_state not in {"fresh", "monitor", "stale"}:
            raise HTTPException(status_code=400, detail="freshness_state must be fresh, monitor, or stale")
        normalized["freshness_state"] = freshness_state
    if review_state:
        if review_state not in {"clean", "warnings", "open_review", "blocked"}:
            raise HTTPException(status_code=400, detail="review_state must be clean, warnings, open_review, or blocked")
        normalized["review_state"] = review_state
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="library_view subscriptions require at least one active filter",
        )
    return normalized


def _normalize_report_list_filter(name: str, value: Optional[str]) -> str | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    if name == "quality_status" and normalized not in REPORT_QUALITY_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=(
                "quality_status must be sales_ready, needs_review, "
                "thin_evidence, or deterministic_fallback"
            ),
        )
    if name == "freshness_state" and normalized not in {"fresh", "monitor", "stale"}:
        raise HTTPException(status_code=400, detail="freshness_state must be fresh, monitor, or stale")
    if name == "review_state" and normalized not in {"clean", "warnings", "open_review", "blocked"}:
        raise HTTPException(status_code=400, detail="review_state must be clean, warnings, open_review, or blocked")
    return normalized


def _report_freshness_state_for_list(row) -> str:
    data_stale = bool(row["data_stale"]) if "data_stale" in row else False
    return report_freshness_payload(
        row["report_date"] or row["created_at"],
        data_stale=data_stale,
    )["state"]


def _report_review_state_for_list(row) -> str:
    return report_review_payload(
        row["blocker_count"] or 0,
        row["warning_count"] or 0,
        row["unresolved_issue_count"] or 0,
    )["state"]


def _report_matches_list_filters(
    row,
    *,
    quality_status: str | None,
    freshness_state: str | None,
    review_state: str | None,
) -> bool:
    if quality_status and str(row["quality_status"] or "").strip().lower() != quality_status:
        return False
    if freshness_state and _report_freshness_state_for_list(row) != freshness_state:
        return False
    if review_state and _report_review_state_for_list(row) != review_state:
        return False
    return True


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
    filter_payload = _safe_json(row["filter_payload"])
    if not isinstance(filter_payload, dict):
        filter_payload = {}
    try:
        last_delivery_freshness_state = row["last_delivery_freshness_state"]
    except Exception:
        last_delivery_freshness_state = None
    return {
        "id": str(row["id"]),
        "scope_type": row["scope_type"],
        "scope_key": row["scope_key"],
        "scope_label": row["scope_label"],
        "filter_payload": filter_payload,
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
        "last_delivery_freshness_state": last_delivery_freshness_state or None,
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
        SELECT s.id, s.scope_type, s.scope_key, s.scope_label, s.filter_payload, s.report_id,
               s.delivery_frequency, s.deliverable_focus, s.freshness_policy,
               s.recipient_emails, s.delivery_note, s.enabled, s.next_delivery_at,
               s.created_at, s.updated_at,
               dl.status AS last_delivery_status,
               dl.delivered_at AS last_delivery_at,
               dl.freshness_state AS last_delivery_freshness_state,
               dl.summary AS last_delivery_summary,
               dl.error AS last_delivery_error,
               COALESCE(array_length(dl.delivered_report_ids, 1), 0) AS last_delivery_report_count
        FROM b2b_report_subscriptions s
        LEFT JOIN LATERAL (
            SELECT status, delivered_at, freshness_state, summary, error, delivered_report_ids
            FROM b2b_report_subscription_delivery_log
            WHERE subscription_id = s.id
              AND status <> 'processing'
            ORDER BY delivered_at DESC NULLS LAST,
                     id DESC
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


async def _fetch_latest_tenant_vendor_report(
    pool,
    *,
    report_type: str,
    vendor_name: str,
):
    return await pool.fetchrow(
        """
        SELECT *
        FROM b2b_intelligence
        WHERE report_type = $1
          AND LOWER(vendor_filter) = LOWER($2)
        ORDER BY report_date DESC NULLS LAST, b2b_intelligence.created_at DESC NULLS LAST, b2b_intelligence.id DESC
        LIMIT 1
        """,
        report_type,
        vendor_name,
    )


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


class BattleCardRequest(BaseModel):
    vendor_name: str = Field(..., min_length=1, max_length=200)
    refresh: bool = False


class ReportSubscriptionUpsertRequest(BaseModel):
    scope_label: str = Field(..., min_length=1, max_length=255)
    filter_payload: dict[str, Any] = Field(default_factory=dict)
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

    @field_validator(
        "company",
        "vendor",
        mode="before",
    )
    @classmethod
    def _trim_required_text(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value is required")
        return text

    @field_validator(
        "pain",
        "role_type",
        "buying_stage",
        "contract_end",
        "decision_timeline",
        "competitor_context",
        "primary_quote",
        "trust_tier",
        "source",
        "review_id",
        "industry",
        "company_size",
        "company_domain",
        "company_country",
        "revenue_range",
        mode="before",
    )
    @classmethod
    def _trim_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("alternatives", mode="before")
    @classmethod
    def _trim_alternatives(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            return value
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or None


class PushToCrmBody(BaseModel):
    opportunities: list[PushToCrmOpportunity] = Field(..., min_length=1, max_length=50)


class WatchlistViewRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    vendor_name: str | None = Field(default=None, max_length=255)
    vendor_names: list[str] | None = Field(default=None)
    category: str | None = Field(default=None, max_length=255)
    source: str | None = Field(default=None, max_length=255)
    min_urgency: float | None = Field(default=None, ge=0, le=10)
    include_stale: bool = True
    named_accounts_only: bool = False
    changed_wedges_only: bool = False
    vendor_alert_threshold: float | None = Field(default=None, ge=0, le=10)
    account_alert_threshold: float | None = Field(default=None, ge=0, le=10)
    stale_days_threshold: int | None = Field(default=None, ge=0, le=365)
    alert_email_enabled: bool = False
    alert_delivery_frequency: str = Field(default="daily", min_length=1, max_length=20)


_VALID_DISPOSITIONS = {"snoozed", "dismissed", "saved"}


class SetDispositionRequest(BaseModel):
    opportunity_key: str = Field(..., min_length=1, max_length=600)
    company: str = Field(..., min_length=1, max_length=200)
    vendor: str = Field(..., min_length=1, max_length=200)
    review_id: str | None = Field(default=None, max_length=200)
    disposition: str = Field(..., min_length=1, max_length=20)
    snoozed_until: str | None = Field(
        default=None,
        description="ISO 8601 timestamp, required when disposition=snoozed",
    )


class BulkDispositionItem(BaseModel):
    opportunity_key: str = Field(..., min_length=1, max_length=600)
    company: str = Field(..., min_length=1, max_length=200)
    vendor: str = Field(..., min_length=1, max_length=200)
    review_id: str | None = Field(default=None, max_length=200)


class BulkSetDispositionRequest(BaseModel):
    items: list[BulkDispositionItem] = Field(..., min_length=1, max_length=100)
    disposition: str = Field(..., min_length=1, max_length=20)
    snoozed_until: str | None = Field(default=None)


class RemoveDispositionsRequest(BaseModel):
    opportunity_keys: list[str] = Field(..., min_length=1, max_length=100)


class WatchlistAlertEmailRequest(BaseModel):
    evaluate_before_send: bool = True


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
               snap.snapshot_date AS latest_snapshot_date,
               rpt.report_date AS latest_accounts_report_date
        FROM tracked_vendors tv
        LEFT JOIN LATERAL (
            SELECT snapshot_date
            FROM b2b_vendor_snapshots snap
            WHERE snap.vendor_name = tv.vendor_name
            ORDER BY snapshot_date DESC
            LIMIT 1
        ) snap ON TRUE
        LEFT JOIN LATERAL (
            SELECT report_date
            FROM b2b_intelligence bi
            WHERE bi.report_type = 'accounts_in_motion'
              AND LOWER(COALESCE(bi.vendor_filter, '')) = LOWER(tv.vendor_name)
            ORDER BY bi.report_date DESC NULLS LAST, bi.created_at DESC NULLS LAST
            LIMIT 1
        ) rpt ON TRUE
        WHERE tv.account_id = $1
        ORDER BY tv.added_at
        """,
        acct,
    )

    from ..autonomous.tasks._b2b_shared import read_best_vendor_signal_rows

    tracked_vendor_names = [row["vendor_name"] for row in rows if row.get("vendor_name")]
    signal_rows = []
    if tracked_vendor_names:
        signal_rows = await read_best_vendor_signal_rows(
            pool,
            vendor_names=tracked_vendor_names,
            limit=len(tracked_vendor_names),
        )
    signal_by_vendor = {
        _normalize_vendor_name(row.get("vendor_name")): row
        for row in signal_rows
        if row.get("vendor_name")
    }

    vendors = []
    reasoning_views = await _load_reasoning_views_for_vendors(
        pool,
        [row["vendor_name"] for row in rows],
    )
    for r in rows:
        signal_row = signal_by_vendor.get(_normalize_vendor_name(r["vendor_name"])) or {}
        vendor = {
            "id": str(r["id"]),
            "vendor_name": r["vendor_name"],
            "track_mode": r["track_mode"],
            "label": r["label"],
            "added_at": str(r["added_at"]) if r["added_at"] else None,
            "avg_urgency": _safe_float(signal_row.get("avg_urgency_score")),
            "churn_intent_count": signal_row.get("churn_intent_count"),
            "total_reviews": signal_row.get("total_reviews"),
            "nps_proxy": _safe_float(signal_row.get("nps_proxy")),
            "last_computed_at": str(signal_row.get("last_computed_at")) if signal_row.get("last_computed_at") else None,
            "latest_snapshot_date": str(r["latest_snapshot_date"]) if r["latest_snapshot_date"] else None,
            "latest_accounts_report_date": str(r["latest_accounts_report_date"]) if r["latest_accounts_report_date"] else None,
            "freshness_status": None,
            "freshness_reason": None,
            "freshness_timestamp": None,
        }
        view = reasoning_views.get(_normalize_vendor_name(vendor["vendor_name"]))
        if view is not None:
            _overlay_reasoning_summary_from_view(vendor, view)
            from ..autonomous.tasks._b2b_synthesis_reader import inject_synthesis_freshness

            inject_synthesis_freshness(
                vendor,
                view,
                requested_as_of=date.today(),
            )
        freshness_status, freshness_reason, freshness_timestamp = _watchlist_vendor_freshness_status(vendor)
        vendor["freshness_status"] = freshness_status
        vendor["freshness_reason"] = freshness_reason
        vendor["freshness_timestamp"] = freshness_timestamp
        vendors.append(vendor)

    return {"vendors": vendors, "count": len(vendors)}


@router.post("/vendors")
async def add_tracked_vendor(req: AddVendorRequest, user: AuthUser = Depends(require_auth)):
    """Add a vendor to track. Enforces vendor_limit in a transaction."""
    _require_b2b_product(user)
    vendor_name = _clean_required_text(req.vendor_name, "vendor_name")
    if req.track_mode not in ("own", "competitor"):
        raise HTTPException(status_code=400, detail="track_mode must be 'own' or 'competitor'")
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)

    canonical_vendor = await resolve_vendor_name(vendor_name)
    label = _clean_optional_text(req.label) or ""
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
    vname = _clean_required_text(vendor_name, "vendor_name")
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)

    canonical_vendor = await resolve_vendor_name(vname)
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
    query_text = _clean_required_text(q, "q")
    pool = _pool_or_503()

    from ..autonomous.tasks._b2b_shared import read_best_vendor_signal_rows

    rows = await read_best_vendor_signal_rows(
        pool,
        vendor_name_query=query_text,
        limit=min(limit, 50),
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
          AND 'high_intent_push' = ANY(event_types)
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
    max_payload_bytes = _coerce_int_with_default(getattr(cfg, "max_payload_bytes", 65536), 65536)
    pushed = 0
    failed: list[dict[str, str]] = []
    company_signal_ids = await _load_company_signal_ids_for_opportunities(pool, body.opportunities)

    for opp in body.opportunities:
        opp_data = opp.model_dump(exclude_none=True)
        opp_data["company_name"] = opp_data.pop("company", "")
        lookup_key = _company_signal_lookup_key(opp.company, opp.vendor)
        if lookup_key is not None:
            company_signal_id = company_signal_ids.get(lookup_key)
            if company_signal_id:
                opp_data["company_signal_id"] = company_signal_id
        envelope = _build_envelope("high_intent_push", opp.vendor, opp_data)

        opp_ok = False
        failure_reason = "delivery_failed"
        for sub in subs:
            payload_bytes = _format_for_channel(sub["channel"], envelope)
            if len(payload_bytes) > max_payload_bytes:
                logger.warning(
                    "CRM push payload too large (%d bytes, max %d) for account=%s vendor=%s company=%s channel=%s",
                    len(payload_bytes),
                    max_payload_bytes,
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
    repo = get_competitive_set_repo()
    requested_name = _clean_required_text(req.name, "name")
    existing_name = await repo.get_by_name_for_account(
        _uuid.UUID(user.account_id),
        requested_name,
    )
    if existing_name:
        raise HTTPException(status_code=409, detail="Competitive set name already exists")
    pool = _pool_or_503()
    payload = await _canonical_competitive_set_payload(
        pool,
        _uuid.UUID(user.account_id),
        name=requested_name,
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
    next_name = _clean_required_text(req.name, "name") if req.name is not None else None
    pool = _pool_or_503()
    repo = get_competitive_set_repo()
    existing = await repo.get_by_id_for_account(competitive_set_id, _uuid.UUID(user.account_id))
    if not existing:
        raise HTTPException(status_code=404, detail="Competitive set not found")
    if next_name is None:
        next_name = existing.name
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
    from ..autonomous.tasks._b2b_shared import (
        read_high_intent_companies,
        read_vendor_signal_overview,
        read_vendor_signal_summary,
    )

    if t_params:
        vendor_count = await pool.fetchval(
            "SELECT COUNT(*) FROM tracked_vendors WHERE account_id = $1",
            t_params[0],
        )
        signal_stats = await read_vendor_signal_overview(
            pool,
            tracked_account_id=t_params[0],
        )
    else:
        summary = await read_vendor_signal_summary(
            pool,
        )
        vendor_count = summary.get("total_vendors", 0)
        signal_stats = await read_vendor_signal_overview(pool)

    # Recent high-intent leads via shared adapter
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
    vendor_names: Optional[list[str]] = Query(None),
    min_urgency: float = Query(0, ge=0, le=10),
    category: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser = Depends(require_auth),
):
    """Churn signals for tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    vendor_name = _clean_optional_text(vendor_name)
    vendor_names = _clean_optional_text_list(vendor_names)
    category = _clean_optional_text(category)
    t_params = _tenant_params(user)
    tracked_account_id = t_params[0] if t_params else None
    from ..autonomous.tasks._b2b_shared import (
        read_vendor_signal_rows,
        read_vendor_signal_summary,
    )

    rows = await read_vendor_signal_rows(
        pool,
        vendor_name_query=None if vendor_names else vendor_name,
        vendor_names=vendor_names,
        min_urgency=min_urgency,
        product_category=category,
        tracked_account_id=tracked_account_id,
        include_snapshot_metrics=True,
        limit=limit,
    )

    summary = await read_vendor_signal_summary(
        pool,
        vendor_name_query=None if vendor_names else vendor_name,
        vendor_names=vendor_names,
        min_urgency=min_urgency,
        product_category=category,
        tracked_account_id=tracked_account_id,
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
    vendor_names: Optional[list[str]] = Query(None),
    category: Optional[str] = Query(None),
    vendor_alert_threshold: float | None = Query(None, ge=0, le=10),
    stale_days_threshold: int | None = Query(None, ge=0, le=365),
    limit: int = Query(10, ge=1, le=100),
    user: AuthUser = Depends(require_auth),
):
    """Slow-burn watchlist for tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    vendor_name = _clean_optional_text(vendor_name)
    vendor_names = _clean_optional_text_list(vendor_names)
    category = _clean_optional_text(category)
    vendor_alert_threshold = _safe_float(vendor_alert_threshold)
    stale_days_threshold = _coerce_optional_int(stale_days_threshold)
    limit = _coerce_optional_int(limit) or 10
    t_params = _tenant_params(user)
    tracked_account_id = t_params[0] if t_params else None
    from ..autonomous.tasks._b2b_shared import read_ranked_vendor_signal_rows

    rows = await read_ranked_vendor_signal_rows(
        pool,
        vendor_name_query=None if vendor_names else vendor_name,
        vendor_names=vendor_names,
        product_category=category,
        tracked_account_id=tracked_account_id,
        require_snapshot_activity=True,
        limit=limit,
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
            from ..autonomous.tasks._b2b_synthesis_reader import inject_synthesis_freshness

            inject_synthesis_freshness(
                signal,
                view,
                requested_as_of=date.today(),
            )
        freshness_status, freshness_reason, freshness_timestamp = _watchlist_vendor_freshness_status(signal)
        signal["freshness_status"] = freshness_status
        signal["freshness_reason"] = freshness_reason
        signal["freshness_timestamp"] = freshness_timestamp
        signal["vendor_alert_hit"] = _threshold_hit_numeric(
            signal.get("avg_urgency_score"),
            vendor_alert_threshold,
        )
        signal["stale_threshold_hit"] = _threshold_hit_stale(
            stale_days_threshold,
            freshness_timestamp=freshness_timestamp,
            fallback_timestamp=signal.get("last_computed_at"),
        )

    return {
        "signals": signals,
        "count": len(rows),
        "vendor_alert_threshold": vendor_alert_threshold,
        "vendor_alert_hit_count": sum(1 for signal in signals if signal.get("vendor_alert_hit")),
        "stale_days_threshold": stale_days_threshold,
        "stale_threshold_hit_count": sum(1 for signal in signals if signal.get("stale_threshold_hit")),
    }


@router.get("/signals/{vendor_name}")
async def get_vendor_detail(vendor_name: str, user: AuthUser = Depends(require_auth)):
    """Full vendor detail: signal + pain + competitors + evidence."""
    _require_b2b_product(user)
    vname = _clean_required_text(vendor_name, "vendor_name")
    pool = _pool_or_503()

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

    from ..autonomous.tasks._b2b_shared import read_vendor_signal_detail

    signal_row = await read_vendor_signal_detail(
        pool,
        vendor_name_query=vname,
        include_snapshot_metrics=True,
    )

    counts = await pool.fetchrow(
        f"""
        SELECT COUNT(*) AS total_reviews,
               COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched
        FROM b2b_reviews
        WHERE vendor_name ILIKE '%' || $1 || '%'
          AND {_canonical_review_predicate()}
        """,
        vname,
    )

    # APPROVED-ENRICHMENT-READ: pain_category
    # Reason: aggregation, GROUP BY + COUNT
    pain_rows = await pool.fetch(
        f"""
        SELECT enrichment->>'pain_category' AS pain, COUNT(*) AS cnt
        FROM b2b_reviews
        WHERE vendor_name ILIKE '%' || $1 || '%'
          AND enrichment_status = 'enriched'
          AND {_canonical_review_predicate()}
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
    vendor_name: Optional[str] = Query(None),
    vendor_names: Optional[list[str]] = Query(None),
    category: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    min_urgency: float = Query(settings.b2b_churn.accounts_in_motion_min_urgency, ge=0, le=10),
    include_stale: bool = Query(True),
    named_accounts_only: bool = Query(False),
    account_alert_threshold: float | None = Query(None, ge=0, le=10),
    stale_days_threshold: int | None = Query(None, ge=0, le=365),
    per_vendor_limit: int = Query(settings.b2b_churn.accounts_in_motion_max_per_vendor, ge=1, le=100),
    limit: int = Query(settings.b2b_churn.accounts_in_motion_feed_max_total, ge=1, le=200),
    user: AuthUser = Depends(require_auth),
):
    """Aggregated persisted accounts-in-motion feed across tracked vendors."""
    _require_b2b_product(user)
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)
    vendor_name = _clean_optional_text(vendor_name)
    vendor_names = _clean_optional_text_list(vendor_names)
    category = _clean_optional_text(category)
    source = _clean_optional_text(source)
    include_stale = include_stale if isinstance(include_stale, bool) else True
    named_accounts_only = bool(named_accounts_only)
    min_urgency = _safe_float(min_urgency, settings.b2b_churn.accounts_in_motion_min_urgency)
    account_alert_threshold = _safe_float(account_alert_threshold)
    stale_days_threshold = _coerce_optional_int(stale_days_threshold)
    per_vendor_limit = _coerce_optional_int(per_vendor_limit) or settings.b2b_churn.accounts_in_motion_max_per_vendor
    limit = _coerce_optional_int(limit) or settings.b2b_churn.accounts_in_motion_feed_max_total

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
            "account_alert_threshold": account_alert_threshold,
            "account_alert_hit_count": 0,
            "stale_days_threshold": stale_days_threshold,
            "stale_threshold_hit_count": 0,
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
                named_accounts_only=named_accounts_only,
            )
            for row in tracked_rows
        ]
    )

    accounts: list[dict] = []
    freshest_report_date: str | None = None
    vendors_with_accounts = 0
    for tracked, report in zip(tracked_rows, vendor_reports):
        if vendor_names:
            if tracked["vendor_name"].lower() not in {v.lower() for v in vendor_names}:
                continue
        elif vendor_name and not _matches_text_filter(tracked["vendor_name"], vendor_name):
            continue
        if not report:
            continue
        report_date = report.get("report_date")
        report_is_stale = bool(report.get("is_stale"))
        if report_is_stale and not include_stale:
            continue
        filtered_accounts: list[dict] = []
        for account in report.get("accounts") or []:
            if not isinstance(account, dict):
                continue
            if category and not _matches_text_filter(account.get("category"), category):
                continue
            if source and not _matches_source_filter(account, source):
                continue
            account_alert_score, account_alert_score_source = _accounts_in_motion_alert_basis(account)
            account_alert_eligible, account_alert_policy_reason = _accounts_in_motion_preview_alert_policy(
                account
            )
            filtered_accounts.append(
                {
                    **account,
                    "watch_vendor": tracked["vendor_name"],
                    "track_mode": tracked["track_mode"],
                    "watchlist_label": tracked["label"],
                    "report_date": report_date,
                    "stale_days": report.get("stale_days"),
                    "is_stale": bool(report.get("is_stale")),
                    "data_source": report.get("data_source"),
                    "freshness_status": report.get("freshness_status"),
                    "freshness_reason": report.get("freshness_reason"),
                    "freshness_timestamp": report.get("freshness_timestamp"),
                    "account_alert_score": account_alert_score,
                    "account_alert_score_source": account_alert_score_source,
                    "account_alert_eligible": account_alert_eligible,
                    "account_alert_policy_reason": account_alert_policy_reason,
                    "account_alert_hit": account_alert_eligible and _threshold_hit_numeric(
                        account_alert_score,
                        account_alert_threshold,
                    ),
                    "stale_threshold_hit": _threshold_hit_stale(
                        stale_days_threshold,
                        stale_days=report.get("stale_days"),
                        freshness_timestamp=report.get("freshness_timestamp"),
                        fallback_timestamp=report_date,
                    ),
                }
            )
        if not filtered_accounts:
            continue
        vendors_with_accounts += 1
        if isinstance(report_date, str) and (
            freshest_report_date is None or report_date > freshest_report_date
        ):
            freshest_report_date = report_date
        accounts.extend(filtered_accounts)

    accounts.sort(
        key=lambda account: (
            bool(account.get("is_stale")),
            -(account.get("opportunity_score") or 0),
            -(_safe_float(account.get("account_alert_score")) or 0),
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
        "account_alert_threshold": account_alert_threshold,
        "account_alert_hit_count": sum(1 for account in limited_accounts if account.get("account_alert_hit")),
        "stale_days_threshold": stale_days_threshold,
        "stale_threshold_hit_count": sum(1 for account in limited_accounts if account.get("stale_threshold_hit")),
        "per_vendor_limit": per_vendor_limit,
        "freshest_report_date": freshest_report_date,
    }


@router.get("/watchlist-views")
async def list_watchlist_views(user: AuthUser = Depends(require_auth)):
    _require_b2b_product(user)
    pool = _pool_or_503()
    rows = await pool.fetch(
        """
        SELECT id, name, vendor_names, category, source, min_urgency,
               include_stale, named_accounts_only, changed_wedges_only,
               vendor_alert_threshold, account_alert_threshold, stale_days_threshold,
               alert_email_enabled, alert_delivery_frequency, next_alert_delivery_at,
               last_alert_delivery_at, last_alert_delivery_status, last_alert_delivery_summary,
               created_at, updated_at
        FROM b2b_watchlist_views
        WHERE account_id = $1
        ORDER BY updated_at DESC, created_at DESC, name ASC
        """,
        _uuid.UUID(user.account_id),
    )
    views = [_watchlist_view_payload(row) for row in rows]
    return {"views": views, "count": len(views)}


@router.post("/watchlist-views", status_code=201)
async def create_watchlist_view(
    req: WatchlistViewRequest,
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    account_id = _uuid.UUID(user.account_id)
    name = _clean_required_watchlist_view_name(req.name)
    alert_delivery_frequency = _clean_watchlist_alert_delivery_frequency(req.alert_delivery_frequency)
    pool = _pool_or_503()
    existing = await pool.fetchval(
        """
        SELECT 1
        FROM b2b_watchlist_views
        WHERE account_id = $1
          AND LOWER(name) = LOWER($2)
        LIMIT 1
        """,
        account_id,
        name,
    )
    if existing:
        raise HTTPException(status_code=409, detail="Saved view name already exists")
    vendor_names = await _resolve_tracked_vendors_for_view(pool, account_id, _merge_vendor_names_from_request(req))
    now = datetime.now(timezone.utc)
    next_alert_delivery_at = watchlist_alert_service.resolve_watchlist_alert_next_delivery_at(
        existing_row=None,
        alert_email_enabled=req.alert_email_enabled,
        alert_delivery_frequency=alert_delivery_frequency,
        now=now,
    )
    row = await pool.fetchrow(
        """
        INSERT INTO b2b_watchlist_views (
            id, account_id, name, vendor_names, category, source, min_urgency,
            include_stale, named_accounts_only, changed_wedges_only,
            vendor_alert_threshold, account_alert_threshold, stale_days_threshold,
            alert_email_enabled, alert_delivery_frequency, next_alert_delivery_at,
            created_at, updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
            $14, $15, $16, $17, $17
        )
        RETURNING id, name, vendor_names, category, source, min_urgency,
                  include_stale, named_accounts_only, changed_wedges_only,
                  vendor_alert_threshold, account_alert_threshold, stale_days_threshold,
                  alert_email_enabled, alert_delivery_frequency, next_alert_delivery_at,
                  last_alert_delivery_at, last_alert_delivery_status, last_alert_delivery_summary,
                  created_at, updated_at
        """,
        _uuid.uuid4(),
        account_id,
        name,
        vendor_names,
        _clean_optional_text(req.category),
        _clean_optional_text(req.source),
        req.min_urgency,
        req.include_stale,
        req.named_accounts_only,
        req.changed_wedges_only,
        req.vendor_alert_threshold,
        req.account_alert_threshold,
        req.stale_days_threshold,
        req.alert_email_enabled,
        alert_delivery_frequency,
        next_alert_delivery_at,
        now,
    )
    return _watchlist_view_payload(row)


@router.put("/watchlist-views/{view_id}")
async def update_watchlist_view(
    view_id: _uuid.UUID,
    req: WatchlistViewRequest,
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    name = _clean_required_watchlist_view_name(req.name)
    request_fields = set(getattr(req, "model_fields_set", set()))
    requested_alert_delivery_frequency = _clean_watchlist_alert_delivery_frequency(req.alert_delivery_frequency)
    pool = _pool_or_503()
    account_id = _uuid.UUID(user.account_id)
    existing = await pool.fetchrow(
        """
        SELECT id, name, vendor_names, category, source, min_urgency,
               include_stale, named_accounts_only, changed_wedges_only,
               vendor_alert_threshold, account_alert_threshold, stale_days_threshold,
               alert_email_enabled, alert_delivery_frequency, next_alert_delivery_at
        FROM b2b_watchlist_views
        WHERE id = $1
          AND account_id = $2
        """,
        view_id,
        account_id,
    )
    if not existing:
        raise HTTPException(status_code=404, detail="Saved view not found")
    duplicate = await pool.fetchval(
        """
        SELECT 1
        FROM b2b_watchlist_views
        WHERE account_id = $1
          AND LOWER(name) = LOWER($2)
          AND id <> $3
        LIMIT 1
        """,
        account_id,
        name,
        view_id,
    )
    if duplicate:
        raise HTTPException(status_code=409, detail="Saved view name already exists")
    vendor_names = await _resolve_tracked_vendors_for_view(
        pool,
        account_id,
        _merge_vendor_names_from_request(req)
        if "vendor_names" in request_fields or "vendor_name" in request_fields
        else list(existing["vendor_names"] or []) or None,
    )
    now = datetime.now(timezone.utc)
    category = (
        _clean_optional_text(req.category)
        if "category" in request_fields
        else _clean_optional_text(existing["category"])
    )
    source = (
        _clean_optional_text(req.source)
        if "source" in request_fields
        else _clean_optional_text(existing["source"])
    )
    min_urgency = req.min_urgency if "min_urgency" in request_fields else existing["min_urgency"]
    include_stale = req.include_stale if "include_stale" in request_fields else bool(existing["include_stale"])
    named_accounts_only = (
        req.named_accounts_only
        if "named_accounts_only" in request_fields
        else bool(existing["named_accounts_only"])
    )
    changed_wedges_only = (
        req.changed_wedges_only
        if "changed_wedges_only" in request_fields
        else bool(existing["changed_wedges_only"])
    )
    vendor_alert_threshold = (
        req.vendor_alert_threshold
        if "vendor_alert_threshold" in request_fields
        else existing["vendor_alert_threshold"]
    )
    account_alert_threshold = (
        req.account_alert_threshold
        if "account_alert_threshold" in request_fields
        else existing["account_alert_threshold"]
    )
    stale_days_threshold = (
        req.stale_days_threshold
        if "stale_days_threshold" in request_fields
        else existing["stale_days_threshold"]
    )
    alert_delivery_enabled = (
        bool(req.alert_email_enabled)
        if "alert_email_enabled" in request_fields
        else bool(existing["alert_email_enabled"])
    )
    alert_delivery_frequency = (
        requested_alert_delivery_frequency
        if "alert_delivery_frequency" in request_fields
        else watchlist_alert_service.clean_watchlist_alert_delivery_frequency(existing["alert_delivery_frequency"])
    )
    next_alert_delivery_at = watchlist_alert_service.resolve_watchlist_alert_next_delivery_at(
        existing_row=existing,
        alert_email_enabled=alert_delivery_enabled,
        alert_delivery_frequency=alert_delivery_frequency,
        now=now,
    )
    row = await pool.fetchrow(
        """
        UPDATE b2b_watchlist_views
        SET name = $3,
            vendor_names = $4,
            category = $5,
            source = $6,
            min_urgency = $7,
            include_stale = $8,
            named_accounts_only = $9,
            changed_wedges_only = $10,
            vendor_alert_threshold = $11,
            account_alert_threshold = $12,
            stale_days_threshold = $13,
            alert_email_enabled = $14,
            alert_delivery_frequency = $15,
            next_alert_delivery_at = $16,
            updated_at = $17
        WHERE id = $1
          AND account_id = $2
        RETURNING id, name, vendor_names, category, source, min_urgency,
                  include_stale, named_accounts_only, changed_wedges_only,
                  vendor_alert_threshold, account_alert_threshold, stale_days_threshold,
                  alert_email_enabled, alert_delivery_frequency, next_alert_delivery_at,
                  last_alert_delivery_at, last_alert_delivery_status, last_alert_delivery_summary,
                  created_at, updated_at
        """,
        view_id,
        account_id,
        name,
        vendor_names,
        category,
        source,
        min_urgency,
        include_stale,
        named_accounts_only,
        changed_wedges_only,
        vendor_alert_threshold,
        account_alert_threshold,
        stale_days_threshold,
        alert_delivery_enabled,
        alert_delivery_frequency,
        next_alert_delivery_at,
        now,
    )
    return _watchlist_view_payload(row)


@router.delete("/watchlist-views/{view_id}")
async def delete_watchlist_view(
    view_id: _uuid.UUID,
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    pool = _pool_or_503()
    deleted_row = await pool.fetchrow(
        """
        DELETE FROM b2b_watchlist_views
        WHERE id = $1
          AND account_id = $2
        RETURNING id
        """,
        view_id,
        _uuid.UUID(user.account_id),
    )
    if not deleted_row:
        raise HTTPException(status_code=404, detail="Saved view not found")
    return {"deleted": True, "watchlist_view_id": str(view_id)}


@router.get("/watchlist-views/{view_id}/alert-events")
async def list_watchlist_alert_events(
    view_id: _uuid.UUID,
    status: str = Query("open"),
    limit: int = Query(25, ge=1, le=200),
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    status_text = _clean_optional_text(status)
    status_value = (status_text or "open").lower()
    if status_value not in {"open", "resolved", "all"}:
        raise HTTPException(status_code=422, detail="status must be one of: open, resolved, all")

    pool = _pool_or_503()
    account_id = _uuid.UUID(user.account_id)
    view_row = await _fetch_watchlist_view_row(pool, account_id=account_id, view_id=view_id)
    if not view_row:
        raise HTTPException(status_code=404, detail="Saved view not found")

    params: list[Any] = [account_id, view_id]
    status_clause = ""
    if status_value != "all":
        params.append(status_value)
        status_clause = f"AND status = ${len(params)}"
    params.append(limit)
    rows = await pool.fetch(
        f"""
        SELECT id, watchlist_view_id, event_type, threshold_field, entity_type, entity_key,
               vendor_name, company_name, category, source, threshold_value, summary,
               payload, status, first_seen_at, last_seen_at, resolved_at, created_at, updated_at
        FROM b2b_watchlist_alert_events
        WHERE account_id = $1
          AND watchlist_view_id = $2
          {status_clause}
        ORDER BY CASE WHEN status = 'open' THEN 0 ELSE 1 END,
                 last_seen_at DESC,
                 created_at DESC
        LIMIT ${len(params)}
        """,
        *params,
    )
    events = [watchlist_alert_service.serialize_watchlist_alert_event(row) for row in rows]
    return {
        "watchlist_view_id": str(view_id),
        "watchlist_view_name": view_row["name"],
        "status": status_value,
        "events": events,
        "count": len(events),
    }

@router.post("/watchlist-views/{view_id}/alert-events/evaluate")
async def evaluate_watchlist_alert_events(
    view_id: _uuid.UUID,
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    pool = _pool_or_503()
    account_id = _uuid.UUID(user.account_id)
    view_row = await _fetch_watchlist_view_row(pool, account_id=account_id, view_id=view_id)
    if not view_row:
        raise HTTPException(status_code=404, detail="Saved view not found")
    return await watchlist_alert_service.evaluate_watchlist_alert_events_for_view(
        pool,
        account_id=account_id,
        view_id=view_id,
        view_row=view_row,
        user=user,
        slow_burn_loader=list_tenant_slow_burn_watchlist,
        accounts_loader=list_tenant_accounts_in_motion_feed,
    )


@router.get("/watchlist-views/{view_id}/alert-email-log")
async def list_watchlist_alert_email_log(
    view_id: _uuid.UUID,
    limit: int = Query(10, ge=1, le=100),
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    pool = _pool_or_503()
    account_id = _uuid.UUID(user.account_id)
    view_row = await _fetch_watchlist_view_row(pool, account_id=account_id, view_id=view_id)
    if not view_row:
        raise HTTPException(status_code=404, detail="Saved view not found")
    rows = await pool.fetch(
        """
        SELECT id, recipient_emails, message_ids, event_count, status, summary, error,
               delivered_at, created_at, updated_at, scheduled_for, delivery_frequency, delivery_mode
        FROM b2b_watchlist_alert_email_log
        WHERE account_id = $1
          AND watchlist_view_id = $2
        ORDER BY created_at DESC
        LIMIT $3
        """,
        account_id,
        view_id,
        limit,
    )
    deliveries = [
        {
            "id": str(_row_value(row, "id")),
            "recipient_emails": list(_row_value(row, "recipient_emails") or []),
            "message_ids": list(_row_value(row, "message_ids") or []),
            "event_count": int(_row_value(row, "event_count") or 0),
            "status": _row_value(row, "status"),
            "summary": _row_value(row, "summary"),
            "error": _row_value(row, "error"),
            "delivered_at": str(_row_value(row, "delivered_at")) if _row_value(row, "delivered_at") else None,
            "created_at": str(_row_value(row, "created_at")) if _row_value(row, "created_at") else None,
            "updated_at": str(_row_value(row, "updated_at")) if _row_value(row, "updated_at") else None,
            "scheduled_for": str(_row_value(row, "scheduled_for")) if _row_value(row, "scheduled_for") else None,
            "delivery_frequency": _row_value(row, "delivery_frequency"),
            "delivery_mode": _row_value(row, "delivery_mode"),
        }
        for row in rows
    ]
    return {
        "watchlist_view_id": str(view_id),
        "watchlist_view_name": view_row["name"],
        "deliveries": deliveries,
        "count": len(deliveries),
    }


@router.post("/watchlist-views/{view_id}/alert-events/deliver-email")
async def deliver_watchlist_alert_email(
    view_id: _uuid.UUID,
    body: WatchlistAlertEmailRequest,
    user: AuthUser = Depends(require_auth),
):
    _require_b2b_product(user)
    pool = _pool_or_503()
    account_id = _uuid.UUID(user.account_id)
    view_row = await _fetch_watchlist_view_row(pool, account_id=account_id, view_id=view_id)
    if not view_row:
        raise HTTPException(status_code=404, detail="Saved view not found")

    if body.evaluate_before_send:
        evaluation = await evaluate_watchlist_alert_events(view_id=view_id, user=user)
        events = evaluation["events"]
        suppressed_preview_summary = evaluation.get("suppressed_preview_summary")
    else:
        existing = await list_watchlist_alert_events(view_id=view_id, status="open", limit=25, user=user)
        events = existing["events"]
        suppressed_preview_summary = None

    now = datetime.now(timezone.utc)
    event_ids = [_uuid.UUID(str(event["id"])) for event in events]
    event_count = len(events)
    recipients = await watchlist_alert_service.resolve_watchlist_alert_recipients(pool, account_id=account_id)
    recipient_emails = [item["email"] for item in recipients]
    account_name = await pool.fetchval(
        "SELECT name FROM saas_accounts WHERE id = $1",
        account_id,
    ) or "your account"

    if not events:
        summary = (
            "No open alert events to deliver"
            f"{watchlist_alert_service.suppressed_preview_summary_suffix(suppressed_preview_summary)}"
        )
        log_id = _uuid.uuid4()
        await watchlist_alert_service.record_watchlist_alert_email_log(
            pool,
            log_id=log_id,
            account_id=account_id,
            view_id=view_id,
            event_ids=event_ids,
            recipient_emails=recipient_emails,
            message_ids=[],
            event_count=0,
            status="no_events",
            summary=summary,
            error=None,
            delivered_at=now,
            recorded_at=now,
        )
        await watchlist_alert_service.update_watchlist_view_delivery_state(
            pool,
            view_id=view_id,
            account_id=account_id,
            delivered_at=now,
            status="no_events",
            summary=summary,
            recorded_at=now,
        )
        return {
            "watchlist_view_id": str(view_id),
            "watchlist_view_name": view_row["name"],
            "status": "no_events",
            "recipient_emails": recipient_emails,
            "event_count": 0,
            "message_ids": [],
            "summary": summary,
            "suppressed_preview_summary": suppressed_preview_summary,
        }

    if not recipient_emails:
        summary = "Watchlist alert email delivery failed before send"
        await watchlist_alert_service.record_watchlist_alert_email_log(
            pool,
            log_id=_uuid.uuid4(),
            account_id=account_id,
            view_id=view_id,
            event_ids=event_ids,
            recipient_emails=[],
            message_ids=[],
            event_count=event_count,
            status="failed",
            summary=summary,
            error="No active owner email is configured for this account",
            delivered_at=None,
            recorded_at=now,
        )
        await watchlist_alert_service.update_watchlist_view_delivery_state(
            pool,
            view_id=view_id,
            account_id=account_id,
            delivered_at=now,
            status="failed",
            summary=summary,
            recorded_at=now,
        )
        raise HTTPException(status_code=422, detail="No active owner email is configured for this account")
    if not watchlist_alert_service.watchlist_alert_sender_configured():
        summary = "Watchlist alert email delivery failed before send"
        await watchlist_alert_service.record_watchlist_alert_email_log(
            pool,
            log_id=_uuid.uuid4(),
            account_id=account_id,
            view_id=view_id,
            event_ids=event_ids,
            recipient_emails=recipient_emails,
            message_ids=[],
            event_count=event_count,
            status="failed",
            summary=summary,
            error="Campaign sender not configured for watchlist alert email delivery",
            delivered_at=None,
            recorded_at=now,
        )
        await watchlist_alert_service.update_watchlist_view_delivery_state(
            pool,
            view_id=view_id,
            account_id=account_id,
            delivered_at=now,
            status="failed",
            summary=summary,
            recorded_at=now,
        )
        raise HTTPException(status_code=503, detail="Campaign sender not configured for watchlist alert email delivery")

    try:
        sender = get_campaign_sender()
    except Exception as exc:
        summary = "Watchlist alert email delivery failed before send"
        await watchlist_alert_service.record_watchlist_alert_email_log(
            pool,
            log_id=_uuid.uuid4(),
            account_id=account_id,
            view_id=view_id,
            event_ids=event_ids,
            recipient_emails=recipient_emails,
            message_ids=[],
            event_count=event_count,
            status="failed",
            summary=summary,
            error=f"Campaign sender unavailable: {exc}",
            delivered_at=None,
            recorded_at=now,
        )
        await watchlist_alert_service.update_watchlist_view_delivery_state(
            pool,
            view_id=view_id,
            account_id=account_id,
            delivered_at=now,
            status="failed",
            summary=summary,
            recorded_at=now,
        )
        raise HTTPException(status_code=503, detail=f"Campaign sender unavailable: {exc}")

    from_addr = watchlist_alert_service.watchlist_alert_from_email()
    if not from_addr:
        summary = "Watchlist alert email delivery failed before send"
        await watchlist_alert_service.record_watchlist_alert_email_log(
            pool,
            log_id=_uuid.uuid4(),
            account_id=account_id,
            view_id=view_id,
            event_ids=event_ids,
            recipient_emails=recipient_emails,
            message_ids=[],
            event_count=event_count,
            status="failed",
            summary=summary,
            error="Campaign sender from-address is not configured",
            delivered_at=None,
            recorded_at=now,
        )
        await watchlist_alert_service.update_watchlist_view_delivery_state(
            pool,
            view_id=view_id,
            account_id=account_id,
            delivered_at=now,
            status="failed",
            summary=summary,
            recorded_at=now,
        )
        raise HTTPException(status_code=503, detail="Campaign sender from-address is not configured")

    summary_line = f"{event_count} open saved-view alert{'s' if event_count != 1 else ''} are ready for review."
    html_body = watchlist_alert_service.render_watchlist_alert_email_html(
        account_name=str(account_name),
        view_name=str(view_row["name"]),
        summary_line=summary_line,
        events=events,
    )
    subject = f"Watchlist alerts: {view_row['name']}"

    delivered_count = 0
    suppressed_count = 0
    message_ids: list[str] = []
    errors: list[str] = []
    for recipient in recipient_emails:
        sent, message_id, send_error, suppressed = await watchlist_alert_service.send_watchlist_alert_email(
            pool,
            sender,
            recipient=recipient,
            from_addr=from_addr,
            subject=subject,
            html_body=html_body,
            view_name=str(view_row["name"]),
        )
        if sent:
            delivered_count += 1
            if message_id:
                message_ids.append(message_id)
        elif suppressed:
            suppressed_count += 1
            if send_error:
                errors.append(send_error)
        elif send_error:
            errors.append(send_error)

    status = "sent"
    if delivered_count == 0:
        status = "failed"
    elif delivered_count < len(recipient_emails):
        status = "partial"
    summary = (
        f"Delivered watchlist alert email to {delivered_count} of {len(recipient_emails)} recipient"
        f"{'' if len(recipient_emails) == 1 else 's'}"
    )
    if suppressed_count > 0:
        summary += f" ({suppressed_count} suppressed)"
    error_text = "; ".join(errors) if errors else None
    delivered_at = now if delivered_count > 0 else None
    await watchlist_alert_service.record_watchlist_alert_email_log(
        pool,
        log_id=_uuid.uuid4(),
        account_id=account_id,
        view_id=view_id,
        event_ids=event_ids,
        recipient_emails=recipient_emails,
        message_ids=message_ids,
        event_count=event_count,
        status=status,
        summary=summary,
        error=error_text,
        delivered_at=delivered_at,
        recorded_at=now,
    )
    await watchlist_alert_service.update_watchlist_view_delivery_state(
        pool,
        view_id=view_id,
        account_id=account_id,
        delivered_at=delivered_at or now,
        status=status,
        summary=summary,
        recorded_at=now,
    )
    return {
        "watchlist_view_id": str(view_id),
        "watchlist_view_name": view_row["name"],
        "status": status,
        "recipient_emails": recipient_emails,
        "event_count": event_count,
        "message_ids": message_ids,
        "summary": summary,
        "error": error_text,
        "suppressed_preview_summary": suppressed_preview_summary,
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
    vname = _clean_required_text(vendor_name, "vendor_name")
    pool = _pool_or_503()

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
    vname = _clean_required_text(vendor_name, "vendor_name")
    pool = _pool_or_503()

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
    conditions.append(_canonical_review_predicate())
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
    conditions = ["enrichment_status = 'enriched'", _canonical_review_predicate()]
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
        review_scope = (
            "WHERE vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = $1)"
            f" AND {_canonical_review_predicate()}"
        )
        scrape_scope = "WHERE vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = $1)"
        scope_params = t_params
    else:
        review_scope = f"WHERE {_canonical_review_predicate()}"

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
    vendor_name = _clean_optional_text(vendor_name)
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
    company_name = _clean_required_text(company, "company")
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
        company=company_name,
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
        "company": company_name,
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
    primary_vendor = _clean_required_text(body.primary_vendor, "primary_vendor")
    comparison_vendor = _clean_required_text(body.comparison_vendor, "comparison_vendor")
    if primary_vendor.lower() == comparison_vendor.lower():
        raise HTTPException(status_code=400, detail="Choose two different vendors")
    pool = _pool_or_503()

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
    primary_company = _clean_required_text(body.primary_company, "primary_company")
    comparison_company = _clean_required_text(body.comparison_company, "comparison_company")
    if primary_company.lower() == comparison_company.lower():
        raise HTTPException(status_code=400, detail="Choose two different companies")
    pool = _pool_or_503()

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
    company_name = _clean_required_text(body.company_name, "company_name")
    pool = _pool_or_503()

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


@router.post("/reports/battle-card")
async def generate_tenant_battle_card_report(
    body: BattleCardRequest,
    user: AuthUser = Depends(require_auth),
):
    """Return a persisted battle card artifact for a tracked vendor."""
    _require_b2b_product(user)
    vendor_name = _clean_required_text(body.vendor_name, "vendor_name")
    pool = _pool_or_503()

    tracked_vendor = vendor_name
    if settings.saas_auth.enabled and not _is_admin_user(user):
        tracked_vendor = await _resolve_tracked_vendor_for_view(
            pool,
            _uuid.UUID(user.account_id),
            vendor_name,
        )
        if not tracked_vendor:
            raise HTTPException(status_code=403, detail="Vendor must be in your tracked vendor list")

    existing = await _fetch_latest_tenant_vendor_report(
        pool,
        report_type="battle_card",
        vendor_name=tracked_vendor,
    )
    if existing and not body.refresh:
        return {
            "status": "ready",
            "report_id": str(existing["id"]),
            "vendor_name": tracked_vendor,
            "reused": True,
            "message": "Using the latest persisted battle card.",
        }

    task_repo = get_scheduled_task_repo()
    battle_card_task = await task_repo.get_by_name("b2b_battle_cards")
    if not battle_card_task:
        raise HTTPException(status_code=503, detail="b2b_battle_cards task is not registered")

    battle_card_task.metadata = {
        **(battle_card_task.metadata or {}),
        "scope_name": tracked_vendor,
        "scope_trigger": "tenant_manual_request",
        "test_vendors": [tracked_vendor],
    }
    scheduler = get_task_scheduler()
    run_result = await scheduler.run_now(battle_card_task)

    latest = await _fetch_latest_tenant_vendor_report(
        pool,
        report_type="battle_card",
        vendor_name=tracked_vendor,
    )
    if not latest:
        detail = str((run_result or {}).get("_skip_synthesis") or "").strip()
        raise HTTPException(
            status_code=404,
            detail=detail or f"No persisted battle card is available for {tracked_vendor}",
        )

    return {
        "status": "ready",
        "report_id": str(latest["id"]),
        "vendor_name": tracked_vendor,
        "reused": False,
        "message": "Battle card ready.",
    }


@router.get("/reports")
async def list_tenant_reports(
    report_type: Optional[str] = Query(None),
    vendor_filter: Optional[str] = Query(None),
    quality_status: Optional[str] = None,
    freshness_state: Optional[str] = None,
    review_state: Optional[str] = None,
    include_stale: bool = Query(False),
    limit: int = Query(10, ge=1, le=200),
    user: AuthUser = Depends(require_auth),
):
    """Reports scoped to tracked vendors."""
    _require_b2b_product(user)
    report_type = _clean_optional_text(report_type)
    vendor_filter = _clean_optional_text(vendor_filter)
    normalized_quality_status = _normalize_report_list_filter("quality_status", quality_status)
    normalized_freshness_state = _normalize_report_list_filter("freshness_state", freshness_state)
    normalized_review_state = _normalize_report_list_filter("review_state", review_state)
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
    scan_limit = capped
    if normalized_quality_status or normalized_freshness_state or normalized_review_state:
        scan_limit = min(max(capped * 5, capped), 200)
    params.append(scan_limit)
    report_subscription_account_idx = idx + 1
    params.append(user.account_id)

    rows = await pool.fetch(
        f"""
        SELECT b2b_intelligence.id, report_date, report_type, executive_summary,
               vendor_filter, category_filter, status, b2b_intelligence.created_at,
               intelligence_data,
               latest_failure_step, latest_error_code, latest_error_summary,
               COALESCE((intelligence_data->>'data_stale')::boolean, false) AS data_stale,
               intelligence_data->>'data_as_of_date' AS evidence_data_as_of_date,
               intelligence_data->>'as_of_date' AS evidence_as_of_date,
               intelligence_data->>'report_date' AS evidence_report_date,
               intelligence_data->>'analysis_window_days' AS evidence_analysis_window_days,
               intelligence_data->>'window_days' AS evidence_window_days,
               intelligence_data->>'evidence_window_days' AS evidence_fallback_window_days,
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
               END AS quality_score,
               rs.id AS report_subscription_id,
               rs.scope_type AS report_subscription_scope_type,
               rs.scope_key AS report_subscription_scope_key,
               rs.scope_label AS report_subscription_scope_label,
               rs.enabled AS report_subscription_enabled
        FROM b2b_intelligence
        LEFT JOIN LATERAL (
            SELECT id, scope_type, scope_key, scope_label, enabled
            FROM b2b_report_subscriptions
            WHERE account_id = ${report_subscription_account_idx}::uuid
              AND scope_type = 'report'
              AND scope_key = b2b_intelligence.id::text
            ORDER BY updated_at DESC NULLS LAST, id DESC
            LIMIT 1
        ) rs ON TRUE
        {where}
        ORDER BY report_date DESC NULLS LAST, b2b_intelligence.created_at DESC NULLS LAST, b2b_intelligence.id DESC
        LIMIT ${idx}
        """,
        *params,
    )

    if normalized_quality_status or normalized_freshness_state or normalized_review_state:
        rows = [
            row for row in rows
            if _report_matches_list_filters(
                row,
                quality_status=normalized_quality_status,
                freshness_state=normalized_freshness_state,
                review_state=normalized_review_state,
            )
        ][:capped]

    reports = []
    for r in rows:
        blocker_count = r["blocker_count"] or 0
        warning_count = r["warning_count"] or 0
        unresolved_issue_count = r["unresolved_issue_count"] or 0
        preview_fields = _extract_report_account_preview_fields(r.get("intelligence_data"))
        evidence_snapshot = report_evidence_snapshot_payload(
            report_date=r["report_date"],
            intelligence_data={
                "data_as_of_date": r.get("evidence_data_as_of_date"),
                "as_of_date": r.get("evidence_as_of_date"),
                "report_date": r.get("evidence_report_date"),
                "analysis_window_days": r.get("evidence_analysis_window_days"),
                "window_days": r.get("evidence_window_days"),
                "evidence_window_days": r.get("evidence_fallback_window_days"),
            },
        )
        report_subscription = None
        if r["report_subscription_id"]:
            report_subscription = {
                "id": str(r["report_subscription_id"]),
                "scope_type": r["report_subscription_scope_type"],
                "scope_key": r["report_subscription_scope_key"],
                "scope_label": r["report_subscription_scope_label"],
                "enabled": bool(r["report_subscription_enabled"]),
            }
        trust = report_trust_payload(
            report_date=r["report_date"],
            created_at=r["created_at"],
            data_stale=bool(r["data_stale"]) if "data_stale" in r else False,
            blocker_count=blocker_count,
            warning_count=warning_count,
            unresolved_issue_count=unresolved_issue_count,
            status=r["status"],
        )
        reports.append(
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
                "blocker_count": blocker_count,
                "warning_count": warning_count,
                "unresolved_issue_count": unresolved_issue_count,
                "quality_status": r["quality_status"],
                "quality_score": r["quality_score"],
                "report_subscription": report_subscription,
                "has_pdf_export": trust["artifact_state"] == "ready",
                "artifact_state": trust["artifact_state"],
                "artifact_label": trust["artifact_label"],
                "freshness_state": trust["freshness_state"],
                "freshness_label": trust["freshness_label"],
                "review_state": trust["review_state"],
                "review_label": trust["review_label"],
                "trust": trust,
                "as_of_date": evidence_snapshot["as_of_date"],
                "analysis_window_days": evidence_snapshot["analysis_window_days"],
                "created_at": str(r["created_at"]) if r["created_at"] else None,
                **preview_fields,
            }
        )

    return {"reports": reports, "count": len(reports)}


@router.get("/reports/{report_id}")
async def get_tenant_report(report_id: str, user: AuthUser = Depends(require_auth)):
    """Report detail (verify vendor in tracked)."""
    _require_b2b_product(user)
    report_id = _clean_optional_text(report_id)
    try:
        rid = _uuid.UUID(report_id or "")
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
    data_density = _safe_json(row["data_density"])
    if isinstance(intelligence_data, dict):
        quality = _safe_json(intelligence_data.get("battle_card_quality"))
        quality_status = intelligence_data.get("quality_status")
        if not quality_status and isinstance(quality, dict):
            quality_status = quality.get("status")
        if isinstance(quality, dict):
            quality_score = quality.get("score")
    section_evidence = report_section_evidence_payload(intelligence_data)
    evidence_snapshot = report_evidence_snapshot_payload(
        report_date=row["report_date"],
        intelligence_data=intelligence_data,
    )
    data_stale = False
    if isinstance(intelligence_data, dict):
        data_stale = bool(intelligence_data.get("data_stale") is True)
    blocker_count = row["blocker_count"] or 0
    warning_count = row["warning_count"] or 0
    open_issue_count = unresolved_issue_count or 0
    trust = report_trust_payload(
        report_date=row["report_date"],
        created_at=row["created_at"],
        data_stale=data_stale,
        blocker_count=blocker_count,
        warning_count=warning_count,
        unresolved_issue_count=open_issue_count,
        status=row["status"],
    )
    subscription_row = await _fetch_report_subscription_row(
        pool,
        account_id=user.account_id,
        scope_type="report",
        scope_key=str(row["id"]),
    )
    report_subscription = None
    if subscription_row:
        report_subscription = {
            "id": str(subscription_row["id"]),
            "scope_type": subscription_row["scope_type"],
            "scope_key": subscription_row["scope_key"],
            "scope_label": subscription_row["scope_label"],
            "enabled": bool(subscription_row["enabled"]),
        }

    return {
        "id": str(row["id"]),
        "report_date": str(row["report_date"]) if row["report_date"] else None,
        "report_type": row["report_type"],
        "vendor_filter": row["vendor_filter"],
        "category_filter": row["category_filter"],
        "executive_summary": row["executive_summary"],
        "intelligence_data": intelligence_data,
        "as_of_date": evidence_snapshot["as_of_date"],
        "analysis_window_days": evidence_snapshot["analysis_window_days"],
        "section_evidence": section_evidence,
        "data_density": data_density,
        "status": row["status"],
        "latest_failure_step": row["latest_failure_step"],
        "latest_error_code": row["latest_error_code"],
        "latest_error_summary": row["latest_error_summary"],
        "blocker_count": blocker_count,
        "warning_count": warning_count,
        "unresolved_issue_count": open_issue_count,
        "quality_status": quality_status,
        "quality_score": quality_score,
        "report_subscription": report_subscription,
        "has_pdf_export": trust["artifact_state"] == "ready",
        "artifact_state": trust["artifact_state"],
        "artifact_label": trust["artifact_label"],
        "freshness_state": trust["freshness_state"],
        "freshness_label": trust["freshness_label"],
        "review_state": trust["review_state"],
        "review_label": trust["review_label"],
        "trust": trust,
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
    normalized_scope_type, normalized_scope_key = _normalize_report_subscription_scope(
        scope_type,
        scope_key,
    )

    report_id: _uuid.UUID | None = None
    if normalized_scope_type == "report":
        report_id = _coerce_uuid(normalized_scope_key)
        if not report_id:
            raise HTTPException(status_code=400, detail="scope_key must be a report UUID")

    pool = _pool_or_503()
    if report_id:
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
    scope_label = _clean_required_text(body.scope_label, "scope_label")
    delivery_note = _clean_optional_text(body.delivery_note)
    normalized_scope_type, normalized_scope_key = _normalize_report_subscription_scope(
        scope_type,
        scope_key,
    )
    normalized_recipients = _normalize_report_subscription_recipients(body.recipients)
    if body.enabled and not normalized_recipients:
        raise HTTPException(
            status_code=400,
            detail="Add at least one recipient before enabling recurring delivery",
        )

    normalized_filter_payload = _normalize_report_subscription_filter_payload(
        normalized_scope_type,
        body.filter_payload,
    )

    report_id: _uuid.UUID | None = None
    if normalized_scope_type == "report":
        report_id = _coerce_uuid(normalized_scope_key)
        if not report_id:
            raise HTTPException(status_code=400, detail="scope_key must be a report UUID")

    pool = _pool_or_503()
    account_id = _uuid.UUID(user.account_id)
    user_id = _coerce_uuid(getattr(user, "user_id", None))

    if report_id:
        await _load_accessible_tenant_report(pool, report_id, user)

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
            filter_payload,
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
            $6::jsonb,
            $7,
            $8,
            $9,
            $10::text[],
            $11,
            $12,
            $13,
            $14::uuid,
            $14::uuid
        )
        ON CONFLICT (account_id, scope_type, scope_key)
        DO UPDATE
        SET report_id = EXCLUDED.report_id,
            scope_label = EXCLUDED.scope_label,
            filter_payload = EXCLUDED.filter_payload,
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
        json.dumps(normalized_filter_payload),
        body.delivery_frequency,
        body.deliverable_focus,
        body.freshness_policy,
        normalized_recipients,
        delivery_note,
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
    pain_category = _clean_optional_text(pain_category)
    company = _clean_optional_text(company)
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
    review_id = _clean_optional_text(review_id)
    try:
        rid = _uuid.UUID(review_id or "")
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
    status = _clean_optional_text(status)
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
    vname = _clean_required_text(req.vendor_name, "vendor_name")
    company_filter = _clean_optional_text(req.company_filter)
    pool = _pool_or_503()

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
            company_filter=company_filter,
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


# ---------------------------------------------------------------------------
# Opportunity dispositions (4 endpoints)
# ---------------------------------------------------------------------------


def _disposition_payload(row) -> dict:
    return {
        "id": str(row["id"]),
        "opportunity_key": row["opportunity_key"],
        "company": row["company"],
        "vendor": row["vendor"],
        "review_id": row["review_id"],
        "disposition": row["disposition"],
        "snoozed_until": row["snoozed_until"].isoformat() if row["snoozed_until"] else None,
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


def _validate_disposition(disposition: str, snoozed_until: str | None) -> datetime | None:
    if disposition not in _VALID_DISPOSITIONS:
        raise HTTPException(
            status_code=422,
            detail=f"disposition must be one of: {', '.join(sorted(_VALID_DISPOSITIONS))}",
        )
    parsed_snooze: datetime | None = None
    if disposition == "snoozed":
        if not snoozed_until:
            raise HTTPException(status_code=422, detail="snoozed_until is required when disposition=snoozed")
        try:
            parsed_snooze = datetime.fromisoformat(snoozed_until)
            if parsed_snooze.tzinfo is None:
                parsed_snooze = parsed_snooze.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail="snoozed_until must be a valid ISO 8601 timestamp")
    return parsed_snooze


@router.get("/opportunity-dispositions")
async def list_opportunity_dispositions(
    disposition: str | None = Query(None, description="Filter: snoozed, dismissed, saved"),
    user: AuthUser = Depends(require_auth),
):
    """List persisted opportunity dispositions for this account."""
    _require_b2b_product(user)
    disposition_value = _clean_optional_text(disposition)
    if disposition_value and disposition_value not in _VALID_DISPOSITIONS:
        raise HTTPException(status_code=422, detail=f"disposition must be one of: {', '.join(sorted(_VALID_DISPOSITIONS))}")

    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)

    # Expire stale snoozes
    await pool.execute(
        "DELETE FROM b2b_opportunity_dispositions WHERE account_id = $1 AND disposition = 'snoozed' AND snoozed_until <= NOW()",
        acct,
    )

    if disposition_value:
        rows = await pool.fetch(
            """
            SELECT id, opportunity_key, company, vendor, review_id,
                   disposition, snoozed_until, created_at, updated_at
            FROM b2b_opportunity_dispositions
            WHERE account_id = $1 AND disposition = $2
            ORDER BY updated_at DESC
            """,
            acct,
            disposition_value,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, opportunity_key, company, vendor, review_id,
                   disposition, snoozed_until, created_at, updated_at
            FROM b2b_opportunity_dispositions
            WHERE account_id = $1
            ORDER BY updated_at DESC
            """,
            acct,
        )

    dispositions = [_disposition_payload(r) for r in rows]
    return {"dispositions": dispositions, "count": len(dispositions)}


@router.post("/opportunity-dispositions", status_code=200)
async def set_opportunity_disposition(
    req: SetDispositionRequest,
    user: AuthUser = Depends(require_auth),
):
    """Upsert a single opportunity disposition (save, snooze, or dismiss)."""
    _require_b2b_product(user)
    parsed_snooze = _validate_disposition(req.disposition, req.snoozed_until)
    opportunity_key = _clean_required_text(req.opportunity_key, "opportunity_key")
    company = _clean_required_text(req.company, "company")
    vendor = _clean_required_text(req.vendor, "vendor")
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)
    now = datetime.now(timezone.utc)

    row = await pool.fetchrow(
        """
        INSERT INTO b2b_opportunity_dispositions
            (id, account_id, opportunity_key, company, vendor, review_id,
             disposition, snoozed_until, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)
        ON CONFLICT (account_id, opportunity_key) DO UPDATE SET
            disposition = EXCLUDED.disposition,
            snoozed_until = EXCLUDED.snoozed_until,
            updated_at = NOW()
        RETURNING id, opportunity_key, company, vendor, review_id,
                  disposition, snoozed_until, created_at, updated_at
        """,
        _uuid.uuid4(),
        acct,
        opportunity_key,
        company,
        vendor,
        _clean_optional_text(req.review_id),
        req.disposition,
        parsed_snooze,
        now,
    )
    return _disposition_payload(row)


@router.post("/opportunity-dispositions/bulk", status_code=200)
async def bulk_set_opportunity_dispositions(
    req: BulkSetDispositionRequest,
    user: AuthUser = Depends(require_auth),
):
    """Bulk upsert opportunity dispositions."""
    _require_b2b_product(user)
    parsed_snooze = _validate_disposition(req.disposition, req.snoozed_until)
    normalized_items: list[tuple[str, str, str, str | None]] = []
    for item in req.items:
        normalized_items.append((
            _clean_required_text(item.opportunity_key, "opportunity_key"),
            _clean_required_text(item.company, "company"),
            _clean_required_text(item.vendor, "vendor"),
            _clean_optional_text(item.review_id),
        ))
    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)
    now = datetime.now(timezone.utc)

    updated = 0
    for opportunity_key, company, vendor, review_id in normalized_items:
        await pool.execute(
            """
            INSERT INTO b2b_opportunity_dispositions
                (id, account_id, opportunity_key, company, vendor, review_id,
                 disposition, snoozed_until, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)
            ON CONFLICT (account_id, opportunity_key) DO UPDATE SET
                disposition = EXCLUDED.disposition,
                snoozed_until = EXCLUDED.snoozed_until,
                updated_at = NOW()
            """,
            _uuid.uuid4(),
            acct,
            opportunity_key,
            company,
            vendor,
            review_id,
            req.disposition,
            parsed_snooze,
            now,
        )
        updated += 1

    return {"updated": updated}


@router.post("/opportunity-dispositions/remove", status_code=200)
async def remove_opportunity_dispositions(
    req: RemoveDispositionsRequest,
    user: AuthUser = Depends(require_auth),
):
    """Remove dispositions (restore opportunities to active)."""
    _require_b2b_product(user)
    keys: list[str] = []
    for key in req.opportunity_keys:
        cleaned = _clean_optional_text(key)
        if cleaned:
            keys.append(cleaned)
    if not keys:
        return {"removed": 0}

    pool = _pool_or_503()
    acct = _uuid.UUID(user.account_id)

    result = await pool.execute(
        """
        DELETE FROM b2b_opportunity_dispositions
        WHERE account_id = $1
          AND opportunity_key = ANY($2::text[])
        """,
        acct,
        keys,
    )
    # asyncpg returns "DELETE N"
    removed = int(result.split()[-1]) if result else 0
    return {"removed": removed}


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
