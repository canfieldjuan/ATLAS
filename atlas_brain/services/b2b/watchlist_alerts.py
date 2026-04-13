"""Shared watchlist alert evaluation and delivery helpers."""

from __future__ import annotations

import json
import uuid as _uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from ...api.b2b_dashboard import (
    _accounts_in_motion_alert_basis,
    _normalize_vendor_name,
)
from ...autonomous.tasks.campaign_send import _unsub_headers, _wrap_with_footer
from ...autonomous.tasks.campaign_suppression import is_suppressed
from ...config import settings
from ...templates.email.watchlist_alert_delivery import render_watchlist_alert_delivery_html

_WATCHLIST_ALERT_DELIVERY_FREQUENCIES = {
    "daily": 1,
    "weekly": 7,
}


def _safe_json(val: Any) -> Any:
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return val


def _safe_float(val: Any, default: float | None = None) -> float | None:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _clean_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_text_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    cleaned: list[str] = []
    for item in value:
        text = _clean_optional_text(item)
        if text:
            cleaned.append(text)
    return cleaned


def _watchlist_alert_account_focus_payload(row: dict[str, Any]) -> dict[str, str] | None:
    vendor_name = _clean_optional_text(row.get("vendor"))
    company_name = _clean_optional_text(row.get("company"))
    report_date = _clean_optional_text(row.get("report_date"))
    if not vendor_name or not company_name or not report_date:
        return None
    return {
        "vendor": vendor_name,
        "company": company_name,
        "report_date": report_date,
        "watch_vendor": _clean_optional_text(row.get("watch_vendor")) or vendor_name,
        "category": _clean_optional_text(row.get("category")) or "",
        "track_mode": _clean_optional_text(row.get("track_mode")) or "",
    }


def _normalize_watchlist_alert_account_focus(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    vendor_name = _clean_optional_text(value.get("vendor"))
    company_name = _clean_optional_text(value.get("company"))
    report_date = _clean_optional_text(value.get("report_date"))
    if not vendor_name or not company_name or not report_date:
        return None
    return {
        "vendor": vendor_name,
        "company": company_name,
        "report_date": report_date,
        "watch_vendor": _clean_optional_text(value.get("watch_vendor")) or vendor_name,
        "category": _clean_optional_text(value.get("category")) or "",
        "track_mode": _clean_optional_text(value.get("track_mode")) or "",
    }


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


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        return default


def clean_watchlist_alert_delivery_frequency(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "daily"
    if text not in _WATCHLIST_ALERT_DELIVERY_FREQUENCIES:
        supported = ", ".join(sorted(_WATCHLIST_ALERT_DELIVERY_FREQUENCIES))
        raise ValueError(f"alert_delivery_frequency must be one of: {supported}")
    return text


def watchlist_alert_delivery_interval(frequency: str) -> timedelta:
    return timedelta(days=_WATCHLIST_ALERT_DELIVERY_FREQUENCIES.get(str(frequency or "daily"), 1))


def next_watchlist_alert_delivery_at(
    frequency: str,
    *,
    anchor: datetime | None = None,
    from_now: datetime | None = None,
) -> datetime:
    next_due = _parse_timestamp_value(anchor) or _parse_timestamp_value(from_now) or datetime.now(timezone.utc)
    interval = watchlist_alert_delivery_interval(frequency)
    now = datetime.now(timezone.utc)
    while next_due <= now:
        next_due += interval
    return next_due.replace(microsecond=0)


def resolve_watchlist_alert_next_delivery_at(
    *,
    existing_row: Any,
    alert_email_enabled: bool,
    alert_delivery_frequency: str,
    now: datetime | None = None,
) -> datetime | None:
    if not alert_email_enabled:
        return None
    if not existing_row:
        return next_watchlist_alert_delivery_at(alert_delivery_frequency, from_now=now)
    previous_frequency = clean_watchlist_alert_delivery_frequency(existing_row.get("alert_delivery_frequency"))
    previous_next_delivery_at = _parse_timestamp_value(existing_row.get("next_alert_delivery_at"))
    if not bool(existing_row.get("alert_email_enabled")):
        return next_watchlist_alert_delivery_at(alert_delivery_frequency, from_now=now)
    if previous_frequency != alert_delivery_frequency:
        return next_watchlist_alert_delivery_at(alert_delivery_frequency, from_now=now)
    if previous_next_delivery_at is not None:
        return previous_next_delivery_at
    return next_watchlist_alert_delivery_at(alert_delivery_frequency, from_now=now)


def watchlist_view_payload(row: Any) -> dict[str, Any]:
    preview_alerts_enabled = bool(_row_value(row, "preview_alerts_enabled", True))
    preview_alert_min_confidence_override = _safe_float(
        _row_value(row, "preview_alert_min_confidence")
    )
    raw_preview_alert_require_budget_authority = _row_value(
        row, "preview_alert_require_budget_authority"
    )
    preview_alert_require_budget_authority_override = (
        raw_preview_alert_require_budget_authority
        if isinstance(raw_preview_alert_require_budget_authority, bool)
        else None
    )
    preview_account_alert_policy = {
        "applies_to_preview_only": True,
        "enabled": preview_alerts_enabled,
        "enabled_source": "view",
        "min_confidence": (
            preview_alert_min_confidence_override
            if preview_alert_min_confidence_override is not None
            else float(settings.b2b_churn.accounts_in_motion_preview_alert_min_confidence)
        ),
        "min_confidence_source": (
            "view" if preview_alert_min_confidence_override is not None else "global"
        ),
        "require_budget_authority": (
            preview_alert_require_budget_authority_override
            if preview_alert_require_budget_authority_override is not None
            else bool(settings.b2b_churn.accounts_in_motion_preview_alert_require_budget_authority)
        ),
        "require_budget_authority_source": (
            "view" if preview_alert_require_budget_authority_override is not None else "global"
        ),
        "override_min_confidence": preview_alert_min_confidence_override,
        "override_require_budget_authority": preview_alert_require_budget_authority_override,
    }
    return {
        "id": str(_row_value(row, "id")),
        "name": _row_value(row, "name"),
        "vendor_name": (list(_row_value(row, "vendor_names") or []) or [None])[0],
        "vendor_names": list(_row_value(row, "vendor_names") or []),
        "category": _row_value(row, "category"),
        "source": _row_value(row, "source"),
        "min_urgency": _safe_float(_row_value(row, "min_urgency")),
        "include_stale": bool(_row_value(row, "include_stale")),
        "named_accounts_only": bool(_row_value(row, "named_accounts_only")),
        "changed_wedges_only": bool(_row_value(row, "changed_wedges_only")),
        "vendor_alert_threshold": _safe_float(_row_value(row, "vendor_alert_threshold")),
        "account_alert_threshold": _safe_float(_row_value(row, "account_alert_threshold")),
        "preview_alerts_enabled": preview_alerts_enabled,
        "stale_days_threshold": _coerce_optional_int(_row_value(row, "stale_days_threshold")),
        "alert_email_enabled": bool(_row_value(row, "alert_email_enabled")),
        "alert_delivery_frequency": str(_row_value(row, "alert_delivery_frequency") or "daily"),
        "next_alert_delivery_at": str(_row_value(row, "next_alert_delivery_at")) if _row_value(row, "next_alert_delivery_at") else None,
        "last_alert_delivery_at": str(_row_value(row, "last_alert_delivery_at")) if _row_value(row, "last_alert_delivery_at") else None,
        "last_alert_delivery_status": _row_value(row, "last_alert_delivery_status"),
        "last_alert_delivery_summary": _row_value(row, "last_alert_delivery_summary"),
        "preview_account_alert_policy": preview_account_alert_policy,
        "created_at": str(_row_value(row, "created_at")) if _row_value(row, "created_at") else None,
        "updated_at": str(_row_value(row, "updated_at")) if _row_value(row, "updated_at") else None,
    }


def _suppressed_preview_reason_detail(
    *,
    policy_reason: str,
    preview_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    policy = preview_policy if isinstance(preview_policy, dict) else {}
    if policy_reason == "preview_policy_disabled":
        return {
            "summary": "Preview-backed account alerts are disabled for this view.",
            "short_summary": "preview alerts disabled for this view",
            "enabled": False,
            "enabled_source": policy.get("enabled_source") or "view",
        }
    if policy_reason == "preview_low_confidence":
        min_confidence = _safe_float(policy.get("min_confidence"))
        return {
            "summary": (
                "Preview-backed account alerts require "
                f"confidence >= {min_confidence:.2f}."
                if min_confidence is not None
                else "Preview-backed account alerts require higher confidence."
            ),
            "short_summary": (
                f"confidence >= {min_confidence:.2f} required"
                if min_confidence is not None
                else "higher confidence required"
            ),
            "min_confidence": min_confidence,
            "min_confidence_source": policy.get("min_confidence_source") or "global",
        }
    if policy_reason == "preview_missing_budget_authority":
        return {
            "summary": "Preview-backed account alerts require budget authority.",
            "short_summary": "budget authority required",
            "require_budget_authority": True,
            "require_budget_authority_source": (
                policy.get("require_budget_authority_source") or "global"
            ),
        }
    return {
        "summary": "Preview-backed account alerts were blocked by policy.",
        "short_summary": "blocked by policy",
    }


def serialize_watchlist_alert_event(row: Any) -> dict[str, Any]:
    payload = _row_value(row, "payload")
    payload_json = _safe_json(payload) if payload is not None else {}
    payload_dict = payload_json if isinstance(payload_json, dict) else {}
    account_alert_score = _safe_float(payload_dict.get("account_alert_score"))
    account_alert_score_source = _clean_optional_text(payload_dict.get("account_alert_score_source"))
    if account_alert_score is None:
        account_alert_score = _safe_float(payload_dict.get("urgency"))
        if account_alert_score is not None and not account_alert_score_source:
            account_alert_score_source = "urgency"
    return {
        "id": str(_row_value(row, "id")),
        "watchlist_view_id": str(_row_value(row, "watchlist_view_id")),
        "event_type": _row_value(row, "event_type"),
        "threshold_field": _row_value(row, "threshold_field"),
        "entity_type": _row_value(row, "entity_type"),
        "entity_key": _row_value(row, "entity_key"),
        "vendor_name": _row_value(row, "vendor_name"),
        "company_name": _row_value(row, "company_name"),
        "category": _row_value(row, "category"),
        "source": _row_value(row, "source"),
        "threshold_value": _safe_float(_row_value(row, "threshold_value")),
        "summary": _row_value(row, "summary"),
        "payload": payload_json,
        "reasoning_reference_ids": payload_dict.get("reasoning_reference_ids") if isinstance(payload_dict.get("reasoning_reference_ids"), dict) else None,
        "source_review_ids": _clean_text_list(payload_dict.get("source_review_ids")),
        "account_review_focus": _normalize_watchlist_alert_account_focus(payload_dict.get("account_review_focus")),
        "account_alert_score": account_alert_score,
        "account_alert_score_source": account_alert_score_source,
        "account_alert_eligible": bool(payload_dict.get("account_alert_eligible", True)),
        "account_alert_policy_reason": _clean_optional_text(payload_dict.get("account_alert_policy_reason")),
        "preview_signal_score": _safe_float(payload_dict.get("preview_signal_score")),
        "account_reasoning_preview_only": bool(payload_dict.get("account_reasoning_preview_only")),
        "account_pressure_disclaimer": _clean_optional_text(payload_dict.get("account_pressure_disclaimer")),
        "status": _row_value(row, "status"),
        "first_seen_at": str(_row_value(row, "first_seen_at")) if _row_value(row, "first_seen_at") else None,
        "last_seen_at": str(_row_value(row, "last_seen_at")) if _row_value(row, "last_seen_at") else None,
        "resolved_at": str(_row_value(row, "resolved_at")) if _row_value(row, "resolved_at") else None,
        "reopen_count": int(_row_value(row, "reopen_count") or 0),
        "created_at": str(_row_value(row, "created_at")) if _row_value(row, "created_at") else None,
        "updated_at": str(_row_value(row, "updated_at")) if _row_value(row, "updated_at") else None,
    }


def _watchlist_alert_source_name(row: dict[str, Any]) -> str | None:
    distribution = row.get("source_distribution")
    if isinstance(distribution, dict) and distribution:
        ranked = sorted(
            (
                (str(key).strip(), int(value or 0))
                for key, value in distribution.items()
                if str(key).strip()
            ),
            key=lambda item: (-item[1], item[0].lower()),
        )
        if ranked:
            return ranked[0][0]
    for review in row.get("source_reviews") or []:
        if isinstance(review, dict):
            source_name = _clean_optional_text(review.get("source"))
            if source_name:
                return source_name
    return None


def _threshold_hit_numeric(value: Any, threshold: float | None) -> bool:
    if threshold is None:
        return False
    score = _safe_float(value)
    if score is None:
        return False
    return score >= threshold


def summarize_suppressed_preview_accounts(
    *,
    watchlist_view: dict[str, Any],
    accounts_feed: dict[str, Any],
) -> dict[str, Any]:
    account_threshold = _safe_float(watchlist_view.get("account_alert_threshold"))
    preview_policy = watchlist_view.get("preview_account_alert_policy")
    if account_threshold is None:
        return {
            "count": 0,
            "threshold_value": None,
            "preview_account_alert_policy": preview_policy,
            "reason_details": {},
            "reasons": {},
            "vendors": [],
            "accounts": [],
        }

    reasons: dict[str, int] = {}
    reason_details: dict[str, dict[str, Any]] = {}
    vendors: dict[str, int] = {}
    accounts: list[dict[str, Any]] = []
    for account in accounts_feed.get("accounts") or []:
        if not isinstance(account, dict):
            continue
        if not bool(account.get("account_reasoning_preview_only")):
            continue
        if bool(account.get("account_alert_eligible", True)):
            continue
        policy_reason = _clean_optional_text(account.get("account_alert_policy_reason"))
        if not policy_reason:
            continue
        account_alert_score, account_alert_score_source = _accounts_in_motion_alert_basis(account)
        if not _threshold_hit_numeric(account_alert_score, account_threshold):
            continue
        vendor_name = _clean_optional_text(account.get("vendor"))
        company_name = _clean_optional_text(account.get("company"))
        category = _clean_optional_text(account.get("category"))
        source_name = _watchlist_alert_source_name(account)
        reasons[policy_reason] = reasons.get(policy_reason, 0) + 1
        if policy_reason not in reason_details:
            reason_details[policy_reason] = _suppressed_preview_reason_detail(
                policy_reason=policy_reason,
                preview_policy=preview_policy if isinstance(preview_policy, dict) else None,
            )
        if vendor_name:
            vendors[vendor_name] = vendors.get(vendor_name, 0) + 1
        accounts.append(
            {
                "vendor_name": vendor_name,
                "company_name": company_name,
                "category": category,
                "source": source_name,
                "account_alert_score": account_alert_score,
                "account_alert_score_source": account_alert_score_source,
                "account_alert_policy_reason": policy_reason,
                "confidence": _safe_float(account.get("confidence")),
                "budget_authority": bool(account.get("budget_authority")),
                "account_pressure_disclaimer": _clean_optional_text(
                    account.get("account_pressure_disclaimer")
                ),
            }
        )

    accounts.sort(
        key=lambda row: (
            -(row.get("account_alert_score") or 0.0),
            str(row.get("vendor_name") or "").lower(),
            str(row.get("company_name") or "").lower(),
        )
    )
    vendor_rows = [
        {"vendor_name": vendor_name, "count": count}
        for vendor_name, count in sorted(vendors.items(), key=lambda item: (-item[1], item[0].lower()))
    ]
    return {
        "count": len(accounts),
        "threshold_value": account_threshold,
        "preview_account_alert_policy": preview_policy,
        "reason_details": reason_details,
        "reasons": reasons,
        "vendors": vendor_rows,
        "accounts": accounts,
    }


def suppressed_preview_summary_suffix(summary: Any) -> str:
    if not isinstance(summary, dict):
        return ""
    count = _coerce_optional_int(summary.get("count")) or 0
    if count <= 0:
        return ""
    label = "preview-backed account alert" if count == 1 else "preview-backed account alerts"
    reasons = summary.get("reasons")
    if isinstance(reasons, dict) and len(reasons) == 1:
        policy_reason = next(iter(reasons))
        reason_details = summary.get("reason_details")
        if isinstance(reason_details, dict):
            reason_detail = reason_details.get(policy_reason)
            if isinstance(reason_detail, dict):
                short_summary = _clean_optional_text(reason_detail.get("short_summary"))
                if short_summary:
                    return f" ({count} {label} blocked by policy: {short_summary})"
    return f" ({count} {label} blocked by policy)"


def _watchlist_alert_entity_key(
    *,
    event_type: str,
    entity_type: str,
    vendor_name: str | None,
    company_name: str | None = None,
    category: str | None = None,
    source: str | None = None,
    report_date: str | None = None,
) -> str:
    parts = [event_type, entity_type, _normalize_vendor_name(vendor_name)]
    if company_name:
        parts.append(company_name.strip().lower())
    if category:
        parts.append(category.strip().lower())
    if source:
        parts.append(source.strip().lower())
    if report_date:
        parts.append(report_date.strip().lower())
    return ":".join(part for part in parts if part)


def build_watchlist_alert_candidates(
    *,
    watchlist_view: dict[str, Any],
    feed: dict[str, Any],
    accounts_feed: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    vendor_threshold = _safe_float(watchlist_view.get("vendor_alert_threshold"))
    account_threshold = _safe_float(watchlist_view.get("account_alert_threshold"))
    stale_threshold = _coerce_optional_int(watchlist_view.get("stale_days_threshold"))

    for signal in feed.get("signals") or []:
        vendor_name = _clean_optional_text(signal.get("vendor_name"))
        category = _clean_optional_text(signal.get("product_category"))
        signal_reasoning_reference_ids = signal.get("reasoning_reference_ids")
        if signal.get("vendor_alert_hit") and vendor_threshold is not None:
            candidates.append({
                "event_type": "vendor_alert",
                "threshold_field": "vendor_alert_threshold",
                "entity_type": "vendor",
                "entity_key": _watchlist_alert_entity_key(
                    event_type="vendor_alert",
                    entity_type="vendor",
                    vendor_name=vendor_name,
                ),
                "vendor_name": vendor_name,
                "company_name": None,
                "category": category,
                "source": None,
                "threshold_value": vendor_threshold,
                "summary": (
                    f"{vendor_name} crossed the vendor alert threshold at "
                    f"{_safe_float(signal.get('avg_urgency_score'), 0.0):.1f}"
                ),
                "payload": {
                    "avg_urgency_score": signal.get("avg_urgency_score"),
                    "freshness_status": signal.get("freshness_status"),
                    "freshness_timestamp": signal.get("freshness_timestamp"),
                    "synthesis_wedge_label": signal.get("synthesis_wedge_label"),
                    "reasoning_reference_ids": signal_reasoning_reference_ids,
                },
            })
        if signal.get("stale_threshold_hit") and stale_threshold is not None:
            candidates.append({
                "event_type": "stale_data",
                "threshold_field": "stale_days_threshold",
                "entity_type": "vendor",
                "entity_key": _watchlist_alert_entity_key(
                    event_type="stale_data",
                    entity_type="vendor",
                    vendor_name=vendor_name,
                ),
                "vendor_name": vendor_name,
                "company_name": None,
                "category": category,
                "source": None,
                "threshold_value": stale_threshold,
                "summary": f"{vendor_name} is older than the stale-data policy",
                "payload": {
                    "freshness_status": signal.get("freshness_status"),
                    "freshness_reason": signal.get("freshness_reason"),
                    "freshness_timestamp": signal.get("freshness_timestamp"),
                    "reasoning_reference_ids": signal_reasoning_reference_ids,
                },
            })

    for account in accounts_feed.get("accounts") or []:
        vendor_name = _clean_optional_text(account.get("vendor"))
        company_name = _clean_optional_text(account.get("company"))
        category = _clean_optional_text(account.get("category"))
        source_name = _watchlist_alert_source_name(account)
        account_alert_score, account_alert_score_source = _accounts_in_motion_alert_basis(account)
        reasoning_reference_ids = account.get("reasoning_reference_ids")
        source_review_ids = account.get("source_review_ids")
        account_review_focus = _watchlist_alert_account_focus_payload(account)
        preview_only = bool(account.get("account_reasoning_preview_only"))
        entity_type = "account" if company_name else "signal_cluster"
        entity_key = _watchlist_alert_entity_key(
            event_type="account_alert",
            entity_type=entity_type,
            vendor_name=vendor_name,
            company_name=company_name,
            category=category,
            source=source_name,
            report_date=_clean_optional_text(account.get("report_date")),
        )
        subject = company_name or f"{vendor_name} signal cluster"
        if account.get("account_alert_hit") and account_threshold is not None:
            if preview_only:
                summary = (
                    f"Early account signal for {subject} crossed the account alert threshold at "
                    f"{_safe_float(account_alert_score, 0.0):.1f}"
                )
            else:
                summary = (
                    f"{subject} crossed the account alert threshold at "
                    f"{_safe_float(account_alert_score, 0.0):.1f}"
                )
            candidates.append({
                "event_type": "account_alert",
                "threshold_field": "account_alert_threshold",
                "entity_type": entity_type,
                "entity_key": entity_key,
                "vendor_name": vendor_name,
                "company_name": company_name,
                "category": category,
                "source": source_name,
                "threshold_value": account_threshold,
                "summary": summary,
                "payload": {
                    "urgency": account.get("urgency"),
                    "account_alert_score": account_alert_score,
                    "account_alert_score_source": account_alert_score_source,
                    "account_alert_eligible": bool(account.get("account_alert_eligible", True)),
                    "account_alert_policy_reason": account.get("account_alert_policy_reason"),
                    "preview_signal_score": account.get("preview_signal_score"),
                    "confidence": account.get("confidence"),
                    "report_date": account.get("report_date"),
                    "freshness_status": account.get("freshness_status"),
                    "reasoning_reference_ids": reasoning_reference_ids,
                    "source_review_ids": source_review_ids,
                    "account_review_focus": account_review_focus,
                    "account_reasoning_preview_only": preview_only,
                    "account_pressure_disclaimer": account.get("account_pressure_disclaimer"),
                },
            })
        if account.get("stale_threshold_hit") and stale_threshold is not None:
            candidates.append({
                "event_type": "stale_data",
                "threshold_field": "stale_days_threshold",
                "entity_type": entity_type,
                "entity_key": _watchlist_alert_entity_key(
                    event_type="stale_data",
                    entity_type=entity_type,
                    vendor_name=vendor_name,
                    company_name=company_name,
                    category=category,
                    source=source_name,
                    report_date=_clean_optional_text(account.get("report_date")),
                ),
                "vendor_name": vendor_name,
                "company_name": company_name,
                "category": category,
                "source": source_name,
                "threshold_value": stale_threshold,
                "summary": f"{subject} is older than the stale-data policy",
                "payload": {
                    "stale_days": account.get("stale_days"),
                    "report_date": account.get("report_date"),
                    "freshness_status": account.get("freshness_status"),
                    "freshness_reason": account.get("freshness_reason"),
                    "reasoning_reference_ids": reasoning_reference_ids,
                    "source_review_ids": source_review_ids,
                    "account_review_focus": account_review_focus,
                },
            })
    return candidates


async def evaluate_watchlist_alert_events_for_view(
    pool: Any,
    *,
    account_id: _uuid.UUID,
    view_id: _uuid.UUID,
    view_row: Any,
    user: Any,
    slow_burn_loader: Callable[..., Awaitable[dict[str, Any]]],
    accounts_loader: Callable[..., Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    feed = await slow_burn_loader(
        vendor_names=list(_row_value(view_row, "vendor_names") or []) or None,
        category=_row_value(view_row, "category"),
        vendor_alert_threshold=_safe_float(_row_value(view_row, "vendor_alert_threshold")),
        stale_days_threshold=_coerce_optional_int(_row_value(view_row, "stale_days_threshold")),
        user=user,
    )
    accounts_feed = await accounts_loader(
        vendor_names=list(_row_value(view_row, "vendor_names") or []) or None,
        category=_row_value(view_row, "category"),
        source=_row_value(view_row, "source"),
        min_urgency=_safe_float(_row_value(view_row, "min_urgency"), settings.b2b_churn.accounts_in_motion_min_urgency),
        include_stale=bool(_row_value(view_row, "include_stale")),
        named_accounts_only=bool(_row_value(view_row, "named_accounts_only")),
        account_alert_threshold=_safe_float(_row_value(view_row, "account_alert_threshold")),
        preview_alerts_enabled=bool(_row_value(view_row, "preview_alerts_enabled", True)),
        preview_alert_min_confidence=_safe_float(_row_value(view_row, "preview_alert_min_confidence")),
        preview_alert_require_budget_authority=(
            _row_value(view_row, "preview_alert_require_budget_authority")
            if isinstance(_row_value(view_row, "preview_alert_require_budget_authority"), bool)
            else None
        ),
        stale_days_threshold=_coerce_optional_int(_row_value(view_row, "stale_days_threshold")),
        per_vendor_limit=settings.b2b_churn.accounts_in_motion_max_per_vendor,
        limit=settings.b2b_churn.accounts_in_motion_feed_max_total,
        user=user,
    )
    candidates = build_watchlist_alert_candidates(
        watchlist_view=watchlist_view_payload(view_row),
        feed=feed,
        accounts_feed=accounts_feed,
    )
    suppressed_preview_summary = summarize_suppressed_preview_accounts(
        watchlist_view=watchlist_view_payload(view_row),
        accounts_feed=accounts_feed,
    )
    now = datetime.now(timezone.utc)
    # Use a transaction when available (real asyncpg pool) to prevent
    # concurrent evaluate/resolve race conditions.
    _acquire = getattr(pool, 'acquire', None)
    if _acquire is not None:
        _conn_ctx = _acquire()
    else:
        from contextlib import asynccontextmanager as _acm
        @_acm
        async def _noop_ctx():
            yield pool
        _conn_ctx = _noop_ctx()
    async with _conn_ctx as conn:
        _txn_fn = getattr(conn, 'transaction', None)
        if _txn_fn is not None:
            _txn_ctx = _txn_fn()
        else:
            from contextlib import asynccontextmanager as _acm2
            @_acm2
            async def _noop_txn():
                yield
            _txn_ctx = _noop_txn()
        async with _txn_ctx:
            existing_open_rows = await conn.fetch(
        """
        SELECT id, event_type, entity_key
        FROM b2b_watchlist_alert_events
        WHERE account_id = $1
          AND watchlist_view_id = $2
          AND status = 'open'
        """,
            account_id,
            view_id,
        )
            existing_open = {
                (str(row["event_type"]), str(row["entity_key"])): row["id"]
                for row in existing_open_rows
            }
            current_keys = {(str(item["event_type"]), str(item["entity_key"])) for item in candidates}
            new_open_count = sum(1 for key in current_keys if key not in existing_open)

            persisted_rows = []
            for item in candidates:
                row = await conn.fetchrow(
                    """
                    INSERT INTO b2b_watchlist_alert_events (
                        id, account_id, watchlist_view_id, event_type, threshold_field, entity_type,
                        entity_key, vendor_name, company_name, category, source, threshold_value,
                        summary, payload, status, first_seen_at, last_seen_at, resolved_at, created_at, updated_at
                    )
                    VALUES (
                        $1, $2, $3, $4, $5, $6,
                        $7, $8, $9, $10, $11, $12,
                        $13, $14::jsonb, 'open', $15, $15, NULL, $15, $15
                    )
                    ON CONFLICT (watchlist_view_id, event_type, entity_key)
                    DO UPDATE SET
                        threshold_field = EXCLUDED.threshold_field,
                        entity_type = EXCLUDED.entity_type,
                        vendor_name = EXCLUDED.vendor_name,
                        company_name = EXCLUDED.company_name,
                        category = EXCLUDED.category,
                        source = EXCLUDED.source,
                        threshold_value = EXCLUDED.threshold_value,
                        summary = EXCLUDED.summary,
                        payload = EXCLUDED.payload,
                        status = 'open',
                        last_seen_at = EXCLUDED.last_seen_at,
                        resolved_at = NULL,
                        reopen_count = b2b_watchlist_alert_events.reopen_count + CASE
                            WHEN b2b_watchlist_alert_events.status = 'resolved' THEN 1 ELSE 0 END,
                        updated_at = EXCLUDED.updated_at
                    RETURNING id, watchlist_view_id, event_type, threshold_field, entity_type, entity_key,
                              vendor_name, company_name, category, source, threshold_value, summary,
                              payload, status, first_seen_at, last_seen_at, resolved_at, reopen_count, created_at, updated_at
                    """,
                    _uuid.uuid4(),
                    account_id,
                    view_id,
                    item["event_type"],
                    item["threshold_field"],
                    item["entity_type"],
                    item["entity_key"],
                    item["vendor_name"],
                    item["company_name"],
                    item["category"],
                    item["source"],
                    item["threshold_value"],
                    item["summary"],
                    json.dumps(item["payload"] or {}),
                    now,
                )
                persisted_rows.append(row)

            resolved_ids = [event_id for key, event_id in existing_open.items() if key not in current_keys]
            if resolved_ids:
                await conn.execute(
                    """
                    UPDATE b2b_watchlist_alert_events
                    SET status = 'resolved',
                        resolved_at = $2,
                        updated_at = $2
                    WHERE id = ANY($1::uuid[])
                    """,
                    resolved_ids,
                    now,
                )

    events = [serialize_watchlist_alert_event(row) for row in persisted_rows]
    return {
        "watchlist_view_id": str(view_id),
        "watchlist_view_name": _row_value(view_row, "name"),
        "evaluated_at": now.isoformat(),
        "events": events,
        "count": len(events),
        "new_open_event_count": new_open_count,
        "resolved_event_count": len(resolved_ids),
        "suppressed_preview_summary": suppressed_preview_summary,
    }


def watchlist_alert_sender_configured() -> bool:
    if not settings.b2b_alert.email_enabled:
        return False
    campaign_cfg = settings.campaign_sequence
    if campaign_cfg.sender_type == "ses":
        return bool(campaign_cfg.ses_from_email)
    return bool(campaign_cfg.resend_api_key and campaign_cfg.resend_from_email)


def watchlist_alert_from_email() -> str:
    campaign_cfg = settings.campaign_sequence
    sender_name = settings.b2b_alert.sender_name.strip()
    from_email = (
        str(campaign_cfg.ses_from_email or "").strip()
        if campaign_cfg.sender_type == "ses"
        else str(campaign_cfg.resend_from_email or "").strip()
    )
    if not from_email:
        return ""
    return f"{sender_name} <{from_email}>" if sender_name else from_email


def watchlist_manage_url() -> str | None:
    configured = str(settings.b2b_alert.dashboard_base_url or "").strip()
    if configured:
        return configured.rstrip("/")
    frontend = str(settings.saas_auth.frontend_base_url or "").strip()
    if not frontend:
        return None
    return f"{frontend.rstrip('/')}/watchlists"


async def resolve_watchlist_alert_recipients(
    pool: Any,
    *,
    account_id: _uuid.UUID,
) -> list[dict[str, str]]:
    rows = await pool.fetch(
        """
        SELECT email, full_name
        FROM saas_users
        WHERE account_id = $1
          AND is_active = TRUE
          AND role = 'owner'
          AND email IS NOT NULL
          AND BTRIM(email) <> ''
        ORDER BY created_at ASC, email ASC
        """,
        account_id,
    )
    recipients: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        email = str(_row_value(row, "email") or "").strip().lower()
        if not email or email in seen:
            continue
        seen.add(email)
        recipients.append({
            "email": email,
            "full_name": str(_row_value(row, "full_name") or "").strip(),
        })
    return recipients


def watchlist_alert_event_email_rows(events: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events[:12]:
        vendor_name = str(event.get("vendor_name") or "").strip()
        company_name = str(event.get("company_name") or "").strip()
        label_parts = [str(event.get("event_type") or "").replace("_", " ").strip()]
        if vendor_name:
            label_parts.append(vendor_name)
        summary = str(event.get("summary") or "").strip() or "Watchlist alert"
        detail_parts: list[str] = []
        if company_name:
            detail_parts.append(f"Account: {company_name}")
        category = str(event.get("category") or "").strip()
        if category:
            detail_parts.append(f"Category: {category}")
        source = str(event.get("source") or "").strip()
        if source:
            detail_parts.append(f"Source: {source}")
        threshold_value = event.get("threshold_value")
        if threshold_value is not None:
            detail_parts.append(f"Threshold: {threshold_value}")
        rows.append({
            "label": " - ".join(part.title() for part in label_parts if part),
            "summary": summary,
            "detail": " | ".join(detail_parts),
        })
    return rows


async def send_watchlist_alert_email(
    pool: Any,
    sender: Any,
    *,
    recipient: str,
    from_addr: str,
    subject: str,
    html_body: str,
    view_name: str,
) -> tuple[bool, str | None, str | None, bool]:
    suppression = await is_suppressed(pool, email=recipient)
    if suppression:
        reason = str(suppression.get("reason") or "suppressed").strip() or "suppressed"
        return False, None, f"{recipient}: suppressed ({reason})", True

    body = _wrap_with_footer(html_body, recipient, settings.campaign_sequence)
    headers = _unsub_headers(settings.campaign_sequence.unsubscribe_base_url, recipient)
    try:
        result = await sender.send(
            to=recipient,
            from_email=from_addr,
            subject=subject,
            body=body,
            headers=headers or None,
            tags=[
                {"name": "task", "value": "watchlist_alert_email"},
                {"name": "view", "value": view_name[:64] or "watchlist"},
            ],
        )
        message_id = str(result.get("id") or "").strip() or None
        return True, message_id, None, False
    except Exception as exc:
        return False, None, f"{recipient}: {exc}", False


async def record_watchlist_alert_email_log(
    pool: Any,
    *,
    log_id: _uuid.UUID,
    account_id: _uuid.UUID,
    view_id: _uuid.UUID,
    event_ids: list[_uuid.UUID],
    recipient_emails: list[str],
    message_ids: list[str],
    event_count: int,
    status: str,
    summary: str,
    error: str | None,
    delivered_at: datetime | None,
    recorded_at: datetime,
    scheduled_for: datetime | None = None,
    delivery_frequency: str | None = None,
    delivery_mode: str = "live",
    suppressed_preview_summary: dict[str, Any] | None = None,
) -> None:
    await pool.execute(
        """
        INSERT INTO b2b_watchlist_alert_email_log (
            id, account_id, watchlist_view_id, event_ids, recipient_emails, message_ids,
            event_count, status, summary, error, delivered_at, created_at, updated_at,
            scheduled_for, delivery_frequency, delivery_mode, suppressed_preview_summary
        )
        VALUES (
            $1, $2, $3, $4::uuid[], $5::text[], $6::text[], $7, $8, $9, $10, $11, $12, $12,
            $13, $14, $15, $16::jsonb
        )
        """,
        log_id,
        account_id,
        view_id,
        event_ids,
        recipient_emails,
        message_ids,
        event_count,
        status,
        summary,
        error,
        delivered_at,
        recorded_at,
        scheduled_for,
        delivery_frequency,
        delivery_mode,
        json.dumps(suppressed_preview_summary) if suppressed_preview_summary else None,
    )


async def update_watchlist_alert_email_log(
    pool: Any,
    *,
    log_id: _uuid.UUID,
    event_ids: list[_uuid.UUID],
    recipient_emails: list[str],
    message_ids: list[str],
    event_count: int,
    status: str,
    summary: str,
    error: str | None,
    delivered_at: datetime | None,
    recorded_at: datetime,
    suppressed_preview_summary: dict[str, Any] | None = None,
) -> None:
    await pool.execute(
        """
        UPDATE b2b_watchlist_alert_email_log
        SET event_ids = $2::uuid[],
            recipient_emails = $3::text[],
            message_ids = $4::text[],
            event_count = $5,
            status = $6,
            summary = $7,
            error = $8,
            delivered_at = $9,
            updated_at = $10,
            suppressed_preview_summary = $11::jsonb
        WHERE id = $1
        """,
        log_id,
        event_ids,
        recipient_emails,
        message_ids,
        event_count,
        status,
        summary,
        error,
        delivered_at,
        recorded_at,
        json.dumps(suppressed_preview_summary) if suppressed_preview_summary else None,
    )


async def update_watchlist_view_delivery_state(
    pool: Any,
    *,
    view_id: _uuid.UUID,
    account_id: _uuid.UUID,
    delivered_at: datetime | None,
    status: str,
    summary: str,
    recorded_at: datetime,
) -> None:
    await pool.execute(
        """
        UPDATE b2b_watchlist_views
        SET last_alert_delivery_at = $2,
            last_alert_delivery_status = $3,
            last_alert_delivery_summary = $4,
            updated_at = $5
        WHERE id = $1
          AND account_id = $6
        """,
        view_id,
        delivered_at,
        status,
        summary[:1000],
        recorded_at,
        account_id,
    )


def render_watchlist_alert_email_html(
    *,
    account_name: str,
    view_name: str,
    summary_line: str,
    events: list[dict[str, Any]],
) -> str:
    return render_watchlist_alert_delivery_html(
        account_name=account_name,
        view_name=view_name,
        summary_line=summary_line,
        manage_url=watchlist_manage_url(),
        events=watchlist_alert_event_email_rows(events),
    )
