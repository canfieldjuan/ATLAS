"""Recurring delivery for persisted B2B report subscriptions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

from ...config import settings
from ...services.b2b.report_trust import (
    report_artifact_payload,
    report_freshness_label,
    report_review_payload,
    report_section_evidence_payload,
)
from ...services.campaign_sender import get_campaign_sender
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ...templates.email.report_subscription_delivery import (
    render_report_subscription_delivery_html,
)
from ._b2b_shared import has_complete_core_run_marker
from .campaign_send import _unsub_headers, _wrap_with_footer
from .campaign_suppression import is_suppressed

logger = logging.getLogger("atlas.tasks.b2b_report_subscription_delivery")

_TIMESTAMP_KEYS = {
    "report_date",
    "created_at",
    "updated_at",
    "generated_at",
    "last_computed_at",
    "data_as_of_date",
    "as_of_date",
    "computed_at",
}
_REPORT_TYPE_FOCUS_MAP = {
    "battle_cards": {"battle_card", "challenger_brief"},
    "executive_reports": {
        "weekly_churn_feed",
        "vendor_scorecard",
        "vendor_deep_dive",
        "vendor_retention",
        "category_overview",
        "exploratory_overview",
    },
    "comparison_packs": {
        "vendor_comparison",
        "account_comparison",
        "account_deep_dive",
        "displacement_report",
        "challenger_intel",
        "challenger_brief",
    },
}
_REPORT_FREQUENCY_DAYS = {
    "weekly": 7,
    "monthly": 30,
    "quarterly": 90,
}
_QUALITY_GATED_REPORT_TYPES = {"battle_card", "challenger_brief"}
_PERSISTED_ARTIFACT_STATUSES = {"completed", "published"}
_DELIVERY_CONTENT_HASH_STATUSES = ("sent",)
_DELIVERY_LOG_CLAIMABLE_STALE_STATUSES = ("failed", "processing")
_DELIVERY_LOG_RECLAIMABLE_IMMEDIATE_STATUSES = ("dry_run",)
_DELIVERY_BLOCKED_FRESHNESS_STATE = "blocked"


@dataclass
class _ArtifactSelectionResult:
    artifacts: list[dict[str, Any]]
    eligible_before_freshness_count: int = 0
    freshness_blocked_count: int = 0


def _safe_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        import json

        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        return default


def _task_metadata(task: ScheduledTask | Any) -> dict[str, Any]:
    metadata = getattr(task, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _normalize_runtime_scope(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        parsed = _safe_json(text)
        if parsed is not text:
            return _normalize_runtime_scope(parsed)
        return {
            item.strip()
            for item in text.split(",")
            if item and item.strip()
        }
    if isinstance(value, (list, tuple, set)):
        return {
            str(item).strip()
            for item in value
            if str(item).strip()
        }
    text = str(value).strip()
    return {text} if text else set()


def _campaign_from_email() -> str:
    campaign_cfg = settings.campaign_sequence
    if campaign_cfg.sender_type == "ses":
        return str(campaign_cfg.ses_from_email or "").strip()
    return str(campaign_cfg.resend_from_email or "").strip()


def _frontend_base_url() -> str:
    configured = str(settings.b2b_report_delivery.dashboard_base_url or "").strip()
    if configured:
        return configured.rstrip("/")
    return str(settings.saas_auth.frontend_base_url or "").strip().rstrip("/")


def _dry_run_enabled(task: ScheduledTask | Any) -> bool:
    metadata = _task_metadata(task)
    if "dry_run" in metadata:
        return bool(metadata.get("dry_run"))
    return bool(settings.b2b_report_delivery.dry_run)


def _delivery_mode(dry_run: bool) -> str:
    return "dry_run" if dry_run else "live"


def _canary_account_ids(task: ScheduledTask | Any) -> set[str]:
    metadata = _task_metadata(task)
    if "canary_account_ids" in metadata:
        return _normalize_runtime_scope(metadata.get("canary_account_ids"))
    if "live_account_ids" in metadata:
        return _normalize_runtime_scope(metadata.get("live_account_ids"))
    return _normalize_runtime_scope(settings.b2b_report_delivery.canary_account_ids)


def _scope_manage_url(scope_type: str, scope_key: str) -> str | None:
    base_url = _frontend_base_url()
    if not base_url:
        return None
    if scope_type == "report":
        return f"{base_url}/reports/{quote(scope_key, safe='')}"
    return f"{base_url}/reports"


def _scope_manage_url_with_filters(scope_type: str, scope_key: str, filter_payload: Any) -> str | None:
    base_url = _scope_manage_url(scope_type, scope_key)
    if not base_url or scope_type != "library_view":
        return base_url
    normalized = _normalize_subscription_filter_payload(_safe_json(filter_payload))
    if not normalized:
        return base_url

    params: list[str] = []
    if normalized.get("report_type"):
        params.append(f"report_type={quote(normalized['report_type'], safe='')}")
    if normalized.get("vendor_filter"):
        params.append(f"vendor_filter={quote(normalized['vendor_filter'], safe='')}")
    if normalized.get("quality_status"):
        params.append(f"quality_status={quote(normalized['quality_status'], safe='')}")
    if normalized.get("freshness_state"):
        params.append(f"freshness_state={quote(normalized['freshness_state'], safe='')}")
    if normalized.get("review_state"):
        params.append(f"review_state={quote(normalized['review_state'], safe='')}")
    if not params:
        return base_url
    return f"{base_url}?{'&'.join(params)}"


def _report_url(report_id: Any) -> str | None:
    base_url = _frontend_base_url()
    if not base_url or not report_id:
        return None
    return f"{base_url}/reports/{quote(str(report_id), safe='')}"


def _format_report_type_label(report_type: str) -> str:
    return str(report_type or "report").replace("_", " ").title()


def _format_section_label(section_key: str) -> str:
    return str(section_key or "section").replace("_", " ").title()


def _report_display_title(row) -> str:
    report_type = str(row["report_type"] or "")
    vendor_filter = str(row["vendor_filter"] or "").strip()
    category_filter = str(row["category_filter"] or "").strip()
    if report_type in {"vendor_comparison", "account_comparison"} and vendor_filter and category_filter:
        return f"{vendor_filter} vs {category_filter}"
    if report_type == "challenger_brief" and vendor_filter and category_filter:
        return f"{vendor_filter} -> {category_filter}"
    return vendor_filter or category_filter or _format_report_type_label(report_type)


def _quality_status(row, intelligence_data: dict[str, Any]) -> str:
    quality_record = _safe_json(intelligence_data.get("battle_card_quality"))
    quality_status = str(row["quality_status"] or intelligence_data.get("quality_status") or "").strip().lower()
    if not quality_status and isinstance(quality_record, dict):
        quality_status = str(quality_record.get("status") or "").strip().lower()
    return quality_status


def _row_int(row, key: str) -> int:
    try:
        return int(row[key] or 0)
    except Exception:
        return 0


def _report_trust_label(row, intelligence_data: dict[str, Any]) -> str:
    quality_status = _quality_status(row, intelligence_data)
    report_type = str(row["report_type"] or "")
    status = str(row["status"] or "").strip().lower()

    if report_type in _QUALITY_GATED_REPORT_TYPES and quality_status:
        if quality_status == "sales_ready":
            return "Evidence-backed"
        if quality_status == "needs_review":
            return "Needs review"
        if quality_status == "thin_evidence":
            return "Thin evidence"
        return "Fallback render"

    if status in _PERSISTED_ARTIFACT_STATUSES:
        return "Persisted artifact"
    return "In workflow"


def _section_evidence_summary(
    section_evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "witness_backed_count": 0,
        "partial_count": 0,
        "thin_count": 0,
        "partial_sections": [],
        "thin_sections": [],
    }
    if not isinstance(section_evidence, dict):
        return summary

    for field_key in sorted(section_evidence.keys()):
        entry = section_evidence.get(field_key)
        if not isinstance(entry, dict):
            continue
        state = str(entry.get("state") or "").strip().lower()
        label = _format_section_label(field_key)
        if state == "witness_backed":
            summary["witness_backed_count"] += 1
        elif state == "partial":
            summary["partial_count"] += 1
            summary["partial_sections"].append(label)
        elif state == "thin":
            summary["thin_count"] += 1
            summary["thin_sections"].append(label)
    return summary


def _artifact_ready(row, intelligence_data: dict[str, Any]) -> bool:
    report_type = str(row["report_type"] or "").strip().lower()
    status = str(row["status"] or "").strip().lower()
    quality_status = _quality_status(row, intelligence_data)

    if report_type == "battle_card":
        return status == "sales_ready" or quality_status == "sales_ready"
    return status in _PERSISTED_ARTIFACT_STATUSES


def _parse_timestamp_candidate(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _collect_timestamp_candidates(value: Any, results: list[datetime], *, depth: int = 0, seen: set[int] | None = None) -> None:
    if depth > 5 or value is None:
        return
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list)):
        marker = id(value)
        if marker in seen:
            return
        seen.add(marker)

    if isinstance(value, list):
        for item in value:
            _collect_timestamp_candidates(item, results, depth=depth + 1, seen=seen)
        return

    if not isinstance(value, dict):
        return

    for key, child in value.items():
        if key in _TIMESTAMP_KEYS:
            parsed = _parse_timestamp_candidate(child)
            if parsed is not None:
                results.append(parsed)
        if isinstance(child, (dict, list)):
            _collect_timestamp_candidates(child, results, depth=depth + 1, seen=seen)


def _relative_age_label(age_hours: float) -> str:
    rounded = max(1, int(round(age_hours)))
    if rounded < 24:
        return f"{rounded}h ago"
    if rounded < 24 * 7:
        return f"{max(1, int(round(rounded / 24)))}d ago"
    return f"{max(1, int(round(rounded / (24 * 7))))}w ago"


def _derive_freshness(row, intelligence_data: dict[str, Any], data_density: dict[str, Any]) -> dict[str, Any]:
    fresh_hours = settings.b2b_report_delivery.fresh_hours
    monitor_hours = settings.b2b_report_delivery.monitor_hours
    candidates: list[datetime] = []
    for candidate in (
        row["report_date"],
        row["created_at"],
    ):
        parsed = _parse_timestamp_candidate(candidate)
        if parsed is not None:
            candidates.append(parsed)

    _collect_timestamp_candidates(intelligence_data, candidates)
    _collect_timestamp_candidates(data_density, candidates)

    if not candidates:
        return {
            "state": "unknown",
            "label": "Unknown",
            "detail": "No freshness timestamp was attached to this artifact.",
            "anchor": None,
        }

    anchor = max(candidates)
    age_hours = max(0.0, (datetime.now(timezone.utc) - anchor).total_seconds() / 3600.0)
    if age_hours <= fresh_hours:
        return {
            "state": "fresh",
            "label": report_freshness_label("fresh"),
            "detail": f"Evidence window looks current ({_relative_age_label(age_hours)}).",
            "anchor": anchor,
        }
    if age_hours <= monitor_hours:
        return {
            "state": "monitor",
            "label": report_freshness_label("monitor"),
            "detail": f"Artifact is aging ({_relative_age_label(age_hours)}). Recheck before external use.",
            "anchor": anchor,
        }
    return {
        "state": "stale",
        "label": report_freshness_label("stale"),
        "detail": f"Refresh recommended. Latest evidence is {_relative_age_label(age_hours)} old.",
        "anchor": anchor,
    }


def _freshness_allowed(policy: str, freshness_state: str) -> bool:
    if policy == "any":
        return True
    if policy == "fresh_only":
        return freshness_state == "fresh"
    return freshness_state in {"fresh", "monitor"}


def _tracked_vendor_allowed(row, tracked_vendors: set[str], account_id: Any) -> bool:
    vendor_filter = str(row["vendor_filter"] or "").strip().lower()
    row_account_id = row["account_id"]
    if row_account_id is not None and str(row_account_id) == str(account_id):
        return True
    if not vendor_filter:
        return True
    return vendor_filter in tracked_vendors


def _focus_allowed(scope_type: str, focus: str, report_type: str) -> bool:
    if scope_type == "report" or focus == "all":
        return True
    allowed_types = _REPORT_TYPE_FOCUS_MAP.get(focus)
    if not allowed_types:
        return True
    return report_type in allowed_types


def _normalize_subscription_filter_payload(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
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
        normalized["quality_status"] = quality_status
    if freshness_state:
        normalized["freshness_state"] = freshness_state
    if review_state:
        normalized["review_state"] = review_state
    return normalized


def _review_state(row) -> str:
    blocker_count = _row_int(row, "blocker_count")
    warning_count = _row_int(row, "warning_count")
    unresolved_issue_count = _row_int(row, "unresolved_issue_count")
    if blocker_count > 0:
        return "blocked"
    if unresolved_issue_count > 0:
        return "open_review"
    if warning_count > 0:
        return "warnings"
    return "clean"


def _report_matches_subscription_filters(
    report_row,
    intelligence_data: dict[str, Any],
    filter_payload: dict[str, str],
) -> bool:
    report_type = str(report_row["report_type"] or "").strip()
    vendor_filter = str(report_row["vendor_filter"] or "").strip().lower()
    quality_status = _quality_status(report_row, intelligence_data)

    requested_report_type = str(filter_payload.get("report_type") or "").strip()
    if requested_report_type and report_type != requested_report_type:
        return False

    requested_vendor_filter = str(filter_payload.get("vendor_filter") or "").strip().lower()
    if requested_vendor_filter and requested_vendor_filter not in vendor_filter:
        return False

    requested_quality_status = str(filter_payload.get("quality_status") or "").strip().lower()
    if requested_quality_status and quality_status != requested_quality_status:
        return False

    freshness = _derive_freshness(
        report_row,
        intelligence_data,
        _safe_json(_row_value(report_row, "data_density")) or {},
    )
    requested_freshness_state = str(filter_payload.get("freshness_state") or "").strip().lower()
    if requested_freshness_state and freshness["state"] != requested_freshness_state:
        return False

    requested_review_state = str(filter_payload.get("review_state") or "").strip().lower()
    if requested_review_state and _review_state(report_row) != requested_review_state:
        return False

    return True


def _delivery_eligibility_reason(row, intelligence_data: dict[str, Any]) -> str | None:
    delivery_cfg = settings.b2b_report_delivery
    blocker_count = _row_int(row, "blocker_count")
    unresolved_issue_count = _row_int(row, "unresolved_issue_count")
    report_type = str(row["report_type"] or "").strip().lower()
    quality_status = _quality_status(row, intelligence_data)

    if blocker_count > int(delivery_cfg.max_blocker_count):
        return (
            f"blocker_count={blocker_count} exceeds "
            f"max_blocker_count={delivery_cfg.max_blocker_count}"
        )
    if unresolved_issue_count > int(delivery_cfg.max_open_review_count):
        return (
            f"unresolved_issue_count={unresolved_issue_count} exceeds "
            f"max_open_review_count={delivery_cfg.max_open_review_count}"
        )
    if (
        delivery_cfg.require_sales_ready_for_competitive
        and report_type in _QUALITY_GATED_REPORT_TYPES
        and quality_status
        and quality_status != "sales_ready"
    ):
        return f"quality_status={quality_status}"
    return None


def _shorten(value: str, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _evidence_highlight_from_item(item: Any) -> str | None:
    if not isinstance(item, dict):
        text = str(item or "").strip()
        return _shorten(text) if text else None

    excerpt = str(item.get("excerpt_text") or item.get("quote") or item.get("text") or "").strip()
    if not excerpt:
        return None

    meta: list[str] = []
    reviewer_company = str(item.get("reviewer_company") or "").strip()
    reviewer_title = str(item.get("reviewer_title") or item.get("role") or "").strip()
    competitor = str(item.get("competitor") or "").strip()
    pain_category = str(item.get("pain_category") or "").strip()
    time_anchor = str(item.get("time_anchor") or "").strip()

    if reviewer_company:
        meta.append(reviewer_company)
    if reviewer_title:
        meta.append(reviewer_title)
    if competitor:
        meta.append(f"vs {competitor}")
    if pain_category:
        meta.append(f"pain: {pain_category}")
    if time_anchor:
        meta.append(time_anchor)

    if meta:
        return f"{_shorten(excerpt)} ({'; '.join(meta)})"
    return _shorten(excerpt)


def _extract_evidence_highlights(intelligence_data: dict[str, Any], *, limit: int = 3) -> list[str]:
    for key in ("witness_highlights", "reasoning_witness_highlights"):
        values = intelligence_data.get(key)
        if not isinstance(values, list):
            continue
        highlights: list[str] = []
        for item in values:
            line = _evidence_highlight_from_item(item)
            if not line:
                continue
            highlights.append(line)
            if len(highlights) >= limit:
                return highlights

    values = intelligence_data.get("customer_pain_quotes")
    if isinstance(values, list):
        highlights = []
        for item in values:
            line = _evidence_highlight_from_item(item)
            if not line:
                continue
            highlights.append(line)
            if len(highlights) >= limit:
                return highlights
    return []


def _frequency_label(frequency: str) -> str:
    return str(frequency or "weekly").replace("_", " ").title()


def _delivery_interval(frequency: str) -> timedelta:
    days = _REPORT_FREQUENCY_DAYS.get(str(frequency or "weekly"), 7)
    return timedelta(days=days)


def _next_delivery_at(frequency: str, anchor: datetime | None = None) -> datetime:
    next_due = _parse_timestamp_candidate(anchor) or datetime.now(timezone.utc)
    interval = _delivery_interval(frequency)
    now = datetime.now(timezone.utc)
    while next_due <= now:
        next_due += interval
    return next_due.replace(microsecond=0)


def _subscription_summary(scope_type: str, artifacts: list[dict[str, Any]]) -> str:
    if scope_type == "report" and artifacts:
        return f"{artifacts[0]['title']} is ready from your saved recurring report subscription."
    count = len(artifacts)
    if count == 1:
        return "1 persisted artifact is ready from your saved recurring report subscription."
    return f"{count} persisted artifacts are ready from your saved recurring report subscription."


def _aggregate_freshness_state(artifacts: list[dict[str, Any]]) -> str:
    states = {str(artifact["freshness_state"]) for artifact in artifacts if artifact.get("freshness_state")}
    if not states:
        return "none"
    if len(states) == 1:
        return next(iter(states))
    return "mixed"


def _empty_selection_result() -> _ArtifactSelectionResult:
    return _ArtifactSelectionResult(artifacts=[])


def _selection_is_fully_freshness_blocked(selection: _ArtifactSelectionResult) -> bool:
    return (
        not selection.artifacts
        and selection.eligible_before_freshness_count > 0
        and selection.freshness_blocked_count == selection.eligible_before_freshness_count
    )


async def _blocked_delivery_skip_details(
    pool,
    row,
    selection: _ArtifactSelectionResult,
) -> tuple[str | None, str, str | None]:
    if not _selection_is_fully_freshness_blocked(selection):
        return None, "none", None
    freshness_policy = str(row["freshness_policy"] or "fresh_or_monitor").strip().lower()
    if freshness_policy == "any":
        return None, "none", None
    core_as_of = date.today()
    if await has_complete_core_run_marker(pool, core_as_of):
        return None, "none", None
    summary = (
        "Skipped delivery because today's core churn materialization is incomplete, "
        "and no authoritative persisted artifacts were fresh enough for the saved policy."
    )
    return summary, _DELIVERY_BLOCKED_FRESHNESS_STATE, "core_incomplete"


def _delivery_content_hash(row, artifacts: list[dict[str, Any]]) -> str:
    payload = {
        "scope_type": str(row["scope_type"] or ""),
        "scope_key": str(row["scope_key"] or ""),
        "scope_label": str(row["scope_label"] or ""),
        "filter_payload": _normalize_subscription_filter_payload(
            _safe_json(_row_value(row, "filter_payload"))
        ),
        "delivery_note": str(row["delivery_note"] or "").strip(),
        "artifacts": [
            {
                "report_id": str(artifact["report_id"]),
                "report_type": str(artifact["report_type"] or ""),
                "title": str(artifact["title"] or ""),
                "trust_label": str(artifact["trust_label"] or ""),
                "artifact_state": str(artifact.get("artifact_state") or ""),
                "artifact_label": str(artifact.get("artifact_label") or ""),
                "quality_status": str(artifact.get("quality_status") or ""),
                "blocker_count": int(artifact.get("blocker_count") or 0),
                "warning_count": int(artifact.get("warning_count") or 0),
                "unresolved_issue_count": int(artifact.get("unresolved_issue_count") or 0),
                "freshness_state": str(artifact.get("freshness_state") or ""),
                "freshness_label": str(artifact.get("freshness_label") or ""),
                "review_state": str(artifact.get("review_state") or ""),
                "review_label": str(artifact.get("review_label") or ""),
                "executive_summary": str(artifact["executive_summary"] or ""),
                "evidence_highlights": [str(item or "") for item in artifact.get("evidence_highlights") or []],
                "section_evidence": [
                    {
                        "section": str(section_key or ""),
                        "state": str((section_value or {}).get("state") or ""),
                        "label": str((section_value or {}).get("label") or ""),
                        "detail": str((section_value or {}).get("detail") or ""),
                        "witness_count": int((section_value or {}).get("witness_count") or 0),
                        "metric_count": int((section_value or {}).get("metric_count") or 0),
                    }
                    for section_key, section_value in sorted(
                        (artifact.get("section_evidence") or {}).items(),
                        key=lambda item: str(item[0] or ""),
                    )
                ],
                "section_evidence_summary": {
                    "witness_backed_count": int((artifact.get("section_evidence_summary") or {}).get("witness_backed_count") or 0),
                    "partial_count": int((artifact.get("section_evidence_summary") or {}).get("partial_count") or 0),
                    "thin_count": int((artifact.get("section_evidence_summary") or {}).get("thin_count") or 0),
                    "partial_sections": [str(item or "") for item in (artifact.get("section_evidence_summary") or {}).get("partial_sections") or []],
                    "thin_sections": [str(item or "") for item in (artifact.get("section_evidence_summary") or {}).get("thin_sections") or []],
                },
            }
            for artifact in artifacts
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _delivery_status_summary(
    status: str,
    artifact_count: int,
    recipient_count: int,
    successful_recipient_count: int = 0,
) -> str:
    if status == "sent":
        return f"Delivered {artifact_count} artifact(s) to {recipient_count} recipient(s)."
    if status == "partial":
        return (
            f"Partially delivered {artifact_count} artifact(s) to "
            f"{successful_recipient_count} of {recipient_count} recipient(s)."
        )
    if status == "skipped":
        return "Skipped delivery because no eligible persisted artifacts matched the saved policy."
    return "Recurring delivery failed before any recipient received the package."


async def _tracked_vendors(pool, account_id: Any) -> set[str]:
    rows = await pool.fetch(
        """
        SELECT vendor_name
        FROM tracked_vendors
        WHERE account_id = $1
        """,
        account_id,
    )
    return {
        str(row["vendor_name"] or "").strip().lower()
        for row in rows
        if str(row["vendor_name"] or "").strip()
    }


async def _overridden_report_ids(pool, account_id: Any) -> set[str]:
    rows = await pool.fetch(
        """
        SELECT report_id
        FROM b2b_report_subscriptions
        WHERE account_id = $1
          AND scope_type = 'report'
          AND enabled = TRUE
          AND report_id IS NOT NULL
        """,
        account_id,
    )
    return {str(row["report_id"]) for row in rows if row["report_id"] is not None}


async def _latest_delivery_content_hash(pool, subscription_id: Any) -> str | None:
    value = await pool.fetchval(
        """
        SELECT content_hash
        FROM b2b_report_subscription_delivery_log
        WHERE subscription_id = $1
          AND status = ANY($2::text[])
          AND content_hash IS NOT NULL
        ORDER BY delivered_at DESC
        LIMIT 1
        """,
        subscription_id,
        list(_DELIVERY_CONTENT_HASH_STATUSES),
    )
    text = str(value or "").strip()
    return text or None


async def _claim_delivery_attempt(pool, row, *, dry_run: bool) -> Any | None:
    stale_claim_seconds = settings.b2b_report_delivery.stale_claim_seconds
    delivery_mode = _delivery_mode(dry_run)
    return await pool.fetchrow(
        """
        INSERT INTO b2b_report_subscription_delivery_log (
            subscription_id,
            account_id,
            scheduled_for,
            delivery_mode,
            scope_type,
            scope_key,
            recipient_emails,
            delivery_frequency,
            deliverable_focus,
            freshness_policy,
            status,
            delivered_at
        )
        VALUES (
            $1,
            $2,
            $3,
            $4,
            $5,
            $6,
            $7::text[],
            $8,
            $9,
            $10,
            'processing',
            NOW()
        )
        ON CONFLICT (subscription_id, scheduled_for, delivery_mode) DO UPDATE
        SET recipient_emails = EXCLUDED.recipient_emails,
            delivery_frequency = EXCLUDED.delivery_frequency,
            deliverable_focus = EXCLUDED.deliverable_focus,
            freshness_policy = EXCLUDED.freshness_policy,
            delivered_report_ids = '{}'::uuid[],
            message_ids = '{}'::text[],
            freshness_state = 'none',
            status = 'processing',
            content_hash = NULL,
            summary = '',
            error = NULL,
            delivered_at = NOW()
        WHERE (
                b2b_report_subscription_delivery_log.status = ANY($11::text[])
                AND b2b_report_subscription_delivery_log.delivered_at < NOW() - ($13 || ' seconds')::interval
              )
           OR b2b_report_subscription_delivery_log.status = ANY($12::text[])
        RETURNING id
        """,
        row["id"],
        row["account_id"],
        row["next_delivery_at"],
        delivery_mode,
        row["scope_type"],
        row["scope_key"],
        list(row["recipient_emails"] or []),
        row["delivery_frequency"],
        row["deliverable_focus"],
        row["freshness_policy"],
        list(_DELIVERY_LOG_CLAIMABLE_STALE_STATUSES),
        list(_DELIVERY_LOG_RECLAIMABLE_IMMEDIATE_STATUSES),
        str(stale_claim_seconds),
    )


async def _count_due_non_canary_subscriptions(pool, canary_account_ids: set[str]) -> int:
    if not canary_account_ids:
        return 0
    value = await pool.fetchval(
        """
        SELECT COUNT(*)::int
        FROM b2b_report_subscriptions s
        JOIN saas_accounts sa ON sa.id = s.account_id
        WHERE s.enabled = TRUE
          AND s.next_delivery_at IS NOT NULL
          AND s.next_delivery_at <= NOW()
          AND sa.product IN ('b2b_retention', 'b2b_challenger')
          AND sa.plan_status IN ('trialing', 'active', 'past_due')
          AND NOT (s.account_id::text = ANY($1::text[]))
        """,
        list(canary_account_ids),
    )
    return int(value or 0)


async def _fetch_due_rows(pool, limit: int, canary_account_ids: set[str]) -> list[Any]:
    if canary_account_ids:
        return await pool.fetch(
            """
            SELECT s.id, s.account_id, sa.name AS account_name,
                   s.report_id, s.scope_type, s.scope_key, s.scope_label, s.filter_payload,
                   s.delivery_frequency, s.deliverable_focus, s.freshness_policy,
                   s.recipient_emails, s.delivery_note, s.next_delivery_at
            FROM b2b_report_subscriptions s
            JOIN saas_accounts sa ON sa.id = s.account_id
            WHERE s.enabled = TRUE
              AND s.next_delivery_at IS NOT NULL
              AND s.next_delivery_at <= NOW()
              AND sa.product IN ('b2b_retention', 'b2b_challenger')
              AND sa.plan_status IN ('trialing', 'active', 'past_due')
              AND s.account_id::text = ANY($2::text[])
            ORDER BY s.next_delivery_at ASC, s.created_at ASC
            LIMIT $1
            """,
            limit,
            list(canary_account_ids),
        )
    return await pool.fetch(
        """
        SELECT s.id, s.account_id, sa.name AS account_name,
               s.report_id, s.scope_type, s.scope_key, s.scope_label, s.filter_payload,
               s.delivery_frequency, s.deliverable_focus, s.freshness_policy,
               s.recipient_emails, s.delivery_note, s.next_delivery_at
        FROM b2b_report_subscriptions s
        JOIN saas_accounts sa ON sa.id = s.account_id
        WHERE s.enabled = TRUE
          AND s.next_delivery_at IS NOT NULL
          AND s.next_delivery_at <= NOW()
          AND sa.product IN ('b2b_retention', 'b2b_challenger')
          AND sa.plan_status IN ('trialing', 'active', 'past_due')
        ORDER BY s.next_delivery_at ASC, s.created_at ASC
        LIMIT $1
        """,
        limit,
    )


async def _finalize_delivery_attempt(
    pool,
    *,
    delivery_log_id: Any,
    status: str,
    freshness_state: str,
    delivered_report_ids: list[Any],
    message_ids: list[str],
    content_hash: str | None,
    summary: str,
    error: str | None,
) -> None:
    await pool.execute(
        """
        UPDATE b2b_report_subscription_delivery_log
        SET status = $2,
            freshness_state = $3,
            delivered_report_ids = $4::uuid[],
            message_ids = $5::text[],
            content_hash = $6,
            summary = $7,
            error = $8,
            delivered_at = NOW()
        WHERE id = $1
        """,
        delivery_log_id,
        status,
        freshness_state,
        delivered_report_ids,
        message_ids,
        content_hash,
        summary[:1000],
        (error or "")[:2000] or None,
    )


async def _advance_subscription(
    pool,
    subscription_id: Any,
    delivery_frequency: str,
    scheduled_for: datetime | None,
) -> None:
    await pool.execute(
        """
        UPDATE b2b_report_subscriptions
        SET next_delivery_at = $2
        WHERE id = $1
        """,
        subscription_id,
        _next_delivery_at(delivery_frequency, scheduled_for),
    )


def _should_advance_subscription(status: str, *, skip_reason: str | None = None) -> bool:
    if status in {"sent", "partial"}:
        return True
    if status != "skipped":
        return False
    return skip_reason in {
        "unchanged_package",
        "all_recipients_suppressed",
    }


async def _fetch_report_row(pool, report_id: Any) -> Any | None:
    return await pool.fetchrow(
        """
        SELECT id, account_id, report_date, report_type, executive_summary,
               vendor_filter, category_filter, status, created_at,
               blocker_count, warning_count,
               (
                 SELECT COUNT(*)
                 FROM pipeline_visibility_reviews r
                 JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
                 WHERE r.status = 'open'
                   AND e.entity_type = 'churn_report'
                   AND e.entity_id = b2b_intelligence.id::text
               ) AS unresolved_issue_count,
               intelligence_data, data_density,
               COALESCE(intelligence_data->>'quality_status', intelligence_data->'battle_card_quality'->>'status')
                   AS quality_status
        FROM b2b_intelligence
        WHERE id = $1
        """,
        report_id,
    )


async def _fetch_library_rows(
    pool,
    account_id: Any,
    tracked_vendors: set[str],
    focus: str,
    filter_payload: dict[str, str] | None = None,
) -> list[Any]:
    allowed_types = _REPORT_TYPE_FOCUS_MAP.get(focus)
    requested_report_type = str((filter_payload or {}).get("report_type") or "").strip()
    if requested_report_type:
        if allowed_types and requested_report_type not in allowed_types:
            return []
        allowed_types = {requested_report_type}
    requested_vendor_filter = str((filter_payload or {}).get("vendor_filter") or "").strip()
    scan_limit = settings.b2b_report_delivery.max_reports_per_delivery * 5
    return await pool.fetch(
        """
        SELECT id, account_id, report_date, report_type, executive_summary,
               vendor_filter, category_filter, status, created_at,
               blocker_count, warning_count,
               (
                 SELECT COUNT(*)
                 FROM pipeline_visibility_reviews r
                 JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
                 WHERE r.status = 'open'
                   AND e.entity_type = 'churn_report'
                   AND e.entity_id = b2b_intelligence.id::text
               ) AS unresolved_issue_count,
               intelligence_data, data_density,
               COALESCE(intelligence_data->>'quality_status', intelligence_data->'battle_card_quality'->>'status')
                   AS quality_status
        FROM b2b_intelligence
        WHERE status = ANY($5::text[])
          AND (
                vendor_filter IS NULL
                OR vendor_filter = ''
                OR LOWER(vendor_filter) = ANY($2::text[])
                OR account_id = $1
          )
          AND ($3::text[] IS NULL OR report_type = ANY($3::text[]))
          AND ($6::text = '' OR vendor_filter ILIKE '%' || $6 || '%')
        ORDER BY report_date DESC NULLS LAST, created_at DESC NULLS LAST, id DESC
        LIMIT $4
        """,
        account_id,
        list(tracked_vendors),
        list(allowed_types) if allowed_types else None,
        scan_limit,
        ["completed", "published", "sales_ready"],
        requested_vendor_filter,
    )


def _build_delivery_artifact(row) -> dict[str, Any]:
    intelligence_data = _safe_json(row["intelligence_data"])
    if not isinstance(intelligence_data, dict):
        intelligence_data = {}
    data_density = _safe_json(row["data_density"])
    if not isinstance(data_density, dict):
        data_density = {}
    freshness = _derive_freshness(row, intelligence_data, data_density)
    artifact_state = report_artifact_payload(row["status"])
    review = report_review_payload(
        _row_int(row, "blocker_count"),
        _row_int(row, "warning_count"),
        _row_int(row, "unresolved_issue_count"),
    )
    section_evidence = report_section_evidence_payload(intelligence_data)
    executive_summary = str(
        row["executive_summary"]
        or intelligence_data.get("executive_summary")
        or "No executive summary is attached to this artifact yet."
    ).strip()
    return {
        "report_id": row["id"],
        "report_type": str(row["report_type"] or ""),
        "title": _report_display_title(row),
        "type_label": _format_report_type_label(str(row["report_type"] or "")),
        "trust_label": _report_trust_label(row, intelligence_data),
        "artifact_state": artifact_state["state"],
        "artifact_label": artifact_state["label"],
        "quality_status": _quality_status(row, intelligence_data),
        "section_evidence": section_evidence,
        "section_evidence_summary": _section_evidence_summary(section_evidence),
        "blocker_count": _row_int(row, "blocker_count"),
        "warning_count": _row_int(row, "warning_count"),
        "unresolved_issue_count": _row_int(row, "unresolved_issue_count"),
        "freshness_state": freshness["state"],
        "freshness_label": freshness["label"],
        "freshness_detail": freshness["detail"],
        "review_state": review["state"],
        "review_label": review["label"],
        "trust": {
            "artifact_state": artifact_state["state"],
            "artifact_label": artifact_state["label"],
            "freshness_state": freshness["state"],
            "freshness_label": freshness["label"],
            "review_state": review["state"],
            "review_label": review["label"],
        },
        "executive_summary": executive_summary,
        "evidence_highlights": _extract_evidence_highlights(intelligence_data),
        "report_url": _report_url(row["id"]),
    }


async def _resolve_artifact_selection(pool, row, tracked_vendors: set[str]) -> _ArtifactSelectionResult:
    scope_type = str(row["scope_type"] or "")
    freshness_policy = str(row["freshness_policy"] or "fresh_or_monitor")
    filter_payload = _normalize_subscription_filter_payload(
        _safe_json(_row_value(row, "filter_payload"))
    )

    if scope_type == "report":
        report_row = await _fetch_report_row(pool, row["report_id"])
        if not report_row:
            return _empty_selection_result()
        intelligence_data = _safe_json(report_row["intelligence_data"])
        if not isinstance(intelligence_data, dict):
            intelligence_data = {}
        if not _artifact_ready(report_row, intelligence_data):
            return _empty_selection_result()
        if not _tracked_vendor_allowed(report_row, tracked_vendors, row["account_id"]):
            return _empty_selection_result()
        if _delivery_eligibility_reason(report_row, intelligence_data):
            return _empty_selection_result()
        artifact = _build_delivery_artifact(report_row)
        selection = _ArtifactSelectionResult(
            artifacts=[],
            eligible_before_freshness_count=1,
        )
        if not _freshness_allowed(freshness_policy, artifact["freshness_state"]):
            selection.freshness_blocked_count = 1
            return selection
        selection.artifacts.append(artifact)
        return selection

    rows = await _fetch_library_rows(
        pool,
        row["account_id"],
        tracked_vendors,
        str(row["deliverable_focus"] or "all"),
        filter_payload if scope_type == "library_view" else None,
    )
    overridden_report_ids: set[str] = set()
    if settings.b2b_report_delivery.report_scope_overrides_library:
        overridden_report_ids = await _overridden_report_ids(pool, row["account_id"])
    selection = _empty_selection_result()
    for report_row in rows:
        report_type = str(report_row["report_type"] or "")
        if not _focus_allowed(scope_type, str(row["deliverable_focus"] or "all"), report_type):
            continue
        if str(report_row["id"]) in overridden_report_ids:
            continue
        intelligence_data = _safe_json(report_row["intelligence_data"])
        if not isinstance(intelligence_data, dict):
            intelligence_data = {}
        if scope_type == "library_view" and not _report_matches_subscription_filters(
            report_row,
            intelligence_data,
            filter_payload,
        ):
            continue
        if not _artifact_ready(report_row, intelligence_data):
            continue
        if _delivery_eligibility_reason(report_row, intelligence_data):
            continue
        artifact = _build_delivery_artifact(report_row)
        selection.eligible_before_freshness_count += 1
        if not _freshness_allowed(freshness_policy, artifact["freshness_state"]):
            selection.freshness_blocked_count += 1
            continue
        selection.artifacts.append(artifact)
        if len(selection.artifacts) >= settings.b2b_report_delivery.max_reports_per_delivery:
            break
    return selection


async def _resolve_artifacts(pool, row, tracked_vendors: set[str]) -> list[dict[str, Any]]:
    selection = await _resolve_artifact_selection(pool, row, tracked_vendors)
    return selection.artifacts


async def _send_subscription_email(
    pool,
    sender,
    *,
    recipient: str,
    from_addr: str,
    subject: str,
    html_body: str,
    tags: list[dict[str, str]],
) -> tuple[bool, str | None, str | None, bool]:
    suppression = await is_suppressed(pool, email=recipient)
    if suppression:
        reason = str(suppression.get("reason") or "suppressed").strip() or "suppressed"
        return False, None, f"{recipient}: suppressed ({reason})", True

    headers = _unsub_headers(settings.campaign_sequence.unsubscribe_base_url, recipient)
    body = _wrap_with_footer(html_body, recipient, settings.campaign_sequence)
    delivery_cfg = settings.b2b_report_delivery
    attempts = max(1, int(delivery_cfg.max_send_attempts_per_recipient))
    backoff_seconds = max(0, int(delivery_cfg.retry_backoff_seconds))
    last_error: str | None = None

    for attempt in range(1, attempts + 1):
        try:
            result = await sender.send(
                to=recipient,
                from_email=from_addr,
                subject=subject,
                body=body,
                headers=headers or None,
                tags=tags,
            )
            message_id = str(result.get("id") or "").strip() or None
            return True, message_id, None, False
        except Exception as exc:
            last_error = str(exc)
            if attempt < attempts and backoff_seconds > 0:
                await asyncio.sleep(backoff_seconds)

    return False, None, f"{recipient}: {last_error or 'send failed'}", False


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Deliver due persisted report subscriptions through the campaign sender."""
    delivery_cfg = settings.b2b_report_delivery
    if not delivery_cfg.enabled:
        return {"_skip_synthesis": "Recurring report delivery disabled"}

    dry_run = _dry_run_enabled(task)
    canary_account_ids = _canary_account_ids(task)

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    deferred_non_canary = await _count_due_non_canary_subscriptions(pool, canary_account_ids)
    due_rows = await _fetch_due_rows(
        pool,
        int(delivery_cfg.max_subscriptions_per_run),
        canary_account_ids,
    )

    if not due_rows:
        return {
            "_skip_synthesis": (
                "No due report subscriptions in canary scope"
                if canary_account_ids
                else "No due report subscriptions"
            ),
            "dry_run": dry_run,
            "deferred_non_canary": deferred_non_canary,
        }

    sender = None
    from_email = ""
    if not dry_run:
        from_email = _campaign_from_email()
        if not from_email:
            return {"_skip_synthesis": "Campaign sender not configured"}

        try:
            sender = get_campaign_sender()
        except Exception as exc:
            logger.warning("Recurring report delivery sender unavailable: %s", exc)
            return {"_skip_synthesis": "Campaign sender unavailable"}

    attempts_claimed = 0
    delivered = 0
    partially_delivered = 0
    dry_run_deliveries = 0
    skipped = 0
    failed = 0

    for row in due_rows:
        claim = await _claim_delivery_attempt(pool, row, dry_run=dry_run)
        if not claim:
            continue

        attempts_claimed += 1
        delivery_log_id = claim["id"]
        recipients = list(row["recipient_emails"] or [])
        status = "failed"
        summary = "Recurring delivery failed before any recipient received the package."
        error_message = None
        message_ids: list[str] = []
        delivered_report_ids: list[Any] = []
        freshness_state = "none"
        content_hash: str | None = None
        skip_reason: str | None = None

        try:
            tracked_vendors = await _tracked_vendors(pool, row["account_id"])
            selection = await _resolve_artifact_selection(pool, row, tracked_vendors)
            artifacts = selection.artifacts
            delivered_report_ids = [artifact["report_id"] for artifact in artifacts]
            freshness_state = _aggregate_freshness_state(artifacts)

            if not recipients:
                error_message = "Subscription has no recipients"
                status = "failed"
            elif not artifacts:
                blocked_summary, blocked_freshness_state, blocked_skip_reason = await _blocked_delivery_skip_details(
                    pool,
                    row,
                    selection,
                )
                status = "skipped"
                summary = blocked_summary or (
                    "Skipped delivery because no eligible persisted artifacts matched the saved policy."
                )
                freshness_state = blocked_freshness_state
                skip_reason = blocked_skip_reason or "no_eligible_artifacts"
            else:
                content_hash = _delivery_content_hash(row, artifacts)
                if delivery_cfg.suppress_unchanged_deliveries:
                    latest_content_hash = await _latest_delivery_content_hash(pool, row["id"])
                    if latest_content_hash and latest_content_hash == content_hash:
                        status = "skipped"
                        summary = (
                            "Skipped delivery because the eligible report package has not materially "
                            "changed since the last completed delivery."
                        )
                        skip_reason = "unchanged_package"

                if dry_run and status != "skipped":
                    status = "dry_run"
                    summary = (
                        f"Dry run: would deliver {len(artifacts)} artifact(s) "
                        f"to {len(recipients)} recipient(s)."
                    )

                if status not in {"skipped", "dry_run"}:
                    tags = [
                        {"name": "task", "value": "b2b_report_subscription_delivery"},
                        {"name": "subscription_id", "value": str(row["id"])},
                        {"name": "scope_type", "value": str(row["scope_type"] or "")},
                    ]
                    successful_recipient_count = 0
                    suppressed_recipient_count = 0
                    send_failures: list[str] = []
                    sender_name = str(delivery_cfg.sender_name or "").strip()
                    from_addr = f"{sender_name} <{from_email}>" if sender_name else from_email
                    subject = (
                        f"{artifacts[0]['title']} | Recurring delivery"
                        if row["scope_type"] == "report" and len(artifacts) == 1
                        else f"{row['account_name']}: {len(artifacts)} report artifact(s) ready"
                    )
                    summary_line = _subscription_summary(str(row["scope_type"] or ""), artifacts)
                    manage_url = _scope_manage_url_with_filters(
                        str(row["scope_type"] or ""),
                        str(row["scope_key"] or ""),
                        _row_value(row, "filter_payload"),
                    )
                    html_body = render_report_subscription_delivery_html(
                        account_name=str(row["account_name"] or ""),
                        scope_label=str(row["scope_label"] or "Recurring delivery"),
                        summary_line=summary_line,
                        frequency_label=_frequency_label(str(row["delivery_frequency"] or "")),
                        manage_url=manage_url,
                        delivery_note=str(row["delivery_note"] or ""),
                        artifacts=artifacts,
                    )

                    for recipient in recipients:
                        sent, message_id, send_error, suppressed = await _send_subscription_email(
                            pool,
                            sender,
                            recipient=recipient,
                            from_addr=from_addr,
                            subject=subject,
                            html_body=html_body,
                            tags=tags,
                        )
                        if sent:
                            successful_recipient_count += 1
                            if message_id:
                                message_ids.append(message_id)
                        elif send_error:
                            if suppressed:
                                suppressed_recipient_count += 1
                            send_failures.append(send_error)

                    if suppressed_recipient_count == len(recipients):
                        status = "skipped"
                        error_message = "; ".join(send_failures)
                        summary = "Skipped delivery because every recipient on the subscription is currently suppressed."
                        skip_reason = "all_recipients_suppressed"
                    elif send_failures and successful_recipient_count == 0:
                        status = "failed"
                        error_message = "; ".join(send_failures)
                    elif send_failures:
                        status = "partial"
                        error_message = "; ".join(send_failures)
                    else:
                        status = "sent"

                    if status != "skipped":
                        summary = _delivery_status_summary(
                            status,
                            len(artifacts),
                            len(recipients),
                            successful_recipient_count,
                        )

            await _finalize_delivery_attempt(
                pool,
                delivery_log_id=delivery_log_id,
                status=status,
                freshness_state=freshness_state,
                delivered_report_ids=delivered_report_ids,
                message_ids=message_ids,
                content_hash=content_hash,
                summary=summary,
                error=error_message,
            )

            if not dry_run and _should_advance_subscription(status, skip_reason=skip_reason):
                await _advance_subscription(
                    pool,
                    row["id"],
                    str(row["delivery_frequency"] or "weekly"),
                    row["next_delivery_at"],
                )

            if status == "sent":
                delivered += 1
            elif status == "partial":
                partially_delivered += 1
            elif status == "dry_run":
                dry_run_deliveries += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
        except Exception as exc:
            logger.exception("Recurring report delivery failed for subscription %s", row["id"])
            await _finalize_delivery_attempt(
                pool,
                delivery_log_id=delivery_log_id,
                status="failed",
                freshness_state=freshness_state,
                delivered_report_ids=delivered_report_ids,
                message_ids=message_ids,
                content_hash=content_hash,
                summary=summary,
                error=str(exc),
            )
            failed += 1

    if attempts_claimed == 0:
        return {"_skip_synthesis": "No report subscriptions were claimable"}

    return {
        "_skip_synthesis": "Recurring report delivery complete",
        "due_subscriptions": len(due_rows),
        "attempts_claimed": attempts_claimed,
        "delivered": delivered,
        "partially_delivered": partially_delivered,
        "dry_run_deliveries": dry_run_deliveries,
        "skipped": skipped,
        "failed": failed,
        "dry_run": dry_run,
        "deferred_non_canary": deferred_non_canary,
    }
