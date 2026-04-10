from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

REPORT_QUALITY_STATUSES = frozenset({
    "sales_ready",
    "needs_review",
    "thin_evidence",
    "deterministic_fallback",
})


def coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def report_freshness_label(state: str, *, unknown_label: str = "Freshness unknown") -> str:
    normalized = str(state or "").strip().lower()
    if normalized == "fresh":
        return "Fresh"
    if normalized == "monitor":
        return "Monitor"
    if normalized == "stale":
        return "Stale"
    return unknown_label


def report_freshness_payload(
    anchor: Any,
    *,
    data_stale: bool = False,
    now: datetime | None = None,
    fresh_hours: int = 24,
    monitor_hours: int = 168,
    unknown_label: str = "Freshness unknown",
) -> dict[str, str]:
    if data_stale:
        return {"state": "stale", "label": report_freshness_label("stale", unknown_label=unknown_label)}
    dt = coerce_datetime(anchor)
    if not dt:
        return {"state": "unknown", "label": unknown_label}
    current = now or datetime.now(timezone.utc)
    age_hours = (current - dt.astimezone(timezone.utc)).total_seconds() / 3600
    if age_hours < fresh_hours:
        state = "fresh"
    elif age_hours < monitor_hours:
        state = "monitor"
    else:
        state = "stale"
    return {"state": state, "label": report_freshness_label(state, unknown_label=unknown_label)}


def report_review_payload(
    blocker_count: int,
    warning_count: int,
    unresolved_issue_count: int,
) -> dict[str, str]:
    if blocker_count > 0:
        return {"state": "blocked", "label": "Blocked"}
    if unresolved_issue_count > 0:
        return {"state": "open_review", "label": "Open Review"}
    if warning_count > 0:
        return {"state": "warnings", "label": "Warnings"}
    return {"state": "clean", "label": "Clean"}


def report_artifact_payload(status: Any) -> dict[str, str]:
    normalized = str(status or "").strip().lower()
    if normalized in {"completed", "complete", "succeeded", "success", "ready", "published", "sales_ready"}:
        return {"state": "ready", "label": "Ready"}
    if normalized in {"failed", "error", "cancelled", "blocked"}:
        return {"state": "failed", "label": "Attention needed"}
    if normalized in {"queued", "pending", "running", "processing", "enriching", "repairing"}:
        return {"state": "processing", "label": "Processing"}
    return {"state": "unknown", "label": "Status unknown"}


def report_trust_payload(
    *,
    report_date: Any,
    created_at: Any,
    data_stale: bool,
    blocker_count: int,
    warning_count: int,
    unresolved_issue_count: int,
    status: Any = None,
) -> dict[str, str]:
    freshness = report_freshness_payload(report_date or created_at, data_stale=data_stale)
    review = report_review_payload(blocker_count, warning_count, unresolved_issue_count)
    trust = {
        "freshness_state": freshness["state"],
        "freshness_label": freshness["label"],
        "review_state": review["state"],
        "review_label": review["label"],
    }
    if status is not None:
        artifact = report_artifact_payload(status)
        trust["artifact_state"] = artifact["state"]
        trust["artifact_label"] = artifact["label"]
    return trust
