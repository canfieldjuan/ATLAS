from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

REPORT_QUALITY_STATUSES = frozenset({
    "sales_ready",
    "needs_review",
    "thin_evidence",
    "deterministic_fallback",
})
REPORT_SECTION_EVIDENCE_SUFFIXES = frozenset({
    "_reference_ids",
    "_witness_highlights",
})
REPORT_SECTION_EVIDENCE_SKIP_KEYS = frozenset({
    "executive_summary",
    "quality_status",
    "battle_card_quality",
    "data_stale",
    "reasoning_reference_ids",
    "reasoning_witness_highlights",
    "reference_ids",
    "witness_highlights",
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
    anchor_is_date_only = isinstance(anchor, date) and not isinstance(anchor, datetime)
    dt = coerce_datetime(anchor)
    if not dt:
        return {"state": "unknown", "label": unknown_label}
    current = now or datetime.now(timezone.utc)
    age_hours = (current - dt.astimezone(timezone.utc)).total_seconds() / 3600
    if anchor_is_date_only:
        state = "monitor" if age_hours < monitor_hours else "stale"
    elif age_hours < fresh_hours:
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


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _extract_string_ids(value: Any, key: str) -> list[str]:
    if not _is_record(value):
        return []
    raw = value.get(key)
    if not isinstance(raw, list):
        return []
    return [item.strip() for item in raw if isinstance(item, str) and item.strip()]


def _extract_witness_highlights(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        item for item in value
        if _is_record(item) and isinstance(item.get("witness_id"), str) and item["witness_id"].strip()
    ]


def _is_section_evidence_metadata_key(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in REPORT_SECTION_EVIDENCE_SUFFIXES)


def _is_renderable_section(key: str, value: Any) -> bool:
    if key in REPORT_SECTION_EVIDENCE_SKIP_KEYS or _is_section_evidence_metadata_key(key):
        return False
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    if _is_record(value) and len(value) == 0:
        return False
    return True


def report_section_evidence_payload(
    intelligence_data: Any,
) -> dict[str, dict[str, Any]]:
    if not _is_record(intelligence_data):
        return {}

    section_evidence: dict[str, dict[str, Any]] = {}
    for field_key, value in intelligence_data.items():
        if not _is_renderable_section(field_key, value):
            continue

        section_object = value if _is_record(value) else None
        section_reference_sources = [
            section_object.get("reasoning_reference_ids") if section_object else None,
            section_object.get("reference_ids") if section_object else None,
            intelligence_data.get(f"{field_key}_reference_ids"),
        ]
        section_highlight_sources = [
            section_object.get("reasoning_witness_highlights") if section_object else None,
            section_object.get("witness_highlights") if section_object else None,
            intelligence_data.get(f"{field_key}_witness_highlights"),
        ]

        witness_ids = {
            witness_id
            for source in section_reference_sources
            for witness_id in _extract_string_ids(source, "witness_ids")
        }
        metric_ids = {
            metric_id
            for source in section_reference_sources
            for metric_id in _extract_string_ids(source, "metric_ids")
        }
        section_highlights = [
            highlight
            for source in section_highlight_sources
            for highlight in _extract_witness_highlights(source)
        ]
        witness_ids.update(
            str(highlight["witness_id"]).strip()
            for highlight in section_highlights
            if str(highlight["witness_id"]).strip()
        )

        if witness_ids:
            label = "Witness-backed"
            detail = f"{len(witness_ids)} linked witness citation{'s' if len(witness_ids) != 1 else ''}"
            state = "witness_backed"
        elif metric_ids or any(_is_record(source) for source in section_reference_sources) or any(isinstance(source, list) for source in section_highlight_sources):
            label = "Partial evidence"
            detail = "Section has evidence metadata, but no linked witness citations yet."
            state = "partial"
        else:
            label = "Thin evidence"
            detail = "No linked witness citations for this section yet."
            state = "thin"

        section_evidence[field_key] = {
            "state": state,
            "label": label,
            "detail": detail,
            "witness_count": len(witness_ids),
            "metric_count": len(metric_ids),
        }

    return section_evidence


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
