"""Pure delta computation for paid deflection report models."""

from __future__ import annotations

import re
from typing import Any, Mapping


DEFLECTION_DELTA_SCHEMA_VERSION = "deflection_delta.v1"
SUPPORTED_DEFLECTION_MODEL_SCHEMA_VERSION = "deflection.v1"

_UNRESOLVED = {"Needs answer", "Needs review"}
_RESURFACED = "Already covered but still recurring"
_CSAT_FIELDS = (
    "status",
    "csat_present_count",
    "negative_csat_ticket_count",
    "numeric_average",
)
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def compute_deflection_delta(
    current_model: Mapping[str, Any],
    baseline_model: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two paid `deflection.v1` models without DB, network, or rendering."""

    _require_model(current_model, "current_model")
    _require_model(baseline_model, "baseline_model")
    current_rows = _action_rows(current_model)
    baseline_rows = _action_rows(baseline_model)
    ambiguous_keys = _non_unique_keys(current_rows) | _non_unique_keys(baseline_rows)
    current = _index_rows(current_rows, "current", ambiguous_keys)
    baseline = _index_rows(baseline_rows, "baseline", ambiguous_keys)
    items = [
        _delta_row(key, current.get(key), baseline.get(key))
        for key in sorted(set(current) | set(baseline))
    ]
    return {
        "schema_version": DEFLECTION_DELTA_SCHEMA_VERSION,
        "current": _metadata(current_model),
        "baseline": _metadata(baseline_model),
        "summary": _summary(items, current, baseline),
        "items": items,
    }


def _require_model(model: Mapping[str, Any], label: str) -> None:
    if _text(model.get("schema_version")) != SUPPORTED_DEFLECTION_MODEL_SCHEMA_VERSION:
        raise ValueError(
            f"{label} must use schema_version "
            f"{SUPPORTED_DEFLECTION_MODEL_SCHEMA_VERSION!r}"
        )


def _index_rows(
    rows: list[Mapping[str, Any]],
    side: str,
    ambiguous_keys: set[str],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        repeat_key = _text(row.get("repeat_key"))
        cluster_id = _text(row.get("cluster_id"))
        confidence = _text(row.get("identity_confidence")) or "low"
        key = repeat_key or cluster_id
        if confidence == "low" or not key or key in ambiguous_keys:
            key = f"unmatched:{side}:{index}"
            confidence = "low"
        indexed[key] = {
            "identity_key": key,
            "repeat_key": repeat_key,
            "cluster_id": cluster_id,
            "identity_basis": _text(row.get("identity_basis")) or "unknown",
            "identity_confidence": confidence,
            "row": dict(row),
        }
    return indexed


def _non_unique_keys(rows: list[Mapping[str, Any]]) -> set[str]:
    counts: dict[str, int] = {}
    for row in rows:
        key = _text(row.get("repeat_key")) or _text(row.get("cluster_id"))
        if key:
            counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count > 1}


def _action_rows(model: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sections = model.get("sections")
    if not isinstance(sections, list):
        return []
    by_id = {
        _text(section.get("id")): section
        for section in sections
        if isinstance(section, Mapping)
    }
    section = by_id.get("backlog_table") or by_id.get("priority_fix_queue") or {}
    data = section.get("data") if isinstance(section, Mapping) else {}
    items = data.get("items") if isinstance(data, Mapping) else ()
    return [item for item in items if isinstance(item, Mapping)] if isinstance(items, list) else []


def _delta_row(
    key: str,
    current: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    current_row = current.get("row") if current else None
    baseline_row = baseline.get("row") if baseline else None
    current_count = _number(_get(current_row, "ticket_count"))
    baseline_count = _number(_get(baseline_row, "ticket_count"))
    current_cost = _number(_get(current_row, "estimated_support_cost"))
    baseline_cost = _number(_get(baseline_row, "estimated_support_cost"))
    current_status = _text(_get(current_row, "status"))
    baseline_status = _text(_get(baseline_row, "status"))
    identity = current or baseline or {}
    effective_confidence = _effective_confidence(current, baseline)
    return {
        "identity_key": key,
        "repeat_key": _text(identity.get("repeat_key")),
        "cluster_id": _text(identity.get("cluster_id")),
        "identity_basis": _text(identity.get("identity_basis")),
        "identity_confidence": effective_confidence,
        "question": _text(_get(current_row, "question") or _get(baseline_row, "question")),
        "owner_lane": _text(_get(current_row, "owner_lane") or _get(baseline_row, "owner_lane")),
        "fix_type": _text(_get(current_row, "fix_type") or _get(baseline_row, "fix_type")),
        "current_status": current_status or None,
        "baseline_status": baseline_status or None,
        "current_ticket_count": current_count,
        "baseline_ticket_count": baseline_count,
        "ticket_count_delta": current_count - baseline_count,
        "current_estimated_support_cost": current_cost,
        "baseline_estimated_support_cost": baseline_cost,
        "support_cost_delta": round(current_cost - baseline_cost, 2),
        "current_csat_signal": _csat(current_row),
        "baseline_csat_signal": _csat(baseline_row),
        "change_types": _change_types(current_row, baseline_row, effective_confidence),
    }


def _change_types(
    current_row: Mapping[str, Any] | None,
    baseline_row: Mapping[str, Any] | None,
    effective_confidence: str,
) -> list[str]:
    current_status = _text(_get(current_row, "status"))
    baseline_status = _text(_get(baseline_row, "status"))
    count_delta = _number(_get(current_row, "ticket_count")) - _number(
        _get(baseline_row, "ticket_count")
    )
    cost_delta = _number(_get(current_row, "estimated_support_cost")) - _number(
        _get(baseline_row, "estimated_support_cost")
    )
    changes: list[str] = []
    if current_row and not baseline_row:
        changes.append("NEW")
    if baseline_row and not current_row:
        changes.append("RESOLVED")
    if current_status == _RESURFACED:
        changes.append("RESURFACED")
    if current_row and baseline_row and current_status in _UNRESOLVED:
        if baseline_status in _UNRESOLVED:
            changes.append("STILL_UNRESOLVED")
    if count_delta > 0:
        changes.append("GROWING")
    elif count_delta < 0:
        changes.append("SHRINKING")
    if current_row and baseline_row and current_status != baseline_status:
        changes.append("STATUS_CHANGED")
    if round(cost_delta, 2) != 0:
        changes.append("COST_CHANGED")
    if _csat_value(current_row) != _csat_value(baseline_row):
        changes.append("CSAT_CHANGED")
    if effective_confidence == "low":
        changes.append("LOW_CONFIDENCE_IDENTITY")
    return changes or ["STABLE"]


def _effective_confidence(
    current: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
) -> str:
    confidences = [
        _text(item.get("identity_confidence"))
        for item in (current, baseline)
        if isinstance(item, Mapping)
    ]
    if "low" in confidences:
        return "low"
    if "medium" in confidences:
        return "medium"
    return confidences[0] if confidences else "low"


def _summary(
    items: list[Mapping[str, Any]],
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    summary = {
        "current_item_count": len(current),
        "baseline_item_count": len(baseline),
        "matched_item_count": sum(
            1 for item in items if item["current_status"] and item["baseline_status"]
        ),
        "support_cost_delta": round(sum(float(item["support_cost_delta"]) for item in items), 2),
    }
    for change in (
        "NEW",
        "RESOLVED",
        "RESURFACED",
        "GROWING",
        "SHRINKING",
        "STILL_UNRESOLVED",
        "STATUS_CHANGED",
        "COST_CHANGED",
        "CSAT_CHANGED",
        "LOW_CONFIDENCE_IDENTITY",
        "STABLE",
    ):
        summary[f"{change.lower()}_count"] = sum(
            1 for item in items if change in item["change_types"]
        )
    return summary


def _metadata(model: Mapping[str, Any]) -> dict[str, Any]:
    summary = model.get("summary") if isinstance(model.get("summary"), Mapping) else {}
    return {
        "schema_version": _text(model.get("schema_version")),
        "title": _text(model.get("title")),
        "source_date_start": _text(summary.get("source_date_start")),
        "source_date_end": _text(summary.get("source_date_end")),
        "source_window_days": _int(summary.get("source_window_days")),
    }


def _csat(row: Mapping[str, Any] | None) -> dict[str, Any]:
    signal = _get(row, "csat_signal")
    if not isinstance(signal, Mapping):
        signal = {}
    return {
        "status": _text(signal.get("status")) or "insufficient_data",
        "csat_present_count": _int(signal.get("csat_present_count")),
        "negative_csat_ticket_count": _int(signal.get("negative_csat_ticket_count")),
        "numeric_average": signal.get("numeric_average"),
    }


def _csat_value(row: Mapping[str, Any] | None) -> tuple[Any, ...]:
    signal = _csat(row)
    return tuple(signal.get(field) for field in _CSAT_FIELDS)


def _get(row: Mapping[str, Any] | None, field: str) -> Any:
    return row.get(field) if isinstance(row, Mapping) else None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid deflection delta number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and _NUMBER_RE.fullmatch(value.strip()):
        return float(value)
    raise ValueError(f"invalid deflection delta number: {value!r}")


def _int(value: Any) -> int:
    number = _number(value)
    if not number.is_integer():
        raise ValueError(f"invalid deflection delta integer: {value!r}")
    return int(number)
