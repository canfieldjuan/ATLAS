"""Small campaign reasoning context adapters for copied campaign tasks."""

from __future__ import annotations

from typing import Any


_THESIS_LIMIT = 2
_TIMING_WINDOW_LIMIT = 2
_PROOF_POINT_LIMIT = 2
_ACCOUNT_SIGNAL_LIMIT = 2
_COVERAGE_LIMIT_LIMIT = 3
_DELTA_ITEM_LIMIT = 3


def _copy_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def campaign_reasoning_scope_summary(
    scope_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(scope_manifest, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in (
        "selection_strategy",
        "reviews_considered_total",
        "reviews_in_scope",
        "witnesses_in_scope",
        "witness_mix",
    ):
        value = scope_manifest.get(key)
        if value not in (None, "", [], {}):
            summary[key] = value
    return summary


def campaign_reasoning_atom_context(
    consumer_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(consumer_context, dict):
        return {}
    context: dict[str, Any] = {}
    theses = [
        {
            "wedge": _clean_text(item.get("wedge")),
            "summary": _clean_text(item.get("summary")),
            "why_now": _clean_text(item.get("why_now")),
            "confidence": _clean_text(item.get("confidence")),
        }
        for item in _copy_list(consumer_context.get("theses"))[:_THESIS_LIMIT]
        if isinstance(item, dict)
    ]
    theses = [item for item in theses if item["summary"] or item["why_now"]]
    if theses:
        context["top_theses"] = theses

    timing_windows = [
        {
            "window_type": _clean_text(item.get("window_type")),
            "anchor": _clean_text(item.get("start_or_anchor")),
            "urgency": _clean_text(item.get("urgency")),
            "recommended_action": _clean_text(item.get("recommended_action")),
        }
        for item in _copy_list(consumer_context.get("timing_windows"))[
            :_TIMING_WINDOW_LIMIT
        ]
        if isinstance(item, dict)
    ]
    timing_windows = [item for item in timing_windows if item["anchor"]]
    if timing_windows:
        context["timing_windows"] = timing_windows

    proof_points = [
        {
            "label": _clean_text(item.get("label")),
            "value": item.get("value"),
            "interpretation": _clean_text(item.get("interpretation")),
        }
        for item in _copy_list(consumer_context.get("proof_points"))[
            :_PROOF_POINT_LIMIT
        ]
        if isinstance(item, dict)
    ]
    proof_points = [item for item in proof_points if item["label"]]
    if proof_points:
        context["proof_points"] = proof_points

    account_signals = [
        {
            "company": _clean_text(item.get("company")),
            "buying_stage": _clean_text(item.get("buying_stage")),
            "competitor_context": _clean_text(item.get("competitor_context")),
            "primary_pain": _clean_text(item.get("primary_pain")),
        }
        for item in _copy_list(consumer_context.get("account_signals"))[
            :_ACCOUNT_SIGNAL_LIMIT
        ]
        if isinstance(item, dict)
    ]
    account_signals = [
        item
        for item in account_signals
        if item["company"] or item["primary_pain"] or item["competitor_context"]
    ]
    if account_signals:
        context["account_signals"] = account_signals

    coverage_limits = [
        _clean_text(item.get("label"))
        for item in _copy_list(consumer_context.get("coverage_limits"))[
            :_COVERAGE_LIMIT_LIMIT
        ]
        if isinstance(item, dict) and _clean_text(item.get("label"))
    ]
    if coverage_limits:
        context["coverage_limits"] = coverage_limits
    return context


def campaign_reasoning_delta_summary(
    reasoning_delta: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(reasoning_delta, dict):
        return {}
    summary: dict[str, Any] = {"changed": bool(reasoning_delta.get("changed"))}
    for key in (
        "wedge_changed",
        "confidence_changed",
        "top_destination_changed",
    ):
        if key in reasoning_delta:
            summary[key] = bool(reasoning_delta.get(key))
    for key in ("theses_added", "new_timing_windows", "new_account_signals"):
        value = reasoning_delta.get(key)
        if isinstance(value, list) and value:
            summary[key] = value[:_DELTA_ITEM_LIMIT]
    return summary


__all__ = [
    "campaign_reasoning_atom_context",
    "campaign_reasoning_delta_summary",
    "campaign_reasoning_scope_summary",
]
