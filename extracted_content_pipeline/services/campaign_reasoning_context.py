"""Small campaign reasoning context adapters for copied campaign tasks."""

from __future__ import annotations

from typing import Any

from ..campaign_ports import CampaignReasoningContext


_THESIS_LIMIT = 2
_TIMING_WINDOW_LIMIT = 2
_PROOF_POINT_LIMIT = 2
_ACCOUNT_SIGNAL_LIMIT = 2
_COVERAGE_LIMIT_LIMIT = 3
_DELTA_ITEM_LIMIT = 3


def _copy_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _copy_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_mapping_list(value: Any, *, limit: int | None = None) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for item in _copy_list(value):
        if not isinstance(item, dict):
            continue
        row = {
            str(key): item_value
            for key, item_value in item.items()
            if item_value not in (None, "", [], {})
        }
        if row:
            rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return tuple(rows)


def _clean_reference_ids(value: Any) -> dict[str, tuple[str, ...]]:
    refs: dict[str, tuple[str, ...]] = {}
    if not isinstance(value, dict):
        return refs
    for key, raw_values in value.items():
        if isinstance(raw_values, list):
            values = tuple(
                _clean_text(item)
                for item in raw_values
                if _clean_text(item)
            )
        else:
            text = _clean_text(raw_values)
            values = (text,) if text else ()
        if values:
            refs[str(key)] = values
    return refs


def _clean_anchor_examples(value: Any) -> dict[str, tuple[dict[str, Any], ...]]:
    anchors: dict[str, tuple[dict[str, Any], ...]] = {}
    if not isinstance(value, dict):
        return anchors
    for label, rows in value.items():
        cleaned = _clean_mapping_list(rows)
        if cleaned:
            anchors[_clean_text(label) or "default"] = cleaned
    return anchors


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def _first_list(*values: Any) -> list[Any]:
    for value in values:
        if isinstance(value, list) and value:
            return value
    return []


def _clean_scalar_list(value: Any) -> tuple[str, ...]:
    return tuple(
        _clean_text(item)
        for item in _copy_list(value)
        if _clean_text(item)
    )


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


def normalize_campaign_reasoning_context(value: Any) -> CampaignReasoningContext:
    """Normalize host-provided compressed context into the campaign contract.

    Accepts already-normalized context, Atlas-style ``reasoning_*`` keys, or a
    nested ``reasoning_context`` blob. It intentionally does not import
    ``_b2b_pool_compression``; the host owns compression and passes the
    resulting evidence here.
    """
    if isinstance(value, CampaignReasoningContext):
        return value
    payload = _copy_dict(value)
    nested = _copy_dict(payload.get("reasoning_context"))
    atom_context = _first_dict(
        payload.get("reasoning_atom_context"),
        nested.get("reasoning_atom_context"),
        nested.get("atom_context"),
        nested,
    )
    scope_summary = _first_dict(
        payload.get("reasoning_scope_summary"),
        nested.get("reasoning_scope_summary"),
        nested.get("scope_summary"),
    )
    delta_summary = _first_dict(
        payload.get("reasoning_delta_summary"),
        nested.get("reasoning_delta_summary"),
        nested.get("delta_summary"),
    )
    normalized_delta = (
        campaign_reasoning_delta_summary(delta_summary) if delta_summary else {}
    )
    return CampaignReasoningContext(
        anchor_examples=_clean_anchor_examples(
            _first_dict(
                payload.get("reasoning_anchor_examples"),
                payload.get("anchor_examples"),
                nested.get("reasoning_anchor_examples"),
                nested.get("anchor_examples"),
            )
        ),
        witness_highlights=_clean_mapping_list(
            _first_list(
                payload.get("reasoning_witness_highlights"),
                payload.get("witness_highlights"),
                nested.get("reasoning_witness_highlights"),
                nested.get("witness_highlights"),
            )
        ),
        reference_ids=_clean_reference_ids(
            _first_dict(
                payload.get("reasoning_reference_ids"),
                payload.get("reference_ids"),
                nested.get("reasoning_reference_ids"),
                nested.get("reference_ids"),
            )
        ),
        account_signals=_clean_mapping_list(atom_context.get("account_signals")),
        timing_windows=_clean_mapping_list(atom_context.get("timing_windows")),
        proof_points=_clean_mapping_list(atom_context.get("proof_points")),
        coverage_limits=_clean_scalar_list(atom_context.get("coverage_limits")),
        scope_summary=campaign_reasoning_scope_summary(scope_summary),
        delta_summary=normalized_delta,
    )


def campaign_reasoning_context_payload(
    context: CampaignReasoningContext,
) -> dict[str, Any]:
    """Return prompt-visible normalized context."""
    return context.as_dict()


def campaign_reasoning_context_metadata(
    context: CampaignReasoningContext,
) -> dict[str, Any]:
    """Return storage metadata fields expected by campaign consumers."""
    if not context.has_content():
        return {}
    metadata: dict[str, Any] = {"reasoning_context": context.as_dict()}
    if context.anchor_examples:
        metadata["reasoning_anchor_examples"] = {
            str(label): [dict(row) for row in rows]
            for label, rows in context.anchor_examples.items()
        }
    if context.witness_highlights:
        metadata["reasoning_witness_highlights"] = [
            dict(row) for row in context.witness_highlights
        ]
    if context.reference_ids:
        metadata["reasoning_reference_ids"] = {
            str(key): [str(item) for item in values]
            for key, values in context.reference_ids.items()
        }
    return metadata


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
    "campaign_reasoning_context_metadata",
    "campaign_reasoning_context_payload",
    "campaign_reasoning_atom_context",
    "campaign_reasoning_delta_summary",
    "campaign_reasoning_scope_summary",
    "normalize_campaign_reasoning_context",
]
