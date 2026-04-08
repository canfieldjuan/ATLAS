from __future__ import annotations

from typing import Any

_THESIS_LIMIT = 2
_TIMING_WINDOW_LIMIT = 2
_PROOF_POINT_LIMIT = 2
_ACCOUNT_SIGNAL_LIMIT = 2
_COVERAGE_LIMIT_LIMIT = 3
_DELTA_ITEM_LIMIT = 3


def _copy_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _copy_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_list(values: Any) -> list[str]:
    return [
        _clean_text(value)
        for value in _copy_list(values)
        if _clean_text(value)
    ]


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
        for item in _copy_list(consumer_context.get("timing_windows"))[:_TIMING_WINDOW_LIMIT]
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
        for item in _copy_list(consumer_context.get("proof_points"))[:_PROOF_POINT_LIMIT]
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
        for item in _copy_list(consumer_context.get("account_signals"))[:_ACCOUNT_SIGNAL_LIMIT]
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
        for item in _copy_list(consumer_context.get("coverage_limits"))[:_COVERAGE_LIMIT_LIMIT]
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


def campaign_review_reasoning_context(
    company_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(company_context, dict):
        return {}
    reasoning_context = _copy_dict(company_context.get("reasoning_context"))
    scope_summary = campaign_reasoning_scope_summary(
        _copy_dict(company_context.get("reasoning_scope_summary"))
        or _copy_dict(reasoning_context.get("scope_summary")),
    )
    atom_context = _copy_dict(company_context.get("reasoning_atom_context")) or _copy_dict(
        reasoning_context.get("atom_context"),
    )
    delta_summary = campaign_reasoning_delta_summary(
        _copy_dict(company_context.get("reasoning_delta_summary"))
        or _copy_dict(reasoning_context.get("delta_summary")),
    )
    enriched_account_signals = _enrich_review_account_signals(company_context, atom_context)
    if enriched_account_signals:
        atom_context["account_signals"] = enriched_account_signals
    result: dict[str, Any] = {}
    if scope_summary:
        result["reasoning_scope_summary"] = scope_summary
    if atom_context:
        result["reasoning_atom_context"] = atom_context
    if delta_summary:
        result["reasoning_delta_summary"] = delta_summary
    return result


def _enrich_review_account_signals(
    company_context: dict[str, Any],
    atom_context: dict[str, Any],
) -> list[dict[str, Any]]:
    seed_company = _clean_text(company_context.get("company"))
    seed_competitor = _clean_text(
        (_clean_list(company_context.get("competitors_considering")) or [""])[0],
    )
    seed_pain = _clean_text(
        (_clean_list(company_context.get("pain_categories")) or [""])[0],
    )
    seed_quote = _clean_text((_clean_list(company_context.get("key_quotes")) or [""])[0])
    seed = {
        "company": seed_company,
        "role_type": _clean_text(company_context.get("role_type")),
        "buying_stage": _clean_text(company_context.get("buying_stage")),
        "urgency": company_context.get("urgency"),
        "competitor_context": seed_competitor,
        "contract_end": _clean_text(company_context.get("contract_end")),
        "decision_timeline": _clean_text(company_context.get("decision_timeline")),
        "primary_pain": seed_pain,
        "quote": seed_quote,
        "trust_tier": "medium",
    }
    signals: list[dict[str, Any]] = []
    for index, item in enumerate(_copy_list(atom_context.get("account_signals"))[:_ACCOUNT_SIGNAL_LIMIT]):
        if not isinstance(item, dict):
            continue
        signal = {
            "company": _clean_text(item.get("company")) or seed["company"],
            "role_type": _clean_text(item.get("role_type")) or seed["role_type"],
            "buying_stage": _clean_text(item.get("buying_stage")) or seed["buying_stage"],
            "urgency": item.get("urgency") if item.get("urgency") not in (None, "") else seed["urgency"],
            "competitor_context": _clean_text(item.get("competitor_context")) or seed["competitor_context"],
            "contract_end": _clean_text(item.get("contract_end")) or seed["contract_end"],
            "decision_timeline": _clean_text(item.get("decision_timeline")) or seed["decision_timeline"],
            "primary_pain": _clean_text(item.get("primary_pain")) or seed["primary_pain"],
            "quote": _clean_text(item.get("quote")) or (seed["quote"] if index == 0 else ""),
            "trust_tier": _clean_text(item.get("trust_tier")) or seed["trust_tier"],
        }
        if any(
            signal[key]
            for key in (
                "company",
                "buying_stage",
                "competitor_context",
                "contract_end",
                "decision_timeline",
                "primary_pain",
                "quote",
            )
        ) or signal["urgency"] not in (None, ""):
            signals.append(signal)
    if signals:
        return signals
    if any(
        seed[key]
        for key in (
            "company",
            "role_type",
            "buying_stage",
            "competitor_context",
            "contract_end",
            "decision_timeline",
            "primary_pain",
            "quote",
        )
    ) or seed["urgency"] not in (None, ""):
        return [seed]
    return []
