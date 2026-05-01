from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnrichmentTimelineDeps:
    contains_any: Any
    normalize_compare_text: Any
    has_commercial_context: Any
    has_strong_commercial_context: Any
    has_technical_context: Any
    has_consumer_context: Any
    normalized_low_fidelity_noisy_sources: Any
    text_mentions_name: Any
    timeline_month_day_re: Any
    timeline_slash_date_re: Any
    timeline_iso_date_re: Any
    timeline_explicit_anchor_phrases: tuple[str, ...]
    timeline_relative_anchor_re: Any
    timeline_contract_event_patterns: tuple[Any, ...]
    timeline_decision_deadline_patterns: tuple[str, ...]
    timeline_contract_end_patterns: tuple[str, ...]
    timeline_immediate_patterns: tuple[str, ...]
    timeline_quarter_patterns: tuple[str, ...]
    timeline_year_patterns: tuple[str, ...]
    timeline_decision_patterns: tuple[str, ...]
    timeline_ambiguous_vendor_tokens: set[str]
    timeline_ambiguous_vendor_product_context_patterns: tuple[str, ...]


def normalize_timeline_anchor(anchor: Any) -> str | None:
    text = re.sub(r"\s+", " ", str(anchor or "")).strip(" \t\r\n'\".,;:()[]{}")
    return text.lower() if text else None


def extract_concrete_timeline_anchor(text: Any, *, deps: EnrichmentTimelineDeps) -> str | None:
    raw_text = str(text or "")
    if not raw_text.strip():
        return None
    for pattern in (deps.timeline_month_day_re, deps.timeline_slash_date_re, deps.timeline_iso_date_re):
        match = pattern.search(raw_text)
        if match:
            return normalize_timeline_anchor(match.group(0))
    lowered = raw_text.lower()
    for phrase in deps.timeline_explicit_anchor_phrases:
        index = lowered.find(phrase)
        if index >= 0:
            return normalize_timeline_anchor(raw_text[index:index + len(phrase)])
    match = deps.timeline_relative_anchor_re.search(raw_text)
    if match:
        return normalize_timeline_anchor(match.group(0))
    return None


def extract_contract_end_event_anchor(text: Any, *, deps: EnrichmentTimelineDeps) -> str | None:
    raw_text = str(text or "")
    if not raw_text.strip():
        return None
    for pattern in deps.timeline_contract_event_patterns:
        match = pattern.search(raw_text)
        if not match:
            continue
        anchor = normalize_timeline_anchor(match.group(0))
        if not anchor:
            continue
        if "renew" in anchor:
            return "renewal"
        if "current contract" in anchor:
            return "current contract end"
        return anchor
    return None


def has_timeline_commercial_signal(
    result: dict[str, Any],
    source_row: dict[str, Any] | None = None,
    *,
    deps: EnrichmentTimelineDeps,
) -> bool:
    churn = result.get("churn_signals") or {}
    review_norm = ""
    review_blob = ""
    source = ""
    if source_row is not None:
        review_blob = " ".join(
            str(source_row.get(field) or "")
            for field in ("summary", "review_text", "pros", "cons")
        )
        source = str(source_row.get("source") or "").strip().lower()
        review_norm = deps.normalize_compare_text(review_blob)

    structured_churn = any((
        bool(churn.get("intent_to_leave")),
        bool(churn.get("actively_evaluating")),
        bool(churn.get("migration_in_progress")),
        bool(churn.get("contract_renewal_mentioned")),
    ))
    strong_signal = any((
        structured_churn,
        bool(result.get("competitors_mentioned")),
        bool(result.get("pricing_phrases")),
        deps.has_strong_commercial_context(review_norm),
    ))
    soft_signal = any((
        bool(result.get("specific_complaints")),
        bool(result.get("event_mentions")),
    ))
    if source_row is not None:
        noisy_sources = deps.normalized_low_fidelity_noisy_sources()
        if source in noisy_sources:
            vendor_norm = deps.normalize_compare_text(source_row.get("vendor_name"))
            product_norm = deps.normalize_compare_text(source_row.get("product_name"))
            product_hit = (
                bool(source_row.get("product_name"))
                and product_norm != vendor_norm
                and deps.text_mentions_name(review_norm, source_row.get("product_name"))
            )
            vendor_hit = (
                bool(source_row.get("vendor_name"))
                and deps.text_mentions_name(review_norm, source_row.get("vendor_name"))
            )
            if vendor_norm in deps.timeline_ambiguous_vendor_tokens and vendor_hit:
                vendor_hit = deps.contains_any(review_blob, deps.timeline_ambiguous_vendor_product_context_patterns)
            vendor_reference = product_hit or vendor_hit
            if not vendor_reference and not structured_churn:
                return False

    return any((
        strong_signal,
        soft_signal and deps.has_commercial_context(review_norm),
    ))


def derive_concrete_timeline_fields(
    result: dict[str, Any],
    source_row: dict[str, Any] | None = None,
    *,
    deps: EnrichmentTimelineDeps,
) -> tuple[str | None, str | None]:
    churn = result.get("churn_signals") or {}
    timeline = result.get("timeline") or {}
    contract_end = normalize_timeline_anchor(timeline.get("contract_end"))
    evaluation_deadline = normalize_timeline_anchor(timeline.get("evaluation_deadline"))
    if contract_end and evaluation_deadline:
        return contract_end, evaluation_deadline

    candidates: list[tuple[str, str]] = []
    for event in result.get("event_mentions") or []:
        if not isinstance(event, dict):
            continue
        anchor = extract_concrete_timeline_anchor(event.get("timeframe"), deps=deps)
        if not anchor:
            continue
        context = " ".join(str(event.get(key) or "") for key in ("event", "detail", "timeframe"))
        candidates.append((anchor, context.lower()))

    if source_row is not None and has_timeline_commercial_signal(result, source_row, deps=deps):
        review_blob = " ".join(
            str(source_row.get(field) or "")
            for field in ("summary", "review_text", "pros", "cons")
        )
        anchor = extract_concrete_timeline_anchor(review_blob, deps=deps)
        if anchor:
            candidates.append((anchor, review_blob.lower()))

    for anchor, context in candidates:
        if not evaluation_deadline and (deps.contains_any(context, deps.timeline_decision_deadline_patterns) or " before " in context):
            evaluation_deadline = anchor
            continue
        if not contract_end and (
            deps.contains_any(context, deps.timeline_contract_end_patterns)
            or bool(churn.get("contract_renewal_mentioned"))
        ):
            contract_end = anchor
            continue
        if not evaluation_deadline and (
            bool(churn.get("actively_evaluating"))
            or bool(churn.get("migration_in_progress"))
            or bool(churn.get("intent_to_leave"))
        ):
            evaluation_deadline = anchor
            continue

    if not contract_end and source_row is not None and has_timeline_commercial_signal(result, source_row, deps=deps):
        review_blob = " ".join(
            str(source_row.get(field) or "")
            for field in ("summary", "review_text", "pros", "cons")
        )
        contract_event_anchor = extract_contract_end_event_anchor(review_blob, deps=deps)
        if contract_event_anchor:
            contract_end = contract_event_anchor

    return contract_end, evaluation_deadline


def derive_decision_timeline(
    result: dict[str, Any],
    source_row: dict[str, Any] | None = None,
    *,
    deps: EnrichmentTimelineDeps,
) -> str:
    churn = result.get("churn_signals") or {}
    timeline = result.get("timeline") or {}
    event_mentions = result.get("event_mentions") or []
    parts = [
        str(churn.get("renewal_timing") or ""),
        str(timeline.get("contract_end") or ""),
        str(timeline.get("evaluation_deadline") or ""),
    ]
    for event in event_mentions:
        if isinstance(event, dict):
            parts.append(str(event.get("timeframe") or ""))
    text = " ".join(parts).lower()
    if deps.contains_any(text, deps.timeline_immediate_patterns):
        return "immediate"
    if deps.contains_any(text, deps.timeline_quarter_patterns):
        return "within_quarter"
    if deps.contains_any(text, deps.timeline_year_patterns):
        return "within_year"

    if source_row is not None:
        review_blob = " ".join(
            str(source_row.get(field) or "")
            for field in ("summary", "review_text", "pros", "cons")
        ).lower()
        has_commercial_signal = has_timeline_commercial_signal(result, source_row, deps=deps)
        if has_commercial_signal and deps.contains_any(review_blob, deps.timeline_decision_patterns):
            if deps.contains_any(review_blob, deps.timeline_immediate_patterns):
                return "immediate"
            if deps.contains_any(review_blob, deps.timeline_quarter_patterns):
                return "within_quarter"
            if deps.contains_any(review_blob, deps.timeline_year_patterns):
                return "within_year"
    return "unknown"
