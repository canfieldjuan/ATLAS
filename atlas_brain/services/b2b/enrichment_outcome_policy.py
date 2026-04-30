from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnrichmentOutcomePolicyDeps:
    normalized_low_fidelity_noisy_sources: Any
    normalize_compare_text: Any
    text_mentions_name: Any
    normalized_name_tokens: Any
    has_commercial_context: Any
    has_strong_commercial_context: Any
    has_technical_context: Any
    has_consumer_context: Any
    dedupe_reason_codes: Any
    normalize_company_name: Any


def is_no_signal_result(result: dict[str, Any], source_row: dict[str, Any]) -> bool:
    churn = result.get("churn_signals") or {}
    if any(bool(value) for value in churn.values()):
        return False
    if result.get("competitors_mentioned"):
        return False
    if result.get("specific_complaints") or result.get("quotable_phrases"):
        return False
    if result.get("pricing_phrases") or result.get("recommendation_language"):
        return False
    if result.get("event_mentions") or result.get("feature_gaps"):
        return False
    content_type = str(source_row.get("content_type") or "").strip().lower()
    if content_type in {"community_discussion", "comment"}:
        return True
    rating = source_row.get("rating")
    try:
        return float(rating or 0) >= 3.0
    except (TypeError, ValueError):
        return True


def trusted_reviewer_company_name(
    source_row: dict[str, Any] | None,
    *,
    deps: EnrichmentOutcomePolicyDeps,
) -> str | None:
    row = source_row if isinstance(source_row, dict) else {}
    company = str(row.get("reviewer_company") or "").strip()
    if not company:
        return None
    company_norm = deps.normalize_company_name(company) or company.lower()
    vendor_norm = deps.normalize_company_name(str(row.get("vendor_name") or "")) or ""
    if vendor_norm and company_norm == vendor_norm:
        return None
    return company


def witness_metrics(result: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(result, dict):
        return 0, 0
    spans = result.get("evidence_spans")
    if not isinstance(spans, list):
        return 0, 0
    witness_count = 0
    for span in spans:
        if not isinstance(span, dict):
            continue
        if not str(span.get("text") or "").strip():
            continue
        witness_count += 1
    return (1 if witness_count > 0 else 0), witness_count


def detect_low_fidelity_reasons(
    row: dict[str, Any],
    result: dict[str, Any],
    *,
    deps: EnrichmentOutcomePolicyDeps,
) -> list[str]:
    source = str(row.get("source") or "").strip().lower()
    noisy_sources = deps.normalized_low_fidelity_noisy_sources()
    if source not in noisy_sources and source != "trustpilot":
        return []

    combined_text = " ".join(
        str(row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    )
    combined_norm = deps.normalize_compare_text(combined_text)
    if not combined_norm:
        return ["empty_noisy_context"]

    summary_norm = deps.normalize_compare_text(row.get("summary"))
    vendor_hit = any(
        deps.text_mentions_name(combined_norm, row.get(field))
        for field in ("vendor_name", "product_name")
        if row.get(field)
    )
    competitor_hit = any(
        deps.text_mentions_name(combined_norm, comp.get("name"))
        for comp in (result.get("competitors_mentioned") or [])
        if isinstance(comp, dict) and comp.get("name")
    )
    summary_tokens = deps.normalized_name_tokens(row.get("summary"))
    urgency = float(result.get("urgency_score") or 0)
    reasons: list[str] = []
    if source in noisy_sources:
        if not vendor_hit:
            reasons.append("vendor_absent_noisy_source")
        if not vendor_hit and competitor_hit:
            reasons.append("competitor_only_context")
        if (
            source in {"twitter", "quora", "reddit", "hackernews"}
            and len(combined_norm) < 160
            and not (vendor_hit and deps.has_commercial_context(combined_norm))
        ):
            reasons.append("thin_social_context")
        if (
            source == "software_advice"
            and len(combined_norm) < 140
            and not deps.has_strong_commercial_context(combined_norm)
        ):
            reasons.append("thin_review_platform_context")
        if source == "quora" and summary_tokens and len(summary_tokens) <= 3 and not vendor_hit:
            reasons.append("author_style_summary")
    if source in {"stackoverflow", "github"}:
        if (
            urgency <= 5
            and deps.has_technical_context(summary_norm, combined_norm)
            and not deps.has_commercial_context(combined_norm)
        ):
            reasons.append("technical_question_context")
    if source == "trustpilot":
        if deps.has_consumer_context(combined_norm) and not deps.has_strong_commercial_context(combined_norm):
            reasons.append("consumer_support_context")
    return deps.dedupe_reason_codes(reasons)
