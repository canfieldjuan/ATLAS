"""Shared blog quality audit and revalidation helpers."""

from __future__ import annotations

import json
from typing import Any

from ..autonomous.tasks._b2b_specificity import (
    _contains_term,
    specificity_audit_snapshot,
    surface_specificity_context,
)
from .campaign_quality import coerce_json_dict

_BLOG_FAILURE_STEPS = {
    "generation": "quality_gate",
    "manual_generate": "manual_generate",
    "publish": "publish_validation",
    "backfill": "backfill",
}

_BLOG_FAILURE_CODES = {
    "generation": "quality_gate_rejection",
    "manual_generate": "manual_generate_quality_rejection",
    "publish": "publish_revalidation_failed",
    "backfill": "blog_quality_backfill_refresh",
}

_BLOG_QUALITY_CONTRACT_PREFIXES = (
    "content_too_short:",
    "missing_chart_placeholder:",
    "duplicate_chart_placeholder:",
    "unknown_chart_placeholders:",
    "unresolved_placeholders:",
    "too_few_sourced_quotes:",
    "missing_vendor_mentions:",
    "placeholder_links_href_hash",
    "nonexistent_internal_links:",
)


def _safe_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _missing_context_inputs(context: dict[str, Any] | None) -> list[str]:
    resolved = context if isinstance(context, dict) else {}
    missing: list[str] = []
    if not resolved.get("anchor_examples"):
        missing.append("reasoning_anchor_examples")
    if not resolved.get("witness_highlights"):
        missing.append("reasoning_witness_highlights")
    if not resolved.get("reference_ids"):
        missing.append("reasoning_reference_ids")
    return missing


def latest_blog_quality_audit(data_context: Any) -> dict[str, Any]:
    context = coerce_json_dict(data_context)
    audit = context.get("latest_quality_audit")
    return dict(audit) if isinstance(audit, dict) else {}


def blog_failure_explanation(data_context: Any) -> dict[str, Any] | None:
    audit = latest_blog_quality_audit(data_context)
    explanation = audit.get("failure_explanation")
    return dict(explanation) if isinstance(explanation, dict) else None


def blog_quality_summary(data_context: Any) -> dict[str, Any]:
    audit = latest_blog_quality_audit(data_context)
    if audit:
        return audit
    legacy = coerce_json_dict(data_context).get("generation_quality")
    return dict(legacy) if isinstance(legacy, dict) else {}


def blog_quality_context_details(data_context: Any) -> dict[str, Any]:
    context = surface_specificity_context(
        coerce_json_dict(data_context),
        surface="blog",
    )
    return {
        "context": context,
        "context_sources": ["data_context"] if context else [],
        "missing_inputs": _missing_context_inputs(context),
    }


def _ordered_required_terms(signal_terms: dict[str, Any] | None) -> list[str]:
    if not isinstance(signal_terms, dict):
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for group_name in (
        "numeric_terms",
        "timing_terms",
        "competitor_terms",
        "pain_terms",
        "workflow_terms",
    ):
        values = signal_terms.get(group_name)
        if not isinstance(values, list):
            continue
        for value in values:
            token = str(value or "").strip()
            key = token.lower()
            if not token or key in seen:
                continue
            seen.add(key)
            ordered.append(token)
    return ordered


def build_blog_failure_explanation(
    *,
    audit: dict[str, Any],
    boundary: str,
    specificity_context: dict[str, Any] | None,
    specificity_snapshot: dict[str, Any] | None,
    missing_inputs: list[str],
    context_sources: list[str],
    content_text: str,
) -> dict[str, Any]:
    resolved_context = specificity_context if isinstance(specificity_context, dict) else {}
    snapshot = specificity_snapshot if isinstance(specificity_snapshot, dict) else {}
    blocking_issues = [
        str(issue).strip()
        for issue in (audit.get("blocking_issues") or [])
        if str(issue or "").strip()
    ]
    warnings = [
        str(issue).strip()
        for issue in (audit.get("warnings") or [])
        if str(issue or "").strip()
    ]
    threshold = audit.get("threshold")
    score = audit.get("score")
    primary_blocker = blocking_issues[0] if blocking_issues else None
    if (
        primary_blocker is None
        and audit.get("status") == "fail"
        and score is not None
        and threshold is not None
        and score < threshold
    ):
        primary_blocker = "quality_score_below_threshold"

    signal_terms = snapshot.get("signal_terms") if isinstance(snapshot.get("signal_terms"), dict) else {}
    required_proof_terms = _ordered_required_terms(signal_terms)
    normalized_content = str(content_text or "")
    used_proof_terms = [
        term
        for term in required_proof_terms
        if _contains_term(normalized_content, term)
    ]
    used_terms_lower = {term.lower() for term in used_proof_terms}
    unused_proof_terms = [
        term for term in required_proof_terms if term.lower() not in used_terms_lower
    ]

    matched_groups = [
        str(group).strip()
        for group in (snapshot.get("matched_groups") or [])
        if str(group or "").strip()
    ]
    available_groups = [
        str(group).strip()
        for group in (snapshot.get("available_groups") or [])
        if str(group or "").strip()
    ]
    missing_groups = [
        str(group).strip()
        for group in (snapshot.get("missing_groups") or [])
        if str(group or "").strip()
    ]

    anchor_count = int(snapshot.get("anchor_count") or 0)
    highlight_count = int(snapshot.get("highlight_count") or 0)
    reference_id_counts = snapshot.get("reference_id_counts")
    if not isinstance(reference_id_counts, dict):
        reference_id_counts = {}

    has_specificity_failure = any(
        issue.startswith("witness_specificity:")
        for issue in blocking_issues
    )
    has_available_evidence = bool(
        anchor_count
        or highlight_count
        or available_groups
        or reference_id_counts
        or required_proof_terms
    )
    has_unsupported_claim = any(
        issue == "unsupported_category_outcome_assertion"
        or issue.startswith("unsupported_data_claim:")
        or issue.startswith("critical_warning_unresolved:unsupported_data_claim:")
        or issue.startswith("chart_scope_ambiguity:")
        or issue.startswith("critical_warning_unresolved:chart_scope_ambiguity:")
        for issue in blocking_issues + warnings
    )
    has_quality_contract_failure = bool(
        primary_blocker == "quality_score_below_threshold"
        or any(
            issue == prefix or issue.startswith(prefix)
            for issue in blocking_issues
            for prefix in _BLOG_QUALITY_CONTRACT_PREFIXES
        )
    )

    cause_candidates: list[str] = []
    if str(audit.get("status") or "") == "fail":
        if has_unsupported_claim:
            cause_candidates.append("unsupported_claim")
        if has_specificity_failure and has_available_evidence:
            cause_candidates.append("content_ignored_available_evidence")
        elif has_specificity_failure and not has_unsupported_claim:
            cause_candidates.append("upstream_data_missing")
        if has_quality_contract_failure:
            cause_candidates.append("quality_contract_failure")
        if not cause_candidates and missing_inputs:
            cause_candidates.append("upstream_data_missing")

    deduped_causes: list[str] = []
    for candidate in cause_candidates:
        if candidate not in deduped_causes:
            deduped_causes.append(candidate)

    cause_type: str | None = None
    if len(deduped_causes) > 1:
        cause_type = "mixed"
    elif deduped_causes:
        cause_type = deduped_causes[0]

    anchor_labels = [
        str(label).strip()
        for label in (snapshot.get("anchor_labels") or [])
        if str(label or "").strip()
    ]

    return {
        "boundary": boundary,
        "primary_blocker": primary_blocker,
        "cause_type": cause_type,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "matched_groups": matched_groups,
        "available_groups": available_groups,
        "missing_groups": missing_groups,
        "required_proof_terms": required_proof_terms,
        "used_proof_terms": used_proof_terms,
        "unused_proof_terms": unused_proof_terms,
        "missing_inputs": list(missing_inputs),
        "missing_primary_inputs": list(missing_inputs),
        "context_sources": list(context_sources),
        "fallback_used": False,
        "reasoning_view_found": False,
        "anchor_count": anchor_count,
        "highlight_count": highlight_count,
        "reference_id_counts": reference_id_counts,
        "anchor_labels": anchor_labels,
        "context_has_anchor_examples": bool(resolved_context.get("anchor_examples")),
        "context_has_witness_highlights": bool(resolved_context.get("witness_highlights")),
        "context_has_reference_ids": bool(resolved_context.get("reference_ids")),
    }


def merge_blog_revalidation_data_context(
    *,
    data_context: Any,
    audit: dict[str, Any],
    boundary: str,
) -> dict[str, Any]:
    merged = coerce_json_dict(data_context)
    merged["latest_quality_audit"] = {
        **audit,
        "boundary": boundary,
    }
    if boundary in {"generation", "manual_generate"}:
        merged["generation_quality"] = dict(audit)
    return merged


def blog_quality_projection(
    audit: dict[str, Any],
    *,
    boundary: str,
) -> dict[str, Any]:
    blocking_issues = list(audit.get("blocking_issues") or [])
    warnings = list(audit.get("warnings") or [])
    status = str(audit.get("status") or "")
    summary = ", ".join(str(issue) for issue in blocking_issues[:3]).strip()
    if not summary and status == "fail":
        score = audit.get("score")
        threshold = audit.get("threshold")
        if score is not None and threshold is not None and score < threshold:
            summary = f"quality_score_below_threshold:{score}_lt_{threshold}"
        else:
            summary = "blog_quality_revalidation_failed"

    if status == "pass":
        return {
            "score": audit.get("score"),
            "threshold": audit.get("threshold"),
            "blocker_count": len(blocking_issues),
            "warning_count": len(warnings),
            "failure_step": None,
            "error_code": None,
            "error_summary": None,
            "rejection_reason": None,
        }

    return {
        "score": audit.get("score"),
        "threshold": audit.get("threshold"),
        "blocker_count": len(blocking_issues),
        "warning_count": len(warnings),
        "failure_step": _BLOG_FAILURE_STEPS.get(boundary, boundary),
        "error_code": _BLOG_FAILURE_CODES.get(boundary, "blog_quality_revalidation_failed"),
        "error_summary": summary,
        "rejection_reason": summary,
    }


def blog_row_content(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        row = dict(row)
    return {
        "title": row.get("title") or "",
        "description": row.get("description") or "",
        "content": row.get("content") or "",
        "seo_title": row.get("seo_title") or "",
        "seo_description": row.get("seo_description") or "",
        "target_keyword": row.get("target_keyword") or "",
        "secondary_keywords": _safe_json(row.get("secondary_keywords")),
        "faq": _safe_json(row.get("faq")),
    }


def blog_row_to_blueprint(row: Any):
    from ..autonomous.tasks.b2b_blog_post_generation import ChartSpec, PostBlueprint

    if not isinstance(row, dict):
        row = dict(row)
    raw_charts = _safe_json(row.get("charts", []))
    charts: list[ChartSpec] = []
    for idx, chart in enumerate(raw_charts if isinstance(raw_charts, list) else []):
        if not isinstance(chart, dict):
            continue
        charts.append(
            ChartSpec(
                chart_id=str(chart.get("chart_id") or f"chart_{idx + 1}"),
                chart_type=str(chart.get("chart_type") or "bar"),
                title=str(chart.get("title") or f"Chart {idx + 1}"),
                data=chart.get("data") if isinstance(chart.get("data"), list) else [],
                config=chart.get("config") if isinstance(chart.get("config"), dict) else {},
            )
        )

    data_context = coerce_json_dict(_safe_json(row.get("data_context")))
    quotable_phrases = data_context.get("quotable_phrases")
    if not isinstance(quotable_phrases, list):
        quotable_phrases = []

    return PostBlueprint(
        topic_type=str(row.get("topic_type") or ""),
        slug=str(row.get("slug") or ""),
        suggested_title=str(row.get("title") or ""),
        tags=_safe_json(row.get("tags", [])) if isinstance(_safe_json(row.get("tags", [])), list) else [],
        data_context=data_context,
        sections=[],
        charts=charts,
        quotable_phrases=quotable_phrases,
        cta=_safe_json(row.get("cta")) if isinstance(_safe_json(row.get("cta")), dict) else None,
    )


def blog_quality_revalidation(
    *,
    blueprint,
    content: dict[str, Any] | None,
    boundary: str,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from ..autonomous.tasks.b2b_blog_post_generation import (
        _apply_blog_quality_gate,
        _with_unresolved_critical_warnings,
    )

    resolved_content = dict(content or {})
    if report is None:
        _, resolved_report = _apply_blog_quality_gate(blueprint, resolved_content)
    else:
        resolved_report = dict(report)
    resolved_report = _with_unresolved_critical_warnings(resolved_report)

    context_details = blog_quality_context_details(getattr(blueprint, "data_context", {}))
    specificity_context = context_details["context"]
    content_text = str(resolved_content.get("content") or "")
    specificity_snapshot = {}
    if specificity_context:
        specificity_snapshot = specificity_audit_snapshot(
            content_text,
            anchor_examples=specificity_context.get("anchor_examples"),
            witness_highlights=specificity_context.get("witness_highlights"),
            reference_ids=specificity_context.get("reference_ids"),
            allow_company_names=False,
            min_anchor_hits=1,
            require_anchor_support=False,
            require_timing_or_numeric_when_available=False,
            include_competitor_terms=True,
        )

    failure_explanation = build_blog_failure_explanation(
        audit=resolved_report,
        boundary=boundary,
        specificity_context=specificity_context,
        specificity_snapshot=specificity_snapshot,
        missing_inputs=list(context_details["missing_inputs"]),
        context_sources=list(context_details["context_sources"]),
        content_text=content_text,
    )

    audit = {
        **resolved_report,
        "boundary": boundary,
        "primary_blocker": failure_explanation.get("primary_blocker"),
        "cause_type": failure_explanation.get("cause_type"),
        "missing_inputs": failure_explanation.get("missing_inputs"),
        "context_sources": failure_explanation.get("context_sources"),
        "matched_groups": failure_explanation.get("matched_groups"),
        "available_groups": failure_explanation.get("available_groups"),
        "missing_groups": failure_explanation.get("missing_groups"),
        "required_proof_terms": failure_explanation.get("required_proof_terms"),
        "used_proof_terms": failure_explanation.get("used_proof_terms"),
        "unused_proof_terms": failure_explanation.get("unused_proof_terms"),
        "anchor_count": failure_explanation.get("anchor_count"),
        "anchor_labels": failure_explanation.get("anchor_labels"),
        "highlight_count": failure_explanation.get("highlight_count"),
        "reference_ids": (specificity_snapshot or {}).get("reference_ids") or specificity_context.get("reference_ids") or {},
        "reference_id_counts": failure_explanation.get("reference_id_counts"),
        "failure_explanation": failure_explanation,
    }
    merged_context = merge_blog_revalidation_data_context(
        data_context=getattr(blueprint, "data_context", {}),
        audit=audit,
        boundary=boundary,
    )
    return {
        "audit": audit,
        "data_context": merged_context,
        "specificity_context": specificity_context,
    }
