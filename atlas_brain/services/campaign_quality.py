"""Shared campaign quality revalidation helpers."""

from __future__ import annotations

import json
from typing import Any

from ..autonomous.tasks._b2b_specificity import (
    campaign_policy_audit_snapshot,
    merge_specificity_contexts,
    surface_specificity_context,
)
from ..config import settings


def coerce_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def campaign_specificity_context(
    metadata: Any,
    company_context: Any,
) -> dict[str, Any]:
    return campaign_specificity_context_details(metadata, company_context)["context"]


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


def campaign_specificity_context_details(
    metadata: Any,
    company_context: Any,
) -> dict[str, Any]:
    metadata_context = surface_specificity_context(
        coerce_json_dict(metadata),
        surface="campaign",
        nested_keys=("briefing_context",),
    )
    company_specificity = surface_specificity_context(
        coerce_json_dict(company_context),
        surface="campaign",
        nested_keys=("briefing_context",),
    )
    resolved_context = merge_specificity_contexts(metadata_context, company_specificity)
    context_sources: list[str] = []
    if metadata_context:
        context_sources.append("metadata")
    if company_specificity:
        context_sources.append("company_context")
    return {
        "context": resolved_context,
        "metadata_context": metadata_context,
        "company_context": company_specificity,
        "context_sources": context_sources,
        "missing_primary_inputs": _missing_context_inputs(resolved_context),
    }


def campaign_specificity_context_has_witnesses(
    context: dict[str, Any] | None,
) -> bool:
    if not isinstance(context, dict):
        return False
    return bool(context.get("anchor_examples") or context.get("witness_highlights"))


def campaign_specificity_from_consumer_context(
    consumer_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(consumer_context, dict):
        return {}
    resolved: dict[str, Any] = {}
    anchors = consumer_context.get("anchor_examples")
    if isinstance(anchors, dict) and anchors:
        resolved["anchor_examples"] = anchors
    highlights = consumer_context.get("witness_highlights")
    if isinstance(highlights, list) and highlights:
        resolved["witness_highlights"] = highlights
    reference_ids = consumer_context.get("reference_ids")
    if isinstance(reference_ids, dict) and reference_ids:
        resolved["reference_ids"] = reference_ids
    return resolved


async def campaign_specificity_context_with_fallback(
    pool,
    *,
    campaign: dict[str, Any],
    company_context: Any = None,
) -> dict[str, Any]:
    resolution = await campaign_specificity_context_resolution_with_fallback(
        pool,
        campaign=campaign,
        company_context=company_context,
    )
    return resolution["context"]


async def campaign_specificity_context_resolution_with_fallback(
    pool,
    *,
    campaign: dict[str, Any],
    company_context: Any = None,
) -> dict[str, Any]:
    metadata = coerce_json_dict(campaign.get("metadata"))
    primary_resolution = campaign_specificity_context_details(metadata, company_context)
    context = primary_resolution["context"]
    if campaign_specificity_context_has_witnesses(context):
        return {
            **primary_resolution,
            "fallback_used": False,
            "reasoning_view_found": False,
            "missing_inputs": _missing_context_inputs(context),
            "context": context,
        }

    vendor_name = str(
        campaign.get("vendor_name")
        or campaign.get("company_name")
        or ""
    ).strip()
    if not vendor_name:
        return {
            **primary_resolution,
            "fallback_used": False,
            "reasoning_view_found": False,
            "missing_inputs": _missing_context_inputs(context),
            "context": context,
        }

    from ..autonomous.tasks._b2b_synthesis_reader import load_best_reasoning_view

    view = await load_best_reasoning_view(
        pool,
        vendor_name,
        allow_legacy_fallback=False,
    )
    if view is None:
        return {
            **primary_resolution,
            "fallback_used": False,
            "reasoning_view_found": False,
            "missing_inputs": _missing_context_inputs(context),
            "context": context,
        }
    fallback_context = campaign_specificity_from_consumer_context(
        view.filtered_consumer_context("campaign")
    )
    merged_context = merge_specificity_contexts(
        context,
        fallback_context,
    )
    context_sources = list(primary_resolution["context_sources"])
    if fallback_context:
        context_sources.append("reasoning_fallback")
    return {
        **primary_resolution,
        "context": merged_context,
        "context_sources": context_sources,
        "fallback_used": bool(fallback_context),
        "reasoning_view_found": True,
        "missing_inputs": _missing_context_inputs(merged_context),
    }


_CAMPAIGN_POLICY_PREFIXES = (
    "report_tier_language:",
    "competitor_name_in_email_cold:",
    "incumbent_name_in_email_cold:",
    "private_account_name_leak:",
)

_CAMPAIGN_EVIDENCE_BLOCKERS = (
    "missing_exact_proof_term",
    "content does not reference any witness-backed anchor despite anchors being available",
    "content omits a concrete timing or numeric anchor even though one is available",
)


def _reference_id_counts(reference_ids: Any) -> dict[str, int]:
    if not isinstance(reference_ids, dict):
        return {}
    counts: dict[str, int] = {}
    for key, value in reference_ids.items():
        if isinstance(value, list):
            counts[str(key)] = len([item for item in value if str(item or "").strip()])
        elif isinstance(value, dict):
            counts[str(key)] = len(value)
        elif value not in (None, "", [], {}):
            counts[str(key)] = 1
    return counts


def build_campaign_failure_explanation(
    *,
    audit: dict[str, Any],
    boundary: str,
    specificity_context: dict[str, Any] | None,
    missing_inputs: list[str],
    missing_primary_inputs: list[str],
    context_sources: list[str],
    fallback_used: bool,
    reasoning_view_found: bool,
) -> dict[str, Any]:
    resolved_context = specificity_context if isinstance(specificity_context, dict) else {}
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
    available_groups = [
        str(group).strip()
        for group in (audit.get("available_groups") or [])
        if str(group or "").strip()
    ]
    matched_groups = [
        str(group).strip()
        for group in (audit.get("matched_groups") or [])
        if str(group or "").strip()
    ]
    missing_groups = [
        str(group).strip()
        for group in (audit.get("missing_groups") or [])
        if str(group or "").strip()
    ]
    required_proof_terms = [
        str(term).strip()
        for term in (audit.get("required_proof_terms") or audit.get("campaign_proof_terms") or [])
        if str(term or "").strip()
    ]
    used_proof_terms = [
        str(term).strip()
        for term in (audit.get("used_proof_terms") or [])
        if str(term or "").strip()
    ]
    unused_proof_terms = [
        str(term).strip()
        for term in (audit.get("unused_proof_terms") or [])
        if str(term or "").strip()
    ]
    anchor_examples = resolved_context.get("anchor_examples")
    witness_highlights = resolved_context.get("witness_highlights")
    reference_ids = resolved_context.get("reference_ids")
    anchor_count = int(audit.get("anchor_count") or 0)
    highlight_count = int(audit.get("highlight_count") or 0)
    reference_id_counts = _reference_id_counts(reference_ids or audit.get("reference_ids"))

    has_policy_violation = any(
        issue.startswith(_CAMPAIGN_POLICY_PREFIXES) for issue in blocking_issues
    )
    has_evidence_failure = any(
        issue in _CAMPAIGN_EVIDENCE_BLOCKERS for issue in blocking_issues
    )
    has_available_evidence = bool(
        anchor_count
        or highlight_count
        or available_groups
        or required_proof_terms
        or reference_id_counts
    )
    missing_context_data = bool(missing_inputs)

    cause_type: str | None = None
    if blocking_issues:
        categories = 0
        if has_policy_violation:
            categories += 1
        if has_evidence_failure and has_available_evidence:
            categories += 1
        if missing_context_data and not has_available_evidence:
            categories += 1
        if categories > 1 or (has_policy_violation and missing_context_data):
            cause_type = "mixed"
        elif has_policy_violation:
            cause_type = "policy_violation"
        elif has_evidence_failure and has_available_evidence:
            cause_type = "content_ignored_available_evidence"
        elif missing_context_data:
            cause_type = "upstream_data_missing"
        else:
            cause_type = "content_ignored_available_evidence"

    return {
        "boundary": boundary,
        "primary_blocker": blocking_issues[0] if blocking_issues else None,
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
        "missing_primary_inputs": list(missing_primary_inputs),
        "context_sources": list(context_sources),
        "fallback_used": bool(fallback_used),
        "reasoning_view_found": bool(reasoning_view_found),
        "anchor_count": anchor_count,
        "highlight_count": highlight_count,
        "reference_id_counts": reference_id_counts,
        "anchor_labels": [
            str(label).strip()
            for label in (audit.get("anchor_labels") or [])
            if str(label or "").strip()
        ],
        "context_has_anchor_examples": bool(anchor_examples),
        "context_has_witness_highlights": bool(witness_highlights),
        "context_has_reference_ids": bool(reference_ids),
    }


def merge_campaign_revalidation_metadata(
    *,
    campaign: dict[str, Any],
    metadata: Any,
    audit: dict[str, Any],
    boundary: str,
    specificity_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_metadata = coerce_json_dict(metadata)
    tier = str(campaign.get("tier") or merged_metadata.get("tier") or "").strip()
    if tier and not merged_metadata.get("tier"):
        merged_metadata["tier"] = tier
    target_mode = str(
        campaign.get("target_mode") or merged_metadata.get("target_mode") or ""
    ).strip()
    if target_mode and not merged_metadata.get("target_mode"):
        merged_metadata["target_mode"] = target_mode
    if isinstance(specificity_context, dict):
        anchors = specificity_context.get("anchor_examples")
        if (
            isinstance(anchors, dict)
            and anchors
            and not merged_metadata.get("reasoning_anchor_examples")
        ):
            merged_metadata["reasoning_anchor_examples"] = anchors
        highlights = specificity_context.get("witness_highlights")
        if (
            isinstance(highlights, list)
            and highlights
            and not merged_metadata.get("reasoning_witness_highlights")
        ):
            merged_metadata["reasoning_witness_highlights"] = highlights
        reference_ids = specificity_context.get("reference_ids")
        if (
            isinstance(reference_ids, dict)
            and reference_ids
            and not merged_metadata.get("reasoning_reference_ids")
        ):
            merged_metadata["reasoning_reference_ids"] = reference_ids
    proof_terms = audit.get("campaign_proof_terms")
    if (
        isinstance(proof_terms, list)
        and proof_terms
        and not merged_metadata.get("campaign_proof_terms")
    ):
        merged_metadata["campaign_proof_terms"] = proof_terms
    merged_metadata["latest_specificity_audit"] = {
        **audit,
        "boundary": boundary,
    }
    return merged_metadata


def campaign_quality_revalidation(
    *,
    campaign: dict[str, Any],
    boundary: str,
    company_context: Any = None,
    metadata: Any = None,
    specificity_context: dict[str, Any] | None = None,
    resolution_details: dict[str, Any] | None = None,
    min_anchor_hits: int | None = None,
    require_anchor_support: bool | None = None,
    require_timing_or_numeric_when_available: bool | None = None,
    proof_term_limit: int | None = None,
) -> dict[str, Any]:
    resolved_metadata = coerce_json_dict(
        campaign.get("metadata") if metadata is None else metadata
    )
    if isinstance(resolution_details, dict):
        resolved_context = (
            resolution_details.get("context")
            if isinstance(resolution_details.get("context"), dict)
            else {}
        )
        if not resolved_context and isinstance(specificity_context, dict):
            resolved_context = specificity_context
        resolution_details = {
            **resolution_details,
            "missing_inputs": list(
                resolution_details.get("missing_inputs")
                or _missing_context_inputs(resolved_context)
            ),
            "missing_primary_inputs": list(
                resolution_details.get("missing_primary_inputs")
                or _missing_context_inputs(resolved_context)
            ),
            "context_sources": list(resolution_details.get("context_sources") or []),
            "fallback_used": bool(resolution_details.get("fallback_used")),
            "reasoning_view_found": bool(
                resolution_details.get("reasoning_view_found")
            ),
        }
    elif isinstance(specificity_context, dict):
        resolved_context = specificity_context
        resolution_details = {
            "context": resolved_context,
            "context_sources": [
                source
                for source in ("metadata", "company_context")
                if surface_specificity_context(
                    resolved_metadata if source == "metadata" else coerce_json_dict(company_context),
                    surface="campaign",
                    nested_keys=("briefing_context",),
                )
            ],
            "missing_primary_inputs": _missing_context_inputs(resolved_context),
            "missing_inputs": _missing_context_inputs(resolved_context),
            "fallback_used": False,
            "reasoning_view_found": False,
        }
    else:
        resolution_details = campaign_specificity_context_details(
            resolved_metadata,
            company_context,
        )
        resolution_details["missing_inputs"] = _missing_context_inputs(
            resolution_details["context"]
        )
        resolution_details["fallback_used"] = False
        resolution_details["reasoning_view_found"] = False
        resolved_context = resolution_details["context"]
    merged_campaign = {
        **campaign,
        "metadata": resolved_metadata,
    }
    resolved_min_anchor_hits = int(
        settings.b2b_campaign.specificity_min_anchor_hits
        if min_anchor_hits is None else min_anchor_hits
    )
    resolved_require_anchor_support = bool(
        settings.b2b_campaign.specificity_require_anchor_support
        if require_anchor_support is None else require_anchor_support
    )
    resolved_require_timing_or_numeric = bool(
        settings.b2b_campaign.specificity_require_timing_or_numeric_when_available
        if require_timing_or_numeric_when_available is None
        else require_timing_or_numeric_when_available
    )
    resolved_proof_term_limit = int(
        settings.b2b_campaign.specificity_revision_term_limit
        if proof_term_limit is None else proof_term_limit
    )
    audit = campaign_policy_audit_snapshot(
        subject=str(campaign.get("subject") or ""),
        body=str(campaign.get("body") or ""),
        cta=str(campaign.get("cta") or ""),
        campaign=merged_campaign,
        anchor_examples=resolved_context.get("anchor_examples"),
        witness_highlights=resolved_context.get("witness_highlights"),
        reference_ids=resolved_context.get("reference_ids"),
        campaign_proof_terms=resolved_metadata.get("campaign_proof_terms"),
        min_anchor_hits=resolved_min_anchor_hits,
        require_anchor_support=resolved_require_anchor_support,
        require_timing_or_numeric_when_available=resolved_require_timing_or_numeric,
        proof_term_limit=resolved_proof_term_limit,
    )
    failure_explanation = build_campaign_failure_explanation(
        audit=audit,
        boundary=boundary,
        specificity_context=resolved_context,
        missing_inputs=list(resolution_details.get("missing_inputs") or []),
        missing_primary_inputs=list(
            resolution_details.get("missing_primary_inputs") or []
        ),
        context_sources=list(resolution_details.get("context_sources") or []),
        fallback_used=bool(resolution_details.get("fallback_used")),
        reasoning_view_found=bool(resolution_details.get("reasoning_view_found")),
    )
    audit = {
        **audit,
        "primary_blocker": failure_explanation.get("primary_blocker"),
        "cause_type": failure_explanation.get("cause_type"),
        "missing_inputs": failure_explanation.get("missing_inputs"),
        "context_sources": failure_explanation.get("context_sources"),
        "failure_explanation": failure_explanation,
    }
    merged_metadata = merge_campaign_revalidation_metadata(
        campaign=merged_campaign,
        metadata=resolved_metadata,
        audit=audit,
        boundary=boundary,
        specificity_context=resolved_context,
    )
    return {
        "audit": audit,
        "metadata": merged_metadata,
        "specificity_context": resolved_context,
    }


async def campaign_quality_revalidation_with_fallback(
    pool,
    *,
    campaign: dict[str, Any],
    boundary: str,
    company_context: Any = None,
    metadata: Any = None,
) -> dict[str, Any]:
    resolved_metadata = coerce_json_dict(
        campaign.get("metadata") if metadata is None else metadata
    )
    resolution_details = await campaign_specificity_context_resolution_with_fallback(
        pool,
        campaign={
            **campaign,
            "metadata": resolved_metadata,
        },
        company_context=company_context,
    )
    resolved_context = resolution_details["context"]
    return campaign_quality_revalidation(
        campaign={
            **campaign,
            "metadata": resolved_metadata,
        },
        boundary=boundary,
        company_context=company_context,
        metadata=resolved_metadata,
        specificity_context=resolved_context,
        resolution_details=resolution_details,
    )
