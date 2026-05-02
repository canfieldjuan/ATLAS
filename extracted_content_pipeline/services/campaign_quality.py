"""Standalone campaign quality revalidation seam."""

from __future__ import annotations

import json
import re
from typing import Any


_PLACEHOLDER_RE = re.compile(r"\[(?:Name|Company|Your Name|First Name|Title)\]|\{\{.+?\}\}")


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


def _copy_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _specificity_context(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _proof_terms(metadata: dict[str, Any], context: dict[str, Any]) -> list[str]:
    terms = metadata.get("campaign_proof_terms")
    if not isinstance(terms, list):
        terms = context.get("campaign_proof_terms")
    return [
        str(term or "").strip()
        for term in _copy_list(terms)
        if str(term or "").strip()
    ]


def _blocking_issues(campaign: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    subject = str(campaign.get("subject") or "")
    body = str(campaign.get("body") or "")
    if _PLACEHOLDER_RE.search(subject) or _PLACEHOLDER_RE.search(body):
        issues.append("placeholder_token")
    return issues


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
    """Return the copied campaign task's expected revalidation envelope.

    The extracted package keeps this seam intentionally conservative and local:
    it preserves existing metadata/proof-term plumbing and blocks obvious
    placeholder tokens, while full Atlas-specific evidence-policy scoring stays
    outside this import-readiness slice.
    """
    del company_context
    del resolution_details
    del min_anchor_hits
    del require_anchor_support
    del require_timing_or_numeric_when_available
    del proof_term_limit

    resolved_metadata = coerce_json_dict(
        campaign.get("metadata") if metadata is None else metadata
    )
    resolved_context = _specificity_context(specificity_context)
    blocking_issues = _blocking_issues(campaign)
    warnings: list[str] = []
    proof_terms = _proof_terms(resolved_metadata, resolved_context)
    audit = {
        "boundary": boundary,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "campaign_proof_terms": proof_terms,
        "available_groups": [],
        "matched_groups": [],
        "missing_groups": [],
        "used_proof_terms": [],
        "unused_proof_terms": proof_terms,
        "primary_blocker": blocking_issues[0] if blocking_issues else None,
        "cause_type": "placeholder" if blocking_issues else None,
        "missing_inputs": [],
        "context_sources": ["specificity_context"] if resolved_context else [],
    }
    merged_metadata = {
        **resolved_metadata,
        "latest_specificity_audit": audit,
    }
    if proof_terms:
        merged_metadata["campaign_proof_terms"] = proof_terms
    return {
        "audit": audit,
        "metadata": merged_metadata,
        "specificity_context": resolved_context,
    }


__all__ = ["campaign_quality_revalidation", "coerce_json_dict"]
