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


def _flatten_anchor_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    anchors = context.get("anchor_examples")
    if isinstance(anchors, dict):
        for group_rows in anchors.values():
            for row in _copy_list(group_rows):
                if isinstance(row, dict):
                    rows.append(dict(row))
    return rows


def _context_terms(rows: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for row in rows:
        for key in ("excerpt_text", "quote", "text", "anchor", "value"):
            term = str(row.get(key) or "").strip()
            if term:
                terms.append(term)
    return terms


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


def _matched_terms(content: str, terms: list[str], *, limit: int | None = None) -> list[str]:
    lowered = content.lower()
    matched: list[str] = []
    for term in terms:
        normalized = term.lower()
        if normalized and normalized in lowered and term not in matched:
            matched.append(term)
        if limit is not None and len(matched) >= limit:
            break
    return matched


def _timing_or_numeric_available(context: dict[str, Any], terms: list[str]) -> bool:
    if _copy_list(context.get("timing_windows")):
        return True
    if _copy_list(context.get("proof_points")):
        return True
    return any(re.search(r"\d|q[1-4]|renewal|month|quarter|year", term.lower()) for term in terms)


def _has_timing_or_numeric(content: str) -> bool:
    return bool(re.search(r"\d|q[1-4]|renewal|month|quarter|year", content.lower()))


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

    resolved_metadata = coerce_json_dict(
        campaign.get("metadata") if metadata is None else metadata
    )
    resolved_context = _specificity_context(specificity_context)
    blocking_issues = _blocking_issues(campaign)
    warnings: list[str] = []
    anchor_rows = _flatten_anchor_rows(resolved_context)
    anchor_terms = _context_terms(anchor_rows)
    proof_terms = _proof_terms(resolved_metadata, resolved_context) or anchor_terms
    content = " ".join(
        str(campaign.get(key) or "")
        for key in ("subject", "body", "cta")
    )
    min_hits = int(min_anchor_hits or 1)
    require_anchor = True if require_anchor_support is None else bool(require_anchor_support)
    require_timing = (
        True
        if require_timing_or_numeric_when_available is None
        else bool(require_timing_or_numeric_when_available)
    )
    matched_terms = _matched_terms(content, proof_terms, limit=proof_term_limit)
    if require_anchor and anchor_rows and len(matched_terms) < min_hits:
        blocking_issues.append("missing_anchor_support")
    if (
        require_timing
        and _timing_or_numeric_available(resolved_context, proof_terms)
        and not _has_timing_or_numeric(content)
    ):
        blocking_issues.append("missing_timing_or_numeric")

    anchors = resolved_context.get("anchor_examples")
    available_groups = list(anchors.keys()) if isinstance(anchors, dict) else []
    matched_groups = available_groups if matched_terms else []
    missing_groups = [
        group for group in available_groups if group not in matched_groups
    ]
    audit = {
        "boundary": boundary,
        "status": "fail" if blocking_issues else "pass",
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "campaign_proof_terms": proof_terms,
        "available_groups": available_groups,
        "matched_groups": matched_groups,
        "missing_groups": missing_groups,
        "used_proof_terms": matched_terms,
        "unused_proof_terms": [term for term in proof_terms if term not in matched_terms],
        "primary_blocker": blocking_issues[0] if blocking_issues else None,
        "cause_type": blocking_issues[0] if blocking_issues else None,
        "missing_inputs": [],
        "context_sources": ["specificity_context"] if resolved_context else [],
    }
    audit["failure_explanation"] = {
        "boundary": boundary,
        "cause_type": audit["cause_type"],
        "anchor_count": len(anchor_rows),
        "matched_anchor_count": len(matched_terms),
        "context_sources": audit["context_sources"],
    }
    merged_metadata = {
        **resolved_metadata,
        "latest_specificity_audit": audit,
    }
    for key in ("tier", "target_mode", "channel", "cta"):
        value = campaign.get(key)
        if key not in merged_metadata and value not in (None, "", [], {}):
            merged_metadata[key] = value
    context_metadata_keys = {
        "anchor_examples": "reasoning_anchor_examples",
        "witness_highlights": "reasoning_witness_highlights",
        "reference_ids": "reasoning_reference_ids",
    }
    for context_key, metadata_key in context_metadata_keys.items():
        value = resolved_context.get(context_key)
        if value not in (None, "", [], {}):
            merged_metadata[metadata_key] = value
    if proof_terms:
        merged_metadata["campaign_proof_terms"] = proof_terms
    return {
        "audit": audit,
        "metadata": merged_metadata,
        "specificity_context": resolved_context,
    }


__all__ = ["campaign_quality_revalidation", "coerce_json_dict"]
