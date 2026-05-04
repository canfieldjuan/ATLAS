"""Witness specificity pack: deterministic validators for witness-anchor coverage.

Owned by ``extracted_quality_gate`` (PR-B5b). The public surface lifts
the deterministic specificity helpers that previously lived in
``atlas_brain.autonomous.tasks._b2b_specificity``. None of these
functions touch the database, the clock, or the network -- they
operate on already-fetched anchor / witness rows and the candidate
content text.

Two API styles ship in this module:

  * **Functional helpers** -- the legacy entry points
    (``surface_specificity_context``, ``merge_specificity_contexts``,
    ``specificity_signal_terms``, ``evaluate_specificity_support``,
    ``specificity_audit_snapshot``, ``campaign_proof_terms_from_audit``)
    keep their original signatures so the Atlas wrappers can be a
    one-line re-export. The blog and campaign generators call these
    directly.
  * **Pack contract** -- ``evaluate_witness_specificity(input, *, policy)``
    matches the same shape as ``evaluate_blog_post`` and
    ``evaluate_campaign``: takes a ``QualityInput`` + ``QualityPolicy``,
    returns a ``QualityReport``. This is the surface a future product
    pack composes against.

Recognised ``input.context`` keys (for the pack contract):

  * ``surface``: str -- ``"blog"`` / ``"campaign"`` / ``"report"``;
    drives the default for ``allow_company_names`` (False on
    blog/campaign, True elsewhere).
  * ``anchor_examples``: dict[str, list[dict]] -- per-label witness
    rows already sanitized for the surface.
  * ``witness_highlights``: list[dict] -- ungrouped witness rows.
  * ``reference_ids``: dict[str, list | dict] -- canonical reference
    IDs surfaced from the upstream context payload.

Recognised ``policy.thresholds`` keys (all optional):

  * ``min_anchor_hits``: int (default 1)
  * ``require_anchor_support``: bool (default True)
  * ``require_timing_or_numeric_when_available``: bool (default False)
  * ``include_competitor_terms``: bool (default True)
  * ``allow_company_names``: bool (default depends on ``surface``)
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from typing import Any

from .types import (
    GateDecision,
    GateFinding,
    GateSeverity,
    QualityInput,
    QualityPolicy,
    QualityReport,
)


# ---- Regex / text helpers ----

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_NUMERIC_TEXT_RE = re.compile(
    r"(?:[$€£]?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|k|m|b|x))?(?:/[a-z]+)?)",
    re.IGNORECASE,
)
_TIMING_TEXT_RE = re.compile(
    r"\b(?:renewal|deadline|contract|quarter|month|week|year|q[1-4]|fy\d{2,4}|today|tomorrow|immediate)\b",
    re.IGNORECASE,
)


def _normalize_text(value: Any) -> str:
    """Strip HTML, lowercase, collapse whitespace.

    Mirrors the legacy helper in ``_b2b_specificity.py`` byte for byte
    so the pack matches existing fixtures and call sites.
    """
    text = _HTML_TAG_RE.sub(" ", str(value or ""))
    text = text.lower()
    return _WHITESPACE_RE.sub(" ", text).strip()


def _contains_term(normalized_text: str, term: str) -> bool:
    """Whole-token substring match (no embedded matches)."""
    clean_term = _normalize_text(term)
    if not normalized_text or not clean_term:
        return False
    pattern = re.compile(
        r"(?<![a-z0-9])" + re.escape(clean_term) + r"(?![a-z0-9])"
    )
    return bool(pattern.search(normalized_text))


def _copy_json_dict(value: Any) -> dict[str, Any]:
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _copy_json_list(value: Any) -> list[Any]:
    return copy.deepcopy(value) if isinstance(value, list) else []


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


def _coerce_anchor_examples(value: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        return {}
    resolved: dict[str, list[dict[str, Any]]] = {}
    for label, rows in value.items():
        if not isinstance(rows, list):
            continue
        clean_rows = [dict(row) for row in rows if isinstance(row, dict)]
        if clean_rows:
            resolved[str(label)] = clean_rows
    return resolved


def _coerce_witness_highlights(value: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _redact_company(text: str, company: str) -> str:
    clean_text = str(text or "").strip()
    clean_company = str(company or "").strip()
    if not clean_text or not clean_company:
        return clean_text
    pattern = re.compile(re.escape(clean_company), re.IGNORECASE)
    redacted = pattern.sub("a customer", clean_text)
    return _WHITESPACE_RE.sub(" ", redacted).strip()


def _sanitize_witness_row(
    witness: dict[str, Any],
    *,
    allow_company_names: bool,
) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key in (
        "witness_id",
        "witness_type",
        "time_anchor",
        "competitor",
        "pain_category",
        "signal_type",
        "replacement_mode",
        "operating_model_shift",
        "productivity_delta_claim",
        "org_pressure_type",
        "reviewer_title",
        "company_size",
        "industry",
        "source",
        "source_id",
        "_sid",
        "grounding_status",
        "phrase_polarity",
        "phrase_subject",
        "phrase_role",
        "phrase_verbatim",
        "pain_confidence",
    ):
        value = witness.get(key)
        if value in (None, "", [], {}):
            continue
        clean[key] = copy.deepcopy(value)

    excerpt = str(witness.get("excerpt_text") or "").strip()
    reviewer_company = str(witness.get("reviewer_company") or "").strip()
    if excerpt:
        clean["excerpt_text"] = (
            excerpt if allow_company_names else _redact_company(excerpt, reviewer_company)
        )
    if allow_company_names and reviewer_company:
        clean["reviewer_company"] = reviewer_company

    numeric_literals = _copy_json_dict(witness.get("numeric_literals"))
    if numeric_literals:
        clean["numeric_literals"] = numeric_literals
    signal_tags = _copy_json_list(witness.get("signal_tags"))
    if signal_tags:
        clean["signal_tags"] = signal_tags
    return clean


def _witness_marker(witness: dict[str, Any]) -> str:
    witness_id = str(witness.get("witness_id") or "").strip()
    if witness_id:
        return witness_id
    excerpt = _normalize_text(witness.get("excerpt_text"))
    if excerpt:
        return excerpt
    time_anchor = _normalize_text(witness.get("time_anchor"))
    competitor = _normalize_text(witness.get("competitor"))
    pain = _normalize_text(witness.get("pain_category"))
    return "|".join(part for part in (time_anchor, competitor, pain) if part)


def _numeric_terms_from_witness(witness: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    numeric_literals = witness.get("numeric_literals")
    if isinstance(numeric_literals, dict):
        for values in numeric_literals.values():
            if isinstance(values, list):
                for value in values:
                    token = _normalize_text(value)
                    if token and not (token.isdigit() and len(token) < 3):
                        terms.add(token)
            else:
                token = _normalize_text(values)
                if token and not (token.isdigit() and len(token) < 3):
                    terms.add(token)
    excerpt = str(witness.get("excerpt_text") or "")
    for match in _NUMERIC_TEXT_RE.findall(excerpt):
        token = _normalize_text(match)
        if token and not (token.isdigit() and len(token) < 3):
            terms.add(token)
    return terms


def _timing_terms_from_witness(witness: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    time_anchor = str(witness.get("time_anchor") or "").strip()
    if time_anchor:
        terms.add(_normalize_text(time_anchor))
    excerpt = str(witness.get("excerpt_text") or "")
    for match in _TIMING_TEXT_RE.findall(excerpt):
        token = _normalize_text(match)
        if token:
            terms.add(token)
    return terms


def _list_terms(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    terms: set[str] = set()
    for item in value:
        token = _normalize_text(item)
        if token:
            terms.add(token)
    return terms


# ---- Public functional API (legacy entry points) ----


def surface_specificity_context(
    source: dict[str, Any] | None,
    *,
    surface: str,
    nested_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a sanitized specificity context for a render surface.

    Reads ``reasoning_anchor_examples`` / ``anchor_examples``,
    ``reasoning_witness_highlights`` / ``witness_highlights``, and
    ``reasoning_reference_ids`` / ``reference_ids`` from ``source`` (and
    optionally from nested keys). Returns a dict that downstream packs
    can pass into ``evaluate_specificity_support`` without re-deriving
    the surface-specific redaction policy.

    For ``surface in {"blog", "campaign"}`` company names are redacted
    out of excerpt text; other surfaces (battle cards, internal
    reports) keep the names intact.
    """
    allow_company_names = surface not in {"blog", "campaign"}
    sources: list[dict[str, Any]] = []
    if isinstance(source, dict):
        sources.append(source)
        for key in nested_keys:
            nested = source.get(key)
            if isinstance(nested, dict):
                sources.append(nested)

    anchor_examples: dict[str, list[dict[str, Any]]] = {}
    witness_highlights: list[dict[str, Any]] = []
    reference_ids: dict[str, Any] = {}

    for payload in sources:
        if not anchor_examples:
            anchor_examples = _coerce_anchor_examples(
                payload.get("reasoning_anchor_examples") or payload.get("anchor_examples")
            )
        if not witness_highlights:
            witness_highlights = _coerce_witness_highlights(
                payload.get("reasoning_witness_highlights") or payload.get("witness_highlights")
            )
        if not reference_ids:
            candidate_refs = payload.get("reasoning_reference_ids") or payload.get("reference_ids")
            if isinstance(candidate_refs, dict):
                reference_ids = _copy_json_dict(candidate_refs)

    sanitized_anchors: dict[str, list[dict[str, Any]]] = {}
    for label, rows in anchor_examples.items():
        clean_rows = [
            _sanitize_witness_row(row, allow_company_names=allow_company_names)
            for row in rows
        ]
        clean_rows = [row for row in clean_rows if row]
        if clean_rows:
            sanitized_anchors[label] = clean_rows

    sanitized_highlights = [
        _sanitize_witness_row(row, allow_company_names=allow_company_names)
        for row in witness_highlights
    ]
    sanitized_highlights = [row for row in sanitized_highlights if row]

    context: dict[str, Any] = {}
    if sanitized_anchors:
        context["anchor_examples"] = sanitized_anchors
    if sanitized_highlights:
        context["witness_highlights"] = sanitized_highlights
    if reference_ids:
        context["reference_ids"] = reference_ids
    return context


def merge_specificity_contexts(*contexts: dict[str, Any] | None) -> dict[str, Any]:
    """Combine N specificity contexts, de-duping witnesses by marker."""
    anchor_examples: dict[str, list[dict[str, Any]]] = {}
    witness_highlights: list[dict[str, Any]] = []
    reference_ids: dict[str, Any] = {}
    seen_highlights: set[str] = set()

    for context in contexts:
        if not isinstance(context, dict):
            continue
        for label, rows in _coerce_anchor_examples(context.get("anchor_examples")).items():
            bucket = anchor_examples.setdefault(label, [])
            seen_bucket = {_witness_marker(row) for row in bucket if _witness_marker(row)}
            for row in rows:
                marker = _witness_marker(row)
                if marker and marker in seen_bucket:
                    continue
                bucket.append(copy.deepcopy(row))
                if marker:
                    seen_bucket.add(marker)
        for row in _coerce_witness_highlights(context.get("witness_highlights")):
            marker = _witness_marker(row)
            if marker and marker in seen_highlights:
                continue
            witness_highlights.append(copy.deepcopy(row))
            if marker:
                seen_highlights.add(marker)
        candidate_refs = context.get("reference_ids")
        if isinstance(candidate_refs, dict):
            for key, value in candidate_refs.items():
                if key not in reference_ids:
                    reference_ids[key] = copy.deepcopy(value)
                    continue
                if isinstance(reference_ids[key], list) and isinstance(value, list):
                    seen_values = {str(item) for item in reference_ids[key]}
                    for item in value:
                        if str(item) in seen_values:
                            continue
                        reference_ids[key].append(copy.deepcopy(item))
                        seen_values.add(str(item))

    merged: dict[str, Any] = {}
    if anchor_examples:
        merged["anchor_examples"] = anchor_examples
    if witness_highlights:
        merged["witness_highlights"] = witness_highlights
    if reference_ids:
        merged["reference_ids"] = reference_ids
    return merged


def specificity_signal_terms(
    *,
    anchor_examples: dict[str, list[dict[str, Any]]] | None = None,
    witness_highlights: list[dict[str, Any]] | None = None,
    allow_company_names: bool,
) -> dict[str, set[str]]:
    """Derive normalized term sets the content can be checked against."""
    companies: set[str] = set()
    timing_terms: set[str] = set()
    numeric_terms: set[str] = set()
    competitor_terms: set[str] = set()
    pain_terms: set[str] = set()
    workflow_terms: set[str] = set()

    rows: list[dict[str, Any]] = []
    for group_rows in (anchor_examples or {}).values():
        rows.extend(group_rows)
    rows.extend(witness_highlights or [])

    for witness in rows:
        if allow_company_names:
            reviewer_company = _normalize_text(witness.get("reviewer_company"))
            if reviewer_company:
                companies.add(reviewer_company)
        competitor = _normalize_text(witness.get("competitor"))
        if competitor:
            competitor_terms.add(competitor)
        timing_terms.update(_timing_terms_from_witness(witness))
        numeric_terms.update(_numeric_terms_from_witness(witness))

        for key in ("pain_category", "signal_type", "org_pressure_type"):
            token = _normalize_text(witness.get(key))
            if token:
                pain_terms.add(token)
        pain_terms.update(_list_terms(witness.get("signal_tags")))

        for key in ("replacement_mode", "operating_model_shift", "productivity_delta_claim"):
            token = _normalize_text(witness.get(key))
            if token and token != "none" and token != "unknown":
                workflow_terms.add(token)

    return {
        "companies": companies,
        "timing_terms": timing_terms,
        "numeric_terms": numeric_terms,
        "competitor_terms": competitor_terms,
        "pain_terms": pain_terms,
        "workflow_terms": workflow_terms,
    }


def evaluate_specificity_support(
    text: str,
    *,
    anchor_examples: dict[str, list[dict[str, Any]]] | None = None,
    witness_highlights: list[dict[str, Any]] | None = None,
    reference_ids: dict[str, Any] | None = None,
    allow_company_names: bool,
    min_anchor_hits: int = 1,
    require_anchor_support: bool = True,
    require_timing_or_numeric_when_available: bool = False,
    include_competitor_terms: bool = True,
    numeric_term_filter: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Score the specificity support of ``text`` against the witness pool."""
    normalized_text = _normalize_text(text)
    blocking_issues: list[str] = []
    warnings: list[str] = []

    refs = reference_ids if isinstance(reference_ids, dict) else {}
    witness_refs = [
        str(value).strip()
        for value in (refs.get("witness_ids") or [])
        if str(value or "").strip()
    ]
    if witness_refs and not (anchor_examples or witness_highlights):
        warnings.append("witness references exist but no anchor examples or witness highlights were surfaced")

    terms = specificity_signal_terms(
        anchor_examples=anchor_examples,
        witness_highlights=witness_highlights,
        allow_company_names=allow_company_names,
    )
    if numeric_term_filter is not None:
        terms["numeric_terms"] = {
            term for term in terms["numeric_terms"] if numeric_term_filter(term)
        }
    active_groups = {
        "timing_terms": terms["timing_terms"],
        "numeric_terms": terms["numeric_terms"],
        "pain_terms": terms["pain_terms"],
        "workflow_terms": terms["workflow_terms"],
    }
    if include_competitor_terms:
        active_groups["competitor_terms"] = terms["competitor_terms"]
    if allow_company_names:
        active_groups["companies"] = terms["companies"]

    matched_groups = [
        group
        for group, group_terms in active_groups.items()
        if group_terms and any(_contains_term(normalized_text, term) for term in group_terms)
    ]

    anchors_available = bool(anchor_examples or witness_highlights)
    structured_terms_available = any(group_terms for group_terms in active_groups.values())

    if require_anchor_support and anchors_available and structured_terms_available:
        if len(matched_groups) < max(1, int(min_anchor_hits)):
            blocking_issues.append(
                "content does not reference any witness-backed anchor despite anchors being available"
            )
    elif anchors_available and not structured_terms_available:
        warnings.append(
            "witness-backed anchors are available but do not expose concrete timing, numeric, competitor, or pain terms"
        )

    has_timing_or_numeric = bool(terms["timing_terms"] or terms["numeric_terms"])
    matched_timing_or_numeric = any(
        group in matched_groups for group in ("timing_terms", "numeric_terms")
    )
    if require_timing_or_numeric_when_available and has_timing_or_numeric and not matched_timing_or_numeric:
        blocking_issues.append(
            "content omits a concrete timing or numeric anchor even though one is available"
        )

    if allow_company_names and terms["companies"] and "companies" not in matched_groups:
        warnings.append("named-account anchor exists but content does not mention a named example")
    if terms["numeric_terms"] and "numeric_terms" not in matched_groups:
        warnings.append("money-backed anchor exists but content does not mention the concrete spend signal")
    if terms["timing_terms"] and "timing_terms" not in matched_groups:
        warnings.append("timing anchor exists but content does not mention the live trigger window")
    if include_competitor_terms and terms["competitor_terms"] and "competitor_terms" not in matched_groups:
        warnings.append("competitor-backed anchor exists but content does not mention the competitive alternative")

    return {
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "matched_groups": matched_groups,
        "signal_terms": terms,
    }


def specificity_audit_snapshot(
    text: str,
    *,
    anchor_examples: dict[str, list[dict[str, Any]]] | None = None,
    witness_highlights: list[dict[str, Any]] | None = None,
    reference_ids: dict[str, Any] | None = None,
    allow_company_names: bool,
    min_anchor_hits: int = 1,
    require_anchor_support: bool = True,
    require_timing_or_numeric_when_available: bool = False,
    include_competitor_terms: bool = True,
) -> dict[str, Any]:
    """Snapshot of ``evaluate_specificity_support`` with rendered counts.

    The output dict is JSON-safe (sorted lists for term sets, raw dict
    for reference_ids) so it can be persisted to ``b2b_audit`` rows.
    """
    evaluation = evaluate_specificity_support(
        text,
        anchor_examples=anchor_examples,
        witness_highlights=witness_highlights,
        reference_ids=reference_ids,
        allow_company_names=allow_company_names,
        min_anchor_hits=min_anchor_hits,
        require_anchor_support=require_anchor_support,
        require_timing_or_numeric_when_available=require_timing_or_numeric_when_available,
        include_competitor_terms=include_competitor_terms,
    )
    signal_terms = {
        key: sorted(values)
        for key, values in (evaluation.get("signal_terms") or {}).items()
        if values
    }
    matched_groups = list(evaluation.get("matched_groups") or [])
    available_groups = sorted(signal_terms.keys())
    return {
        "status": "fail" if evaluation.get("blocking_issues") else "pass",
        "blocking_issues": list(evaluation.get("blocking_issues") or []),
        "warnings": list(evaluation.get("warnings") or []),
        "matched_groups": matched_groups,
        "available_groups": available_groups,
        "missing_groups": [
            group for group in available_groups if group not in set(matched_groups)
        ],
        "signal_terms": signal_terms,
        "anchor_count": sum(len(rows) for rows in (anchor_examples or {}).values()),
        "anchor_labels": sorted((anchor_examples or {}).keys()),
        "highlight_count": len(witness_highlights or []),
        "reference_ids": _copy_json_dict(reference_ids),
        "reference_id_counts": _reference_id_counts(reference_ids),
    }


def campaign_proof_terms_from_audit(
    audit: dict[str, Any] | None,
    *,
    channel: str,
    limit: int,
) -> list[str]:
    """Derive proof terms from an audit's signal_terms structure.

    Used by the campaign pack as a fallback when the caller did not
    pre-resolve proof terms. Stays in this module because it operates
    on the audit dict shape this pack produces.
    """
    if not isinstance(audit, dict):
        return []
    signal_terms = audit.get("signal_terms")
    if not isinstance(signal_terms, dict):
        return []

    blocking_issues = [
        str(issue).strip().lower()
        for issue in (audit.get("blocking_issues") or [])
        if str(issue or "").strip()
    ]
    prefer_numeric_timing = any(
        "timing or numeric anchor" in issue for issue in blocking_issues
    )
    group_order = ["numeric_terms", "timing_terms"]
    if not prefer_numeric_timing:
        if channel != "email_cold":
            group_order.append("competitor_terms")
        group_order.extend(["pain_terms", "workflow_terms"])

    resolved: list[str] = []
    seen: set[str] = set()
    for group_name in group_order:
        values = signal_terms.get(group_name)
        if not isinstance(values, list):
            continue
        ranked_values = sorted(
            (str(value or "").strip() for value in values),
            key=lambda token: (-len(token), token.lower()),
        )
        for token in ranked_values:
            if not token:
                continue
            rendered = token.replace("_", " ")
            marker = rendered.lower()
            if marker in seen:
                continue
            resolved.append(rendered)
            seen.add(marker)
            if len(resolved) >= max(1, int(limit)):
                return resolved
    return resolved


# ---- Pack contract ----


def evaluate_witness_specificity(
    input: QualityInput,
    *,
    policy: QualityPolicy | None = None,
) -> QualityReport:
    """Pack-contract entry point. See module docstring for context keys."""
    body = str(input.content or "")
    context = dict(input.context or {})
    surface = str(context.get("surface") or "").strip()
    anchor_examples = context.get("anchor_examples")
    if not isinstance(anchor_examples, dict):
        anchor_examples = None
    witness_highlights = context.get("witness_highlights")
    if not isinstance(witness_highlights, list):
        witness_highlights = None
    reference_ids = context.get("reference_ids")
    if not isinstance(reference_ids, dict):
        reference_ids = None

    thresholds = policy.thresholds if policy is not None else {}
    allow_company_names = thresholds.get("allow_company_names")
    if allow_company_names is None:
        allow_company_names = surface not in {"blog", "campaign"}
    min_anchor_hits = int(thresholds.get("min_anchor_hits", 1))
    require_anchor_support = bool(thresholds.get("require_anchor_support", True))
    require_timing_or_numeric_when_available = bool(
        thresholds.get("require_timing_or_numeric_when_available", False)
    )
    include_competitor_terms = bool(thresholds.get("include_competitor_terms", True))

    evaluation = evaluate_specificity_support(
        body,
        anchor_examples=anchor_examples,
        witness_highlights=witness_highlights,
        reference_ids=reference_ids,
        allow_company_names=bool(allow_company_names),
        min_anchor_hits=min_anchor_hits,
        require_anchor_support=require_anchor_support,
        require_timing_or_numeric_when_available=require_timing_or_numeric_when_available,
        include_competitor_terms=include_competitor_terms,
    )

    findings: list[GateFinding] = []
    for issue in evaluation.get("blocking_issues") or ():
        findings.append(
            GateFinding(
                code="witness_specificity_blocker",
                message=str(issue),
                severity=GateSeverity.BLOCKER,
            )
        )
    for warning in evaluation.get("warnings") or ():
        findings.append(
            GateFinding(
                code="witness_specificity_warning",
                message=str(warning),
                severity=GateSeverity.WARNING,
            )
        )

    if any(f.severity == GateSeverity.BLOCKER for f in findings):
        decision = GateDecision.BLOCK
    elif findings:
        decision = GateDecision.WARN
    else:
        decision = GateDecision.PASS

    matched_groups = list(evaluation.get("matched_groups") or ())
    signal_terms = {
        key: sorted(values)
        for key, values in (evaluation.get("signal_terms") or {}).items()
        if values
    }
    metadata = {
        "matched_groups": tuple(matched_groups),
        "signal_terms": signal_terms,
        "available_groups": tuple(sorted(signal_terms.keys())),
        "missing_groups": tuple(
            group for group in sorted(signal_terms.keys()) if group not in set(matched_groups)
        ),
    }
    return QualityReport(
        passed=not any(f.severity == GateSeverity.BLOCKER for f in findings),
        decision=decision,
        findings=tuple(findings),
        metadata=metadata,
    )


__all__ = [
    "campaign_proof_terms_from_audit",
    "evaluate_specificity_support",
    "evaluate_witness_specificity",
    "merge_specificity_contexts",
    "specificity_audit_snapshot",
    "specificity_signal_terms",
    "surface_specificity_context",
]
