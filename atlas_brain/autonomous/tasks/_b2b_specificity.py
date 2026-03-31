import copy
import re
from typing import Any


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_NUMERIC_TEXT_RE = re.compile(
    r"(?:[$\u20ac\u00a3]?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|k|m|b|x))?(?:/[a-z]+)?)",
    re.IGNORECASE,
)
_TIMING_TEXT_RE = re.compile(
    r"\b(?:renewal|deadline|contract|quarter|month|week|year|q[1-4]|fy\d{2,4}|today|tomorrow|immediate)\b",
    re.IGNORECASE,
)


def _normalize_text(value: Any) -> str:
    text = _HTML_TAG_RE.sub(" ", str(value or ""))
    text = text.lower()
    return _WHITESPACE_RE.sub(" ", text).strip()


def _copy_json_dict(value: Any) -> dict[str, Any]:
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _copy_json_list(value: Any) -> list[Any]:
    return copy.deepcopy(value) if isinstance(value, list) else []


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


def surface_specificity_context(
    source: dict[str, Any] | None,
    *,
    surface: str,
    nested_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
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


def merge_specificity_contexts(*contexts: dict[str, Any] | None) -> dict[str, Any]:
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


def specificity_signal_terms(
    *,
    anchor_examples: dict[str, list[dict[str, Any]]] | None = None,
    witness_highlights: list[dict[str, Any]] | None = None,
    allow_company_names: bool,
) -> dict[str, set[str]]:
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


def _contains_term(normalized_text: str, term: str) -> bool:
    clean_term = _normalize_text(term)
    if not normalized_text or not clean_term:
        return False
    pattern = re.compile(
        r"(?<![a-z0-9])" + re.escape(clean_term) + r"(?![a-z0-9])"
    )
    return bool(pattern.search(normalized_text))


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
) -> dict[str, Any]:
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
    return {
        "status": "fail" if evaluation.get("blocking_issues") else "pass",
        "blocking_issues": list(evaluation.get("blocking_issues") or []),
        "warnings": list(evaluation.get("warnings") or []),
        "matched_groups": list(evaluation.get("matched_groups") or []),
        "available_groups": sorted(signal_terms.keys()),
        "signal_terms": signal_terms,
        "anchor_count": sum(len(rows) for rows in (anchor_examples or {}).values()),
        "anchor_labels": sorted((anchor_examples or {}).keys()),
        "highlight_count": len(witness_highlights or []),
        "reference_ids": _copy_json_dict(reference_ids),
    }


def latest_specificity_audit(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    current = metadata.get("latest_specificity_audit")
    if isinstance(current, dict):
        return copy.deepcopy(current)
    generation = metadata.get("generation_audit")
    if not isinstance(generation, dict):
        return {}
    specificity = generation.get("specificity")
    if isinstance(specificity, dict):
        return copy.deepcopy(specificity)
    if generation:
        return {
            "status": generation.get("status"),
            "blocking_issues": [],
            "warnings": [],
            "matched_groups": [],
        }
    return {}


def specificity_quality_summary(metadata: Any) -> dict[str, Any]:
    audit = latest_specificity_audit(metadata)
    blocking_issues = list(audit.get("blocking_issues") or [])
    warnings = list(audit.get("warnings") or [])
    return {
        "quality_status": audit.get("status"),
        "blocker_count": len(blocking_issues),
        "warning_count": len(warnings),
        "latest_error_summary": blocking_issues[0] if blocking_issues else None,
    }
