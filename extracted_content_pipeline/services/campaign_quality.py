"""Standalone campaign quality revalidation seam."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any


_PLACEHOLDER_RE = re.compile(r"\[(?:Name|Company|Your Name|First Name|Title)\]|\{\{.+?\}\}")
_PERCENT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<first>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?:-\s*(?P<second>\d+(?:,\d{3})*(?:\.\d+)?)\s*)?%"
)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[$]\s*)?(?P<first>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?:\s*-\s*(?P<second>\d+(?:,\d{3})*(?:\.\d+)?))?"
    r"\s*(?:[kKmMbB])?\+?"
)
_QUARTER_RE = re.compile(r"\bq(?P<quarter>[1-4])\b", re.IGNORECASE)
_SCAN_CLAIM_RE = re.compile(
    r"\b(?:we|i|our team)\s+"
    r"(?:scanned|analy[sz]ed|reviewed|looked at)\b",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_AGGREGATE_CLAIM_RE = re.compile(
    r"\b(?:daily|weekly|monthly|every\s+week|per\s+week|"
    r"every\s+(?:billing\s+)?cycle|"
    r"next\s+(?:billing\s+)?cycle|"
    r"across\s+(?:accounts|teams|queues|customers)|"
    r"another\s+[-\w]+(?:\s+[-\w]+){0,4}\s+(?:question|ticket|account|customer|case)|"
    r"same\s+pattern|"
    r"pile\s+up|"
    r"questions?\s+like\s+this|"
    r"ticket\s+questions?|"
    r"support\s+queue|"
    r"support\s+ticket\s+patterns?|"
    r"support\s+questions|"
    r"(?:billing\s+)?questions?\s+lands?\s+in\s+support|"
    r"(?:question|answer|it)\s+(?:will|would|could)\s+lands?\s+in\s+support(?:\s+again)?|"
    r"keeps?\s+coming\s+up|"
    r"same\s+support\s+question|"
    r"users?\s+(?:(?:still|again|often|also)\s+)?"
    r"(?:hit|contact|open|file|send).{0,24}\bsupport|"
    r"stood\s+out|"
    r"(?:usually|typically)\s+means|"
    r"keeps?\s+(?:asking|coming|showing|hitting)|"
    r"(?:will|would|could)\s+come\s+back|"
    r"repeat(?:ed)?\s+(?:questions?|tickets?|cases?|requests?)|"
    r"(?:questions?|tickets?|cases?|requests?)\s+repeat|"
    r"same\s+\w+(?:\s+\w+){0,4}\s+again|"
    r"same\s+\w+(?:\s+\w+){0,4}\s+repeats?)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}
_TIME_WORDS = "minutes?|hours?|days?|weeks?|months?"
_TIMING_CLAIM_RE = re.compile(
    rf"\btakes?\s+(?:about\s+)?(?:a|an|\d+|{'|'.join(_NUMBER_WORDS)})?\s*"
    rf"(?:{_TIME_WORDS})\b|"
    rf"\b(?:send|deliver|draft|article).{{0,40}}\b(?:in|within)\s+"
    rf"(?:a|an|\d+|{'|'.join(_NUMBER_WORDS)})\s+(?:{_TIME_WORDS})\b",
    re.IGNORECASE,
)
_NUMBER_WORD_RE_TEMPLATE = r"(?<![-A-Za-z0-9]){word}(?![-A-Za-z0-9])"
_SCHEDULING_DURATION_RE = re.compile(
    r"\b(?:schedule|book|call|meeting|walkthrough|chat)\b.*\bminutes?\b|"
    r"\bminutes?\b.*\b(?:schedule|book|call|meeting|walkthrough|chat)\b",
    re.IGNORECASE,
)
_GENERATED_VALUE_KEYS = frozenset({
    "angle_reasoning",
    "body",
    "campaign_revalidation",
    "cold_email_context",
    "content",
    "cta",
    "email_body",
    "generation_usage",
    "metadata",
    "subject",
})
_SOURCE_EXCLUDED_KEYS = frozenset({
    "account_id",
    "contact_email",
    "email",
    "id",
    "scope",
    "source_id",
    "source_url",
    "target_id",
    "ticket_id",
    "user_id",
    "url",
})


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


def _sentences(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in _SENTENCE_RE.finditer(text)
        if match.group(0).strip()
    ]


def _number_token(value: str) -> str:
    text = value.replace(",", "").strip().lower()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _numeric_tokens(value: Any) -> set[str]:
    text = str(value or "")
    tokens: set[str] = set()
    for match in _PERCENT_RE.finditer(text):
        tokens.add(f"pct:{_number_token(match.group('first'))}")
        second = match.group("second")
        if second is not None:
            tokens.add(f"pct:{_number_token(second)}")
    for match in _NUMBER_RE.finditer(text):
        tokens.add(f"num:{_number_token(match.group('first'))}")
        second = match.group("second")
        if second is not None:
            tokens.add(f"num:{_number_token(second)}")
    for match in _QUARTER_RE.finditer(text):
        tokens.add(f"quarter:q{match.group('quarter')}")
    for word, number in _NUMBER_WORDS.items():
        if re.search(_NUMBER_WORD_RE_TEMPLATE.format(word=word), text, re.IGNORECASE):
            tokens.add(f"num:{number}")
    return tokens


def _source_text_values(value: Any, *, key: str = "") -> list[str]:
    normalized_key = key.strip().lower().replace(" ", "_")
    if normalized_key in _GENERATED_VALUE_KEYS or normalized_key in _SOURCE_EXCLUDED_KEYS:
        return []
    if isinstance(value, Mapping):
        values: list[str] = []
        for item_key, item_value in value.items():
            values.extend(_source_text_values(item_value, key=str(item_key)))
        return values
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = []
        for item in value:
            values.extend(_source_text_values(item, key=key))
        return values
    if value in (None, "", [], {}):
        return []
    return [str(value)]


def _source_numeric_tokens(
    campaign: dict[str, Any],
    context: dict[str, Any],
    proof_terms: list[str],
) -> set[str]:
    values = [
        *_source_text_values(campaign),
        *_source_text_values(context),
        *proof_terms,
    ]
    tokens: set[str] = set()
    for value in values:
        tokens.update(_numeric_tokens(value))
    evidence = campaign.get("evidence")
    if (
        isinstance(evidence, Sequence)
        and not isinstance(evidence, (str, bytes, bytearray))
        and len(evidence) == 1
    ):
        tokens.add("num:1")
    return tokens


def _unsupported_numeric_claims(content: str, allowed_tokens: set[str]) -> list[str]:
    unsupported: list[str] = []
    for sentence in _sentences(content):
        if _SCHEDULING_DURATION_RE.search(sentence):
            continue
        tokens = _numeric_tokens(sentence)
        if (tokens and not tokens.issubset(allowed_tokens)) or _TIMING_CLAIM_RE.search(sentence):
            unsupported.append(sentence)
    return unsupported


def _generated_content_blocks(campaign: dict[str, Any]) -> list[str]:
    return [
        str(campaign.get(key) or "").strip()
        for key in ("subject", "body", "cta")
        if str(campaign.get(key) or "").strip()
    ]


def _unsupported_scan_claims(content: str, source_values: list[str]) -> list[str]:
    source_text = " ".join(re.findall(r"[a-z0-9]+", " ".join(source_values).lower()))
    unsupported: list[str] = []
    for sentence in _sentences(content):
        if not _SCAN_CLAIM_RE.search(sentence):
            continue
        normalized = " ".join(re.findall(r"[a-z0-9]+", sentence.lower()))
        if normalized and normalized not in source_text:
            unsupported.append(sentence)
    return unsupported


def _unsupported_aggregate_claims(content: str, source_values: list[str]) -> list[str]:
    source_text = " ".join(re.findall(r"[a-z0-9]+", " ".join(source_values).lower()))
    unsupported: list[str] = []
    for sentence in _sentences(content):
        if not _AGGREGATE_CLAIM_RE.search(sentence):
            continue
        normalized = " ".join(re.findall(r"[a-z0-9]+", sentence.lower()))
        if normalized and normalized not in source_text:
            unsupported.append(sentence)
    return unsupported


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
    content = ". ".join(_generated_content_blocks(campaign))
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
    source_values = [
        *_source_text_values(campaign),
        *_source_text_values(resolved_context),
        *proof_terms,
    ]
    allowed_numeric_tokens = _source_numeric_tokens(
        campaign,
        resolved_context,
        proof_terms,
    )
    unsupported_numeric_claims = _unsupported_numeric_claims(
        content,
        allowed_numeric_tokens,
    )
    if unsupported_numeric_claims:
        blocking_issues.append("unsupported_numeric_claim")
    unsupported_scan_claims = _unsupported_scan_claims(content, source_values)
    if unsupported_scan_claims:
        blocking_issues.append("unsupported_scan_claim")
    unsupported_aggregate_claims = _unsupported_aggregate_claims(
        content,
        source_values,
    )
    if unsupported_aggregate_claims:
        blocking_issues.append("unsupported_aggregate_claim")

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
        "unsupported_numeric_claims": unsupported_numeric_claims,
        "unsupported_scan_claims": unsupported_scan_claims,
        "unsupported_aggregate_claims": unsupported_aggregate_claims,
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
