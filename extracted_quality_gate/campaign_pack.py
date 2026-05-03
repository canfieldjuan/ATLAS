"""Campaign quality pack: deterministic validators for outbound campaigns.

Owned by ``extracted_quality_gate`` (PR-B4b). The single public entry
point ``evaluate_campaign`` is pure: no DB, no clock, no network. It
takes a :class:`QualityInput` (subject + body + cta in
``content``/``context``) and a campaign payload in ``context`` and
returns a :class:`QualityReport`.

Specificity audit (witness anchor support, evidence coverage) stays
Atlas-side -- per the PR-B5 framing, that ships as its own pack
later. The Atlas wrapper runs ``specificity_audit_snapshot`` first
and passes its output (blocking issues, warnings) to this pack as
``context['specificity_blocking_issues']`` / ``context['specificity_warnings']``;
the pack appends its own findings.

Pack scope:

  * Proof-term coverage: each ``required_proof_terms`` entry must
    appear in the body (case-insensitive whole-token match). Missing
    terms add a single ``missing_exact_proof_term`` blocker (only
    when ``require_anchor_support`` is True and anchors / witnesses
    are available -- otherwise enforcement falls back to specificity).
  * Report-tier banned language: if ``campaign.tier == "report"``,
    the body+CTA cannot use words like "dashboard", "live feed",
    "free trial", "software", "platform" -- those are product-tier
    language, not analyst-report language.
  * Forbidden terms: cold-email channels cannot name
    competitors / incumbents the recipient already knows, gated by
    ``campaign.target_mode``. ``vendor_retention`` blocks
    competitor names; ``challenger_intel`` blocks incumbent names.
  * Private account name leak: anchor / witness data may surface
    ``reviewer_company`` strings; if any of them appear in the
    outbound message, that is a confidentiality breach.

Public API:

    evaluate_campaign(
        input: QualityInput,
        *,
        policy: QualityPolicy | None = None,
    ) -> QualityReport

Recognised ``input.context`` keys:

  * ``subject``: str
  * ``body``: str (body markdown / plain)
  * ``cta``: str
  * ``campaign``: dict -- payload with ``channel``, ``target_mode``,
    ``tier``, ``metadata``, ``signal_summary``, ``competitors_considering``,
    ``incumbent_archetypes`` (matches the legacy ``campaign`` arg
    of ``campaign_policy_audit_snapshot``).
  * ``required_proof_terms``: tuple[str, ...] (caller-resolved).
  * ``anchor_examples``: dict[str, list[dict]]
  * ``witness_highlights``: tuple[dict, ...]
  * ``specificity_blocking_issues``: tuple[str, ...] (from atlas-side
    audit; the pack passes them through into the legacy report shape)
  * ``specificity_warnings``: tuple[str, ...] (likewise)

Recognised ``policy.thresholds`` keys (all optional):

  * ``require_anchor_support``: bool (default True)
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .types import (
    GateDecision,
    GateFinding,
    GateSeverity,
    QualityInput,
    QualityPolicy,
    QualityReport,
)


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_REPORT_TIER_BANNED = re.compile(
    r"\b(dashboard|live feed|free trial|software|platform)\b",
    re.IGNORECASE,
)


def _normalize_text(value: Any) -> str:
    """Strip HTML, lowercase, collapse whitespace.

    Mirrors the legacy ``_normalize_text`` in
    ``atlas_brain.autonomous.tasks._b2b_specificity`` so the pack
    behaves identically on the same inputs.
    """
    text = _HTML_TAG_RE.sub(" ", str(value or ""))
    text = text.lower()
    return _WHITESPACE_RE.sub(" ", text).strip()


def _contains_term(normalized_text: str, term: str) -> bool:
    """Whole-token match (no embedded substring matches)."""
    clean_term = _normalize_text(term)
    if not normalized_text or not clean_term:
        return False
    pattern = re.compile(
        r"(?<![a-z0-9])" + re.escape(clean_term) + r"(?![a-z0-9])"
    )
    return bool(pattern.search(normalized_text))


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    """Stable de-dup by lowercase marker. Mirrors the legacy helper."""
    resolved: list[str] = []
    seen: set[str] = set()
    for value in values:
        marker = str(value or "").strip().lower()
        if not marker or marker in seen:
            continue
        seen.add(marker)
        resolved.append(str(value).strip())
    return resolved


def _campaign_collection(payload: Mapping[str, Any], key: str) -> Any:
    """Read ``payload[key]`` with metadata-fallback semantics."""
    value = payload.get(key)
    if value not in (None, "", [], {}):
        return value
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _campaign_name_terms(value: Any) -> list[str]:
    """Extract vendor / incumbent / alternative names from a payload section."""
    terms: list[str] = []
    seen: set[str] = set()
    rows: list[Any] = []
    if isinstance(value, list):
        rows = list(value)
    elif isinstance(value, dict):
        rows = [value]

    for row in rows:
        name = ""
        if isinstance(row, dict):
            for key in ("name", "vendor_name", "incumbent_vendor", "alternative_vendor"):
                candidate = str(row.get(key) or "").strip()
                if candidate:
                    name = candidate
                    break
        else:
            name = str(row or "").strip()
        marker = _normalize_text(name)
        if not marker or marker in seen:
            continue
        seen.add(marker)
        terms.append(name)
    return terms


def _campaign_private_company_terms(
    anchor_examples: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    witness_highlights: Sequence[Mapping[str, Any]] | None,
) -> list[str]:
    """Surface private ``reviewer_company`` strings that must not leak."""
    terms: list[str] = []
    seen: set[str] = set()
    rows: list[Mapping[str, Any]] = []
    for group_rows in (anchor_examples or {}).values():
        rows.extend(group_rows or ())
    rows.extend(witness_highlights or ())
    for row in rows:
        company = str(row.get("reviewer_company") or "").strip()
        marker = _normalize_text(company)
        if not marker or marker in seen:
            continue
        seen.add(marker)
        terms.append(company)
    return terms


def evaluate_campaign(
    input: QualityInput,
    *,
    policy: QualityPolicy | None = None,
) -> QualityReport:
    """Run the deterministic campaign-quality validators.

    Returns a :class:`QualityReport`. ``decision`` is:

      * ``BLOCK`` when any blocker fires (proof-term gap, banned
        report-tier word, forbidden competitor/incumbent name in
        cold email, private company leak, or any specificity
        blocker passed in via context).
      * ``WARN`` when only specificity warnings flow through.
      * ``PASS`` otherwise.

    The report's ``metadata`` mirrors the legacy
    ``campaign_policy_audit_snapshot`` dict shape so Atlas-side
    callers can preserve their existing telemetry.
    """
    context = dict(input.context or {})
    subject = str(context.get("subject") or "")
    body = str(context.get("body") or input.content or "")
    cta = str(context.get("cta") or "")
    payload: Mapping[str, Any] = context.get("campaign") or {}

    # Channel / target_mode / tier with metadata fallback (matches legacy)
    channel = str(payload.get("channel") or "").strip()
    target_mode = str(payload.get("target_mode") or "").strip()
    tier = str(payload.get("tier") or "").strip()
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        if not tier:
            tier = str(metadata.get("tier") or "").strip()
        if not target_mode:
            target_mode = str(metadata.get("target_mode") or "").strip()

    # Proof-term policy switch.
    require_anchor_support = True
    if policy is not None:
        threshold = policy.thresholds.get("require_anchor_support")
        if isinstance(threshold, bool):
            require_anchor_support = threshold

    findings: list[GateFinding] = []

    # ---- Specificity audit pass-through (atlas computes; pack mirrors) ----
    specificity_blockers = tuple(
        str(issue).strip()
        for issue in (context.get("specificity_blocking_issues") or ())
        if str(issue or "").strip()
    )
    specificity_warnings = tuple(
        str(warning).strip()
        for warning in (context.get("specificity_warnings") or ())
        if str(warning or "").strip()
    )
    for issue in specificity_blockers:
        findings.append(
            GateFinding(
                code="specificity_audit_blocker",
                message=issue,
                severity=GateSeverity.BLOCKER,
            )
        )
    for warning in specificity_warnings:
        findings.append(
            GateFinding(
                code="specificity_audit_warning",
                message=warning,
                severity=GateSeverity.WARNING,
            )
        )

    # ---- Proof-term coverage ----
    required_proof_terms = _dedupe_strings(
        list(context.get("required_proof_terms") or ())
    )
    normalized_body = _normalize_text(body)
    used_proof_terms = [
        term for term in required_proof_terms if _contains_term(normalized_body, term)
    ]
    anchor_examples = context.get("anchor_examples")
    witness_highlights = context.get("witness_highlights") or ()
    has_anchor_or_witness = bool(anchor_examples) or bool(witness_highlights)
    if (
        required_proof_terms
        and require_anchor_support
        and has_anchor_or_witness
        and not used_proof_terms
    ):
        findings.append(
            GateFinding(
                code="missing_exact_proof_term",
                message="missing_exact_proof_term",
                severity=GateSeverity.BLOCKER,
                metadata={"required_proof_terms": tuple(required_proof_terms)},
            )
        )

    # ---- Report-tier banned language (body + CTA only; subject is exempt) ----
    normalized_report = _normalize_text(" ".join(part for part in (body, cta) if part))
    if tier.lower() == "report":
        report_match = _REPORT_TIER_BANNED.search(normalized_report)
        if report_match:
            findings.append(
                GateFinding(
                    code="report_tier_language",
                    message=f"report_tier_language:{report_match.group(1)}",
                    severity=GateSeverity.BLOCKER,
                    metadata={"banned_word": report_match.group(1)},
                )
            )

    # ---- Forbidden terms in cold email ----
    normalized_message = _normalize_text(
        " ".join(part for part in (subject, body, cta) if part)
    )
    forbidden_label, forbidden_terms = _resolve_forbidden_terms(
        channel=channel, target_mode=target_mode, payload=payload
    )
    for term in _dedupe_strings(forbidden_terms):
        if _contains_term(normalized_message, term):
            findings.append(
                GateFinding(
                    code=forbidden_label,
                    message=f"{forbidden_label}:{term}",
                    severity=GateSeverity.BLOCKER,
                    metadata={"term": term},
                )
            )

    # ---- Private account name leak ----
    private_terms = _campaign_private_company_terms(
        anchor_examples if isinstance(anchor_examples, Mapping) else None,
        witness_highlights,
    )
    for company in private_terms:
        if _contains_term(normalized_message, company):
            findings.append(
                GateFinding(
                    code="private_account_name_leak",
                    message=f"private_account_name_leak:{company}",
                    severity=GateSeverity.BLOCKER,
                    metadata={"company": company},
                )
            )

    return _build_report(
        findings=findings,
        required_proof_terms=required_proof_terms,
        used_proof_terms=used_proof_terms,
    )


def _resolve_forbidden_terms(
    *,
    channel: str,
    target_mode: str,
    payload: Mapping[str, Any],
) -> tuple[str, list[str]]:
    """Resolve the forbidden-term label + list for the (channel, target_mode) pair.

    Only ``email_cold`` is gated today. ``vendor_retention`` blocks
    competitor names; ``challenger_intel`` blocks incumbent names.
    Other channel/target_mode combos return an empty list.
    """
    if channel != "email_cold":
        return ("", [])

    forbidden_terms: list[str] = []
    if target_mode == "vendor_retention":
        label = "competitor_name_in_email_cold"
        forbidden_terms.extend(
            _campaign_name_terms(_campaign_collection(payload, "competitors_considering"))
        )
        signal_summary = payload.get("signal_summary")
        if isinstance(signal_summary, dict):
            forbidden_terms.extend(
                _campaign_name_terms(signal_summary.get("competitor_distribution"))
            )
        return (label, forbidden_terms)

    if target_mode == "challenger_intel":
        label = "incumbent_name_in_email_cold"
        forbidden_terms.extend(
            _campaign_name_terms(_campaign_collection(payload, "competitors_considering"))
        )
        signal_summary = payload.get("signal_summary")
        if isinstance(signal_summary, dict):
            forbidden_terms.extend(
                _campaign_name_terms(signal_summary.get("incumbents_losing"))
            )
        incumbent_archetypes = payload.get("incumbent_archetypes")
        if isinstance(incumbent_archetypes, dict):
            for rows in incumbent_archetypes.values():
                forbidden_terms.extend(_campaign_name_terms(rows))
        return (label, forbidden_terms)

    return ("", [])


def _build_report(
    *,
    findings: list[GateFinding],
    required_proof_terms: list[str],
    used_proof_terms: list[str],
) -> QualityReport:
    blockers = [f for f in findings if f.severity == GateSeverity.BLOCKER]
    warnings = [f for f in findings if f.severity == GateSeverity.WARNING]

    if blockers:
        decision = GateDecision.BLOCK
    elif warnings:
        decision = GateDecision.WARN
    else:
        decision = GateDecision.PASS

    blocking_messages = _dedupe_strings([f.message for f in blockers])
    warning_messages = _dedupe_strings([f.message for f in warnings])
    metadata = {
        "status": "fail" if blocking_messages else "pass",
        "blocking_issues": tuple(blocking_messages),
        "warnings": tuple(warning_messages),
        "campaign_proof_terms": tuple(required_proof_terms),
        "required_proof_terms": tuple(required_proof_terms),
        "used_proof_terms": tuple(used_proof_terms),
        "unused_proof_terms": tuple(
            term for term in required_proof_terms if term not in used_proof_terms
        ),
        "primary_blocker": blocking_messages[0] if blocking_messages else None,
    }
    return QualityReport(
        passed=not blocking_messages,
        decision=decision,
        findings=tuple(findings),
        metadata=metadata,
    )


__all__ = [
    "evaluate_campaign",
]
