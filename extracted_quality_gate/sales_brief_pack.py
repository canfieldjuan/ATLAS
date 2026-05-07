"""Sales-brief quality pack: deterministic validators for sales briefs.

Sibling to ``report_pack`` (PR-Reports-1b), ``landing_page_pack``
(PR-LandingPage-1b), and ``campaign_pack`` / ``blog_pack``. Validates
the structured ``SalesBriefDraft`` shape from
``extracted_content_pipeline.sales_brief_ports``: title, headline,
ordered sections, reference ids. Pure-function discipline (no DB, no
LLM, no clock) -- sanitization belongs in the wrapper.

Public API:

    evaluate_sales_brief(
        input: QualityInput,
        *,
        policy: QualityPolicy | None = None,
    ) -> QualityReport

The ``input`` carries the structured sales-brief payload through
``input.context``. Recognised keys:

  - ``title`` (str)
  - ``headline`` (str): one-line punchy framing
  - ``sections`` (Sequence[Mapping]): each ``{"id", "title",
    "body_markdown", "claim_ids", "evidence_ids", ...}`` (mirrors
    ``report_pack``)
  - ``reference_ids`` (Sequence[str]): cited source ids at the
    brief level
  - ``metadata`` (Mapping): generation metadata; ``confidence`` may
    appear here as a float in [0, 1]

Recognised ``policy.thresholds`` keys:
  - ``min_sections`` (int): default 1
  - ``max_headline_chars`` (int): warn when ``headline`` is above this;
    default 280 (a reasonable elevator-line ceiling -- the rep should
    be able to skim it in 5 seconds before walking into the meeting).
    Note the deliberate asymmetry with the skill prompt, which tells
    the LLM to aim for ~140 chars (a tighter "punchy" target). The
    pack ceiling is the harder bound: the LLM can overshoot the
    aspirational prompt target by ~2x and still pass the gate, but a
    rambling 300-char "headline" warns. Two-tier guidance: aim short
    in the prompt, tolerate up to 280 in the gate.
  - ``min_confidence`` (float): warn when ``metadata["confidence"]`` is
    below this; default 0.0 (no floor)
  - ``pass_score`` (int): default 70
  - ``blocking_penalty`` (int): default 18 (per blocker)
  - ``warning_penalty`` (int): default 6 (per warning)

Recognised ``policy.metadata`` keys:
  - ``blocked_phrasing`` (Sequence[str] | str): word-boundary blocked
    phrases. Bare string is auto-wrapped (mirrors report_pack).
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


_DEFAULT_THRESHOLDS: Mapping[str, Any] = {
    "min_sections": 1,
    "max_headline_chars": 280,
    "min_confidence": 0.0,
    "pass_score": 70,
    "blocking_penalty": 18,
    "warning_penalty": 6,
}


def _threshold_int(policy: QualityPolicy | None, key: str) -> int:
    if policy is not None:
        value = policy.thresholds.get(key)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
    return int(_DEFAULT_THRESHOLDS[key])


def _threshold_float(policy: QualityPolicy | None, key: str) -> float:
    if policy is not None:
        value = policy.thresholds.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return float(_DEFAULT_THRESHOLDS[key])


def _blocked_phrases(policy: QualityPolicy | None) -> tuple[str, ...]:
    """Mirror of ``report_pack._blocked_phrases``: bare string auto-wraps."""
    if policy is None:
        return ()
    raw = policy.metadata.get("blocked_phrasing")
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    if not isinstance(raw, Sequence):
        return ()
    return tuple(str(item) for item in raw if str(item).strip())


def evaluate_sales_brief(
    input: QualityInput,
    *,
    policy: QualityPolicy | None = None,
) -> QualityReport:
    """Run the deterministic sales-brief-quality validators."""

    context = dict(input.context or {})
    title = str(context.get("title") or "").strip()
    headline = str(context.get("headline") or "").strip()
    sections_raw = context.get("sections") or ()
    sections: list[Mapping[str, Any]] = [
        section for section in sections_raw if isinstance(section, Mapping)
    ]
    reference_ids: tuple[str, ...] = tuple(
        str(item).strip()
        for item in (context.get("reference_ids") or ())
        if str(item).strip()
    )
    metadata: Mapping[str, Any] = (
        context.get("metadata") if isinstance(context.get("metadata"), Mapping) else {}
    )

    findings: list[GateFinding] = []

    # ---- Title / headline ----
    if not title:
        findings.append(GateFinding(code="no_title", message="no_title", severity=GateSeverity.BLOCKER))
    if not headline:
        findings.append(
            GateFinding(code="no_headline", message="no_headline", severity=GateSeverity.BLOCKER)
        )
    else:
        max_headline_chars = _threshold_int(policy, "max_headline_chars")
        if max_headline_chars > 0 and len(headline) > max_headline_chars:
            findings.append(
                GateFinding(
                    code="headline_too_long",
                    message=f"headline_too_long:{len(headline)}>{max_headline_chars}",
                    severity=GateSeverity.WARNING,
                    metadata={"length": len(headline), "max": max_headline_chars},
                )
            )

    # ---- Sections ----
    min_sections = _threshold_int(policy, "min_sections")
    if len(sections) < min_sections:
        findings.append(
            GateFinding(
                code="no_sections",
                message=f"no_sections:{len(sections)}_below_min_{min_sections}",
                severity=GateSeverity.BLOCKER,
                metadata={"section_count": len(sections), "min_sections": min_sections},
            )
        )

    section_evidence_present = False
    for index, section in enumerate(sections):
        section_title = str(section.get("title") or "").strip()
        if not section_title:
            findings.append(
                GateFinding(
                    code="section_missing_title",
                    message=f"section_missing_title:{index}",
                    severity=GateSeverity.BLOCKER,
                    metadata={"section_index": index},
                )
            )
        section_body = str(section.get("body_markdown") or "").strip()
        if not section_body:
            findings.append(
                GateFinding(
                    code="section_missing_body",
                    message=f"section_missing_body:{index}",
                    severity=GateSeverity.BLOCKER,
                    metadata={"section_index": index},
                )
            )
        evidence_ids = section.get("evidence_ids") or ()
        if isinstance(evidence_ids, Sequence) and not isinstance(evidence_ids, (str, bytes)):
            if any(str(item).strip() for item in evidence_ids):
                section_evidence_present = True

    # ---- References ----
    # A sales brief without source ids is useless to the rep -- they
    # need to know where the claims come from before walking into the
    # meeting. Mirrors the report_pack rule.
    if not reference_ids and not section_evidence_present:
        findings.append(
            GateFinding(
                code="no_references",
                message="no_references",
                severity=GateSeverity.BLOCKER,
            )
        )

    # ---- Confidence floor ----
    min_confidence = _threshold_float(policy, "min_confidence")
    raw_confidence = metadata.get("confidence")
    confidence: float | None = None
    if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
        confidence = max(0.0, min(1.0, float(raw_confidence)))
    if min_confidence > 0.0:
        if confidence is None:
            findings.append(
                GateFinding(
                    code="missing_confidence",
                    message="missing_confidence",
                    severity=GateSeverity.WARNING,
                )
            )
        elif confidence < min_confidence:
            findings.append(
                GateFinding(
                    code="confidence_below_min",
                    message=f"confidence_below_min:{confidence:.2f}<{min_confidence:.2f}",
                    severity=GateSeverity.WARNING,
                    metadata={"confidence": confidence, "min_confidence": min_confidence},
                )
            )

    # ---- Blocked phrasing (case-insensitive word-boundary) ----
    phrases = _blocked_phrases(policy)
    if phrases:
        haystack_parts: list[str] = []
        if title:
            haystack_parts.append(title)
        if headline:
            haystack_parts.append(headline)
        for section in sections:
            for key in ("title", "body_markdown"):
                value = section.get(key)
                if isinstance(value, str) and value.strip():
                    haystack_parts.append(value)
        haystack = "\n".join(haystack_parts)
        for phrase in phrases:
            phrase_str = str(phrase).strip()
            if not phrase_str:
                continue
            pattern = re.compile(rf"\b{re.escape(phrase_str)}\b", re.IGNORECASE)
            if pattern.search(haystack):
                findings.append(
                    GateFinding(
                        code="blocked_phrasing",
                        message=f"blocked_phrasing:{phrase}",
                        severity=GateSeverity.BLOCKER,
                        metadata={"phrase": phrase_str},
                    )
                )

    return _build_report(findings=findings, sections=sections, policy=policy)


def _build_report(
    *,
    findings: list[GateFinding],
    sections: list[Mapping[str, Any]],
    policy: QualityPolicy | None,
) -> QualityReport:
    blockers = [f for f in findings if f.severity == GateSeverity.BLOCKER]
    warnings = [f for f in findings if f.severity == GateSeverity.WARNING]
    blocking_penalty = _threshold_int(policy, "blocking_penalty")
    warning_penalty = _threshold_int(policy, "warning_penalty")
    pass_score = _threshold_int(policy, "pass_score")
    score = max(
        0,
        100 - (blocking_penalty * len(blockers)) - (warning_penalty * len(warnings)),
    )
    passed = (not blockers) and score >= pass_score
    if blockers or score < pass_score:
        decision = GateDecision.BLOCK
    elif warnings:
        decision = GateDecision.WARN
    else:
        decision = GateDecision.PASS
    metadata = {
        "score": score,
        "threshold": pass_score,
        "status": "pass" if passed else "fail",
        "blocking_issues": tuple(f.message for f in blockers),
        "blocking_codes": tuple(f.code for f in blockers),
        "warnings": tuple(f.message for f in warnings),
        "warning_codes": tuple(f.code for f in warnings),
        "section_count": len(sections),
    }
    return QualityReport(
        passed=passed,
        decision=decision,
        findings=tuple(findings),
        metadata=metadata,
    )


__all__ = ["evaluate_sales_brief"]
