"""Report quality pack: deterministic validators for structured reports.

The Reports content asset (PR-Reports-1) ships an output shape with
title, summary, ordered sections, and reference ids (see
``extracted_content_pipeline.report_ports.ReportDraft``). This pack
runs the deterministic checks that gate that output before persistence
and human approval.

Pure-function discipline matches the rest of the gate kernel: no DB,
no clock, no LLM, no network. Sanitization (markdown cleanup, etc.)
belongs in the wrapper, not here.

Public API:

    evaluate_report(
        input: QualityInput,
        *,
        policy: QualityPolicy | None = None,
    ) -> QualityReport

The ``input`` carries the structured report payload through
``input.context`` so the same ``QualityInput`` shape used by other
packs works unchanged. Recognised ``input.context`` keys:

  - ``title`` (str): report title
  - ``summary`` (str): executive summary
  - ``sections`` (Sequence[Mapping]): each section is
    ``{"id", "title", "body_markdown", "claim_ids", "evidence_ids"}``
  - ``reference_ids`` (Sequence[str]): cited source ids at the report
    level
  - ``metadata`` (Mapping): generation metadata; ``confidence`` may
    appear here as a float in [0, 1]

Recognised ``policy.thresholds`` keys (all optional, all have defaults):
  - ``min_confidence`` (float): warn when ``metadata["confidence"]`` is
    below this threshold; default 0.0 (no floor)
  - ``min_sections`` (int): blocker when section count is below this;
    default 1
  - ``pass_score`` (int): default 70
  - ``blocking_penalty`` (int): default 18 (per blocker)
  - ``warning_penalty`` (int): default 6 (per warning)

Recognised ``policy.metadata`` keys:
  - ``blocked_phrasing`` (Sequence[str]): substrings (matched
    case-insensitively at word boundaries) that disqualify the report
    if they appear in the title, summary, or any section's body. Same
    semantic as ``extracted_reasoning_core.api.validate_reasoning_output``.
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
    "min_confidence": 0.0,
    "min_sections": 1,
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
    if policy is None:
        return ()
    raw = policy.metadata.get("blocked_phrasing") or ()
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        return ()
    return tuple(str(item) for item in raw if str(item).strip())


def evaluate_report(
    input: QualityInput,
    *,
    policy: QualityPolicy | None = None,
) -> QualityReport:
    """Run the deterministic report-quality validators.

    Returns a :class:`QualityReport` with ``decision`` set to
    ``BLOCK`` when any blocker fires or score < pass_score,
    ``WARN`` when warnings fire but no blockers, otherwise ``PASS``.
    """

    context = dict(input.context or {})
    title = str(context.get("title") or "").strip()
    summary = str(context.get("summary") or "").strip()
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

    # ---- Title ----
    if not title:
        findings.append(
            GateFinding(
                code="no_title",
                message="no_title",
                severity=GateSeverity.BLOCKER,
            )
        )

    # ---- Summary ----
    if not summary:
        findings.append(
            GateFinding(
                code="no_summary",
                message="no_summary",
                severity=GateSeverity.BLOCKER,
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

    # ---- Blocked phrasing (case-insensitive word-boundary, mirrors
    # ---- extracted_reasoning_core.api.validate_reasoning_output) ----
    phrases = _blocked_phrases(policy)
    if phrases:
        haystack_parts: list[str] = []
        if title:
            haystack_parts.append(title)
        if summary:
            haystack_parts.append(summary)
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


__all__ = [
    "evaluate_report",
]
