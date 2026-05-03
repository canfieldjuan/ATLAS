"""Deterministic safety-gate scanner: content patterns + risk scoring.

Owned by ``extracted_quality_gate`` (PR-B3). The two public entry
points -- ``check_content`` and ``assess_risk`` -- are pure: no DB,
no network, no clock. They take inputs, return frozen dataclasses,
and are safe to call from any context (sync or async).

The Atlas-side ``SafetyGate`` wrapper layers approvals, audit logging,
and DB persistence on top of these primitives via the ``ApprovalStore``
and ``AuditLog`` ports defined in :mod:`extracted_quality_gate.ports`.

Why deterministic-first: a content/risk scan that does not touch the
database (a) can be unit-tested without fixtures, (b) can run in a
worker that has no DB connection (e.g. a pre-flight check at the edge),
and (c) decouples policy evolution from persistence schema evolution.

Pattern catalogue (``_PROHIBITED_PATTERNS``) is a stable, additive
list. Adding a label is non-breaking; renaming a label IS a breaking
change because operator dashboards key off the label string.
"""

from __future__ import annotations

import re
from typing import Mapping

from .types import (
    ContentFlag,
    ContentScanResult,
    RiskAssessment,
    RiskLevel,
)


# Patterns that must never appear in intervention output.
# Catches deceptive, coercive, or identity-misrepresenting content.
# (regex, label) -- label is a stable string used in audit logs and
# operator dashboards; renaming a label is a breaking change.
_PROHIBITED_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bimpersonat(?:e|ing|ion)\b", "impersonation"),
    (r"\bfabricat(?:e|ed|ing)\s+(?:facts?|evidence|data)", "fabricated_facts"),
    (r"\bblackmail\b", "blackmail"),
    (r"\bextort(?:ion|ing)?\b", "extortion"),
    (r"\bthreaten(?:s|ed|ing)?\s+(?:to\s+)?(?:harm|violence|physical)", "threat_of_harm"),
    (r"\bmanipulat(?:e|ing)\s+(?:evidence|records|data)", "evidence_manipulation"),
    (r"\bdoxx?(?:ing|ed)?\b", "doxxing"),
    (r"\bphishing\b", "phishing"),
    (r"\bsocial\s+engineer(?:ing)?\b", "social_engineering"),
)

# Compiled once at module load. Kept private so callers cannot mutate
# the registry between scans.
_COMPILED_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), label)
    for pat, label in _PROHIBITED_PATTERNS
)


_RISK_ORDER: Mapping[RiskLevel, int] = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


def check_content(text: str) -> ContentScanResult:
    """Scan ``text`` for prohibited patterns.

    Empty / falsy input passes by definition. The scan walks every
    compiled pattern over the full text, so a single input can produce
    multiple flags (e.g. a paragraph that contains both "phishing" and
    "social engineering").
    """
    if not text:
        return ContentScanResult(passed=True, blocked=False, flags=())

    flags: list[ContentFlag] = []
    for compiled, label in _COMPILED_PATTERNS:
        for match in compiled.finditer(text):
            flags.append(
                ContentFlag(
                    pattern=label,
                    match=match.group(),
                    position=match.start(),
                )
            )

    blocked = bool(flags)
    return ContentScanResult(
        passed=not blocked,
        blocked=blocked,
        flags=tuple(flags),
    )


def assess_risk(
    sensor_summary: Mapping[str, object] | None,
    pressure: Mapping[str, object] | None,
    content_check: ContentScanResult | None = None,
    *,
    auto_approve_max_risk: RiskLevel = RiskLevel.MEDIUM,
) -> RiskAssessment:
    """Compose a single ``RiskAssessment`` from upstream signals.

    Inputs are loosely-typed mappings so callers do not have to convert
    Atlas-shaped dicts into core types -- the scanner reads the
    documented keys and ignores the rest.

    Recognized keys:
      ``sensor_summary["dominant_risk_level"]`` (str: LOW/MEDIUM/HIGH/CRITICAL)
      ``pressure["pressure_score"]`` (numeric, 0-10 scale)
      ``content_check`` (a :class:`ContentScanResult`; ``blocked=True``
        adds 3 to the risk score and contributes a ``Content flags:``
        factor with the underlying labels)

    Scoring is intentionally simple and stable -- a tuned ML model
    has no place in the deterministic core. Tune by editing constants.
    """
    factors: list[str] = []
    risk_score = 0

    # Sensor-based risk
    sensor_level_raw = (sensor_summary or {}).get("dominant_risk_level", "LOW")
    try:
        sensor_level = RiskLevel(str(sensor_level_raw))
    except ValueError:
        sensor_level = RiskLevel.LOW
    risk_score += _RISK_ORDER[sensor_level]
    if sensor_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        factors.append(f"Sensor composite: {sensor_level.value}")

    # Pressure-based risk
    pressure_score = (pressure or {}).get("pressure_score", 0)
    if isinstance(pressure_score, (int, float)):
        if pressure_score >= 8:
            risk_score += 2
            factors.append(f"Critical pressure: {pressure_score}/10")
        elif pressure_score >= 6:
            risk_score += 1
            factors.append(f"Elevated pressure: {pressure_score}/10")

    # Content-filter risk
    if content_check is not None and content_check.blocked:
        risk_score += 3
        flag_labels = [flag.pattern for flag in content_check.flags]
        factors.append(f"Content flags: {', '.join(flag_labels)}")

    # Map score -> level
    if risk_score >= 4:
        level = RiskLevel.CRITICAL
    elif risk_score >= 3:
        level = RiskLevel.HIGH
    elif risk_score >= 1:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    auto_eligible = _RISK_ORDER[level] <= _RISK_ORDER[auto_approve_max_risk]

    return RiskAssessment(
        risk_level=level,
        risk_score=risk_score,
        auto_approve_eligible=auto_eligible,
        factors=tuple(factors),
    )


__all__ = [
    "assess_risk",
    "check_content",
]
