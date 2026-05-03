"""Public quality-gate data types.

These types are intentionally generic. Product-specific packs can map
their own rows into these reports without importing Atlas or another
product's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class GateSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


class GateDecision(StrEnum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    APPROVAL_REQUIRED = "approval_required"


@dataclass(frozen=True)
class GateFinding:
    code: str
    message: str
    severity: GateSeverity
    field_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityReport:
    passed: bool
    decision: GateDecision
    findings: tuple[GateFinding, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def blockers(self) -> tuple[GateFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if finding.severity == GateSeverity.BLOCKER
        )

    @property
    def warnings(self) -> tuple[GateFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if finding.severity == GateSeverity.WARNING
        )


@dataclass(frozen=True)
class QualityPolicy:
    name: str
    version: str = "v1"
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityInput:
    artifact_type: str
    artifact_id: str | None = None
    content: str | None = None
    evidence: tuple[Mapping[str, Any], ...] = ()
    claims: tuple[Mapping[str, Any], ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "GateDecision",
    "GateFinding",
    "GateSeverity",
    "QualityInput",
    "QualityPolicy",
    "QualityReport",
]
