"""Public dataclasses for the extracted reasoning core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from .ports import Clock, LLMClient, ReasoningStateStore, SemanticCacheStore
from .wedge_registry import Wedge, WedgeMeta


ReasoningDepth = Literal["L1", "L2", "L3", "L4", "L5"]


@dataclass(frozen=True)
class EvidenceItem:
    source_type: str
    source_id: str
    text: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningInput:
    entity_id: str
    entity_type: str
    goal: str
    evidence: Sequence[EvidenceItem]
    context: Mapping[str, Any] = field(default_factory=dict)
    pack_name: str | None = None


@dataclass(frozen=True)
class ReasoningResult:
    summary: str
    claims: Sequence[Mapping[str, Any]]
    confidence: float
    tier: ReasoningDepth
    state: Mapping[str, Any]
    trace: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningPorts:
    llm: LLMClient | None = None
    semantic_cache: SemanticCacheStore | None = None
    state_store: ReasoningStateStore | None = None
    clock: Clock | None = None


@dataclass(frozen=True)
class ArchetypeMatch:
    archetype_id: str
    label: str
    score: float
    evidence_hits: Sequence[str] = field(default_factory=tuple)
    missing_evidence: Sequence[str] = field(default_factory=tuple)
    risk_label: str = ""


@dataclass(frozen=True)
class EvidencePolicy:
    required_pools: Sequence[str] = field(default_factory=tuple)
    min_confidence: float = 0.0
    confidence_labels: Mapping[str, float] = field(default_factory=dict)
    suppression_rules: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceDecision:
    allowed: bool
    confidence: float = 0.0
    reasons: Sequence[str] = field(default_factory=tuple)
    missing_evidence: Sequence[str] = field(default_factory=tuple)
    trace: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TemporalEvidence:
    velocity: Mapping[str, Any] = field(default_factory=dict)
    trend: Mapping[str, Any] = field(default_factory=dict)
    anomaly: Mapping[str, Any] = field(default_factory=dict)
    recency: Mapping[str, Any] = field(default_factory=dict)
    baselines: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NarrativePlan:
    claims: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    sections: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    evidence_requirements: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    state_hints: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningPack:
    name: str
    version: str = "v1"
    prompts: Mapping[str, str] = field(default_factory=dict)
    policies: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FalsificationPolicy:
    rules: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    conservative: bool = True


@dataclass(frozen=True)
class FalsificationResult:
    triggered_conditions: Sequence[str] = field(default_factory=tuple)
    non_triggered_conditions: Sequence[str] = field(default_factory=tuple)
    should_invalidate: bool = False
    trace: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputPolicy:
    required_claim_types: Sequence[str] = field(default_factory=tuple)
    require_citations: bool = True
    min_confidence: float = 0.0
    blocked_phrasing: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    blockers: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)
    repaired_fields: Mapping[str, Any] = field(default_factory=dict)
    trace: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "ArchetypeMatch",
    "EvidenceDecision",
    "EvidenceItem",
    "EvidencePolicy",
    "FalsificationPolicy",
    "FalsificationResult",
    "NarrativePlan",
    "OutputPolicy",
    "ReasoningDepth",
    "ReasoningInput",
    "ReasoningPack",
    "ReasoningPorts",
    "ReasoningResult",
    "TemporalEvidence",
    "ValidationReport",
    "Wedge",
    "WedgeMeta",
]
