"""Public dataclasses for the extracted reasoning core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from .ports import Clock, EventSink, LLMClient, ReasoningStateStore, SemanticCacheStore, TraceSink
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
    event_sink: EventSink | None = None
    trace_sink: TraceSink | None = None


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


@dataclass(frozen=True, slots=True)
class ConclusionResult:
    """Outcome of a single per-vendor conclusion rule evaluation.

    Mirrors the shape `EvidenceEngine.evaluate_conclusion` returns in
    atlas. Public so products that need the richer per-rule outcome
    (vs the simpler `EvidenceDecision`) can consume it from the
    canonical location.
    """

    conclusion_id: str
    met: bool
    confidence: str
    fallback_label: str | None = None
    fallback_action: str | None = None


@dataclass(frozen=True, slots=True)
class SuppressionResult:
    """Per-section render-suppression decision.

    Mirrors the shape `EvidenceEngine.evaluate_suppression` returns in
    atlas. Public so report renderers can consume the same outcome
    types from the canonical location.
    """

    suppress: bool = False
    degrade: bool = False
    disclaimer: str | None = None
    fallback_label: str | None = None


@dataclass(frozen=True)
class VendorVelocity:
    """Rate-of-change metrics for a vendor metric over a snapshot window."""

    vendor_name: str
    metric: str
    current_value: float
    previous_value: float
    velocity: float          # change per day
    days_between: int
    acceleration: float | None = None  # change in velocity (needs 3+ points)


@dataclass(frozen=True)
class LongTermTrend:
    """Long-term linear trend (30-day / 90-day slopes)."""

    metric: str
    slope_30d: float | None = None
    slope_90d: float | None = None
    volatility: float | None = None  # standard deviation of changes
    data_points: int = 0


@dataclass(frozen=True)
class CategoryPercentile:
    """Rolling percentile baselines for a metric within a product category."""

    product_category: str
    metric: str
    p25: float
    p50: float
    p75: float
    sample_count: int


@dataclass(frozen=True)
class AnomalyScore:
    """Z-score anomaly detection for a single vendor metric."""

    vendor_name: str
    metric: str
    value: float
    z_score: float
    p_value: float | None = None
    is_anomaly: bool = False    # |z| > 2.0


@dataclass(frozen=True)
class TemporalEvidence:
    """Complete temporal analysis for a vendor.

    Promoted from PR #79's coarse `Mapping[str, Any]` placeholder to
    the rich shape per the PR #82 audit's contract amendment. The 4
    sub-types (`VendorVelocity`, `LongTermTrend`, `CategoryPercentile`,
    `AnomalyScore`) are public so products can render velocity / trend
    charts without losing type safety at the boundary.
    """

    vendor_name: str
    snapshot_days: int
    velocities: list[VendorVelocity] = field(default_factory=list)
    trends: list[LongTermTrend] = field(default_factory=list)
    anomalies: list[AnomalyScore] = field(default_factory=list)
    category_baselines: list[CategoryPercentile] = field(default_factory=list)
    insufficient_data: bool = False


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
    "AnomalyScore",
    "ArchetypeMatch",
    "CategoryPercentile",
    "ConclusionResult",
    "EvidenceDecision",
    "EvidenceItem",
    "EvidencePolicy",
    "FalsificationPolicy",
    "FalsificationResult",
    "LongTermTrend",
    "NarrativePlan",
    "OutputPolicy",
    "ReasoningDepth",
    "ReasoningInput",
    "ReasoningPack",
    "ReasoningPorts",
    "ReasoningResult",
    "SuppressionResult",
    "TemporalEvidence",
    "ValidationReport",
    "VendorVelocity",
    "Wedge",
    "WedgeMeta",
]
