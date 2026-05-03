"""Public API for the extracted reasoning core."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .types import (
    ArchetypeMatch,
    EvidenceDecision,
    EvidenceItem,
    EvidencePolicy,
    FalsificationPolicy,
    FalsificationResult,
    NarrativePlan,
    OutputPolicy,
    ReasoningDepth,
    ReasoningInput,
    ReasoningPack,
    ReasoningPorts,
    ReasoningResult,
    TemporalEvidence,
    ValidationReport,
)
from .state import ReasoningAgentState
from .tiers import (
    TIER_CONFIGS,
    Tier,
    TierConfig,
    build_tiered_pattern_sig,
    gather_tier_context,
    get_tier_config,
    needs_refresh,
)
from .wedge_registry import (
    WEDGE_ENUM_VALUES,
    Wedge,
    WedgeMeta,
    get_required_pools,
    get_sales_motion,
    get_wedge_meta,
    validate_wedge,
    wedge_from_archetype,
)


def score_archetypes(
    evidence: Mapping[str, Any],
    temporal: Mapping[str, Any] | None = None,
    *,
    limit: int = 3,
) -> Sequence[ArchetypeMatch]:
    """Score evidence against shared archetypes.

    The implementation lands when `archetypes.py` is consolidated into the
    core. The public name is reserved now so products can depend on the stable
    API shape instead of importing product-local forks.
    """
    del evidence
    del temporal
    del limit
    raise NotImplementedError("score_archetypes lands with archetype consolidation")


def evaluate_evidence(
    evidence: Mapping[str, Any],
    *,
    policy: EvidencePolicy | None = None,
) -> EvidenceDecision:
    """Evaluate evidence against a shared policy."""
    del evidence
    del policy
    raise NotImplementedError("evaluate_evidence lands with evidence consolidation")


def build_temporal_evidence(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    baselines: Mapping[str, Any] | None = None,
) -> TemporalEvidence:
    """Build normalized temporal evidence from snapshots."""
    del snapshots
    del baselines
    raise NotImplementedError("build_temporal_evidence lands with temporal consolidation")


def build_narrative_plan(
    context: Mapping[str, Any],
    *,
    pack: ReasoningPack,
) -> NarrativePlan:
    """Build a product-neutral narrative plan."""
    del context
    del pack
    raise NotImplementedError("build_narrative_plan lands with narrative consolidation")


async def run_reasoning(
    reasoning_input: ReasoningInput,
    *,
    depth: ReasoningDepth = "L2",
    pack: ReasoningPack | None = None,
    ports: ReasoningPorts | None = None,
) -> ReasoningResult:
    """Run the shared reasoning engine."""
    del reasoning_input
    del depth
    del pack
    del ports
    raise NotImplementedError("run_reasoning lands with graph/state consolidation")


async def continue_reasoning(
    state: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    ports: ReasoningPorts | None = None,
) -> ReasoningResult:
    """Continue a prior reasoning state with a new event."""
    del state
    del event
    del ports
    raise NotImplementedError("continue_reasoning lands with graph/state consolidation")


async def check_falsification(
    claim: Mapping[str, Any],
    fresh_evidence: Sequence[EvidenceItem],
    *,
    policy: FalsificationPolicy | None = None,
    ports: ReasoningPorts | None = None,
) -> FalsificationResult:
    """Check whether fresh evidence falsifies a prior claim."""
    del claim
    del fresh_evidence
    del policy
    del ports
    raise NotImplementedError("check_falsification lands with falsification consolidation")


def compute_evidence_hash(evidence: Mapping[str, Any]) -> str:
    """Compute the stable reasoning evidence hash."""
    del evidence
    raise NotImplementedError("compute_evidence_hash lands with semantic-cache split")


def build_semantic_cache_key(
    reasoning_input: ReasoningInput,
    *,
    tier: str,
    pack_name: str | None = None,
) -> str:
    """Build a stable semantic-cache key for a reasoning input."""
    del reasoning_input
    del tier
    del pack_name
    raise NotImplementedError("build_semantic_cache_key lands with semantic-cache split")


def load_reasoning_pack(name: str) -> ReasoningPack:
    """Load a named reasoning pack."""
    del name
    raise NotImplementedError("load_reasoning_pack lands with pack registry")


def validate_reasoning_output(
    result: ReasoningResult,
    *,
    policy: OutputPolicy | None = None,
) -> ValidationReport:
    """Validate a reasoned output against output policy."""
    del result
    del policy
    raise NotImplementedError("validate_reasoning_output lands with validation policy")


__all__ = [
    "WEDGE_ENUM_VALUES",
    "ArchetypeMatch",
    "EvidenceDecision",
    "EvidenceItem",
    "EvidencePolicy",
    "FalsificationPolicy",
    "FalsificationResult",
    "NarrativePlan",
    "OutputPolicy",
    "ReasoningDepth",
    "ReasoningAgentState",
    "ReasoningInput",
    "ReasoningPack",
    "ReasoningPorts",
    "ReasoningResult",
    "TemporalEvidence",
    "ValidationReport",
    "Wedge",
    "WedgeMeta",
    "TIER_CONFIGS",
    "Tier",
    "TierConfig",
    "build_narrative_plan",
    "build_semantic_cache_key",
    "build_temporal_evidence",
    "build_tiered_pattern_sig",
    "check_falsification",
    "compute_evidence_hash",
    "evaluate_evidence",
    "gather_tier_context",
    "get_required_pools",
    "get_sales_motion",
    "get_tier_config",
    "get_wedge_meta",
    "load_reasoning_pack",
    "needs_refresh",
    "run_reasoning",
    "score_archetypes",
    "continue_reasoning",
    "validate_reasoning_output",
    "validate_wedge",
    "wedge_from_archetype",
]
