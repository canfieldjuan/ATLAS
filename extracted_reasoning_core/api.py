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

    Returns the top `limit` matches as public `ArchetypeMatch` instances,
    sorted by score descending. Internal scoring runs against the
    canonical 10-archetype catalog from `extracted_reasoning_core.archetypes`;
    the rich internal `_ArchetypeMatchInternal` shape is converted to the
    public contract via the module's `_to_public_match` adapter so callers
    consume the stable `types.ArchetypeMatch` shape.

    `evidence` is the flat snapshot dict; `temporal` is an optional
    overlay (e.g., output of `TemporalEngine.to_evidence_dict`) that gets
    merged before scoring. `limit` defaults to 3 (matches the prior
    `top_matches` convention).
    """
    from . import archetypes as _archetypes

    matches = _archetypes.score_evidence(dict(evidence), dict(temporal) if temporal else None)
    capped = matches[: max(0, int(limit))]
    return tuple(_archetypes._to_public_match(m) for m in capped)


def evaluate_evidence(
    evidence: Mapping[str, Any],
    *,
    policy: EvidencePolicy | None = None,
) -> EvidenceDecision:
    """Evaluate evidence against a shared policy.

    Public helper that returns the simple "is this evidence allowed
    through" decision. Routes vendor evidence through the slim
    `EvidenceEngine` (PR-C1d) and reduces the per-conclusion outcomes
    to a single `EvidenceDecision`:

      * `allowed=False` if the engine returns the `insufficient_data`
        short-circuit OR if no conclusion fires (`met=True`); in that
        case the engine's fallback labels feed the `reasons` field.
      * `allowed=True` when at least one conclusion is met. `confidence`
        is the highest-tier confidence string the met conclusions
        carry, mapped through the optional `policy.confidence_labels`
        when provided.

    Callers that need the richer per-conclusion / per-suppression
    outputs should use `EvidenceEngine` directly via
    `extracted_reasoning_core.evidence_engine.get_evidence_engine`.
    """
    from . import evidence_engine as _ee

    engine = _ee.get_evidence_engine()
    results = engine.evaluate_conclusions(dict(evidence))

    # insufficient_data short-circuit -> single result with met=True,
    # confidence="insufficient". Translate to allowed=False.
    if (
        len(results) == 1
        and results[0].conclusion_id == "insufficient_data"
        and results[0].met
    ):
        reasons = tuple(
            x for x in (results[0].fallback_label,) if x
        )
        return EvidenceDecision(
            allowed=False,
            confidence=0.0,
            reasons=reasons or ("insufficient_data",),
        )

    met = [r for r in results if r.met]
    if not met:
        # No conclusion fired; surface fallback reasons from any
        # not-met results that carried fallback_label hints.
        reasons = tuple(r.fallback_label for r in results if r.fallback_label)
        return EvidenceDecision(
            allowed=False,
            confidence=0.0,
            reasons=reasons,
        )

    # Pick the strongest met-confidence string and translate to a
    # numeric score. Policy can override the mapping.
    rank = {"high": 0.9, "medium": 0.6, "low": 0.3, "insufficient": 0.0}
    if policy and policy.confidence_labels:
        rank = {**rank, **{k: float(v) for k, v in policy.confidence_labels.items()}}
    confidence = max(rank.get(r.confidence, 0.0) for r in met)
    if policy and confidence < policy.min_confidence:
        return EvidenceDecision(
            allowed=False,
            confidence=confidence,
            reasons=("confidence_below_policy_min",),
        )
    return EvidenceDecision(
        allowed=True,
        confidence=confidence,
        reasons=tuple(r.conclusion_id for r in met),
    )


def build_temporal_evidence(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    baselines: Mapping[str, Any] | None = None,
) -> TemporalEvidence:
    """Build normalized temporal evidence from already-loaded snapshots.

    Pure-function path: caller supplies a sorted (oldest-first) sequence of
    snapshot dicts and gets back the rich `TemporalEvidence` shape with
    velocities and long-term trends computed in-memory. No DB access; this
    is the in-process companion to `TemporalEngine.analyze_vendor` which
    handles DB-backed snapshots and category baselines.

    Velocities require >= 2 snapshots (`MIN_DAYS_FOR_VELOCITY`); long-term
    trends require >= 14 (`MIN_DAYS_FOR_TREND`). Below the velocity floor
    the function returns a `TemporalEvidence` with `insufficient_data=True`
    and the right `snapshot_days` count.

    `baselines` accepts a `Mapping` carrying optional category-percentile
    data. Atlas's full anomaly-vs-baseline pipeline (which requires
    DB-backed category lookups via `_compute_percentiles`) is intentionally
    out of scope here; callers needing it should use `TemporalEngine`
    directly. When `baselines` is `None` or empty, the returned
    `TemporalEvidence` has empty `anomalies` and `category_baselines`
    lists. A future PR can extend this entry point to honor a structured
    baselines payload without breaking callers.
    """
    from .temporal import (
        MIN_DAYS_FOR_TREND,
        MIN_DAYS_FOR_VELOCITY,
        TemporalEngine,
    )

    snaps = [dict(s) for s in snapshots]
    vendor_name = ""
    if snaps:
        vendor_name = str(snaps[-1].get("vendor_name") or "")

    if len(snaps) < MIN_DAYS_FOR_VELOCITY:
        return TemporalEvidence(
            vendor_name=vendor_name,
            snapshot_days=len(snaps),
            insufficient_data=True,
        )

    # `TemporalEngine` exposes the in-memory helpers we need; pool=None is
    # safe because we never call the DB-backed methods (`analyze_vendor`,
    # `_compute_percentiles`, `_infer_category`) from this entry point.
    engine = TemporalEngine(pool=None)
    velocities = engine._compute_velocities(vendor_name, snaps)
    trends = (
        engine._compute_long_term_trends(vendor_name, snaps)
        if len(snaps) >= MIN_DAYS_FOR_TREND
        else []
    )

    # `baselines` is an optional advisory input today; structured anomaly
    # support is a follow-up. The argument is accepted (and unpacked into a
    # local) so the public signature is honored even though the value is
    # not yet consumed.
    _baselines = dict(baselines) if baselines else {}
    del _baselines

    return TemporalEvidence(
        vendor_name=vendor_name,
        snapshot_days=len(snaps),
        velocities=velocities,
        trends=trends,
    )


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
