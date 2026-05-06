"""Public API for the extracted reasoning core."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from ._synthesis import (
    SynthesisLoopResult,
    clean_reasoning_text,
    evidence_to_mapping,
    extract_claims,
    extract_confidence,
    extract_summary,
    invoke_synthesis_loop,
    parse_llm_json,
    response_text,
    synthesis_config_from_pack,
)
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


class ConfigurationError(RuntimeError):
    """Raised when reasoning core is missing a required host port."""


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
        reasons = tuple(r.fallback_label for r in results if r.fallback_label)
        return EvidenceDecision(
            allowed=False,
            confidence=0.0,
            reasons=reasons,
        )

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

    engine = TemporalEngine(pool=None)
    velocities = engine._compute_velocities(vendor_name, snaps)
    trends = (
        engine._compute_long_term_trends(vendor_name, snaps)
        if len(snaps) >= MIN_DAYS_FOR_TREND
        else []
    )

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
    """Build a product-neutral narrative plan from a reasoning context.

    Pure deterministic transformation: take a reasoning context (either
    a ``ReasoningResult.state`` dict or a raw synthesis JSON), apply
    pack policies, and produce an ordered ``NarrativePlan`` with
    sections, evidence requirements, and state hints. No LLM call.

    Pack policies honored (all optional):
      * ``min_confidence`` (default 0.0): drop claims below this score.
      * ``max_sections`` (default 10): cap the number of sections.
      * ``claim_ordering`` (default "by_confidence"): "by_confidence"
        sorts claims descending by confidence; any other value
        (including "preserve_input") keeps the order from the source.
      * ``default_section`` (default "main"): section name used when a
        claim doesn't carry a ``section`` field.

    ``state_hints`` includes ``dropped_below_confidence`` (claims removed
    by ``min_confidence``) and ``dropped_due_to_section_cap`` (claims
    removed because their section exceeded ``max_sections``) so callers
    can distinguish missing data from rendering choices.
    """

    raw_synthesis = context.get("raw_synthesis") if isinstance(context.get("raw_synthesis"), Mapping) else None
    source = raw_synthesis if raw_synthesis else context

    raw_claims = extract_claims(source)
    summary = extract_summary(source)
    confidence = extract_confidence(source, raw_claims)

    policies = dict(pack.policies or {})
    min_confidence = float(policies.get("min_confidence", 0.0))
    max_sections = max(1, int(policies.get("max_sections", 10)))
    claim_ordering = str(policies.get("claim_ordering", "by_confidence")).lower()
    default_section = str(policies.get("default_section", "main"))

    # Filter and order claims.
    filtered: list[Mapping[str, Any]] = []
    dropped = 0
    for claim in raw_claims:
        claim_conf = _claim_confidence(claim)
        if claim_conf < min_confidence:
            dropped += 1
            continue
        filtered.append(claim)
    if claim_ordering == "by_confidence":
        filtered.sort(key=_claim_confidence, reverse=True)

    # Group by section.
    section_order: list[str] = []
    section_claims: dict[str, list[Mapping[str, Any]]] = {}
    for claim in filtered:
        section_name = str(claim.get("section") or default_section)
        if section_name not in section_claims:
            section_claims[section_name] = []
            section_order.append(section_name)
        section_claims[section_name].append(claim)

    if len(section_order) > max_sections:
        capped_section_names = section_order[max_sections:]
        dropped_due_to_section_cap = sum(
            len(section_claims[s]) for s in capped_section_names
        )
        section_order = section_order[:max_sections]
    else:
        dropped_due_to_section_cap = 0

    sections: list[Mapping[str, Any]] = []
    evidence_requirements: list[Mapping[str, Any]] = []
    plan_claims: list[Mapping[str, Any]] = []

    for section_name in section_order:
        claims_in_section = tuple(section_claims[section_name])
        plan_claims.extend(claims_in_section)
        section_source_ids = _collect_source_ids(claims_in_section)
        sections.append({
            "id": section_name,
            "title": section_name.replace("_", " ").strip().title() or section_name,
            "claim_count": len(claims_in_section),
            "claim_ids": tuple(_claim_id(c) for c in claims_in_section),
        })
        evidence_requirements.append({
            "section_id": section_name,
            "cited_source_ids": section_source_ids,
            "claim_count": len(claims_in_section),
        })

    return NarrativePlan(
        claims=tuple(plan_claims),
        sections=tuple(sections),
        evidence_requirements=tuple(evidence_requirements),
        state_hints={
            "summary": summary,
            "overall_confidence": confidence,
            "claim_count": len(plan_claims),
            "section_count": len(sections),
            "dropped_below_confidence": dropped,
            "dropped_due_to_section_cap": dropped_due_to_section_cap,
            "pack": pack.name,
            "pack_version": pack.version,
            "depth": context.get("depth") or context.get("tier") or "",
        },
    )


def _claim_confidence(claim: Mapping[str, Any]) -> float:
    raw = claim.get("confidence")
    if raw is None:
        return 0.0
    rank = {"high": 0.9, "medium": 0.6, "low": 0.3, "insufficient": 0.0}
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return rank.get(str(raw).strip().lower(), 0.0)


def _claim_id(claim: Mapping[str, Any]) -> str:
    return str(claim.get("claim_id") or claim.get("id") or claim.get("claim") or "")[:200]


def _collect_source_ids(claims: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    seen: list[str] = []
    for claim in claims:
        for sid in claim.get("source_ids") or ():
            sid_str = str(sid)
            if sid_str and sid_str not in seen:
                seen.append(sid_str)
    return tuple(seen)


async def run_reasoning(
    reasoning_input: ReasoningInput,
    *,
    depth: ReasoningDepth = "L2",
    pack: ReasoningPack | None = None,
    ports: ReasoningPorts | None = None,
) -> ReasoningResult:
    """Run the single-pass reasoning synthesis flow.

    Calls the host LLM port once, validates the JSON response against the
    summary/claims contract, and retries with compact validation feedback
    when validation fails. Host-owned witness compression stays outside
    this package and is supplied either through
    ``ReasoningPorts.witness_context`` or via ``reasoning_input.context``.
    """

    port_bundle = ports or ReasoningPorts()
    if port_bundle.llm is None:
        raise ConfigurationError(
            "run_reasoning requires ReasoningPorts.llm; provide an "
            "extracted_llm_infrastructure-backed LLM port."
        )

    effective_pack = pack or ReasoningPack(
        name=reasoning_input.pack_name or "reasoning_synthesis",
        prompts={},
        policies={},
    )
    config = synthesis_config_from_pack(effective_pack)

    witness_context = await _resolve_witness_context(
        reasoning_input,
        depth=depth,
        pack=effective_pack,
        ports=port_bundle,
    )
    system_prompt = _resolve_reasoning_prompt(effective_pack)
    payload = _build_reasoning_payload(
        reasoning_input,
        depth=depth,
        pack=effective_pack,
        witness_context=witness_context,
    )
    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    trace_sink = port_bundle.trace_sink
    span = None
    if trace_sink is not None:
        span = trace_sink.start_span(
            "extracted_reasoning_core.run_reasoning",
            metadata={
                "entity_id": reasoning_input.entity_id,
                "entity_type": reasoning_input.entity_type,
                "depth": depth,
                "pack": effective_pack.name,
            },
        )

    try:
        loop = await invoke_synthesis_loop(
            system_prompt=system_prompt,
            payload_text=payload_text,
            llm_metadata={
                "entity_id": reasoning_input.entity_id,
                "entity_type": reasoning_input.entity_type,
                "goal": reasoning_input.goal,
                "depth": depth,
                "pack": effective_pack.name,
                "reasoning_mode": "single_pass_synthesis",
            },
            config=config,
            llm_port=port_bundle.llm,
        )

        if loop.succeeded:
            result = _synthesis_success_result(
                loop=loop,
                reasoning_input=reasoning_input,
                depth=depth,
                pack=effective_pack,
                witness_context=witness_context,
            )
            if port_bundle.event_sink is not None:
                await port_bundle.event_sink.emit(
                    "reasoning.synthesis.completed",
                    "extracted_reasoning_core.run_reasoning",
                    {"attempts_used": len(loop.attempts), "tokens_used": loop.total_tokens},
                    entity_type=reasoning_input.entity_type,
                    entity_id=reasoning_input.entity_id,
                )
            if trace_sink is not None and span is not None:
                trace_sink.end_span(
                    span,
                    status="ok",
                    metadata={
                        "attempts_used": len(loop.attempts),
                        "tokens_used": loop.total_tokens,
                    },
                )
            return result

        if port_bundle.event_sink is not None:
            await port_bundle.event_sink.emit(
                "reasoning.synthesis.validation_failed",
                "extracted_reasoning_core.run_reasoning",
                {
                    "attempts_used": config.max_attempts,
                    "tokens_used": loop.total_tokens,
                    "errors": loop.failure_reasons,
                },
                entity_type=reasoning_input.entity_type,
                entity_id=reasoning_input.entity_id,
            )
        if trace_sink is not None and span is not None:
            trace_sink.end_span(
                span,
                status="error",
                metadata={
                    "attempts_used": config.max_attempts,
                    "error_text": loop.error_text,
                },
            )
        return _synthesis_failure_result(
            loop=loop,
            depth=depth,
            max_attempts=config.max_attempts,
            witness_context=witness_context,
        )
    except Exception:
        if trace_sink is not None and span is not None:
            trace_sink.end_span(span, status="error", metadata={"stage": "exception"})
        raise


async def continue_reasoning(
    state: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    ports: ReasoningPorts | None = None,
) -> ReasoningResult:
    """Continue a prior reasoning state with a new event.

    Multi-pass refinement: given a prior ``ReasoningResult.state`` and a
    new event (with new evidence), produce an updated ``ReasoningResult``
    that supersedes the prior conclusions. Lineage (generation count,
    prior synthesis hash, events consumed) is tracked in the returned
    state so callers can chain continuations.

    The function does NOT run a falsification pass; that is reserved for
    ``check_falsification``. If the new event carries a
    ``falsification_hint``, it is surfaced in the trace but not acted on
    separately.
    """

    port_bundle = ports or ReasoningPorts()
    if port_bundle.llm is None:
        raise ConfigurationError(
            "continue_reasoning requires ReasoningPorts.llm; provide an "
            "extracted_llm_infrastructure-backed LLM port."
        )

    prior_status = str(state.get("status") or "")
    if prior_status != "completed":
        return _continuation_invalid_state_result(state=state, event=event)

    raw_synthesis = dict(state.get("raw_synthesis") or {})
    prior_summary = extract_summary(raw_synthesis)
    prior_claims = extract_claims(raw_synthesis)
    prior_confidence = extract_confidence(raw_synthesis, prior_claims)
    prior_synthesis_hash = _hash_prior_synthesis(raw_synthesis)
    prior_generation = int(state.get("generation") or 1)

    entity_id = str(state.get("entity_id") or "")
    entity_type = str(state.get("entity_type") or "")
    goal = str(state.get("goal") or "")
    depth_value: ReasoningDepth = state.get("depth") or state.get("tier") or "L2"  # type: ignore[assignment]
    pack_name = str(state.get("pack") or "reasoning_synthesis")
    pack_version = str(state.get("pack_version") or "1.0")
    pack = ReasoningPack(
        name=pack_name,
        version=pack_version,
        prompts=dict(state.get("pack_prompts") or {}),
        policies=dict(state.get("pack_policies") or {}),
    )

    event_evidence_raw = event.get("evidence") or ()
    event_evidence = tuple(_coerce_evidence_item(item) for item in event_evidence_raw)
    reasoning_input = ReasoningInput(
        entity_id=entity_id,
        entity_type=entity_type,
        goal=goal,
        evidence=event_evidence,
    )

    witness_context = await _resolve_witness_context(
        reasoning_input,
        depth=depth_value,
        pack=pack,
        ports=port_bundle,
    )
    config = synthesis_config_from_pack(pack)
    system_prompt = _resolve_continuation_prompt(pack)
    payload = _build_continuation_payload(
        prior_summary=prior_summary,
        prior_claims=prior_claims,
        prior_confidence=prior_confidence,
        event=event,
        witness_context=witness_context,
        entity_id=entity_id,
        entity_type=entity_type,
        goal=goal,
        depth=depth_value,
        pack=pack,
    )
    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    falsification_hint_seen = bool(event.get("falsification_hint"))
    event_type = str(event.get("event_type") or "")

    trace_sink = port_bundle.trace_sink
    span = None
    if trace_sink is not None:
        span = trace_sink.start_span(
            "extracted_reasoning_core.continue_reasoning",
            metadata={
                "entity_id": entity_id,
                "entity_type": entity_type,
                "depth": depth_value,
                "pack": pack_name,
                "event_type": event_type,
                "prior_generation": prior_generation,
            },
        )

    try:
        loop = await invoke_synthesis_loop(
            system_prompt=system_prompt,
            payload_text=payload_text,
            llm_metadata={
                "entity_id": entity_id,
                "entity_type": entity_type,
                "goal": goal,
                "depth": depth_value,
                "pack": pack_name,
                "event_type": event_type,
                "reasoning_mode": "multi_pass_continuation",
            },
            config=config,
            llm_port=port_bundle.llm,
        )

        if loop.succeeded:
            result = _continuation_success_result(
                loop=loop,
                entity_id=entity_id,
                entity_type=entity_type,
                goal=goal,
                pack_name=pack_name,
                pack_version=pack_version,
                pack_policies=pack.policies,
                pack_prompts=pack.prompts,
                depth=depth_value,
                generation=prior_generation + 1,
                prior_synthesis_hash=prior_synthesis_hash,
                prior_summary=prior_summary,
                prior_attempts_used=int(state.get("attempts_used") or 0),
                event_type=event_type,
                falsification_hint_seen=falsification_hint_seen,
                witness_context=witness_context,
            )
            if port_bundle.event_sink is not None:
                await port_bundle.event_sink.emit(
                    "reasoning.continuation.completed",
                    "extracted_reasoning_core.continue_reasoning",
                    {
                        "attempts_used": len(loop.attempts),
                        "tokens_used": loop.total_tokens,
                        "generation": prior_generation + 1,
                    },
                    entity_type=entity_type,
                    entity_id=entity_id,
                )
            if trace_sink is not None and span is not None:
                trace_sink.end_span(
                    span,
                    status="ok",
                    metadata={
                        "attempts_used": len(loop.attempts),
                        "tokens_used": loop.total_tokens,
                        "generation": prior_generation + 1,
                    },
                )
            return result

        if port_bundle.event_sink is not None:
            await port_bundle.event_sink.emit(
                "reasoning.continuation.validation_failed",
                "extracted_reasoning_core.continue_reasoning",
                {
                    "attempts_used": config.max_attempts,
                    "tokens_used": loop.total_tokens,
                    "errors": loop.failure_reasons,
                    "generation": prior_generation + 1,
                },
                entity_type=entity_type,
                entity_id=entity_id,
            )
        if trace_sink is not None and span is not None:
            trace_sink.end_span(
                span,
                status="error",
                metadata={
                    "attempts_used": config.max_attempts,
                    "error_text": loop.error_text,
                },
            )
        return _continuation_failure_result(
            loop=loop,
            depth=depth_value,
            max_attempts=config.max_attempts,
            prior_summary=prior_summary,
            prior_synthesis_hash=prior_synthesis_hash,
            generation=prior_generation + 1,
            event_type=event_type,
            falsification_hint_seen=falsification_hint_seen,
            witness_context=witness_context,
        )
    except Exception:
        if trace_sink is not None and span is not None:
            trace_sink.end_span(span, status="error", metadata={"stage": "exception"})
        raise


async def _resolve_witness_context(
    reasoning_input: ReasoningInput,
    *,
    depth: ReasoningDepth,
    pack: ReasoningPack,
    ports: ReasoningPorts,
) -> Mapping[str, Any]:
    if ports.witness_context is not None:
        return dict(await ports.witness_context.get_witness_context(
            reasoning_input,
            depth=depth,
            pack=pack,
        ))
    context = dict(reasoning_input.context or {})
    for key in ("witness_context", "compressed_witness_context", "witness_pack"):
        value = context.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _resolve_reasoning_prompt(pack: ReasoningPack) -> str:
    for key in ("reasoning_synthesis", "system", "prompt"):
        prompt = str((pack.prompts or {}).get(key) or "").strip()
        if prompt:
            return prompt
    from .skills.registry import get_skill_registry

    registry = get_skill_registry()
    return registry.get_prompt("digest/reasoning_synthesis") or (
        "Return a valid JSON reasoning synthesis with summary, claims, and confidence."
    )


def _resolve_continuation_prompt(pack: ReasoningPack) -> str:
    for key in ("reasoning_continuation", "continuation", "system_continuation"):
        prompt = str((pack.prompts or {}).get(key) or "").strip()
        if prompt:
            return prompt
    from .skills.registry import get_skill_registry

    registry = get_skill_registry()
    return registry.get_prompt("digest/reasoning_continuation") or (
        "Return a complete refined JSON reasoning synthesis incorporating "
        "the new event, with summary, claims (each with revised_from), and "
        "confidence."
    )


def _build_reasoning_payload(
    reasoning_input: ReasoningInput,
    *,
    depth: ReasoningDepth,
    pack: ReasoningPack,
    witness_context: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "entity_id": reasoning_input.entity_id,
        "entity_type": reasoning_input.entity_type,
        "goal": reasoning_input.goal,
        "depth": depth,
        "pack": {"name": pack.name, "version": pack.version},
        "evidence": tuple(evidence_to_mapping(item) for item in reasoning_input.evidence),
        "context": dict(reasoning_input.context or {}),
        "witness_context": dict(witness_context or {}),
    }


def _build_continuation_payload(
    *,
    prior_summary: str,
    prior_claims: Sequence[Mapping[str, Any]],
    prior_confidence: float,
    event: Mapping[str, Any],
    witness_context: Mapping[str, Any],
    entity_id: str,
    entity_type: str,
    goal: str,
    depth: ReasoningDepth,
    pack: ReasoningPack,
) -> Mapping[str, Any]:
    event_evidence = tuple(
        evidence_to_mapping(item) for item in (event.get("evidence") or ())
    )
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "goal": goal,
        "depth": depth,
        "pack": {"name": pack.name, "version": pack.version},
        "prior_summary": prior_summary,
        "prior_claims": tuple(dict(c) for c in prior_claims),
        "prior_confidence": prior_confidence,
        "event": {
            "event_type": str(event.get("event_type") or ""),
            "timestamp": event.get("timestamp"),
            "source_id": event.get("source_id"),
            "metadata": dict(event.get("metadata") or {}),
            "evidence": event_evidence,
            "falsification_hint": bool(event.get("falsification_hint")),
        },
        "witness_context": dict(witness_context or {}),
    }


def _synthesis_success_result(
    *,
    loop: SynthesisLoopResult,
    reasoning_input: ReasoningInput,
    depth: ReasoningDepth,
    pack: ReasoningPack,
    witness_context: Mapping[str, Any],
) -> ReasoningResult:
    candidate = loop.valid_candidate or {}
    claims = extract_claims(candidate)
    confidence = extract_confidence(candidate, claims)
    return ReasoningResult(
        summary=extract_summary(candidate),
        claims=claims,
        confidence=confidence,
        tier=depth,
        state={
            "status": "completed",
            "succeeded": True,
            "entity_id": reasoning_input.entity_id,
            "entity_type": reasoning_input.entity_type,
            "goal": reasoning_input.goal,
            "pack": pack.name,
            "pack_version": pack.version,
            "pack_policies": dict(pack.policies or {}),
            "pack_prompts": dict(pack.prompts or {}),
            "attempts_used": len(loop.attempts),
            "tokens_used": loop.total_tokens,
            "depth": depth,
            "raw_synthesis": dict(candidate),
        },
        trace={
            "attempts": loop.attempts,
            "witness_context_present": bool(witness_context),
            "validation_warnings": tuple(
                warning
                for attempt in loop.attempts
                for warning in attempt.get("warnings", ())
            ),
        },
    )


def _synthesis_failure_result(
    *,
    loop: SynthesisLoopResult,
    depth: ReasoningDepth,
    max_attempts: int,
    witness_context: Mapping[str, Any],
) -> ReasoningResult:
    return ReasoningResult(
        summary="Reasoning synthesis failed validation",
        claims=(),
        confidence=0.0,
        tier=depth,
        state={
            "status": "failed",
            "succeeded": False,
            "stage": "validation",
            "error_text": loop.error_text,
            "reasons": loop.failure_reasons,
            "attempts_used": max_attempts,
            "tokens_used": loop.total_tokens,
        },
        trace={
            "attempts": loop.attempts,
            "last_response": loop.last_text,
            "last_candidate": loop.last_candidate or {},
            "witness_context_present": bool(witness_context),
        },
    )


def _continuation_success_result(
    *,
    loop: SynthesisLoopResult,
    entity_id: str,
    entity_type: str,
    goal: str,
    pack_name: str,
    pack_version: str,
    pack_policies: Mapping[str, Any],
    pack_prompts: Mapping[str, Any],
    depth: ReasoningDepth,
    generation: int,
    prior_synthesis_hash: str,
    prior_summary: str,
    prior_attempts_used: int,
    event_type: str,
    falsification_hint_seen: bool,
    witness_context: Mapping[str, Any],
) -> ReasoningResult:
    candidate = loop.valid_candidate or {}
    claims = extract_claims(candidate)
    confidence = extract_confidence(candidate, claims)
    return ReasoningResult(
        summary=extract_summary(candidate),
        claims=claims,
        confidence=confidence,
        tier=depth,
        state={
            "status": "completed",
            "succeeded": True,
            "stage": "continuation",
            "entity_id": entity_id,
            "entity_type": entity_type,
            "goal": goal,
            "pack": pack_name,
            "pack_version": pack_version,
            "pack_policies": dict(pack_policies or {}),
            "pack_prompts": dict(pack_prompts or {}),
            "generation": generation,
            "prior_synthesis_hash": prior_synthesis_hash,
            "prior_attempts_used": prior_attempts_used,
            "events_consumed": (event_type,),
            "attempts_used": len(loop.attempts),
            "tokens_used": loop.total_tokens,
            "depth": depth,
            "raw_synthesis": dict(candidate),
        },
        trace={
            "attempts": loop.attempts,
            "witness_context_present": bool(witness_context),
            "prior_summary": prior_summary,
            "event_type": event_type,
            "falsification_hint_seen": falsification_hint_seen,
            "validation_warnings": tuple(
                warning
                for attempt in loop.attempts
                for warning in attempt.get("warnings", ())
            ),
        },
    )


def _continuation_failure_result(
    *,
    loop: SynthesisLoopResult,
    depth: ReasoningDepth,
    max_attempts: int,
    prior_summary: str,
    prior_synthesis_hash: str,
    generation: int,
    event_type: str,
    falsification_hint_seen: bool,
    witness_context: Mapping[str, Any],
) -> ReasoningResult:
    return ReasoningResult(
        summary="Reasoning continuation failed validation",
        claims=(),
        confidence=0.0,
        tier=depth,
        state={
            "status": "failed",
            "succeeded": False,
            "stage": "continuation_validation",
            "error_text": loop.error_text,
            "reasons": loop.failure_reasons,
            "generation": generation,
            "prior_synthesis_hash": prior_synthesis_hash,
            "events_consumed": (event_type,),
            "attempts_used": max_attempts,
            "tokens_used": loop.total_tokens,
        },
        trace={
            "attempts": loop.attempts,
            "last_response": loop.last_text,
            "last_candidate": loop.last_candidate or {},
            "witness_context_present": bool(witness_context),
            "prior_summary": prior_summary,
            "event_type": event_type,
            "falsification_hint_seen": falsification_hint_seen,
        },
    )


def _continuation_invalid_state_result(
    *,
    state: Mapping[str, Any],
    event: Mapping[str, Any],
) -> ReasoningResult:
    prior_status = str(state.get("status") or "unknown")
    depth_value: ReasoningDepth = state.get("depth") or state.get("tier") or "L2"  # type: ignore[assignment]
    return ReasoningResult(
        summary="Continuation rejected: prior reasoning state is not completed",
        claims=(),
        confidence=0.0,
        tier=depth_value,
        state={
            "status": "failed",
            "succeeded": False,
            "stage": "continuation_input",
            "reasons": ("prior_state_not_completed", f"prior_status={prior_status}"),
            "attempts_used": 0,
            "tokens_used": 0,
        },
        trace={
            "attempts": (),
            "prior_status": prior_status,
            "event_type": str(event.get("event_type") or ""),
        },
    )


def _hash_prior_synthesis(raw_synthesis: Mapping[str, Any]) -> str:
    text = json.dumps(raw_synthesis, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coerce_evidence_item(item: Any) -> Any:
    if isinstance(item, EvidenceItem):
        return item
    if isinstance(item, Mapping):
        kwargs = dict(item)
        try:
            return EvidenceItem(**kwargs)
        except TypeError:
            return EvidenceItem(
                source_type=str(kwargs.get("source_type") or "unknown"),
                source_id=str(kwargs.get("source_id") or ""),
                text=str(kwargs.get("text") or ""),
            )
    return EvidenceItem(source_type="unknown", source_id="", text=str(item))


async def check_falsification(
    claim: Mapping[str, Any],
    fresh_evidence: Sequence[EvidenceItem],
    *,
    policy: FalsificationPolicy | None = None,
    ports: ReasoningPorts | None = None,
) -> FalsificationResult:
    """Check whether fresh evidence falsifies a prior claim.

    Sends the claim, the fresh evidence, and the policy's rules to the
    host LLM port and asks which conditions are triggered. Returns a
    structured ``FalsificationResult`` with triggered/non-triggered
    condition lists and a single ``should_invalidate`` verdict.

    Conservative semantics (``policy.conservative=True``, the default):
    if the LLM returns malformed JSON or refuses to commit, no
    conditions are reported triggered and ``should_invalidate`` is
    ``False`` -- the prior claim survives ambiguity.
    """

    port_bundle = ports or ReasoningPorts()
    if port_bundle.llm is None:
        raise ConfigurationError(
            "check_falsification requires ReasoningPorts.llm; provide an "
            "extracted_llm_infrastructure-backed LLM port."
        )

    effective_policy = policy or FalsificationPolicy()
    # Synthesize stable ids for any rule missing id/name/condition_id so the
    # LLM, the policy surface, and the result all reference the same handle.
    # Collision-safe: skip past any rule_<i> a host already assigned explicitly.
    explicit_ids = {_rule_id(r) for r in effective_policy.rules if _rule_id(r)}
    normalized_rules: list[dict[str, Any]] = []
    rule_ids_list: list[str] = []
    synth_counter = 0
    for rule in effective_policy.rules:
        rule_dict = dict(rule)
        rid = _rule_id(rule_dict)
        if not rid:
            while f"rule_{synth_counter}" in explicit_ids:
                synth_counter += 1
            rid = f"rule_{synth_counter}"
            synth_counter += 1
        rule_dict["id"] = rid
        normalized_rules.append(rule_dict)
        rule_ids_list.append(rid)
    rule_ids = tuple(rule_ids_list)

    claim_id = str((claim or {}).get("claim_id") or (claim or {}).get("id") or "")
    payload = {
        "claim": dict(claim or {}),
        "fresh_evidence": tuple(evidence_to_mapping(item) for item in fresh_evidence),
        "rules": tuple(normalized_rules),
        "conservative": bool(effective_policy.conservative),
    }
    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    system_prompt = _resolve_falsification_prompt()

    trace_sink = port_bundle.trace_sink
    span = None
    if trace_sink is not None:
        span = trace_sink.start_span(
            "extracted_reasoning_core.check_falsification",
            metadata={
                "claim_id": claim_id,
                "rule_count": len(rule_ids),
                "conservative": effective_policy.conservative,
            },
        )

    try:
        response = await port_bundle.llm.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload_text},
            ],
            max_tokens=effective_policy.max_tokens,
            temperature=effective_policy.temperature,
            metadata={
                "reasoning_mode": "falsification_check",
                "claim_id": claim_id,
                "rule_count": len(rule_ids),
            },
        )

        raw_text = clean_reasoning_text(response_text(response))
        parsed = parse_llm_json(raw_text)

        triggered: tuple[str, ...] = ()
        non_triggered: tuple[str, ...] = ()
        should_invalidate = False
        parse_failed = False
        raw_triggered: list[Any] = []
        raw_non_triggered: list[Any] = []

        if isinstance(parsed, Mapping) and not parsed.get("_parse_fallback"):
            raw_triggered = list(parsed.get("triggered_conditions") or [])
            raw_non_triggered = list(parsed.get("non_triggered_conditions") or [])
            triggered = tuple(_normalize_condition(c) for c in raw_triggered if _normalize_condition(c))
            non_triggered = tuple(_normalize_condition(c) for c in raw_non_triggered if _normalize_condition(c))
            verdict = parsed.get("should_invalidate")
            if verdict is None:
                # Conservative: never auto-invalidate. Aggressive: any trigger invalidates.
                should_invalidate = bool(triggered) and not effective_policy.conservative
            else:
                should_invalidate = bool(verdict)
            if effective_policy.conservative and not triggered:
                # Override LLM verdict: no triggered conditions means no invalidation.
                should_invalidate = False
        else:
            parse_failed = True
            # Conservative fallback: surface no triggers, do not invalidate.

        if rule_ids:
            seen = set(triggered) | set(non_triggered)
            unseen = tuple(rid for rid in rule_ids if rid not in seen)
            non_triggered = non_triggered + unseen

        usage = dict(response.get("usage") or {}) if isinstance(response, Mapping) else {}
        tokens_used = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)

        if port_bundle.event_sink is not None:
            event_name = (
                "reasoning.falsification.parse_failed"
                if parse_failed
                else "reasoning.falsification.completed"
            )
            await port_bundle.event_sink.emit(
                event_name,
                "extracted_reasoning_core.check_falsification",
                {
                    "triggered_count": len(triggered),
                    "should_invalidate": should_invalidate,
                    "tokens_used": tokens_used,
                    "claim_id": claim_id,
                },
                entity_type="claim",
                entity_id=claim_id,
            )
        if trace_sink is not None and span is not None:
            trace_sink.end_span(
                span,
                status="error" if parse_failed else "ok",
                metadata={
                    "triggered_count": len(triggered),
                    "should_invalidate": should_invalidate,
                    "tokens_used": tokens_used,
                    "parse_failed": parse_failed,
                },
            )

        return FalsificationResult(
            triggered_conditions=triggered,
            non_triggered_conditions=non_triggered,
            should_invalidate=should_invalidate,
            trace={
                "conservative": effective_policy.conservative,
                "rule_ids": rule_ids,
                "parse_failed": parse_failed,
                "raw_triggered": tuple(str(c) for c in raw_triggered),
                "raw_non_triggered": tuple(str(c) for c in raw_non_triggered),
                "tokens_used": tokens_used,
                "claim_id": claim_id,
            },
        )
    except Exception:
        if trace_sink is not None and span is not None:
            trace_sink.end_span(span, status="error", metadata={"stage": "exception"})
        raise


def _resolve_falsification_prompt() -> str:
    from .skills.registry import get_skill_registry

    registry = get_skill_registry()
    return registry.get_prompt("digest/reasoning_falsification") or (
        "Given a claim, fresh evidence, and a list of falsification rules, "
        "return a JSON object with triggered_conditions (array of rule ids), "
        "non_triggered_conditions (array of rule ids), and should_invalidate "
        "(boolean). Be conservative when conservative=true."
    )


def _rule_id(rule: Mapping[str, Any]) -> str:
    return str(rule.get("id") or rule.get("name") or rule.get("condition_id") or "").strip()


def _normalize_condition(value: Any) -> str:
    if isinstance(value, Mapping):
        return _rule_id(value)
    return str(value or "").strip()


def compute_evidence_hash(evidence: Mapping[str, Any]) -> str:
    """Compute a stable hex digest for an evidence mapping.

    Delegates to :func:`extracted_reasoning_core.semantic_cache_keys.compute_evidence_hash`
    so this public surface and the existing semantic-cache primitives
    produce identical fingerprints. Returns a 16-char hex prefix of the
    sha256 digest (matches the established ``reasoning_semantic_cache``
    storage convention).
    """
    from .semantic_cache_keys import compute_evidence_hash as _compute

    return _compute(dict(evidence or {}))


def build_semantic_cache_key(
    reasoning_input: ReasoningInput,
    *,
    tier: str,
    pack_name: str | None = None,
) -> str:
    """Build a stable semantic-cache key for a reasoning input.

    The key encodes everything that should make two reasoning calls
    cache-equivalent: entity identity, goal, evidence content, context,
    pack name, and tier. Format: ``reasoning/{tier}/{pack}/{digest}`` so
    the leading segments are human-readable while the digest (16-char
    hex from :func:`compute_evidence_hash`) captures full content
    sensitivity.

    The default pack name (``"reasoning_synthesis"``) matches the fallback
    that ``run_reasoning`` and ``continue_reasoning`` apply, so cache
    lookups for inputs without an explicit pack still align with what
    synthesis actually computed against.

    ``tier`` and ``pack_name`` must not contain ``/`` (raises
    ``ValueError``) since the slash separates key segments.
    """
    effective_pack = pack_name or reasoning_input.pack_name or "reasoning_synthesis"
    for label, value in (("tier", tier), ("pack_name", effective_pack)):
        if "/" in value:
            raise ValueError(
                f"semantic cache key {label} cannot contain '/': {value!r}"
            )
    payload = {
        "entity_id": reasoning_input.entity_id,
        "entity_type": reasoning_input.entity_type,
        "goal": reasoning_input.goal,
        "evidence": [evidence_to_mapping(item) for item in reasoning_input.evidence],
        "context": dict(reasoning_input.context or {}),
        "pack_name": effective_pack,
        "tier": tier,
    }
    digest = compute_evidence_hash(payload)
    return f"reasoning/{tier}/{effective_pack}/{digest}"


def load_reasoning_pack(name: str) -> ReasoningPack | None:
    """Load a named reasoning pack from the shared pack registry.

    Adapts a ``pack_registry.Pack`` (registered by an owning product at
    import time) into the public ``ReasoningPack`` shape that
    ``run_reasoning`` and ``continue_reasoning`` consume. When multiple
    versions of a pack are registered, returns the highest by **proper
    semver comparison** (``packaging.version.Version``) rather than the
    lexicographic comparison the underlying ``pack_registry.get_pack``
    uses -- so ``1.10.0`` correctly outranks ``1.9.0`` here.

    Returns ``None`` when no pack is registered under ``name`` -- core
    never raises on unknown names, matching the existing
    ``pack_registry.get_pack`` ergonomic so callers can run without any
    pack registered. Callers needing strict-load semantics should check
    for ``None`` explicitly or call ``pack_registry.get_pack`` with an
    explicit version.

    The registry's free-form ``metadata`` is mapped to
    ``ReasoningPack.policies`` so consumers like
    ``synthesis_config_from_pack`` pick up policy flags
    (``max_attempts``, ``temperature``, etc.) directly.
    """
    from packaging.version import InvalidVersion, Version

    from .pack_registry import list_packs

    candidates = [p for p in list_packs() if p.name == name]
    if not candidates:
        return None

    def _semver_key(p: Any) -> Any:
        try:
            return (0, Version(p.version))
        except InvalidVersion:
            # Non-semver versions sort below any valid semver; among
            # themselves, fall back to lexicographic so behavior is at
            # least deterministic.
            return (-1, p.version)

    pack = max(candidates, key=_semver_key)
    return ReasoningPack(
        name=pack.name,
        version=pack.version,
        prompts=dict(pack.prompts or {}),
        policies=dict(pack.metadata or {}),
    )


def validate_reasoning_output(
    result: ReasoningResult,
    *,
    policy: OutputPolicy | None = None,
) -> ValidationReport:
    """Validate a reasoned output against an output policy.

    Pure deterministic check (no LLM call). Each policy field maps to a
    specific blocker class:

      * ``min_confidence`` → ``confidence_below_min`` blocker if the
        result's overall confidence is under the threshold.
      * ``require_citations`` → ``claim_missing_citations:<idx>``
        blocker for each claim with no ``source_ids``.
      * ``required_claim_types`` → ``missing_required_claim_type:<type>``
        blocker for each declared type absent from any claim's
        ``type``/``category`` field.
      * ``blocked_phrasing`` → ``blocked_phrasing:<phrase>`` blocker if
        any blocked phrase (case-insensitive, **word-boundary** match)
        appears in the result summary or any claim's prose fields. The
        scanned claim keys are: ``claim``, ``summary``, ``narrative``,
        ``explanation``, ``rationale``, ``description``. Other claim
        fields (id, type, source_ids, etc.) are intentionally not
        scanned to avoid false positives on identifier-shaped strings.
        Word-boundary matching means ``"promise"`` does NOT block
        ``"compromise"`` and ``"free"`` does NOT block ``"freedom"``.

    Empty claims raises a single ``no_claims`` blocker (always invalid).

    ``passed`` is ``True`` iff there are no blockers. Warnings are not
    used in v1 (reserved for future soft checks). ``repaired_fields`` is
    always empty -- this validator inspects, it does not mutate.
    """
    effective_policy = policy or OutputPolicy()
    blockers: list[str] = []
    warnings: list[str] = []

    if not result.claims:
        blockers.append("no_claims")

    if result.confidence < effective_policy.min_confidence:
        blockers.append("confidence_below_min")

    if effective_policy.require_citations:
        for index, claim in enumerate(result.claims):
            source_ids = claim.get("source_ids") or ()
            if not source_ids:
                blockers.append(f"claim_missing_citations:{index}")

    if effective_policy.required_claim_types:
        present_types = {
            str(claim.get("type") or claim.get("category") or "").strip()
            for claim in result.claims
        }
        present_types.discard("")
        for required in effective_policy.required_claim_types:
            if required not in present_types:
                blockers.append(f"missing_required_claim_type:{required}")

    if effective_policy.blocked_phrasing:
        prose_keys = ("claim", "summary", "narrative", "explanation", "rationale", "description")
        haystack_parts = [result.summary or ""]
        for claim in result.claims:
            for key in prose_keys:
                value = claim.get(key)
                if isinstance(value, str):
                    haystack_parts.append(value)
        haystack = "\n".join(haystack_parts)
        for phrase in effective_policy.blocked_phrasing:
            phrase_str = str(phrase)
            if not phrase_str:
                continue
            pattern = re.compile(rf"\b{re.escape(phrase_str)}\b", re.IGNORECASE)
            if pattern.search(haystack):
                blockers.append(f"blocked_phrasing:{phrase}")

    return ValidationReport(
        passed=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        repaired_fields={},
        trace={
            "claim_count": len(result.claims),
            "confidence": result.confidence,
            "tier": result.tier,
            "policy_required_types": tuple(effective_policy.required_claim_types),
            "policy_min_confidence": effective_policy.min_confidence,
        },
    )


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
    "ConfigurationError",
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
    "continue_reasoning",
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
    "validate_reasoning_output",
    "validate_wedge",
    "wedge_from_archetype",
]
