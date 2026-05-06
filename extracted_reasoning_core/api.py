"""Public API for the extracted reasoning core."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
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
    """Run the single-pass reasoning synthesis flow.

    This is the extracted, product-neutral lift of the production
    per-vendor synthesis loop: build one prompt from already-prepared
    evidence/witness context, call the shared LLM port, validate the JSON
    response, and retry with compact validation feedback when validation
    fails. Host-owned witness compression stays outside this package and
    is supplied either through ``ReasoningPorts.witness_context`` or the
    input context.
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
    policies = dict(effective_pack.policies or {})
    max_attempts = max(1, int(policies.get("max_attempts", 2)))
    feedback_limit = max(1, int(policies.get("feedback_limit", 5)))
    max_tokens = max(1, int(policies.get("max_tokens", 16384)))
    temperature = float(policies.get("temperature", 0.3))

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

    attempts: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    last_text = ""
    last_candidate: dict[str, Any] | None = None
    total_tokens = 0

    try:
        for attempt_index in range(max_attempts):
            attempt_no = attempt_index + 1
            messages: list[Mapping[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload_text},
            ]
            if attempt_index > 0 and last_text:
                messages.append({"role": "assistant", "content": last_text})
            if attempt_index > 0 and failure_reasons:
                feedback = "\n".join(
                    f"- {reason}" for reason in failure_reasons[:feedback_limit]
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was rejected. Return a complete "
                        "corrected JSON object only.\nFix these issues:\n"
                        f"{feedback}"
                    ),
                })

            response = await port_bundle.llm.complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                metadata={
                    "entity_id": reasoning_input.entity_id,
                    "entity_type": reasoning_input.entity_type,
                    "goal": reasoning_input.goal,
                    "depth": depth,
                    "pack": effective_pack.name,
                    "attempt_no": attempt_no,
                    "reasoning_mode": "single_pass_synthesis",
                },
            )
            text = _response_text(response)
            last_text = _clean_reasoning_text(text)
            usage = dict(response.get("usage") or {}) if isinstance(response, Mapping) else {}
            attempt_tokens = _usage_tokens(usage)
            total_tokens += attempt_tokens

            parsed = _parse_llm_json(last_text)
            validation = _validate_reasoning_candidate(parsed)
            attempts.append({
                "attempt_no": attempt_no,
                "valid": validation["valid"],
                "errors": tuple(validation["errors"]),
                "warnings": tuple(validation["warnings"]),
                "tokens_used": attempt_tokens,
            })

            if validation["valid"]:
                assert isinstance(parsed, dict)
                last_candidate = parsed
                result = _candidate_to_reasoning_result(
                    parsed,
                    reasoning_input=reasoning_input,
                    depth=depth,
                    pack=effective_pack,
                    witness_context=witness_context,
                    attempts=attempts,
                    tokens_used=total_tokens,
                )
                if port_bundle.event_sink is not None:
                    await port_bundle.event_sink.emit(
                        "reasoning.synthesis.completed",
                        "extracted_reasoning_core.run_reasoning",
                        {"attempts_used": attempt_no, "tokens_used": total_tokens},
                        entity_type=reasoning_input.entity_type,
                        entity_id=reasoning_input.entity_id,
                    )
                if trace_sink is not None and span is not None:
                    trace_sink.end_span(
                        span,
                        status="ok",
                        metadata={"attempts_used": attempt_no, "tokens_used": total_tokens},
                    )
                return result

            failure_reasons = list(validation["errors"])

        error_text = "; ".join(failure_reasons[:2])[:200] or "validation failed"
        if port_bundle.event_sink is not None:
            await port_bundle.event_sink.emit(
                "reasoning.synthesis.validation_failed",
                "extracted_reasoning_core.run_reasoning",
                {
                    "attempts_used": max_attempts,
                    "tokens_used": total_tokens,
                    "errors": tuple(failure_reasons[:feedback_limit]),
                },
                entity_type=reasoning_input.entity_type,
                entity_id=reasoning_input.entity_id,
            )
        if trace_sink is not None and span is not None:
            trace_sink.end_span(
                span,
                status="error",
                metadata={"attempts_used": max_attempts, "error_text": error_text},
            )
        return ReasoningResult(
            summary="Reasoning synthesis failed validation",
            claims=(),
            confidence=0.0,
            tier=depth,
            state={
                "status": "failed",
                "succeeded": False,
                "stage": "validation",
                "error_text": error_text,
                "reasons": tuple(failure_reasons[:feedback_limit]),
                "attempts_used": max_attempts,
                "tokens_used": total_tokens,
            },
            trace={
                "attempts": tuple(attempts),
                "last_response": last_text,
                "last_candidate": last_candidate or {},
                "witness_context_present": bool(witness_context),
            },
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
        "evidence": tuple(_evidence_to_mapping(item) for item in reasoning_input.evidence),
        "context": dict(reasoning_input.context or {}),
        "witness_context": dict(witness_context or {}),
    }


def _evidence_to_mapping(item: EvidenceItem) -> Mapping[str, Any]:
    if is_dataclass(item):
        return asdict(item)
    return dict(item)  # type: ignore[arg-type]


def _response_text(response: Mapping[str, Any]) -> str:
    for key in ("response", "content", "text"):
        value = response.get(key)
        if value is not None:
            return str(value)
    message = response.get("message")
    if isinstance(message, Mapping) and message.get("content") is not None:
        return str(message.get("content"))
    return json.dumps(response, default=str)


def _clean_reasoning_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    if "<scratchpad>" in cleaned:
        cleaned = cleaned.split("</scratchpad>")[-1].strip()
    return cleaned


def _parse_llm_json(text: str) -> Any:
    try:
        from extracted_llm_infrastructure.pipelines.llm import parse_json_response
    except ImportError:
        parse_json_response = None
    if parse_json_response is not None:
        return parse_json_response(text, recover_truncated=True)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _validate_reasoning_candidate(candidate: Any) -> Mapping[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(candidate, dict):
        return {"valid": False, "errors": ("LLM did not return a JSON object",), "warnings": ()}
    if candidate.get("_parse_fallback"):
        return {"valid": False, "errors": ("LLM did not return valid JSON",), "warnings": ()}
    summary = _extract_summary(candidate)
    claims = _extract_claims(candidate)
    if not summary:
        errors.append("missing_summary")
    if not claims:
        errors.append("missing_claims")
    confidence = _extract_confidence(candidate, claims)
    if confidence <= 0.0:
        warnings.append("zero_or_missing_confidence")
    return {"valid": not errors, "errors": tuple(errors), "warnings": tuple(warnings)}


def _extract_summary(candidate: Mapping[str, Any]) -> str:
    for key in ("summary", "executive_summary", "causal_narrative"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            nested = _extract_summary(value)
            if nested:
                return nested
    contracts = candidate.get("reasoning_contracts")
    if isinstance(contracts, Mapping):
        for value in contracts.values():
            if isinstance(value, Mapping):
                nested = _extract_summary(value)
                if nested:
                    return nested
    return ""


def _extract_claims(candidate: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = candidate.get("claims")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        claims = tuple(dict(item) if isinstance(item, Mapping) else {"claim": str(item)} for item in raw)
        if claims:
            return claims
    contracts = candidate.get("reasoning_contracts")
    found: list[Mapping[str, Any]] = []
    if isinstance(contracts, Mapping):
        _collect_contract_claims(contracts, found)
    return tuple(found)


def _collect_contract_claims(value: Mapping[str, Any], found: list[Mapping[str, Any]]) -> None:
    for key, item in value.items():
        if not isinstance(item, Mapping):
            continue
        claim_text = item.get("claim") or item.get("summary") or item.get("narrative")
        if claim_text:
            found.append({"claim": str(claim_text), "section": str(key), **dict(item)})
        nested = item.get("claims") or item.get("sections")
        if isinstance(nested, Mapping):
            _collect_contract_claims(nested, found)


def _extract_confidence(candidate: Mapping[str, Any], claims: Sequence[Mapping[str, Any]]) -> float:
    rank = {"high": 0.9, "medium": 0.6, "low": 0.3, "insufficient": 0.0}
    raw = candidate.get("confidence")
    if raw is not None:
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return rank.get(str(raw).strip().lower(), 0.0)
    values: list[float] = []
    for claim in claims:
        value = claim.get("confidence")
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(rank.get(str(value or "").strip().lower(), 0.0))
    return max(values) if values else 0.0


def _candidate_to_reasoning_result(
    candidate: Mapping[str, Any],
    *,
    reasoning_input: ReasoningInput,
    depth: ReasoningDepth,
    pack: ReasoningPack,
    witness_context: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    tokens_used: int,
) -> ReasoningResult:
    claims = _extract_claims(candidate)
    confidence = _extract_confidence(candidate, claims)
    return ReasoningResult(
        summary=_extract_summary(candidate),
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
            "attempts_used": len(attempts),
            "tokens_used": tokens_used,
            "raw_synthesis": dict(candidate),
        },
        trace={
            "attempts": tuple(attempts),
            "witness_context_present": bool(witness_context),
            "validation_warnings": tuple(
                warning
                for attempt in attempts
                for warning in attempt.get("warnings", ())
            ),
        },
    )


def _usage_tokens(usage: Mapping[str, Any]) -> int:
    return int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)


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
