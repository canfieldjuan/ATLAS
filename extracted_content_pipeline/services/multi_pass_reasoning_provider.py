"""Bridge from extracted_reasoning_core into the content-pipeline port.

This module is the consumer-side wire-up that lets AI Content Ops use
the extracted_reasoning_core producer surface (PR-D20a..g). It
implements ``CampaignReasoningContextProvider`` so any host that
already consumes the existing port shape gets multi-pass reasoning by
swapping the provider, without changing any campaign-generation code.

Per-call flow:
  1. ``run_reasoning`` produces the initial reasoning state from the
     opportunity's evidence.
  2. If ``opportunity["events"]`` is non-empty AND
     ``config.enable_multi_pass`` is true, each event is fed through
     ``continue_reasoning`` in sequence so later events refine the
     prior state. The chain stops on the first continuation that fails
     validation; the last successful state is what feeds the campaign
     context.
  3. The final ``ReasoningResult`` is translated into the
     ``CampaignReasoningContext`` the campaign generator already
     consumes.

Falsification (``check_falsification``), narrative planning
(``build_narrative_plan``), and output validation
(``validate_reasoning_output``) remain reachable via the shared
``ReasoningPorts`` for host code that wants them; this provider keeps
its surface narrow.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from ..campaign_ports import CampaignReasoningContext, TenantScope

if TYPE_CHECKING:
    from extracted_reasoning_core.types import FalsificationPolicy, ReasoningPack


@dataclass(frozen=True)
class MultiPassReasoningProviderConfig:
    """Optional per-call defaults used when an opportunity doesn't supply them."""

    default_goal: str = "synthesize vendor pressure"
    default_depth: str = "L2"
    pack_name: str | None = None
    top_thesis_limit: int = 5
    # When true (default), opportunity["events"] are chained through
    # continue_reasoning so each event refines the prior state. Set to
    # false to fall back to the single-pass run_reasoning behavior even
    # when events are present (useful for host-side budget controls).
    enable_multi_pass: bool = True
    # Hard cap on how many events from opportunity["events"] are
    # consumed in one call. Protects against runaway token spend if a
    # buggy upstream feeds an unbounded list.
    max_continuations: int = 8
    # Optional falsification policy. When set, after the chain completes
    # each surviving claim is run through check_falsification against the
    # aggregated evidence (opportunity + event evidence). Falsified
    # claim ids are surfaced in scope_summary["falsification"]; the
    # claims are dropped from top_theses iff drop_falsified is True.
    falsification_policy: "FalsificationPolicy | None" = None
    # When True (and falsification_policy is set), claims marked
    # should_invalidate=True are removed from the resulting
    # top_theses/witness_highlights. When False (default), they remain
    # but are flagged in scope_summary["falsification"]["falsified_claim_ids"]
    # so downstream renderers can decide what to do.
    drop_falsified: bool = False
    # Optional ReasoningPack for narrative-plan structuring. When set,
    # build_narrative_plan runs after the chain (and after falsification
    # gating, if any) on the final result. The resulting NarrativePlan
    # is surfaced as canonical_reasoning["narrative_plan"]. Hosts that
    # want to load a pack from the registry can do so themselves via
    # extracted_reasoning_core.api.load_reasoning_pack and pass the
    # result here. When None (default), no narrative plan is produced.
    narrative_plan_pack: "ReasoningPack | None" = None


class MultiPassCampaignReasoningProvider:
    """``CampaignReasoningContextProvider`` backed by ``extracted_reasoning_core``.

    Constructor accepts a ``ReasoningPorts`` bundle (must include an LLM
    port) and optional config. Each ``read_campaign_reasoning_context``
    call builds a ``ReasoningInput`` from the opportunity, calls
    ``run_reasoning``, and translates the resulting ``ReasoningResult``
    into the ``CampaignReasoningContext`` the campaign generator already
    consumes.
    """

    def __init__(
        self,
        ports: Any,
        *,
        config: MultiPassReasoningProviderConfig | None = None,
    ) -> None:
        self._ports = ports
        self._config = config or MultiPassReasoningProviderConfig()

    async def read_campaign_reasoning_context(
        self,
        *,
        scope: TenantScope,
        target_id: str,
        target_mode: str,
        opportunity: Mapping[str, Any],
    ) -> CampaignReasoningContext | None:
        del scope  # tenant scoping is the host's responsibility upstream

        from extracted_reasoning_core.api import (
            continue_reasoning,
            load_reasoning_pack,
            run_reasoning,
        )
        from extracted_reasoning_core.types import ReasoningInput

        evidence_items = tuple(
            _coerce_evidence(item) for item in (opportunity.get("evidence") or ())
        )
        goal = str(opportunity.get("goal") or self._config.default_goal)
        depth = str(opportunity.get("depth") or self._config.default_depth)
        # Per-opportunity pack_name overrides the provider-bound default so a
        # single provider can serve heterogeneous targets (e.g. different packs
        # per vendor segment) without instantiating one provider per pack.
        pack_name = opportunity.get("pack_name") or self._config.pack_name
        reasoning_input = ReasoningInput(
            entity_id=str(target_id),
            entity_type=str(target_mode or opportunity.get("entity_type") or ""),
            goal=goal,
            evidence=evidence_items,
            context=dict(opportunity.get("context") or {}),
            pack_name=pack_name,
        )

        pack = load_reasoning_pack(pack_name) if pack_name else None

        result = await run_reasoning(
            reasoning_input,
            depth=depth,  # type: ignore[arg-type]
            pack=pack,
            ports=self._ports,
        )

        if str(result.state.get("status") or "") != "completed":
            # Validation-failure path. Surface a defensive empty context
            # rather than raising; campaign generation handles None / empty
            # context as host-decision territory.
            return None

        # Chain continue_reasoning for any events on the opportunity. Each
        # event refines the prior state. The chain stops on the first
        # continuation that fails validation -- the last successful state
        # is what feeds the campaign context.
        events = list(opportunity.get("events") or ())
        events_total = len(events)
        events_consumed = 0
        chain_halted_on_failure = False
        if self._config.enable_multi_pass and events:
            limit = max(0, int(self._config.max_continuations))
            for event in events[:limit]:
                if not isinstance(event, Mapping):
                    raise TypeError(
                        "opportunity['events'] entries must be mappings with at "
                        f"least an 'event_type' key; got {type(event).__name__}: "
                        f"{event!r}"
                    )
                next_result = await continue_reasoning(
                    result.state,
                    event,
                    ports=self._ports,
                )
                if str(next_result.state.get("status") or "") != "completed":
                    # Continuation failed validation -- keep the prior
                    # successful state and stop the chain.
                    chain_halted_on_failure = True
                    break
                result = next_result
                events_consumed += 1

        events_truncated = max(0, events_total - max(0, int(self._config.max_continuations))) if self._config.enable_multi_pass else events_total
        chain_telemetry = {
            "events_total": events_total,
            "events_consumed": events_consumed,
            "events_truncated": events_truncated,
            "chain_halted_on_failure": chain_halted_on_failure,
        }

        # Optional falsification gate. Aggregates evidence from the
        # opportunity and every consumed event, then runs
        # check_falsification on each surviving claim. Falsified claim
        # ids are surfaced in scope_summary regardless of drop_falsified;
        # the drop_falsified flag controls whether those claims are also
        # removed from the rendered top_theses.
        falsification_summary: Mapping[str, Any] | None = None
        falsified_claim_ids: tuple[str, ...] = ()
        if self._config.falsification_policy is not None and result.claims:
            consumed_event_evidence: list[Any] = []
            if self._config.enable_multi_pass:
                for event in events[:events_consumed]:
                    for item in event.get("evidence") or ():
                        consumed_event_evidence.append(_coerce_evidence(item))
            fresh_evidence = evidence_items + tuple(consumed_event_evidence)

            # Run falsification checks in parallel -- they're independent
            # per-claim LLM calls. Each check is wrapped so a single
            # failure (network blip, LLM error) doesn't cascade and
            # collapse the whole context; the failed claim's trace
            # records the error and skips invalidation.
            checks = [
                _falsify_one(
                    claim,
                    fresh_evidence,
                    policy=self._config.falsification_policy,
                    ports=self._ports,
                )
                for claim in result.claims
            ]
            check_outcomes = await asyncio.gather(*checks)

            falsified_ids: list[str] = []
            falsification_traces: list[Mapping[str, Any]] = []
            for claim, outcome in zip(result.claims, check_outcomes):
                claim_id = _claim_identity(claim)
                trace_entry: dict[str, Any] = {"claim_id": claim_id}
                if outcome["error"] is not None:
                    trace_entry["should_invalidate"] = False
                    trace_entry["triggered_conditions"] = ()
                    trace_entry["error"] = outcome["error"]
                else:
                    fr = outcome["result"]
                    trace_entry["should_invalidate"] = fr.should_invalidate
                    trace_entry["triggered_conditions"] = tuple(fr.triggered_conditions)
                    if fr.should_invalidate:
                        falsified_ids.append(claim_id)
                falsification_traces.append(trace_entry)
            falsified_claim_ids = tuple(falsified_ids)
            falsification_summary = {
                "evaluated_claim_count": len(result.claims),
                "falsified_count": len(falsified_ids),
                "falsified_claim_ids": falsified_claim_ids,
                "drop_falsified": self._config.drop_falsified,
                "traces": tuple(falsification_traces),
            }

        # Optional narrative plan. When a pack is configured, run
        # build_narrative_plan over the final state and surface the
        # resulting NarrativePlan in canonical_reasoning. If
        # drop_falsified=True, the narrative-plan input is pre-filtered
        # to remove falsified claims so the rendered plan stays
        # consistent with the rendered top_theses.
        narrative_plan_summary: Mapping[str, Any] | None = None
        if self._config.narrative_plan_pack is not None:
            from extracted_reasoning_core.api import build_narrative_plan

            if self._config.drop_falsified and falsified_claim_ids:
                drop_set = set(falsified_claim_ids)
                raw = dict(result.state.get("raw_synthesis") or {})
                raw_claims = list(raw.get("claims") or ())
                raw["claims"] = [c for c in raw_claims if _claim_identity(c) not in drop_set]
                narrative_context: Mapping[str, Any] = {**dict(result.state), "raw_synthesis": raw}
            else:
                narrative_context = result.state

            plan = build_narrative_plan(narrative_context, pack=self._config.narrative_plan_pack)
            narrative_plan_summary = {
                "claims": tuple(dict(c) for c in plan.claims),
                "sections": tuple(dict(s) for s in plan.sections),
                "evidence_requirements": tuple(dict(e) for e in plan.evidence_requirements),
                "state_hints": dict(plan.state_hints),
            }

        return _result_to_campaign_context(
            result,
            limit=self._config.top_thesis_limit,
            chain_telemetry=chain_telemetry,
            falsification_summary=falsification_summary,
            drop_claim_ids=falsified_claim_ids if self._config.drop_falsified else (),
            narrative_plan=narrative_plan_summary,
        )


def _coerce_evidence(item: Any) -> Any:
    from extracted_reasoning_core.types import EvidenceItem

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


def _result_to_campaign_context(
    result: Any,
    *,
    limit: int,
    chain_telemetry: Mapping[str, Any] | None = None,
    falsification_summary: Mapping[str, Any] | None = None,
    drop_claim_ids: tuple[str, ...] = (),
    narrative_plan: Mapping[str, Any] | None = None,
) -> CampaignReasoningContext:
    raw_claims = list(result.claims or ())
    drop_set = set(drop_claim_ids)
    claims = (
        [c for c in raw_claims if _claim_identity(c) not in drop_set]
        if drop_set
        else raw_claims
    )
    # Top theses ordered by confidence descending (ties keep input order).
    sorted_claims = sorted(
        enumerate(claims),
        key=lambda pair: -_claim_confidence(pair[1]),
    )
    top = [claim for _, claim in sorted_claims[: max(0, int(limit))]]

    top_theses = tuple(
        {
            "claim": str(c.get("claim") or ""),
            "confidence": _claim_confidence(c),
            "source_ids": tuple(str(s) for s in (c.get("source_ids") or ())),
            "section": str(c.get("section") or ""),
        }
        for c in top
    )
    witness_highlights = tuple(
        {
            "claim": str(c.get("claim") or ""),
            "confidence": _claim_confidence(c),
        }
        for c in top
    )
    all_source_ids: list[str] = []
    for claim in claims:
        for sid in claim.get("source_ids") or ():
            sid_str = str(sid)
            if sid_str and sid_str not in all_source_ids:
                all_source_ids.append(sid_str)

    canonical_reasoning: dict[str, Any] = {
        "summary": result.summary,
        "confidence": result.confidence,
        "tier": result.tier,
        "generation": int(result.state.get("generation") or 1),
        "raw_synthesis": dict(result.state.get("raw_synthesis") or {}),
    }
    if narrative_plan is not None:
        canonical_reasoning["narrative_plan"] = dict(narrative_plan)

    return CampaignReasoningContext(
        anchor_examples={},
        witness_highlights=witness_highlights,
        reference_ids={"top_theses": tuple(all_source_ids)},
        top_theses=top_theses,
        account_signals=(),
        timing_windows=(),
        proof_points=(),
        coverage_limits=(),
        canonical_reasoning=canonical_reasoning,
        scope_summary={
            "entity_id": result.state.get("entity_id") or "",
            "entity_type": result.state.get("entity_type") or "",
            "goal": result.state.get("goal") or "",
            "pack": result.state.get("pack") or "",
            "tokens_used": int(result.state.get("tokens_used") or 0),
            "attempts_used": int(result.state.get("attempts_used") or 0),
            **dict(chain_telemetry or {}),
            **({"falsification": dict(falsification_summary)} if falsification_summary is not None else {}),
        },
        delta_summary={},
    )


async def _falsify_one(
    claim: Mapping[str, Any],
    fresh_evidence: tuple[Any, ...],
    *,
    policy: Any,
    ports: Any,
) -> dict[str, Any]:
    """Run check_falsification for one claim, capturing errors per-claim.

    Returns ``{"result": FalsificationResult, "error": None}`` on success
    and ``{"result": None, "error": "<exc class>: <msg>"}`` on failure
    so a single bad claim doesn't collapse the whole bridge call. The
    ``ConfigurationError`` (missing LLM port) is intentionally left to
    propagate -- that's a host config problem, not a runtime fault.
    """
    from extracted_reasoning_core.api import ConfigurationError, check_falsification

    try:
        fr = await check_falsification(claim, fresh_evidence, policy=policy, ports=ports)
        return {"result": fr, "error": None}
    except ConfigurationError:
        raise
    except Exception as exc:  # noqa: BLE001 -- intentional fail-soft per claim
        return {"result": None, "error": f"{type(exc).__name__}: {exc}"}


def _claim_identity(claim: Mapping[str, Any]) -> str:
    return str(claim.get("claim_id") or claim.get("id") or claim.get("claim") or "")[:200]


def _claim_confidence(claim: Mapping[str, Any]) -> float:
    rank = {"high": 0.9, "medium": 0.6, "low": 0.3, "insufficient": 0.0}
    raw = claim.get("confidence")
    if raw is None:
        return 0.0
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return rank.get(str(raw).strip().lower(), 0.0)


__all__ = [
    "MultiPassCampaignReasoningProvider",
    "MultiPassReasoningProviderConfig",
]
