from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping, Sequence

import pytest

from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    TenantScope,
)
from extracted_content_pipeline.services.multi_pass_reasoning_provider import (
    MultiPassCampaignReasoningProvider,
    MultiPassReasoningProviderConfig,
)
from extracted_reasoning_core.api import ConfigurationError
from extracted_reasoning_core.types import ReasoningPorts


class FakeLLMPort:
    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append({"max_tokens": max_tokens, "metadata": dict(metadata or {})})
        return self.responses.pop(0)


def _opportunity() -> dict[str, Any]:
    return {
        "vendor_id": "acme",
        "evidence": [
            {"source_type": "review", "source_id": "r1", "text": "Renewal pricing too high."},
            {"source_type": "ticket", "source_id": "t9", "text": "Onboarding pain."},
        ],
        "context": {"industry": "saas"},
    }


@pytest.mark.asyncio
async def test_multi_pass_provider_translates_run_reasoning_result_into_campaign_context() -> None:
    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "summary": "Pricing pressure dominates the displacement signal.",
                "claims": [
                    {"claim": "Renewal pricing drives churn.", "confidence": "high", "source_ids": ["r1"]},
                    {"claim": "Onboarding friction is secondary.", "confidence": 0.55, "source_ids": ["t9"]},
                ],
                "confidence": 0.8,
            }),
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
    ])
    provider = MultiPassCampaignReasoningProvider(ports=ReasoningPorts(llm=llm))

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert isinstance(context, CampaignReasoningContext)
    # Top theses ordered by confidence (high → ~0.55).
    assert context.top_theses[0]["claim"] == "Renewal pricing drives churn."
    assert context.top_theses[0]["confidence"] == pytest.approx(0.9)
    assert context.top_theses[1]["claim"] == "Onboarding friction is secondary."
    # Reference ids aggregate unique source ids across claims.
    assert set(context.reference_ids["top_theses"]) == {"r1", "t9"}
    assert context.canonical_reasoning["summary"].startswith("Pricing pressure")
    assert context.canonical_reasoning["confidence"] == pytest.approx(0.8)
    assert context.canonical_reasoning["generation"] == 1
    assert context.scope_summary["entity_id"] == "acme"
    assert context.scope_summary["entity_type"] == "vendor"
    # LLM was called with the right reasoning_mode tag, proving the
    # reasoning-core surface is the one that actually ran.
    assert llm.calls[0]["metadata"]["reasoning_mode"] == "single_pass_synthesis"


@pytest.mark.asyncio
async def test_multi_pass_provider_returns_none_when_run_reasoning_fails_validation() -> None:
    # Two responses both missing claims → run_reasoning returns failed result.
    llm = FakeLLMPort([
        {"response": json.dumps({"summary": "no claims yet"}), "usage": {}},
        {"response": json.dumps({"summary": "still no claims"}), "usage": {}},
    ])
    provider = MultiPassCampaignReasoningProvider(ports=ReasoningPorts(llm=llm))

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert context is None


@pytest.mark.asyncio
async def test_multi_pass_provider_raises_configuration_error_without_llm_port() -> None:
    provider = MultiPassCampaignReasoningProvider(ports=ReasoningPorts())

    with pytest.raises(ConfigurationError, match="ReasoningPorts.llm"):
        await provider.read_campaign_reasoning_context(
            scope=TenantScope(),
            target_id="acme",
            target_mode="vendor",
            opportunity=_opportunity(),
        )


@pytest.mark.asyncio
async def test_multi_pass_provider_honors_top_thesis_limit() -> None:
    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "summary": "Multiple drivers identified.",
                "claims": [
                    {"claim": f"Claim {i}", "confidence": 0.9 - 0.1 * i, "source_ids": [f"r{i}"]}
                    for i in range(5)
                ],
                "confidence": 0.7,
            }),
            "usage": {},
        }
    ])
    provider = MultiPassCampaignReasoningProvider(
        ports=ReasoningPorts(llm=llm),
        config=MultiPassReasoningProviderConfig(top_thesis_limit=2),
    )

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert context is not None
    assert len(context.top_theses) == 2
    # Reference ids still aggregate from all claims (limit applies to top_theses only).
    assert len(context.reference_ids["top_theses"]) == 5


@pytest.mark.asyncio
async def test_multi_pass_provider_honors_per_opportunity_pack_name_override() -> None:
    """An opportunity-level pack_name overrides the provider-bound default."""

    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "summary": "ok",
                "claims": [{"claim": "ok", "confidence": 0.7, "source_ids": ["r1"]}],
                "confidence": 0.7,
            }),
            "usage": {},
        }
    ])
    provider = MultiPassCampaignReasoningProvider(
        ports=ReasoningPorts(llm=llm),
        config=MultiPassReasoningProviderConfig(pack_name="provider_default"),
    )
    opportunity = _opportunity()
    opportunity["pack_name"] = "per_opportunity_override"

    await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=opportunity,
    )

    # The per-opportunity pack_name flows through to the LLM call metadata
    # rather than the provider-bound default.
    assert llm.calls[0]["metadata"]["pack"] == "per_opportunity_override"


def _initial_synthesis_response() -> dict[str, Any]:
    return {
        "response": json.dumps({
            "summary": "Initial synthesis: pricing pressure dominates.",
            "claims": [
                {"claim": "Renewal pricing drives churn.", "confidence": 0.8, "source_ids": ["r1"]},
            ],
            "confidence": 0.8,
        }),
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def _continuation_response(*, summary: str) -> dict[str, Any]:
    return {
        "response": json.dumps({
            "summary": summary,
            "claims": [
                {
                    "claim": f"Refined claim for {summary}",
                    "confidence": 0.85,
                    "source_ids": ["r1", "evt"],
                    "revised_from": "Renewal pricing drives churn.",
                },
            ],
            "confidence": 0.85,
        }),
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


@pytest.mark.asyncio
async def test_multi_pass_provider_chains_continue_reasoning_for_each_event() -> None:
    """opportunity["events"] are chained through continue_reasoning. Generation increments per event."""

    llm = FakeLLMPort([
        _initial_synthesis_response(),
        _continuation_response(summary="After event 1"),
        _continuation_response(summary="After event 2"),
    ])
    provider = MultiPassCampaignReasoningProvider(ports=ReasoningPorts(llm=llm))

    opp = _opportunity()
    opp["events"] = [
        {"event_type": "support_ticket", "evidence": [{"source_type": "ticket", "source_id": "t1", "text": "x"}]},
        {"event_type": "renewal_signal", "evidence": []},
    ]

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=opp,
    )

    assert context is not None
    # 1 run_reasoning + 2 continuations = 3 LLM calls
    assert len(llm.calls) == 3
    assert llm.calls[0]["metadata"]["reasoning_mode"] == "single_pass_synthesis"
    assert llm.calls[1]["metadata"]["reasoning_mode"] == "multi_pass_continuation"
    assert llm.calls[2]["metadata"]["reasoning_mode"] == "multi_pass_continuation"
    # Final state reflects the last continuation.
    assert context.canonical_reasoning["summary"] == "After event 2"
    # Generation increments: 1 (initial) → 2 (event 1) → 3 (event 2).
    assert context.canonical_reasoning["generation"] == 3
    # Chain telemetry surfaces in scope_summary.
    assert context.scope_summary["events_total"] == 2
    assert context.scope_summary["events_consumed"] == 2
    assert context.scope_summary["events_truncated"] == 0
    assert context.scope_summary["chain_halted_on_failure"] is False


@pytest.mark.asyncio
async def test_multi_pass_provider_skips_continuation_when_enable_multi_pass_is_false() -> None:
    llm = FakeLLMPort([_initial_synthesis_response()])
    provider = MultiPassCampaignReasoningProvider(
        ports=ReasoningPorts(llm=llm),
        config=MultiPassReasoningProviderConfig(enable_multi_pass=False),
    )

    opp = _opportunity()
    opp["events"] = [{"event_type": "would_be_ignored", "evidence": []}]

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=opp,
    )

    assert context is not None
    # Only the run_reasoning call; the event is silently ignored.
    assert len(llm.calls) == 1
    assert context.canonical_reasoning["generation"] == 1
    # Telemetry reflects that 1 event was provided but 0 consumed and the
    # whole list was effectively truncated by the disabled flag.
    assert context.scope_summary["events_total"] == 1
    assert context.scope_summary["events_consumed"] == 0
    assert context.scope_summary["events_truncated"] == 1
    assert context.scope_summary["chain_halted_on_failure"] is False


@pytest.mark.asyncio
async def test_multi_pass_provider_caps_continuation_chain_at_max_continuations() -> None:
    llm = FakeLLMPort([
        _initial_synthesis_response(),
        _continuation_response(summary="event 1"),
        _continuation_response(summary="event 2"),
    ])
    provider = MultiPassCampaignReasoningProvider(
        ports=ReasoningPorts(llm=llm),
        config=MultiPassReasoningProviderConfig(max_continuations=2),
    )

    opp = _opportunity()
    # 4 events but cap is 2.
    opp["events"] = [{"event_type": f"e{i}", "evidence": []} for i in range(4)]

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=opp,
    )

    assert context is not None
    # 1 run_reasoning + 2 continuations (capped) = 3 calls; events 3 and 4 ignored.
    assert len(llm.calls) == 3
    assert context.canonical_reasoning["generation"] == 3
    # Telemetry surfaces the truncation rather than letting it silently disappear.
    assert context.scope_summary["events_total"] == 4
    assert context.scope_summary["events_consumed"] == 2
    assert context.scope_summary["events_truncated"] == 2
    assert context.scope_summary["chain_halted_on_failure"] is False


@pytest.mark.asyncio
async def test_multi_pass_provider_keeps_prior_state_when_continuation_fails_validation() -> None:
    """If a mid-chain continuation fails validation, the chain stops and the last good state wins."""

    llm = FakeLLMPort([
        _initial_synthesis_response(),
        _continuation_response(summary="After event 1"),  # succeeds
        # Event 2 returns invalid JSON twice (max_attempts=2 default) → validation failure.
        {"response": json.dumps({"summary": "no claims"}), "usage": {}},
        {"response": json.dumps({"summary": "still no claims"}), "usage": {}},
    ])
    provider = MultiPassCampaignReasoningProvider(ports=ReasoningPorts(llm=llm))

    opp = _opportunity()
    opp["events"] = [
        {"event_type": "good", "evidence": []},
        {"event_type": "broken", "evidence": []},
    ]

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=opp,
    )

    assert context is not None
    # Result reflects event 1 (the last successful state), not event 2.
    assert context.canonical_reasoning["summary"] == "After event 1"
    assert context.canonical_reasoning["generation"] == 2
    # Telemetry signals the mid-chain validation halt.
    assert context.scope_summary["events_total"] == 2
    assert context.scope_summary["events_consumed"] == 1
    assert context.scope_summary["chain_halted_on_failure"] is True


@pytest.mark.asyncio
async def test_multi_pass_provider_rejects_non_mapping_events_with_clear_error() -> None:
    """A host accidentally passing strings (or any non-Mapping) gets a TypeError, not silent wrap."""

    llm = FakeLLMPort([_initial_synthesis_response()])
    provider = MultiPassCampaignReasoningProvider(ports=ReasoningPorts(llm=llm))
    opp = _opportunity()
    opp["events"] = ["accidentally_a_string"]

    with pytest.raises(TypeError, match="event_type"):
        await provider.read_campaign_reasoning_context(
            scope=TenantScope(),
            target_id="acme",
            target_mode="vendor",
            opportunity=opp,
        )


def _falsifiable_synthesis_response() -> dict[str, Any]:
    return {
        "response": json.dumps({
            "summary": "Two drivers identified.",
            "claims": [
                {"claim_id": "c1", "claim": "Renewal pricing drives churn.", "confidence": 0.8, "source_ids": ["r1"]},
                {"claim_id": "c2", "claim": "Onboarding friction is secondary.", "confidence": 0.6, "source_ids": ["r2"]},
            ],
            "confidence": 0.8,
        }),
        "usage": {"input_tokens": 4, "output_tokens": 2},
    }


def _falsification_response(*, triggered: list[str], should_invalidate: bool) -> dict[str, Any]:
    return {
        "response": json.dumps({
            "triggered_conditions": triggered,
            "non_triggered_conditions": [],
            "should_invalidate": should_invalidate,
        }),
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


@pytest.mark.asyncio
async def test_multi_pass_provider_runs_falsification_gate_when_policy_set() -> None:
    """With falsification_policy set, each claim is evaluated; falsified ids surfaced."""

    from extracted_reasoning_core.types import FalsificationPolicy

    llm = FakeLLMPort([
        _falsifiable_synthesis_response(),
        # Falsification call for c1 → falsified.
        _falsification_response(triggered=["renewal_signal_lost"], should_invalidate=True),
        # Falsification call for c2 → not falsified.
        _falsification_response(triggered=[], should_invalidate=False),
    ])
    provider = MultiPassCampaignReasoningProvider(
        ports=ReasoningPorts(llm=llm),
        config=MultiPassReasoningProviderConfig(
            falsification_policy=FalsificationPolicy(
                rules=({"id": "renewal_signal_lost"},),
                conservative=False,
            ),
        ),
    )

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert context is not None
    f = context.scope_summary["falsification"]
    assert f["evaluated_claim_count"] == 2
    assert f["falsified_count"] == 1
    assert f["falsified_claim_ids"] == ("c1",)
    assert f["drop_falsified"] is False
    # drop_falsified=False → falsified claim still in top_theses.
    top_ids = [t["claim"] for t in context.top_theses]
    assert "Renewal pricing drives churn." in top_ids
    # 1 run_reasoning + 2 check_falsification = 3 calls.
    assert len(llm.calls) == 3
    assert llm.calls[1]["metadata"]["reasoning_mode"] == "falsification_check"


@pytest.mark.asyncio
async def test_multi_pass_provider_drops_falsified_claims_when_drop_falsified_true() -> None:
    from extracted_reasoning_core.types import FalsificationPolicy

    llm = FakeLLMPort([
        _falsifiable_synthesis_response(),
        _falsification_response(triggered=["renewal_signal_lost"], should_invalidate=True),
        _falsification_response(triggered=[], should_invalidate=False),
    ])
    provider = MultiPassCampaignReasoningProvider(
        ports=ReasoningPorts(llm=llm),
        config=MultiPassReasoningProviderConfig(
            falsification_policy=FalsificationPolicy(
                rules=({"id": "renewal_signal_lost"},),
                conservative=False,
            ),
            drop_falsified=True,
        ),
    )

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert context is not None
    # c1 was falsified and drop_falsified=True → not in top_theses.
    top_claims = [t["claim"] for t in context.top_theses]
    assert "Renewal pricing drives churn." not in top_claims
    assert "Onboarding friction is secondary." in top_claims
    f = context.scope_summary["falsification"]
    assert f["falsified_claim_ids"] == ("c1",)
    assert f["drop_falsified"] is True


@pytest.mark.asyncio
async def test_multi_pass_provider_skips_falsification_when_policy_unset() -> None:
    """No falsification_policy → no falsification calls (existing D21b behavior preserved)."""

    llm = FakeLLMPort([_falsifiable_synthesis_response()])
    provider = MultiPassCampaignReasoningProvider(ports=ReasoningPorts(llm=llm))

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert context is not None
    # Only the run_reasoning call.
    assert len(llm.calls) == 1
    # No falsification key in scope_summary when policy is unset.
    assert "falsification" not in context.scope_summary


class _RaisingFalsificationLLM:
    """LLM port that returns a valid synthesis once, then raises on every
    subsequent call (simulating a network blip during falsification)."""

    def __init__(self, synthesis_response: Mapping[str, Any]) -> None:
        self._synthesis_response = synthesis_response
        self._first = True
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append({"metadata": dict(metadata or {})})
        if self._first:
            self._first = False
            return self._synthesis_response
        raise ConnectionError("simulated LLM outage")


@pytest.mark.asyncio
async def test_multi_pass_provider_runs_falsification_checks_in_parallel() -> None:
    """asyncio.gather is used so N claims = ~1x latency, not Nx."""

    from extracted_reasoning_core.types import FalsificationPolicy

    barrier = asyncio.Event()
    in_flight = 0
    max_in_flight = 0

    class TrackingLLM:
        def __init__(self, responses: list[Mapping[str, Any]]) -> None:
            self.responses = list(responses)
            self.calls: list[dict[str, Any]] = []

        async def complete(
            self,
            messages: Sequence[Mapping[str, Any]],
            *,
            max_tokens: int,
            temperature: float,
            metadata: Mapping[str, Any] | None = None,
        ) -> Mapping[str, Any]:
            nonlocal in_flight, max_in_flight
            self.calls.append({"metadata": dict(metadata or {})})
            mode = (metadata or {}).get("reasoning_mode")
            if mode == "falsification_check":
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                # Hold each falsification call until barrier fires; if
                # they were sequential this would deadlock on the first
                # call. Parallelism means all N enter before the barrier
                # is set.
                if in_flight >= 2:
                    barrier.set()
                await barrier.wait()
                in_flight -= 1
            return self.responses.pop(0)

    llm = TrackingLLM([
        _falsifiable_synthesis_response(),
        _falsification_response(triggered=[], should_invalidate=False),
        _falsification_response(triggered=[], should_invalidate=False),
    ])
    provider = MultiPassCampaignReasoningProvider(
        ports=ReasoningPorts(llm=llm),
        config=MultiPassReasoningProviderConfig(
            falsification_policy=FalsificationPolicy(rules=({"id": "x"},), conservative=False),
        ),
    )

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert context is not None
    # If the calls were sequential, max_in_flight would be 1 (each call
    # would complete before the next started). Parallelism puts both
    # falsification calls in flight before either completes.
    assert max_in_flight >= 2


@pytest.mark.asyncio
async def test_multi_pass_provider_fails_soft_on_check_falsification_error() -> None:
    """A check_falsification raise on one claim is captured in the trace; the rest still ship."""

    from extracted_reasoning_core.types import FalsificationPolicy

    llm = _RaisingFalsificationLLM(_falsifiable_synthesis_response())
    provider = MultiPassCampaignReasoningProvider(
        ports=ReasoningPorts(llm=llm),
        config=MultiPassReasoningProviderConfig(
            falsification_policy=FalsificationPolicy(rules=({"id": "x"},), conservative=False),
        ),
    )

    # Should NOT raise -- the connection error is captured per-claim.
    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert context is not None
    f = context.scope_summary["falsification"]
    # Both claims errored, neither falsified.
    assert f["falsified_count"] == 0
    assert all(t.get("error", "").startswith("ConnectionError") for t in f["traces"])
    # Synthesis claims still survive in the rendered context.
    assert len(context.top_theses) > 0


@pytest.mark.asyncio
async def test_multi_pass_provider_attaches_narrative_plan_when_pack_configured() -> None:
    """narrative_plan_pack triggers build_narrative_plan; result lands in canonical_reasoning."""

    from extracted_reasoning_core.types import ReasoningPack

    llm = FakeLLMPort([_falsifiable_synthesis_response()])
    provider = MultiPassCampaignReasoningProvider(
        ports=ReasoningPorts(llm=llm),
        config=MultiPassReasoningProviderConfig(
            narrative_plan_pack=ReasoningPack(
                name="content_ops_default",
                policies={"max_sections": 3},
            ),
        ),
    )

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert context is not None
    plan = context.canonical_reasoning.get("narrative_plan")
    assert plan is not None
    # Both synthesis claims are present in the plan output.
    assert len(plan["claims"]) == 2
    # Sections derived from the claims' "section" field (or default).
    assert len(plan["sections"]) >= 1
    # state_hints surfaces overall metadata.
    assert plan["state_hints"]["claim_count"] == 2


@pytest.mark.asyncio
async def test_multi_pass_provider_skips_narrative_plan_when_pack_unset() -> None:
    """No narrative_plan_pack → no narrative_plan in canonical_reasoning."""

    llm = FakeLLMPort([_falsifiable_synthesis_response()])
    provider = MultiPassCampaignReasoningProvider(ports=ReasoningPorts(llm=llm))

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert context is not None
    assert "narrative_plan" not in context.canonical_reasoning


@pytest.mark.asyncio
async def test_multi_pass_provider_narrative_plan_excludes_falsified_when_drop_falsified_true() -> None:
    """drop_falsified=True keeps top_theses and narrative_plan consistent."""

    from extracted_reasoning_core.types import FalsificationPolicy, ReasoningPack

    llm = FakeLLMPort([
        _falsifiable_synthesis_response(),
        _falsification_response(triggered=["renewal_signal_lost"], should_invalidate=True),
        _falsification_response(triggered=[], should_invalidate=False),
    ])
    provider = MultiPassCampaignReasoningProvider(
        ports=ReasoningPorts(llm=llm),
        config=MultiPassReasoningProviderConfig(
            falsification_policy=FalsificationPolicy(rules=({"id": "renewal_signal_lost"},), conservative=False),
            drop_falsified=True,
            narrative_plan_pack=ReasoningPack(name="default"),
        ),
    )

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert context is not None
    plan = context.canonical_reasoning["narrative_plan"]
    plan_claim_ids = [c.get("claim_id") for c in plan["claims"]]
    # c1 was falsified and drop_falsified=True → not in the narrative plan.
    assert "c1" not in plan_claim_ids
    assert "c2" in plan_claim_ids
