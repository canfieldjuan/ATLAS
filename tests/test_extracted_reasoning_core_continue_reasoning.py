from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import pytest

from extracted_reasoning_core.api import ConfigurationError, continue_reasoning
from extracted_reasoning_core.types import (
    EvidenceItem,
    ReasoningPack,
    ReasoningPorts,
    ReasoningResult,
)


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
        self.calls.append({
            "messages": tuple(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": dict(metadata or {}),
        })
        return self.responses.pop(0)


class FakeWitnessContextPort:
    def __init__(self, context: Mapping[str, Any]) -> None:
        self.context = dict(context)
        self.calls: list[dict[str, Any]] = []

    async def get_witness_context(
        self,
        reasoning_input: Any,
        *,
        depth: str,
        pack: Any | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append({
            "entity_id": reasoning_input.entity_id,
            "depth": depth,
            "pack": getattr(pack, "name", None),
        })
        return self.context


def _prior_raw_synthesis() -> dict[str, Any]:
    return {
        "summary": "Pricing pressure is the strongest supported wedge.",
        "claims": [
            {
                "claim": "Renewal pricing is creating displacement pressure.",
                "confidence": "high",
                "source_ids": ["r1"],
            },
            {
                "claim": "Onboarding friction is a secondary driver.",
                "confidence": "medium",
                "source_ids": ["r2"],
            },
        ],
        "confidence": 0.84,
    }


def _completed_state() -> dict[str, Any]:
    raw = _prior_raw_synthesis()
    return {
        "status": "completed",
        "succeeded": True,
        "entity_id": "acme",
        "entity_type": "vendor",
        "goal": "synthesize vendor pressure",
        "pack": "reasoning_synthesis",
        "pack_version": "1.0",
        "attempts_used": 1,
        "tokens_used": 15,
        "raw_synthesis": raw,
    }


def _expected_prior_hash(raw: Mapping[str, Any]) -> str:
    text = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.asyncio
async def test_continue_reasoning_happy_path_tracks_lineage() -> None:
    raw = _prior_raw_synthesis()
    state = _completed_state()
    state["raw_synthesis"] = raw

    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "summary": "Pricing pressure persists; onboarding friction is reinforced by the new ticket data.",
                "claims": [
                    {
                        "claim": "Renewal pricing remains the dominant displacement driver.",
                        "confidence": "high",
                        "source_ids": ["r1", "ticket-9"],
                        "revised_from": "Renewal pricing is creating displacement pressure.",
                    },
                    {
                        "claim": "Onboarding friction is now a primary driver, not secondary.",
                        "confidence": "high",
                        "source_ids": ["r2", "ticket-9"],
                        "revised_from": "Onboarding friction is a secondary driver.",
                    },
                ],
                "confidence": 0.88,
            }),
            "usage": {"input_tokens": 8, "output_tokens": 4},
        }
    ])
    witness = FakeWitnessContextPort({"witness_pack": [{"id": "w2"}]})

    event = {
        "event_type": "support_ticket_received",
        "evidence": [
            {
                "source_type": "ticket",
                "source_id": "ticket-9",
                "text": "Multiple onboarding pain points reported during renewal.",
            },
        ],
    }

    result = await continue_reasoning(
        state,
        event,
        ports=ReasoningPorts(llm=llm, witness_context=witness),
    )

    assert isinstance(result, ReasoningResult)
    assert result.summary.startswith("Pricing pressure persists")
    assert result.tier == "L2"
    assert result.state["status"] == "completed"
    assert result.state["stage"] == "continuation"
    assert result.state["generation"] == 2
    assert result.state["prior_synthesis_hash"] == _expected_prior_hash(raw)
    assert result.state["events_consumed"] == ("support_ticket_received",)
    assert result.state["prior_attempts_used"] == 1
    assert result.state["attempts_used"] == 1
    assert result.state["tokens_used"] == 12
    assert result.trace["event_type"] == "support_ticket_received"
    assert result.trace["prior_summary"] == raw["summary"]
    assert result.trace["falsification_hint_seen"] is False
    assert witness.calls == [{"entity_id": "acme", "depth": "L2", "pack": "reasoning_synthesis"}]
    assert llm.calls[0]["metadata"]["reasoning_mode"] == "multi_pass_continuation"
    assert llm.calls[0]["metadata"]["event_type"] == "support_ticket_received"


@pytest.mark.asyncio
async def test_continue_reasoning_preserves_revised_from_in_claims() -> None:
    state = _completed_state()
    contradicted_text = "Onboarding friction is a secondary driver."

    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "summary": "Onboarding friction is now the dominant driver.",
                "claims": [
                    {
                        "claim": "Onboarding friction now dominates churn risk.",
                        "confidence": 0.9,
                        "source_ids": ["ticket-12", "ticket-13"],
                        "revised_from": contradicted_text,
                    },
                ],
                "confidence": 0.9,
            }),
            "usage": {"input_tokens": 6, "output_tokens": 3},
        }
    ])

    event = {
        "event_type": "support_ticket_burst",
        "falsification_hint": True,
        "evidence": [
            {"source_type": "ticket", "source_id": "ticket-12", "text": "Customers stuck in onboarding."},
            {"source_type": "ticket", "source_id": "ticket-13", "text": "Five-day delay before activation."},
        ],
    }

    result = await continue_reasoning(
        state,
        event,
        ports=ReasoningPorts(llm=llm, witness_context=FakeWitnessContextPort({})),
    )

    revised = [c for c in result.claims if c.get("revised_from")]
    assert revised, "expected at least one claim with revised_from set"
    assert revised[0]["revised_from"] == contradicted_text
    assert result.trace["falsification_hint_seen"] is True


@pytest.mark.asyncio
async def test_continue_reasoning_retries_validation_failure_and_returns_final_failure_shape() -> None:
    state = _completed_state()

    llm = FakeLLMPort([
        {"response": json.dumps({"summary": "Missing claims."}), "usage": {"input_tokens": 3, "output_tokens": 2}},
        {"response": json.dumps({"summary": "Still missing claims."}), "usage": {"input_tokens": 4, "output_tokens": 2}},
    ])

    result = await continue_reasoning(
        state,
        {
            "event_type": "noop",
            "evidence": [],
        },
        ports=ReasoningPorts(
            llm=llm,
            witness_context=FakeWitnessContextPort({}),
        ),
    )

    assert len(llm.calls) == 2
    assert "previous response was rejected" in llm.calls[1]["messages"][-1]["content"]
    assert "missing_claims" in llm.calls[1]["messages"][-1]["content"]
    assert result.summary == "Reasoning continuation failed validation"
    assert result.claims == ()
    assert result.confidence == 0.0
    assert result.state["status"] == "failed"
    assert result.state["stage"] == "continuation_validation"
    assert result.state["attempts_used"] == 2
    assert result.state["generation"] == 2
    assert result.state["events_consumed"] == ("noop",)
    assert result.state["reasons"] == ("missing_claims",)
    assert result.state["error_text"] == "missing_claims"
    assert tuple(a["valid"] for a in result.trace["attempts"]) == (False, False)


@pytest.mark.asyncio
async def test_continue_reasoning_preserves_depth_across_continuations() -> None:
    """Regression: L3 chains must not silently downgrade to L2.

    Prior to the depth-in-state fix, _synthesis_success_result and
    _continuation_success_result both omitted the depth key, so
    continue_reasoning's fallback always resolved to "L2" regardless of
    the original tier.
    """

    state = _completed_state()
    state["depth"] = "L3"

    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "summary": "Continued at the original tier.",
                "claims": [{"claim": "Continuation preserves L3.", "confidence": 0.7, "source_ids": []}],
                "confidence": 0.7,
            }),
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
    ])

    result = await continue_reasoning(
        state,
        {"event_type": "tier_check", "evidence": []},
        ports=ReasoningPorts(llm=llm, witness_context=FakeWitnessContextPort({})),
    )

    assert result.tier == "L3"
    assert result.state["depth"] == "L3"
    assert llm.calls[0]["metadata"]["depth"] == "L3"


@pytest.mark.asyncio
async def test_continue_reasoning_missing_llm_port_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="ReasoningPorts.llm"):
        await continue_reasoning(
            _completed_state(),
            {"event_type": "noop", "evidence": []},
            ports=ReasoningPorts(witness_context=FakeWitnessContextPort({})),
        )


@pytest.mark.asyncio
async def test_continue_reasoning_rejects_non_completed_prior_state_without_llm_call() -> None:
    failed_prior = {
        "status": "failed",
        "succeeded": False,
        "stage": "validation",
        "error_text": "missing_claims",
        "reasons": ("missing_claims",),
        "attempts_used": 2,
        "tokens_used": 11,
    }
    llm = FakeLLMPort([])

    result = await continue_reasoning(
        failed_prior,
        {"event_type": "anything", "evidence": []},
        ports=ReasoningPorts(llm=llm, witness_context=FakeWitnessContextPort({})),
    )

    assert llm.calls == []
    assert result.state["status"] == "failed"
    assert result.state["stage"] == "continuation_input"
    assert "prior_state_not_completed" in result.state["reasons"]
    assert "prior_status=failed" in result.state["reasons"]
