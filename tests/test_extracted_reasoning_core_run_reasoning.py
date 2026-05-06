from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pytest

from extracted_reasoning_core.api import ConfigurationError, run_reasoning
from extracted_reasoning_core.types import (
    EvidenceItem,
    ReasoningInput,
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


def _input() -> ReasoningInput:
    return ReasoningInput(
        entity_id="acme",
        entity_type="vendor",
        goal="synthesize vendor pressure",
        evidence=(
            EvidenceItem(
                source_type="review",
                source_id="r1",
                text="Buyers mention renewal pricing pressure.",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_run_reasoning_happy_path_populates_result_with_ports() -> None:
    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "summary": "Pricing pressure is the strongest supported wedge.",
                "claims": [
                    {
                        "claim": "Renewal pricing is creating displacement pressure.",
                        "confidence": "high",
                        "source_ids": ["r1"],
                    }
                ],
                "confidence": 0.84,
            }),
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    ])
    witness = FakeWitnessContextPort({"witness_pack": [{"id": "w1"}]})

    result = await run_reasoning(
        _input(),
        depth="L2",
        pack=ReasoningPack(
            name="reasoning_synthesis",
            prompts={"reasoning_synthesis": "Return JSON."},
            policies={"max_attempts": 2, "max_tokens": 1000, "temperature": 0.1},
        ),
        ports=ReasoningPorts(llm=llm, witness_context=witness),
    )

    assert isinstance(result, ReasoningResult)
    assert result.summary == "Pricing pressure is the strongest supported wedge."
    assert result.claims[0]["claim"] == "Renewal pricing is creating displacement pressure."
    assert result.confidence == pytest.approx(0.84)
    assert result.tier == "L2"
    assert result.state["status"] == "completed"
    assert result.state["attempts_used"] == 1
    assert result.state["tokens_used"] == 15
    assert result.trace["witness_context_present"] is True
    assert witness.calls == [{"entity_id": "acme", "depth": "L2", "pack": "reasoning_synthesis"}]
    assert llm.calls[0]["metadata"]["reasoning_mode"] == "single_pass_synthesis"


@pytest.mark.asyncio
async def test_run_reasoning_retries_validation_failure_and_returns_final_failure_shape() -> None:
    llm = FakeLLMPort([
        {
            "response": json.dumps({"summary": "Missing claims."}),
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
        {
            "response": json.dumps({"summary": "Still missing claims."}),
            "usage": {"input_tokens": 4, "output_tokens": 2},
        },
    ])

    result = await run_reasoning(
        _input(),
        pack=ReasoningPack(
            name="reasoning_synthesis",
            prompts={"reasoning_synthesis": "Return JSON."},
            policies={"max_attempts": 2},
        ),
        ports=ReasoningPorts(
            llm=llm,
            witness_context=FakeWitnessContextPort({"witness_pack": []}),
        ),
    )

    assert len(llm.calls) == 2
    assert "previous response was rejected" in llm.calls[1]["messages"][-1]["content"]
    assert "missing_claims" in llm.calls[1]["messages"][-1]["content"]
    assert result.summary == "Reasoning synthesis failed validation"
    assert result.claims == ()
    assert result.confidence == 0.0
    assert result.state == {
        "status": "failed",
        "succeeded": False,
        "stage": "validation",
        "error_text": "missing_claims",
        "reasons": ("missing_claims",),
        "attempts_used": 2,
        "tokens_used": 11,
    }
    assert tuple(a["valid"] for a in result.trace["attempts"]) == (False, False)
    assert tuple(a["errors"] for a in result.trace["attempts"]) == (("missing_claims",), ("missing_claims",))


@pytest.mark.asyncio
async def test_run_reasoning_missing_llm_port_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="ReasoningPorts.llm"):
        await run_reasoning(
            _input(),
            ports=ReasoningPorts(witness_context=FakeWitnessContextPort({})),
        )
