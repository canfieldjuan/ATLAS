from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pytest

from extracted_reasoning_core.api import ConfigurationError, check_falsification
from extracted_reasoning_core.types import (
    EvidenceItem,
    FalsificationPolicy,
    FalsificationResult,
    ReasoningPorts,
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


_CLAIM = {
    "claim_id": "c1",
    "claim": "Renewal pricing is the dominant displacement driver.",
    "confidence": 0.84,
    "source_ids": ["r1"],
}
_EVIDENCE = (
    EvidenceItem(source_type="ticket", source_id="t-9", text="Customer renewed without protest."),
)
_POLICY = FalsificationPolicy(
    rules=(
        {"id": "renewal_signal_lost", "predicate": "fresh evidence shows renewal completed"},
        {"id": "competitor_won_segment", "predicate": "fresh evidence shows displacement"},
    ),
    conservative=True,
)


@pytest.mark.asyncio
async def test_check_falsification_happy_path_returns_structured_result() -> None:
    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "triggered_conditions": ["renewal_signal_lost"],
                "non_triggered_conditions": ["competitor_won_segment"],
                "should_invalidate": True,
            }),
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
    ])

    result = await check_falsification(
        _CLAIM,
        _EVIDENCE,
        policy=FalsificationPolicy(rules=_POLICY.rules, conservative=False),
        ports=ReasoningPorts(llm=llm),
    )

    assert isinstance(result, FalsificationResult)
    assert result.triggered_conditions == ("renewal_signal_lost",)
    assert "competitor_won_segment" in result.non_triggered_conditions
    assert result.should_invalidate is True
    assert result.trace["rule_ids"] == ("renewal_signal_lost", "competitor_won_segment")
    assert result.trace["claim_id"] == "c1"
    assert result.trace["tokens_used"] == 8
    assert llm.calls[0]["metadata"]["reasoning_mode"] == "falsification_check"


@pytest.mark.asyncio
async def test_check_falsification_conservative_blocks_invalidation_when_no_triggers() -> None:
    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "triggered_conditions": [],
                "non_triggered_conditions": ["renewal_signal_lost", "competitor_won_segment"],
                "should_invalidate": True,
            }),
            "usage": {"input_tokens": 4, "output_tokens": 2},
        }
    ])

    result = await check_falsification(
        _CLAIM,
        _EVIDENCE,
        policy=_POLICY,
        ports=ReasoningPorts(llm=llm),
    )

    assert result.triggered_conditions == ()
    assert result.should_invalidate is False
    assert result.trace["conservative"] is True


@pytest.mark.asyncio
async def test_check_falsification_parse_failure_is_conservative_no_invalidation() -> None:
    llm = FakeLLMPort([
        {"response": "this is not json", "usage": {"input_tokens": 2, "output_tokens": 1}},
    ])

    result = await check_falsification(
        _CLAIM,
        _EVIDENCE,
        policy=_POLICY,
        ports=ReasoningPorts(llm=llm),
    )

    assert result.triggered_conditions == ()
    assert result.should_invalidate is False
    assert result.trace["parse_failed"] is True
    # All declared rules show as non-triggered when the LLM returns nothing parseable.
    assert set(result.non_triggered_conditions) == {"renewal_signal_lost", "competitor_won_segment"}


@pytest.mark.asyncio
async def test_check_falsification_missing_llm_port_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="ReasoningPorts.llm"):
        await check_falsification(_CLAIM, _EVIDENCE, policy=_POLICY)
