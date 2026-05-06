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


@pytest.mark.asyncio
async def test_check_falsification_aggressive_derives_verdict_from_triggered_when_llm_omits_it() -> None:
    """When LLM returns no should_invalidate boolean, aggressive mode invalidates iff anything triggered."""

    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "triggered_conditions": ["renewal_signal_lost"],
                "non_triggered_conditions": ["competitor_won_segment"],
            }),
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    ])

    result = await check_falsification(
        _CLAIM,
        _EVIDENCE,
        policy=FalsificationPolicy(rules=_POLICY.rules, conservative=False),
        ports=ReasoningPorts(llm=llm),
    )

    assert result.triggered_conditions == ("renewal_signal_lost",)
    assert result.should_invalidate is True


@pytest.mark.asyncio
async def test_check_falsification_synthesizes_ids_for_anonymous_rules() -> None:
    """Rules without id/name/condition_id get synthesized rule_{i} ids that flow through."""

    anonymous_rules = (
        {"predicate": "fresh evidence shows renewal completed"},
        {"predicate": "fresh evidence shows displacement"},
    )
    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "triggered_conditions": ["rule_0"],
                "non_triggered_conditions": [],
                "should_invalidate": True,
            }),
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
    ])

    result = await check_falsification(
        _CLAIM,
        _EVIDENCE,
        policy=FalsificationPolicy(rules=anonymous_rules, conservative=False),
        ports=ReasoningPorts(llm=llm),
    )

    # Synthesized ids appear in the result coverage and the LLM payload.
    assert result.trace["rule_ids"] == ("rule_0", "rule_1")
    assert result.triggered_conditions == ("rule_0",)
    assert "rule_1" in result.non_triggered_conditions
    sent_payload = json.loads(llm.calls[0]["messages"][1]["content"])
    assert [r["id"] for r in sent_payload["rules"]] == ["rule_0", "rule_1"]


@pytest.mark.asyncio
async def test_check_falsification_id_synthesis_skips_explicit_rule_n_collisions() -> None:
    """If a host explicitly names a rule rule_<i>, synthesized ids skip past it."""

    mixed_rules = (
        {"id": "rule_0", "predicate": "host already used rule_0 explicitly"},
        {"predicate": "anonymous A"},
        {"id": "rule_2", "predicate": "host already used rule_2 explicitly"},
        {"predicate": "anonymous B"},
    )
    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "triggered_conditions": [],
                "non_triggered_conditions": [],
            }),
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ])

    result = await check_falsification(
        _CLAIM,
        _EVIDENCE,
        policy=FalsificationPolicy(rules=mixed_rules, conservative=False),
        ports=ReasoningPorts(llm=llm),
    )

    sent_ids = [r["id"] for r in json.loads(llm.calls[0]["messages"][1]["content"])["rules"]]
    # Anonymous rules get rule_1 and rule_3 — synthesizer skips past explicit rule_0 / rule_2.
    assert sent_ids == ["rule_0", "rule_1", "rule_2", "rule_3"]
    assert len(set(sent_ids)) == 4  # no duplicates
    assert result.trace["rule_ids"] == ("rule_0", "rule_1", "rule_2", "rule_3")


@pytest.mark.asyncio
async def test_check_falsification_honors_policy_max_tokens_and_temperature() -> None:
    llm = FakeLLMPort([
        {
            "response": json.dumps({"triggered_conditions": [], "non_triggered_conditions": []}),
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ])

    await check_falsification(
        _CLAIM,
        _EVIDENCE,
        policy=FalsificationPolicy(rules=_POLICY.rules, max_tokens=128, temperature=0.42),
        ports=ReasoningPorts(llm=llm),
    )

    assert llm.calls[0]["max_tokens"] == 128
    assert llm.calls[0]["temperature"] == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_check_falsification_emits_completed_event_and_trace_span() -> None:
    """Observability parity with run_reasoning / continue_reasoning."""

    events: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []

    class FakeEventSink:
        async def emit(self, name: str, source: str, payload: Mapping[str, Any], **kwargs: Any) -> None:
            events.append({"name": name, "source": source, "payload": dict(payload), "kwargs": dict(kwargs)})

    class FakeTraceSink:
        def start_span(self, name: str, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
            span = {"name": name, "start_metadata": dict(metadata or {}), "end_metadata": None, "status": None}
            spans.append(span)
            return span

        def end_span(self, span: dict[str, Any], *, status: str, metadata: Mapping[str, Any] | None = None) -> None:
            span["status"] = status
            span["end_metadata"] = dict(metadata or {})

    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "triggered_conditions": ["renewal_signal_lost"],
                "non_triggered_conditions": ["competitor_won_segment"],
                "should_invalidate": True,
            }),
            "usage": {"input_tokens": 4, "output_tokens": 2},
        }
    ])

    await check_falsification(
        _CLAIM,
        _EVIDENCE,
        policy=FalsificationPolicy(rules=_POLICY.rules, conservative=False),
        ports=ReasoningPorts(llm=llm, event_sink=FakeEventSink(), trace_sink=FakeTraceSink()),
    )

    assert events[0]["name"] == "reasoning.falsification.completed"
    assert events[0]["payload"]["triggered_count"] == 1
    assert events[0]["payload"]["should_invalidate"] is True
    assert spans[0]["name"] == "extracted_reasoning_core.check_falsification"
    assert spans[0]["status"] == "ok"
    assert spans[0]["end_metadata"]["triggered_count"] == 1


@pytest.mark.asyncio
async def test_check_falsification_emits_parse_failed_event_on_malformed_response() -> None:
    events: list[dict[str, Any]] = []

    class FakeEventSink:
        async def emit(self, name: str, source: str, payload: Mapping[str, Any], **kwargs: Any) -> None:
            events.append({"name": name})

    llm = FakeLLMPort([
        {"response": "this is not json", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ])

    await check_falsification(
        _CLAIM,
        _EVIDENCE,
        policy=_POLICY,
        ports=ReasoningPorts(llm=llm, event_sink=FakeEventSink()),
    )

    assert events[0]["name"] == "reasoning.falsification.parse_failed"
