"""Unit tests for ``extracted_reasoning_core.graph_nodes``.

PR-C4e2 promoted the LLM-driven nodes ``node_triage`` and
``node_synthesize`` into core. They run against the
``LLMClient`` Protocol instead of atlas's concrete LLM service, so
these tests exercise both the happy path (LLM available, JSON parses)
and every fallback the original atlas code carried (LLM unavailable,
LLM raises, malformed JSON).

A fake ``LLMClient`` (matches the Port shape exactly) records the
calls so we can pin the metadata-packed kwargs the adapter relies on
(``json_mode``, ``response_format``).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from extracted_reasoning_core.graph_nodes import node_synthesize, node_triage


_TRIAGE_PROMPT = "You are a triage classifier."
_SYNTH_PROMPT = "You are a notification summarizer."


class _RecordingLLMClient:
    """Fake LLMClient that returns a canned response and records calls."""

    def __init__(
        self,
        response_text: str,
        usage: dict[str, int] | None = None,
    ) -> None:
        self._response_text = response_text
        self._usage = usage or {}
        self.calls: list[dict[str, Any]] = []
        self.exc: Exception | None = None

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "metadata": dict(metadata) if metadata else None,
            }
        )
        if self.exc is not None:
            raise self.exc
        return {"response": self._response_text, "usage": dict(self._usage)}


# ----------------------------------------------------------------------
# node_triage
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_triage_writes_parsed_priority_and_accumulates_tokens() -> None:
    fake = _RecordingLLMClient(
        response_text='{"priority": "high", "needs_reasoning": true, "reasoning": "vendor at risk"}',
        usage={"input_tokens": 12, "output_tokens": 7},
    )
    state: dict = {
        "event_type": "vendor.archetype_assigned",
        "source": "reasoning_core",
        "entity_type": "vendor",
        "entity_id": "acme",
        "payload": {"vendor": "Acme"},
        "total_input_tokens": 5,
        "total_output_tokens": 3,
    }

    await node_triage(state, fake, triage_system_prompt=_TRIAGE_PROMPT)

    assert state["triage_priority"] == "high"
    assert state["needs_reasoning"] is True
    assert state["triage_reasoning"] == "vendor at risk"
    # Token totals accumulate (didn't overwrite the prior 5/3).
    assert state["total_input_tokens"] == 17
    assert state["total_output_tokens"] == 10

    assert len(fake.calls) == 1
    call = fake.calls[0]
    # System prompt is the first message.
    assert call["messages"][0] == {"role": "system", "content": _TRIAGE_PROMPT}
    # User message carries the event description.
    assert "vendor.archetype_assigned" in call["messages"][1]["content"]
    # JSON-mode flags reach the LLMClient via metadata so the adapter
    # can pass response_format through to the underlying chat call.
    assert call["metadata"]["json_mode"] is True
    assert call["metadata"]["response_format"] == {"type": "json_object"}
    # Triage uses a low temperature.
    assert call["temperature"] == 0.1


@pytest.mark.asyncio
async def test_node_triage_falls_back_when_llm_is_none() -> None:
    state: dict = {"event_type": "system.tick"}
    await node_triage(state, None, triage_system_prompt=_TRIAGE_PROMPT)
    assert state["triage_priority"] == "medium"
    assert state["needs_reasoning"] is True
    assert state["triage_reasoning"] == "Triage LLM unavailable, defaulting to reason"


@pytest.mark.asyncio
async def test_node_triage_falls_back_when_llm_raises() -> None:
    fake = _RecordingLLMClient(response_text="")
    fake.exc = RuntimeError("upstream timeout")
    state: dict = {"event_type": "system.tick"}
    await node_triage(state, fake, triage_system_prompt=_TRIAGE_PROMPT)
    assert state["triage_priority"] == "medium"
    assert state["needs_reasoning"] is True
    assert "Triage parse error" in state["triage_reasoning"]


@pytest.mark.asyncio
async def test_node_triage_default_priority_when_parsed_keys_missing() -> None:
    # If the model returns valid JSON ``{}`` (no ``priority`` field),
    # fall back to ``medium`` priority + needs_reasoning via the
    # parsed-key defaults. parse_ok is True here -- the response
    # parsed cleanly, just had no fields.
    fake = _RecordingLLMClient(response_text="{}")
    state: dict = {"event_type": "system.tick"}
    await node_triage(state, fake, triage_system_prompt=_TRIAGE_PROMPT)
    assert state["triage_priority"] == "medium"
    assert state["needs_reasoning"] is True
    assert state["triage_reasoning"] == ""


@pytest.mark.asyncio
async def test_node_triage_falls_back_on_unparseable_output() -> None:
    # The model returned a non-JSON response. complete_with_json
    # swallows the JSONDecodeError and signals via parse_ok=False;
    # node_triage must apply the same "default to reason on parse
    # failure" fallback atlas's pre-extraction code did. Without
    # this branch, malformed triage output would silently leak
    # through with empty triage_reasoning.
    fake = _RecordingLLMClient(response_text="not JSON at all")
    state: dict = {"event_type": "system.tick"}
    await node_triage(state, fake, triage_system_prompt=_TRIAGE_PROMPT)
    assert state["triage_priority"] == "medium"
    assert state["needs_reasoning"] is True
    assert "Triage parse error" in state["triage_reasoning"]


@pytest.mark.asyncio
async def test_node_triage_threads_timeout_through_metadata() -> None:
    fake = _RecordingLLMClient(response_text='{"priority": "low"}')
    state: dict = {"event_type": "system.tick"}
    await node_triage(
        state, fake,
        triage_system_prompt=_TRIAGE_PROMPT,
        timeout=45.0,
    )
    # Timeout reaches the LLMClient via metadata so the adapter can
    # apply the deadline (asyncio.wait_for + chat kwarg).
    assert fake.calls[0]["metadata"]["timeout"] == 45.0


@pytest.mark.asyncio
async def test_node_synthesize_threads_timeout_through_metadata() -> None:
    fake = _RecordingLLMClient(response_text="A summary line.")
    state: dict = {
        "should_notify": True,
        "event_type": "vendor.churn",
        "action_results": [],
        "rationale": "...",
        "connections_found": [],
    }
    await node_synthesize(
        state, fake,
        synthesis_system_prompt=_SYNTH_PROMPT,
        timeout=30.0,
    )
    assert fake.calls[0]["metadata"]["timeout"] == 30.0


# ----------------------------------------------------------------------
# node_synthesize
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_synthesize_skips_when_should_notify_false() -> None:
    state: dict = {"should_notify": False}
    fake = _RecordingLLMClient(response_text="should not be called")
    await node_synthesize(state, fake, synthesis_system_prompt=_SYNTH_PROMPT)
    assert state["summary"] == ""
    # No LLM call when notification gate is closed.
    assert fake.calls == []


@pytest.mark.asyncio
async def test_node_synthesize_uses_deterministic_fallback_when_llm_none() -> None:
    state: dict = {
        "should_notify": True,
        "event_type": "vendor.churn",
        "action_results": [{"tool": "log_interaction", "success": True}],
        "rationale": "Vendor signaling churn risk.",
        "connections_found": [],
    }
    await node_synthesize(state, None, synthesis_system_prompt=_SYNTH_PROMPT)
    # Fallback wraps rationale text + appends action phrase.
    assert "Vendor signaling churn risk" in state["summary"]
    assert "log interaction" in state["summary"]


@pytest.mark.asyncio
async def test_node_synthesize_writes_sanitized_summary_and_tokens() -> None:
    fake = _RecordingLLMClient(
        response_text="Vendor X is at risk. Outreach scheduled for Monday.",
        usage={"input_tokens": 30, "output_tokens": 12},
    )
    state: dict = {
        "should_notify": True,
        "event_type": "vendor.churn",
        "action_results": [],
        "rationale": "...",
        "connections_found": [],
        "total_input_tokens": 100,
        "total_output_tokens": 50,
    }
    await node_synthesize(state, fake, synthesis_system_prompt=_SYNTH_PROMPT)

    assert "Vendor X is at risk" in state["summary"]
    assert "Outreach scheduled" in state["summary"]
    assert state["total_input_tokens"] == 130
    assert state["total_output_tokens"] == 62

    assert len(fake.calls) == 1
    call = fake.calls[0]
    # Synthesis is plain text, not JSON -- no json_mode metadata.
    assert call["metadata"] is None
    assert call["max_tokens"] == 256
    assert call["temperature"] == 0.3


@pytest.mark.asyncio
async def test_node_synthesize_falls_back_on_llm_failure() -> None:
    fake = _RecordingLLMClient(response_text="")
    fake.exc = RuntimeError("synthesis failed")
    state: dict = {
        "should_notify": True,
        "event_type": "vendor.churn",
        "action_results": [{"tool": "log_interaction", "success": True}],
        "rationale": "Vendor X showing churn risk.",
        "connections_found": [],
    }
    await node_synthesize(state, fake, synthesis_system_prompt=_SYNTH_PROMPT)
    # LLM raised; deterministic fallback fires.
    assert "Vendor X showing churn risk" in state["summary"]


@pytest.mark.asyncio
async def test_node_synthesize_falls_back_on_empty_response() -> None:
    fake = _RecordingLLMClient(response_text="", usage={"input_tokens": 5, "output_tokens": 0})
    state: dict = {
        "should_notify": True,
        "event_type": "vendor.churn",
        "action_results": [],
        "rationale": "",
        "connections_found": ["Vendor signal: pricing pressure"],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }
    await node_synthesize(state, fake, synthesis_system_prompt=_SYNTH_PROMPT)
    # Empty response → fallback uses connections.
    assert "Vendor signal: pricing pressure" in state["summary"]
    # Tokens still accumulate even when response is empty.
    assert state["total_input_tokens"] == 5
