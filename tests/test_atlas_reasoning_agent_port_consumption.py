"""Verify atlas's ``ReasoningAgentGraph`` consumes the TraceSink port.

PR-C4d migrates ``atlas_brain.reasoning.agent.process_event`` from a
direct ``tracer.start_span/end_span`` call to ``self.ports.trace_sink.*``
on an injected ``ReasoningPorts`` bundle. These tests pin:

- ``ReasoningAgentGraph(ports=fake)`` actually uses the injected
  TraceSink (no fall-through to the default-builder path)
- ``process_event`` opens exactly one span named ``reasoning.process``
  and packs the lifted typed kwargs (``model_name`` /
  ``model_provider`` / ``session_id`` / ``input_tokens`` /
  ``output_tokens`` / ``input_data`` / ``output_data``) into the
  Port's free-form metadata bag, leaving ``business`` and ``reasoning``
  as nested metadata for atlas tracer's special handling
- Both success and failure paths close the span exactly once
  with the right Port-side ``status`` value (``ok`` / ``error``)

Atlas's heavy chains (``atlas_brain.services``, ``atlas_brain.pipelines``,
``atlas_brain.config``) and the actual reasoning graph are all stubbed
in ``sys.modules`` so the test runs in standalone CI without the full
atlas dep stack. Only the wiring layer is exercised; the FTL tracer and
real LLM are out of scope -- their adapter translation is covered by
``test_atlas_reasoning_port_adapters.py``.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Mapping
from uuid import UUID

import pytest


# ----------------------------------------------------------------------
# Module stubs for the heavy atlas chain
# ----------------------------------------------------------------------


_STUBBED_MODULE_NAMES = (
    "atlas_brain.services",
    "atlas_brain.services.tracing",
    "atlas_brain.pipelines",
    "atlas_brain.pipelines.llm",
    "atlas_brain.config",
)


def _build_atlas_stubs() -> dict[str, types.ModuleType]:
    """Construct stub modules matching the interface ``agent.process_event``
    lazy-imports. Returns a name->module dict for the fixture to install."""
    services_pkg = types.ModuleType("atlas_brain.services")
    services_pkg.__path__ = []  # mark as package for submodule imports

    tracing_mod = types.ModuleType("atlas_brain.services.tracing")
    tracing_mod.tracer = object()  # never used; default-builder path is bypassed
    tracing_mod.build_business_trace_context = lambda **kwargs: {
        k: v for k, v in kwargs.items() if v is not None
    }
    tracing_mod.build_reasoning_trace_context = lambda **kwargs: {
        k: v for k, v in kwargs.items() if v is not None
    }

    pipelines_pkg = types.ModuleType("atlas_brain.pipelines")
    pipelines_pkg.__path__ = []

    pipelines_llm_mod = types.ModuleType("atlas_brain.pipelines.llm")

    class _StubLLM:
        model = "stub-model"
        name = "stub-provider"

    pipelines_llm_mod.get_pipeline_llm = lambda **kwargs: _StubLLM()

    config_mod = types.ModuleType("atlas_brain.config")

    class _StubReasoning:
        graph_reasoning_workload = "test"
        graph_openrouter_model = ""

    class _StubSettings:
        reasoning = _StubReasoning()

    config_mod.settings = _StubSettings()

    return {
        "atlas_brain.services": services_pkg,
        "atlas_brain.services.tracing": tracing_mod,
        "atlas_brain.pipelines": pipelines_pkg,
        "atlas_brain.pipelines.llm": pipelines_llm_mod,
        "atlas_brain.config": config_mod,
    }


@pytest.fixture(scope="module", autouse=True)
def _stub_atlas_chain():
    """Install stubs for the test module, restore originals on teardown.

    Without restoration, ``sys.modules`` would carry the stubs into
    sibling test files in the same pytest invocation -- e.g. a future
    ``tests/test_tracing_context.py`` that imports the real
    ``atlas_brain.services.tracing.build_business_trace_context``
    would see this file's stub instead. Save the prior entries (or
    record which names were absent), install the stubs, and on
    teardown either put the originals back or remove the stub entries.
    """
    saved: dict[str, types.ModuleType | None] = {
        name: sys.modules.get(name) for name in _STUBBED_MODULE_NAMES
    }

    stubs = _build_atlas_stubs()
    for name, mod in stubs.items():
        sys.modules[name] = mod

    try:
        yield
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------


class _RecordingTraceSink:
    """Minimal TraceSink fake that records calls in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def start_span(
        self,
        name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        record = {"span": object(), "name": name, "metadata": dict(metadata) if metadata else None}
        self.events.append(("start", record))
        return record["span"]

    def end_span(
        self,
        span: Any,
        *,
        status: str = "ok",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append(
            (
                "end",
                {
                    "span": span,
                    "status": status,
                    "metadata": dict(metadata) if metadata else None,
                },
            )
        )


class _StubEventSink:
    async def emit(
        self,
        event_type: str,
        source: str,
        payload: Mapping[str, Any],
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> str:
        return "stub-id"


def _build_fake_ports(trace_sink: _RecordingTraceSink) -> Any:
    from extracted_reasoning_core.types import ReasoningPorts

    return ReasoningPorts(
        event_sink=_StubEventSink(),
        trace_sink=trace_sink,
    )


def _make_event() -> Any:
    from atlas_brain.reasoning.events import AtlasEvent

    return AtlasEvent(
        id=UUID("11111111-2222-3333-4444-555555555555"),
        event_type="vendor.archetype_assigned",
        source="reasoning_core",
        entity_type="vendor",
        entity_id="acme",
        payload={"vendor": "Acme"},
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_event_success_path_uses_injected_trace_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas_brain.reasoning import agent as agent_mod

    # Stub the actual reasoning graph -- we're testing the wiring layer
    # around it, not the graph itself.
    graph_mod = types.ModuleType("atlas_brain.reasoning.graph")

    async def _fake_run_reasoning_graph(state: dict) -> dict:
        return {
            **state,
            "triage_priority": "high",
            "needs_reasoning": True,
            "queued": False,
            "connections_found": ["connection-1"],
            "planned_actions": [{"tool": "noop"}],
            "action_results": [{"ok": True}],
            "notification_sent": True,
            "summary": "all good",
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "triage_reasoning": "high priority",
            "rationale": "...",
            "reasoning_output": "raw output",
        }

    graph_mod.run_reasoning_graph = _fake_run_reasoning_graph
    monkeypatch.setitem(sys.modules, "atlas_brain.reasoning.graph", graph_mod)

    trace_sink = _RecordingTraceSink()
    graph = agent_mod.ReasoningAgentGraph(ports=_build_fake_ports(trace_sink))
    result = await graph.process_event(_make_event())

    # Exactly one start + one end, in order.
    assert [e[0] for e in trace_sink.events] == ["start", "end"]

    start = trace_sink.events[0][1]
    assert start["name"] == "reasoning.process"
    # Lifted-via-metadata typed kwargs the adapter will route to atlas
    # tracer.
    assert start["metadata"]["model_name"] == "stub-model"
    assert start["metadata"]["model_provider"] == "stub-provider"
    assert start["metadata"]["session_id"] == "11111111-2222-3333-4444-555555555555"
    # Business context envelope stays nested in metadata (NOT lifted).
    assert "business" in start["metadata"]
    assert start["metadata"]["business"]["workflow"] == "reasoning_agent"
    assert start["metadata"]["business"]["event_type"] == "vendor.archetype_assigned"

    end = trace_sink.events[1][1]
    assert end["span"] is start["span"] if isinstance(start, dict) else True
    assert end["status"] == "ok"
    assert end["metadata"]["input_tokens"] == 100
    assert end["metadata"]["output_tokens"] == 50
    assert end["metadata"]["input_data"]["event_type"] == "vendor.archetype_assigned"
    assert end["metadata"]["output_data"]["status"] == "completed"
    # Reasoning context stays nested (NOT lifted) so atlas's
    # _derive_reasoning_text promotion still fires. The exact shape
    # under "reasoning" comes from build_reasoning_trace_context (atlas
    # tracing module) -- the stub here passes the raw kwargs through,
    # so we assert against the kwarg names rather than atlas's
    # post-transform shape.
    assert "reasoning" in end["metadata"]
    assert end["metadata"]["reasoning"]["triage_reasoning"] == "high priority"
    assert end["metadata"]["reasoning"]["raw_reasoning"] == "raw output"

    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_process_event_failure_path_closes_span_with_error_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas_brain.reasoning import agent as agent_mod

    graph_mod = types.ModuleType("atlas_brain.reasoning.graph")

    async def _failing_run_reasoning_graph(state: dict) -> dict:
        raise RuntimeError("graph blew up")

    graph_mod.run_reasoning_graph = _failing_run_reasoning_graph
    monkeypatch.setitem(sys.modules, "atlas_brain.reasoning.graph", graph_mod)

    trace_sink = _RecordingTraceSink()
    graph = agent_mod.ReasoningAgentGraph(ports=_build_fake_ports(trace_sink))
    result = await graph.process_event(_make_event())

    assert [e[0] for e in trace_sink.events] == ["start", "end"]
    end = trace_sink.events[1][1]
    assert end["status"] == "error"
    assert end["metadata"]["error_message"] == "reasoning graph failed"
    assert end["metadata"]["error_type"] == "ReasoningGraphError"
    assert end["metadata"]["input_data"]["event_type"] == "vendor.archetype_assigned"

    assert result["status"] == "error"


def test_explicit_ports_bypass_default_builder() -> None:
    # Construct with an explicit ports bundle and confirm the
    # default-builder path is never reached. (If it were, the import
    # of atlas_brain.services.tracing.tracer for the default-build
    # would happen on first .ports access -- our stub installs that
    # symbol, but the point is structural: explicit ports take
    # precedence and the default-builder doesn't run.)
    from atlas_brain.reasoning.agent import ReasoningAgentGraph

    trace_sink = _RecordingTraceSink()
    fake_ports = _build_fake_ports(trace_sink)
    graph = ReasoningAgentGraph(ports=fake_ports)

    assert graph.ports is fake_ports
