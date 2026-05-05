"""Verify ``run_reflection`` opens + closes a span via the TraceSink port.

Atlas's cron-driven reflection cycle had zero tracing prior to this
slice -- if the LLM call to Claude stalled or the pattern detectors
failed, nothing surfaced in FTL. PR-C4d2 wraps the cycle in a
``reasoning.reflection`` span so it's visible alongside the reactive
``reasoning.process`` spans that ``ReasoningAgentGraph`` opens.

These tests pin:

- ``run_reflection(ports=fake)`` actually uses the injected TraceSink
  (no fall-through to the default-builder path)
- One ``reasoning.reflection`` span opens at the start and closes
  exactly once with the cycle result dict packed as metadata + status
  ``"ok"``
- An exception inside the cycle still closes the span -- with status
  ``"error"`` and ``error_message`` / ``error_type`` -- and re-raises

Atlas's heavy chains (services / pipelines / config) plus the
reflection-internal helpers (patterns detectors, graph helpers, the
LLM client) are stubbed in ``sys.modules`` so the test runs in
standalone CI without the full atlas dep stack. Stubs are restored on
teardown (lesson from PR-C4d's Codex review -- otherwise sibling test
files in the same pytest invocation would see this file's stubs).
"""

from __future__ import annotations

import sys
import types
from typing import Any, Mapping

import pytest


# ----------------------------------------------------------------------
# sys.modules stubs (installed/restored by the autouse fixture)
# ----------------------------------------------------------------------


_STUBBED_MODULE_NAMES = (
    "atlas_brain.services",
    "atlas_brain.services.tracing",
    "atlas_brain.pipelines",
    "atlas_brain.pipelines.llm",
    "atlas_brain.config",
    "atlas_brain.reasoning.patterns",
    "atlas_brain.reasoning.graph",
    "atlas_brain.reasoning.graph_prompts",
)


def _build_atlas_stubs(
    pattern_findings: list[dict[str, Any]],
) -> dict[str, types.ModuleType]:
    services_pkg = types.ModuleType("atlas_brain.services")
    services_pkg.__path__ = []
    tracing_mod = types.ModuleType("atlas_brain.services.tracing")
    tracing_mod.tracer = object()  # never reached; explicit ports bypass default

    pipelines_pkg = types.ModuleType("atlas_brain.pipelines")
    pipelines_pkg.__path__ = []
    pipelines_llm_mod = types.ModuleType("atlas_brain.pipelines.llm")
    # Default: LLM is unavailable, so reflection takes the no-LLM path
    # and notifies on rule-based findings. Tests can override per-test.
    pipelines_llm_mod.get_pipeline_llm = lambda **kwargs: None

    config_mod = types.ModuleType("atlas_brain.config")

    class _StubReasoning:
        graph_synthesis_workload = "test"
        max_tokens = 1024
        temperature = 0.3

    class _StubAlerts:
        ntfy_enabled = False
        ntfy_url = ""
        ntfy_topic = ""

    class _StubSettings:
        reasoning = _StubReasoning()
        alerts = _StubAlerts()

    config_mod.settings = _StubSettings()

    patterns_mod = types.ModuleType("atlas_brain.reasoning.patterns")

    async def _fake_run_all_pattern_detectors() -> list[dict[str, Any]]:
        return pattern_findings

    patterns_mod.run_all_pattern_detectors = _fake_run_all_pattern_detectors

    graph_mod = types.ModuleType("atlas_brain.reasoning.graph")

    async def _fake_llm_generate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"response": '{"findings": []}', "usage": {}}

    def _fake_parse_llm_json(text: str) -> dict[str, Any]:
        import json as _json
        return _json.loads(text)

    graph_mod._llm_generate = _fake_llm_generate
    graph_mod._parse_llm_json = _fake_parse_llm_json

    graph_prompts_mod = types.ModuleType("atlas_brain.reasoning.graph_prompts")
    graph_prompts_mod.REFLECTION_SYSTEM = "system"

    return {
        "atlas_brain.services": services_pkg,
        "atlas_brain.services.tracing": tracing_mod,
        "atlas_brain.pipelines": pipelines_pkg,
        "atlas_brain.pipelines.llm": pipelines_llm_mod,
        "atlas_brain.config": config_mod,
        "atlas_brain.reasoning.patterns": patterns_mod,
        "atlas_brain.reasoning.graph": graph_mod,
        "atlas_brain.reasoning.graph_prompts": graph_prompts_mod,
    }


@pytest.fixture
def install_stubs():
    """Per-test fixture so each test can choose its own pattern findings.

    Returns a callable that takes a ``pattern_findings`` list, installs
    stubs in ``sys.modules`` (saving prior entries first), and registers
    teardown to restore the originals.
    """
    saved: dict[str, types.ModuleType | None] = {}

    def _install(pattern_findings: list[dict[str, Any]]) -> None:
        for name in _STUBBED_MODULE_NAMES:
            saved[name] = sys.modules.get(name)
        for name, mod in _build_atlas_stubs(pattern_findings).items():
            sys.modules[name] = mod

    yield _install

    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------


class _RecordingTraceSink:
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


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_reflection_opens_and_closes_span_with_no_findings(
    install_stubs,
) -> None:
    install_stubs([])  # rule-based detectors return nothing
    from atlas_brain.reasoning.reflection import run_reflection

    trace_sink = _RecordingTraceSink()
    result = await run_reflection(ports=_build_fake_ports(trace_sink))

    assert [e[0] for e in trace_sink.events] == ["start", "end"]
    start = trace_sink.events[0][1]
    end = trace_sink.events[1][1]
    assert start["name"] == "reasoning.reflection"
    assert start["metadata"] is None  # no metadata at span start

    assert end["status"] == "ok"
    # The cycle result dict gets packed into end metadata so the
    # span carries findings/actions/notifications counts.
    assert end["metadata"] == {"findings": 0, "actions": 0, "notifications": 0}
    assert result == {"findings": 0, "actions": 0, "notifications": 0}


@pytest.mark.asyncio
async def test_run_reflection_packs_no_llm_path_counts_into_end_metadata(
    install_stubs,
) -> None:
    # With pattern findings present but no LLM available (the stubbed
    # get_pipeline_llm returns None), reflection notifies on every
    # rule-based finding. The end-span metadata reflects that.
    findings = [{"description": "stale thread"}, {"description": "drifted"}]
    install_stubs(findings)
    from atlas_brain.reasoning.reflection import run_reflection

    trace_sink = _RecordingTraceSink()
    result = await run_reflection(ports=_build_fake_ports(trace_sink))

    end = trace_sink.events[1][1]
    assert end["status"] == "ok"
    assert end["metadata"] == {
        "findings": 2,
        "actions": 0,
        "notifications": 2,
    }
    assert result == {
        "findings": 2,
        "actions": 0,
        "notifications": 2,
    }


@pytest.mark.asyncio
async def test_run_reflection_closes_span_with_error_status_on_exception(
    install_stubs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Wire a pattern detector that raises -- the span must still close
    # with status="error" and the exception must propagate so the
    # autonomous task framework can surface it.
    install_stubs([])

    async def _raises() -> list[dict[str, Any]]:
        raise RuntimeError("detector blew up")

    monkeypatch.setattr(
        sys.modules["atlas_brain.reasoning.patterns"],
        "run_all_pattern_detectors",
        _raises,
    )
    from atlas_brain.reasoning.reflection import run_reflection

    trace_sink = _RecordingTraceSink()
    with pytest.raises(RuntimeError, match="detector blew up"):
        await run_reflection(ports=_build_fake_ports(trace_sink))

    assert [e[0] for e in trace_sink.events] == ["start", "end"]
    end = trace_sink.events[1][1]
    assert end["status"] == "error"
    assert end["metadata"]["error_message"] == "detector blew up"
    assert end["metadata"]["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_explicit_ports_bypass_default_builder(install_stubs) -> None:
    # If the default-builder path were taken, run_reflection would import
    # ``from .agent import _build_default_ports`` and that would in turn
    # trigger ``atlas_brain.services.tracing.tracer`` -- our stub provides
    # a no-op tracer so it'd technically succeed, but the point is
    # structural: the explicit ports we pass must be the ones used.
    install_stubs([])
    from atlas_brain.reasoning.reflection import run_reflection

    trace_sink = _RecordingTraceSink()
    fake_ports = _build_fake_ports(trace_sink)
    await run_reflection(ports=fake_ports)

    # If the explicit bundle were ignored, our recording trace_sink
    # would never see the events.
    assert len(trace_sink.events) == 2
