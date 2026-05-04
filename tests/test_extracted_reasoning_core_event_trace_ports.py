"""Unit tests for the EventSink and TraceSink ports added in PR-C4a / PR 6.

These tests verify the Protocol contracts for the two new reasoning-side
hooks that the audit's PR 6 calls for:

  - EventSink: host event bus hook (atlas: ``atlas_events`` table + LISTEN/NOTIFY)
  - TraceSink: host tracing hook (atlas: tracer.start_span/end_span)

Reasoning core uses these as ports so the runner stays host-agnostic.
Atlas-side adapters land in subsequent PR-C4b/c slices; this slice
just establishes the seams.

All tests are pure -- no atlas imports, no Postgres -- so they wire
cleanly into the standalone extracted-pipeline CI.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from extracted_reasoning_core.ports import EventSink, TraceSink


# ----------------------------------------------------------------------
# EventSink contract
# ----------------------------------------------------------------------


class _FakeEventSink:
    """Minimal duck-typed implementation that records emit() calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def emit(
        self,
        event_type: str,
        source: str,
        payload: Mapping[str, Any],
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> str:
        self.calls.append({
            "event_type": event_type,
            "source": source,
            "payload": dict(payload),
            "entity_type": entity_type,
            "entity_id": entity_id,
        })
        return f"event-{len(self.calls)}"


def test_event_sink_protocol_is_runtime_satisfiable() -> None:
    sink: EventSink = _FakeEventSink()
    # Static-type assertion via annotation only -- the body just exercises
    # the duck-typed call to confirm the Protocol shape matches.
    assert hasattr(sink, "emit")


@pytest.mark.asyncio
async def test_event_sink_emit_records_call_and_returns_event_id() -> None:
    sink = _FakeEventSink()
    event_id = await sink.emit(
        "vendor.archetype_assigned",
        "reasoning_core",
        {"vendor": "Acme", "archetype": "pricing_shock"},
        entity_type="vendor",
        entity_id="acme",
    )
    assert event_id == "event-1"
    assert len(sink.calls) == 1
    call = sink.calls[0]
    assert call["event_type"] == "vendor.archetype_assigned"
    assert call["source"] == "reasoning_core"
    assert call["payload"] == {"vendor": "Acme", "archetype": "pricing_shock"}
    assert call["entity_type"] == "vendor"
    assert call["entity_id"] == "acme"


@pytest.mark.asyncio
async def test_event_sink_emit_accepts_no_entity_kwargs() -> None:
    # Both entity kwargs are optional; sinks must tolerate omitting them.
    sink = _FakeEventSink()
    event_id = await sink.emit(
        "system.reasoning_completed",
        "reasoning_core",
        {"depth": "L3"},
    )
    assert event_id == "event-1"
    assert sink.calls[0]["entity_type"] is None
    assert sink.calls[0]["entity_id"] is None


# ----------------------------------------------------------------------
# TraceSink contract
# ----------------------------------------------------------------------


class _FakeTraceSink:
    """Minimal duck-typed implementation that records start/end calls."""

    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.ends: list[dict[str, Any]] = []

    def start_span(
        self,
        name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        span = {"name": name, "metadata": dict(metadata) if metadata else {}}
        self.starts.append(span)
        return span

    def end_span(
        self,
        span: Any,
        *,
        status: str = "ok",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.ends.append({
            "span": span,
            "status": status,
            "metadata": dict(metadata) if metadata else {},
        })


def test_trace_sink_protocol_is_runtime_satisfiable() -> None:
    sink: TraceSink = _FakeTraceSink()
    assert hasattr(sink, "start_span")
    assert hasattr(sink, "end_span")


def test_trace_sink_round_trip_passes_span_handle_back() -> None:
    sink = _FakeTraceSink()
    span = sink.start_span("reasoning.process", metadata={"depth": "L2"})
    sink.end_span(span, status="ok", metadata={"latency_ms": 1234})

    assert len(sink.starts) == 1
    assert sink.starts[0]["name"] == "reasoning.process"
    assert sink.starts[0]["metadata"] == {"depth": "L2"}

    assert len(sink.ends) == 1
    # The span handle returned by start_span must be the same object passed
    # to end_span -- the Protocol explicitly says the handle is opaque
    # to reasoning core.
    assert sink.ends[0]["span"] is span
    assert sink.ends[0]["status"] == "ok"
    assert sink.ends[0]["metadata"] == {"latency_ms": 1234}


def test_trace_sink_metadata_is_optional() -> None:
    sink = _FakeTraceSink()
    span = sink.start_span("reasoning.cache_lookup")
    sink.end_span(span)

    assert sink.starts[0]["metadata"] == {}
    assert sink.ends[0]["status"] == "ok"
    assert sink.ends[0]["metadata"] == {}


def test_trace_sink_supports_error_status() -> None:
    sink = _FakeTraceSink()
    span = sink.start_span("reasoning.process")
    sink.end_span(span, status="error", metadata={"exception": "RuntimeError"})

    assert sink.ends[0]["status"] == "error"
    assert sink.ends[0]["metadata"]["exception"] == "RuntimeError"


# ----------------------------------------------------------------------
# Combined ports surface
# ----------------------------------------------------------------------


def test_ports_module_exports_new_ports() -> None:
    # Pin the public surface so callers reading from
    # ``extracted_reasoning_core.ports.__all__`` can rely on the names.
    from extracted_reasoning_core import ports as ports_module

    assert "EventSink" in ports_module.__all__
    assert "TraceSink" in ports_module.__all__
