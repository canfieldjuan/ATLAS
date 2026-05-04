"""Verify atlas-side adapters satisfy the reasoning core port Protocols.

PR-C4a added ``EventSink`` and ``TraceSink`` Protocols to
``extracted_reasoning_core.ports`` (both ``@runtime_checkable``).
PR-C4c (this slice) provides the atlas implementations in
``atlas_brain.reasoning.port_adapters``. These tests pin:

- ``isinstance(sink, EventSink)`` / ``isinstance(sink, TraceSink)``
  (Protocol satisfaction guard adapter wiring will rely on)
- the adapters call the injected ``emit_event`` / tracer with the
  translated argument shapes the Port contracts require
- the Port-side ``status`` vocabulary is mapped to atlas's
  ``completed``/``failed`` convention, with passthrough for unknown
  values so callers can extend the vocabulary without losing data

Both adapters take their atlas-side dependency by constructor injection
(see ``port_adapters`` module docstring for the import-isolation
rationale), so tests pass recording fakes directly -- no monkeypatch
needed and no path that reaches real I/O.
"""

from __future__ import annotations

from typing import Any, Mapping
from uuid import UUID

import pytest

from atlas_brain.reasoning.port_adapters import AtlasEventSink, AtlasTraceSink
from extracted_reasoning_core.ports import EventSink, TraceSink


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------


class _RecordingEmit:
    def __init__(self, event_id: UUID) -> None:
        self._event_id = event_id
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        event_type: str,
        source: str,
        payload: dict[str, Any],
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> UUID:
        self.calls.append(
            {
                "event_type": event_type,
                "source": source,
                "payload": payload,
                "entity_type": entity_type,
                "entity_id": entity_id,
            }
        )
        return self._event_id


class _RecordingTracer:
    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.ends: list[dict[str, Any]] = []

    def start_span(
        self,
        span_name: str,
        operation_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "span_name": span_name,
            "operation_type": operation_type,
            "metadata": metadata,
        }
        self.starts.append(record)
        return record

    def end_span(
        self,
        ctx: Any,
        status: str = "completed",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.ends.append(
            {
                "ctx": ctx,
                "status": status,
                "metadata": dict(metadata) if metadata else None,
            }
        )


# ----------------------------------------------------------------------
# Protocol satisfaction
# ----------------------------------------------------------------------


def test_atlas_event_sink_satisfies_event_sink_protocol() -> None:
    # Real runtime Protocol check (the Protocol is @runtime_checkable
    # per PR-C4a). Catches the case where the adapter's emit signature
    # drifts away from the Port contract.
    sink = AtlasEventSink(emit_event=_RecordingEmit(UUID(int=0)))
    assert isinstance(sink, EventSink)


def test_atlas_trace_sink_satisfies_trace_sink_protocol() -> None:
    sink = AtlasTraceSink(tracer=_RecordingTracer())
    assert isinstance(sink, TraceSink)


# ----------------------------------------------------------------------
# AtlasEventSink behavior
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_atlas_event_sink_emit_translates_args_and_stringifies_uuid() -> None:
    fake = _RecordingEmit(UUID("11111111-2222-3333-4444-555555555555"))
    sink = AtlasEventSink(emit_event=fake)

    event_id = await sink.emit(
        "vendor.archetype_assigned",
        "reasoning_core",
        {"vendor": "Acme", "archetype": "pricing_shock"},
        entity_type="vendor",
        entity_id="acme",
    )

    # Atlas's emit_event returns a UUID; the Port contract returns str.
    # The adapter must stringify -- otherwise downstream code that does
    # ``len(event_id)`` or string comparison on the returned id breaks.
    assert event_id == "11111111-2222-3333-4444-555555555555"
    assert isinstance(event_id, str)

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["event_type"] == "vendor.archetype_assigned"
    assert call["source"] == "reasoning_core"
    assert call["payload"] == {"vendor": "Acme", "archetype": "pricing_shock"}
    assert call["entity_type"] == "vendor"
    assert call["entity_id"] == "acme"


@pytest.mark.asyncio
async def test_atlas_event_sink_emit_omits_optional_entity_kwargs() -> None:
    fake = _RecordingEmit(UUID(int=0))
    sink = AtlasEventSink(emit_event=fake)

    await sink.emit(
        "system.reasoning_completed",
        "reasoning_core",
        {"depth": "L3"},
    )
    call = fake.calls[0]
    assert call["entity_type"] is None
    assert call["entity_id"] is None


@pytest.mark.asyncio
async def test_atlas_event_sink_emit_copies_payload_to_dict() -> None:
    # Mapping inputs (immutable views, custom mappings) must reach the
    # underlying emit_event as a plain dict, since atlas's emit_event
    # JSON-encodes via __import__("json").dumps which expects a dict.
    fake = _RecordingEmit(UUID(int=0))
    sink = AtlasEventSink(emit_event=fake)

    class _ROMapping(Mapping[str, Any]):
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    await sink.emit("e", "s", _ROMapping({"k": "v"}))
    assert fake.calls[0]["payload"] == {"k": "v"}
    assert isinstance(fake.calls[0]["payload"], dict)


# ----------------------------------------------------------------------
# AtlasTraceSink behavior
# ----------------------------------------------------------------------


def test_atlas_trace_sink_start_span_uses_reasoning_operation_type() -> None:
    # The Port contract is host-agnostic -- it never names operation_type.
    # The adapter pins it to "reasoning" so atlas-side trace queries can
    # filter on a stable value.
    fake = _RecordingTracer()
    sink = AtlasTraceSink(tracer=fake)

    sink.start_span("reasoning.process", metadata={"depth": "L2"})

    assert len(fake.starts) == 1
    record = fake.starts[0]
    assert record["span_name"] == "reasoning.process"
    assert record["operation_type"] == "reasoning"
    assert record["metadata"] == {"depth": "L2"}


def test_atlas_trace_sink_round_trips_span_handle() -> None:
    # Per the Port contract, the span returned from start_span is opaque
    # to reasoning core and must be passed back to end_span unchanged.
    fake = _RecordingTracer()
    sink = AtlasTraceSink(tracer=fake)

    span = sink.start_span("reasoning.cache_lookup")
    sink.end_span(span, status="ok", metadata={"hit": True})

    assert len(fake.ends) == 1
    assert fake.ends[0]["ctx"] is span
    assert fake.ends[0]["metadata"] == {"hit": True}


def test_atlas_trace_sink_status_ok_maps_to_completed() -> None:
    fake = _RecordingTracer()
    sink = AtlasTraceSink(tracer=fake)
    span = sink.start_span("reasoning.process")
    sink.end_span(span)  # status defaults to "ok"
    assert fake.ends[0]["status"] == "completed"


def test_atlas_trace_sink_status_error_maps_to_failed() -> None:
    fake = _RecordingTracer()
    sink = AtlasTraceSink(tracer=fake)
    span = sink.start_span("reasoning.process")
    sink.end_span(span, status="error", metadata={"exception": "RuntimeError"})
    assert fake.ends[0]["status"] == "failed"


def test_atlas_trace_sink_unknown_status_passes_through() -> None:
    # Unknown status values flow through untouched -- callers can extend
    # the vocabulary (e.g. "cancelled", "skipped") without losing data.
    fake = _RecordingTracer()
    sink = AtlasTraceSink(tracer=fake)
    span = sink.start_span("reasoning.process")
    sink.end_span(span, status="cancelled")
    assert fake.ends[0]["status"] == "cancelled"


def test_atlas_trace_sink_metadata_optional_on_both_calls() -> None:
    fake = _RecordingTracer()
    sink = AtlasTraceSink(tracer=fake)
    span = sink.start_span("reasoning.process")
    sink.end_span(span)

    assert fake.starts[0]["metadata"] is None
    assert fake.ends[0]["metadata"] is None
