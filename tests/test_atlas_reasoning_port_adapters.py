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

import asyncio
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
    """Fake atlas tracer.

    Mirrors the kwargs ``AtlasTraceSink`` actually passes to atlas's
    real tracer in :mod:`atlas_brain.services.tracing` -- specifically
    the typed kwargs the adapter lifts out of the Port's metadata bag
    (``model_name``/``model_provider``/``session_id`` on start, and
    ``input_tokens``/``output_tokens``/``input_data``/``output_data``/
    ``error_message``/``error_type`` on end). Unknown kwargs are
    captured under ``extra`` so a regression that adds a new typed
    extraction without a matching test will surface as an unexpected
    kwarg.
    """

    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.ends: list[dict[str, Any]] = []

    def start_span(
        self,
        span_name: str,
        operation_type: str,
        metadata: dict[str, Any] | None = None,
        *,
        model_name: str | None = None,
        model_provider: str | None = None,
        session_id: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        record = {
            "span_name": span_name,
            "operation_type": operation_type,
            "metadata": metadata,
            "model_name": model_name,
            "model_provider": model_provider,
            "session_id": session_id,
            "extra": extra,
        }
        self.starts.append(record)
        return record

    def end_span(
        self,
        ctx: Any,
        status: str = "completed",
        metadata: Mapping[str, Any] | None = None,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        error_message: str | None = None,
        error_type: str | None = None,
        **extra: Any,
    ) -> None:
        self.ends.append(
            {
                "ctx": ctx,
                "status": status,
                "metadata": dict(metadata) if metadata else None,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_data": input_data,
                "output_data": output_data,
                "error_message": error_message,
                "error_type": error_type,
                "extra": extra,
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
async def test_atlas_event_sink_emit_propagates_none_when_entity_kwargs_unset() -> None:
    # When the caller doesn't supply entity_type/entity_id, the adapter
    # must propagate them as ``None`` to atlas's ``emit_event`` (matching
    # atlas's ``Optional[str] = None`` parameter contract). The Port
    # signature also marks both as optional, so tests must pin that the
    # default flows through unchanged.
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


# ----------------------------------------------------------------------
# Metadata-key extraction (PR-C4d)
#
# Atlas's tracer (atlas_brain.services.tracing) carries model_name,
# model_provider, session_id on the SpanContext (set at start_span)
# and accepts input_tokens, output_tokens, input_data, output_data,
# error_message, error_type as typed kwargs to end_span. Cost calc,
# trace payload columns, and llm_usage rows all depend on these
# arriving as kwargs -- not nested in metadata. The Port contract
# can't widen to expose them, so AtlasTraceSink lifts specific keys
# out of metadata.
# ----------------------------------------------------------------------


def test_start_span_lifts_model_kwargs_out_of_metadata() -> None:
    fake = _RecordingTracer()
    sink = AtlasTraceSink(tracer=fake)

    sink.start_span(
        "reasoning.process",
        metadata={
            "model_name": "qwen3:14b",
            "model_provider": "ollama",
            "session_id": "evt-42",
            "depth": "L3",
        },
    )

    record = fake.starts[0]
    # The three lifted keys must arrive as typed kwargs.
    assert record["model_name"] == "qwen3:14b"
    assert record["model_provider"] == "ollama"
    assert record["session_id"] == "evt-42"
    # Anything else stays in metadata.
    assert record["metadata"] == {"depth": "L3"}
    # Sanity: no key lands in **extra (which would mean the adapter
    # is passing an unexpected kwarg to the underlying tracer).
    assert record["extra"] == {}


def test_start_span_keeps_reasoning_and_business_in_metadata() -> None:
    # ``reasoning`` is promoted to a top-level payload field by atlas's
    # _derive_reasoning_text. ``business`` is the structured business
    # context envelope. Both are deliberately NOT extracted -- they need
    # to stay nested in metadata for the existing atlas tracer codepaths
    # to keep working.
    fake = _RecordingTracer()
    sink = AtlasTraceSink(tracer=fake)

    sink.start_span(
        "reasoning.process",
        metadata={
            "reasoning": {"summary": "triaged as low priority"},
            "business": {"workflow": "reasoning_agent"},
            "model_name": "qwen3:14b",
        },
    )

    record = fake.starts[0]
    assert record["model_name"] == "qwen3:14b"
    assert record["metadata"] == {
        "reasoning": {"summary": "triaged as low priority"},
        "business": {"workflow": "reasoning_agent"},
    }


def test_start_span_metadata_becomes_none_when_only_lifted_keys_present() -> None:
    # If every metadata key is lifted into typed kwargs, metadata should
    # arrive as None (not an empty dict) -- atlas's tracer accepts
    # Optional[dict] and treats None as "no metadata".
    fake = _RecordingTracer()
    sink = AtlasTraceSink(tracer=fake)

    sink.start_span(
        "reasoning.process",
        metadata={
            "model_name": "qwen3:14b",
            "model_provider": "ollama",
            "session_id": "evt-1",
        },
    )

    record = fake.starts[0]
    assert record["metadata"] is None


def test_end_span_lifts_typed_kwargs_out_of_metadata() -> None:
    fake = _RecordingTracer()
    sink = AtlasTraceSink(tracer=fake)
    span = sink.start_span("reasoning.process")

    sink.end_span(
        span,
        status="ok",
        metadata={
            "input_tokens": 1234,
            "output_tokens": 567,
            "input_data": {"event_type": "test"},
            "output_data": {"summary": "done"},
            "reasoning": {"summary": "triaged"},
            "business": {"workflow": "x"},
        },
    )

    record = fake.ends[0]
    assert record["status"] == "completed"
    assert record["input_tokens"] == 1234
    assert record["output_tokens"] == 567
    assert record["input_data"] == {"event_type": "test"}
    assert record["output_data"] == {"summary": "done"}
    # reasoning + business stay in metadata.
    assert record["metadata"] == {
        "reasoning": {"summary": "triaged"},
        "business": {"workflow": "x"},
    }


def test_end_span_lifts_error_kwargs_on_error_status() -> None:
    fake = _RecordingTracer()
    sink = AtlasTraceSink(tracer=fake)
    span = sink.start_span("reasoning.process")

    sink.end_span(
        span,
        status="error",
        metadata={
            "error_message": "reasoning graph failed",
            "error_type": "ReasoningGraphError",
            "input_data": {"event_type": "test"},
        },
    )

    record = fake.ends[0]
    assert record["status"] == "failed"
    assert record["error_message"] == "reasoning graph failed"
    assert record["error_type"] == "ReasoningGraphError"
    assert record["input_data"] == {"event_type": "test"}
    assert record["metadata"] is None


def test_extraction_only_lifts_keys_present_in_metadata() -> None:
    # Verify the adapter doesn't pass spurious kwargs (e.g. None values
    # for keys the caller didn't supply) -- atlas's tracer treats
    # ``input_tokens=None`` and "input_tokens omitted" identically, but
    # the kwargs noise would clutter signatures and risk shadowing
    # future tracer params.
    fake = _RecordingTracer()
    sink = AtlasTraceSink(tracer=fake)
    span = sink.start_span("reasoning.process", metadata={"model_name": "x"})
    sink.end_span(span, metadata={"input_tokens": 10})

    # start_span got model_name; the other lifted keys default to None
    # on the fake (matching atlas tracer behavior) but were never
    # passed by the adapter -- captured by the start_record itself.
    start_record = fake.starts[0]
    assert start_record["model_name"] == "x"
    # _RecordingTracer's defaults reflect "kwarg not passed" for the
    # other two; the adapter shouldn't have passed them. Same for end.
    end_record = fake.ends[0]
    assert end_record["input_tokens"] == 10
    assert end_record["output_tokens"] is None
    assert end_record["error_message"] is None


# ----------------------------------------------------------------------
# AtlasLLMClient (PR-C4e2)
#
# Adapter wraps atlas's sync ``LLMService.chat`` so it satisfies the
# core ``LLMClient.complete()`` Protocol. Tests verify:
#
# - sync->async dispatch (chat is called via asyncio.to_thread)
# - Mapping[str,Any] -> atlas Message conversion
# - metadata-key extraction (json_mode/response_format/timeout) into
#   chat()'s typed kwargs
# - Protocol satisfaction
#
# The adapter lazy-imports atlas's ``Message`` at call time to keep
# port_adapters import-light. Tests stub a fake ``Message`` in
# sys.modules['atlas_brain.services.protocols'] so the import resolves
# without pulling the heavy services/__init__ chain. Stubs are
# restored on teardown (PR-C4d Codex lesson).
# ----------------------------------------------------------------------


import sys as _sys
import types as _types

from atlas_brain.reasoning.port_adapters import AtlasLLMClient
from extracted_reasoning_core.ports import LLMClient


class _FakeMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class _RecordingLLMService:
    """Fake atlas LLMService -- records ``chat()`` calls."""

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self._response = response or {"response": "ok", "usage": {"input_tokens": 1}}
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self._response


@pytest.fixture
def stub_protocols_module():
    """Install a fake ``atlas_brain.services.protocols`` so the adapter's
    lazy ``from ..services.protocols import Message`` resolves without
    triggering ``atlas_brain.services.__init__``'s heavy imports."""
    saved_services = _sys.modules.get("atlas_brain.services")
    saved_protocols = _sys.modules.get("atlas_brain.services.protocols")

    services_pkg = _types.ModuleType("atlas_brain.services")
    services_pkg.__path__ = []
    protocols_mod = _types.ModuleType("atlas_brain.services.protocols")
    protocols_mod.Message = _FakeMessage

    _sys.modules["atlas_brain.services"] = services_pkg
    _sys.modules["atlas_brain.services.protocols"] = protocols_mod

    yield

    if saved_protocols is None:
        _sys.modules.pop("atlas_brain.services.protocols", None)
    else:
        _sys.modules["atlas_brain.services.protocols"] = saved_protocols
    if saved_services is None:
        _sys.modules.pop("atlas_brain.services", None)
    else:
        _sys.modules["atlas_brain.services"] = saved_services


def test_atlas_llm_client_satisfies_llm_client_protocol() -> None:
    # LLMClient is not @runtime_checkable, so we can only do a
    # structural hasattr check here. A future PR that flips the
    # Protocol to runtime-checkable can tighten this.
    client = AtlasLLMClient(_RecordingLLMService())
    assert hasattr(client, "complete")
    assert callable(getattr(client, "complete", None))
    # Static analyzers should also accept it where LLMClient is required.
    _: LLMClient = client


@pytest.mark.asyncio
async def test_atlas_llm_client_dispatches_to_chat_via_to_thread(
    stub_protocols_module,
) -> None:
    fake_service = _RecordingLLMService(
        response={"response": "hi", "usage": {"input_tokens": 4, "output_tokens": 2}},
    )
    client = AtlasLLMClient(fake_service)

    result = await client.complete(
        [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ],
        max_tokens=128,
        temperature=0.3,
    )

    assert result == {"response": "hi", "usage": {"input_tokens": 4, "output_tokens": 2}}
    assert len(fake_service.calls) == 1
    call = fake_service.calls[0]
    # Messages converted to atlas Message objects (the Mapping->dataclass
    # translation the adapter performs at the boundary).
    assert isinstance(call["messages"], list)
    assert len(call["messages"]) == 2
    assert isinstance(call["messages"][0], _FakeMessage)
    assert call["messages"][0].role == "system"
    assert call["messages"][0].content == "system text"
    assert call["max_tokens"] == 128
    assert call["temperature"] == 0.3


@pytest.mark.asyncio
async def test_atlas_llm_client_lifts_json_mode_from_metadata(
    stub_protocols_module,
) -> None:
    fake_service = _RecordingLLMService()
    client = AtlasLLMClient(fake_service)

    await client.complete(
        [{"role": "user", "content": "x"}],
        max_tokens=10,
        temperature=0.0,
        metadata={
            "json_mode": True,
            "response_format": {"type": "json_object"},
            "timeout": 60.0,
            "irrelevant": "stays-in-metadata-but-not-passed-to-chat",
        },
    )

    call = fake_service.calls[0]
    # The three known keys are lifted into typed kwargs.
    assert call["json_mode"] is True
    assert call["response_format"] == {"type": "json_object"}
    assert call["timeout"] == 60.0
    # Unrelated metadata keys are NOT forwarded as kwargs (would
    # otherwise risk shadowing future LLMService kwargs).
    assert "irrelevant" not in call


@pytest.mark.asyncio
async def test_atlas_llm_client_complete_returns_dict_for_non_dict_chat_result(
    stub_protocols_module,
) -> None:
    # Some atlas LLMService impls return a plain string. The adapter
    # must wrap it so the LLMClient contract (returns Mapping) holds.
    class _StringChatService:
        def chat(self, **kwargs: Any) -> str:
            return "raw string response"

    client = AtlasLLMClient(_StringChatService())
    result = await client.complete(
        [{"role": "user", "content": "x"}],
        max_tokens=10,
        temperature=0.0,
    )
    assert result == {"response": "raw string response"}


@pytest.mark.asyncio
async def test_atlas_llm_client_enforces_metadata_timeout_via_wait_for(
    stub_protocols_module,
) -> None:
    # Atlas's pre-extraction _llm_generate wrapped chat() in
    # asyncio.wait_for(timeout=120). PR-C4e2 broke that until the
    # adapter was taught to honor the metadata timeout itself --
    # forwarding ``timeout`` to chat() alone isn't enough because
    # some Provider impls ignore the kwarg or block in non-cooperative
    # C extensions. Pin the wait_for behavior so a regression here
    # would surface as a TimeoutError, not an indefinite hang.
    import time

    class _SlowChatService:
        def chat(self, **kwargs: Any) -> dict[str, Any]:
            time.sleep(2.0)  # Longer than our 0.05s timeout below.
            return {"response": "should never be reached"}

    client = AtlasLLMClient(_SlowChatService())
    with pytest.raises(asyncio.TimeoutError):
        await client.complete(
            [{"role": "user", "content": "x"}],
            max_tokens=10,
            temperature=0.0,
            metadata={"timeout": 0.05},
        )


@pytest.mark.asyncio
async def test_atlas_llm_client_no_timeout_when_metadata_omits_it(
    stub_protocols_module,
) -> None:
    # Without an explicit timeout in metadata, the adapter awaits the
    # to_thread coroutine directly -- no outer wait_for. This
    # preserves backward compat with callers that don't care about
    # deadlines (and doesn't impose an arbitrary default).
    fake = _RecordingLLMService(response={"response": "ok"})
    client = AtlasLLMClient(fake)
    result = await client.complete(
        [{"role": "user", "content": "x"}],
        max_tokens=10,
        temperature=0.0,
    )
    assert result == {"response": "ok"}
    # Sanity: no timeout kwarg got passed to chat() when none in metadata.
    assert "timeout" not in fake.calls[0]
