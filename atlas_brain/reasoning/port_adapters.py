"""Atlas-side adapters that satisfy reasoning core's host-facing ports.

PR-C4a established the ``EventSink`` and ``TraceSink`` Protocols in
``extracted_reasoning_core.ports`` so the extracted reasoning runner can
emit events and open spans without reaching into host internals.
PR-C4c (this module) is the atlas-side implementation of those ports:

- ``AtlasEventSink`` wraps :func:`atlas_brain.reasoning.events.emit_event`,
  which writes to the ``atlas_events`` table and triggers Postgres
  LISTEN/NOTIFY downstream. The Protocol returns the event identifier
  as a ``str``; atlas's ``emit_event`` returns ``UUID``, so the adapter
  stringifies on the way out.

- ``AtlasTraceSink`` wraps the FTL tracer in
  :mod:`atlas_brain.services.tracing`. Atlas's ``tracer.start_span``
  requires a positional ``operation_type`` that the Port deliberately
  doesn't expose (the Port is host-agnostic). The adapter hardcodes
  ``"reasoning"`` -- if a future host wants finer span typing, that
  belongs in metadata, not at the port boundary.

  Port-side ``status`` is ``"ok"``/``"error"`` by convention; atlas's
  tracer speaks ``"completed"``/``"failed"``. The adapter translates
  both known values and passes anything else through unchanged so
  callers can extend the vocabulary without losing data.

The dependencies (the ``emit_event`` callable, the tracer object) are
injected at construction. This keeps the adapter module
import-lightweight -- it never reaches into ``atlas_brain.services.*``
or other heavy atlas internals -- so it can be exercised by the
standalone-CI test suite. The production wiring site does the heavy
imports once and hands instances to ``ReasoningPorts``.

PR-C4c does not wire these into the runner -- that's PR-C4d/e. This
slice just provides the atlas-side surface that future port-instantiation
code can hand to ``ReasoningPorts``.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping


_DEFAULT_OPERATION_TYPE = "reasoning"


class AtlasEventSink:
    """``EventSink`` adapter over atlas's ``atlas_events`` writer."""

    def __init__(
        self,
        emit_event: Callable[..., Awaitable[Any]],
    ) -> None:
        self._emit_event = emit_event

    async def emit(
        self,
        event_type: str,
        source: str,
        payload: Mapping[str, Any],
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> str:
        event_id = await self._emit_event(
            event_type,
            source,
            dict(payload),
            entity_type=entity_type,
            entity_id=entity_id,
        )
        return str(event_id)


class AtlasTraceSink:
    """``TraceSink`` adapter over atlas's FTL tracer."""

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    def start_span(
        self,
        name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        return self._tracer.start_span(
            span_name=name,
            operation_type=_DEFAULT_OPERATION_TYPE,
            metadata=dict(metadata) if metadata else None,
        )

    def end_span(
        self,
        span: Any,
        *,
        status: str = "ok",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._tracer.end_span(
            span,
            status=_translate_status(status),
            metadata=dict(metadata) if metadata else None,
        )


def _translate_status(port_status: str) -> str:
    if port_status == "ok":
        return "completed"
    if port_status == "error":
        return "failed"
    return port_status


__all__ = ["AtlasEventSink", "AtlasTraceSink"]
