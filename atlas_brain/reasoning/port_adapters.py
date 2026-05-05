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

PR-C4d extends ``AtlasTraceSink`` to lift specific keys out of the Port
contract's free-form ``metadata`` and route them to atlas's tracer as
typed kwargs. This preserves the host's existing FTL trace payload
shape (top-level ``model_name`` / ``model_provider`` / ``session_tag``
columns, token counts, ``input_data``/``output_data``) without
widening the Port contract -- which has to stay host-agnostic. Two keys
are deliberately *not* extracted: ``reasoning`` (atlas's
``_derive_reasoning_text`` promotes ``metadata["reasoning"]`` to a
top-level field) and ``business`` (the business-context envelope is
captured as nested metadata by design). Callers stuff those into the
metadata dict as-is.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Mapping, Sequence


_DEFAULT_OPERATION_TYPE = "reasoning"

# Keys that ``AtlasLLMClient.complete`` lifts out of the Port's
# free-form ``metadata`` and forwards to atlas's ``LLMService.chat``
# as typed kwargs. ``json_mode`` and ``response_format`` control
# structured-output behavior; ``timeout`` is both forwarded to
# ``chat()`` (in case the underlying client honors it) and used to
# wrap the ``asyncio.to_thread`` await in ``asyncio.wait_for``.
#
# Metadata keys NOT in this tuple are deliberately discarded -- atlas's
# ``chat`` signature is fixed and unrecognized kwargs would raise
# ``TypeError``. To extend the contract, add the key here AND update
# the matching atlas-side signature.
_LLM_COMPLETE_METADATA_KWARGS: tuple[str, ...] = (
    "json_mode",
    "response_format",
    "timeout",
)

# Keys that ``AtlasTraceSink.start_span`` lifts out of metadata into
# atlas tracer kwargs. These map to fields atlas's ``SpanContext``
# carries through to the trace payload + ``llm_usage`` columns.
_START_SPAN_METADATA_KWARGS: tuple[str, ...] = (
    "model_name",
    "model_provider",
    "session_id",
)

# Keys that ``AtlasTraceSink.end_span`` lifts out of metadata. These map
# to atlas tracer's ``end_span`` typed kwargs that affect cost calc
# (tokens), payload top-level columns (input_data/output_data), and
# the failure envelope (error_message/error_type).
_END_SPAN_METADATA_KWARGS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "input_data",
    "output_data",
    "error_message",
    "error_type",
)


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
        kwargs, remaining = _split_metadata(metadata, _START_SPAN_METADATA_KWARGS)
        return self._tracer.start_span(
            span_name=name,
            operation_type=_DEFAULT_OPERATION_TYPE,
            metadata=remaining,
            **kwargs,
        )

    def end_span(
        self,
        span: Any,
        *,
        status: str = "ok",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        kwargs, remaining = _split_metadata(metadata, _END_SPAN_METADATA_KWARGS)
        self._tracer.end_span(
            span,
            status=_translate_status(status),
            metadata=remaining,
            **kwargs,
        )


class AtlasLLMClient:
    """``LLMClient`` adapter over atlas's ``LLMService.chat``.

    Wraps an :class:`atlas_brain.services.protocols.LLMService` so it
    satisfies :class:`extracted_reasoning_core.ports.LLMClient`. Three
    translation responsibilities:

    - **sync -> async**: atlas's ``chat`` is sync (the native model
      call); the Port is ``async``. We dispatch via
      :func:`asyncio.to_thread` so the event loop stays responsive
      while the model runs.
    - **timeout enforcement**: when the metadata bag carries a
      ``timeout`` (in seconds), the adapter wraps the
      ``asyncio.to_thread`` await in :func:`asyncio.wait_for`. This
      preserves the deadline semantics atlas's pre-extraction
      ``_llm_generate`` enforced -- forwarding ``timeout`` as a kwarg
      to ``chat`` alone isn't enough because some Provider impls
      ignore it or block in non-cooperative C extensions.
    - **metadata -> typed kwargs**: the Port carries ``json_mode`` /
      ``response_format`` / ``timeout`` in the free-form metadata bag
      so reasoning-core nodes don't have to know about atlas's chat
      signature. The adapter lifts those three keys back out and
      forwards them as typed kwargs to ``chat`` (in addition to
      using ``timeout`` for the outer ``wait_for``). Other metadata
      keys are deliberately *not* forwarded -- atlas's chat signature
      is fixed and unrecognized kwargs would raise ``TypeError``;
      callers that need to extend the contract should add new
      typed-extraction entries to ``_LLM_COMPLETE_METADATA_KWARGS``.

    The adapter accepts ``llm_service`` as ``Any`` rather than a typed
    parameter -- atlas's ``LLMService`` Protocol lives behind
    ``atlas_brain.services.__init__`` which would pull in the heavy
    LLM-impl chain at import time, defeating the import-light goal of
    this module.
    """

    def __init__(self, llm_service: Any) -> None:
        self._llm_service = llm_service

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        kwargs, _remaining = _split_metadata(metadata, _LLM_COMPLETE_METADATA_KWARGS)
        # Atlas's chat() expects a list of Message objects; the Port
        # passes raw mappings. Convert here -- atlas's Message is a
        # dataclass with ``role`` + ``content``, so a dict with those
        # keys is structurally compatible. Importing atlas's Message
        # at call-time avoids the heavy services package init at
        # adapter-module-import time.
        from ..services.protocols import Message

        message_list = [
            Message(role=m["role"], content=m["content"]) for m in messages
        ]

        chat_kwargs: dict[str, Any] = {
            "messages": message_list,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        chat_kwargs.update(kwargs)

        timeout = kwargs.get("timeout")
        coro = asyncio.to_thread(self._llm_service.chat, **chat_kwargs)
        if timeout is not None and timeout > 0:
            result = await asyncio.wait_for(coro, timeout=timeout)
        else:
            result = await coro
        return result if isinstance(result, dict) else {"response": str(result)}


def _translate_status(port_status: str) -> str:
    if port_status == "ok":
        return "completed"
    if port_status == "error":
        return "failed"
    return port_status


def _split_metadata(
    metadata: Mapping[str, Any] | None,
    typed_keys: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Split a metadata mapping into (typed_kwargs, remaining_metadata).

    Used by ``AtlasTraceSink`` to lift host-tracer-specific keys out of
    the Port contract's free-form metadata. Returns ``(kwargs, None)``
    when no metadata remains after extraction so callers can pass the
    underlying tracer ``metadata=None`` cleanly.
    """
    if not metadata:
        return {}, None
    remaining = dict(metadata)
    kwargs: dict[str, Any] = {}
    for key in typed_keys:
        if key in remaining:
            kwargs[key] = remaining.pop(key)
    return kwargs, remaining or None


__all__ = ["AtlasEventSink", "AtlasLLMClient", "AtlasTraceSink"]
