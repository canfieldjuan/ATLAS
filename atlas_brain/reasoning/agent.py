"""ReasoningAgentGraph -- singleton wrapper around the reasoning graph."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional

from .events import AtlasEvent
from .state import ReasoningAgentState

if TYPE_CHECKING:
    from extracted_reasoning_core.types import ReasoningPorts

logger = logging.getLogger("atlas.reasoning.agent")

_instance: Optional["ReasoningAgentGraph"] = None


def _build_default_ports() -> "ReasoningPorts":
    """Build the default ``ReasoningPorts`` bundle for atlas runtime.

    Imports atlas's heavy services package (``services.tracing`` chain)
    and ``reasoning.events.emit_event`` -- this is the wiring site, the
    single place in atlas where those imports tie into the host-agnostic
    Port contract. Splitting it out keeps the constructor of
    ``ReasoningAgentGraph`` trivially testable: tests pass an explicit
    ``ports=`` and bypass the heavy default chain entirely.
    """
    from extracted_reasoning_core.types import ReasoningPorts
    from .events import emit_event
    from .port_adapters import AtlasEventSink, AtlasTraceSink
    from ..services.tracing import tracer

    return ReasoningPorts(
        event_sink=AtlasEventSink(emit_event=emit_event),
        trace_sink=AtlasTraceSink(tracer=tracer),
    )


class ReasoningAgentGraph:
    """Singleton that processes events through the reasoning graph."""

    def __init__(self, ports: Optional["ReasoningPorts"] = None) -> None:
        # ``ports`` is the seam introduced in PR-C4d. Production wiring
        # leaves it None and the singleton builds the atlas-default
        # bundle; tests pass an explicit fake bundle to avoid hitting
        # atlas's heavy services chain.
        self._ports = ports

    @property
    def ports(self) -> "ReasoningPorts":
        if self._ports is None:
            self._ports = _build_default_ports()
        return self._ports

    async def process_event(self, event: AtlasEvent) -> dict[str, Any]:
        """Run a single event through the reasoning graph.

        Returns the processing result dict (stored in atlas_events.processing_result).
        """
        from .graph import run_reasoning_graph
        from ..services.tracing import (
            build_business_trace_context,
            build_reasoning_trace_context,
        )

        state: ReasoningAgentState = {
            "event_id": str(event.id) if event.id else "",
            "event_type": event.event_type,
            "source": event.source,
            "entity_type": event.entity_type,
            "entity_id": event.entity_id,
            "payload": event.payload,
        }
        from ..config import settings
        from ..pipelines.llm import get_pipeline_llm

        _rlm = get_pipeline_llm(
            workload=settings.reasoning.graph_reasoning_workload,
            auto_activate_ollama=False,
            openrouter_model=settings.reasoning.graph_openrouter_model or None,
        )
        # Pack atlas-specific span kwargs into the Port's free-form
        # metadata. ``AtlasTraceSink`` lifts the typed keys back out
        # before calling atlas's tracer; see port_adapters.py for the
        # extraction list.
        start_metadata: dict[str, Any] = {
            "business": build_business_trace_context(
                workflow="reasoning_agent",
                event_type=event.event_type,
                source_name=event.source,
                entity_type=event.entity_type,
                entity_id=event.entity_id,
            ),
        }
        if _rlm is not None:
            model_name = getattr(_rlm, "model", getattr(_rlm, "model_id", None))
            model_provider = getattr(_rlm, "name", None)
            if model_name:
                start_metadata["model_name"] = model_name
            if model_provider:
                start_metadata["model_provider"] = model_provider
        if event.id:
            start_metadata["session_id"] = str(event.id)

        span = self.ports.trace_sink.start_span(
            "reasoning.process",
            metadata=start_metadata,
        )

        try:
            result_state = await run_reasoning_graph(state)
        except Exception:
            self.ports.trace_sink.end_span(
                span,
                status="error",
                metadata={
                    "input_data": {"event_type": event.event_type, "payload": event.payload},
                    "error_message": "reasoning graph failed",
                    "error_type": "ReasoningGraphError",
                },
            )
            logger.error(
                "Reasoning graph failed for event %s", event.id, exc_info=True
            )
            return {
                "status": "error",
                "triage_priority": state.get("triage_priority", "unknown"),
            }

        result = {
            "status": "completed",
            "triage_priority": result_state.get("triage_priority", "unknown"),
            "needs_reasoning": result_state.get("needs_reasoning", False),
            "queued": result_state.get("queued", False),
            "connections": result_state.get("connections_found", []),
            "actions_planned": len(result_state.get("planned_actions", [])),
            "actions_executed": len(result_state.get("action_results", [])),
            "notified": result_state.get("notification_sent", False),
            "summary": result_state.get("summary", ""),
        }
        end_metadata: dict[str, Any] = {
            "input_data": {"event_type": event.event_type, "payload": event.payload},
            "output_data": result,
            "reasoning": build_reasoning_trace_context(
                decision={
                    "triage_priority": result_state.get("triage_priority"),
                    "needs_reasoning": result_state.get("needs_reasoning"),
                    "queued": result_state.get("queued"),
                    "planned_actions": result_state.get("planned_actions", []),
                },
                evidence={
                    "connections_found": result_state.get("connections_found", []),
                    "action_results": result_state.get("action_results", []),
                },
                triage_reasoning=result_state.get("triage_reasoning"),
                rationale=result_state.get("rationale"),
                raw_reasoning=result_state.get("reasoning_output"),
            ),
        }
        if result_state.get("total_input_tokens") is not None:
            end_metadata["input_tokens"] = result_state.get("total_input_tokens")
        if result_state.get("total_output_tokens") is not None:
            end_metadata["output_tokens"] = result_state.get("total_output_tokens")

        self.ports.trace_sink.end_span(span, status="ok", metadata=end_metadata)
        return result

    async def process_drained_events(
        self, events: list[AtlasEvent]
    ) -> list[dict[str, Any]]:
        """Process a batch of drained events (after lock release).

        Provides accumulated context from the voice session.
        """
        results = []
        for event in events:
            try:
                result = await asyncio.wait_for(
                    self.process_event(event), timeout=120.0,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Drained event %s timed out after 120s", event.id,
                )
                result = {"status": "timeout", "event_id": str(event.id)}
            results.append(result)
        return results


def get_reasoning_agent() -> ReasoningAgentGraph:
    """Get or create the reasoning agent singleton."""
    global _instance
    if _instance is None:
        _instance = ReasoningAgentGraph()
    return _instance
