"""ReasoningAgentGraph -- singleton wrapper around the reasoning graph."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from .events import AtlasEvent
from .state import ReasoningAgentState

logger = logging.getLogger("atlas.reasoning.agent")

_instance: Optional["ReasoningAgentGraph"] = None


class ReasoningAgentGraph:
    """Singleton that processes events through the reasoning graph."""

    def __init__(self) -> None:
        pass

    async def process_event(self, event: AtlasEvent) -> dict[str, Any]:
        """Run a single event through the reasoning graph.

        Returns the processing result dict (stored in atlas_events.processing_result).
        """
        from .graph import run_reasoning_graph
        from ..services.tracing import (
            build_business_trace_context,
            build_reasoning_trace_context,
            tracer,
        )

        state: ReasoningAgentState = {
            "event_id": str(event.id) if event.id else "",
            "event_type": event.event_type,
            "source": event.source,
            "entity_type": event.entity_type,
            "entity_id": event.entity_id,
            "payload": event.payload,
        }
        from ..services.llm_router import get_llm as _get_llm
        _rlm = _get_llm("reasoning")
        span = tracer.start_span(
            span_name="reasoning.process",
            operation_type="reasoning",
            model_name=getattr(_rlm, "model", getattr(_rlm, "model_id", None)) if _rlm else None,
            model_provider=getattr(_rlm, "name", None) if _rlm else None,
            session_id=str(event.id) if event.id else None,
            metadata={
                "business": build_business_trace_context(
                    workflow="reasoning_agent",
                    event_type=event.event_type,
                    source_name=event.source,
                    entity_type=event.entity_type,
                    entity_id=event.entity_id,
                ),
            },
        )

        try:
            result_state = await run_reasoning_graph(state)
        except Exception:
            tracer.end_span(
                span,
                status="failed",
                input_data={"event_type": event.event_type, "payload": event.payload},
                error_message="reasoning graph failed",
                error_type="ReasoningGraphError",
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
        tracer.end_span(
            span,
            status="completed",
            input_data={"event_type": event.event_type, "payload": event.payload},
            output_data=result,
            input_tokens=result_state.get("total_input_tokens"),
            output_tokens=result_state.get("total_output_tokens"),
            metadata={
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
            },
        )
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
