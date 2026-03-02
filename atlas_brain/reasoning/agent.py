"""ReasoningAgentGraph -- singleton wrapper around the reasoning graph."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional
from uuid import UUID

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

        state: ReasoningAgentState = {
            "event_id": str(event.id) if event.id else "",
            "event_type": event.event_type,
            "source": event.source,
            "entity_type": event.entity_type,
            "entity_id": event.entity_id,
            "payload": event.payload,
        }

        try:
            result_state = await run_reasoning_graph(state)
        except Exception:
            logger.error(
                "Reasoning graph failed for event %s", event.id, exc_info=True
            )
            return {
                "status": "error",
                "triage_priority": state.get("triage_priority", "unknown"),
            }

        return {
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
