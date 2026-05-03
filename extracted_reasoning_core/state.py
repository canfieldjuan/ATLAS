"""State definition for extracted reasoning runners."""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class ReasoningAgentState(TypedDict, total=False):
    """State flowing through reasoning graph nodes."""

    event_id: str
    event_type: str
    source: str
    entity_type: Optional[str]
    entity_id: Optional[str]
    payload: dict[str, Any]
    triage_priority: str
    triage_reasoning: str
    needs_reasoning: bool
    context: dict[str, Any]
    entity_locked: bool
    lock_holder: Optional[str]
    queued: bool
    reasoning_output: str
    connections_found: list[str]
    recommended_actions: list[dict[str, Any]]
    rationale: str
    planned_actions: list[dict[str, Any]]
    action_results: list[dict[str, Any]]
    summary: str
    should_notify: bool
    notification_sent: bool
    total_input_tokens: int
    total_output_tokens: int


__all__ = ["ReasoningAgentState"]
