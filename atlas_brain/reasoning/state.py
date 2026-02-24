"""State definition for the ReasoningAgent LangGraph."""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class ReasoningAgentState(TypedDict, total=False):
    """State flowing through the reasoning graph nodes."""

    # Input
    event_id: str
    event_type: str
    source: str
    entity_type: Optional[str]
    entity_id: Optional[str]
    payload: dict[str, Any]

    # Triage
    triage_priority: str  # "skip", "low", "medium", "high"
    triage_reasoning: str
    needs_reasoning: bool

    # Context aggregation
    crm_context: Optional[dict[str, Any]]
    email_history: list[dict[str, Any]]
    voice_turns: list[dict[str, Any]]
    calendar_events: list[dict[str, Any]]
    sms_messages: list[dict[str, Any]]
    graph_facts: list[str]
    recent_events: list[dict[str, Any]]
    market_context: list[dict[str, Any]]
    news_context: list[dict[str, Any]]

    # Lock check
    entity_locked: bool
    lock_holder: Optional[str]
    queued: bool

    # Reasoning
    reasoning_output: str
    connections_found: list[str]
    recommended_actions: list[dict[str, Any]]
    rationale: str

    # Action planning
    planned_actions: list[dict[str, Any]]

    # Execution
    action_results: list[dict[str, Any]]

    # Synthesis
    summary: str
    should_notify: bool
    notification_sent: bool
