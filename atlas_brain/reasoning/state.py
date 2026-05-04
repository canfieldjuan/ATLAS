"""State definition for the ReasoningAgent LangGraph.

Atlas's `ReasoningAgentState` is a refinement of the canonical
``extracted_reasoning_core.state.ReasoningAgentState``: it inherits every
host-agnostic field (event identity, triage, planning, action results,
synthesis flags, token tracking) and adds the atlas-specific context
slots that the atlas reasoning runner populates.

The seam is structural -- a function annotated against core's TypedDict
accepts an atlas state literal because every key core declares is also
declared here. Whether atlas's granular context fields ultimately migrate
into core (or stay host-side and feed through ports) is a decision
deferred to PR-C4d/e; this slice only anchors atlas's type to the core
contract so future port wiring can rely on it.

Note: core's TypedDict carries a generic ``context: dict[str, Any]``
escape hatch. Atlas does not populate it -- atlas spreads context into
the ten typed slots below -- so reads of ``state["context"]`` on an
atlas state will simply miss. Don't add a parallel write path.
"""

from __future__ import annotations

from typing import Any, Optional

from extracted_reasoning_core.state import ReasoningAgentState as _CoreReasoningAgentState


class ReasoningAgentState(_CoreReasoningAgentState, total=False):
    """Atlas reasoning graph state (extends core's TypedDict)."""

    crm_context: Optional[dict[str, Any]]
    email_history: list[dict[str, Any]]
    voice_turns: list[dict[str, Any]]
    calendar_events: list[dict[str, Any]]
    sms_messages: list[dict[str, Any]]
    graph_facts: list[str]
    recent_events: list[dict[str, Any]]
    market_context: list[dict[str, Any]]
    news_context: list[dict[str, Any]]
    b2b_churn: dict[str, Any]
