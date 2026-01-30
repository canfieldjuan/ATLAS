"""
Shared state schemas for LangGraph agents.

These TypedDict classes define the state that flows through
the LangGraph StateGraphs, replacing ThinkResult + ActResult.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import BaseMessage


def add_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    """Reducer that appends messages to the list."""
    return left + right


@dataclass
class ActionResult:
    """Result from executing an action or tool."""

    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Intent:
    """Parsed intent from user input."""

    action: str  # turn_on, turn_off, toggle, set_brightness, query, conversation
    target_type: Optional[str] = None  # light, switch, media_player, tool, etc.
    target_name: Optional[str] = None  # kitchen, living room, etc.
    target_id: Optional[str] = None  # entity_id if resolved
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_query: str = ""


class AgentState(TypedDict, total=False):
    """
    Base state for all LangGraph agents.

    Uses TypedDict for LangGraph compatibility with optional fields.
    """

    # Input
    input_text: str
    input_type: str  # "text", "voice", "vision"

    # Session context
    session_id: Optional[str]
    user_id: Optional[str]
    speaker_id: Optional[str]
    speaker_confidence: float

    # Conversation (with message reducer for streaming)
    messages: Annotated[list[BaseMessage], add_messages]

    # Runtime context
    runtime_context: dict[str, Any]

    # Classification result
    action_type: str  # "device_command", "tool_use", "conversation", "none"
    confidence: float

    # Parsed intent (for device commands)
    intent: Optional[Intent]

    # Action execution result
    action_result: Optional[ActionResult]

    # Tool results
    tool_results: dict[str, Any]

    # Final response
    response: str

    # Error handling
    error: Optional[str]
    error_code: Optional[str]

    # Timing (milliseconds)
    classify_ms: float
    think_ms: float
    act_ms: float
    respond_ms: float
    total_ms: float


class HomeAgentState(AgentState):
    """
    State for HomeAgent LangGraph.

    Handles device commands with optional pronoun resolution.
    """

    # Entity tracking for pronoun resolution
    last_entity_type: Optional[str]
    last_entity_name: Optional[str]
    last_entity_id: Optional[str]

    # Device-specific
    resolved_entity_id: Optional[str]

    # Whether we need LLM for response (vs template)
    needs_llm: bool


class AtlasAgentState(AgentState):
    """
    State for AtlasAgent LangGraph.

    Main router agent that delegates to sub-agents or handles directly.
    """

    # Mode routing
    current_mode: str  # "home", "receptionist", "default"

    # Memory/context retrieval
    retrieved_context: Optional[str]
    memory_ms: float

    # Sub-agent delegation
    delegate_to: Optional[str]  # "home", "receptionist", None

    # LLM tool calling
    tools_to_call: list[str]
    tools_executed: list[str]


class ReceptionistAgentState(AgentState):
    """
    State for ReceptionistAgent LangGraph.

    Handles phone calls with appointment booking flow.
    """

    # Call state
    call_phase: str  # "greeting", "gathering", "confirming", "booking", "farewell"
    call_id: Optional[str]
    caller_number: Optional[str]

    # Gathered information
    customer_name: Optional[str]
    customer_phone: Optional[str]
    customer_address: Optional[str]
    appointment_time: Optional[str]
    service_type: Optional[str]

    # Booking result
    booking_confirmed: bool
    booking_id: Optional[str]

    # Phone-specific
    is_phone_call: bool
    use_phone_tts: bool
