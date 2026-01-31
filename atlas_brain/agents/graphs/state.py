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


class BookingWorkflowState(TypedDict, total=False):
    """
    State for multi-step booking workflow.

    Demonstrates graph-driven tool chaining:
    lookup_customer -> check_availability -> book_appointment

    The LLM only makes decisions at specific nodes.
    LangGraph handles orchestration and routing.
    """

    # Input
    input_text: str
    session_id: Optional[str]

    # Parsed from input (LLM extracts)
    customer_name: Optional[str]
    customer_phone: Optional[str]
    requested_date: Optional[str]
    requested_time: Optional[str]
    service_type: Optional[str]

    # From lookup_customer tool
    customer_found: bool
    customer_id: Optional[str]
    customer_email: Optional[str]
    customer_address: Optional[str]

    # From check_availability tool
    slot_available: bool
    alternative_slots: list[str]

    # From book_appointment tool
    booking_confirmed: bool
    booking_id: Optional[str]
    confirmation_details: dict[str, Any]

    # Workflow control
    current_step: str  # "parse", "lookup", "availability", "book", "complete"
    needs_info: list[str]  # Fields still needed from user
    awaiting_user_input: bool

    # Output
    response: str
    error: Optional[str]

    # Timing
    total_ms: float
    step_timings: dict[str, float]


class ReminderWorkflowState(TypedDict, total=False):
    """
    State for reminder management workflow.

    Handles: create, list, complete, delete reminders.
    Routes based on detected intent within the graph.
    """

    # Input
    input_text: str
    session_id: Optional[str]

    # Intent classification (determined by graph)
    intent: str  # "create", "list", "complete", "delete", "unknown"

    # Parsed from input (for create)
    reminder_message: Optional[str]
    reminder_time: Optional[str]  # Natural language time ("in 30 minutes")
    repeat_pattern: Optional[str]  # "daily", "weekly", etc.

    # Parsed datetime (after parsing)
    parsed_due_at: Optional[str]  # ISO format

    # For complete/delete operations
    target_reminder_id: Optional[str]
    target_by_index: Optional[int]  # "delete the first one"
    complete_all: bool

    # Results from operations
    reminder_created: bool
    created_reminder_id: Optional[str]

    reminders_listed: bool
    reminder_list: list[dict[str, Any]]
    reminder_count: int

    reminder_completed: bool
    completed_reminder_id: Optional[str]

    reminder_deleted: bool
    deleted_reminder_id: Optional[str]

    # Workflow control
    current_step: str  # "classify", "parse", "execute", "respond"
    needs_clarification: bool
    clarification_prompt: Optional[str]

    # Output
    response: str
    error: Optional[str]

    # Timing
    total_ms: float
    step_timings: dict[str, float]


class SecurityWorkflowState(TypedDict, total=False):
    """
    State for security system workflow.

    Handles: cameras, detections, zones.
    Routes based on detected intent within the graph.
    """

    # Input
    input_text: str
    session_id: Optional[str]

    # Intent classification (determined by graph)
    # camera_list, camera_status, camera_record_start, camera_record_stop, camera_ptz
    # detection_current, detection_query, detection_person_location, detection_motion
    # zone_list, zone_status, zone_arm, zone_disarm
    intent: str

    # Camera parameters
    camera_name: Optional[str]
    camera_id: Optional[str]
    record_duration: Optional[int]  # seconds
    ptz_action: Optional[str]  # "pan_left", "pan_right", "tilt_up", etc.
    ptz_amount: Optional[float]

    # Detection parameters
    detection_type: Optional[str]  # "person", "vehicle", "motion", etc.
    location: Optional[str]  # location to check
    time_range_start: Optional[str]  # ISO format
    time_range_end: Optional[str]
    min_confidence: Optional[float]
    limit: Optional[int]

    # Zone parameters
    zone_name: Optional[str]
    zone_id: Optional[str]
    arm_mode: Optional[str]  # "away", "home", "night"

    # Results - Camera operations
    cameras_listed: bool
    camera_list: list[dict[str, Any]]
    camera_count: int
    camera_status: Optional[dict[str, Any]]
    recording_started: bool
    recording_stopped: bool

    # Results - Detection operations
    detections_retrieved: bool
    detection_list: list[dict[str, Any]]
    detection_count: int
    person_found: bool
    person_location_result: Optional[dict[str, Any]]
    motion_events: list[dict[str, Any]]

    # Results - Zone operations
    zones_listed: bool
    zone_list: list[dict[str, Any]]
    zone_count: int
    zone_status_result: Optional[dict[str, Any]]
    zone_armed: bool
    zone_disarmed: bool

    # Workflow control
    current_step: str  # "classify", "parse", "execute", "respond"
    needs_clarification: bool
    clarification_prompt: Optional[str]

    # Output
    response: str
    error: Optional[str]

    # Timing
    total_ms: float
    step_timings: dict[str, float]


class PresenceWorkflowState(TypedDict, total=False):
    """
    State for presence-aware device control workflow.

    Handles: lights, media, scenes near user, and location queries.
    Routes based on detected intent within the graph.
    """

    # Input
    input_text: str
    session_id: Optional[str]
    user_id: Optional[str]

    # Intent classification
    # lights_control, media_control, scene_set, where_am_i
    intent: str

    # Presence context (resolved from presence service)
    current_room_id: Optional[str]
    current_room_name: Optional[str]
    presence_confidence: float
    presence_source: Optional[str]

    # Lights parameters
    light_action: Optional[str]  # on, off, toggle
    brightness: Optional[int]  # 0-100
    light_entities: list[str]

    # Media parameters
    media_action: Optional[str]  # on, off, play, pause, stop
    media_entities: list[str]

    # Scene parameters
    scene_name: Optional[str]  # bright, dim, cozy, movie, focus, relax, off

    # Results
    action_executed: bool
    devices_controlled: list[str]
    location_reported: bool

    # Workflow control
    current_step: str
    needs_clarification: bool
    clarification_prompt: Optional[str]

    # Output
    response: str
    error: Optional[str]

    # Timing
    total_ms: float
    step_timings: dict[str, float]