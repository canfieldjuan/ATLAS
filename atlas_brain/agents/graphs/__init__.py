"""
LangGraph-based agent implementations.

This package contains the LangGraph StateGraph implementations
for Atlas agents, replacing the custom React pattern.
"""

from .state import (
    AgentState,
    HomeAgentState,
    AtlasAgentState,
    ReceptionistAgentState,
    BookingWorkflowState,
    ReminderWorkflowState,
    SecurityWorkflowState,
    PresenceWorkflowState,
)
from .home import HomeAgentGraph, get_home_agent_langgraph
from .atlas import AtlasAgentGraph, get_atlas_agent_langgraph
from .receptionist import ReceptionistAgentGraph, get_receptionist_agent_langgraph
from .streaming import (
    StreamingHomeAgent,
    StreamingAtlasAgent,
    get_streaming_home_agent,
    get_streaming_atlas_agent,
    stream_to_tts,
)
from .booking import (
    build_booking_graph,
    compile_booking_graph,
    run_booking_workflow,
)
from .reminder import (
    build_reminder_graph,
    compile_reminder_graph,
    run_reminder_workflow,
)
from .security import (
    build_security_graph,
    compile_security_graph,
    run_security_workflow,
)
from .presence import (
    build_presence_graph,
    compile_presence_graph,
    run_presence_workflow,
)

__all__ = [
    # State schemas
    "AgentState",
    "HomeAgentState",
    "AtlasAgentState",
    "ReceptionistAgentState",
    "BookingWorkflowState",
    "ReminderWorkflowState",
    "SecurityWorkflowState",
    # Agent graphs
    "HomeAgentGraph",
    "AtlasAgentGraph",
    "ReceptionistAgentGraph",
    # Factory functions
    "get_home_agent_langgraph",
    "get_atlas_agent_langgraph",
    "get_receptionist_agent_langgraph",
    # Streaming agents
    "StreamingHomeAgent",
    "StreamingAtlasAgent",
    "get_streaming_home_agent",
    "get_streaming_atlas_agent",
    "stream_to_tts",
    # Booking workflow
    "build_booking_graph",
    "compile_booking_graph",
    "run_booking_workflow",
    # Reminder workflow
    "build_reminder_graph",
    "compile_reminder_graph",
    "run_reminder_workflow",
    # Security workflow
    "build_security_graph",
    "compile_security_graph",
    "run_security_workflow",
    # Presence workflow
    "PresenceWorkflowState",
    "build_presence_graph",
    "compile_presence_graph",
    "run_presence_workflow",
]
