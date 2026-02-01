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
    EmailWorkflowState,
    CalendarWorkflowState,
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
from .email import (
    build_email_graph,
    compile_email_graph,
    run_email_workflow,
    send_email_confirmed,
)
from .calendar import (
    build_calendar_graph,
    compile_calendar_graph,
    run_calendar_workflow,
)
from .workflow_state import (
    ActiveWorkflowState,
    WorkflowStateManager,
    get_workflow_state_manager,
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
    # Email workflow
    "EmailWorkflowState",
    "build_email_graph",
    "compile_email_graph",
    "run_email_workflow",
    "send_email_confirmed",
    # Calendar workflow
    "CalendarWorkflowState",
    "build_calendar_graph",
    "compile_calendar_graph",
    "run_calendar_workflow",
    # Workflow state manager
    "ActiveWorkflowState",
    "WorkflowStateManager",
    "get_workflow_state_manager",
]
