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
    run_booking_workflow,
    BOOKING_WORKFLOW_TYPE,
)
from .reminder import (
    run_reminder_workflow,
    REMINDER_WORKFLOW_TYPE,
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
    run_email_workflow,
    EMAIL_WORKFLOW_TYPE,
)
from .email_query import (
    run_email_query_workflow,
    EMAIL_QUERY_WORKFLOW_TYPE,
)
from .calendar import (
    run_calendar_workflow,
    CALENDAR_WORKFLOW_TYPE,
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
    "run_booking_workflow",
    "BOOKING_WORKFLOW_TYPE",
    # Reminder workflow
    "run_reminder_workflow",
    "REMINDER_WORKFLOW_TYPE",
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
    "run_email_workflow",
    "EMAIL_WORKFLOW_TYPE",
    # Email query workflow
    "run_email_query_workflow",
    "EMAIL_QUERY_WORKFLOW_TYPE",
    # Calendar workflow
    "CalendarWorkflowState",
    "run_calendar_workflow",
    "CALENDAR_WORKFLOW_TYPE",
    # Workflow state manager
    "ActiveWorkflowState",
    "WorkflowStateManager",
    "get_workflow_state_manager",
]
