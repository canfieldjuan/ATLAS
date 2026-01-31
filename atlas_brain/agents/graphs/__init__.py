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

__all__ = [
    # State schemas
    "AgentState",
    "HomeAgentState",
    "AtlasAgentState",
    "ReceptionistAgentState",
    "BookingWorkflowState",
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
]
