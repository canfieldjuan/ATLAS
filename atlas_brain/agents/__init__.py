"""
Agent system for Atlas.

Agents handle reasoning, tool execution, and response generation.
Uses LangGraph for agent orchestration.

Example usage:
    from atlas_brain.agents import get_agent

    agent = get_agent("atlas")
    result = await agent.process(
        input_text="turn on the living room lights",
        session_id="abc-123",
    )
    print(result.response_text)
"""

from .protocols import (
    # Enums
    AgentState,
    # Data classes
    AgentInfo,
    AgentContext,
    ThinkResult,
    ActResult,
    AgentResult,
    # Protocols
    Agent,
    AgentMemory,
    AgentTools,
)

from .memory import (
    AtlasAgentMemory,
    get_agent_memory,
    reset_agent_memory,
)

from .tools import (
    AtlasAgentTools,
    get_agent_tools,
    reset_agent_tools,
)

from .entity_tracker import (
    EntityTracker,
    TrackedEntity,
    has_pronoun,
    extract_pronoun,
)

from .interface import (
    AgentInterface,
    LangGraphAgentAdapter,
    get_agent,
    process_with_fallback,
    reset_agent_cache,
)

__all__ = [
    # Enums
    "AgentState",
    # Data classes
    "AgentInfo",
    "AgentContext",
    "ThinkResult",
    "ActResult",
    "AgentResult",
    # Protocols
    "Agent",
    "AgentMemory",
    "AgentTools",
    # Memory system
    "AtlasAgentMemory",
    "get_agent_memory",
    "reset_agent_memory",
    # Tools system
    "AtlasAgentTools",
    "get_agent_tools",
    "reset_agent_tools",
    # Entity tracking
    "EntityTracker",
    "TrackedEntity",
    "has_pronoun",
    "extract_pronoun",
    # Unified Interface
    "AgentInterface",
    "LangGraphAgentAdapter",
    "get_agent",
    "process_with_fallback",
    "reset_agent_cache",
]
