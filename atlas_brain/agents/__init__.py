"""
Agent system for Atlas.

Agents handle reasoning, tool execution, and response generation.
The Orchestrator handles audio I/O and delegates to Agents.

Example usage:
    from atlas_brain.agents import AtlasAgent, AgentContext

    agent = AtlasAgent()
    context = AgentContext(
        input_text="turn on the living room lights",
        session_id="abc-123",
    )
    result = await agent.run(context)
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

from .base import (
    BaseAgent,
    Timer,
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

from .atlas import (
    AtlasAgent,
    get_atlas_agent,
    reset_atlas_agent,
)

from .entity_tracker import (
    EntityTracker,
    TrackedEntity,
    has_pronoun,
    extract_pronoun,
)

from .receptionist import (
    ReceptionistAgent,
    get_receptionist_agent,
    create_receptionist_agent,
    reset_receptionist_agent,
)

from .home import (
    HomeAgent,
    get_home_agent,
    create_home_agent,
)

from .interface import (
    AgentInterface,
    LegacyAgentAdapter,
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
    # Base class
    "BaseAgent",
    "Timer",
    # Memory system
    "AtlasAgentMemory",
    "get_agent_memory",
    "reset_agent_memory",
    # Tools system
    "AtlasAgentTools",
    "get_agent_tools",
    "reset_agent_tools",
    # Atlas Agent
    "AtlasAgent",
    "get_atlas_agent",
    "reset_atlas_agent",
    # Entity tracking
    "EntityTracker",
    "TrackedEntity",
    "has_pronoun",
    "extract_pronoun",
    # Receptionist Agent
    "ReceptionistAgent",
    "get_receptionist_agent",
    "create_receptionist_agent",
    "reset_receptionist_agent",
    # Home Agent
    "HomeAgent",
    "get_home_agent",
    "create_home_agent",
    # Unified Interface
    "AgentInterface",
    "LegacyAgentAdapter",
    "LangGraphAgentAdapter",
    "get_agent",
    "process_with_fallback",
    "reset_agent_cache",
]
