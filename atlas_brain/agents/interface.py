"""
Unified Agent Interface.

Provides a common interface for LangGraph agent implementations.
"""

import logging
from typing import Any, Optional, Protocol, runtime_checkable

from .protocols import AgentResult

logger = logging.getLogger("atlas.agents.interface")


@runtime_checkable
class AgentInterface(Protocol):
    """Common interface for all agent implementations."""

    async def process(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        input_type: str = "text",
        runtime_context: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Process input and return result.

        Args:
            input_text: User input text
            session_id: Session ID for persistence
            speaker_id: Identified speaker name
            input_type: Input source type (text, voice, vision)
            runtime_context: Additional runtime context

        Returns:
            AgentResult with response and metadata
        """
        ...


class LangGraphAgentAdapter:
    """Adapts LangGraph agent to unified interface."""

    def __init__(self, graph: Any):
        """
        Initialize adapter with a LangGraph agent.

        Args:
            graph: A LangGraph agent (HomeAgentGraph, AtlasAgentGraph, etc.)
        """
        self._graph = graph

    async def process(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        input_type: str = "text",
        runtime_context: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """Process input through LangGraph agent."""
        result = await self._graph.run(
            input_text=input_text,
            session_id=session_id,
            speaker_id=speaker_id,
            input_type=input_type,
            runtime_context=runtime_context or {},
        )

        # Convert dict result to AgentResult
        timing = result.get("timing", {})
        return AgentResult(
            success=result.get("success", False),
            response_text=result.get("response_text"),
            action_type=result.get("action_type", "none"),
            intent=result.get("intent"),
            action_results=result.get("action_results", []),
            error=result.get("error"),
            total_ms=timing.get("total", 0),
            think_ms=timing.get("think", 0) + timing.get("classify", 0),
            act_ms=timing.get("act", 0),
            llm_ms=timing.get("respond", 0),
        )


# Agent factory singletons
_atlas_agent: Optional[AgentInterface] = None
_home_agent: Optional[AgentInterface] = None


def get_agent(
    agent_type: str = "atlas",
    session_id: Optional[str] = None,
    backend: Optional[str] = None,
    business_context: Optional[Any] = None,
) -> AgentInterface:
    """
    Get an agent instance.

    Args:
        agent_type: Type of agent ("atlas", "home", "receptionist")
        session_id: Session ID for the agent
        backend: Ignored (kept for backwards compatibility)
        business_context: Business context for receptionist agent

    Returns:
        Agent adapter implementing AgentInterface
    """
    if agent_type == "atlas":
        from .graphs import get_atlas_agent_langgraph
        graph = get_atlas_agent_langgraph(session_id=session_id)
        return LangGraphAgentAdapter(graph)

    elif agent_type == "home":
        from .graphs import get_home_agent_langgraph
        graph = get_home_agent_langgraph(session_id=session_id)
        return LangGraphAgentAdapter(graph)

    elif agent_type == "receptionist":
        from .graphs import get_receptionist_agent_langgraph
        graph = get_receptionist_agent_langgraph(
            business_context=business_context,
            session_id=session_id,
        )
        return LangGraphAgentAdapter(graph)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


async def process_with_fallback(
    input_text: str,
    agent_type: str = "atlas",
    session_id: Optional[str] = None,
    speaker_id: Optional[str] = None,
    input_type: str = "text",
    runtime_context: Optional[dict[str, Any]] = None,
) -> AgentResult:
    """
    Process input with error handling.

    Args:
        input_text: User input text
        agent_type: Type of agent to use
        session_id: Session ID
        speaker_id: Speaker ID
        input_type: Input type
        runtime_context: Runtime context

    Returns:
        AgentResult from agent processing
    """
    try:
        agent = get_agent(
            agent_type=agent_type,
            session_id=session_id,
        )
        return await agent.process(
            input_text=input_text,
            session_id=session_id,
            speaker_id=speaker_id,
            input_type=input_type,
            runtime_context=runtime_context,
        )

    except Exception as e:
        logger.exception("Agent processing failed: %s", e)
        return AgentResult(
            success=False,
            error=str(e),
            response_text="I encountered an error processing your request.",
        )


def reset_agent_cache() -> None:
    """Reset cached agent instances."""
    global _atlas_agent, _home_agent
    _atlas_agent = None
    _home_agent = None
