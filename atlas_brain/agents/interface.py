"""
Unified Agent Interface.

Provides a common interface for both legacy (BaseAgent) and new (LangGraph)
agent implementations, enabling gradual migration without breaking changes.
"""

import logging
from typing import Any, Optional, Protocol, runtime_checkable

from .protocols import AgentContext, AgentResult

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


class LegacyAgentAdapter:
    """Adapts old BaseAgent to unified interface."""

    def __init__(self, agent: Any):
        """
        Initialize adapter with a legacy agent.

        Args:
            agent: A BaseAgent instance (AtlasAgent, HomeAgent, etc.)
        """
        self._agent = agent

    async def process(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        input_type: str = "text",
        runtime_context: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """Process input through legacy agent."""
        context = AgentContext(
            input_text=input_text,
            input_type=input_type,
            session_id=session_id,
            speaker_id=speaker_id,
            runtime_context=runtime_context or {},
        )
        return await self._agent.run(context)


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
) -> AgentInterface:
    """
    Get an agent instance with the configured backend.

    Args:
        agent_type: Type of agent ("atlas", "home", "receptionist")
        session_id: Session ID for the agent
        backend: Override backend ("legacy" or "langgraph")

    Returns:
        Agent adapter implementing AgentInterface
    """
    from ..config import settings

    # Determine backend
    use_backend = backend or settings.agent.backend

    if use_backend == "langgraph":
        return _get_langgraph_agent(agent_type, session_id)
    else:
        return _get_legacy_agent(agent_type, session_id)


def _get_legacy_agent(
    agent_type: str,
    session_id: Optional[str] = None,
) -> LegacyAgentAdapter:
    """Get legacy agent wrapped in adapter."""
    if agent_type == "atlas":
        from .atlas import get_atlas_agent
        agent = get_atlas_agent(session_id=session_id)
        return LegacyAgentAdapter(agent)

    elif agent_type == "home":
        from .home import get_home_agent
        agent = get_home_agent(session_id=session_id)
        return LegacyAgentAdapter(agent)

    elif agent_type == "receptionist":
        from .receptionist import create_receptionist_agent
        agent = create_receptionist_agent()
        return LegacyAgentAdapter(agent)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def _get_langgraph_agent(
    agent_type: str,
    session_id: Optional[str] = None,
) -> LangGraphAgentAdapter:
    """Get LangGraph agent wrapped in adapter."""
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
        graph = get_receptionist_agent_langgraph(session_id=session_id)
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
    Process input with fallback to legacy agent on failure.

    Args:
        input_text: User input text
        agent_type: Type of agent to use
        session_id: Session ID
        speaker_id: Speaker ID
        input_type: Input type
        runtime_context: Runtime context

    Returns:
        AgentResult from primary or fallback agent
    """
    from ..config import settings

    # Try primary backend
    try:
        agent = get_agent(
            agent_type=agent_type,
            session_id=session_id,
            backend=settings.agent.backend,
        )
        result = await agent.process(
            input_text=input_text,
            session_id=session_id,
            speaker_id=speaker_id,
            input_type=input_type,
            runtime_context=runtime_context,
        )

        if result.success:
            return result

        # If primary failed and fallback is enabled
        if settings.agent.fallback_enabled and settings.agent.backend == "langgraph":
            logger.warning(
                "LangGraph agent failed, falling back to legacy: %s",
                result.error,
            )
            return await _fallback_to_legacy(
                input_text=input_text,
                agent_type=agent_type,
                session_id=session_id,
                speaker_id=speaker_id,
                input_type=input_type,
                runtime_context=runtime_context,
            )

        return result

    except Exception as e:
        logger.exception("Agent processing failed: %s", e)

        # Try fallback if enabled
        if settings.agent.fallback_enabled and settings.agent.backend == "langgraph":
            logger.warning("Falling back to legacy agent after exception")
            return await _fallback_to_legacy(
                input_text=input_text,
                agent_type=agent_type,
                session_id=session_id,
                speaker_id=speaker_id,
                input_type=input_type,
                runtime_context=runtime_context,
            )

        return AgentResult(
            success=False,
            error=str(e),
            response_text="I encountered an error processing your request.",
        )


async def _fallback_to_legacy(
    input_text: str,
    agent_type: str,
    session_id: Optional[str],
    speaker_id: Optional[str],
    input_type: str,
    runtime_context: Optional[dict[str, Any]],
) -> AgentResult:
    """Fall back to legacy agent."""
    agent = _get_legacy_agent(agent_type, session_id)
    return await agent.process(
        input_text=input_text,
        session_id=session_id,
        speaker_id=speaker_id,
        input_type=input_type,
        runtime_context=runtime_context,
    )


def reset_agent_cache() -> None:
    """Reset cached agent instances."""
    global _atlas_agent, _home_agent
    _atlas_agent = None
    _home_agent = None
