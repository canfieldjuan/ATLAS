"""
Base Agent implementation with shared utilities.

Provides common functionality for all Agent implementations,
including timing, logging, and default implementations.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from .protocols import (
    ActResult,
    AgentContext,
    AgentInfo,
    AgentMemory,
    AgentResult,
    AgentState,
    AgentTools,
    ThinkResult,
)

logger = logging.getLogger("atlas.agents")


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000


class BaseAgent(ABC):
    """
    Base class for Agent implementations.

    Provides:
    - Logging and timing utilities
    - Default implementations of protocol methods
    - State management
    - Error handling patterns

    Subclasses must implement:
    - _do_think(): Core thinking logic
    - _do_act(): Core action execution
    - _do_respond(): Core response generation
    """

    def __init__(
        self,
        name: str,
        description: str,
        memory: Optional[AgentMemory] = None,
        tools: Optional[AgentTools] = None,
    ):
        """
        Initialize base agent.

        Args:
            name: Agent identifier
            description: Human-readable description
            memory: AgentMemory implementation (optional, can set later)
            tools: AgentTools implementation (optional, can set later)
        """
        self._name = name
        self._description = description
        self._memory = memory
        self._tools = tools
        self._state = AgentState.IDLE
        self._capabilities: list[str] = []

        self._logger = logging.getLogger(f"atlas.agents.{name}")

    @property
    def info(self) -> AgentInfo:
        """Return metadata about this agent."""
        return AgentInfo(
            name=self._name,
            description=self._description,
            capabilities=self._capabilities,
        )

    @property
    def state(self) -> AgentState:
        """Return current agent state."""
        return self._state

    @property
    def memory(self) -> Optional[AgentMemory]:
        """Return memory interface."""
        return self._memory

    @memory.setter
    def memory(self, value: AgentMemory) -> None:
        """Set memory interface."""
        self._memory = value

    @property
    def tools(self) -> Optional[AgentTools]:
        """Return tools interface."""
        return self._tools

    @tools.setter
    def tools(self, value: AgentTools) -> None:
        """Set tools interface."""
        self._tools = value

    def reset(self) -> None:
        """Reset agent state to IDLE."""
        self._state = AgentState.IDLE
        self._logger.debug("Agent reset to IDLE state")

    async def run(
        self,
        context: AgentContext,
    ) -> AgentResult:
        """
        Main entry point: Process input and return result.

        Orchestrates the think -> act -> respond flow with
        proper error handling and timing.

        Args:
            context: AgentContext with input, session info, and history

        Returns:
            AgentResult with response and action results
        """
        start_time = time.perf_counter()

        self._logger.info(
            "Processing input: %.50s... (session=%s, speaker=%s)",
            context.input_text,
            context.session_id,
            context.speaker_id,
        )

        result = AgentResult(success=False)

        try:
            # Phase 1: Think - Analyze and decide
            self._state = AgentState.THINKING
            with Timer() as think_timer:
                think_result = await self.think(context)
            result.think_ms = think_timer.duration_ms
            result.memory_ms = getattr(think_result, "memory_ms", 0.0)

            self._logger.debug(
                "Think result: action_type=%s, confidence=%.2f, duration=%.1fms",
                think_result.action_type,
                think_result.confidence,
                think_timer.duration_ms,
            )

            # Phase 2: Act - Execute if needed
            act_result: Optional[ActResult] = None
            if think_result.action_type in ("device_command", "tool_use"):
                self._state = AgentState.EXECUTING
                with Timer() as act_timer:
                    act_result = await self.act(context, think_result)
                result.act_ms = act_timer.duration_ms
                result.tools_ms = getattr(act_result, "tools_ms", 0.0)
                result.action_results = act_result.action_results

                self._logger.debug(
                    "Act result: success=%s, duration=%.1fms",
                    act_result.success,
                    act_timer.duration_ms,
                )

            # Phase 3: Respond - Generate response
            self._state = AgentState.RESPONDING
            with Timer() as respond_timer:
                response_text = await self.respond(context, think_result, act_result)
            result.llm_ms = respond_timer.duration_ms

            self._logger.debug(
                "Response generated: %.50s... (%.1fms)",
                response_text,
                respond_timer.duration_ms,
            )

            # Phase 4: Store - Persist conversation
            with Timer() as storage_timer:
                await self.store_turn(context, AgentResult(
                    success=True,
                    response_text=response_text,
                    action_type=think_result.action_type,
                    intent=think_result.intent,
                    action_results=result.action_results,
                ))
            result.storage_ms = storage_timer.duration_ms

            # Build final result
            result.success = True
            result.response_text = response_text
            result.action_type = think_result.action_type
            result.intent = think_result.intent

        except Exception as e:
            self._state = AgentState.ERROR
            self._logger.exception("Error during agent processing: %s", e)
            result.success = False
            result.error = str(e)
            result.error_code = "AGENT_ERROR"
            result.response_text = f"I'm sorry, I encountered an error: {e}"

        finally:
            # Calculate total time
            end_time = time.perf_counter()
            result.total_ms = (end_time - start_time) * 1000

            # Reset state
            self._state = AgentState.IDLE

            self._logger.info(
                "Processing complete: success=%s, total=%.1fms, breakdown=%s",
                result.success,
                result.total_ms,
                result.timing_breakdown(),
            )

        return result

    async def think(
        self,
        context: AgentContext,
    ) -> ThinkResult:
        """
        Analyze input and decide what to do.

        Wraps _do_think() with timing and error handling.
        """
        try:
            return await self._do_think(context)
        except Exception as e:
            self._logger.exception("Error in think phase: %s", e)
            return ThinkResult(
                action_type="none",
                confidence=0.0,
                reasoning=f"Error during analysis: {e}",
            )

    async def act(
        self,
        context: AgentContext,
        think_result: ThinkResult,
    ) -> ActResult:
        """
        Execute actions based on think result.

        Wraps _do_act() with timing and error handling.
        """
        try:
            return await self._do_act(context, think_result)
        except Exception as e:
            self._logger.exception("Error in act phase: %s", e)
            return ActResult(
                success=False,
                action_type=think_result.action_type,
                error=str(e),
                error_code="ACT_ERROR",
            )

    async def respond(
        self,
        context: AgentContext,
        think_result: ThinkResult,
        act_result: Optional[ActResult],
    ) -> str:
        """
        Generate response text.

        Wraps _do_respond() with error handling.
        """
        try:
            return await self._do_respond(context, think_result, act_result)
        except Exception as e:
            self._logger.exception("Error in respond phase: %s", e)
            return f"I'm sorry, I had trouble generating a response: {e}"

    async def store_turn(
        self,
        context: AgentContext,
        result: AgentResult,
    ) -> None:
        """
        Persist the conversation turn.

        Default implementation uses memory interface if available.
        """
        if not self._memory or not context.session_id:
            self._logger.debug("Skipping turn storage (no memory or session)")
            return

        try:
            # Store user turn
            await self._memory.add_turn(
                session_id=context.session_id,
                role="user",
                content=context.input_text,
                speaker_id=context.speaker_id,
                intent=result.intent.action if result.intent and hasattr(result.intent, "action") else None,
                turn_type="command" if result.action_type == "device_command" else "conversation",
            )

            # Store assistant turn
            if result.response_text:
                await self._memory.add_turn(
                    session_id=context.session_id,
                    role="assistant",
                    content=result.response_text,
                    turn_type="command" if result.action_type == "device_command" else "conversation",
                )

            self._logger.debug("Stored conversation turns for session %s", context.session_id)

        except Exception as e:
            self._logger.warning("Failed to store conversation turn: %s", e)

    # Abstract methods for subclasses to implement

    @abstractmethod
    async def _do_think(
        self,
        context: AgentContext,
    ) -> ThinkResult:
        """
        Core thinking logic - must be implemented by subclasses.

        Responsibilities:
        - Parse intent from input
        - Retrieve relevant memory/context
        - Decide: conversation, device command, tool use, or none
        """
        ...

    @abstractmethod
    async def _do_act(
        self,
        context: AgentContext,
        think_result: ThinkResult,
    ) -> ActResult:
        """
        Core action execution - must be implemented by subclasses.

        Responsibilities:
        - Execute device commands via ActionDispatcher
        - Run tools (weather, traffic, etc.)
        - Gather data for response generation
        """
        ...

    @abstractmethod
    async def _do_respond(
        self,
        context: AgentContext,
        think_result: ThinkResult,
        act_result: Optional[ActResult],
    ) -> str:
        """
        Core response generation - must be implemented by subclasses.

        Responsibilities:
        - For device commands: Generate confirmation message
        - For conversations: Call LLM with context
        - For errors: Generate error message
        """
        ...
