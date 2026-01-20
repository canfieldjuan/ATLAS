"""
Home Agent - Device control agent for Atlas Brain.

Handles device commands: lights, TV, scenes.
Fast path - no state machine, direct execution.
"""

import asyncio
import logging
import time
from typing import Any, Optional

from .base import BaseAgent
from .protocols import (
    ActResult,
    AgentContext,
    ThinkResult,
)
from .tools import AtlasAgentTools, get_agent_tools
from .entity_tracker import EntityTracker, has_pronoun, extract_pronoun

logger = logging.getLogger("atlas.agents.home")

# Global CUDA lock - shared across agents
_cuda_lock: Optional[asyncio.Lock] = None


def _get_cuda_lock() -> asyncio.Lock:
    """Get or create global CUDA lock."""
    global _cuda_lock
    if _cuda_lock is None:
        _cuda_lock = asyncio.Lock()
    return _cuda_lock


class HomeAgent(BaseAgent):
    """
    Home device control agent.

    Handles:
    - Light control (on/off, brightness, color)
    - Media control (TV, speakers)
    - Scene activation
    - Device queries

    Fast path execution - no state machine needed.
    """

    def __init__(
        self,
        tools: Optional[AtlasAgentTools] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize Home agent.

        Args:
            tools: Tools system (lazy-loaded if None)
            session_id: Default session ID for persistence
        """
        super().__init__(
            name="home",
            description="Home device control agent",
            tools=tools,
        )

        self._capabilities = [
            "device_control",
            "lights",
            "media",
            "scenes",
        ]

        self._session_id = session_id
        self._llm = None
        self._entity_tracker = EntityTracker()

    def _get_tools(self) -> AtlasAgentTools:
        """Get or create tools system."""
        if self._tools is None:
            self._tools = get_agent_tools()
        return self._tools

    def _get_llm(self):
        """Get or create LLM service."""
        if self._llm is None:
            from ..services import llm_registry
            self._llm = llm_registry.get_active()
        return self._llm

    async def _do_think(
        self,
        context: AgentContext,
    ) -> ThinkResult:
        """
        Analyze input and decide what to do.

        For HomeAgent, we only handle device commands.
        """
        tools = self._get_tools()

        result = ThinkResult(
            action_type="none",
            confidence=0.0,
        )

        start_time = time.perf_counter()

        # Parse intent
        intent = await tools.parse_intent(context.input_text)

        # Resolve pronouns if needed
        if intent and not intent.target_name and has_pronoun(context.input_text):
            pronoun = extract_pronoun(context.input_text)
            if pronoun:
                # Don't filter by type if target_type is generic "device"
                filter_type = intent.target_type if intent.target_type != "device" else None
                resolved = self._entity_tracker.resolve_pronoun(
                    pronoun,
                    entity_type=filter_type,
                )
                if resolved:
                    intent.target_name = resolved.entity_name
                    # Use resolved type, especially if original was generic "device"
                    if intent.target_type == "device" or not intent.target_type:
                        intent.target_type = resolved.entity_type
                    if resolved.entity_id:
                        intent.target_id = resolved.entity_id
                    logger.info(
                        "Resolved '%s' -> %s/%s",
                        pronoun,
                        resolved.entity_type,
                        resolved.entity_name,
                    )

        if intent:
            result.intent = intent
            result.confidence = intent.confidence

            # Device actions we handle
            device_actions = {
                "turn_on", "turn_off", "toggle",
                "set_brightness", "set_temperature",
            }

            # Device types we handle
            device_types = {
                "media_player", "light", "switch",
                "climate", "cover", "fan", "scene",
            }

            if intent.action in device_actions:
                result.action_type = "device_command"
                result.needs_llm = False
                logger.debug(
                    "Device command: %s %s/%s (conf=%.2f)",
                    intent.action,
                    intent.target_type,
                    intent.target_name,
                    intent.confidence,
                )
            elif intent.action == "query" and intent.target_type in device_types:
                result.action_type = "device_command"
                result.needs_llm = False
                logger.debug(
                    "Device query: %s %s/%s (conf=%.2f)",
                    intent.action,
                    intent.target_type,
                    intent.target_name,
                    intent.confidence,
                )
            elif intent.action == "query" and intent.target_type == "tool":
                result.action_type = "tool_use"
                result.needs_llm = True
                logger.debug(
                    "Tool query: %s (conf=%.2f)",
                    intent.target_name,
                    intent.confidence,
                )
            else:
                # Not a device command - conversation fallback
                result.action_type = "conversation"
                result.needs_llm = True
        else:
            result.action_type = "conversation"
            result.needs_llm = True
            result.confidence = 0.5

        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result

    async def _do_act(
        self,
        context: AgentContext,
        think_result: ThinkResult,
    ) -> ActResult:
        """
        Execute device commands.
        """
        tools = self._get_tools()
        result = ActResult(
            success=False,
            action_type=think_result.action_type,
        )

        start_time = time.perf_counter()

        if think_result.action_type == "device_command" and think_result.intent:
            try:
                action_result = await tools.execute_intent(think_result.intent)
                result.success = action_result.get("success", False)
                result.action_results.append(action_result)
                result.response_data["action_message"] = action_result.get("message", "")

                logger.info(
                    "Action executed: %s -> %s",
                    think_result.intent.action,
                    "success" if result.success else "failed",
                )

                # Track entity for pronoun resolution
                if result.success and think_result.intent.target_name:
                    self._entity_tracker.track(
                        entity_type=think_result.intent.target_type or "device",
                        entity_name=think_result.intent.target_name,
                        entity_id=think_result.intent.target_id,
                    )

            except Exception as e:
                logger.warning("Action execution failed: %s", e)
                result.error = str(e)
                result.error_code = "EXECUTION_ERROR"

        elif think_result.action_type == "tool_use" and think_result.intent:
            # Execute tool directly for faster response
            target_name = think_result.intent.target_name
            params = think_result.intent.parameters or {}

            if target_name:
                try:
                    tool_result = await tools.execute_tool_by_intent(
                        target_name, params
                    )
                    result.success = tool_result.get("success", False)
                    result.tool_results[target_name] = tool_result
                    result.response_data["tool_message"] = tool_result.get("message", "")
                    logger.info(
                        "Tool executed: %s -> %s",
                        target_name,
                        "success" if result.success else "failed",
                    )
                except Exception as e:
                    logger.warning("Tool execution failed: %s", e)
                    result.error = str(e)
                    result.error_code = "TOOL_ERROR"
            else:
                # No target, defer to LLM tool calling
                result.success = True
                logger.info("Tool query with no target, deferring to LLM")

        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result

    async def _do_respond(
        self,
        context: AgentContext,
        think_result: ThinkResult,
        act_result: Optional[ActResult],
    ) -> str:
        """
        Generate response for device commands.
        """
        # Error case
        if act_result and act_result.error:
            return f"I'm sorry, there was an error: {act_result.error}"

        # Device command response
        if think_result.action_type == "device_command":
            return await self._generate_device_response(context, think_result, act_result)

        # Tool use - check if already executed in act phase
        if think_result.action_type == "tool_use":
            # Fast path: tool was executed in act phase, use its message
            if act_result and act_result.tool_results:
                tool_message = act_result.response_data.get("tool_message", "")
                if tool_message:
                    return tool_message
            # Slow path: fall back to LLM tool calling
            return await self._generate_llm_response_with_tools(context, think_result, act_result)

        # Conversation fallback - simple response
        return f"I heard: {context.input_text}"

    async def _generate_device_response(
        self,
        context: AgentContext,
        think_result: ThinkResult,
        act_result: Optional[ActResult],
    ) -> str:
        """Generate simple response for device command.

        Uses templates instead of LLM for faster, more predictable responses.
        """
        intent = think_result.intent
        success = act_result.success if act_result else False
        action_message = act_result.response_data.get("action_message", "") if act_result else ""

        if intent:
            target = intent.target_name or "device"
        else:
            target = "device"

        if success:
            if action_message:
                return action_message
            return f"Done."
        else:
            if action_message:
                return f"Sorry, {action_message}"
            return "Sorry, I couldn't do that."

    async def _generate_llm_response_with_tools(
        self,
        context: AgentContext,
        think_result: ThinkResult,
        act_result: Optional[ActResult],
    ) -> str:
        """Generate response using LLM tool calling loop."""
        from ..services.protocols import Message
        from ..services.tool_executor import execute_with_tools

        llm = self._get_llm()
        if llm is None:
            return f"I heard: {context.input_text}"

        # Build system prompt - keep it simple for reliable tool calling
        system_msg = "You are a helpful assistant. Use tools when needed."

        messages = [Message(role="system", content=system_msg)]

        # Add conversation history if available
        if context.conversation_history:
            for turn in context.conversation_history[-6:]:
                messages.append(Message(
                    role=turn.get("role", "user"),
                    content=turn.get("content", ""),
                ))

        # Add current user message
        messages.append(Message(role="user", content=context.input_text))

        # Get target tool from intent (if available)
        # Passing a single tool to the LLM is more reliable (100% vs ~33%)
        target_tool = None
        if think_result.intent and think_result.intent.target_name:
            target_tool = think_result.intent.target_name
            logger.info("Tool use with target: %s", target_tool)

        # Execute with tool calling loop
        cuda_lock = _get_cuda_lock()
        async with cuda_lock:
            result = await execute_with_tools(
                llm=llm,
                messages=messages,
                max_tokens=150,
                temperature=0.3,
                target_tool=target_tool,
            )

        response = result.get("response", "").strip()
        tools_executed = result.get("tools_executed", [])
        logger.info("Tool LLM result: tools=%s, response='%s'", tools_executed, response)

        if tools_executed:
            logger.info("Tools executed via LLM: %s", tools_executed)
            # Store tools in act_result so they flow to the API response
            if act_result is not None:
                for tool_name in tools_executed:
                    act_result.action_results.append({"tool": tool_name})

        return response if response else "I couldn't process that request."


# Factory function
_home_agent: Optional[HomeAgent] = None


def get_home_agent(session_id: Optional[str] = None) -> HomeAgent:
    """Get or create HomeAgent instance."""
    global _home_agent
    if _home_agent is None:
        _home_agent = HomeAgent(session_id=session_id)
    return _home_agent


def create_home_agent(session_id: Optional[str] = None) -> HomeAgent:
    """Create a new HomeAgent instance."""
    return HomeAgent(session_id=session_id)
