"""
Atlas Agent - Main agent implementation for Atlas Brain.

Handles reasoning, tool execution, and response generation.
The Orchestrator handles audio I/O and delegates to this Agent.
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any, Optional

from .base import BaseAgent, Timer
from .memory import AtlasAgentMemory, get_agent_memory
from .protocols import (
    ActResult,
    AgentContext,
    AgentResult,
    ThinkResult,
)
from .tools import AtlasAgentTools, get_agent_tools
from .entity_tracker import EntityTracker, has_pronoun, extract_pronoun

# Mode management imports
from ..modes.manager import get_mode_manager, ModeManager
from ..modes.config import ModeType

logger = logging.getLogger("atlas.agents.atlas")

# Global CUDA lock - shared with orchestrator to prevent conflicts
_cuda_lock: Optional[asyncio.Lock] = None

# Wake word strip pattern - removes "hey jarvis", "hey atlas", etc. from start of query
_WAKE_WORD_PATTERN = re.compile(
    r"^(?:hey\s+)?(?:jarvis|atlas|computer|assistant)[,.\s]*",
    re.IGNORECASE,
)


def _strip_wake_word(text: str) -> str:
    """Strip wake word prefix from text."""
    stripped = _WAKE_WORD_PATTERN.sub("", text).strip()
    if stripped != text.strip():
        logger.debug("Stripped wake word: '%s' -> '%s'", text[:30], stripped[:30])
    return stripped if stripped else text


def _get_cuda_lock() -> asyncio.Lock:
    """Get or create global CUDA lock."""
    global _cuda_lock
    if _cuda_lock is None:
        _cuda_lock = asyncio.Lock()
    return _cuda_lock


class AtlasAgent(BaseAgent):
    """
    Main Atlas agent implementation.

    Responsibilities:
    - Understanding user intent (think)
    - Executing device commands and tools (act)
    - Generating responses via LLM or templates (respond)
    - Managing conversation context and memory

    Usage:
        agent = AtlasAgent()
        context = AgentContext(
            input_text="turn on the living room lights",
            session_id="abc-123",
        )
        result = await agent.run(context)
        print(result.response_text)  # "Living room lights turned on"
    """

    def __init__(
        self,
        memory: Optional[AtlasAgentMemory] = None,
        tools: Optional[AtlasAgentTools] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize Atlas agent.

        Args:
            memory: Memory system (lazy-loaded if None)
            tools: Tools system (lazy-loaded if None)
            session_id: Default session ID for persistence
        """
        super().__init__(
            name="atlas",
            description="Atlas home automation assistant",
            memory=memory,
            tools=tools,
        )

        self._capabilities = [
            "device_control",
            "conversation",
            "weather",
            "traffic",
            "time",
            "location",
            "calendar",
            "reminder",
        ]

        self._session_id = session_id
        self._llm = None

        # Entity tracker for pronoun resolution
        self._entity_tracker = EntityTracker()

        # Mode management - AtlasAgent acts as router
        self._mode_manager: ModeManager = get_mode_manager()
        self._mode_agents: dict = {}  # Lazy-loaded to avoid circular imports
        self._workflow_state: Optional[str] = None  # Active workflow name or None

    def _init_mode_agents(self) -> None:
        """Initialize mode agents (lazy to avoid circular imports)."""
        if self._mode_agents:
            return  # Already initialized

        # Import here to avoid circular imports
        # Use singleton getters so agents are shared across AtlasAgent instances
        from .home import get_home_agent

        # Note: ReceptionistAgent is NOT included here - it's for phone calls only
        # (see atlas_brain/comms/phone_processor.py). Voice commands in RECEPTIONIST
        # mode are handled by AtlasAgent directly via the fallback path.
        self._mode_agents = {
            ModeType.HOME: get_home_agent(session_id=self._session_id),
        }
        logger.info("Mode agents initialized: %s", list(self._mode_agents.keys()))

    # Lazy loading

    def _get_memory(self) -> AtlasAgentMemory:
        """Get or create memory system."""
        if self._memory is None:
            self._memory = get_agent_memory()
        return self._memory

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

    # Router logic - AtlasAgent delegates to mode-specific agents

    async def run(self, context: AgentContext) -> AgentResult:
        """
        Main entry point with mode routing.

        Handles:
        1. Check mode timeout (before updating activity)
        2. Update activity timestamp
        3. Check for mode switch command
        4. Pre-classify query (conversations bypass mode agents)
        5. Delegate to current mode agent for device/tool queries
        """
        # Initialize mode agents on first run
        self._init_mode_agents()

        # Strip wake word from input (ASR may include "Hey Jarvis" etc.)
        context.input_text = _strip_wake_word(context.input_text)

        # 1. Check timeout BEFORE updating activity (skip if workflow active)
        if not self._workflow_state:
            self._mode_manager.check_timeout()

        # 2. Update activity timestamp (resets timeout for next check)
        self._mode_manager.update_activity()

        # 3. Check for mode switch command
        mode_switch = self._mode_manager.parse_mode_switch(context.input_text)
        if mode_switch:
            self._mode_manager.switch_mode(mode_switch)
            # Clear workflow state on manual switch
            self._workflow_state = None
            return AgentResult(
                success=True,
                response_text=f"Switched to {mode_switch.value} mode.",
                action_type="mode_switch",
            )

        # 4. Pre-classify query - conversations bypass mode agents
        from ..config import settings
        if settings.intent_router.enabled:
            tools = self._get_tools()
            route_result = await tools.route_intent(context.input_text)
            if route_result.action_category == "conversation":
                logger.info(
                    "Conversation detected (conf=%.2f), using AtlasAgent directly",
                    route_result.confidence,
                )
                return await super().run(context)

        # 5. Delegate to current mode agent for device/tool queries
        current_mode = self._mode_manager.current_mode
        agent = self._mode_agents.get(current_mode)

        if agent:
            logger.info("Routing to %s agent (mode=%s)", agent.info.name, current_mode.value)
            return await agent.run(context)

        # Fallback to original AtlasAgent behavior (for modes without agents)
        logger.info("No agent for mode %s, using AtlasAgent directly", current_mode.value)
        return await super().run(context)

    # Core agent methods

    async def _do_think(
        self,
        context: AgentContext,
    ) -> ThinkResult:
        """
        Analyze input and decide what to do.

        Uses unified intent parsing for devices, tools, and conversation.
        """
        tools = self._get_tools()

        result = ThinkResult(
            action_type="none",
            confidence=0.0,
        )

        start_time = time.perf_counter()

        # Step 1: Fast intent routing (if enabled)
        from ..config import settings
        route_result = None
        use_fast_path = False

        if settings.intent_router.enabled:
            route_result = await tools.route_intent(context.input_text)
            threshold = settings.intent_router.confidence_threshold

            # Fast path for high-confidence tool queries (parameterless only)
            if (route_result.action_category == "tool_use"
                    and route_result.confidence >= threshold
                    and route_result.fast_path_ok):
                result.action_type = "tool_use"
                result.confidence = route_result.confidence
                result.needs_llm = False  # Direct tool execution
                result.tools_to_call = [route_result.tool_name]
                use_fast_path = True
                logger.info(
                    "Fast route: tool_use/%s (conf=%.2f, %.0fms)",
                    route_result.tool_name,
                    route_result.confidence,
                    route_result.route_time_ms,
                )

            # Fast path for high-confidence conversation
            elif (route_result.action_category == "conversation"
                    and route_result.confidence >= threshold):
                result.action_type = "conversation"
                result.confidence = route_result.confidence
                result.needs_llm = True
                use_fast_path = True
                logger.info(
                    "Fast route: conversation (conf=%.2f, %.0fms)",
                    route_result.confidence,
                    route_result.route_time_ms,
                )

            # Parameterized tool - use LLM tool calling (not fast path)
            elif (route_result.action_category == "tool_use"
                    and route_result.confidence >= threshold
                    and route_result.tool_name
                    and not route_result.fast_path_ok):
                result.action_type = "tool_use"
                result.confidence = route_result.confidence
                result.needs_llm = True
                use_fast_path = True  # Skip intent parsing, go straight to LLM
                logger.info(
                    "LLM tool route: %s (conf=%.2f, %.0fms)",
                    route_result.tool_name,
                    route_result.confidence,
                    route_result.route_time_ms,
                )

        # Step 2: Full intent parsing (for device commands or low confidence)
        intent = None
        if not use_fast_path:
            intent = await tools.parse_intent(context.input_text)

            # Resolve pronouns if intent has no target but query contains pronoun
            if intent and not intent.target_name and has_pronoun(context.input_text):
                pronoun = extract_pronoun(context.input_text)
                if pronoun:
                    resolved = self._entity_tracker.resolve_pronoun(
                        pronoun,
                        entity_type=intent.target_type,
                    )
                    if resolved:
                        intent.target_name = resolved.entity_name
                        intent.target_type = intent.target_type or resolved.entity_type
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

                # Determine action type based on intent
                device_actions = {
                    "turn_on", "turn_off", "toggle",
                    "set_brightness", "set_temperature",
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
                elif intent.action == "query" and intent.target_type in [
                    "media_player", "light", "switch", "climate", "cover", "fan"
                ]:
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
                elif intent.action == "conversation":
                    result.action_type = "conversation"
                    result.needs_llm = True
                    logger.debug(
                        "Conversation detected (conf=%.2f)",
                        intent.confidence,
                    )
                else:
                    result.action_type = "conversation"
                    result.needs_llm = True
            else:
                # No intent parsed - default to conversation
                result.action_type = "conversation"
                result.needs_llm = True
                result.confidence = 0.5

        # Step 2: Retrieve memory context if this is a conversation
        if result.needs_llm:
            try:
                from ..config import settings

                if settings.memory.enabled and settings.memory.retrieve_context:
                    memory_start = time.perf_counter()
                    memory_client = self._get_memory_client()

                    if memory_client:
                        # 2 second timeout
                        memory_context = await asyncio.wait_for(
                            memory_client.get_context_for_query(
                                context.input_text,
                                num_results=settings.memory.context_results,
                            ),
                            timeout=2.0,
                        )
                        result.retrieved_context = memory_context
                        memory_ms = (time.perf_counter() - memory_start) * 1000
                        logger.debug(
                            "Retrieved memory context: %d chars in %.0fms",
                            len(memory_context) if memory_context else 0,
                            memory_ms,
                        )
            except asyncio.TimeoutError:
                logger.warning("Memory retrieval timed out (>2s)")
            except Exception as e:
                logger.warning("Memory retrieval failed: %s", e)

        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _get_memory_client(self):
        """Get memory client for RAG context retrieval."""
        try:
            from ..services.memory import get_memory_client
            return get_memory_client()
        except Exception:
            return None

    def _filter_tool_data(
        self,
        tool_name: str,
        data: dict,
        query: str,
    ) -> dict:
        """
        Filter tool data to only include fields relevant to the query.

        This ensures the LLM gets focused context and produces specific answers.
        """
        # Normalize tool name (intent uses "time", registry uses "get_time")
        if tool_name in ("time", "get_time"):
            # Exclude internal fields
            exclude = {"iso", "time_24h", "timezone"}

            # Check what specifically was asked
            wants_time = "time" in query and "date" not in query
            wants_day = "day" in query and "time" not in query
            wants_date = "date" in query and "time" not in query

            if wants_time:
                return {"time": data.get("time")}
            elif wants_day:
                return {"day_of_week": data.get("day_of_week")}
            elif wants_date:
                return {"date": data.get("date")}
            else:
                # General query - return time, day, date
                return {k: v for k, v in data.items() if k not in exclude and v}

        # Calendar: return event summaries or empty indicator
        if tool_name in ("calendar", "get_calendar"):
            events = data.get("events", [])
            if events:
                # Return formatted event list
                event_strs = []
                for e in events[:5]:  # Limit to 5
                    summary = e.get("summary", "Untitled")
                    start = e.get("start", "")
                    event_strs.append(f"{summary} at {start}")
                return {"upcoming_events": "; ".join(event_strs)}
            else:
                # No events - return None to use message fallback
                return {}

        # Reminder: use message directly (data has IDs, not useful for LLM)
        if tool_name in ("reminder", "set_reminder", "list_reminders", "complete_reminder"):
            # Return empty to fall back to the well-formatted message
            return {}

        # Default: return all data except internal fields
        exclude = {"iso", "time_24h", "raw", "debug", "cached"}
        return {k: v for k, v in data.items() if k not in exclude and v}

    async def _do_act(
        self,
        context: AgentContext,
        think_result: ThinkResult,
    ) -> ActResult:
        """
        Execute actions based on think result.

        For device commands: Execute via ActionDispatcher
        For tool use: Execute relevant tools
        """
        tools = self._get_tools()
        result = ActResult(
            success=False,
            action_type=think_result.action_type,
        )

        start_time = time.perf_counter()

        if think_result.action_type == "device_command" and think_result.intent:
            # Execute device command
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

                # Track entity for pronoun resolution on success
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

        elif think_result.action_type == "tool_use" and think_result.tools_to_call:
            # Fast path: execute tool from router result
            tool_name = think_result.tools_to_call[0]
            try:
                tool_result = await tools.execute_tool(tool_name, {})
                result.success = tool_result.get("success", False)
                result.tool_results[tool_name] = tool_result
                result.response_data["tool_message"] = tool_result.get("message", "")
                logger.info(
                    "Fast tool executed: %s -> %s",
                    tool_name,
                    "success" if result.success else "failed",
                )
            except Exception as e:
                logger.warning("Fast tool execution failed: %s", e)
                result.error = str(e)
                result.error_code = "TOOL_ERROR"

        elif think_result.action_type == "tool_use" and think_result.intent:
            # Slow path: execute tool from LLM intent
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
        Generate response text.

        For device commands: Use action result message (fast)
        For conversations: Use LLM with context (slower)
        """
        # Error case
        if act_result and act_result.error:
            return f"I'm sorry, there was an error: {act_result.error}"

        # Device command response - use LLM for natural response
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

        # Conversation response - use LLM
        return await self._generate_llm_response(context, think_result, act_result)

    async def _generate_llm_response(
        self,
        context: AgentContext,
        think_result: ThinkResult,
        act_result: Optional[ActResult],
    ) -> str:
        """Generate a conversational response using the LLM."""
        from ..config import settings
        from ..services.protocols import Message

        llm = self._get_llm()
        if llm is None:
            return f"I heard: {context.input_text}"

        # Build system prompt
        system_parts = [
            "You are Atlas, a capable personal assistant.",
            "You can control smart home devices, answer questions, have conversations, and help with various tasks.",
            "Be conversational, helpful, and concise. Keep responses to 1-3 sentences unless more detail is needed.",
        ]

        # Add tool context if available
        tool_context = act_result.response_data.get("tool_context") if act_result else None
        if tool_context:
            system_parts.append(
                "Answer ONLY what was specifically asked using the data below. "
                "Do not include extra information unless requested."
            )
            system_parts.append(f"\nDATA:\n{tool_context}")

        # Add speaker info
        if context.speaker_id and context.speaker_id != "unknown":
            system_parts.append(f"The speaker is {context.speaker_id}.")

        # Add memory context
        if think_result.retrieved_context:
            system_parts.append(f"\n{think_result.retrieved_context}")

        system_msg = " ".join(system_parts)
        messages = [Message(role="system", content=system_msg)]

        # Add conversation history
        if context.conversation_history:
            for turn in context.conversation_history[-6:]:  # Last 3 exchanges
                messages.append(Message(
                    role=turn.get("role", "user"),
                    content=turn.get("content", ""),
                ))

        # Add current message
        messages.append(Message(role="user", content=context.input_text))

        # Call LLM with CUDA lock
        try:
            cuda_lock = _get_cuda_lock()
            async with cuda_lock:
                llm_result = llm.chat(
                    messages=messages,
                    max_tokens=100,
                    temperature=0.7,
                )
            response = llm_result.get("response", "").strip()
            if response:
                return response
        except Exception as e:
            logger.warning("LLM response generation failed: %s", e)

        return f"I heard: {context.input_text}"

    async def _generate_device_response(
        self,
        context: AgentContext,
        think_result: ThinkResult,
        act_result: Optional[ActResult],
    ) -> str:
        """Generate natural response for device commands using LLM."""
        from ..services.protocols import Message

        # Get action details
        intent = think_result.intent
        action_message = ""
        if act_result and act_result.response_data:
            action_message = act_result.response_data.get("action_message", "")

        success = act_result.success if act_result else False

        # Build context for LLM
        if intent:
            action = intent.action or "unknown"
            target = intent.target_name or "device"
            target_type = intent.target_type or "device"
        else:
            action = "command"
            target = "device"
            target_type = "device"

        llm = self._get_llm()
        logger.info("Device response using LLM: %s", llm.model if hasattr(llm, 'model') else 'unknown')
        if llm is None:
            # Fallback to simple response
            if success:
                return f"Done. {action_message}" if action_message else "Done."
            else:
                return f"I couldn't complete that action. {action_message}"

        # Build prompt for natural response
        system_msg = (
            "You are Atlas, a home assistant. Generate a brief, natural confirmation "
            "for the device action that was just performed. Keep it to one short sentence. "
            "Be conversational but concise."
        )

        if success:
            user_prompt = (
                f"I just successfully executed '{action}' on the {target_type} '{target}'. "
                f"Result: {action_message if action_message else 'Success'}. "
                f"Generate a brief, natural confirmation."
            )
        else:
            user_prompt = (
                f"I tried to execute '{action}' on the {target_type} '{target}' but it failed. "
                f"Error: {action_message if action_message else 'Unknown error'}. "
                f"Generate a brief, apologetic response."
            )

        messages = [
            Message(role="system", content=system_msg),
            Message(role="user", content=user_prompt),
        ]

        try:
            cuda_lock = _get_cuda_lock()
            async with cuda_lock:
                llm_result = llm.chat(
                    messages=messages,
                    max_tokens=50,
                    temperature=0.7,
                )
            response = llm_result.get("response", "").strip()
            if response:
                return response
        except Exception as e:
            logger.warning("Device response generation failed: %s", e)

        # Fallback
        if success:
            return f"Done. {action_message}" if action_message else "Done."
        else:
            return f"I couldn't complete that. {action_message}"

    async def _generate_llm_response_with_tools(
        self,
        context: AgentContext,
        think_result: ThinkResult,
        act_result: Optional[ActResult],
    ) -> str:
        """Generate response using Gorilla for tools, Cloud LLM for chat."""
        from ..config import settings

        # Try Gorilla router first (local tool calling)
        if settings.tool_router.enabled:
            try:
                from ..services.tool_router import process_query

                gorilla_result = await process_query(context.input_text)

                if not gorilla_result.needs_cloud_llm:
                    # Gorilla handled it - tool was executed
                    logger.info(
                        "Gorilla executed tools: %s (%.0fms)",
                        gorilla_result.tools_executed,
                        gorilla_result.latency_ms,
                    )
                    # Store tools in act_result for API response
                    if act_result is not None:
                        for tool_name in gorilla_result.tools_executed:
                            act_result.action_results.append({"tool": tool_name})
                    return gorilla_result.response

                logger.info("Gorilla: no tool detected, routing to cloud LLM")

            except Exception as e:
                logger.warning("Gorilla router failed, falling back to cloud: %s", e)

        # Fall back to Cloud LLM for chat (no tool calling)
        return await self._generate_llm_response(context, think_result, act_result)

    async def store_turn(
        self,
        context: AgentContext,
        result: AgentResult,
    ) -> None:
        """
        Persist the conversation turn.

        Override to use session_id from context or instance.
        """
        session_id = context.session_id or self._session_id
        if not session_id:
            logger.debug("Skipping turn storage (no session)")
            return

        memory = self._get_memory()

        try:
            # Determine turn type
            turn_type = "command" if result.action_type == "device_command" else "conversation"
            intent_str = result.intent.action if result.intent and hasattr(result.intent, "action") else None

            # Store user turn
            await memory.add_turn(
                session_id=session_id,
                role="user",
                content=context.input_text,
                speaker_id=context.speaker_id,
                intent=intent_str,
                turn_type=turn_type,
            )

            # Store assistant turn
            if result.response_text:
                await memory.add_turn(
                    session_id=session_id,
                    role="assistant",
                    content=result.response_text,
                    turn_type=turn_type,
                )

            logger.debug("Stored conversation turns for session %s", session_id)

        except Exception as e:
            logger.warning("Failed to store conversation turn: %s", e)

    async def load_conversation_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 10,
        include_commands: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Load conversation history from storage.

        Args:
            session_id: Session to load from (uses instance session if None)
            limit: Maximum turns to load
            include_commands: Include device commands in history

        Returns:
            List of conversation turn dictionaries
        """
        session_id = session_id or self._session_id
        if not session_id:
            return []

        memory = self._get_memory()
        turn_type = None if include_commands else "conversation"

        return await memory.get_conversation_history(
            session_id=session_id,
            limit=limit,
            turn_type=turn_type,
        )


# Global agent instance
_atlas_agent: Optional[AtlasAgent] = None


def get_atlas_agent(session_id: Optional[str] = None) -> AtlasAgent:
    """
    Get or create the global Atlas agent instance.

    Args:
        session_id: Session ID for persistence

    Returns:
        AtlasAgent instance
    """
    global _atlas_agent
    if _atlas_agent is None:
        _atlas_agent = AtlasAgent(session_id=session_id)
    elif session_id:
        _atlas_agent._session_id = session_id
    return _atlas_agent


def reset_atlas_agent() -> None:
    """Reset the global Atlas agent instance."""
    global _atlas_agent
    _atlas_agent = None
