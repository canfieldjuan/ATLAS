"""
AtlasAgent LangGraph implementation.

Main router agent that delegates to sub-agents or handles directly.
Supports mode-based routing and conversation handling.
"""

import asyncio
import logging
import re
import time
from typing import Any, Literal, Optional

from langgraph.graph import END, StateGraph

from .state import ActionResult, AtlasAgentState, Intent
from .home import HomeAgentGraph
from ..entity_tracker import EntityTracker, extract_pronoun, has_pronoun
from ..tools import AtlasAgentTools, get_agent_tools
from ..memory import AtlasAgentMemory, get_agent_memory
from .workflow_state import get_workflow_state_manager
from .booking import run_booking_workflow, BOOKING_WORKFLOW_TYPE
from .reminder import run_reminder_workflow, REMINDER_WORKFLOW_TYPE
from .email import run_email_workflow, EMAIL_WORKFLOW_TYPE
from .calendar import run_calendar_workflow, CALENDAR_WORKFLOW_TYPE

logger = logging.getLogger("atlas.agents.graphs.atlas")

# Global CUDA lock
_cuda_lock: Optional[asyncio.Lock] = None

# Wake word strip pattern
_WAKE_WORD_PATTERN = re.compile(
    r"^(?:hey\s+)?(?:jarvis|atlas|computer|assistant)[,.\s]*",
    re.IGNORECASE,
)


def _strip_wake_word(text: str) -> str:
    """Strip wake word prefix from text."""
    stripped = _WAKE_WORD_PATTERN.sub("", text).strip()
    return stripped if stripped else text


# Cancel patterns for active workflow interruption
_CANCEL_PATTERNS = [
    re.compile(r"^(?:never\s?mind|cancel|stop|forget\s+it|quit)$", re.IGNORECASE),
    re.compile(r"^(?:I\s+)?(?:don'?t\s+)?(?:want\s+to\s+)?cancel", re.IGNORECASE),
    re.compile(r"^stop\s+(?:that|this|booking|scheduling)", re.IGNORECASE),
]


def _is_cancel_intent(text: str) -> bool:
    """Check if text matches a cancel pattern."""
    text = text.strip()
    for pattern in _CANCEL_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _get_cuda_lock() -> asyncio.Lock:
    """Get or create global CUDA lock."""
    global _cuda_lock
    if _cuda_lock is None:
        _cuda_lock = asyncio.Lock()
    return _cuda_lock


# Node functions


async def preprocess_input(state: AtlasAgentState) -> AtlasAgentState:
    """Preprocess input: strip wake word, check mode switches."""
    input_text = state["input_text"]

    # Strip wake word
    cleaned_text = _strip_wake_word(input_text)

    # Check for mode switch commands
    from ...modes.manager import get_mode_manager

    mode_manager = get_mode_manager()

    # Check timeout before updating activity
    mode_manager.check_timeout()
    mode_manager.update_activity()

    # Check for mode switch
    mode_switch = mode_manager.parse_mode_switch(cleaned_text)
    if mode_switch:
        mode_manager.switch_mode(mode_switch)
        return {
            **state,
            "input_text": cleaned_text,
            "action_type": "mode_switch",
            "response": f"Switched to {mode_switch.value} mode.",
            "current_mode": mode_switch.value,
        }

    return {
        **state,
        "input_text": cleaned_text,
        "current_mode": mode_manager.current_mode.value,
    }


async def check_active_workflow(state: AtlasAgentState) -> AtlasAgentState:
    """Check if session has an active workflow to continue."""
    # Skip if mode switch already handled
    if state.get("action_type") == "mode_switch":
        return state

    session_id = state.get("session_id")
    if not session_id:
        return state

    manager = get_workflow_state_manager()
    workflow = await manager.restore_workflow_state(session_id)

    if workflow is None:
        return state

    # Check if workflow is expired
    if workflow.is_expired():
        logger.info("Workflow expired for session %s, clearing", session_id)
        await manager.clear_workflow_state(session_id)
        return state

    # Check for cancel intent
    input_text = state.get("input_text", "")
    if _is_cancel_intent(input_text):
        await manager.clear_workflow_state(session_id)
        logger.info("User cancelled active workflow for session %s", session_id)
        return {
            **state,
            "action_type": "workflow_cancelled",
            "response": "Okay, I've cancelled that.",
        }

    # Active workflow found - mark for continuation
    logger.info(
        "Continuing %s workflow at step %s for session %s",
        workflow.workflow_type,
        workflow.current_step,
        session_id,
    )
    return {
        **state,
        "active_workflow": {
            "workflow_type": workflow.workflow_type,
            "current_step": workflow.current_step,
            "partial_state": workflow.partial_state,
        },
        "action_type": "workflow_continuation",
    }


async def classify_intent(state: AtlasAgentState) -> AtlasAgentState:
    """
    Classify user input to determine action type.

    Uses fast intent routing (DistilBERT) if available.
    """
    # Skip if mode switch already handled
    if state.get("action_type") == "mode_switch":
        return state

    start_time = time.perf_counter()
    tools = get_agent_tools()
    input_text = state["input_text"]

    from ...config import settings

    action_type = "conversation"
    confidence = 0.5
    tools_to_call: list[str] = []
    delegate_to: Optional[str] = None

    if settings.intent_router.enabled:
        route_result = await tools.route_intent(input_text)
        threshold = settings.intent_router.confidence_threshold

        # Fast path for high-confidence tool queries (parameterless only)
        if (
            route_result.action_category == "tool_use"
            and route_result.confidence >= threshold
            and route_result.fast_path_ok
        ):
            action_type = "tool_use"
            confidence = route_result.confidence
            tools_to_call = [route_result.tool_name] if route_result.tool_name else []
            logger.info(
                "Fast route: tool_use/%s (conf=%.2f)",
                route_result.tool_name,
                route_result.confidence,
            )

        # High-confidence conversation - handle directly
        elif (
            route_result.action_category == "conversation"
            and route_result.confidence >= threshold
        ):
            action_type = "conversation"
            confidence = route_result.confidence
            logger.info("Fast route: conversation (conf=%.2f)", route_result.confidence)

        # Device command - delegate to HomeAgent
        elif (
            route_result.action_category == "device_command"
            and route_result.confidence >= threshold
        ):
            action_type = "device_command"
            confidence = route_result.confidence
            delegate_to = "home"
            logger.info("Fast route: device_command -> HomeAgent")

        # Parameterized tool - use LLM
        elif (
            route_result.action_category == "tool_use"
            and route_result.confidence >= threshold
            and route_result.tool_name
        ):
            action_type = "tool_use"
            confidence = route_result.confidence
            logger.info("LLM tool route: %s", route_result.tool_name)

    classify_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        "action_type": action_type,
        "confidence": confidence,
        "tools_to_call": tools_to_call,
        "delegate_to": delegate_to,
        "classify_ms": classify_ms,
    }


async def retrieve_memory(state: AtlasAgentState) -> AtlasAgentState:
    """Retrieve memory context for conversation queries."""
    # Only retrieve for conversation or tool use needing LLM
    action_type = state.get("action_type", "conversation")
    if action_type not in ("conversation", "tool_use"):
        return state

    from ...config import settings

    if not settings.memory.enabled or not settings.memory.retrieve_context:
        return state

    start_time = time.perf_counter()
    input_text = state["input_text"]

    try:
        from ...services.memory import get_memory_client

        memory_client = get_memory_client()
        if memory_client:
            memory_context = await asyncio.wait_for(
                memory_client.get_context_for_query(
                    input_text,
                    num_results=settings.memory.context_results,
                ),
                timeout=2.0,
            )

            memory_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                "Retrieved memory context: %d chars in %.0fms",
                len(memory_context) if memory_context else 0,
                memory_ms,
            )

            return {
                **state,
                "retrieved_context": memory_context,
                "memory_ms": memory_ms,
            }

    except asyncio.TimeoutError:
        logger.warning("Memory retrieval timed out")
    except Exception as e:
        logger.warning("Memory retrieval failed: %s", e)

    return state


async def parse_intent(state: AtlasAgentState) -> AtlasAgentState:
    """Parse detailed intent from user input."""
    # Skip if delegating or no parsing needed
    if state.get("delegate_to") or state.get("action_type") == "mode_switch":
        return state

    start_time = time.perf_counter()
    tools = get_agent_tools()
    input_text = state["input_text"]
    action_type = state.get("action_type", "conversation")

    intent: Optional[Intent] = None

    # Parse if we need device command details or low confidence
    if action_type in ("device_command", "tool_use") or state.get("confidence", 0) < 0.7:
        parsed = await tools.parse_intent(input_text)

        if parsed:
            intent = Intent(
                action=parsed.action,
                target_type=parsed.target_type,
                target_name=parsed.target_name,
                target_id=getattr(parsed, "target_id", None),
                parameters=parsed.parameters or {},
                confidence=parsed.confidence,
                raw_query=input_text,
            )

            # Refine action_type based on parsed intent
            device_actions = {
                "turn_on",
                "turn_off",
                "toggle",
                "set_brightness",
                "set_temperature",
            }
            device_types = {
                "media_player",
                "light",
                "switch",
                "climate",
                "cover",
                "fan",
                "scene",
            }

            if intent.action in device_actions:
                action_type = "device_command"
            elif intent.action == "query" and intent.target_type in device_types:
                action_type = "device_command"
            elif intent.action == "query" and intent.target_type == "tool":
                action_type = "tool_use"
            elif intent.action == "conversation":
                action_type = "conversation"

    think_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        "intent": intent,
        "action_type": action_type,
        "think_ms": think_ms,
    }


async def execute_action(state: AtlasAgentState) -> AtlasAgentState:
    """Execute tool or device command."""
    # Skip if delegating, conversation, or mode switch
    action_type = state.get("action_type", "none")
    if state.get("delegate_to") or action_type in ("conversation", "mode_switch", "none"):
        return state

    start_time = time.perf_counter()
    tools = get_agent_tools()
    intent = state.get("intent")

    result = ActionResult(success=False, message="No action taken")
    tool_results: dict[str, Any] = {}
    tools_executed: list[str] = []

    if action_type == "device_command" and intent:
        try:
            action_result = await tools.execute_intent(intent)
            result = ActionResult(
                success=action_result.get("success", False),
                message=action_result.get("message", ""),
                data=action_result,
            )
            logger.info(
                "Action executed: %s -> %s",
                intent.action,
                "success" if result.success else "failed",
            )
        except Exception as e:
            logger.warning("Action execution failed: %s", e)
            result = ActionResult(success=False, message=str(e), error=str(e))

    elif action_type == "tool_use":
        # Fast path: tool from router
        tools_to_call = state.get("tools_to_call", [])
        if tools_to_call:
            tool_name = tools_to_call[0]
            try:
                tool_result = await tools.execute_tool(tool_name, {})
                result = ActionResult(
                    success=tool_result.get("success", False),
                    message=tool_result.get("message", ""),
                    data=tool_result,
                )
                tool_results[tool_name] = tool_result
                tools_executed.append(tool_name)
                logger.info("Fast tool executed: %s", tool_name)
            except Exception as e:
                logger.warning("Tool execution failed: %s", e)
                result = ActionResult(success=False, message=str(e), error=str(e))

        # Slow path: tool from LLM intent
        elif intent and intent.target_name:
            target_name = intent.target_name
            params = intent.parameters or {}
            try:
                tool_result = await tools.execute_tool_by_intent(target_name, params)
                result = ActionResult(
                    success=tool_result.get("success", False),
                    message=tool_result.get("message", ""),
                    data=tool_result,
                )
                tool_results[target_name] = tool_result
                tools_executed.append(target_name)
                logger.info("Tool executed: %s", target_name)
            except Exception as e:
                logger.warning("Tool execution failed: %s", e)
                result = ActionResult(success=False, message=str(e), error=str(e))

    act_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        "action_result": result,
        "tool_results": tool_results,
        "tools_executed": tools_executed,
        "act_ms": act_ms,
    }


async def generate_response(state: AtlasAgentState) -> AtlasAgentState:
    """Generate response for the user."""
    # Skip if already responded (mode switch) or delegating
    if state.get("response") or state.get("delegate_to"):
        return state

    start_time = time.perf_counter()
    action_type = state.get("action_type", "conversation")
    action_result = state.get("action_result")
    input_text = state.get("input_text", "")

    response = ""

    # Error case
    if action_result and action_result.error:
        response = f"I'm sorry, there was an error: {action_result.error}"

    # Device command - template response
    elif action_type == "device_command":
        if action_result and action_result.success:
            response = action_result.message or "Done."
        else:
            response = f"Sorry, {action_result.message if action_result else 'I could not complete that.'}"

    # Tool use
    elif action_type == "tool_use":
        tool_results = state.get("tool_results", {})
        if tool_results:
            for tool_name, result in tool_results.items():
                if result.get("message"):
                    response = result["message"]
                    break

        if not response:
            # Fall back to LLM with tool context
            response = await _generate_llm_response(state, with_tools=True)

    # Conversation
    elif action_type == "conversation":
        response = await _generate_llm_response(state)

    else:
        response = f"I heard: {input_text}"

    respond_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        "response": response,
        "respond_ms": respond_ms,
    }


async def delegate_to_home(state: AtlasAgentState) -> AtlasAgentState:
    """Delegate to HomeAgent sub-graph."""
    start_time = time.perf_counter()

    home_agent = HomeAgentGraph(session_id=state.get("session_id"))
    result = await home_agent.run(
        input_text=state["input_text"],
        session_id=state.get("session_id"),
        speaker_id=state.get("speaker_id"),
        runtime_context=state.get("runtime_context", {}),
    )

    total_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        "response": result.get("response_text", ""),
        "action_type": result.get("action_type", "device_command"),
        "intent": result.get("intent"),
        "error": result.get("error"),
        "act_ms": result.get("timing", {}).get("act", 0),
        "respond_ms": result.get("timing", {}).get("respond", 0),
    }


async def continue_workflow(state: AtlasAgentState) -> AtlasAgentState:
    """Continue an active workflow with new user input."""
    start_time = time.perf_counter()
    active_workflow = state.get("active_workflow", {})
    workflow_type = active_workflow.get("workflow_type")
    session_id = state.get("session_id")
    input_text = state.get("input_text", "")

    if workflow_type == BOOKING_WORKFLOW_TYPE:
        result = await run_booking_workflow(
            input_text=input_text,
            session_id=session_id,
        )
        response = result.get("response", "")
        total_ms = (time.perf_counter() - start_time) * 1000

        return {
            **state,
            "response": response,
            "action_type": "tool_use",
            "act_ms": total_ms,
        }

    if workflow_type == REMINDER_WORKFLOW_TYPE:
        result = await run_reminder_workflow(
            input_text=input_text,
            session_id=session_id,
        )
        response = result.get("response", "")
        total_ms = (time.perf_counter() - start_time) * 1000

        return {
            **state,
            "response": response,
            "action_type": "tool_use",
            "act_ms": total_ms,
        }

    if workflow_type == EMAIL_WORKFLOW_TYPE:
        result = await run_email_workflow(
            input_text=input_text,
            session_id=session_id,
        )
        response = result.get("response", "")
        total_ms = (time.perf_counter() - start_time) * 1000

        return {
            **state,
            "response": response,
            "action_type": "tool_use",
            "act_ms": total_ms,
        }

    if workflow_type == CALENDAR_WORKFLOW_TYPE:
        result = await run_calendar_workflow(
            input_text=input_text,
            session_id=session_id,
        )
        response = result.get("response", "")
        total_ms = (time.perf_counter() - start_time) * 1000

        return {
            **state,
            "response": response,
            "action_type": "tool_use",
            "act_ms": total_ms,
        }

    # Unknown workflow type - should not happen
    logger.warning("Unknown workflow type: %s", workflow_type)
    return {
        **state,
        "response": "I'm not sure how to continue. Could you start over?",
        "error": "unknown_workflow_type",
    }


async def _generate_llm_response(
    state: AtlasAgentState,
    with_tools: bool = False,
) -> str:
    """Generate response using LLM."""
    from ...services import llm_registry
    from ...services.protocols import Message

    llm = llm_registry.get_active()
    if llm is None:
        return f"I heard: {state.get('input_text', '')}"

    input_text = state.get("input_text", "")
    retrieved_context = state.get("retrieved_context")
    speaker_id = state.get("speaker_id")

    # Build system prompt
    system_parts = [
        "You are Atlas, a capable personal assistant.",
        "You can control smart home devices, answer questions, and help with various tasks.",
        "Be conversational, helpful, and concise. Keep responses to 1-3 sentences.",
    ]

    # Add tool context if available
    if with_tools:
        tool_results = state.get("tool_results", {})
        if tool_results:
            for tool_name, result in tool_results.items():
                if result.get("data"):
                    system_parts.append(f"\nTool data: {result['data']}")

    # Add speaker info
    if speaker_id and speaker_id != "unknown":
        system_parts.append(f"The speaker is {speaker_id}.")

    # Add memory context
    if retrieved_context:
        system_parts.append(f"\n{retrieved_context}")

    system_msg = " ".join(system_parts)
    messages = [
        Message(role="system", content=system_msg),
        Message(role="user", content=input_text),
    ]

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

    return f"I heard: {input_text}"


# Routing functions


def route_after_classify(
    state: AtlasAgentState,
) -> Literal["delegate_home", "retrieve_memory", "execute", "respond"]:
    """Route based on classification result."""
    action_type = state.get("action_type", "conversation")

    # Mode switch already handled
    if action_type == "mode_switch":
        return "respond"

    # Delegate device commands to HomeAgent
    if state.get("delegate_to") == "home":
        return "delegate_home"

    # Conversation needs memory retrieval
    if action_type == "conversation":
        return "retrieve_memory"

    # Tool use with fast path
    if action_type == "tool_use" and state.get("tools_to_call"):
        return "execute"

    # Otherwise parse intent first
    return "retrieve_memory"


def route_after_memory(
    state: AtlasAgentState,
) -> Literal["parse", "respond"]:
    """Route after memory retrieval."""
    action_type = state.get("action_type", "conversation")

    # Pure conversation goes straight to respond
    if action_type == "conversation" and state.get("confidence", 0) >= 0.7:
        return "respond"

    # Otherwise parse intent
    return "parse"


def route_after_parse(
    state: AtlasAgentState,
) -> Literal["execute", "respond"]:
    """Route after parsing intent."""
    action_type = state.get("action_type", "conversation")

    if action_type in ("device_command", "tool_use"):
        return "execute"

    return "respond"


def route_after_check_workflow(
    state: AtlasAgentState,
) -> Literal["continue_workflow", "classify", "respond"]:
    """Route after checking for active workflow."""
    action_type = state.get("action_type", "")

    # Mode switch or workflow cancelled - go to respond
    if action_type in ("mode_switch", "workflow_cancelled"):
        return "respond"

    # Active workflow found - continue it
    if action_type == "workflow_continuation":
        return "continue_workflow"

    # No active workflow - proceed with normal classification
    return "classify"


# Build the graph


def build_atlas_agent_graph() -> StateGraph:
    """
    Build the AtlasAgent LangGraph.

    Flow:
    preprocess -> check_workflow -> (continue | classify -> ...)
    """
    graph = StateGraph(AtlasAgentState)

    # Add nodes
    graph.add_node("preprocess", preprocess_input)
    graph.add_node("check_workflow", check_active_workflow)
    graph.add_node("continue_workflow", continue_workflow)
    graph.add_node("classify", classify_intent)
    graph.add_node("retrieve_memory", retrieve_memory)
    graph.add_node("parse", parse_intent)
    graph.add_node("execute", execute_action)
    graph.add_node("respond", generate_response)
    graph.add_node("delegate_home", delegate_to_home)

    # Set entry point
    graph.set_entry_point("preprocess")

    # Add edges
    graph.add_edge("preprocess", "check_workflow")

    graph.add_conditional_edges(
        "check_workflow",
        route_after_check_workflow,
        {
            "continue_workflow": "continue_workflow",
            "classify": "classify",
            "respond": "respond",
        },
    )

    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "delegate_home": "delegate_home",
            "retrieve_memory": "retrieve_memory",
            "execute": "execute",
            "respond": "respond",
        },
    )

    graph.add_conditional_edges(
        "retrieve_memory",
        route_after_memory,
        {
            "parse": "parse",
            "respond": "respond",
        },
    )

    graph.add_conditional_edges(
        "parse",
        route_after_parse,
        {
            "execute": "execute",
            "respond": "respond",
        },
    )

    graph.add_edge("execute", "respond")
    graph.add_edge("delegate_home", END)
    graph.add_edge("continue_workflow", END)
    graph.add_edge("respond", END)

    return graph


# Compiled graph singleton
_atlas_graph: Optional[StateGraph] = None


def get_atlas_agent_graph() -> StateGraph:
    """Get or create the compiled AtlasAgent graph."""
    global _atlas_graph
    if _atlas_graph is None:
        _atlas_graph = build_atlas_agent_graph()
    return _atlas_graph


class AtlasAgentGraph:
    """
    AtlasAgent using LangGraph for orchestration.

    Main router agent that handles conversations and delegates
    device commands to specialized sub-agents.
    """

    def __init__(self, session_id: Optional[str] = None):
        """Initialize AtlasAgent graph."""
        self._session_id = session_id
        self._graph = get_atlas_agent_graph()
        self._compiled = self._graph.compile()
        self._entity_tracker = EntityTracker()
        self._memory = get_agent_memory()

    async def run(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Process input and return result.

        Args:
            input_text: User input text
            session_id: Session ID for persistence
            speaker_id: Identified speaker name
            **kwargs: Additional context

        Returns:
            Result dict with response, action_type, timing, etc.
        """
        start_time = time.perf_counter()

        # Build initial state
        initial_state: AtlasAgentState = {
            "input_text": input_text,
            "input_type": kwargs.get("input_type", "text"),
            "session_id": session_id or self._session_id,
            "speaker_id": speaker_id,
            "runtime_context": kwargs.get("runtime_context", {}),
            "messages": [],
            "action_type": "none",
            "confidence": 0.0,
            "current_mode": "home",
            "tools_to_call": [],
            "tools_executed": [],
            "tool_results": {},
        }

        # Run the graph
        try:
            final_state = await self._compiled.ainvoke(initial_state)
        except Exception as e:
            logger.exception("Error running AtlasAgent graph: %s", e)
            final_state = {
                **initial_state,
                "response": f"I'm sorry, I encountered an error: {e}",
                "error": str(e),
            }

        total_ms = (time.perf_counter() - start_time) * 1000

        # Store conversation turn
        await self._store_turn(final_state)

        # Build result
        return {
            "success": final_state.get("error") is None,
            "response_text": final_state.get("response", ""),
            "action_type": final_state.get("action_type", "none"),
            "intent": final_state.get("intent"),
            "error": final_state.get("error"),
            "timing": {
                "total": total_ms,
                "classify": final_state.get("classify_ms", 0),
                "memory": final_state.get("memory_ms", 0),
                "think": final_state.get("think_ms", 0),
                "act": final_state.get("act_ms", 0),
                "respond": final_state.get("respond_ms", 0),
            },
        }

    async def _store_turn(self, state: AtlasAgentState) -> None:
        """Store conversation turn to memory."""
        session_id = state.get("session_id")
        if not session_id:
            return

        try:
            turn_type = (
                "command"
                if state.get("action_type") == "device_command"
                else "conversation"
            )
            intent = state.get("intent")
            intent_str = intent.action if intent else None

            # Store user turn
            await self._memory.add_turn(
                session_id=session_id,
                role="user",
                content=state.get("input_text", ""),
                speaker_id=state.get("speaker_id"),
                intent=intent_str,
                turn_type=turn_type,
            )

            # Store assistant turn
            response = state.get("response")
            if response:
                await self._memory.add_turn(
                    session_id=session_id,
                    role="assistant",
                    content=response,
                    turn_type=turn_type,
                )

            logger.debug("Stored conversation turns for session %s", session_id)

        except Exception as e:
            logger.warning("Failed to store conversation turn: %s", e)


# Factory functions


def get_atlas_agent_langgraph(session_id: Optional[str] = None) -> AtlasAgentGraph:
    """Get AtlasAgent using LangGraph."""
    return AtlasAgentGraph(session_id=session_id)
