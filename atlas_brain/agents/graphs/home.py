"""
HomeAgent LangGraph implementation.

Handles device commands: lights, TV, scenes.
Fast path - StateGraph for device control flow.
"""

import asyncio
import logging
import time
from typing import Any, Literal, Optional

from langgraph.graph import END, StateGraph

from .state import ActionResult, HomeAgentState, Intent
from ..entity_tracker import EntityTracker, extract_pronoun, has_pronoun
from ..tools import AtlasAgentTools, get_agent_tools
from ...utils.cuda_lock import get_cuda_lock

logger = logging.getLogger("atlas.agents.graphs.home")


# Node functions for the StateGraph


async def classify_intent(state: HomeAgentState) -> HomeAgentState:
    """
    Classify user input to determine action type.

    Uses fast intent routing (semantic embeddings) if available.
    """
    start_time = time.perf_counter()
    tools = get_agent_tools()
    input_text = state["input_text"]

    from ...config import settings

    route_result = None
    action_type = "conversation"
    confidence = 0.5
    needs_llm = True
    tools_to_call: list[str] = []

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
            needs_llm = False
            tools_to_call = [route_result.tool_name] if route_result.tool_name else []
            logger.info(
                "Fast route: tool_use/%s (conf=%.2f, %.0fms)",
                route_result.tool_name,
                route_result.confidence,
                route_result.route_time_ms,
            )

        # Fast path for high-confidence conversation
        elif (
            route_result.action_category == "conversation"
            and route_result.confidence >= threshold
        ):
            action_type = "conversation"
            confidence = route_result.confidence
            needs_llm = True
            logger.info(
                "Fast route: conversation (conf=%.2f, %.0fms)",
                route_result.confidence,
                route_result.route_time_ms,
            )

        # Device command from router
        elif (
            route_result.action_category == "device_command"
            and route_result.confidence >= threshold
        ):
            action_type = "device_command"
            confidence = route_result.confidence
            needs_llm = False
            logger.info(
                "Fast route: device_command (conf=%.2f, %.0fms)",
                route_result.confidence,
                route_result.route_time_ms,
            )

        # Parameterized tool - use LLM tool calling
        elif (
            route_result.action_category == "tool_use"
            and route_result.confidence >= threshold
            and route_result.tool_name
            and not route_result.fast_path_ok
        ):
            action_type = "tool_use"
            confidence = route_result.confidence
            needs_llm = True
            logger.info(
                "LLM tool route: %s (conf=%.2f, %.0fms)",
                route_result.tool_name,
                route_result.confidence,
                route_result.route_time_ms,
            )

    classify_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        "action_type": action_type,
        "confidence": confidence,
        "needs_llm": needs_llm,
        "classify_ms": classify_ms,
    }


async def parse_intent(state: HomeAgentState) -> HomeAgentState:
    """
    Parse detailed intent from user input.

    Uses LLM-based intent parsing for device commands.
    """
    start_time = time.perf_counter()
    tools = get_agent_tools()
    input_text = state["input_text"]
    action_type = state.get("action_type", "conversation")

    intent: Optional[Intent] = None

    # Only parse if we need device command details
    if action_type == "device_command" or state.get("confidence", 0) < 0.7:
        parsed = await tools.parse_intent(input_text)

        if parsed:
            # Convert to our Intent dataclass
            intent = Intent(
                action=parsed.action,
                target_type=parsed.target_type,
                target_name=parsed.target_name,
                target_id=getattr(parsed, "target_id", None),
                parameters=parsed.parameters or {},
                confidence=parsed.confidence,
                raw_query=input_text,
            )

            # Resolve pronouns if needed
            if intent and not intent.target_name and has_pronoun(input_text):
                pronoun = extract_pronoun(input_text)
                if pronoun:
                    # Use entity tracker from state if available
                    last_entity_name = state.get("last_entity_name")
                    last_entity_type = state.get("last_entity_type")
                    last_entity_id = state.get("last_entity_id")

                    if last_entity_name:
                        intent.target_name = last_entity_name
                        if last_entity_type:
                            intent.target_type = last_entity_type
                        if last_entity_id:
                            intent.target_id = last_entity_id
                        logger.info(
                            "Resolved '%s' -> %s/%s",
                            pronoun,
                            last_entity_type,
                            last_entity_name,
                        )

            # Refine action_type based on parsed intent
            if intent:
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
                else:
                    action_type = "conversation"

    think_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        "intent": intent,
        "action_type": action_type,
        "think_ms": think_ms,
        "needs_llm": action_type == "conversation" or action_type == "tool_use",
    }


async def execute_action(state: HomeAgentState) -> HomeAgentState:
    """
    Execute device command or tool.
    """
    start_time = time.perf_counter()
    tools = get_agent_tools()
    action_type = state.get("action_type", "none")
    intent = state.get("intent")

    result = ActionResult(success=False, message="No action taken")
    tool_results: dict[str, Any] = {}
    last_entity_name = state.get("last_entity_name")
    last_entity_type = state.get("last_entity_type")
    last_entity_id = state.get("last_entity_id")

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

            # Track entity for pronoun resolution
            if result.success and intent.target_name:
                last_entity_type = intent.target_type or "device"
                last_entity_name = intent.target_name
                last_entity_id = intent.target_id

        except Exception as e:
            logger.warning("Action execution failed: %s", e)
            result = ActionResult(
                success=False,
                message=str(e),
                error=str(e),
            )

    elif action_type == "tool_use":
        # Fast path: tool from router result
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
                logger.info(
                    "Fast tool executed: %s -> %s",
                    tool_name,
                    "success" if result.success else "failed",
                )
            except Exception as e:
                logger.warning("Fast tool execution failed: %s", e)
                result = ActionResult(
                    success=False,
                    message=str(e),
                    error=str(e),
                )

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
                logger.info(
                    "Tool executed: %s -> %s",
                    target_name,
                    "success" if result.success else "failed",
                )
            except Exception as e:
                logger.warning("Tool execution failed: %s", e)
                result = ActionResult(
                    success=False,
                    message=str(e),
                    error=str(e),
                )

    act_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        "action_result": result,
        "tool_results": tool_results,
        "act_ms": act_ms,
        "last_entity_name": last_entity_name,
        "last_entity_type": last_entity_type,
        "last_entity_id": last_entity_id,
    }


async def generate_response(state: HomeAgentState) -> HomeAgentState:
    """
    Generate response for the user.

    Uses templates for device commands, LLM for conversation/tools.
    """
    start_time = time.perf_counter()
    action_type = state.get("action_type", "none")
    action_result = state.get("action_result")
    intent = state.get("intent")
    needs_llm = state.get("needs_llm", False)
    input_text = state.get("input_text", "")

    response = ""
    llm_out: dict[str, Any] | None = None

    # Error case
    if action_result and action_result.error:
        response = f"I'm sorry, there was an error: {action_result.error}"

    # Device command response (template-based)
    elif action_type == "device_command":
        if action_result and action_result.success:
            if action_result.message:
                response = action_result.message
            else:
                response = "Done."
        else:
            if action_result and action_result.message:
                response = f"Sorry, {action_result.message}"
            else:
                response = "Sorry, I couldn't do that."

    # Tool use - check if already executed
    elif action_type == "tool_use":
        tool_results = state.get("tool_results", {})
        if tool_results:
            # Get message from first tool result
            for tool_name, result in tool_results.items():
                if result.get("message"):
                    response = result["message"]
                    break

        if not response and needs_llm:
            # Fall back to LLM tool calling
            llm_out = await _generate_llm_response_with_tools(state)
            response = llm_out["response"]

        if not response:
            response = "I couldn't process that request."

    # Conversation fallback
    elif action_type == "conversation":
        if needs_llm:
            llm_out = await _generate_llm_response(state)
            response = llm_out["response"]
        else:
            response = f"I heard: {input_text}"

    else:
        response = f"I heard: {input_text}"

    respond_ms = (time.perf_counter() - start_time) * 1000

    result = {
        **state,
        "response": response,
        "respond_ms": respond_ms,
        "llm_input_tokens": llm_out.get("input_tokens") if llm_out else None,
        "llm_output_tokens": llm_out.get("output_tokens") if llm_out else None,
        "llm_system_prompt": llm_out.get("system_prompt") if llm_out else None,
        "llm_history_count": llm_out.get("history_count", 0) if llm_out else 0,
        "llm_prompt_eval_duration_ms": llm_out.get("prompt_eval_duration_ms") if llm_out else None,
        "llm_eval_duration_ms": llm_out.get("eval_duration_ms") if llm_out else None,
        "llm_total_duration_ms": llm_out.get("total_duration_ms") if llm_out else None,
        "llm_provider_request_id": llm_out.get("provider_request_id") if llm_out else None,
        "llm_has_response": bool(llm_out and llm_out.get("has_llm_response")),
    }
    return result


async def _generate_llm_response(state: HomeAgentState) -> dict[str, Any]:
    """Generate response using LLM for conversation. Returns dict with response + metadata."""
    from ...services import llm_registry
    from ...services.protocols import Message

    llm = llm_registry.get_active()
    if llm is None:
        return {"response": f"I heard: {state.get('input_text', '')}",
                "input_tokens": None, "output_tokens": None,
                "system_prompt": None, "history_count": 0,
                "prompt_eval_duration_ms": None,
                "eval_duration_ms": None,
                "total_duration_ms": None,
                "provider_request_id": None,
                "has_llm_response": False}

    input_text = state.get("input_text", "")
    system_msg = "You are a helpful home assistant."

    messages = [
        Message(role="system", content=system_msg),
        Message(role="user", content=input_text),
    ]

    cuda_lock = get_cuda_lock()
    async with cuda_lock:
        result = llm.chat(
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )

    response = result.get("response", "").strip() or f"I heard: {input_text}"
    return {
        "response": response,
        "input_tokens": result.get("prompt_eval_count", 0),
        "output_tokens": result.get("eval_count", 0),
        "system_prompt": system_msg,
        "history_count": 0,
        "prompt_eval_duration_ms": result.get("prompt_eval_duration_ms"),
        "eval_duration_ms": result.get("eval_duration_ms"),
        "total_duration_ms": result.get("total_duration_ms"),
        "provider_request_id": result.get("request_id") or result.get("id"),
        "has_llm_response": True,
    }


async def _generate_llm_response_with_tools(state: HomeAgentState) -> dict[str, Any]:
    """Generate response using LLM tool calling loop. Returns dict with response + metadata."""
    from ...services import llm_registry
    from ...services.protocols import Message
    from ...services.tool_executor import execute_with_tools

    llm = llm_registry.get_active()
    if llm is None:
        return {"response": f"I heard: {state.get('input_text', '')}",
                "input_tokens": None, "output_tokens": None,
                "system_prompt": None, "history_count": 0,
                "prompt_eval_duration_ms": None,
                "eval_duration_ms": None,
                "total_duration_ms": None,
                "provider_request_id": None,
                "has_llm_response": False}

    input_text = state.get("input_text", "")
    intent = state.get("intent")

    system_msg = "You are a helpful assistant. Use tools when needed."
    messages = [
        Message(role="system", content=system_msg),
        Message(role="user", content=input_text),
    ]

    # Get target tool from intent
    target_tool = None
    if intent and intent.target_name:
        target_tool = intent.target_name
        logger.info("Tool use with target: %s", target_tool)

    cuda_lock = get_cuda_lock()
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

    return {
        "response": response if response else "I couldn't process that request.",
        "input_tokens": None,  # execute_with_tools doesn't expose token counts
        "output_tokens": None,
        "system_prompt": system_msg,
        "history_count": 0,
        "prompt_eval_duration_ms": None,
        "eval_duration_ms": None,
        "total_duration_ms": None,
        "provider_request_id": None,
        "has_llm_response": True,
    }


# Routing function


def route_by_action_type(
    state: HomeAgentState,
) -> Literal["parse", "execute", "respond"]:
    """Route based on action type from classification."""
    action_type = state.get("action_type", "conversation")
    confidence = state.get("confidence", 0.0)

    # High confidence device command - go straight to parse then execute
    if action_type == "device_command":
        return "parse"

    # High confidence tool use with fast path - go to execute
    if action_type == "tool_use" and not state.get("needs_llm", True):
        return "execute"

    # Tool use needing LLM - parse first
    if action_type == "tool_use":
        return "parse"

    # Conversation or low confidence - go to respond
    return "respond"


def route_after_parse(
    state: HomeAgentState,
) -> Literal["execute", "respond"]:
    """Route after parsing intent."""
    action_type = state.get("action_type", "conversation")

    if action_type in ("device_command", "tool_use"):
        return "execute"

    return "respond"


def route_after_execute(state: HomeAgentState) -> Literal["respond"]:
    """Always go to respond after execution."""
    return "respond"


# Build the graph


def build_home_agent_graph() -> StateGraph:
    """
    Build the HomeAgent LangGraph.

    Flow:
    classify -> (parse -> execute -> respond) | respond
    """
    graph = StateGraph(HomeAgentState)

    # Add nodes
    graph.add_node("classify", classify_intent)
    graph.add_node("parse", parse_intent)
    graph.add_node("execute", execute_action)
    graph.add_node("respond", generate_response)

    # Set entry point
    graph.set_entry_point("classify")

    # Add conditional edges
    graph.add_conditional_edges(
        "classify",
        route_by_action_type,
        {
            "parse": "parse",
            "execute": "execute",
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
    graph.add_edge("respond", END)

    return graph


# Compiled graph singleton
_home_graph: Optional[StateGraph] = None


def get_home_agent_graph() -> StateGraph:
    """Get or create the compiled HomeAgent graph."""
    global _home_graph
    if _home_graph is None:
        _home_graph = build_home_agent_graph()
    return _home_graph


class HomeAgentGraph:
    """
    HomeAgent using LangGraph for orchestration.

    Provides compatibility interface with the original HomeAgent.
    """

    def __init__(self, session_id: Optional[str] = None):
        """Initialize HomeAgent graph."""
        self._session_id = session_id
        self._graph = get_home_agent_graph()
        self._compiled = self._graph.compile()
        self._entity_tracker = EntityTracker()

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
        initial_state: HomeAgentState = {
            "input_text": input_text,
            "input_type": kwargs.get("input_type", "text"),
            "session_id": session_id or self._session_id,
            "speaker_id": speaker_id,
            "runtime_context": kwargs.get("runtime_context", {}),
            "messages": [],
            "action_type": "none",
            "confidence": 0.0,
            "needs_llm": False,
            "tool_results": {},
        }

        # Add entity tracker state
        recent = self._entity_tracker.get_recent(limit=1)
        last_entity = recent[0] if recent else None
        if last_entity:
            initial_state["last_entity_name"] = last_entity.entity_name
            initial_state["last_entity_type"] = last_entity.entity_type
            initial_state["last_entity_id"] = last_entity.entity_id

        # Run the graph
        try:
            final_state = await self._compiled.ainvoke(initial_state)
        except Exception as e:
            logger.exception("Error running HomeAgent graph: %s", e)
            final_state = {
                **initial_state,
                "response": f"I'm sorry, I encountered an error: {e}",
                "error": str(e),
            }

        # Update entity tracker
        if final_state.get("last_entity_name"):
            self._entity_tracker.track(
                entity_type=final_state.get("last_entity_type", "device"),
                entity_name=final_state["last_entity_name"],
                entity_id=final_state.get("last_entity_id"),
            )

        total_ms = (time.perf_counter() - start_time) * 1000

        # Build result
        return {
            "success": final_state.get("error") is None,
            "response_text": final_state.get("response", ""),
            "action_type": final_state.get("action_type", "none"),
            "intent": final_state.get("intent"),
            "error": final_state.get("error"),
            "tools_executed": final_state.get("tools_executed", []),
            "timing": {
                "total": total_ms,
                "classify": final_state.get("classify_ms", 0),
                "think": final_state.get("think_ms", 0),
                "act": final_state.get("act_ms", 0),
                "respond": final_state.get("respond_ms", 0),
            },
            "llm_meta": {
                "input_tokens": final_state.get("llm_input_tokens"),
                "output_tokens": final_state.get("llm_output_tokens"),
                "system_prompt": final_state.get("llm_system_prompt"),
                "history_count": final_state.get("llm_history_count", 0),
                "prompt_eval_duration_ms": final_state.get("llm_prompt_eval_duration_ms"),
                "eval_duration_ms": final_state.get("llm_eval_duration_ms"),
                "total_duration_ms": final_state.get("llm_total_duration_ms"),
                "provider_request_id": final_state.get("llm_provider_request_id"),
                "has_llm_response": final_state.get("llm_has_response", False),
            },
        }


# Factory functions for compatibility


def get_home_agent_langgraph(session_id: Optional[str] = None) -> HomeAgentGraph:
    """Get HomeAgent using LangGraph."""
    return HomeAgentGraph(session_id=session_id)
