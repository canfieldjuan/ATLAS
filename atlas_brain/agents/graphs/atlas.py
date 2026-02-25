"""
AtlasAgent LangGraph implementation.

Main router agent that delegates to sub-agents or handles directly.
Supports mode-based routing and conversation handling.
Uses Command(goto=...) for single-hop routing after classification.
"""

import asyncio
import logging
import re
import time
from typing import Any, Literal, Optional

from langgraph.graph import END, StateGraph
from langgraph.types import Command

from .state import ActionResult, AtlasAgentState, Intent
from .home import HomeAgentGraph
from ..entity_tracker import EntityTracker, extract_pronoun, has_pronoun
from ..tools import AtlasAgentTools, get_agent_tools
from ..memory import AtlasAgentMemory, get_agent_memory
from .workflow_state import get_workflow_state_manager
from ...utils.cuda_lock import get_cuda_lock
from .booking import run_booking_workflow, BOOKING_WORKFLOW_TYPE
from .reminder import run_reminder_workflow, REMINDER_WORKFLOW_TYPE
from .email import run_email_workflow, EMAIL_WORKFLOW_TYPE
from .email_query import run_email_query_workflow, EMAIL_QUERY_WORKFLOW_TYPE
from .call import run_call_workflow, CALL_WORKFLOW_TYPE
from .calendar import run_calendar_workflow, CALENDAR_WORKFLOW_TYPE
from .security import run_security_workflow, SECURITY_WORKFLOW_TYPE
from .presence import run_presence_workflow, PRESENCE_WORKFLOW_TYPE

logger = logging.getLogger("atlas.agents.graphs.atlas")

# Minimum facts to retrieve when an entity is detected via graph traversal
_ENTITY_MIN_FACTS = 5

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
    # Casual rejections / dismissals
    re.compile(r"^no[,.]?\s*(?:I'?m\s+)?(?:good|fine|thanks|thank\s+you|okay|ok|all\s+set|all\s+good)", re.IGNORECASE),
    re.compile(r"^(?:nah|nope|not?\s+(?:now|today|right\s+now|interested|anymore))", re.IGNORECASE),
    re.compile(r"^(?:that'?s\s+)?(?:all|enough|it|okay|ok|fine)[\s,.!]*(?:thanks|thank\s+you)?$", re.IGNORECASE),
    re.compile(r"^(?:I'?m\s+)?(?:done|finished|good\s+for\s+now|all\s+set)", re.IGNORECASE),
    re.compile(r"^(?:don'?t|do\s+not)\s+(?:worry|bother|need)", re.IGNORECASE),
]


_WORKFLOW_NAMES: dict[str, str] = {
    "reminder": "reminder",
    "calendar": "calendar event",
    "booking": "appointment",
    "email": "email draft",
    "email_query": "email query",
    "call": "phone call",
    "security": "security alert",
    "presence": "presence tracking",
}


def _is_cancel_intent(text: str) -> bool:
    """Check if text matches a cancel pattern."""
    text = text.strip()
    for pattern in _CANCEL_PATTERNS:
        if pattern.match(text):
            return True
    return False


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
        from ...config import settings
        return {
            **state,
            "action_type": "workflow_expired",
            "response": settings.voice.error_workflow_expired,
        }

    from ...modes.manager import get_mode_manager
    mode_manager = get_mode_manager()

    # Check for cancel intent
    input_text = state.get("input_text", "")
    if _is_cancel_intent(input_text):
        # In booking workflow, "cancel my appointment" is an appointment
        # cancellation request, not a workflow abort -- let LLM handle it
        _appt_keywords = ("appointment", "booking", "reservation")
        input_lower = input_text.lower()
        is_appt_cancel = (
            workflow.workflow_type == "booking"
            and any(kw in input_lower for kw in _appt_keywords)
        )
        if not is_appt_cancel:
            await manager.clear_workflow_state(session_id)
            mode_manager.set_workflow_active(False)
            logger.info("User cancelled active workflow for session %s", session_id)
            return {
                **state,
                "action_type": "workflow_cancelled",
                "response": "Okay, I've cancelled that.",
            }
        logger.info(
            "Appointment cancel in booking workflow, passing to LLM: %s",
            session_id,
        )

    # Active workflow found - mark for continuation and prevent mode timeout
    mode_manager.set_workflow_active(True)
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


async def classify_and_route(
    state: AtlasAgentState,
) -> Command[Literal[
    "delegate_home", "retrieve_memory", "execute", "respond", "start_workflow"
]]:
    """
    Classify user input and route to the appropriate next node.

    Uses semantic intent router for single-hop classification.
    Returns Command(goto=...) for LangGraph auto-routing.
    """
    # Mode switch already handled
    if state.get("action_type") == "mode_switch":
        return Command(
            update={"classify_ms": 0.0},
            goto="respond",
        )

    start_time = time.perf_counter()
    input_text = state["input_text"]

    from ...config import settings
    from ...services.intent_router import (
        PARAMETERLESS_TOOLS,
        ROUTE_TO_WORKFLOW,
        route_query,
    )

    # Use pre-computed route from voice pipeline if available, otherwise compute
    pre_route = state.get("runtime_context", {}).get("pre_route_result")
    if pre_route is not None:
        route_result = pre_route
        logger.info("Using pre-computed route result (skipping duplicate route_query)")
    else:
        route_result = await route_query(input_text)
    threshold = settings.intent_router.confidence_threshold
    conv_threshold = settings.intent_router.conversation_confidence_threshold

    action_category = route_result.action_category
    confidence = route_result.confidence
    route_name = route_result.raw_label
    tool_name = route_result.tool_name
    entity_name = getattr(route_result, "entity_name", None)

    classify_ms = (time.perf_counter() - start_time) * 1000

    # 1. Workflow routes (reminder, email, calendar_write, booking, call) -> start_workflow
    # In conversation mode, use a lower confidence floor so borderline workflow
    # intents (e.g. "call Juan" at 0.40) don't fall through to the general LLM
    # tool path. Applies to all workflow routes -- during active conversation the
    # user is more likely issuing commands than chatting about them.
    in_conversation = state.get("runtime_context", {}).get("in_conversation_mode", False)
    workflow_conf_floor = (
        settings.intent_router.conversation_workflow_confidence_floor
        if in_conversation else threshold
    )
    if route_name in ROUTE_TO_WORKFLOW and confidence >= workflow_conf_floor:
        workflow_type = ROUTE_TO_WORKFLOW[route_name]
        if in_conversation and confidence < threshold:
            logger.info(
                "Conversation workflow breakout: using lower confidence floor "
                "(%.2f < %.2f) for %s",
                confidence, threshold, workflow_type,
            )
        logger.info(
            "Route -> start_workflow/%s (conf=%.2f, %.0fms)",
            workflow_type, confidence, classify_ms,
        )
        return Command(
            update={
                "action_type": "workflow_start",
                "workflow_to_start": workflow_type,
                "confidence": confidence,
                "classify_ms": classify_ms,
                "entity_name": entity_name,
            },
            goto="start_workflow",
        )

    # 2. Device commands -> delegate_home
    if action_category == "device_command" and confidence >= threshold:
        logger.info("Route -> delegate_home (conf=%.2f, %.0fms)", confidence, classify_ms)
        return Command(
            update={
                "action_type": "device_command",
                "confidence": confidence,
                "delegate_to": "home",
                "classify_ms": classify_ms,
            },
            goto="delegate_home",
        )

    # 3. Parameterless tool fast path -> execute directly
    if (
        action_category == "tool_use"
        and tool_name in PARAMETERLESS_TOOLS
        and confidence >= threshold
    ):
        logger.info(
            "Route -> execute/%s fast path (conf=%.2f, %.0fms)",
            tool_name, confidence, classify_ms,
        )
        return Command(
            update={
                "action_type": "tool_use",
                "confidence": confidence,
                "tools_to_call": [tool_name] if tool_name else [],
                "tool_params": getattr(route_result, "tool_params", {}),
                "classify_ms": classify_ms,
            },
            goto="execute",
        )

    # 4. High-confidence conversation -> retrieve_memory (skip parse)
    if action_category == "conversation" and confidence >= conv_threshold:
        logger.info(
            "Route -> retrieve_memory/conversation (conf=%.2f, %.0fms)",
            confidence, classify_ms,
        )
        return Command(
            update={
                "action_type": "conversation",
                "confidence": confidence,
                "classify_ms": classify_ms,
                "entity_name": entity_name,
            },
            goto="retrieve_memory",
        )

    # 5. Low confidence / parameterized tool -> retrieve_memory -> parse
    logger.info(
        "Route -> retrieve_memory (low conf or param tool, cat=%s, conf=%.2f, %.0fms)",
        action_category, confidence, classify_ms,
    )
    return Command(
        update={
            "action_type": action_category,
            "confidence": confidence,
            "classify_ms": classify_ms,
            "entity_name": entity_name,
            "tool_name_hint": tool_name,
        },
        goto="retrieve_memory",
    )


async def retrieve_memory(
    state: AtlasAgentState,
) -> Command[Literal["parse", "respond"]]:
    """Retrieve structured memory sources, then route to parse or respond."""
    action_type = state.get("action_type", "conversation")

    # Only retrieve for conversation or tool use needing LLM
    if action_type not in ("conversation", "tool_use"):
        return Command(goto="parse")

    from ...config import settings

    conv_threshold = settings.intent_router.conversation_confidence_threshold

    if settings.memory.enabled and settings.memory.retrieve_context:
        start_time = time.perf_counter()
        input_text = state["input_text"]

        try:
            # Query classification: skip RAG for device commands, greetings, etc.
            from ...memory.query_classifier import get_query_classifier

            classifier = get_query_classifier()
            classification = classifier.classify(input_text)

            if not classification.use_rag:
                logger.debug(
                    "Skipping RAG (category=%s): %s",
                    classification.category,
                    input_text[:50],
                )
                memory_ms = (time.perf_counter() - start_time) * 1000
                # Set empty list explicitly so gather_context knows
                # "we searched and found nothing" vs "we didn't search"
                update = {
                    "retrieved_sources": [],
                    "memory_ms": memory_ms,
                }
                if action_type == "conversation" and state.get("confidence", 0) >= conv_threshold:
                    return Command(update=update, goto="respond")
                return Command(update=update, goto="parse")

            # Search graphiti for structured sources (with entity traversal if detected)
            from ...memory.rag_client import get_rag_client

            client = get_rag_client()
            entity_name = state.get("entity_name")
            max_facts = settings.memory.context_results
            if entity_name:
                logger.info("Entity detected: %r -- running parallel search + traversal", entity_name)
                max_facts = max(max_facts, _ENTITY_MIN_FACTS)

            search_result = await client.search_with_traversal(
                query=input_text,
                entity_name=entity_name,
                max_facts=max_facts,
            )

            memory_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Retrieved %d memory sources in %.0fms",
                len(search_result.facts),
                memory_ms,
            )

            update = {
                "retrieved_sources": search_result.facts,
                "memory_ms": memory_ms,
            }

            # Conversation + high confidence -> respond directly
            if action_type == "conversation" and state.get("confidence", 0) >= conv_threshold:
                return Command(update=update, goto="respond")

            return Command(update=update, goto="parse")

        except asyncio.TimeoutError:
            logger.warning("Memory retrieval timed out")
        except Exception as e:
            logger.warning("Memory retrieval failed: %s", e)

        # On failure, set empty list so gather_context doesn't retry
        # the same slow/broken server (avoids 2x timeout latency)
        memory_ms = (time.perf_counter() - start_time) * 1000
        fail_update = {"retrieved_sources": [], "memory_ms": memory_ms}
        if action_type == "conversation" and state.get("confidence", 0) >= conv_threshold:
            return Command(update=fail_update, goto="respond")
        return Command(update=fail_update, goto="parse")

    # Memory disabled -- don't set retrieved_sources so gather_context
    # can still do its own search if include_rag is True
    if action_type == "conversation" and state.get("confidence", 0) >= conv_threshold:
        return Command(goto="respond")

    return Command(goto="parse")


async def parse_intent(
    state: AtlasAgentState,
) -> Command[Literal["execute", "respond"]]:
    """Parse detailed intent from user input, then route to execute or respond."""
    # Skip if delegating, mode switch, or pure conversation (no intent to parse)
    if state.get("delegate_to") or state.get("action_type") in ("mode_switch", "conversation"):
        return Command(goto="respond")

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

    if action_type in ("device_command", "tool_use"):
        return Command(
            update={
                "intent": intent,
                "action_type": action_type,
                "think_ms": think_ms,
            },
            goto="execute",
        )

    return Command(
        update={
            "intent": intent,
            "action_type": action_type,
            "think_ms": think_ms,
        },
        goto="respond",
    )


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
                tool_result = await tools.execute_tool(tool_name, state.get("tool_params") or {})
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

        # Slow path: tool from LLM intent (or router hint fallback)
        else:
            target_name = (
                (intent.target_name if intent else None)
                or state.get("tool_name_hint")
            )
            params = (intent.parameters if intent else None) or {}
            if target_name:
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
    llm_out: dict[str, Any] | None = None

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
            llm_out = await _generate_llm_response(state, with_tools=True)
            response = llm_out["response"]

    # Conversation
    elif action_type == "conversation":
        llm_out = await _generate_llm_response(state)
        response = llm_out["response"]

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
        "rag_graph_used": llm_out.get("rag_graph_used") if llm_out else False,
        "rag_nodes_retrieved": llm_out.get("rag_nodes_retrieved") if llm_out else None,
        "rag_chunks_used": llm_out.get("rag_chunks_used") if llm_out else None,
        "context_tokens": llm_out.get("context_tokens") if llm_out else None,
        "retrieval_latency_ms": llm_out.get("retrieval_latency_ms") if llm_out else None,
    }

    # Propagate tool execution from LLM tool calling (conversation with MCP tools)
    if llm_out and llm_out.get("tools_executed"):
        result["tools_executed"] = llm_out["tools_executed"]
        result["tool_results"] = llm_out.get("tool_results", {})

    return result


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
    from ...modes.manager import get_mode_manager
    from ...services.llm_router import get_llm

    start_time = time.perf_counter()
    active_workflow = state.get("active_workflow", {})
    workflow_type = active_workflow.get("workflow_type")
    session_id = state.get("session_id")
    input_text = state.get("input_text", "")

    result = None

    if workflow_type == BOOKING_WORKFLOW_TYPE:
        result = await run_booking_workflow(
            input_text=input_text,
            session_id=session_id,
            speaker_id=state.get("speaker_id"),
            llm=get_llm("booking"),
        )
    elif workflow_type == REMINDER_WORKFLOW_TYPE:
        result = await run_reminder_workflow(
            input_text=input_text,
            session_id=session_id,
        )
    elif workflow_type == EMAIL_WORKFLOW_TYPE:
        result = await run_email_workflow(
            input_text=input_text,
            session_id=session_id,
            llm=get_llm("email"),
        )
    elif workflow_type == EMAIL_QUERY_WORKFLOW_TYPE:
        result = await run_email_query_workflow(
            input_text=input_text,
            session_id=session_id,
            speaker_id=state.get("speaker_id"),
            llm=get_llm("email_query"),
        )
    elif workflow_type == CALL_WORKFLOW_TYPE:
        result = await run_call_workflow(
            input_text=input_text,
            session_id=session_id,
            speaker_id=state.get("speaker_id"),
            llm=get_llm("call"),
        )
    elif workflow_type == CALENDAR_WORKFLOW_TYPE:
        result = await run_calendar_workflow(
            input_text=input_text,
            session_id=session_id,
        )
    elif workflow_type == SECURITY_WORKFLOW_TYPE:
        result = await run_security_workflow(
            input_text=input_text,
            session_id=session_id,
        )
    elif workflow_type == PRESENCE_WORKFLOW_TYPE:
        result = await run_presence_workflow(
            input_text=input_text,
            session_id=session_id,
            user_id=state.get("speaker_id"),
        )

    if result is None:
        logger.warning("Unknown workflow type: %s", workflow_type)
        get_mode_manager().set_workflow_active(False)
        return {
            **state,
            "response": "I'm not sure how to continue. Could you start over?",
            "error": "unknown_workflow_type",
        }

    response = result.get("response", "")
    total_ms = (time.perf_counter() - start_time) * 1000
    awaiting = result.get("awaiting_user_input", False)
    llm_meta = result.get("llm_meta") or {}

    if not awaiting:
        get_mode_manager().set_workflow_active(False)

    return {
        **state,
        "response": response,
        "action_type": "tool_use",
        "act_ms": total_ms,
        "awaiting_user_input": awaiting,
        "tools_executed": result.get("tools_executed", []),
        "llm_input_tokens": llm_meta.get("input_tokens"),
        "llm_output_tokens": llm_meta.get("output_tokens"),
        "llm_system_prompt": llm_meta.get("system_prompt"),
        "llm_history_count": llm_meta.get("history_count", 0),
        "llm_prompt_eval_duration_ms": llm_meta.get("prompt_eval_duration_ms"),
        "llm_eval_duration_ms": llm_meta.get("eval_duration_ms"),
        "llm_total_duration_ms": llm_meta.get("total_duration_ms"),
        "llm_provider_request_id": llm_meta.get("provider_request_id"),
        "llm_has_response": bool(llm_meta.get("has_llm_response")),
    }


async def start_workflow(state: AtlasAgentState) -> AtlasAgentState:
    """Start a new workflow based on detected intent."""
    from ...modes.manager import get_mode_manager
    from ...services.llm_router import get_llm
    mode_manager = get_mode_manager()
    mode_manager.set_workflow_active(True)

    start_time = time.perf_counter()
    workflow_type = state.get("workflow_to_start")
    session_id = state.get("session_id")
    input_text = state.get("input_text", "")

    logger.info("Starting %s workflow for session %s", workflow_type, session_id)

    if workflow_type == BOOKING_WORKFLOW_TYPE:
        result = await run_booking_workflow(
            input_text=input_text,
            session_id=session_id,
            speaker_id=state.get("speaker_id"),
            llm=get_llm("booking"),
        )
    elif workflow_type == REMINDER_WORKFLOW_TYPE:
        result = await run_reminder_workflow(
            input_text=input_text,
            session_id=session_id,
        )
    elif workflow_type == EMAIL_WORKFLOW_TYPE:
        result = await run_email_workflow(
            input_text=input_text,
            session_id=session_id,
            llm=get_llm("email"),
        )
    elif workflow_type == EMAIL_QUERY_WORKFLOW_TYPE:
        result = await run_email_query_workflow(
            input_text=input_text,
            session_id=session_id,
            speaker_id=state.get("speaker_id"),
            llm=get_llm("email_query"),
        )
    elif workflow_type == CALL_WORKFLOW_TYPE:
        result = await run_call_workflow(
            input_text=input_text,
            session_id=session_id,
            speaker_id=state.get("speaker_id"),
            llm=get_llm("call"),
        )
    elif workflow_type == CALENDAR_WORKFLOW_TYPE:
        result = await run_calendar_workflow(
            input_text=input_text,
            session_id=session_id,
        )
    elif workflow_type == SECURITY_WORKFLOW_TYPE:
        result = await run_security_workflow(
            input_text=input_text,
            session_id=session_id,
        )
    elif workflow_type == PRESENCE_WORKFLOW_TYPE:
        result = await run_presence_workflow(
            input_text=input_text,
            session_id=session_id,
            user_id=state.get("speaker_id"),
        )
    else:
        logger.warning("Unknown workflow type to start: %s", workflow_type)
        return {
            **state,
            "response": "I'm not sure how to help with that.",
            "error": "unknown_workflow_type",
        }

    response = result.get("response", "")
    total_ms = (time.perf_counter() - start_time) * 1000
    awaiting = result.get("awaiting_user_input", False)
    llm_meta = result.get("llm_meta") or {}

    if not awaiting:
        mode_manager.set_workflow_active(False)

    return {
        **state,
        "response": response,
        "action_type": "workflow_started",
        "workflow_type": workflow_type,
        "act_ms": total_ms,
        "awaiting_user_input": awaiting,
        "tools_executed": result.get("tools_executed", []),
        "llm_input_tokens": llm_meta.get("input_tokens"),
        "llm_output_tokens": llm_meta.get("output_tokens"),
        "llm_system_prompt": llm_meta.get("system_prompt"),
        "llm_history_count": llm_meta.get("history_count", 0),
        "llm_prompt_eval_duration_ms": llm_meta.get("prompt_eval_duration_ms"),
        "llm_eval_duration_ms": llm_meta.get("eval_duration_ms"),
        "llm_total_duration_ms": llm_meta.get("total_duration_ms"),
        "llm_provider_request_id": llm_meta.get("provider_request_id"),
        "llm_has_response": bool(llm_meta.get("has_llm_response")),
    }


async def _generate_llm_response(
    state: AtlasAgentState,
    with_tools: bool = False,
) -> dict[str, Any]:
    """Generate response using LLM. Returns dict with response + metadata."""
    from ...services import llm_registry
    from ...services.protocols import Message
    from ...memory.service import get_memory_service

    llm = llm_registry.get_active()
    if llm is None:
        return {"response": f"I heard: {state.get('input_text', '')}",
                "input_tokens": None, "output_tokens": None,
                "system_prompt": None, "history_count": 0,
                "prompt_eval_duration_ms": None,
                "eval_duration_ms": None,
                "total_duration_ms": None,
                "provider_request_id": None,
                "has_llm_response": False,
                "rag_graph_used": False,
                "rag_nodes_retrieved": None,
                "rag_chunks_used": None,
                "context_tokens": None,
                "retrieval_latency_ms": None}

    input_text = state.get("input_text", "")
    retrieved_sources = state.get("retrieved_sources")
    speaker_id = state.get("speaker_id")
    session_id = state.get("session_id")

    # Gather unified context via MemoryService (history + profile + token budget).
    # Pass pre-fetched sources from retrieve_memory so gather_context can
    # track them for feedback without re-searching graphiti.
    # When retrieved_sources is None (voice path), gather_context does its own RAG.
    svc = get_memory_service()
    gather_started = time.perf_counter()
    mem_ctx = await svc.gather_context(
        query=input_text,
        session_id=session_id,
        user_id=state.get("runtime_context", {}).get("speaker_uuid"),
        include_rag=True,
        pre_fetched_sources=retrieved_sources,
        include_history=True,
        include_physical=False,
        max_history=6,
    )
    gather_context_ms = (time.perf_counter() - gather_started) * 1000

    # Stash RAG usage_ids for correction feedback loop
    if mem_ctx.feedback_context and mem_ctx.feedback_context.usage_ids and session_id:
        try:
            from ...memory.feedback import get_feedback_service
            get_feedback_service().stash_session_usage(
                session_id, mem_ctx.feedback_context.usage_ids,
            )
        except Exception:
            pass

    rag_nodes_retrieved = 0
    if mem_ctx.rag_context_used and mem_ctx.rag_result and mem_ctx.rag_result.sources:
        rag_nodes_retrieved = len(mem_ctx.rag_result.sources)

    context_tokens: Optional[int] = None
    if isinstance(mem_ctx.token_usage, dict):
        try:
            rag_tokens = mem_ctx.token_usage.get("rag_context")
            if rag_tokens is not None:
                context_tokens = int(rag_tokens)
        except (TypeError, ValueError):
            context_tokens = None

    retrieval_latency_ms: Optional[float] = None
    memory_ms = state.get("memory_ms")
    if isinstance(memory_ms, (int, float)) and memory_ms > 0:
        retrieval_latency_ms = float(memory_ms)
    elif rag_nodes_retrieved > 0:
        retrieval_latency_ms = gather_context_ms

    # Build history messages from MemoryContext (already chronological order)
    history_messages: list[Message] = [
        Message(role=h["role"], content=h["content"])
        for h in mem_ctx.conversation_history
    ]
    if history_messages:
        logger.debug("Added %d conversation turns to LLM context", len(history_messages))

    # Build system prompt from centralized persona config
    from ...config import settings as _settings
    system_parts = [_settings.persona.system_prompt]

    # Add physical awareness context (people, objects, devices, room, time)
    try:
        from ...orchestration.context import get_context
        awareness = get_context().build_context_string()
        if awareness:
            system_parts.append(f"\nCurrent awareness:\n{awareness}")
    except Exception as e:
        logger.debug("Could not build awareness context: %s", e)

    # Append learned temporal patterns (routines)
    try:
        from ...orchestration.temporal import get_temporal_context
        temporal = await get_temporal_context()
        if temporal:
            system_parts.append(temporal)
    except Exception:
        pass

    # Add user profile from MemoryService context
    if mem_ctx.user_name:
        system_parts.append(f"The user's name is {mem_ctx.user_name}.")
    if mem_ctx.response_style == "brief":
        system_parts.append("Preference: Keep responses short and concise.")
    elif mem_ctx.response_style == "detailed":
        system_parts.append("Preference: Provide thorough explanations.")
    if mem_ctx.expertise_level == "beginner":
        system_parts.append("Level: Explain concepts simply.")
    elif mem_ctx.expertise_level == "expert":
        system_parts.append("Level: Use technical language freely.")

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

    # Add RAG context from GraphRAG knowledge graph
    if mem_ctx.rag_context_used and mem_ctx.rag_result and mem_ctx.rag_result.sources:
        entity_facts = []
        search_facts = []
        for s in mem_ctx.rag_result.sources:
            if not s.fact:
                continue
            if s.source_type == "entity_edge":
                entity_facts.append(s.fact)
            else:
                search_facts.append(s.fact)

        entity_name = state.get("entity_name")
        if entity_facts:
            label = f"Known facts about {entity_name}" if entity_name else "Known facts"
            system_parts.append(f"\n{label}:\n" + "\n".join(f"- {f}" for f in entity_facts))
        if search_facts:
            system_parts.append("\nRelated context:\n" + "\n".join(f"- {f}" for f in search_facts))

    # Inject entity context (recent turns + current room for disambiguation)
    try:
        from ...voice.entity_context import collect_recent_entities, format_entity_context
        entity_dicts = list(mem_ctx.recent_entities)
        try:
            from ...orchestration.context import get_context
            _room = get_context().build_context_dict().get("location")
            if _room:
                entity_dicts.insert(0, {
                    "type": "location",
                    "name": _room,
                    "action": "current room",
                    "source": "context",
                })
        except Exception:
            pass
        if entity_dicts:
            refs = collect_recent_entities(entity_dicts)
            entity_str = format_entity_context(refs)
            if entity_str:
                system_parts.append(f"\n{entity_str}")
    except Exception as e:
        logger.debug("Entity context injection failed: %s", e)

    system_msg = " ".join(system_parts)
    messages = [Message(role="system", content=system_msg)]

    # Add conversation history (provides context for follow-up questions)
    messages.extend(history_messages)

    # Add current user message
    messages.append(Message(role="user", content=input_text))

    # Conversation mode may override max_tokens for both tool-calling and
    # plain chat paths (higher limit for multi-tool responses).
    conv_max_tokens = state.get("runtime_context", {}).get("conversation_max_tokens")

    # Use tool calling for conversation turns when LLM supports it.
    # This gives the model access to MCP tools (CRM, calendar, email,
    # telephony) so it can answer queries like "search for contact Smith"
    # or "what's on my calendar today" during general conversation.
    # Skip for with_tools=True (existing tool result summarization).
    if not with_tools and hasattr(llm, "chat_with_tools"):
        from ...services.tool_executor import execute_with_tools

        try:
            ewt_result = await execute_with_tools(
                llm=llm,
                messages=messages,
                max_tokens=conv_max_tokens if conv_max_tokens else 200,
                temperature=0.7,
            )
            response = ewt_result.get("response", "").strip()
            llm_meta = ewt_result.get("llm_meta", {})
            tools_used = ewt_result.get("tools_executed", [])

            # Use the result when tools were called (response summarizes tool
            # output) or when the LLM produced a genuine text response.
            # When the LLM returned empty and no tools were called,
            # execute_with_tools substitutes a generic fallback -- skip it
            # and try plain chat instead (tool schemas may confuse the model).
            if tools_used:
                return {
                    "response": response,
                    "input_tokens": llm_meta.get("input_tokens"),
                    "output_tokens": llm_meta.get("output_tokens"),
                    "system_prompt": system_msg,
                    "history_count": len(history_messages),
                    "prompt_eval_duration_ms": llm_meta.get("prompt_eval_duration_ms"),
                    "eval_duration_ms": llm_meta.get("eval_duration_ms"),
                    "total_duration_ms": llm_meta.get("total_duration_ms"),
                    "provider_request_id": llm_meta.get("provider_request_id"),
                    "has_llm_response": bool(llm_meta.get("has_llm_response")),
                    "rag_graph_used": rag_nodes_retrieved > 0,
                    "rag_nodes_retrieved": rag_nodes_retrieved if rag_nodes_retrieved > 0 else None,
                    "rag_chunks_used": rag_nodes_retrieved if rag_nodes_retrieved > 0 else None,
                    "context_tokens": context_tokens,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "tools_executed": tools_used,
                    "tool_results": ewt_result.get("tool_results", {}),
                }
            if response:
                from ...services.tool_executor import EMPTY_RESPONSE_FALLBACK

                if response != EMPTY_RESPONSE_FALLBACK:
                    return {
                        "response": response,
                        "input_tokens": llm_meta.get("input_tokens"),
                        "output_tokens": llm_meta.get("output_tokens"),
                        "system_prompt": system_msg,
                        "history_count": len(history_messages),
                        "prompt_eval_duration_ms": llm_meta.get("prompt_eval_duration_ms"),
                        "eval_duration_ms": llm_meta.get("eval_duration_ms"),
                        "total_duration_ms": llm_meta.get("total_duration_ms"),
                        "provider_request_id": llm_meta.get("provider_request_id"),
                        "has_llm_response": bool(llm_meta.get("has_llm_response")),
                        "rag_graph_used": rag_nodes_retrieved > 0,
                        "rag_nodes_retrieved": rag_nodes_retrieved if rag_nodes_retrieved > 0 else None,
                        "rag_chunks_used": rag_nodes_retrieved if rag_nodes_retrieved > 0 else None,
                        "context_tokens": context_tokens,
                        "retrieval_latency_ms": retrieval_latency_ms,
                    }
                logger.info("Tool-calling LLM returned empty, trying plain chat")
        except Exception as e:
            logger.warning("Tool-calling LLM failed, falling back to chat: %s", e)

    # Fallback: plain chat (LLM without tool support, tool calling failed,
    # or with_tools=True for existing tool result summarization).
    try:
        cuda_lock = get_cuda_lock()
        async with cuda_lock:
            llm_result = llm.chat(
                messages=messages,
                max_tokens=conv_max_tokens if conv_max_tokens else 100,
                temperature=0.7,
            )
        response = llm_result.get("response", "").strip()
        if response:
            return {
                "response": response,
                "input_tokens": llm_result.get("prompt_eval_count", 0),
                "output_tokens": llm_result.get("eval_count", 0),
                "system_prompt": system_msg,
                "history_count": len(history_messages),
                "prompt_eval_duration_ms": llm_result.get("prompt_eval_duration_ms"),
                "eval_duration_ms": llm_result.get("eval_duration_ms"),
                "total_duration_ms": llm_result.get("total_duration_ms"),
                "provider_request_id": llm_result.get("request_id") or llm_result.get("id"),
                "has_llm_response": True,
                "rag_graph_used": rag_nodes_retrieved > 0,
                "rag_nodes_retrieved": rag_nodes_retrieved if rag_nodes_retrieved > 0 else None,
                "rag_chunks_used": rag_nodes_retrieved if rag_nodes_retrieved > 0 else None,
                "context_tokens": context_tokens,
                "retrieval_latency_ms": retrieval_latency_ms,
            }
    except Exception as e:
        logger.warning("LLM response generation failed: %s", e)

    return {"response": f"I heard: {input_text}",
            "input_tokens": None, "output_tokens": None,
            "system_prompt": None, "history_count": 0,
            "prompt_eval_duration_ms": None,
            "eval_duration_ms": None,
            "total_duration_ms": None,
            "provider_request_id": None,
            "has_llm_response": False,
            "rag_graph_used": rag_nodes_retrieved > 0,
            "rag_nodes_retrieved": rag_nodes_retrieved if rag_nodes_retrieved > 0 else None,
            "rag_chunks_used": rag_nodes_retrieved if rag_nodes_retrieved > 0 else None,
            "context_tokens": context_tokens,
            "retrieval_latency_ms": retrieval_latency_ms}


# Routing function (only for check_workflow, which doesn't use Command)


def route_after_check_workflow(
    state: AtlasAgentState,
) -> Literal["continue_workflow", "classify", "respond"]:
    """Route after checking for active workflow."""
    action_type = state.get("action_type", "")

    # Mode switch, workflow cancelled, or workflow expired - go to respond
    if action_type in ("mode_switch", "workflow_cancelled", "workflow_expired"):
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
    preprocess -> check_workflow -> (continue | classify_and_route -> ...)
    classify, retrieve_memory, parse return Command(goto=...) for auto-routing.
    """
    graph = StateGraph(AtlasAgentState)

    # Add nodes
    graph.add_node("preprocess", preprocess_input)
    graph.add_node("check_workflow", check_active_workflow)
    graph.add_node("continue_workflow", continue_workflow)
    graph.add_node("start_workflow", start_workflow)
    graph.add_node("classify", classify_and_route)
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

    # classify, retrieve_memory, parse return Command(goto=...) -> auto-routed
    graph.add_edge("execute", "respond")
    graph.add_edge("delegate_home", END)
    graph.add_edge("continue_workflow", END)
    graph.add_edge("start_workflow", END)
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
        effective_session = session_id or self._session_id

        # Pre-pop stashed RAG usage_ids before the graph runs.
        # Must happen before _generate_llm_response can overwrite the stash
        # with the current turn's sources.
        prev_rag_usage_ids: list = []
        if effective_session:
            try:
                from ...memory.feedback import get_feedback_service
                prev_rag_usage_ids = get_feedback_service().pop_session_usage(
                    effective_session,
                )
            except Exception:
                pass

        # Build initial state
        initial_state: AtlasAgentState = {
            "input_text": input_text,
            "input_type": kwargs.get("input_type", "text"),
            "session_id": effective_session,
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

        # Store conversation turn (pass pre-popped usage_ids for correction feedback)
        await self._store_turn(final_state, prev_rag_usage_ids=prev_rag_usage_ids)

        # Build result
        return {
            "success": final_state.get("error") is None,
            "response_text": final_state.get("response", ""),
            "action_type": final_state.get("action_type", "none"),
            "intent": final_state.get("intent"),
            "error": final_state.get("error"),
            "awaiting_user_input": final_state.get("awaiting_user_input", False),
            "tools_executed": final_state.get("tools_executed", []),
            "timing": {
                "total": total_ms,
                "classify": final_state.get("classify_ms", 0),
                "memory": final_state.get("memory_ms", 0),
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
                "rag_graph_used": final_state.get("rag_graph_used", False),
                "rag_nodes_retrieved": final_state.get("rag_nodes_retrieved"),
                "rag_chunks_used": final_state.get("rag_chunks_used"),
                "context_tokens": final_state.get("context_tokens"),
                "retrieval_latency_ms": final_state.get("retrieval_latency_ms"),
            },
        }

    async def _store_turn(
        self,
        state: AtlasAgentState,
        prev_rag_usage_ids: Optional[list] = None,
    ) -> None:
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
            input_text = state.get("input_text", "")

            # Detect memory quality signals (fail-open)
            user_metadata: Optional[dict] = None
            assistant_metadata: Optional[dict] = None
            try:
                from ...memory.quality import get_quality_detector
                signal = get_quality_detector().detect(
                    session_id=session_id,
                    user_content=input_text,
                    turn_type=turn_type,
                )
                user_metadata = signal.to_metadata() or None
                if signal.correction:
                    assistant_metadata = {"memory_quality": {"preceded_by_correction": True}}
                    # Downvote RAG sources from the turn being corrected.
                    # Uses pre-popped ids from run() to avoid the timing bug
                    # where _generate_llm_response overwrites the stash before
                    # we can pop here.
                    if prev_rag_usage_ids:
                        try:
                            from ...memory.feedback import get_feedback_service
                            await get_feedback_service().record_not_helpful(
                                prev_rag_usage_ids, feedback_type="correction",
                            )
                            logger.info(
                                "Downvoted %d RAG sources on correction for session %s",
                                len(prev_rag_usage_ids), session_id,
                            )
                        except Exception as e:
                            logger.debug("Correction feedback failed: %s", e)
                    logger.info(
                        "Quality signal: correction=%s pattern=%s for session %s",
                        signal.correction, signal.correction_pattern, session_id,
                    )
                if signal.repetition:
                    logger.info(
                        "Quality signal: repetition (sim=%.3f) of turn %s for session %s",
                        signal.repetition_similarity or 0,
                        signal.repetition_of_turn_id,
                        session_id,
                    )
            except Exception as e:
                logger.debug("Quality detection skipped: %s", e)

            # Entity extraction from intent and state
            entities = []
            intent_obj = state.get("intent")
            action_type = state.get("action_type", "")
            _speaker = state.get("speaker_id")

            if intent_obj:
                if turn_type == "command" and (intent_obj.target_type or intent_obj.target_name):
                    device_name = " ".join(
                        filter(None, [intent_obj.target_name, intent_obj.target_type])
                    ).strip()
                    if device_name:
                        entities.append({
                            "type": "device",
                            "name": device_name,
                            "action": intent_obj.action or "",
                            "source": "command",
                        })
                elif action_type == "tool_use":
                    _topic_name = intent_obj.target_name
                    if not _topic_name:
                        # Fallback: infer topic from transcript (mirrors launcher.py streaming path)
                        try:
                            from ...voice.entity_context import extract_topic_from_text
                            _topic_name = extract_topic_from_text(
                                state.get("input_text", "") or ""
                            )
                        except Exception:
                            pass
                    if _topic_name:
                        entities.append({
                            "type": "topic",
                            "name": _topic_name,
                            "action": intent_obj.action or "",
                            "source": "tool",
                        })

            # Location from user input then response text (first match wins, mirrors launcher.py)
            _input_text = state.get("input_text", "") or ""
            response_text = state.get("response", "") or ""
            for _loc_text in (_input_text, response_text):
                if _loc_text:
                    try:
                        from ...voice.entity_context import extract_location_from_text
                        location = extract_location_from_text(_loc_text)
                        if location:
                            entities.append({"type": "location", "name": location, "source": "text"})
                            break  # one location per turn
                    except Exception:
                        pass

            # Workflow topic entity from workflow type
            _workflow_type_val = (
                state.get("workflow_type")
                or state.get("workflow_to_start")
                or (state.get("active_workflow") or {}).get("workflow_type")
            )
            if action_type in ("workflow_start", "workflow_continuation", "workflow_started") and _workflow_type_val:
                _wf_name = _WORKFLOW_NAMES.get(_workflow_type_val, _workflow_type_val)
                _wf_action = (
                    "set"
                    if (action_type == "workflow_started" and not state.get("awaiting_user_input"))
                    else "setting"
                )
                entities.append({
                    "type": "topic",
                    "name": _wf_name,
                    "action": _wf_action,
                    "source": "workflow",
                })

            # Person entity from speaker
            if _speaker:
                entities.append({"type": "person", "name": _speaker, "source": "speaker"})

            if entities:
                if assistant_metadata is None:
                    assistant_metadata = {}
                assistant_metadata.setdefault("entities", []).extend(entities)

            # Store user turn
            runtime_ctx = state.get("runtime_context", {})
            await self._memory.add_turn(
                session_id=session_id,
                role="user",
                content=input_text,
                speaker_id=state.get("speaker_id"),
                speaker_uuid=runtime_ctx.get("speaker_uuid"),
                intent=intent_str,
                turn_type=turn_type,
                metadata=user_metadata,
            )

            # Store assistant turn (sanitize to prevent tool-call artifacts
            # from poisoning conversation history on subsequent turns).
            response = state.get("response")
            if response:
                from ...memory.service import _sanitize_for_storage
                await self._memory.add_turn(
                    session_id=session_id,
                    role="assistant",
                    content=_sanitize_for_storage(response),
                    speaker_uuid=runtime_ctx.get("speaker_uuid"),
                    turn_type=turn_type,
                    metadata=assistant_metadata,
                )

            logger.debug("Stored conversation turns for session %s", session_id)

            # Emit event for reasoning agent
            from ...reasoning.producers import emit_if_enabled
            await emit_if_enabled(
                "voice.turn_completed", "atlas_agent",
                {"session_id": session_id, "turn_type": turn_type,
                 "intent": intent_str,
                 "input_preview": (state.get("input") or "")[:200]},
                entity_type="contact" if runtime_ctx.get("speaker_uuid") else None,
                entity_id=runtime_ctx.get("speaker_uuid"),
            )

        except Exception as e:
            logger.warning("Failed to store conversation turn: %s", e)


# Factory functions


def get_atlas_agent_langgraph(session_id: Optional[str] = None) -> AtlasAgentGraph:
    """Get AtlasAgent using LangGraph."""
    return AtlasAgentGraph(session_id=session_id)
