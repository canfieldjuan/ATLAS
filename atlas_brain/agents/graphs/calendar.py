"""
LLM-driven calendar workflow.

The LLM drives the conversation naturally using two tools:
  - create_calendar_event: Create a new event
  - get_calendar: Query upcoming events

Conversation history persists across voice turns via WorkflowStateManager.
"""

import logging
import time
from typing import Optional

from .workflow_state import get_workflow_state_manager
from ...services.protocols import Message

logger = logging.getLogger("atlas.agents.graphs.calendar")

# Workflow type identifier for state persistence
CALENDAR_WORKFLOW_TYPE = "calendar"


def _build_calendar_system_prompt() -> str:
    """Build system prompt for calendar conversation."""
    from datetime import date
    today = date.today()
    today_str = today.strftime("%A, %B %d, %Y")

    return (
        f"You are a helpful calendar assistant. Today is {today_str}.\n\n"
        f"You can create events, query upcoming events, and check for free slots. "
        f"Use the matching tool from your tool list for each action.\n\n"
        f"Be conversational and brief. This is a voice interface -- keep responses "
        f"under 2-3 sentences."
    )


async def run_calendar_workflow(
    input_text: str,
    session_id: Optional[str] = None,
) -> dict:
    """
    Run calendar as an LLM conversation with tools.

    The LLM decides whether to create events or query the calendar
    based on user input. Conversation history persists across turns
    via WorkflowStateManager.
    """
    from ...services import llm_registry
    from ...services.tool_executor import execute_with_tools
    from ...tools import tool_registry

    start_time = time.perf_counter()
    empty_llm_meta = {
        "input_tokens": None,
        "output_tokens": None,
        "prompt_eval_duration_ms": None,
        "eval_duration_ms": None,
        "total_duration_ms": None,
        "provider_request_id": None,
        "has_llm_response": False,
    }
    llm = llm_registry.get_active()

    if llm is None:
        return {
            "response": "I can't manage the calendar right now.",
            "awaiting_user_input": False,
            "llm_meta": empty_llm_meta,
        }

    # Build message history
    messages = [Message(role="system", content=_build_calendar_system_prompt())]

    # Restore conversation context from previous turns
    manager = get_workflow_state_manager()
    saved = await manager.restore_workflow_state(session_id) if session_id else None

    if saved and saved.workflow_type == CALENDAR_WORKFLOW_TYPE:
        for turn in saved.conversation_context:
            messages.append(Message(role=turn["role"], content=turn["content"]))

    # Add current user input
    messages.append(Message(role="user", content=input_text))

    # Get calendar tool schemas (prefer MCP names, fall back to internal)
    from ...services.mcp_client import resolve_tools
    tool_names = resolve_tools([
        "create_event|create_calendar_event",
        "list_events|get_calendar",
        "find_free_slots",
    ])
    tools = tool_registry.get_tool_schemas_filtered(tool_names)

    # Run LLM with tools
    try:
        result = await execute_with_tools(
            llm=llm,
            messages=messages,
            tools_override=tools,
            max_tokens=200,
            temperature=0.7,
        )
    except Exception as e:
        logger.error("Calendar workflow LLM call failed: %s", e)
        if session_id:
            context_turns = [
                {"role": m.role, "content": m.content}
                for m in messages[1:]
            ]
            await manager.save_workflow_state(
                session_id=session_id,
                workflow_type=CALENDAR_WORKFLOW_TYPE,
                current_step="conversation",
                partial_state={},
                conversation_context=context_turns,
            )
        return {
            "response": "Sorry, something went wrong. Could you try again?",
            "awaiting_user_input": True,
            "total_ms": (time.perf_counter() - start_time) * 1000,
            "llm_meta": empty_llm_meta,
        }

    response = result.get("response", "")
    tools_executed = result.get("tools_executed", [])
    llm_meta = result.get("llm_meta") or empty_llm_meta

    # Only mutating tools end the workflow; read-only tools (get_calendar)
    # keep it alive for follow-ups like "add an event at that time"
    _terminal_tools = ("create_event", "create_calendar_event")
    workflow_done = any(t in tools_executed for t in _terminal_tools)

    if workflow_done:
        if session_id:
            await manager.clear_workflow_state(session_id)
        return {
            "response": response or "Done.",
            "awaiting_user_input": False,
            "total_ms": (time.perf_counter() - start_time) * 1000,
            "llm_meta": llm_meta,
        }

    if not response:
        response = "Sorry, could you repeat that?"

    # No tool called -- LLM is asking for more info, save context
    if session_id:
        context_turns = [
            {"role": m.role, "content": m.content}
            for m in messages[1:]
        ]
        context_turns.append({"role": "assistant", "content": response})

        await manager.save_workflow_state(
            session_id=session_id,
            workflow_type=CALENDAR_WORKFLOW_TYPE,
            current_step="conversation",
            partial_state={},
            conversation_context=context_turns,
        )

    return {
        "response": response,
        "awaiting_user_input": True,
        "total_ms": (time.perf_counter() - start_time) * 1000,
        "llm_meta": llm_meta,
    }
