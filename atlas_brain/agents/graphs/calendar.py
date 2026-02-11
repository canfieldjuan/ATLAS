"""
LLM-driven calendar workflow.

The LLM drives the conversation naturally using two tools:
  - create_calendar_event: Create a new event
  - get_calendar: Query upcoming events

Conversation history persists across voice turns via WorkflowStateManager.
"""

import logging
import re
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
        f"You can:\n"
        f"1. Create events -- use create_calendar_event with summary and start_time\n"
        f"2. Query events -- use get_calendar to see upcoming events\n\n"
        f"For create_calendar_event:\n"
        f"  - summary: event title (required)\n"
        f"  - start_time: natural language like 'tomorrow at 3pm' (required)\n"
        f"  - duration_minutes: defaults to 60\n"
        f"  - location: optional\n\n"
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
    llm = llm_registry.get_active()

    if llm is None:
        return {
            "response": "I can't manage the calendar right now.",
            "awaiting_user_input": False,
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

    # Get calendar tool schemas
    tool_names = ["create_calendar_event", "get_calendar"]
    tools = tool_registry.get_tool_schemas_filtered(tool_names)

    # Run LLM with tools
    result = await execute_with_tools(
        llm=llm,
        messages=messages,
        tools_override=tools,
        max_tokens=200,
        temperature=0.7,
    )

    response = result.get("response", "")
    tools_executed = result.get("tools_executed", [])

    # Strip <think> tags and stray tool XML
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"</?tool_call>", "", response)
    response = re.sub(r"<function=\w+>.*?</function>", "", response, flags=re.DOTALL)
    response = response.strip()

    # Calendar operations are typically single-turn
    any_tool_called = len(tools_executed) > 0

    if any_tool_called:
        # Tool executed -- workflow complete, clear state
        if session_id:
            await manager.clear_workflow_state(session_id)
        return {
            "response": response or "Done.",
            "awaiting_user_input": False,
            "total_ms": (time.perf_counter() - start_time) * 1000,
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
    }
