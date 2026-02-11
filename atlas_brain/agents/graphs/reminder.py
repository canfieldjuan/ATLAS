"""
LLM-driven reminder workflow.

The LLM drives the conversation naturally using three tools:
  - set_reminder: Create a new reminder
  - list_reminders: List upcoming reminders
  - complete_reminder: Mark a reminder as done

Conversation history persists across voice turns via WorkflowStateManager.
"""

import logging
import re
import time
from typing import Optional

from .workflow_state import get_workflow_state_manager
from ...services.protocols import Message

logger = logging.getLogger("atlas.agents.graphs.reminder")

# Workflow type identifier for state persistence
REMINDER_WORKFLOW_TYPE = "reminder"


def _build_reminder_system_prompt() -> str:
    """Build system prompt for reminder conversation."""
    from datetime import date
    today = date.today()
    today_str = today.strftime("%A, %B %d, %Y")

    return (
        f"You are a helpful reminder assistant. Today is {today_str}.\n\n"
        f"You can:\n"
        f"1. Create reminders -- use set_reminder with message and when\n"
        f"2. List reminders -- use list_reminders\n"
        f"3. Complete/dismiss reminders -- use complete_reminder\n\n"
        f"For set_reminder, the 'when' parameter accepts natural language like "
        f"'in 30 minutes', 'tomorrow at 5pm', 'next Tuesday morning'.\n"
        f"Optional 'repeat' parameter: 'daily', 'weekly', 'monthly'.\n\n"
        f"Be conversational and brief. This is a voice interface -- keep responses "
        f"under 2 sentences."
    )


async def run_reminder_workflow(
    input_text: str,
    session_id: Optional[str] = None,
) -> dict:
    """
    Run reminder as an LLM conversation with tools.

    The LLM decides whether to create, list, or complete reminders
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
            "response": "I can't manage reminders right now.",
            "awaiting_user_input": False,
        }

    # Build message history
    messages = [Message(role="system", content=_build_reminder_system_prompt())]

    # Restore conversation context from previous turns
    manager = get_workflow_state_manager()
    saved = await manager.restore_workflow_state(session_id) if session_id else None

    if saved and saved.workflow_type == REMINDER_WORKFLOW_TYPE:
        for turn in saved.conversation_context:
            messages.append(Message(role=turn["role"], content=turn["content"]))

    # Add current user input
    messages.append(Message(role="user", content=input_text))

    # Get reminder tool schemas
    tool_names = ["set_reminder", "list_reminders", "complete_reminder"]
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

    # Reminder operations are typically single-turn (tool was called and done)
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
            workflow_type=REMINDER_WORKFLOW_TYPE,
            current_step="conversation",
            partial_state={},
            conversation_context=context_turns,
        )

    return {
        "response": response,
        "awaiting_user_input": True,
        "total_ms": (time.perf_counter() - start_time) * 1000,
    }
