"""
LLM-driven booking workflow.

The LLM drives the conversation naturally using three tools:
  - lookup_customer: Find customer in CRM
  - check_availability: Check calendar slots
  - book_appointment: Create the booking

Conversation history persists across voice turns via WorkflowStateManager.
"""

import logging
import time
from typing import Optional

from .workflow_state import get_workflow_state_manager
from ...services.protocols import Message

logger = logging.getLogger("atlas.agents.graphs.booking")

# Workflow type identifier for state persistence
BOOKING_WORKFLOW_TYPE = "booking"


def _build_booking_system_prompt(speaker_id: Optional[str] = None) -> str:
    """Build system prompt for booking conversation."""
    from datetime import date
    today = date.today()
    today_str = today.strftime("%A, %B %d, %Y")

    speaker_ctx = (
        f"You are speaking with {speaker_id}, the business owner."
        if speaker_id else ""
    )

    return (
        f"You are a friendly scheduling assistant for a cleaning business. "
        f"Today is {today_str}. {speaker_ctx}\n\n"
        f"The business owner is booking appointments for CUSTOMERS (not themselves).\n\n"
        f"Your workflow:\n"
        f"1. If you have the customer's name or phone, use lookup_customer to find them\n"
        f"2. For booking: use check_availability to find open slots, then book_appointment\n"
        f"3. For cancellation: use lookup_customer to find the appointment, then cancel_appointment\n"
        f"4. For rescheduling: use lookup_customer, then reschedule_appointment with new date/time\n\n"
        f"Required for booking: customer_name, customer_phone, date, time.\n"
        f"Optional: customer_email, address, service_type.\n\n"
        f"Be conversational and brief. This is a voice interface -- keep responses "
        f"under 2-3 sentences. Don't list all parameters needed; ask naturally."
    )


async def run_booking_workflow(
    input_text: str,
    session_id: Optional[str] = None,
    speaker_id: Optional[str] = None,
    llm: Optional[object] = None,
) -> dict:
    """
    Run booking as an LLM conversation with tools.

    The LLM drives the conversation. It decides what info to collect,
    when to check availability, and when to book. Conversation history
    persists across turns via WorkflowStateManager.

    Args:
        llm: Optional LLM instance (from llm_router). Falls back to local registry.
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
    if llm is None:
        llm = llm_registry.get_active()

    if llm is None:
        return {
            "response": "I can't process bookings right now.",
            "awaiting_user_input": False,
            "llm_meta": empty_llm_meta,
        }

    # Build message history
    messages = [Message(role="system", content=_build_booking_system_prompt(speaker_id))]

    # Restore conversation context from previous turns
    manager = get_workflow_state_manager()
    saved = await manager.restore_workflow_state(session_id) if session_id else None

    if saved and saved.workflow_type == BOOKING_WORKFLOW_TYPE:
        for turn in saved.conversation_context:
            messages.append(Message(role=turn["role"], content=turn["content"]))

    # Add current user input
    messages.append(Message(role="user", content=input_text))

    # Get scheduling tool schemas
    tool_names = [
        "lookup_customer", "check_availability", "book_appointment",
        "cancel_appointment", "reschedule_appointment",
    ]
    tools = tool_registry.get_tool_schemas_filtered(tool_names)

    # Run LLM with tools (handles tool-call loop internally)
    try:
        result = await execute_with_tools(
            llm=llm,
            messages=messages,
            tools_override=tools,
            max_tokens=200,
            temperature=0.7,
        )
    except Exception as e:
        logger.error("Booking workflow LLM call failed: %s", e)
        # Save current conversation so user can retry without losing context
        if session_id:
            context_turns = [
                {"role": m.role, "content": m.content}
                for m in messages[1:]
            ]
            await manager.save_workflow_state(
                session_id=session_id,
                workflow_type=BOOKING_WORKFLOW_TYPE,
                current_step="conversation",
                partial_state={"speaker_id": speaker_id},
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

    # Determine if workflow is complete (booking, cancellation, or reschedule)
    _terminal_tools = ("book_appointment", "cancel_appointment", "reschedule_appointment")
    booking_done = any(t in tools_executed for t in _terminal_tools)

    if booking_done:
        # Scheduling action completed -- clear state
        if session_id:
            await manager.clear_workflow_state(session_id)
        return {
            "response": response or "Booking complete.",
            "awaiting_user_input": False,
            "total_ms": (time.perf_counter() - start_time) * 1000,
            "llm_meta": llm_meta,
        }

    if not response:
        # LLM returned empty without booking -- keep workflow alive
        response = "Sorry, could you repeat that?"

    # Workflow continues -- save conversation context for next turn
    if session_id:
        # Build context: all messages after system prompt
        context_turns = [
            {"role": m.role, "content": m.content}
            for m in messages[1:]  # skip system prompt
        ]
        # Add assistant response
        context_turns.append({"role": "assistant", "content": response})

        await manager.save_workflow_state(
            session_id=session_id,
            workflow_type=BOOKING_WORKFLOW_TYPE,
            current_step="conversation",
            partial_state={"speaker_id": speaker_id},
            conversation_context=context_turns,
        )

    return {
        "response": response,
        "awaiting_user_input": True,
        "total_ms": (time.perf_counter() - start_time) * 1000,
        "llm_meta": llm_meta,
    }
