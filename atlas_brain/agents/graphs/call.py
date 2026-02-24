"""
LLM-driven call workflow.

Allows users to call a contact via voice: search CRM for the person,
disambiguate if multiple matches, confirm, then place the call via
Twilio. Uses Anthropic Haiku (via triage LLM) for reliable multi-step
tool calling -- same pattern as email_query.

Tools:
  - search_contacts: Find CRM contacts by name
  - make_call: Place a phone call via Twilio (terminal)

Conversation history persists across voice turns via WorkflowStateManager.
"""

import logging
import time
from typing import Optional

from .workflow_state import get_workflow_state_manager
from ...services.protocols import Message

logger = logging.getLogger("atlas.agents.graphs.call")

# Workflow type identifier for state persistence
CALL_WORKFLOW_TYPE = "call"


def _build_call_system_prompt(speaker_id: Optional[str] = None) -> str:
    """Build system prompt for call workflow."""
    speaker_ctx = (
        f"You are speaking with {speaker_id}, the business owner."
        if speaker_id else ""
    )

    return (
        f"You are a phone call assistant for Effingham Office Maids, a cleaning service company. "
        f"{speaker_ctx}\n\n"
        f"Your workflow:\n"
        f"1. Search CRM for the person using search_contacts\n"
        f"2. If multiple matches, read the names and phone numbers and ask which one\n"
        f"3. If one match, confirm the name and number before calling\n"
        f"4. If no matches, say you couldn't find them in contacts\n"
        f"5. Never call without explicit user confirmation\n\n"
        f"Be conversational and brief. This is a voice interface -- keep responses "
        f"under 2 sentences. When confirming, say the person's name and phone number."
    )


async def run_call_workflow(
    input_text: str,
    session_id: Optional[str] = None,
    speaker_id: Optional[str] = None,
    llm: Optional[object] = None,
) -> dict:
    """
    Run call workflow as an LLM conversation with tools.

    The LLM drives the conversation -- it searches CRM, disambiguates,
    confirms, and places the call. Conversation history persists across
    turns via WorkflowStateManager.

    Args:
        input_text: Natural language call request.
        session_id: Optional session ID for multi-turn persistence.
        speaker_id: Optional speaker identifier.
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
            "response": "I can't make calls right now.",
            "awaiting_user_input": False,
            "llm_meta": empty_llm_meta,
        }

    # Build message history
    messages = [Message(role="system", content=_build_call_system_prompt(speaker_id))]

    # Restore conversation context from previous turns
    manager = get_workflow_state_manager()
    saved = await manager.restore_workflow_state(session_id) if session_id else None

    if saved and saved.workflow_type == CALL_WORKFLOW_TYPE:
        for turn in saved.conversation_context:
            messages.append(Message(role=turn["role"], content=turn["content"]))

    # Add current user input
    messages.append(Message(role="user", content=input_text))

    # Get CRM + telephony tool schemas
    from ...services.mcp_client import resolve_tools
    tool_names = resolve_tools([
        "search_contacts",
        "make_call",
    ])
    tools = tool_registry.get_tool_schemas_filtered(tool_names)

    # Run LLM with tools (handles tool-call loop internally)
    try:
        result = await execute_with_tools(
            llm=llm,
            messages=messages,
            tools_override=tools,
            max_tokens=600,
            temperature=0.7,
        )
    except Exception as e:
        logger.error("Call workflow LLM call failed: %s", e)
        if session_id:
            context_turns = [
                {"role": m.role, "content": m.content}
                for m in messages[1:]
            ]
            await manager.save_workflow_state(
                session_id=session_id,
                workflow_type=CALL_WORKFLOW_TYPE,
                current_step="conversation",
                partial_state={"speaker_id": speaker_id},
                conversation_context=context_turns,
            )
        return {
            "response": "Sorry, something went wrong placing the call. Could you try again?",
            "awaiting_user_input": True,
            "total_ms": (time.perf_counter() - start_time) * 1000,
            "llm_meta": empty_llm_meta,
        }

    response = result.get("response", "")
    tools_executed = result.get("tools_executed", [])
    llm_meta = result.get("llm_meta") or empty_llm_meta

    # Terminal tool: make_call completes the workflow
    workflow_done = "make_call" in tools_executed

    if workflow_done:
        if session_id:
            await manager.clear_workflow_state(session_id)

        # Emit reasoning event for cross-domain intelligence
        try:
            from ...reasoning.producers import emit_if_enabled
            await emit_if_enabled(
                "call.completed", "call_workflow",
                {"session_id": session_id, "input_text": input_text,
                 "speaker_id": speaker_id},
            )
        except Exception:
            pass

        return {
            "response": response or "Call placed.",
            "awaiting_user_input": False,
            "total_ms": (time.perf_counter() - start_time) * 1000,
            "llm_meta": llm_meta,
        }

    if not response:
        response = "Sorry, could you repeat that?"

    # Workflow continues -- save conversation context for next turn
    if session_id:
        context_turns = [
            {"role": m.role, "content": m.content}
            for m in messages[1:]  # skip system prompt
        ]
        # Add assistant response
        context_turns.append({"role": "assistant", "content": response})

        await manager.save_workflow_state(
            session_id=session_id,
            workflow_type=CALL_WORKFLOW_TYPE,
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
