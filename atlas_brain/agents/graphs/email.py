"""
LLM-driven email workflow.

The LLM drives the conversation naturally using email tools:
  - send_email: Send a generic email
  - send_estimate_email: Send templated estimate confirmation
  - send_proposal_email: Send templated proposal with auto-PDF
  - query_email_history: Query sent email history
  - lookup_customer: Find customer in CRM (already registered)

Conversation history persists across voice turns via WorkflowStateManager.
"""

import logging
import time
from typing import Optional

from .workflow_state import get_workflow_state_manager
from ...services.protocols import Message

logger = logging.getLogger("atlas.agents.graphs.email")

# Workflow type identifier for state persistence
EMAIL_WORKFLOW_TYPE = "email"


def _build_email_system_prompt(speaker_id: Optional[str] = None) -> str:
    """Build system prompt for email conversation with skills injection."""
    from datetime import date
    from ...skills import get_skill_registry

    today = date.today()
    today_str = today.strftime("%A, %B %d, %Y")

    speaker_ctx = (
        f"You are speaking with {speaker_id}, the business owner."
        if speaker_id else ""
    )

    # Load all email-domain skills
    skills_text = ""
    registry = get_skill_registry()
    email_skills = registry.get_by_domain("email")
    if email_skills:
        skills_parts = []
        for skill in email_skills:
            skills_parts.append(f"### {skill.name}\n{skill.content}")
        skills_text = "\n\n---\n\n".join(skills_parts)

    prompt = (
        f"You are an email assistant for Effingham Office Maids, a cleaning service company. "
        f"Today is {today_str}. {speaker_ctx}\n\n"
        f"You help send emails, estimate confirmations, proposals, and check email history.\n\n"
        f"Your workflow:\n"
        f"1. Collect information conversationally -- don't list all required fields\n"
        f"2. Look up the customer to auto-fill client details when you have a name or phone\n"
        f"3. ALWAYS describe the email before sending (summarize to, subject, key points)\n"
        f"4. Only call a send tool AFTER the user confirms ('yes', 'send it', 'looks good')\n"
        f"5. After sending estimates or proposals, offer to set a follow-up reminder\n\n"
        f"You have tools for: sending generic emails, estimate confirmations, "
        f"cleaning proposals, and querying sent email history. "
        f"Match the user's request to the appropriate tool from your tool list.\n\n"
        f"Be conversational and brief. This is a voice interface -- keep responses "
        f"under 2-3 sentences. Don't list all parameters needed; ask naturally."
    )

    if skills_text:
        prompt += (
            f"\n\n## Email Composition Guidelines\n\n"
            f"When composing emails, follow these domain-specific rules:\n\n"
            f"{skills_text}"
        )

    return prompt


async def run_email_workflow(
    input_text: str,
    session_id: Optional[str] = None,
    speaker_id: Optional[str] = None,
    llm: Optional[object] = None,
) -> dict:
    """
    Run email as an LLM conversation with tools.

    The LLM drives the conversation. It decides what info to collect,
    when to send, and handles drafts conversationally. Conversation
    history persists across turns via WorkflowStateManager.

    Args:
        input_text: Natural language email request.
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
            "response": "I can't process emails right now.",
            "awaiting_user_input": False,
            "llm_meta": empty_llm_meta,
        }

    # Build message history
    messages = [Message(role="system", content=_build_email_system_prompt(speaker_id))]

    # Restore conversation context from previous turns
    manager = get_workflow_state_manager()
    saved = await manager.restore_workflow_state(session_id) if session_id else None

    if saved and saved.workflow_type == EMAIL_WORKFLOW_TYPE:
        for turn in saved.conversation_context:
            messages.append(Message(role=turn["role"], content=turn["content"]))

    # Add current user input
    messages.append(Message(role="user", content=input_text))

    # Get email + CRM tool schemas (prefer MCP names, fall back to internal)
    from ...services.mcp_client import resolve_tools
    tool_names = resolve_tools([
        "send_email",
        "send_estimate|send_estimate_email",
        "send_proposal|send_proposal_email",
        "list_sent_history|query_email_history",
        "lookup_customer",
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
        logger.error("Email workflow LLM call failed: %s", e)
        # Save current conversation so user can retry without losing context
        if session_id:
            context_turns = [
                {"role": m.role, "content": m.content}
                for m in messages[1:]
            ]
            await manager.save_workflow_state(
                session_id=session_id,
                workflow_type=EMAIL_WORKFLOW_TYPE,
                current_step="conversation",
                partial_state={"speaker_id": speaker_id},
                conversation_context=context_turns,
            )
        return {
            "response": "Sorry, something went wrong with the email. Could you try again?",
            "awaiting_user_input": True,
            "total_ms": (time.perf_counter() - start_time) * 1000,
            "llm_meta": empty_llm_meta,
        }

    response = result.get("response", "")
    tools_executed = result.get("tools_executed", [])
    llm_meta = result.get("llm_meta") or empty_llm_meta

    # Terminal tools: any send action or history query completes the workflow
    # Include both MCP and internal names for compatibility
    _terminal_tools = (
        "send_email", "send_estimate", "send_estimate_email",
        "send_proposal", "send_proposal_email",
        "list_sent_history", "query_email_history",
    )
    workflow_done = any(t in tools_executed for t in _terminal_tools)

    if workflow_done:
        # Email action completed -- clear state
        if session_id:
            await manager.clear_workflow_state(session_id)
        return {
            "response": response or "Email operation complete.",
            "awaiting_user_input": False,
            "total_ms": (time.perf_counter() - start_time) * 1000,
            "llm_meta": llm_meta,
        }

    if not response:
        # LLM returned empty without completing -- keep workflow alive
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
            workflow_type=EMAIL_WORKFLOW_TYPE,
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
