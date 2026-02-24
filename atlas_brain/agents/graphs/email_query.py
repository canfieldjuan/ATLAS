"""
LLM-driven email query workflow.

Allows users to query their inbox via voice: read emails, check tone,
summarize threads, and optionally reply. Uses Anthropic Haiku (via
triage LLM) for reliable multi-step tool calling instead of local
qwen3:14b.

Tools (read-focused):
  - list_inbox: List recent inbox messages
  - get_message: Get full message content by UID
  - search_inbox: Search inbox by query string
  - get_thread: Get full email thread
  - search_contacts: Find CRM contacts
  - get_customer_context: Get CRM context for a contact
  - send_email: Reply to an email (terminal)
  - send_estimate / send_proposal: Send templated replies (terminal)

Conversation history persists across voice turns via WorkflowStateManager.
"""

import logging
import time
from typing import Optional

from .workflow_state import get_workflow_state_manager
from ...services.protocols import Message

logger = logging.getLogger("atlas.agents.graphs.email_query")

# Workflow type identifier for state persistence
EMAIL_QUERY_WORKFLOW_TYPE = "email_query"


def _build_email_query_system_prompt(speaker_id: Optional[str] = None) -> str:
    """Build system prompt for email query conversation with skills injection."""
    from datetime import date
    from ...skills import get_skill_registry

    today = date.today()
    today_str = today.strftime("%A, %B %d, %Y")

    speaker_ctx = (
        f"You are speaking with {speaker_id}, the business owner."
        if speaker_id else ""
    )

    # Load email-domain skills (reply guidelines, tone matching, etc.)
    skills_text = ""
    registry = get_skill_registry()
    # Load query-specific skill first, then general email skills
    query_skill = registry.get("email/query")
    if query_skill:
        skills_text = f"### {query_skill.name}\n{query_skill.content}"

    reply_skill = registry.get("email/reply")
    if reply_skill:
        if skills_text:
            skills_text += "\n\n---\n\n"
        skills_text += f"### {reply_skill.name}\n{reply_skill.content}"

    prompt = (
        f"You are an email query assistant for Effingham Office Maids, a cleaning service company. "
        f"Today is {today_str}. {speaker_ctx}\n\n"
        f"You help the user check their inbox, read emails, analyze tone, "
        f"summarize threads, and optionally reply to messages.\n\n"
        f"Your workflow:\n"
        f"1. When the user asks about emails, search or list the inbox to find relevant messages\n"
        f"2. Use get_message or get_thread for full content when needed\n"
        f"3. Summarize content and tone naturally for voice -- keep it brief\n"
        f"4. If the user wants to reply, draft it conversationally and confirm before sending\n"
        f"5. Look up the customer in CRM for context when relevant\n\n"
        f"Be conversational and brief. This is a voice interface -- keep responses "
        f"under 2-3 sentences. Summarize key points, don't read emails verbatim."
    )

    if skills_text:
        prompt += (
            f"\n\n## Email Guidelines\n\n"
            f"{skills_text}"
        )

    return prompt


async def run_email_query_workflow(
    input_text: str,
    session_id: Optional[str] = None,
    speaker_id: Optional[str] = None,
    llm: Optional[object] = None,
) -> dict:
    """
    Run email query as an LLM conversation with tools.

    The LLM drives the conversation -- it searches the inbox, reads
    messages, analyzes tone, and can reply if asked. Conversation
    history persists across turns via WorkflowStateManager.

    Args:
        input_text: Natural language email query.
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
            "response": "I can't check emails right now.",
            "awaiting_user_input": False,
            "llm_meta": empty_llm_meta,
        }

    # Build message history
    messages = [Message(role="system", content=_build_email_query_system_prompt(speaker_id))]

    # Restore conversation context from previous turns
    manager = get_workflow_state_manager()
    saved = await manager.restore_workflow_state(session_id) if session_id else None

    if saved and saved.workflow_type == EMAIL_QUERY_WORKFLOW_TYPE:
        for turn in saved.conversation_context:
            messages.append(Message(role=turn["role"], content=turn["content"]))

    # Add current user input
    messages.append(Message(role="user", content=input_text))

    # Get email read + CRM + reply tool schemas
    from ...services.mcp_client import resolve_tools
    tool_names = resolve_tools([
        "list_inbox",
        "get_message",
        "search_inbox",
        "get_thread",
        "search_contacts",
        "get_customer_context",
        "send_email",
        "send_estimate|send_estimate_email",
        "send_proposal|send_proposal_email",
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
        logger.error("Email query workflow LLM call failed: %s", e)
        # Save current conversation so user can retry without losing context
        if session_id:
            context_turns = [
                {"role": m.role, "content": m.content}
                for m in messages[1:]
            ]
            await manager.save_workflow_state(
                session_id=session_id,
                workflow_type=EMAIL_QUERY_WORKFLOW_TYPE,
                current_step="conversation",
                partial_state={"speaker_id": speaker_id},
                conversation_context=context_turns,
            )
        return {
            "response": "Sorry, something went wrong checking your email. Could you try again?",
            "awaiting_user_input": True,
            "total_ms": (time.perf_counter() - start_time) * 1000,
            "llm_meta": empty_llm_meta,
        }

    response = result.get("response", "")
    tools_executed = result.get("tools_executed", [])
    llm_meta = result.get("llm_meta") or empty_llm_meta

    # Terminal tools: sending actions complete the workflow
    _terminal_tools = (
        "send_email", "send_estimate", "send_estimate_email",
        "send_proposal", "send_proposal_email",
    )
    workflow_done = any(t in tools_executed for t in _terminal_tools)

    if workflow_done:
        # Send action completed -- clear state
        if session_id:
            await manager.clear_workflow_state(session_id)
        return {
            "response": response or "Email sent.",
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
            workflow_type=EMAIL_QUERY_WORKFLOW_TYPE,
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
