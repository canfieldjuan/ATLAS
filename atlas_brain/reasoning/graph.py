"""LangGraph-style state graph for the reasoning agent.

This is a manual state machine (no langgraph dependency required).
Nodes are async functions that transform ReasoningAgentState.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from .state import ReasoningAgentState

logger = logging.getLogger("atlas.reasoning.graph")

# Regex to strip markdown code fences from LLM output
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


async def _llm_generate(llm, prompt: str, system_prompt: str,
                         max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """Call LLM.chat() from async context and return response text."""
    from ..services.protocols import Message

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=prompt),
    ]
    # Prefer async if available; fall back to sync via thread
    if hasattr(llm, "chat_async"):
        return await llm.chat_async(
            messages=messages, max_tokens=max_tokens, temperature=temperature,
        )
    result = await asyncio.to_thread(
        llm.chat, messages=messages, max_tokens=max_tokens, temperature=temperature,
    )
    return result.get("response", "")


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Extract and parse JSON from an LLM response.

    Handles: raw JSON, markdown-fenced JSON, JSON embedded in prose.
    Raises JSONDecodeError if no valid JSON found.
    """
    text = text.strip()
    if not text:
        raise json.JSONDecodeError("Empty response", text, 0)

    # 1. Try raw parse first (ideal case)
    if text.startswith("{"):
        return json.loads(text)

    # 2. Try stripping markdown code fences
    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1).strip())

    # 3. Try finding first { ... last } in the text
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        return json.loads(text[first : last + 1])

    raise json.JSONDecodeError("No JSON object found in response", text, 0)


async def run_reasoning_graph(state: ReasoningAgentState) -> ReasoningAgentState:
    """Execute the full reasoning graph: triage -> context -> lock check ->
    reason -> plan -> execute -> synthesize -> notify.
    """
    # 1. Triage
    state = await _node_triage(state)
    if not state.get("needs_reasoning"):
        return state

    # 2. Aggregate context
    state = await _node_aggregate_context(state)

    # 3. Check entity lock
    state = await _node_check_lock(state)
    if state.get("queued"):
        return state

    # 4. Reason
    state = await _node_reason(state)

    # 5. Plan actions
    state = await _node_plan_actions(state)

    # 6. Execute actions
    state = await _node_execute_actions(state)

    # 7. Synthesize
    state = await _node_synthesize(state)

    # 8. Notify
    state = await _node_notify(state)

    return state


# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------


async def _node_triage(state: ReasoningAgentState) -> ReasoningAgentState:
    """Classify event priority and whether reasoning is needed."""
    from .prompts import TRIAGE_SYSTEM
    from ..services.llm_router import get_llm

    event_desc = (
        f"Event: {state.get('event_type')}\n"
        f"Source: {state.get('source')}\n"
        f"Entity: {state.get('entity_type')}/{state.get('entity_id')}\n"
        f"Payload: {json.dumps(state.get('payload', {}), default=str)[:2000]}"
    )

    llm = get_llm("email_triage")  # reuse triage LLM (Haiku)
    if not llm:
        # No triage LLM -- default to reasoning everything
        state["triage_priority"] = "medium"
        state["needs_reasoning"] = True
        state["triage_reasoning"] = "Triage LLM unavailable, defaulting to reason"
        return state

    try:
        from ..config import settings
        text = await _llm_generate(
            llm, event_desc, TRIAGE_SYSTEM,
            max_tokens=settings.reasoning.triage_max_tokens,
            temperature=0.1,
        )
        parsed = _parse_llm_json(text)
        state["triage_priority"] = parsed.get("priority", "medium")
        state["needs_reasoning"] = parsed.get("needs_reasoning", True)
        state["triage_reasoning"] = parsed.get("reasoning", "")
    except Exception:
        logger.warning("Triage failed, defaulting to reason", exc_info=True)
        state["triage_priority"] = "medium"
        state["needs_reasoning"] = True
        state["triage_reasoning"] = "Triage parse error, defaulting to reason"

    logger.info(
        "Triage: %s priority=%s needs_reasoning=%s",
        state.get("event_type"),
        state.get("triage_priority"),
        state.get("needs_reasoning"),
    )
    return state


async def _node_aggregate_context(
    state: ReasoningAgentState,
) -> ReasoningAgentState:
    """Pull cross-domain context for the entity."""
    from .context_aggregator import aggregate_context

    ctx = await aggregate_context(
        entity_type=state.get("entity_type"),
        entity_id=state.get("entity_id"),
        event_type=state.get("event_type", ""),
        payload=state.get("payload", {}),
    )

    state["crm_context"] = ctx.get("crm")
    state["email_history"] = ctx.get("emails", [])
    state["voice_turns"] = ctx.get("voice", [])
    state["calendar_events"] = ctx.get("calendar", [])
    state["sms_messages"] = ctx.get("sms", [])
    state["graph_facts"] = ctx.get("graph_facts", [])
    state["recent_events"] = ctx.get("recent_events", [])

    return state


async def _node_check_lock(state: ReasoningAgentState) -> ReasoningAgentState:
    """Check if the entity is locked by AtlasAgent."""
    from .entity_locks import EntityLockManager

    entity_type = state.get("entity_type")
    entity_id = state.get("entity_id")

    if not entity_type or not entity_id:
        state["entity_locked"] = False
        state["queued"] = False
        return state

    mgr = EntityLockManager()
    locked, holder = await mgr.is_locked(entity_type, entity_id)
    state["entity_locked"] = locked
    state["lock_holder"] = holder

    if locked:
        # Queue decision for later drain
        event_id = state.get("event_id")
        if event_id:
            from uuid import UUID
            await mgr.queue_for_entity(
                UUID(event_id), entity_type, entity_id
            )
        state["queued"] = True
        logger.info(
            "Entity %s/%s locked by %s -- decision queued",
            entity_type, entity_id, holder,
        )
    else:
        state["queued"] = False

    return state


async def _node_reason(state: ReasoningAgentState) -> ReasoningAgentState:
    """Deep reasoning with full context via Claude Sonnet."""
    from .prompts import REASONING_SYSTEM
    from ..services.llm_router import get_llm
    from ..config import settings

    # Build context prompt
    sections = [
        f"## Event\nType: {state.get('event_type')}\n"
        f"Source: {state.get('source')}\n"
        f"Payload: {json.dumps(state.get('payload', {}), default=str)[:3000]}",
    ]

    if state.get("crm_context"):
        sections.append(
            f"## CRM Context\n{json.dumps(state['crm_context'], default=str)[:2000]}"
        )
    if state.get("email_history"):
        sections.append(
            f"## Recent Emails ({len(state['email_history'])})\n"
            + json.dumps(state["email_history"][:5], default=str)[:2000]
        )
    if state.get("voice_turns"):
        sections.append(
            f"## Recent Voice Turns ({len(state['voice_turns'])})\n"
            + json.dumps(state["voice_turns"][:5], default=str)[:1500]
        )
    if state.get("calendar_events"):
        sections.append(
            f"## Calendar ({len(state['calendar_events'])})\n"
            + json.dumps(state["calendar_events"][:5], default=str)[:1000]
        )
    if state.get("sms_messages"):
        sections.append(
            f"## SMS ({len(state['sms_messages'])})\n"
            + json.dumps(state["sms_messages"][:5], default=str)[:1000]
        )
    if state.get("recent_events"):
        sections.append(
            f"## Recent Events ({len(state['recent_events'])})\n"
            + json.dumps(state["recent_events"][:5], default=str)[:1500]
        )

    prompt = "\n\n".join(sections)

    llm = get_llm("reasoning")
    if not llm:
        state["reasoning_output"] = ""
        state["connections_found"] = []
        state["recommended_actions"] = []
        state["rationale"] = "Reasoning LLM unavailable"
        return state

    try:
        text = await _llm_generate(
            llm, prompt, REASONING_SYSTEM,
            max_tokens=settings.reasoning.max_tokens,
            temperature=settings.reasoning.temperature,
        )
        state["reasoning_output"] = text

        parsed = _parse_llm_json(text)
        state["connections_found"] = parsed.get("connections", [])
        state["recommended_actions"] = parsed.get("actions", [])
        state["rationale"] = parsed.get("rationale", "")
        state["should_notify"] = parsed.get("should_notify", False)
    except json.JSONDecodeError:
        state["connections_found"] = []
        state["recommended_actions"] = []
        state["rationale"] = state.get("reasoning_output", "")
        state["should_notify"] = True
    except Exception:
        logger.error("Reasoning node failed", exc_info=True)
        state["reasoning_output"] = ""
        state["connections_found"] = []
        state["recommended_actions"] = []
        state["rationale"] = "Reasoning failed"

    return state


async def _node_plan_actions(state: ReasoningAgentState) -> ReasoningAgentState:
    """Convert reasoning recommendations into executable action plan.

    Safety: never auto-send email (only draft), never delete,
    never modify CRM without logging.
    """
    SAFE_ACTIONS = {
        "generate_draft", "show_slots", "log_interaction",
        "create_reminder", "send_notification",
    }

    planned = []
    for action in state.get("recommended_actions", []):
        tool = action.get("tool", "")
        if tool not in SAFE_ACTIONS:
            logger.warning("Skipping unsafe action: %s", tool)
            continue
        if action.get("confidence", 0) < 0.5:
            logger.debug("Skipping low-confidence action: %s (%.2f)", tool, action.get("confidence", 0))
            continue
        planned.append(action)

    state["planned_actions"] = planned
    return state


async def _node_execute_actions(
    state: ReasoningAgentState,
) -> ReasoningAgentState:
    """Execute planned actions via existing handlers."""
    results = []

    for action in state.get("planned_actions", []):
        tool = action.get("tool", "")
        params = action.get("params", {})

        try:
            result = await _execute_single_action(tool, params, state)
            results.append({"tool": tool, "success": True, "result": str(result)[:500]})
        except Exception as exc:
            logger.warning("Action %s failed: %s", tool, exc)
            results.append({"tool": tool, "success": False, "error": str(exc)[:500]})

    state["action_results"] = results
    return state


async def _execute_single_action(
    tool: str, params: dict[str, Any], state: ReasoningAgentState
) -> Any:
    """Dispatch a single action to the appropriate handler."""
    if tool == "generate_draft":
        from ..api.email_drafts import generate_draft
        imap_uid = params.get("imap_uid") or params.get("gmail_message_id")
        if imap_uid:
            return await generate_draft(imap_uid)
        return "Missing imap_uid"

    if tool == "show_slots":
        # Delegate to calendar free slots
        return {"action": "show_slots", "status": "queued"}

    if tool == "log_interaction":
        from ..services.crm_provider import get_crm_provider
        crm = get_crm_provider()
        return await crm.log_interaction(
            contact_id=params.get("contact_id", state.get("entity_id", "")),
            interaction_type=params.get("interaction_type", "reasoning_agent"),
            summary=params.get("summary", "Reasoning agent action"),
        )

    if tool == "create_reminder":
        return {"action": "create_reminder", "status": "queued", "params": params}

    if tool == "send_notification":
        message = params.get("message", state.get("rationale", "Reasoning agent alert"))
        await _send_ntfy(message)
        return {"action": "send_notification", "sent": True}

    return {"error": f"Unknown tool: {tool}"}


async def _node_synthesize(state: ReasoningAgentState) -> ReasoningAgentState:
    """Generate a human-readable summary for notification."""
    if not state.get("should_notify"):
        state["summary"] = ""
        return state

    from .prompts import SYNTHESIS_SYSTEM
    from ..services.llm_router import get_llm

    context = (
        f"Event: {state.get('event_type')}\n"
        f"Actions taken: {json.dumps(state.get('action_results', []), default=str)[:1000]}\n"
        f"Rationale: {state.get('rationale', '')[:500]}"
    )

    llm = get_llm("email_triage")  # Haiku for cheap synthesis
    if not llm:
        state["summary"] = state.get("rationale", "Reasoning agent completed.")
        return state

    try:
        text = await _llm_generate(
            llm, context, SYNTHESIS_SYSTEM,
            max_tokens=256, temperature=0.3,
        )
        state["summary"] = text.strip()
    except Exception:
        state["summary"] = state.get("rationale", "Reasoning agent completed.")

    return state


async def _node_notify(state: ReasoningAgentState) -> ReasoningAgentState:
    """Send push notification if warranted."""
    if not state.get("should_notify") or not state.get("summary"):
        state["notification_sent"] = False
        return state

    try:
        await _send_ntfy(state["summary"])
        state["notification_sent"] = True
    except Exception:
        logger.warning("Notification failed", exc_info=True)
        state["notification_sent"] = False

    return state


async def _send_ntfy(message: str) -> None:
    """Send a push notification via ntfy."""
    from ..config import settings

    if not settings.alerts.ntfy_enabled:
        return

    import httpx

    url = f"{settings.alerts.ntfy_url}/{settings.alerts.ntfy_topic}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            url,
            content=message.encode("utf-8"),
            headers={
                "Title": "Atlas Reasoning Agent",
                "Priority": "default",
                "Tags": "brain",
            },
        )
