"""LangGraph-style state graph for the reasoning agent.

This is a manual state machine (no langgraph dependency required).
Nodes are async functions that transform ReasoningAgentState.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from uuid import UUID
from typing import Any

from .state import ReasoningAgentState

logger = logging.getLogger("atlas.reasoning.graph")

# Regex to strip markdown code fences from LLM output
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_SUMMARY_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_SUMMARY_META_LINE_RE = re.compile(
    r"^\s*(summarizing|refining|drafting|thinking|updating|notifying)\b",
    re.IGNORECASE,
)
_SUMMARY_FIRST_PERSON_RE = re.compile(
    r"\b(i|i'm|i am|i need|i should|we should)\b",
    re.IGNORECASE,
)
_SUMMARY_META_CONTENT_RE = re.compile(
    r"\b(the user wants|this event is about|the event is about|push notification|summary of|reasoning agent)\b",
    re.IGNORECASE,
)
_SUMMARY_GENERIC_OPEN_RE = re.compile(
    r"^\s*(assessing|reviewing|monitoring|evaluating|considering|tracking)\b",
    re.IGNORECASE,
)
_MARKDOWN_RE = re.compile(r"[*_`#>]+")
_COMMON_SHORT_FINAL_WORDS = frozenset({
    "and", "for", "the", "you", "aws", "gcp", "crm", "b2b",
})


def _resolve_graph_llm(workload: str):
    """Resolve reasoning-graph LLMs through the pipeline router."""
    from ..config import settings
    from ..pipelines.llm import get_pipeline_llm

    return get_pipeline_llm(
        workload=workload,
        auto_activate_ollama=False,
        openrouter_model=settings.reasoning.graph_openrouter_model or None,
    )


def _clean_summary_text(text: str) -> str:
    """Normalize graph synthesis output into plain notification text."""
    cleaned = _JSON_FENCE_RE.sub(r"\1", text or "")
    cleaned = _MARKDOWN_RE.sub("", cleaned)
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _build_notification_fallback(state: ReasoningAgentState) -> str:
    """Build a deterministic fallback summary from actions and rationale."""
    results = state.get("action_results", [])
    successful = [
        r.get("tool", "").replace("_", " ")
        for r in results
        if r.get("success") and r.get("tool")
    ]
    action_phrase = ""
    if successful:
        action_phrase = " Actions completed: " + ", ".join(successful[:3]) + "."
    connections = state.get("connections_found", []) or []
    connection_phrase = ""
    if connections:
        top_connection = re.sub(r"\s+", " ", str(connections[0])).strip()
        if top_connection:
            connection_phrase = top_connection[:220].rstrip(". ") + "."
    rationale = (state.get("rationale", "") or "").strip()
    rationale = re.sub(r"\s+", " ", rationale)
    if connection_phrase:
        return (connection_phrase + action_phrase).strip()
    if rationale:
        return (rationale[:220].rstrip(". ") + "." + action_phrase).strip()
    if action_phrase:
        return ("Atlas completed follow-up actions." + action_phrase).strip()
    return "Atlas completed reasoning and flagged this event for review."


def _has_suspicious_trailing_fragment(text: str) -> bool:
    """Detect likely truncated sentence endings like 'bef.'."""
    match = re.search(r"\b([A-Za-z]{1,3})[.!?]?$", text.strip())
    if not match:
        return False
    token = match.group(1)
    if token.lower() in _COMMON_SHORT_FINAL_WORDS:
        return False
    return token.islower()


def _sanitize_notification_summary(text: str, state: ReasoningAgentState) -> str:
    """Remove meta narration and keep a short owner-facing summary."""
    cleaned = _clean_summary_text(text)
    pieces: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip(' "\'')
        if not stripped:
            continue
        for sentence in _SUMMARY_SENTENCE_RE.split(stripped):
            if sentence.strip():
                pieces.append(sentence.strip())
    kept: list[str] = []
    for piece in pieces:
        candidate = piece.strip(' "\'')
        if not candidate:
            continue
        if _SUMMARY_META_LINE_RE.match(candidate):
            continue
        if _SUMMARY_FIRST_PERSON_RE.search(candidate):
            continue
        if _SUMMARY_META_CONTENT_RE.search(candidate):
            continue
        if _SUMMARY_GENERIC_OPEN_RE.match(candidate):
            continue
        if _has_suspicious_trailing_fragment(candidate):
            continue
        kept.append(candidate)
        if len(kept) >= 2:
            break
    if not kept:
        return _build_notification_fallback(state)
    summary = " ".join(s.rstrip(".!?") + "." for s in kept)
    return summary[:320].rstrip()


def _valid_uuid(value: Any) -> str | None:
    """Return canonical UUID string when value is a valid UUID-like input."""
    if not value:
        return None
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        return None


async def _llm_generate(llm, prompt: str, system_prompt: str,
                         max_tokens: int = 1024, temperature: float = 0.3,
                         timeout: float = 120.0,
                         json_mode: bool = False) -> dict[str, Any]:
    """Call LLM.chat() from async context and return response with usage.

    Args:
        timeout: Maximum seconds to wait for the LLM response (default 120s).
                 Raises asyncio.TimeoutError if exceeded.
        json_mode: When True, pass response_format={"type":"json_object"} to
                   force structured JSON output (needed for reasoning models
                   like o4-mini that may return prose otherwise).

    Returns:
        Dict with 'response' (str) and 'usage' (dict with input_tokens, output_tokens)
    """
    from ..services.protocols import Message

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=prompt),
    ]
    kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    # Always use sync chat() via thread -- chat_async() returns only str,
    # losing the usage dict needed for token tracking.
    result = await asyncio.wait_for(
        asyncio.to_thread(llm.chat, **kwargs),
        timeout=timeout,
    )
    return {
        "response": result.get("response", ""),
        "usage": result.get("usage", {}),
    }


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
    # Initialize token tracking
    state["total_input_tokens"] = 0
    state["total_output_tokens"] = 0

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
    from .graph_prompts import TRIAGE_SYSTEM
    from ..config import settings

    event_desc = (
        f"Event: {state.get('event_type')}\n"
        f"Source: {state.get('source')}\n"
        f"Entity: {state.get('entity_type')}/{state.get('entity_id')}\n"
        f"Payload: {json.dumps(state.get('payload', {}), default=str)[:2000]}"
    )

    llm = _resolve_graph_llm(settings.reasoning.graph_triage_workload)
    if not llm:
        # No triage LLM -- default to reasoning everything
        state["triage_priority"] = "medium"
        state["needs_reasoning"] = True
        state["triage_reasoning"] = "Triage LLM unavailable, defaulting to reason"
        return state

    try:
        result = await _llm_generate(
            llm, event_desc, TRIAGE_SYSTEM,
            max_tokens=settings.reasoning.triage_max_tokens,
            temperature=0.1,
            json_mode=True,
        )
        text = result["response"]
        usage = result.get("usage", {})
        state["total_input_tokens"] = state.get("total_input_tokens", 0) + usage.get("input_tokens", 0)
        state["total_output_tokens"] = state.get("total_output_tokens", 0) + usage.get("output_tokens", 0)

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
    state["market_context"] = ctx.get("market_data", [])
    state["news_context"] = ctx.get("recent_news", [])
    state["b2b_churn"] = ctx.get("b2b_churn", {})

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
    """Deep reasoning with full context via pipeline-configured LLM routing."""
    from .graph_prompts import REASONING_SYSTEM
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
    if state.get("market_context"):
        sections.append(
            f"## Market Data ({len(state['market_context'])})\n"
            + json.dumps(state["market_context"][:10], default=str)[:2000]
        )
    if state.get("news_context"):
        sections.append(
            f"## Recent News ({len(state['news_context'])})\n"
            + json.dumps(state["news_context"][:10], default=str)[:2000]
        )
    if state.get("b2b_churn"):
        sections.append(
            f"## B2B Churn Intelligence\n"
            + json.dumps(state["b2b_churn"], default=str)[:3000]
        )

    prompt = "\n\n".join(sections)

    llm = _resolve_graph_llm(settings.reasoning.graph_reasoning_workload)
    if not llm:
        state["reasoning_output"] = ""
        state["connections_found"] = []
        state["recommended_actions"] = []
        state["rationale"] = "Reasoning LLM unavailable"
        return state

    try:
        result = await _llm_generate(
            llm, prompt, REASONING_SYSTEM,
            max_tokens=settings.reasoning.max_tokens,
            temperature=settings.reasoning.temperature,
            json_mode=True,
        )
        text = result["response"]
        usage = result.get("usage", {})
        state["total_input_tokens"] = state.get("total_input_tokens", 0) + usage.get("input_tokens", 0)
        state["total_output_tokens"] = state.get("total_output_tokens", 0) + usage.get("output_tokens", 0)

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
            logger.warning("Action %s failed", tool, exc_info=True)
            results.append({"tool": tool, "success": False, "error": type(exc).__name__})

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
        contact_id = _valid_uuid(params.get("contact_id"))
        if not contact_id and state.get("entity_type") == "contact":
            contact_id = _valid_uuid(state.get("entity_id", ""))
        if not contact_id:
            return {
                "action": "log_interaction",
                "status": "skipped",
                "reason": "no valid contact UUID available",
            }
        return await crm.log_interaction(
            contact_id=contact_id,
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

    from .graph_prompts import SYNTHESIS_SYSTEM
    from ..config import settings

    context = (
        f"Event: {state.get('event_type')}\n"
        f"Actions taken: {json.dumps(state.get('action_results', []), default=str)[:1000]}\n"
        f"Rationale: {state.get('rationale', '')[:500]}"
    )

    llm = _resolve_graph_llm(settings.reasoning.graph_synthesis_workload)
    if not llm:
        state["summary"] = _build_notification_fallback(state)
        return state

    try:
        result = await _llm_generate(
            llm, context, SYNTHESIS_SYSTEM,
            max_tokens=256, temperature=0.3,
        )
        text = result["response"]
        usage = result.get("usage", {})
        state["total_input_tokens"] = state.get("total_input_tokens", 0) + usage.get("input_tokens", 0)
        state["total_output_tokens"] = state.get("total_output_tokens", 0) + usage.get("output_tokens", 0)

        state["summary"] = _sanitize_notification_summary(text, state)
    except Exception:
        state["summary"] = _build_notification_fallback(state)

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
