"""LLM-driven graph nodes promoted into core.

PR-C4e2 (this module) is the second sub-slice of the audit's PR 6
graph-engine extraction. ``node_triage`` and ``node_synthesize`` are
fully self-contained -- they read only fields declared on core's
:class:`extracted_reasoning_core.state.ReasoningAgentState` and call
the LLM through the :class:`extracted_reasoning_core.ports.LLMClient`
Protocol. Atlas's ``_node_triage`` / ``_node_synthesize`` become thin
wrappers that resolve an atlas LLM, wrap it in the ``AtlasLLMClient``
adapter, and delegate here.

``_node_reason`` is *not* moved in this slice -- it reads atlas-
specific extended-state fields (``crm_context``, ``b2b_churn``, voice
turns, etc. that atlas adds via the ``ReasoningAgentState`` subclass
established in PR-C4b). Atlas keeps the prompt builder and calls the
shared ``complete_with_json`` helper for the LLM round-trip. PR-C4e3
or a later slice can decide whether to push those fields up to core
or expose a richer prompt-building hook.

Each node accepts ``llm: LLMClient | None``. ``None`` means "LLM
unavailable" -- the node applies the same conservative fallback
behavior atlas's original implementation did (default to medium
priority + needs_reasoning for triage; fall back to deterministic
notification text for synthesize). The fallback path matters because
atlas runs reasoning on every event, not just the LLM-rich ones, and
the triage/synthesize stages must never block the pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .graph_helpers import (
    build_notification_fallback,
    complete_with_json,
    make_chat_messages,
    sanitize_notification_summary,
)
from .ports import LLMClient
from .state import ReasoningAgentState


logger = logging.getLogger("extracted_reasoning_core.graph_nodes")


# Triage prompt budget (atlas's settings.reasoning.triage_max_tokens default).
# Embedded as a default rather than read from settings because core stays
# host-agnostic; atlas's wrapper can pass its configured budget through.
_TRIAGE_MAX_TOKENS_DEFAULT = 256
_TRIAGE_TEMPERATURE = 0.1

_SYNTH_MAX_TOKENS_DEFAULT = 256
_SYNTH_TEMPERATURE_DEFAULT = 0.3


async def node_triage(
    state: ReasoningAgentState,
    llm: LLMClient | None,
    *,
    triage_system_prompt: str,
    max_tokens: int = _TRIAGE_MAX_TOKENS_DEFAULT,
) -> ReasoningAgentState:
    """Classify event priority and whether reasoning is needed.

    Reads ``event_type`` / ``source`` / ``entity_type`` / ``entity_id``
    / ``payload`` from state; writes ``triage_priority``,
    ``needs_reasoning``, ``triage_reasoning``, and accumulates token
    usage into ``total_input_tokens`` / ``total_output_tokens``.

    When ``llm is None`` (atlas LLM unavailable), defaults to medium
    priority + needs_reasoning so the rest of the graph still runs.
    Same behavior as atlas's pre-extraction ``_node_triage`` fallback.
    """
    event_desc = (
        f"Event: {state.get('event_type')}\n"
        f"Source: {state.get('source')}\n"
        f"Entity: {state.get('entity_type')}/{state.get('entity_id')}\n"
        f"Payload: {json.dumps(state.get('payload', {}), default=str)[:2000]}"
    )

    if llm is None:
        state["triage_priority"] = "medium"
        state["needs_reasoning"] = True
        state["triage_reasoning"] = "Triage LLM unavailable, defaulting to reason"
        return state

    try:
        result = await complete_with_json(
            llm,
            triage_system_prompt,
            event_desc,
            max_tokens=max_tokens,
            temperature=_TRIAGE_TEMPERATURE,
            json_mode=True,
        )
    except Exception:
        logger.warning("Triage failed, defaulting to reason", exc_info=True)
        state["triage_priority"] = "medium"
        state["needs_reasoning"] = True
        state["triage_reasoning"] = "Triage parse error, defaulting to reason"
        return state

    usage = result["usage"]
    state["total_input_tokens"] = (
        state.get("total_input_tokens", 0) + int(usage.get("input_tokens", 0) or 0)
    )
    state["total_output_tokens"] = (
        state.get("total_output_tokens", 0) + int(usage.get("output_tokens", 0) or 0)
    )

    parsed = result["parsed"]
    state["triage_priority"] = parsed.get("priority", "medium")
    state["needs_reasoning"] = parsed.get("needs_reasoning", True)
    state["triage_reasoning"] = parsed.get("reasoning", "")

    logger.info(
        "Triage: %s priority=%s needs_reasoning=%s",
        state.get("event_type"),
        state.get("triage_priority"),
        state.get("needs_reasoning"),
    )
    return state


async def node_synthesize(
    state: ReasoningAgentState,
    llm: LLMClient | None,
    *,
    synthesis_system_prompt: str,
    max_tokens: int = _SYNTH_MAX_TOKENS_DEFAULT,
    temperature: float = _SYNTH_TEMPERATURE_DEFAULT,
) -> ReasoningAgentState:
    """Generate a human-readable summary for notification.

    Skipped when ``state["should_notify"]`` is False -- writes empty
    summary and returns. Otherwise builds the synthesis prompt from
    ``event_type`` / ``action_results`` / ``rationale`` and calls the
    LLM. On LLM unavailability or post-processing failure, falls back
    to :func:`build_notification_fallback`.

    Note: this node uses the regular ``complete()`` (not
    ``complete_with_json``) -- the synthesis output is plain text, not
    structured JSON. We build messages and extract the text manually.
    """
    if not state.get("should_notify"):
        state["summary"] = ""
        return state

    context = (
        f"Event: {state.get('event_type')}\n"
        f"Actions taken: {json.dumps(state.get('action_results', []), default=str)[:1000]}\n"
        f"Rationale: {state.get('rationale', '')[:500]}"
    )

    if llm is None:
        state["summary"] = build_notification_fallback(state)
        return state

    try:
        messages = make_chat_messages(synthesis_system_prompt, context)
        result = await llm.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception:
        logger.warning("Synthesis failed, using deterministic fallback", exc_info=True)
        state["summary"] = build_notification_fallback(state)
        return state

    from .graph_helpers import extract_completion_text

    text, usage = extract_completion_text(result)
    state["total_input_tokens"] = (
        state.get("total_input_tokens", 0) + int(usage.get("input_tokens", 0) or 0)
    )
    state["total_output_tokens"] = (
        state.get("total_output_tokens", 0) + int(usage.get("output_tokens", 0) or 0)
    )

    if not text:
        state["summary"] = build_notification_fallback(state)
        return state

    state["summary"] = sanitize_notification_summary(text, state)
    return state


__all__ = ["node_synthesize", "node_triage"]
