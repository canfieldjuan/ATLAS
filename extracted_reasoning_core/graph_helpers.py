"""Host-agnostic helpers extracted from atlas's reasoning graph.

PR-C4e1 (this module) is the first sub-slice of the audit's PR 6 plan
to port the reasoning graph engine into core. These functions are pure
logic -- no atlas imports, no I/O -- so they can land in core ahead of
the LLMClient port alignment (PR-C4e2) and the host-coupled-node
adapters (PR-C4e3).

What lives here:

- ``parse_llm_json`` -- robust JSON extraction from LLM output
  (raw, fenced, or embedded in prose).
- ``valid_uuid_str`` -- canonical-UUID-or-None validator used by action
  dispatch to gate CRM writes on a real contact id.
- ``clean_summary_text`` / ``has_suspicious_trailing_fragment`` /
  ``build_notification_fallback`` / ``sanitize_notification_summary``
  -- the graph-synthesis post-processor that produces the
  owner-facing push-notification text from raw LLM output.
- ``plan_actions`` -- the safe-action filter that gates which
  recommended actions get executed (the ``SAFE_ACTIONS`` allowlist
  + confidence threshold).

Atlas's ``atlas_brain.reasoning.graph`` re-exports these under the
existing private names (e.g. ``_parse_llm_json``) so internal callers
(notably ``reflection.py``) don't need to change import sites in this
slice -- the next sub-slices will redirect those imports as the
LLM-driven nodes follow into core.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from uuid import UUID

from .state import ReasoningAgentState

logger = logging.getLogger("extracted_reasoning_core.graph_helpers")


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

# Actions the graph is willing to execute. Anything outside this set
# is logged-and-skipped at plan time; the planner never gets to silently
# trigger destructive paths (e.g. "send_email", "delete_contact").
SAFE_ACTIONS: frozenset[str] = frozenset({
    "generate_draft",
    "show_slots",
    "log_interaction",
    "create_reminder",
    "send_notification",
})

_PLAN_MIN_CONFIDENCE: float = 0.5


def parse_llm_json(text: str) -> dict[str, Any]:
    """Extract and parse JSON from an LLM response.

    Handles raw JSON, markdown-fenced JSON, and JSON embedded in prose.
    Raises ``json.JSONDecodeError`` if no valid object is found.
    """
    text = text.strip()
    if not text:
        raise json.JSONDecodeError("Empty response", text, 0)

    if text.startswith("{"):
        return json.loads(text)

    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1).strip())

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        return json.loads(text[first : last + 1])

    raise json.JSONDecodeError("No JSON object found in response", text, 0)


def valid_uuid_str(value: Any) -> str | None:
    """Return the canonical UUID string if ``value`` parses as a UUID, else None."""
    if not value:
        return None
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        return None


def clean_summary_text(text: str) -> str:
    """Normalize graph synthesis output into plain notification text."""
    cleaned = _JSON_FENCE_RE.sub(r"\1", text or "")
    cleaned = _MARKDOWN_RE.sub("", cleaned)
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def has_suspicious_trailing_fragment(text: str) -> bool:
    """Detect likely truncated sentence endings like ``"bef."``."""
    match = re.search(r"\b([A-Za-z]{1,3})[.!?]?$", text.strip())
    if not match:
        return False
    token = match.group(1)
    if token.lower() in _COMMON_SHORT_FINAL_WORDS:
        return False
    return token.islower()


def build_notification_fallback(state: ReasoningAgentState) -> str:
    """Build a deterministic fallback summary from actions and rationale.

    Used when the synthesis LLM is unavailable or returns garbage that
    the sanitizer rejects in full. Reads ``action_results``,
    ``connections_found``, and ``rationale`` from state.
    """
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


def sanitize_notification_summary(text: str, state: ReasoningAgentState) -> str:
    """Remove meta narration and keep a short owner-facing summary.

    Falls back to :func:`build_notification_fallback` when the input
    yields no usable sentences (e.g. the LLM returned only meta
    narration like ``"Drafting a summary..."``).
    """
    cleaned = clean_summary_text(text)
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
        if has_suspicious_trailing_fragment(candidate):
            continue
        kept.append(candidate)
        if len(kept) >= 2:
            break
    if not kept:
        return build_notification_fallback(state)
    summary = " ".join(s.rstrip(".!?") + "." for s in kept)
    return summary[:320].rstrip()


async def plan_actions(state: ReasoningAgentState) -> ReasoningAgentState:
    """Filter ``recommended_actions`` down to the safe-and-confident subset.

    Reads ``recommended_actions``; writes ``planned_actions``.
    Skips anything outside :data:`SAFE_ACTIONS` or below the
    confidence floor. ``async`` for parity with the other graph nodes
    even though no I/O happens here.
    """
    planned: list[dict[str, Any]] = []
    for action in state.get("recommended_actions", []):
        tool = action.get("tool", "")
        if tool not in SAFE_ACTIONS:
            logger.warning("Skipping unsafe action: %s", tool)
            continue
        confidence = action.get("confidence", 0)
        if confidence < _PLAN_MIN_CONFIDENCE:
            logger.debug(
                "Skipping low-confidence action: %s (%.2f)", tool, confidence
            )
            continue
        planned.append(action)

    state["planned_actions"] = planned
    return state


__all__ = [
    "SAFE_ACTIONS",
    "build_notification_fallback",
    "clean_summary_text",
    "has_suspicious_trailing_fragment",
    "parse_llm_json",
    "plan_actions",
    "sanitize_notification_summary",
    "valid_uuid_str",
]
