"""Read-only / dry-run tool-surface qualification for the draft-only follow-up worker.

The #2126 follow-up contract (``followup_workflow.py``) fixed the worker's RESULT
shape: server-owned approval, and ``next_permitted_actions`` that can never send.
This module fixes the complementary surface -- the worker's INPUT tools. The qualified
draft-only worker may be handed ONLY read tools; it has no send/mutate capability at
all, so a "dry run" (compose a draft, change nothing) is the only thing it can do.

Guard-class-closure (per docs/GUARD_CLASS_CLOSURE.md): qualification is a fail-closed
ALLOWLIST choke point (``FOLLOWUP_READONLY_TOOLS``), not a denylist of forbidden verbs.
A denylist cannot close the class -- the mutating tools share no single detectable verb
(``send_email``, ``create_contact``, ``record_payment``, and ``log_interaction`` have
nothing lexical in common), so any verb-pattern check would leak ``log_interaction``.
The allowlist inverts that: any tool NOT explicitly a known read tool is disqualified,
so every send/create/update/delete/approve/void/log tool -- and any unknown future tool
-- fails closed by default.

``KNOWN_MUTATING_TOOLS`` is NOT the guard; it exists only to (a) self-audit that the
allowlist is disjoint from real mutating tools (enforced at import) and (b) give sharper
negative tests. All tool names are real Atlas MCP tools verified in the server modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

# CLOSED allowlist: the read-only MCP tools a draft-only follow-up worker may be given.
# Grounded in the real CRM / Email / Calendar server tools (all query/read, no side
# effect). Anything not in this set -- including every mutating tool below and any
# unknown tool -- is disqualified. Membership is deliberately conservative: it is the
# ceiling of what a draft-only worker may touch, not a suggestion of what it must use.
FOLLOWUP_READONLY_TOOLS: frozenset[str] = frozenset(
    {
        # CRM reads (customer lookup + history for the follow-up)
        "search_contacts",
        "get_contact",
        "list_contacts",
        "get_customer_context",
        "get_interactions",
        "get_contact_appointments",
        # Email reads (prior correspondence context)
        "list_inbox",
        "get_message",
        "search_inbox",
        "get_thread",
        "list_sent_history",
        "list_folders",
        # Calendar reads (appointment context)
        "list_calendars",
        "list_events",
        "get_event",
        "find_free_slots",
    }
)

# Real Atlas MCP tools that DO cause a side effect (send / create / update / delete /
# approve / void / record / log). Not the admission guard -- used only for the import-
# time self-audit and negative tests. Representative, not exhaustive: the allowlist
# closure, not this set, is what rejects a mutating tool.
KNOWN_MUTATING_TOOLS: frozenset[str] = frozenset(
    {
        # CRM writes
        "create_contact",
        "update_contact",
        "delete_contact",
        "log_interaction",
        # Email sends
        "send_email",
        "send_estimate",
        "send_proposal",
        # Twilio sends
        "make_call",
        "send_sms",
        # Calendar writes
        "create_event",
        "update_event",
        "delete_event",
        "sync_appointment",
        # Invoicing writes / money movement
        "create_invoice",
        "update_invoice",
        "send_invoice",
        "approve_and_send",
        "record_payment",
        "mark_void",
    }
)

# Fail closed at import if a tool is ever declared both read-only and mutating -- a
# contradiction that would silently poke a hole in the allowlist. A raise (not assert,
# which -O strips) makes the invariant load-bearing.
_overlap = FOLLOWUP_READONLY_TOOLS & KNOWN_MUTATING_TOOLS
if _overlap:
    raise RuntimeError(
        f"follow-up read-only allowlist overlaps mutating tools: {sorted(_overlap)}"
    )
del _overlap


@dataclass(frozen=True)
class ToolSurfaceQualification:
    """Verdict for a proposed follow-up worker tool surface.

    ``qualified`` is True only when every offered tool is within
    ``FOLLOWUP_READONLY_TOOLS`` (so the surface can perform no send/mutation).
    ``disallowed`` lists the offending tool names; ``mutating`` is the subset of those
    that are known side-effecting tools (for a clearer message)."""

    qualified: bool
    offered: tuple[str, ...]
    disallowed: tuple[str, ...]
    mutating: tuple[str, ...]
    reason: str


def _normalize(name: Any) -> str:
    """A tool name is a non-blank string; anything else normalizes to '' so it fails
    the allowlist check (fail closed on a malformed/blank/non-string entry)."""
    return name.strip() if isinstance(name, str) else ""


def qualify_followup_tool_surface(offered: Iterable[Any]) -> ToolSurfaceQualification:
    """Qualify a proposed tool surface as read-only / dry-run, failing closed.

    Every offered tool must be an exact member of ``FOLLOWUP_READONLY_TOOLS``. Any tool
    outside it -- a mutating tool, an unknown tool, a case/whitespace variant, or a
    blank/non-string entry -- disqualifies the surface. An empty surface is trivially
    qualified (it can send nothing). This is the deterministic core a later slice can
    point at a live MCP ``list_tools`` result to prove the worker's real surface sends
    nothing."""
    offered_names: list[str] = []
    disallowed: list[str] = []
    mutating: list[str] = []
    for item in offered:
        name = _normalize(item)
        display = name if name else f"<invalid:{item!r}>"
        offered_names.append(display)
        if not name or name not in FOLLOWUP_READONLY_TOOLS:
            disallowed.append(display)
            if name in KNOWN_MUTATING_TOOLS:
                mutating.append(name)

    disallowed_sorted = tuple(sorted(set(disallowed)))
    mutating_sorted = tuple(sorted(set(mutating)))
    qualified = not disallowed_sorted

    if qualified:
        reason = (
            "all offered tools are read-only follow-up tools; no send/mutation capability"
            if offered_names
            else "no tools offered; trivially read-only"
        )
    else:
        detail = ", ".join(disallowed_sorted)
        note = (
            f" (mutating: {', '.join(mutating_sorted)})" if mutating_sorted else ""
        )
        reason = (
            f"disqualified: {len(disallowed_sorted)} tool(s) outside the read-only "
            f"follow-up surface: {detail}{note}"
        )

    return ToolSurfaceQualification(
        qualified=qualified,
        offered=tuple(offered_names),
        disallowed=disallowed_sorted,
        mutating=mutating_sorted,
        reason=reason,
    )


def is_read_only_tool(name: Any) -> bool:
    """True only for an exact member of the read-only allowlist."""
    return _normalize(name) in FOLLOWUP_READONLY_TOOLS
