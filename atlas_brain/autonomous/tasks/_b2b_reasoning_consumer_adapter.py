"""Consumer adapter helpers for reasoning payload overlays.

Additive adapter layer used by MCP/API consumers to derive stable reasoning
fields from a synthesis view without changing existing response contracts.
"""

from __future__ import annotations

from typing import Any


def reasoning_summary_fields_from_view(view: object) -> dict[str, Any]:
    """Return stable reasoning summary fields derived from a synthesis view."""
    from ._b2b_synthesis_reader import synthesis_view_to_reasoning_entry

    entry = synthesis_view_to_reasoning_entry(view)
    return {
        "archetype": entry.get("archetype"),
        "archetype_confidence": entry.get("confidence"),
        "reasoning_mode": entry.get("mode"),
        "reasoning_risk_level": entry.get("risk_level"),
    }


def reasoning_detail_fields_from_view(view: object) -> dict[str, Any]:
    """Return stable reasoning detail fields derived from a synthesis view.

    List-valued fields are guaranteed to be lists (never None) even when the
    upstream entry has explicit null values for those keys -- dict.get(k, [])
    only falls back to [] when k is missing, NOT when k is present-but-null.
    """
    from ._b2b_synthesis_reader import synthesis_view_to_reasoning_entry

    entry = synthesis_view_to_reasoning_entry(view)
    return {
        "archetype": entry.get("archetype"),
        "archetype_confidence": entry.get("confidence"),
        "reasoning_mode": entry.get("mode"),
        "reasoning_risk_level": entry.get("risk_level"),
        "reasoning_executive_summary": entry.get("executive_summary"),
        "reasoning_key_signals": entry.get("key_signals") or [],
        "reasoning_uncertainty_sources": entry.get("uncertainty_sources") or [],
        "falsification_conditions": entry.get("falsification_conditions") or [],
    }
