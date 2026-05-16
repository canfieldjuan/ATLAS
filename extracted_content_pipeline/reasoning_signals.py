"""Shared signal names for Content Ops reasoning flows."""

from __future__ import annotations

REASONING_VALIDATION_BLOCKED = "reasoning_validation_blocked"


def reasoning_validation_blocked_reason(blockers: list[str] | tuple[str, ...]) -> str:
    """Serialize strict-validation blockers into the current reason string.

    The executor parses comma-separated blocker identifiers from this string.
    Callers should pass stable blocker IDs, not display text that may contain
    commas.
    """

    suffix = f":{','.join(blockers)}" if blockers else ""
    return f"{REASONING_VALIDATION_BLOCKED}{suffix}"
