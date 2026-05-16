"""Shared signal names for Content Ops reasoning flows."""

from __future__ import annotations

REASONING_VALIDATION_BLOCKED = "reasoning_validation_blocked"


def reasoning_validation_blocked_reason(blockers: list[str] | tuple[str, ...]) -> str:
    suffix = f":{','.join(blockers)}" if blockers else ""
    return f"{REASONING_VALIDATION_BLOCKED}{suffix}"
