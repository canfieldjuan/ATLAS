"""Shared text truncation helpers for deterministic content outputs."""

from __future__ import annotations

from typing import Any


def truncate_with_ellipsis(
    value: Any,
    max_chars: Any,
    *,
    suffix: str = "...",
    compact_whitespace: bool = True,
    small_limit: str = "suffix",
) -> str:
    """Return text no longer than max_chars, reserving room for suffix."""

    text = _text(value, compact_whitespace=compact_whitespace)
    limit = _limit(max_chars)
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""

    suffix_text = str(suffix or "")
    if not suffix_text:
        return text[:limit].rstrip()

    reserve = limit - len(suffix_text)
    if reserve <= 0:
        if small_limit == "text":
            return text[:limit]
        return suffix_text[:limit]
    return text[:reserve].rstrip() + suffix_text


def _text(value: Any, *, compact_whitespace: bool) -> str:
    text = str(value or "")
    if compact_whitespace:
        return " ".join(text.split())
    return text.strip()


def _limit(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
