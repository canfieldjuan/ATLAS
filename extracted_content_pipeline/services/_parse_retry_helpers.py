"""Shared parse-retry helpers for generated content asset services."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def clip_invalid_response(text: str, *, limit: int) -> str:
    """Return a stripped response excerpt capped to ``limit`` characters."""

    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip()


def parse_attempt_limit(parse_retry_attempts: int) -> int:
    """Return total parse attempts: one initial attempt plus configured retries."""

    return max(1, int(parse_retry_attempts or 0) + 1)


def retry_prompt_with_invalid_response(
    base_prompt: str,
    *,
    prior_invalid_response: str,
    instruction: str,
) -> str:
    """Append the standard invalid-response retry context when needed."""

    if not prior_invalid_response:
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        f"{instruction} "
        f"Previous response excerpt:\n{prior_invalid_response}"
    )


def accumulate_usage(
    total: Mapping[str, Any],
    usage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Accumulate numeric usage fields while preserving non-numeric metadata."""

    accumulated = dict(total)
    if not isinstance(usage, Mapping):
        return accumulated
    for key, value in usage.items():
        if isinstance(value, bool):
            accumulated[key] = value
        elif isinstance(value, (int, float)):
            prior = accumulated.get(key)
            if isinstance(prior, (int, float)) and not isinstance(prior, bool):
                accumulated[key] = prior + value
            else:
                accumulated[key] = value
        else:
            accumulated[key] = value
    return accumulated
