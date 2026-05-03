"""Public ports for the extracted reasoning core."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Protocol, Sequence


class LLMClient(Protocol):
    """Provider-neutral completion port used by reasoning runners."""

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Return a provider completion response."""


class SemanticCacheStore(Protocol):
    """Port for semantic reasoning-cache storage."""

    async def lookup(self, key: str) -> Mapping[str, Any] | None:
        """Return a cached reasoning entry, if present and usable."""

    async def store(self, key: str, value: Mapping[str, Any]) -> None:
        """Persist a reasoning-cache entry."""

    async def validate(self, key: str, confidence: float | None = None) -> None:
        """Refresh validation metadata for a cached entry."""

    async def invalidate(self, key: str, reason: str = "") -> None:
        """Invalidate a cached entry."""


class ReasoningStateStore(Protocol):
    """Port for long-running reasoning state and continuation checkpoints."""

    async def load_state(self, state_id: str) -> Mapping[str, Any] | None:
        """Return a stored reasoning state by id."""

    async def save_state(self, state_id: str, state: Mapping[str, Any]) -> None:
        """Persist a reasoning state snapshot."""


class Clock(Protocol):
    """Deterministic time source for recency and cache-decay decisions."""

    def now(self) -> datetime:
        """Return the current time."""


__all__ = [
    "Clock",
    "LLMClient",
    "ReasoningStateStore",
    "SemanticCacheStore",
]
