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


class EventSink(Protocol):
    """Port for emitting reasoning-related events to the host's event bus.

    Atlas implements this against ``atlas_events`` (Postgres + LISTEN/NOTIFY).
    Other hosts can plug in any FIFO/log/queue that conforms to the contract.
    Reasoning core emits via this port; it never reaches into the host's
    storage layer directly.

    Returns the event id (opaque string) so callers can reference the
    persisted event in trace metadata, downstream lookups, or reply
    correlation.
    """

    async def emit(
        self,
        event_type: str,
        source: str,
        payload: Mapping[str, Any],
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> str:
        """Persist an event and return its identifier."""


class TraceSink(Protocol):
    """Port for emitting reasoning spans to the host's tracing backend.

    Atlas's tracing wrapper (LangSmith / OTel-style spans + business
    context) implements this. Reasoning core opens a span before LLM /
    cache work and closes it after, attaching status + metadata. Hosts
    that don't trace pass a no-op implementation.

    The ``span`` returned from ``start_span`` is opaque to reasoning core --
    it's a host-side handle that must be passed back to ``end_span``.
    """

    def start_span(
        self,
        name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Open a span and return an opaque span handle."""

    def end_span(
        self,
        span: Any,
        *,
        status: str = "ok",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Close a span. ``status`` is host-defined ('ok'/'error'/etc)."""


class Clock(Protocol):
    """Deterministic time source for recency and cache-decay decisions."""

    def now(self) -> datetime:
        """Return the current time."""


__all__ = [
    "Clock",
    "EventSink",
    "LLMClient",
    "ReasoningStateStore",
    "SemanticCacheStore",
    "TraceSink",
]
