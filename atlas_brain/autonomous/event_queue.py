"""
In-memory event queue with dedup, batching, and debounce.

Routes alert-triggered hook task dispatches through a queue that
collapses rapid-fire duplicate events before invoking HookManager.
Latency-sensitive callbacks (TTS, ntfy, DB persistence) still fire
immediately in AlertManager — only hook dispatch is debounced.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable

from ..alerts.events import AlertEvent
from ..alerts.rules import AlertRule

logger = logging.getLogger("atlas.autonomous.event_queue")


@dataclass
class QueuedEvent:
    """A deduplicated event waiting for batch dispatch."""

    event: AlertEvent
    rule: AlertRule
    message: str
    count: int = 1
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EventQueueConfig:
    """Configuration for EventQueue behavior."""

    debounce_seconds: float = 5.0
    max_batch_size: int = 50
    max_age_seconds: float = 30.0


class EventQueue:
    """
    In-memory event queue with dedup, batching, and debounce.

    Events are keyed by (rule_name, source_id, class_name).  Repeated
    events within the debounce window increment a counter instead of
    creating duplicates.  After the debounce timer fires, the batch is
    flushed to all registered callbacks.

    A max-age timer ensures events are never held longer than
    ``max_age_seconds`` even under continuous traffic.
    """

    def __init__(self, config: EventQueueConfig | None = None):
        self._config = config or EventQueueConfig()
        self._pending: dict[str, QueuedEvent] = {}
        self._flush_task: asyncio.Task | None = None
        self._max_age_task: asyncio.Task | None = None
        self._callbacks: list[Callable[[list[QueuedEvent]], Awaitable[None]]] = []
        self._total_enqueued = 0
        self._total_deduplicated = 0
        self._total_flushed = 0

    # -- Public API -------------------------------------------------------

    def register_callback(
        self,
        callback: Callable[[list[QueuedEvent]], Awaitable[None]],
    ) -> None:
        """Register a batch callback (e.g. HookManager.on_alert_batch)."""
        self._callbacks.append(callback)

    async def enqueue(
        self,
        event: AlertEvent,
        rule: AlertRule,
        message: str,
    ) -> None:
        """Add an event to the queue.  Dedup by key, debounce flush."""
        key = self._dedup_key(event, rule)
        now = datetime.utcnow()
        self._total_enqueued += 1

        if key in self._pending:
            self._pending[key].count += 1
            self._pending[key].last_seen = now
            self._total_deduplicated += 1
            logger.debug("Dedup event %s (count=%d)", key, self._pending[key].count)
        else:
            self._pending[key] = QueuedEvent(
                event=event,
                rule=rule,
                message=message,
                first_seen=now,
                last_seen=now,
            )

        # Enforce max batch size — flush immediately if full
        if len(self._pending) >= self._config.max_batch_size:
            await self._flush()
            return

        self._schedule_flush()

    async def shutdown(self) -> None:
        """Flush remaining events and cancel timers."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        if self._max_age_task and not self._max_age_task.done():
            self._max_age_task.cancel()
        if self._pending:
            await self._flush()

    @property
    def stats(self) -> dict[str, Any]:
        """Return queue statistics."""
        return {
            "pending": len(self._pending),
            "total_enqueued": self._total_enqueued,
            "total_deduplicated": self._total_deduplicated,
            "total_flushed": self._total_flushed,
        }

    # -- Internal ---------------------------------------------------------

    @staticmethod
    def _dedup_key(event: AlertEvent, rule: AlertRule) -> str:
        """(rule_name, source_id, class_name) — collapses repeated detections."""
        class_name = event.get_field("class_name", "")
        return f"{rule.name}:{event.source_id}:{class_name}"

    def _schedule_flush(self) -> None:
        """Debounce: reset flush timer on each new event."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = asyncio.create_task(self._delayed_flush())

        # Ensure max-age timer is running
        if self._max_age_task is None or self._max_age_task.done():
            self._max_age_task = asyncio.create_task(self._max_age_flush())

    async def _delayed_flush(self) -> None:
        """Wait for debounce window, then flush."""
        try:
            await asyncio.sleep(self._config.debounce_seconds)
            await self._flush()
        except asyncio.CancelledError:
            pass

    async def _max_age_flush(self) -> None:
        """Ensure events are never held longer than max_age_seconds."""
        try:
            await asyncio.sleep(self._config.max_age_seconds)
            if self._pending:
                logger.debug("Max-age flush triggered (%d pending)", len(self._pending))
                await self._flush()
        except asyncio.CancelledError:
            pass

    async def _flush(self) -> None:
        """Dispatch all pending events to registered callbacks."""
        if not self._pending:
            return

        batch = list(self._pending.values())
        self._pending.clear()
        self._total_flushed += len(batch)

        # Cancel timers that aren't the current caller to avoid self-cancellation
        current = asyncio.current_task()
        for attr in ("_flush_task", "_max_age_task"):
            task_ref = getattr(self, attr)
            if task_ref and task_ref is not current and not task_ref.done():
                task_ref.cancel()
            setattr(self, attr, None)

        total_events = sum(e.count for e in batch)
        deduped = total_events - len(batch)
        logger.info(
            "Queue flush: %d events (%d deduplicated from %d total)",
            len(batch), deduped, total_events,
        )

        for callback in self._callbacks:
            try:
                await callback(batch)
            except Exception as e:
                logger.error("Queue callback failed: %s", e)
