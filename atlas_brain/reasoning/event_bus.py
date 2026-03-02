"""In-process event bus backed by PostgreSQL LISTEN/NOTIFY."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Awaitable, Optional
from uuid import UUID

from .events import AtlasEvent

logger = logging.getLogger("atlas.reasoning.event_bus")

# Callback signature: async def handler(event: AtlasEvent) -> None
EventCallback = Callable[[AtlasEvent], Awaitable[None]]


class EventBus:
    """Listens for atlas_events via pg_notify, dispatches to subscribers.

    Subscribers register for specific event_type prefixes (e.g. "email."
    matches all email events). A "*" subscription matches everything.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventCallback]] = {}
        self._connection = None  # dedicated asyncpg connection for LISTEN
        self._listen_task: Optional[asyncio.Task] = None
        self._running = False
        self._queue: asyncio.Queue[UUID] = asyncio.Queue(maxsize=10_000)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, prefix: str, callback: EventCallback) -> None:
        """Register a callback for events whose type starts with *prefix*."""
        self._subscribers.setdefault(prefix, []).append(callback)
        logger.debug("Subscribed %s to prefix '%s'", callback, prefix)

    def unsubscribe(self, prefix: str, callback: EventCallback) -> None:
        cbs = self._subscribers.get(prefix, [])
        if callback in cbs:
            cbs.remove(callback)

    async def start(self) -> None:
        """Start listening for pg_notify on the atlas_events channel."""
        if self._running:
            return
        self._running = True
        self._listen_task = asyncio.create_task(self._listen_loop(), name="event_bus_listen")
        logger.info("EventBus started")

    async def stop(self) -> None:
        """Stop the listener and drain."""
        self._running = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        if self._connection and not self._connection.is_closed():
            await self._connection.close()
            self._connection = None
        logger.info("EventBus stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """Acquire a dedicated connection, LISTEN, and dispatch."""
        import asyncpg

        from ..storage.database import get_db_pool

        pool = get_db_pool()

        while self._running:
            try:
                # Get a raw connection from the pool's DSN
                if self._connection is None or self._connection.is_closed():
                    self._connection = await pool.acquire_raw()
                    await self._connection.add_listener(
                        "atlas_events", self._on_notify
                    )
                    logger.info("LISTEN atlas_events established")

                # Process dispatched event IDs
                while self._running:
                    try:
                        event_id = await asyncio.wait_for(
                            self._queue.get(), timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        continue
                    await self._dispatch_event(event_id)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "EventBus listen loop error; reconnecting in 5s",
                    exc_info=True,
                )
                if self._connection and not self._connection.is_closed():
                    try:
                        await self._connection.close()
                    except Exception:
                        pass
                    self._connection = None
                await asyncio.sleep(5)

    def _on_notify(
        self, connection: Any, pid: int, channel: str, payload: str
    ) -> None:
        """pg_notify callback -- enqueue event_id for async dispatch."""
        try:
            event_id = UUID(payload)
            self._queue.put_nowait(event_id)
        except (ValueError, asyncio.QueueFull):
            logger.warning("Failed to enqueue notify payload: %s", payload)

    async def _dispatch_event(self, event_id: UUID) -> None:
        """Load event from DB and fan out to matching subscribers."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            "SELECT id, event_type, source, entity_type, entity_id, payload, created_at"
            " FROM atlas_events WHERE id = $1",
            event_id,
        )
        if not row:
            return

        event = AtlasEvent.from_row(dict(row))

        for prefix, callbacks in list(self._subscribers.items()):
            if prefix == "*" or event.event_type.startswith(prefix):
                for cb in list(callbacks):
                    try:
                        await cb(event)
                    except Exception:
                        logger.error(
                            "Subscriber %s failed for event %s (type=%s)",
                            cb,
                            event_id,
                            event.event_type,
                            exc_info=True,
                        )
