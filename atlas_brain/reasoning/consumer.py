"""Event consumer -- subscribes to EventBus, dispatches to ReasoningAgent."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from .event_bus import EventBus
from .events import AtlasEvent

logger = logging.getLogger("atlas.reasoning.consumer")


class EventConsumer:
    """Subscribes to the EventBus for all events and dispatches to the
    reasoning graph with concurrency control.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._expiry_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return

        from ..config import settings

        self._semaphore = asyncio.Semaphore(
            settings.reasoning.max_concurrent_reasoning
        )
        self._running = True

        # Subscribe to all events
        self._event_bus.subscribe("*", self._on_event)

        # Start periodic lock expiry
        self._expiry_task = asyncio.create_task(
            self._lock_expiry_loop(), name="reasoning_lock_expiry"
        )

        logger.info("EventConsumer started")

    async def stop(self) -> None:
        self._running = False
        self._event_bus.unsubscribe("*", self._on_event)

        if self._expiry_task:
            self._expiry_task.cancel()
            try:
                await self._expiry_task
            except asyncio.CancelledError:
                pass
            self._expiry_task = None

        logger.info("EventConsumer stopped")

    async def _on_event(self, event: AtlasEvent) -> None:
        """Handle incoming event from the EventBus."""
        from ..config import settings

        # Skip stale events
        if event.created_at:
            from datetime import datetime, timezone, timedelta
            max_age = timedelta(hours=settings.reasoning.event_max_age_hours)
            if datetime.now(timezone.utc) - event.created_at > max_age:
                logger.debug("Skipping stale event %s (age > %dh)", event.id, settings.reasoning.event_max_age_hours)
                return

        # Process with concurrency control
        async with self._semaphore:
            await self._process_event(event)

    async def _process_event(self, event: AtlasEvent) -> None:
        """Run event through the reasoning graph and record result."""
        from .agent import get_reasoning_agent
        from ..storage.database import get_db_pool

        agent = get_reasoning_agent()

        try:
            result = await agent.process_event(event)
        except Exception:
            logger.error("ReasoningAgent failed for event %s", event.id, exc_info=True)
            result = {"status": "error"}

        # Record processing result
        try:
            pool = get_db_pool()
            if pool.is_initialized:
                await pool.execute(
                    """
                    UPDATE atlas_events
                    SET processed_at = NOW(), processing_result = $1::jsonb
                    WHERE id = $2
                    """,
                    json.dumps(result, default=str),
                    event.id,
                )
        except Exception:
            logger.warning("Failed to record processing result for event %s", event.id, exc_info=True)

    async def _lock_expiry_loop(self) -> None:
        """Periodically expire stale entity locks."""
        from .entity_locks import EntityLockManager
        from ..config import settings

        mgr = EntityLockManager()
        interval = 60  # check every 60s

        while self._running:
            try:
                await asyncio.sleep(interval)
                expired = await mgr.expire_stale(settings.reasoning.lock_expiry_s)
                if expired:
                    logger.info("Expired %d stale entity locks", expired)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Lock expiry check failed", exc_info=True)
