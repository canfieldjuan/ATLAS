"""
Reminder service with in-memory scheduler for instant delivery.

Uses PostgreSQL for persistence and asyncio timers for sub-second
precision delivery. When a reminder is due, it publishes to the
centralized alert system.
"""

import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Coroutine, Optional
from uuid import UUID

from ..config import settings
from ..storage import DatabaseUnavailableError, DatabaseOperationError
from ..storage.models import Reminder
from ..storage.repositories import get_reminder_repo

logger = logging.getLogger("atlas.services.reminders")

# Type alias for delivery callback
DeliveryCallback = Callable[[Reminder], Coroutine[Any, Any, None]]


class ReminderScheduler:
    """
    In-memory scheduler using asyncio for instant reminder delivery.

    Uses asyncio.call_later for precise timing without polling overhead.
    PostgreSQL is the source of truth; this is the execution layer.
    """

    def __init__(self):
        self._scheduled: dict[UUID, asyncio.TimerHandle] = {}
        self._delivery_callback: Optional[DeliveryCallback] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def set_delivery_callback(self, callback: DeliveryCallback) -> None:
        """Set the callback to invoke when a reminder is due."""
        self._delivery_callback = callback

    def schedule(self, reminder: Reminder) -> bool:
        """
        Schedule a reminder for delivery.

        If the reminder is already past due, delivers immediately.
        Returns True if scheduled successfully.
        """
        if not self._running:
            logger.debug("Scheduler not running, cannot schedule")
            return False

        # Skip already completed or delivered reminders
        if reminder.completed or reminder.delivered:
            logger.debug("Reminder %s already completed/delivered", reminder.id)
            return False

        with self._lock:
            if reminder.id in self._scheduled:
                logger.debug("Reminder %s already scheduled", reminder.id)
                return True

            now = datetime.now(timezone.utc)
            due_at = reminder.due_at

            # Ensure due_at is timezone-aware
            if due_at.tzinfo is None:
                due_at = due_at.replace(tzinfo=timezone.utc)

            delay_seconds = (due_at - now).total_seconds()

            if delay_seconds <= 0:
                # Already due, deliver immediately
                logger.info("Reminder %s is past due, delivering now", reminder.id)
                asyncio.create_task(self._deliver(reminder))
                return True

            # Schedule for future delivery
            if self._loop is None:
                self._loop = asyncio.get_running_loop()

            handle = self._loop.call_later(
                delay_seconds,
                lambda r=reminder: asyncio.create_task(self._deliver(r)),
            )
            self._scheduled[reminder.id] = handle

        logger.debug(
            "Scheduled reminder %s for delivery in %.1f seconds",
            reminder.id,
            delay_seconds,
        )
        return True

    def cancel(self, reminder_id: UUID) -> bool:
        """Cancel a scheduled reminder."""
        with self._lock:
            handle = self._scheduled.pop(reminder_id, None)
        if handle:
            handle.cancel()
            logger.debug("Cancelled reminder %s", reminder_id)
            return True
        return False

    async def _deliver(self, reminder: Reminder) -> None:
        """Deliver a reminder via the callback."""
        with self._lock:
            self._scheduled.pop(reminder.id, None)

        if not self._delivery_callback:
            logger.warning("No delivery callback set, reminder %s not delivered", reminder.id)
            return

        try:
            await self._delivery_callback(reminder)
            logger.info("Delivered reminder %s: %s", reminder.id, reminder.message)
        except Exception as e:
            logger.error("Failed to deliver reminder %s: %s", reminder.id, e)

    def start(self) -> None:
        """Start the scheduler."""
        self._running = True
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
        logger.info("Reminder scheduler started")

    def stop(self) -> None:
        """Stop the scheduler and cancel all pending deliveries."""
        self._running = False
        with self._lock:
            for handle in self._scheduled.values():
                handle.cancel()
            self._scheduled.clear()
        self._scheduled.clear()
        logger.info("Reminder scheduler stopped")

    @property
    def pending_count(self) -> int:
        """Number of reminders currently scheduled."""
        return len(self._scheduled)


class ReminderService:
    """
    High-level reminder service.

    Combines PostgreSQL persistence with in-memory scheduling
    for reliable and instant reminder delivery.
    """

    def __init__(self):
        self._scheduler = ReminderScheduler()
        self._initialized = False
        self._alert_callback: Optional[DeliveryCallback] = None
        self._reload_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """
        Initialize the service.

        Loads pending reminders from database and schedules them.
        Should be called during application startup.
        """
        if self._initialized:
            return

        if not settings.reminder.enabled:
            logger.info("Reminder service disabled by config")
            return

        self._scheduler.start()
        self._scheduler.set_delivery_callback(self._on_reminder_due)

        # Load pending reminders from database
        await self._load_pending_reminders()

        self._initialized = True

        # Start periodic reload task for long-term reminders
        self._reload_task = asyncio.create_task(self._periodic_reload_loop())

        logger.info(
            "Reminder service initialized with %d pending reminders",
            self._scheduler.pending_count,
        )

    async def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        # Cancel periodic reload task
        if self._reload_task and not self._reload_task.done():
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass

        self._scheduler.stop()
        self._initialized = False
        logger.info("Reminder service shutdown")

    def set_alert_callback(self, callback: DeliveryCallback) -> None:
        """
        Set callback for alert system integration.

        This is called when a reminder is due, allowing integration
        with the centralized alert system for TTS delivery.
        """
        self._alert_callback = callback

    async def _send_to_alerts(self, reminder: Reminder) -> None:
        """Send reminder to centralized alert system."""
        try:
            from ..alerts import ReminderAlertEvent, get_alert_manager

            alert_event = ReminderAlertEvent.from_reminder(reminder)
            manager = get_alert_manager()
            await manager.process_event(alert_event)
        except Exception as e:
            logger.error("Failed to send reminder to alerts: %s", e)

    async def _load_pending_reminders(self) -> None:
        """Load all pending reminders from database and schedule them."""
        try:
            repo = get_reminder_repo()

            # Get reminders due in the next 7 days
            # Longer-term reminders will be loaded on next startup or scheduled
            # when we approach their due time
            window = datetime.now(timezone.utc) + timedelta(days=7)
            reminders = await repo.get_pending(before=window)

            for reminder in reminders:
                self._scheduler.schedule(reminder)

            logger.info("Loaded %d pending reminders from database", len(reminders))

        except DatabaseUnavailableError:
            logger.warning("Database unavailable, skipping reminder load")
        except DatabaseOperationError as e:
            logger.error("Failed to load pending reminders: %s", e.cause)

    async def _periodic_reload_loop(self) -> None:
        """
        Periodically reload reminders approaching the scheduling window.

        This ensures reminders scheduled more than 7 days out are eventually
        loaded and scheduled as they approach their due time.
        """
        interval = settings.reminder.scheduler_check_interval_seconds
        reload_interval = 3600.0  # Check for new reminders every hour

        logger.info(
            "Periodic reload loop started (interval=%.0fs)",
            reload_interval,
        )

        try:
            while True:
                await asyncio.sleep(reload_interval)

                if not self._initialized:
                    break

                try:
                    await self._load_pending_reminders()
                except Exception as e:
                    logger.error("Periodic reload failed: %s", e)

        except asyncio.CancelledError:
            logger.debug("Periodic reload loop cancelled")

    async def _on_reminder_due(self, reminder: Reminder) -> None:
        """
        Handle reminder delivery.

        Uses atomic transaction for recurring reminders to ensure
        delivery and rescheduling either both succeed or both fail.
        """
        repo = get_reminder_repo()

        # Handle delivery and rescheduling atomically
        # For recurring reminders, this marks delivered AND creates next occurrence
        # in a single transaction to prevent lost reminders on partial failure
        next_reminder = await repo.deliver_recurring(reminder)

        # Send to centralized alert system
        await self._send_to_alerts(reminder)

        # Also trigger legacy callback if set (for backwards compatibility)
        if self._alert_callback:
            try:
                await self._alert_callback(reminder)
            except Exception as e:
                logger.error("Alert callback failed for reminder %s: %s", reminder.id, e)

        # Schedule next occurrence if this was a recurring reminder
        if next_reminder:
            self._scheduler.schedule(next_reminder)
            logger.info(
                "Rescheduled recurring reminder %s for %s",
                next_reminder.id,
                next_reminder.due_at,
            )

    async def create_reminder(
        self,
        message: str,
        due_at: datetime,
        user_id: Optional[UUID] = None,
        repeat_pattern: Optional[str] = None,
        source: str = "voice",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Reminder]:
        """
        Create a new reminder.

        Persists to database and schedules for delivery.
        Raises DatabaseUnavailableError or DatabaseOperationError on failure.
        """
        if not settings.reminder.enabled:
            logger.warning("Reminder service disabled, cannot create reminder")
            return None

        repo = get_reminder_repo()

        # Check reminder limit per user
        max_reminders = settings.reminder.max_reminders_per_user
        if max_reminders > 0:
            try:
                existing = await repo.get_user_reminders(
                    user_id=user_id,
                    include_completed=False,
                    limit=max_reminders + 1,
                )
                if len(existing) >= max_reminders:
                    logger.warning(
                        "User %s has reached max reminders limit (%d)",
                        user_id,
                        max_reminders,
                    )
                    return None
            except DatabaseUnavailableError:
                # Can't check limit, but allow creation to proceed
                logger.warning("Database unavailable for limit check, proceeding")

        # Ensure due_at is timezone-aware
        if due_at.tzinfo is None:
            due_at = due_at.replace(tzinfo=timezone.utc)

        # Create reminder - exceptions propagate to caller
        reminder = await repo.create(
            message=message,
            due_at=due_at,
            user_id=user_id,
            repeat_pattern=repeat_pattern,
            source=source,
            metadata=metadata,
        )

        self._scheduler.schedule(reminder)
        logger.info("Created and scheduled reminder: %s", reminder.id)

        return reminder

    async def list_reminders(
        self,
        user_id: Optional[UUID] = None,
        include_completed: bool = False,
        limit: int = 50,
    ) -> list[Reminder]:
        """List reminders, optionally filtered by user."""
        repo = get_reminder_repo()
        return await repo.get_user_reminders(
            user_id=user_id,
            include_completed=include_completed,
            limit=limit,
        )

    async def get_reminder(self, reminder_id: UUID) -> Optional[Reminder]:
        """Get a specific reminder by ID."""
        repo = get_reminder_repo()
        return await repo.get_by_id(reminder_id)

    async def complete_reminder(self, reminder_id: UUID) -> bool:
        """
        Mark a reminder as completed (user acknowledged).

        Also cancels any pending delivery.
        """
        repo = get_reminder_repo()

        # Cancel scheduled delivery
        self._scheduler.cancel(reminder_id)

        # Mark as completed in database
        return await repo.mark_completed(reminder_id)

    async def delete_reminder(self, reminder_id: UUID) -> bool:
        """
        Delete a reminder.

        Removes from database and cancels any pending delivery.
        """
        repo = get_reminder_repo()

        # Cancel scheduled delivery
        self._scheduler.cancel(reminder_id)

        # Delete from database
        return await repo.delete(reminder_id)

    async def get_next_reminder(
        self,
        user_id: Optional[UUID] = None,
    ) -> Optional[Reminder]:
        """Get the next upcoming reminder."""
        reminders = await self.list_reminders(
            user_id=user_id,
            include_completed=False,
            limit=1,
        )
        return reminders[0] if reminders else None

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    @property
    def pending_count(self) -> int:
        """Number of reminders scheduled in memory."""
        return self._scheduler.pending_count


# Global service instance
_reminder_service: Optional[ReminderService] = None


def get_reminder_service() -> ReminderService:
    """Get or create the global reminder service."""
    global _reminder_service
    if _reminder_service is None:
        _reminder_service = ReminderService()
    return _reminder_service


async def initialize_reminder_service() -> ReminderService:
    """Initialize and return the reminder service."""
    service = get_reminder_service()
    await service.initialize()
    return service


async def shutdown_reminder_service() -> None:
    """Shutdown the reminder service."""
    global _reminder_service
    if _reminder_service:
        await _reminder_service.shutdown()
        _reminder_service = None
