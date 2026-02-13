"""
Presence tracker for occupancy state management.

Maintains in-memory occupancy state derived from edge security events
(person_entered / person_left).  Persists state transitions to DB and
fires callbacks that integrate with the hook/alert system.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("atlas.autonomous.presence")


class OccupancyState(Enum):
    EMPTY = "empty"
    OCCUPIED = "occupied"
    IDENTIFIED = "identified"


@dataclass
class PresenceState:
    """Current occupancy snapshot."""

    state: OccupancyState = OccupancyState.EMPTY
    occupants: list[str] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    changed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "occupants": list(self.occupants),
            "last_activity": self.last_activity.isoformat(),
            "changed_at": self.changed_at.isoformat(),
        }


@dataclass
class PresenceConfig:
    """Configuration for PresenceTracker behavior."""

    empty_delay_seconds: int = 300
    arrival_cooldown_seconds: int = 300


# Callback signature: (transition, state, person_name)
PresenceCallback = Callable[[str, PresenceState, Optional[str]], Awaitable[None]]


class PresenceTracker:
    """
    Tracks occupancy state from edge security events.

    State machine:
        EMPTY ──person_entered──▶ OCCUPIED / IDENTIFIED
        OCCUPIED ──identify──▶ IDENTIFIED
        IDENTIFIED ──person_left (all)──▶ (wait empty_delay) ──▶ EMPTY
        OCCUPIED ──person_left──▶ (wait empty_delay) ──▶ EMPTY

    Transitions fire callbacks that can trigger hook tasks via AlertManager.
    """

    def __init__(self, config: PresenceConfig | None = None):
        self._config = config or PresenceConfig()
        self._state = PresenceState()
        self._unknown_count: int = 0  # Track unidentified occupants
        self._empty_timer: asyncio.Task | None = None
        self._callbacks: list[PresenceCallback] = []
        self._last_arrival_fire: float = 0.0  # monotonic time

    # -- Public API -------------------------------------------------------

    @property
    def state(self) -> PresenceState:
        return self._state

    def register_callback(self, callback: PresenceCallback) -> None:
        self._callbacks.append(callback)

    async def on_security_event(self, event_type: str, metadata: dict[str, Any]) -> None:
        """
        Called from the WS handler on security events.

        Handles: person_entered, person_left
        """
        if event_type == "person_entered":
            name = metadata.get("name")
            is_known = metadata.get("is_known", False)
            await self._mark_occupied(name if is_known else None)

        elif event_type == "person_left":
            name = metadata.get("name")
            is_known = metadata.get("is_known", False)
            self._remove_occupant(name if is_known else None)
            self._schedule_empty_check()

    async def shutdown(self) -> None:
        """Cancel any pending timers."""
        if self._empty_timer and not self._empty_timer.done():
            self._empty_timer.cancel()

    # -- Internal ---------------------------------------------------------

    async def _mark_occupied(self, name: str | None) -> None:
        """Handle arrival.  Track known occupants, fire transition if was empty."""
        old_state = self._state.state
        now = datetime.utcnow()

        # Cancel pending empty check — someone is here
        if self._empty_timer and not self._empty_timer.done():
            self._empty_timer.cancel()
            self._empty_timer = None

        if name and name != "unknown":
            if name not in self._state.occupants:
                self._state.occupants.append(name)
            self._state.state = OccupancyState.IDENTIFIED
        else:
            self._unknown_count += 1
            if self._state.state == OccupancyState.EMPTY:
                self._state.state = OccupancyState.OCCUPIED

        self._state.last_activity = now

        if old_state == OccupancyState.EMPTY:
            self._state.changed_at = now
            await self._fire_transition("arrival", name)

        logger.debug(
            "Presence: %s → %s (occupants=%s)",
            old_state.value, self._state.state.value, self._state.occupants,
        )

    def _remove_occupant(self, name: str | None) -> None:
        """Remove a known or unknown occupant."""
        if name and name in self._state.occupants:
            self._state.occupants.remove(name)
        else:
            self._unknown_count = max(0, self._unknown_count - 1)
        self._state.last_activity = datetime.utcnow()

    def _schedule_empty_check(self) -> None:
        """After person_left, wait before declaring empty."""
        if self._empty_timer and not self._empty_timer.done():
            self._empty_timer.cancel()
        self._empty_timer = asyncio.create_task(self._check_empty())

    async def _check_empty(self) -> None:
        """If no occupants remain after delay, transition to EMPTY."""
        try:
            await asyncio.sleep(self._config.empty_delay_seconds)
            if not self._state.occupants and self._unknown_count == 0:
                old = self._state.state
                now = datetime.utcnow()
                self._state.state = OccupancyState.EMPTY
                self._state.changed_at = now
                self._unknown_count = 0
                if old != OccupancyState.EMPTY:
                    await self._fire_transition("departure", None)
                    logger.info("Presence: %s → EMPTY", old.value)
        except asyncio.CancelledError:
            pass

    async def _fire_transition(self, transition: str, person: str | None) -> None:
        """Notify callbacks and persist to DB."""
        import time

        # Arrival cooldown — prevent rapid re-fires
        if transition == "arrival":
            now_mono = time.monotonic()
            if (now_mono - self._last_arrival_fire) < self._config.arrival_cooldown_seconds:
                logger.debug("Arrival transition suppressed (cooldown)")
                return
            self._last_arrival_fire = now_mono

        logger.info(
            "Presence transition: %s (person=%s, state=%s, occupants=%s)",
            transition, person, self._state.state.value, self._state.occupants,
        )

        # Persist to DB (best-effort)
        await self._persist_transition(transition, person)

        # Fire callbacks
        for cb in self._callbacks:
            try:
                await cb(transition, self._state, person)
            except Exception as e:
                logger.error("Presence callback failed: %s", e)

    async def _persist_transition(self, transition: str, person: str | None) -> None:
        """Save transition to presence_events table."""
        try:
            from ..storage.database import get_db_pool

            pool = get_db_pool()
            if not pool.is_initialized:
                return

            await pool.execute(
                """
                INSERT INTO presence_events
                    (transition, occupancy_state, occupants, person_name, source_id)
                VALUES ($1, $2, $3, $4, $5)
                """,
                transition,
                self._state.state.value,
                self._state.occupants,
                person,
                "system",
            )
        except Exception as e:
            logger.warning("Failed to persist presence transition: %s", e)


# -- Singleton ------------------------------------------------------------

_presence_tracker: Optional[PresenceTracker] = None


def get_presence_tracker() -> PresenceTracker:
    """Get the global presence tracker."""
    global _presence_tracker
    if _presence_tracker is None:
        _presence_tracker = PresenceTracker()
    return _presence_tracker


def reset_presence_tracker() -> None:
    """Reset the global presence tracker (for testing)."""
    global _presence_tracker
    _presence_tracker = None
