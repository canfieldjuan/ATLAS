"""
Thread-safe state cache for Home Assistant entities.

Provides instant state lookups with automatic TTL expiration.
Used by the WebSocket client to store real-time state updates.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

logger = logging.getLogger("atlas.capabilities.state_cache")


@dataclass
class CachedState:
    """Cached entity state with metadata."""

    entity_id: str
    state: str
    attributes: dict[str, Any]
    last_changed: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    cached_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "entity_id": self.entity_id,
            "state": self.state,
            "attributes": self.attributes,
            "last_changed": self.last_changed.isoformat() if self.last_changed else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "cached_at": self.cached_at.isoformat(),
        }


class EntityStateCache:
    """
    Thread-safe cache for Home Assistant entity states.

    Features:
    - O(1) lookups by entity_id
    - Optional TTL expiration
    - Change listeners for reactive updates
    """

    def __init__(self, default_ttl_seconds: int = 300):
        self._cache: dict[str, CachedState] = {}
        self._ttl = timedelta(seconds=default_ttl_seconds)
        self._listeners: list[Callable[[str, CachedState], None]] = []
        self._lock = asyncio.Lock()

    async def get(self, entity_id: str) -> Optional[CachedState]:
        """Get cached state for entity, None if not cached or expired."""
        async with self._lock:
            cached = self._cache.get(entity_id)
            if cached is None:
                return None

            # Check TTL
            if datetime.now() - cached.cached_at > self._ttl:
                del self._cache[entity_id]
                logger.debug("Cache expired: %s", entity_id)
                return None

            return cached

    async def set(
        self,
        entity_id: str,
        state: str,
        attributes: dict[str, Any],
        last_changed: Optional[datetime] = None,
        last_updated: Optional[datetime] = None,
    ) -> None:
        """Update cached state for entity."""
        async with self._lock:
            cached = CachedState(
                entity_id=entity_id,
                state=state,
                attributes=attributes,
                last_changed=last_changed,
                last_updated=last_updated,
            )
            self._cache[entity_id] = cached

        # Notify listeners (outside lock to prevent deadlocks)
        for listener in self._listeners:
            try:
                listener(entity_id, cached)
            except Exception as e:
                logger.warning("State listener error for %s: %s", entity_id, e)

    async def update_from_event(self, event_data: dict[str, Any]) -> None:
        """
        Update cache from HA state_changed event.

        Expected event_data format:
        {
            "entity_id": "light.living_room",
            "new_state": {
                "state": "on",
                "attributes": {"brightness": 255},
                "last_changed": "2026-01-11T10:30:00+00:00",
                "last_updated": "2026-01-11T10:30:00+00:00"
            }
        }
        """
        entity_id = event_data.get("entity_id", "")
        new_state = event_data.get("new_state")

        if not entity_id or not new_state:
            return

        # Parse timestamps safely
        last_changed = None
        last_updated = None
        try:
            if new_state.get("last_changed"):
                last_changed = datetime.fromisoformat(
                    new_state["last_changed"].replace("Z", "+00:00")
                )
            if new_state.get("last_updated"):
                last_updated = datetime.fromisoformat(
                    new_state["last_updated"].replace("Z", "+00:00")
                )
        except (ValueError, TypeError) as e:
            logger.debug("Timestamp parse error for %s: %s", entity_id, e)

        await self.set(
            entity_id=entity_id,
            state=new_state.get("state", "unknown"),
            attributes=new_state.get("attributes", {}),
            last_changed=last_changed,
            last_updated=last_updated,
        )
        logger.debug("Cache updated from event: %s -> %s", entity_id, new_state.get("state"))

    def add_listener(self, callback: Callable[[str, CachedState], None]) -> None:
        """Add listener for state changes."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str, CachedState], None]) -> None:
        """Remove state change listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    async def clear(self) -> None:
        """Clear all cached states."""
        async with self._lock:
            self._cache.clear()
        logger.info("State cache cleared")

    def get_all_sync(self) -> dict[str, CachedState]:
        """Synchronous read of all cached states (for debugging/API)."""
        return dict(self._cache)

    @property
    def size(self) -> int:
        """Number of cached entities."""
        return len(self._cache)


# Module-level singleton
_state_cache: Optional[EntityStateCache] = None


def get_state_cache() -> EntityStateCache:
    """Get or create the state cache singleton."""
    global _state_cache
    if _state_cache is None:
        _state_cache = EntityStateCache()
    return _state_cache


def reset_state_cache() -> None:
    """Reset the state cache singleton (mainly for testing)."""
    global _state_cache
    _state_cache = None
