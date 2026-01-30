"""
Presence Proxy Service - Fetches presence data from atlas_vision.

This module provides the same interface as the original PresenceService
but fetches data from atlas_vision via HTTP instead of running locally.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import httpx

from ..config import settings

logger = logging.getLogger("atlas.presence.proxy")


class PresenceSource(str, Enum):
    """Source of presence detection."""
    BLE = "ble"
    CAMERA = "camera"
    GPS = "gps"
    MANUAL = "manual"


@dataclass
class RoomState:
    """Current state for a room."""
    room_id: str
    room_name: str
    occupied: bool = False
    confidence: float = 0.0
    last_seen: Optional[datetime] = None
    primary_source: Optional[PresenceSource] = None
    lights: list[str] = field(default_factory=list)
    switches: list[str] = field(default_factory=list)
    media_players: list[str] = field(default_factory=list)
    ha_area: Optional[str] = None


@dataclass
class UserPresence:
    """Current presence state for a user."""
    user_id: str
    current_room: Optional[str] = None
    current_room_name: Optional[str] = None
    confidence: float = 0.0
    last_seen: Optional[datetime] = None
    source: Optional[PresenceSource] = None
    is_stale: bool = False
    entered_current_room_at: Optional[datetime] = None


class PresenceProxyService:
    """
    Proxy service that fetches presence data from atlas_vision.

    Provides the same interface as PresenceService for backwards compatibility.
    """

    def __init__(self, vision_url: Optional[str] = None):
        self.vision_url = vision_url or settings.security.video_processing_url
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def base_url(self) -> str:
        """Get the atlas_vision presence API base URL."""
        return f"{self.vision_url}/presence"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=5.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def get_current_room(self, user_id: Optional[str] = None) -> Optional[str]:
        """
        Get current room ID for a user.

        This is the fast path for tools - returns immediately.
        """
        user_id = user_id or "primary"
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/users/{user_id}/room")
            if response.status_code == 200:
                data = response.json()
                return data.get("room_id")
            elif response.status_code == 404:
                return None
            else:
                logger.warning("Presence API error: %d", response.status_code)
                return None
        except Exception as e:
            logger.error("Failed to get current room: %s", e)
            return None

    async def get_devices_near_user(
        self,
        user_id: Optional[str] = None,
        device_type: str = "lights",
    ) -> list[str]:
        """
        Get device entity IDs near a user.

        Args:
            user_id: User ID, or None for default
            device_type: "lights", "switches", or "media_players"

        Returns:
            List of Home Assistant entity IDs
        """
        user_id = user_id or "primary"
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/users/{user_id}/devices",
                params={"device_type": device_type}
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("devices", [])
            else:
                logger.warning("Devices API error: %d", response.status_code)
                return []
        except Exception as e:
            logger.error("Failed to get devices near user: %s", e)
            return []

    async def get_room_state(self, room_id: str) -> Optional[RoomState]:
        """Get current state for a room."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/rooms/{room_id}")
            if response.status_code == 200:
                data = response.json()
                return RoomState(
                    room_id=data["room_id"],
                    room_name=data["room_name"],
                    occupied=data.get("occupied", False),
                    confidence=data.get("confidence", 0.0),
                    last_seen=datetime.fromisoformat(data["last_seen"]) if data.get("last_seen") else None,
                    primary_source=PresenceSource(data["primary_source"]) if data.get("primary_source") else None,
                )
            elif response.status_code == 404:
                return None
            else:
                logger.warning("Room state API error: %d", response.status_code)
                return None
        except Exception as e:
            logger.error("Failed to get room state: %s", e)
            return None

    async def get_room_devices(self, room_id: str) -> Optional[RoomState]:
        """Get devices in a specific room."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/rooms/{room_id}/devices")
            if response.status_code == 200:
                data = response.json()
                return RoomState(
                    room_id=data["room_id"],
                    room_name=data["room_name"],
                    lights=data.get("lights", []),
                    switches=data.get("switches", []),
                    media_players=data.get("media_players", []),
                    ha_area=data.get("ha_area"),
                )
            elif response.status_code == 404:
                return None
            else:
                logger.warning("Room devices API error: %d", response.status_code)
                return None
        except Exception as e:
            logger.error("Failed to get room devices: %s", e)
            return None

    async def get_user_presence(self, user_id: Optional[str] = None) -> Optional[UserPresence]:
        """Get presence state for a specific user."""
        user_id = user_id or "primary"
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/users/{user_id}")
            if response.status_code == 200:
                data = response.json()
                return UserPresence(
                    user_id=data["user_id"],
                    current_room=data.get("current_room"),
                    current_room_name=data.get("current_room_name"),
                    confidence=data.get("confidence", 0.0),
                    last_seen=datetime.fromisoformat(data["last_seen"]) if data.get("last_seen") else None,
                    source=PresenceSource(data["source"]) if data.get("source") else None,
                    is_stale=data.get("is_stale", False),
                    entered_current_room_at=datetime.fromisoformat(data["entered_current_room_at"]) if data.get("entered_current_room_at") else None,
                )
            elif response.status_code == 404:
                return None
            else:
                logger.warning("User presence API error: %d", response.status_code)
                return None
        except Exception as e:
            logger.error("Failed to get user presence: %s", e)
            return None


# Synchronous wrapper for backwards compatibility with existing tools
class PresenceServiceCompat:
    """
    Synchronous-style wrapper around PresenceProxyService.

    The original PresenceService methods were synchronous. This wrapper
    provides the same interface but runs async calls under the hood.
    """

    def __init__(self, proxy: Optional[PresenceProxyService] = None):
        self._proxy = proxy or PresenceProxyService()
        # Cache for room states to avoid repeated fetches
        self._room_cache: dict[str, RoomState] = {}

    def get_current_room(self, user_id: Optional[str] = None) -> Optional[str]:
        """Get current room ID - runs async fetch synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context - schedule as task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._proxy.get_current_room(user_id)
                    )
                    return future.result(timeout=5.0)
            else:
                return loop.run_until_complete(self._proxy.get_current_room(user_id))
        except Exception as e:
            logger.error("Sync get_current_room failed: %s", e)
            return None

    def get_devices_near_user(
        self,
        user_id: Optional[str] = None,
        device_type: str = "lights",
    ) -> list[str]:
        """Get devices near user - runs async fetch synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._proxy.get_devices_near_user(user_id, device_type)
                    )
                    return future.result(timeout=5.0)
            else:
                return loop.run_until_complete(
                    self._proxy.get_devices_near_user(user_id, device_type)
                )
        except Exception as e:
            logger.error("Sync get_devices_near_user failed: %s", e)
            return []

    def get_room_state(self, room_id: str) -> Optional[RoomState]:
        """Get room state - runs async fetch synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._proxy.get_room_state(room_id)
                    )
                    return future.result(timeout=5.0)
            else:
                return loop.run_until_complete(self._proxy.get_room_state(room_id))
        except Exception as e:
            logger.error("Sync get_room_state failed: %s", e)
            return None

    def get_user_presence(self, user_id: Optional[str] = None) -> Optional[UserPresence]:
        """Get user presence - runs async fetch synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._proxy.get_user_presence(user_id)
                    )
                    return future.result(timeout=5.0)
            else:
                return loop.run_until_complete(self._proxy.get_user_presence(user_id))
        except Exception as e:
            logger.error("Sync get_user_presence failed: %s", e)
            return None


# Singleton instances
_presence_proxy: Optional[PresenceProxyService] = None
_presence_compat: Optional[PresenceServiceCompat] = None


def get_presence_service() -> PresenceServiceCompat:
    """
    Get the presence service instance.

    Returns a compatibility wrapper that provides the same interface
    as the original PresenceService but fetches from atlas_vision.
    """
    global _presence_proxy, _presence_compat
    if _presence_proxy is None:
        _presence_proxy = PresenceProxyService()
    if _presence_compat is None:
        _presence_compat = PresenceServiceCompat(_presence_proxy)
    return _presence_compat


def get_presence_proxy() -> PresenceProxyService:
    """Get the async presence proxy service."""
    global _presence_proxy
    if _presence_proxy is None:
        _presence_proxy = PresenceProxyService()
    return _presence_proxy
