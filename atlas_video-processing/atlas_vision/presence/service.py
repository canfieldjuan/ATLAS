"""
Presence Service - Real-time user location tracking.

Maintains hot state for user locations by fusing signals from:
- ESPresense (BLE beacons via MQTT)
- Camera person detection
- Home Assistant device trackers

The service provides instant room lookups for tools without LLM involvement.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Optional

from .config import PresenceConfig, RoomConfig, DEFAULT_ROOMS, presence_config

logger = logging.getLogger("atlas.vision.presence")


class PresenceSource(str, Enum):
    """Source of presence detection."""
    BLE = "ble"  # ESPresense BLE beacon
    CAMERA = "camera"  # YOLO person detection
    GPS = "gps"  # Home Assistant device tracker
    MANUAL = "manual"  # Manual override


@dataclass
class RoomReading:
    """A single presence reading for a room."""
    room_id: str
    source: PresenceSource
    confidence: float  # 0-1
    distance: Optional[float] = None  # meters (for BLE)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RoomState:
    """Current state for a room."""
    room_id: str
    room_name: str
    occupied: bool = False
    confidence: float = 0.0
    last_seen: Optional[datetime] = None
    primary_source: Optional[PresenceSource] = None

    # For device resolution
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

    # Recent room history (for "where was I?" queries)
    room_history: list[tuple[str, datetime]] = field(default_factory=list)

    # State machine
    entered_current_room_at: Optional[datetime] = None
    pending_room: Optional[str] = None  # Room we might be transitioning to
    pending_since: Optional[datetime] = None

    def is_stale(self, timeout_seconds: float = 60.0) -> bool:
        """Check if presence data is stale."""
        if self.last_seen is None:
            return True
        return (datetime.now() - self.last_seen).total_seconds() > timeout_seconds


class PresenceService:
    """
    Real-time presence tracking service.

    Fuses signals from multiple sources to maintain accurate room-level
    user location. Provides instant lookups for near_user() tools.
    """

    def __init__(
        self,
        config: Optional[PresenceConfig] = None,
        rooms: Optional[list[RoomConfig]] = None,
    ):
        self.config = config or presence_config
        self._rooms = {r.id: r for r in (rooms or DEFAULT_ROOMS)}
        self._room_states: dict[str, RoomState] = {}
        self._user_presence: dict[str, UserPresence] = {}
        self._ble_readings: dict[str, deque[RoomReading]] = {}  # user_id -> recent readings
        self._listeners: list[Callable] = []
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

        # Initialize room states
        for room_id, room_config in self._rooms.items():
            self._room_states[room_id] = RoomState(
                room_id=room_id,
                room_name=room_config.name,
                lights=room_config.lights,
                switches=room_config.switches,
                media_players=room_config.media_players,
                ha_area=room_config.ha_area,
            )

        logger.info(
            "Presence service initialized with %d rooms: %s",
            len(self._rooms),
            list(self._rooms.keys()),
        )

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._running

    async def start(self) -> None:
        """Start the presence service."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Presence service started")

    async def stop(self) -> None:
        """Stop the presence service."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Presence service stopped")

    # === Public API ===

    def get_user_presence(self, user_id: Optional[str] = None) -> Optional[UserPresence]:
        """
        Get current presence for a user.

        Args:
            user_id: User ID, or None for default user

        Returns:
            UserPresence with current room, or None if unknown
        """
        user_id = user_id or self.config.default_user_id
        return self._user_presence.get(user_id)

    def get_current_room(self, user_id: Optional[str] = None) -> Optional[str]:
        """
        Get current room ID for a user.

        This is the fast path for tools - returns immediately.
        """
        presence = self.get_user_presence(user_id)
        if presence and not presence.is_stale(self.config.room_exit_timeout_seconds):
            return presence.current_room
        return None

    def get_room_state(self, room_id: str) -> Optional[RoomState]:
        """Get current state for a room."""
        return self._room_states.get(room_id)

    def get_devices_near_user(
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
        room_id = self.get_current_room(user_id)
        if not room_id:
            logger.debug("No current room for user %s", user_id)
            return []

        room_state = self._room_states.get(room_id)
        if not room_state:
            return []

        return getattr(room_state, device_type, [])

    def get_all_room_states(self) -> dict[str, RoomState]:
        """Get all room states."""
        return self._room_states.copy()

    def get_all_user_presence(self) -> dict[str, UserPresence]:
        """Get all user presence states."""
        return self._user_presence.copy()

    # === Signal Ingestion ===

    async def handle_ble_reading(
        self,
        device_id: str,
        room_id: str,
        distance: float,
        rssi: Optional[int] = None,
    ) -> None:
        """
        Handle a BLE reading from ESPresense.

        Args:
            device_id: BLE device identifier (e.g., "iphone_juan")
            room_id: ESPresense room/node ID
            distance: Estimated distance in meters
            rssi: Signal strength (optional)
        """
        # Map device to user
        user_id = self.config.tracked_devices.get(device_id)
        if not user_id:
            # Use default user for untracked devices
            user_id = self.config.default_user_id

        # Map ESPresense room ID to our room ID
        mapped_room = self._map_espresense_room(room_id)
        if not mapped_room:
            logger.debug("Unknown ESPresense room: %s", room_id)
            return

        # Calculate confidence based on distance
        if distance <= self.config.ble_distance_threshold:
            # Closer = higher confidence (inverse relationship)
            confidence = max(0.0, 1.0 - (distance / self.config.ble_distance_threshold))
        else:
            confidence = 0.0

        reading = RoomReading(
            room_id=mapped_room,
            source=PresenceSource.BLE,
            confidence=confidence,
            distance=distance,
        )

        await self._process_reading(user_id, reading)

    async def handle_camera_detection(
        self,
        camera_source: str,
        person_detected: bool,
        track_id: Optional[int] = None,
        confidence: float = 0.8,
    ) -> None:
        """
        Handle a camera person detection event.

        Args:
            camera_source: Camera source ID
            person_detected: Whether a person is currently detected
            track_id: Track ID for the person (for re-identification)
            confidence: Detection confidence
        """
        # Map camera to room
        room_id = self._map_camera_to_room(camera_source)
        if not room_id:
            logger.debug("Unknown camera source: %s", camera_source)
            return

        # For now, assume single user - future: use track_id for multi-user
        user_id = self.config.default_user_id

        if person_detected:
            reading = RoomReading(
                room_id=room_id,
                source=PresenceSource.CAMERA,
                confidence=confidence,
            )
            await self._process_reading(user_id, reading)
        else:
            # Person left camera view - will be handled by timeout
            pass

    async def handle_gps_update(
        self,
        user_id: str,
        state: str,  # "home", "not_home", or zone name
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> None:
        """
        Handle GPS update from Home Assistant device tracker.

        This is a fallback - only used when BLE/camera not available.
        """
        if state == "not_home":
            # Clear presence when user leaves home
            if user_id in self._user_presence:
                old_room = self._user_presence[user_id].current_room
                self._user_presence[user_id].current_room = None
                self._user_presence[user_id].current_room_name = None
                self._user_presence[user_id].last_seen = datetime.now()
                logger.info("User %s left home (was in %s)", user_id, old_room)
                await self._notify_change(user_id, old_room, None)

    # === Internal Processing ===

    async def _process_reading(self, user_id: str, reading: RoomReading) -> None:
        """Process a presence reading and update state."""
        # Ensure user exists
        if user_id not in self._user_presence:
            self._user_presence[user_id] = UserPresence(user_id=user_id)

        presence = self._user_presence[user_id]

        # For BLE, apply smoothing
        if reading.source == PresenceSource.BLE:
            if user_id not in self._ble_readings:
                self._ble_readings[user_id] = deque(maxlen=self.config.ble_smoothing_window)
            self._ble_readings[user_id].append(reading)
            reading = self._get_smoothed_reading(user_id)

        # Check if reading meets threshold
        if reading.confidence < self.config.room_enter_threshold:
            return

        now = datetime.now()
        old_room = presence.current_room

        # Same room - just update timestamp
        if reading.room_id == presence.current_room:
            presence.last_seen = now
            presence.confidence = reading.confidence
            presence.source = reading.source
            return

        # Different room - apply hysteresis
        if presence.pending_room == reading.room_id:
            # Already pending transition to this room
            elapsed = (now - presence.pending_since).total_seconds()
            if elapsed >= self.config.hysteresis_seconds:
                # Transition confirmed
                self._transition_room(presence, reading.room_id, reading, now)
                await self._notify_change(user_id, old_room, reading.room_id)
        else:
            # Start pending transition
            presence.pending_room = reading.room_id
            presence.pending_since = now
            logger.debug(
                "User %s pending transition: %s -> %s",
                user_id,
                presence.current_room,
                reading.room_id,
            )

    def _transition_room(
        self,
        presence: UserPresence,
        new_room: str,
        reading: RoomReading,
        now: datetime,
    ) -> None:
        """Execute room transition."""
        old_room = presence.current_room

        # Update room history
        if old_room:
            presence.room_history.append((old_room, now))
            # Keep last 20 entries
            presence.room_history = presence.room_history[-20:]

            # Mark old room as unoccupied
            if old_room in self._room_states:
                self._room_states[old_room].occupied = False

        # Update presence
        presence.current_room = new_room
        presence.current_room_name = self._rooms.get(new_room, RoomConfig(id=new_room, name=new_room)).name
        presence.confidence = reading.confidence
        presence.last_seen = now
        presence.source = reading.source
        presence.entered_current_room_at = now
        presence.pending_room = None
        presence.pending_since = None

        # Mark new room as occupied
        if new_room in self._room_states:
            self._room_states[new_room].occupied = True
            self._room_states[new_room].confidence = reading.confidence
            self._room_states[new_room].last_seen = now
            self._room_states[new_room].primary_source = reading.source

        logger.info(
            "User %s moved: %s -> %s (confidence: %.2f, source: %s)",
            presence.user_id,
            old_room or "unknown",
            new_room,
            reading.confidence,
            reading.source.value,
        )

    def _get_smoothed_reading(self, user_id: str) -> RoomReading:
        """Get smoothed BLE reading from recent history."""
        readings = self._ble_readings.get(user_id, [])
        if not readings:
            return RoomReading(room_id="unknown", source=PresenceSource.BLE, confidence=0.0)

        # Find room with most readings
        room_counts: dict[str, list[RoomReading]] = {}
        for r in readings:
            if r.room_id not in room_counts:
                room_counts[r.room_id] = []
            room_counts[r.room_id].append(r)

        # Pick room with highest count, then highest avg confidence
        best_room = None
        best_score = -1
        for room_id, room_readings in room_counts.items():
            avg_confidence = sum(r.confidence for r in room_readings) / len(room_readings)
            score = len(room_readings) * 10 + avg_confidence  # Weight count heavily
            if score > best_score:
                best_score = score
                best_room = room_id

        if best_room:
            room_readings = room_counts[best_room]
            avg_confidence = sum(r.confidence for r in room_readings) / len(room_readings)
            avg_distance = sum(r.distance or 0 for r in room_readings) / len(room_readings)
            return RoomReading(
                room_id=best_room,
                source=PresenceSource.BLE,
                confidence=avg_confidence,
                distance=avg_distance,
            )

        return readings[-1]  # Fallback to most recent

    def _map_espresense_room(self, espresense_room: str) -> Optional[str]:
        """Map ESPresense room ID to our room ID."""
        for room_id, room_config in self._rooms.items():
            if espresense_room in room_config.espresense_devices:
                return room_id
        # Try direct match
        if espresense_room in self._rooms:
            return espresense_room
        return None

    def _map_camera_to_room(self, camera_source: str) -> Optional[str]:
        """Map camera source ID to room ID."""
        for room_id, room_config in self._rooms.items():
            if camera_source in room_config.camera_sources:
                return room_id
        return None

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale presence data."""
        while self._running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                await self._cleanup_stale()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup error: %s", e)

    async def _cleanup_stale(self) -> None:
        """Clean up stale presence data."""
        now = datetime.now()
        timeout = timedelta(seconds=self.config.room_exit_timeout_seconds)

        for user_id, presence in self._user_presence.items():
            if presence.last_seen and (now - presence.last_seen) > timeout:
                if presence.current_room:
                    old_room = presence.current_room
                    logger.info(
                        "User %s presence timed out in %s",
                        user_id,
                        old_room,
                    )
                    # Mark room as unoccupied
                    if old_room in self._room_states:
                        self._room_states[old_room].occupied = False

                    presence.current_room = None
                    presence.current_room_name = None
                    await self._notify_change(user_id, old_room, None)

    # === Listeners ===

    def add_listener(self, callback: Callable[[str, Optional[str], Optional[str]], None]) -> None:
        """
        Add a listener for room changes.

        Callback receives: (user_id, old_room, new_room)
        """
        self._listeners.append(callback)

    async def _notify_change(
        self,
        user_id: str,
        old_room: Optional[str],
        new_room: Optional[str],
    ) -> None:
        """Notify listeners of room change."""
        for listener in self._listeners:
            try:
                result = listener(user_id, old_room, new_room)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Listener error: %s", e)


# Singleton instance
_presence_service: Optional[PresenceService] = None


def get_presence_service() -> PresenceService:
    """Get the singleton presence service instance."""
    global _presence_service
    if _presence_service is None:
        _presence_service = PresenceService()
    return _presence_service


def set_presence_service(service: PresenceService) -> None:
    """Set the singleton presence service instance."""
    global _presence_service
    _presence_service = service
