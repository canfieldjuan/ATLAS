"""
Presence detection and room-level location tracking.

This module provides real-time user location tracking by fusing signals from:
- BLE beacons (ESPresense)
- Camera person detection (YOLO)
- Home Assistant device trackers (GPS fallback)

The presence service maintains hot state (current room, confidence) that tools
can query instantly without involving the LLM in location resolution.

Architecture:
    ESPresense MQTT ─┐
                     ├──► Presence Service ──► {room_id, confidence, last_seen}
    Camera Events ───┤                              │
                     │                              ▼
    HA Tracker ──────┘                    near_user() tools resolve
                                          devices without LLM knowing room
"""

from .service import (
    PresenceService,
    UserPresence,
    RoomState,
    RoomReading,
    PresenceSource,
    get_presence_service,
    set_presence_service,
)
from .config import PresenceConfig, RoomConfig, presence_config, DEFAULT_ROOMS
from .espresense import (
    ESPresenseSubscriber,
    start_espresense_subscriber,
    stop_espresense_subscriber,
    get_espresense_subscriber,
)
from .camera import (
    CameraPresenceConsumer,
    start_camera_presence_consumer,
    get_camera_consumer,
)

__all__ = [
    # Core service
    "PresenceService",
    "UserPresence",
    "RoomState",
    "RoomReading",
    "PresenceSource",
    "get_presence_service",
    "set_presence_service",
    # Config
    "PresenceConfig",
    "RoomConfig",
    "presence_config",
    "DEFAULT_ROOMS",
    # ESPresense
    "ESPresenseSubscriber",
    "start_espresense_subscriber",
    "stop_espresense_subscriber",
    "get_espresense_subscriber",
    # Camera
    "CameraPresenceConsumer",
    "start_camera_presence_consumer",
    "get_camera_consumer",
]
