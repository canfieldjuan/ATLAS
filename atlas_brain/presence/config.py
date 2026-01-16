"""
Configuration for the presence detection system.
"""

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RoomConfig(BaseModel):
    """Configuration for a single room."""

    id: str  # e.g., "living_room"
    name: str  # e.g., "Living Room"

    # ESPresense device IDs that map to this room
    espresense_devices: list[str] = Field(default_factory=list)

    # Camera source IDs that cover this room
    camera_sources: list[str] = Field(default_factory=list)

    # Home Assistant area (for device resolution)
    ha_area: Optional[str] = None

    # Devices in this room (entity_ids)
    lights: list[str] = Field(default_factory=list)
    switches: list[str] = Field(default_factory=list)
    media_players: list[str] = Field(default_factory=list)


class PresenceConfig(BaseSettings):
    """Presence service configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_PRESENCE_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable presence tracking")

    # ESPresense MQTT settings
    espresense_enabled: bool = Field(default=True, description="Use ESPresense BLE tracking")
    espresense_topic_prefix: str = Field(
        default="espresense/rooms",
        description="MQTT topic prefix for ESPresense (e.g., espresense/rooms/{room}/{device})"
    )

    # Camera-based presence
    camera_enabled: bool = Field(default=True, description="Use camera person detection")

    # State machine tuning
    room_enter_threshold: float = Field(
        default=0.7,
        description="Confidence threshold to enter a room (0-1)"
    )
    room_exit_timeout_seconds: float = Field(
        default=30.0,
        description="Seconds without detection before leaving room"
    )
    hysteresis_seconds: float = Field(
        default=2.0,
        description="Minimum time before switching rooms (prevents flapping)"
    )

    # BLE signal processing
    ble_distance_threshold: float = Field(
        default=3.0,
        description="Max distance in meters to consider 'in room'"
    )
    ble_smoothing_window: int = Field(
        default=3,
        description="Number of readings to smooth for stability"
    )

    # Default user (for single-user setup)
    default_user_id: str = Field(
        default="primary",
        description="Default user ID for single-user homes"
    )

    # Device identifiers to track (BLE MAC addresses, iBeacon UUIDs, etc.)
    tracked_devices: dict[str, str] = Field(
        default_factory=dict,
        description="Map of device_id to user_id (e.g., {'iphone_juan': 'juan'})"
    )


# Room definitions - can be loaded from DB or config file
# This is the default/example configuration
DEFAULT_ROOMS: list[RoomConfig] = [
    RoomConfig(
        id="living_room",
        name="Living Room",
        espresense_devices=["living-room"],
        camera_sources=["living_room_cam"],
        ha_area="living_room",
        lights=["light.living_room", "light.living_room_lamp"],
        switches=[],
        media_players=["media_player.living_room_tv"],
    ),
    RoomConfig(
        id="kitchen",
        name="Kitchen",
        espresense_devices=["kitchen"],
        camera_sources=["kitchen_cam"],
        ha_area="kitchen",
        lights=["light.kitchen", "light.kitchen_counter"],
        switches=[],
        media_players=[],
    ),
    RoomConfig(
        id="bedroom",
        name="Bedroom",
        espresense_devices=["bedroom"],
        camera_sources=["bedroom_cam"],
        ha_area="bedroom",
        lights=["light.bedroom", "light.bedroom_lamp"],
        switches=[],
        media_players=["media_player.bedroom_tv"],
    ),
    RoomConfig(
        id="office",
        name="Office",
        espresense_devices=["office"],
        camera_sources=["office_cam"],
        ha_area="office",
        lights=["light.office", "light.desk_lamp"],
        switches=[],
        media_players=[],
    ),
]


# Singleton
presence_config = PresenceConfig()
