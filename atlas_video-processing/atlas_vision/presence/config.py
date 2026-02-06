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
        env_prefix="ATLAS_VISION_PRESENCE_",
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
# Maps rooms to actual registered device IDs in the capability registry
#
# Camera source naming convention:
#   - webcam_*: Local USB webcams (e.g., webcam_office)
#   - wyze_*: Wyze cameras via wyze-bridge (e.g., wyze_living_room)
#
# Wyze-bridge camera names are derived from Wyze app names:
#   "Living Room Cam" -> rtsp://localhost:8554/living-room-cam
#   The source_id should match what you set in ATLAS_RTSP_CAMERAS_JSON
#
DEFAULT_ROOMS: list[RoomConfig] = [
    RoomConfig(
        id="living_room",
        name="Living Room",
        espresense_devices=["living-room"],
        camera_sources=["wyze_living_room"],
        ha_area="living_room",
        lights=["living_room_light"],
        switches=[],
        media_players=["media_player.living_room_tv"],
    ),
    RoomConfig(
        id="kitchen",
        name="Kitchen",
        espresense_devices=["kitchen"],
        camera_sources=["wyze_kitchen"],
        ha_area="kitchen",
        lights=["kitchen_light"],
        switches=["fan_switch"],
        media_players=[],
    ),
    RoomConfig(
        id="bedroom",
        name="Bedroom",
        espresense_devices=["bedroom"],
        camera_sources=["wyze_bedroom"],
        ha_area="bedroom",
        lights=["bedroom_light"],
        switches=[],
        media_players=[],
    ),
    RoomConfig(
        id="office",
        name="Office",
        espresense_devices=["office"],
        camera_sources=["webcam_office"],
        ha_area="office",
        lights=[],
        switches=["input_boolean.office_light", "input_boolean.test_light"],
        media_players=[],
    ),
    RoomConfig(
        id="outdoor_front",
        name="Front Yard",
        espresense_devices=[],
        camera_sources=["wyze_outdoor"],
        ha_area=None,
        lights=[],
        switches=[],
        media_players=[],
    ),
]


# Singleton
presence_config = PresenceConfig()
