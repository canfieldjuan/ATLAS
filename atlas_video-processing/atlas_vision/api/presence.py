"""
Presence API endpoints.

Provides REST API for querying user presence and room states.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..presence import (
    get_presence_service,
    presence_config,
    PresenceSource,
)

logger = logging.getLogger("atlas.vision.api.presence")
router = APIRouter()


# === Response Models ===


class UserPresenceResponse(BaseModel):
    """User presence state."""
    user_id: str
    current_room: Optional[str] = None
    current_room_name: Optional[str] = None
    confidence: float = 0.0
    last_seen: Optional[datetime] = None
    source: Optional[str] = None
    is_stale: bool = False
    entered_current_room_at: Optional[datetime] = None


class RoomStateResponse(BaseModel):
    """Room state."""
    room_id: str
    room_name: str
    occupied: bool = False
    confidence: float = 0.0
    last_seen: Optional[datetime] = None
    primary_source: Optional[str] = None


class RoomDevicesResponse(BaseModel):
    """Devices in a room."""
    room_id: str
    room_name: str
    lights: list[str] = Field(default_factory=list)
    switches: list[str] = Field(default_factory=list)
    media_players: list[str] = Field(default_factory=list)
    ha_area: Optional[str] = None


class PresenceHealthResponse(BaseModel):
    """Presence service health."""
    enabled: bool
    running: bool
    espresense_enabled: bool
    espresense_running: bool
    camera_enabled: bool
    camera_registered: bool
    tracked_users: int
    rooms_configured: int
    occupied_rooms: int


class NearUserDevicesResponse(BaseModel):
    """Devices near a user."""
    user_id: str
    current_room: Optional[str] = None
    current_room_name: Optional[str] = None
    devices: list[str] = Field(default_factory=list)
    device_type: str


# === Endpoints ===


@router.get("/health", response_model=PresenceHealthResponse)
async def get_presence_health():
    """Get presence service health status."""
    from ..presence import get_espresense_subscriber, get_camera_consumer

    service = get_presence_service()
    espresense = get_espresense_subscriber()
    camera = get_camera_consumer()

    room_states = service.get_all_room_states()
    occupied_count = sum(1 for r in room_states.values() if r.occupied)

    return PresenceHealthResponse(
        enabled=presence_config.enabled,
        running=service.is_running,
        espresense_enabled=presence_config.espresense_enabled,
        espresense_running=espresense.is_running if espresense else False,
        camera_enabled=presence_config.camera_enabled,
        camera_registered=camera.is_registered if camera else False,
        tracked_users=len(service.get_all_user_presence()),
        rooms_configured=len(room_states),
        occupied_rooms=occupied_count,
    )


@router.get("/users", response_model=list[UserPresenceResponse])
async def list_user_presence():
    """Get presence state for all tracked users."""
    service = get_presence_service()
    users = service.get_all_user_presence()

    result = []
    for user_id, presence in users.items():
        result.append(UserPresenceResponse(
            user_id=presence.user_id,
            current_room=presence.current_room,
            current_room_name=presence.current_room_name,
            confidence=presence.confidence,
            last_seen=presence.last_seen,
            source=presence.source.value if presence.source else None,
            is_stale=presence.is_stale(presence_config.room_exit_timeout_seconds),
            entered_current_room_at=presence.entered_current_room_at,
        ))

    return result


@router.get("/users/{user_id}", response_model=UserPresenceResponse)
async def get_user_presence(user_id: str):
    """Get presence state for a specific user."""
    service = get_presence_service()
    presence = service.get_user_presence(user_id)

    if presence is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    return UserPresenceResponse(
        user_id=presence.user_id,
        current_room=presence.current_room,
        current_room_name=presence.current_room_name,
        confidence=presence.confidence,
        last_seen=presence.last_seen,
        source=presence.source.value if presence.source else None,
        is_stale=presence.is_stale(presence_config.room_exit_timeout_seconds),
        entered_current_room_at=presence.entered_current_room_at,
    )


@router.get("/users/{user_id}/room", response_model=dict)
async def get_user_room(user_id: str):
    """
    Get just the current room for a user.

    Fast path for near_user() tool queries.
    """
    service = get_presence_service()
    room_id = service.get_current_room(user_id)

    if room_id is None:
        return {"user_id": user_id, "room_id": None, "room_name": None}

    room_state = service.get_room_state(room_id)
    room_name = room_state.room_name if room_state else room_id

    return {"user_id": user_id, "room_id": room_id, "room_name": room_name}


@router.get("/users/{user_id}/devices", response_model=NearUserDevicesResponse)
async def get_devices_near_user(
    user_id: str,
    device_type: str = Query(default="lights", regex="^(lights|switches|media_players)$"),
):
    """
    Get devices near a user.

    This is the primary endpoint for near_user() tools to resolve
    device entity IDs without LLM involvement.
    """
    service = get_presence_service()

    devices = service.get_devices_near_user(user_id, device_type)
    presence = service.get_user_presence(user_id)

    return NearUserDevicesResponse(
        user_id=user_id,
        current_room=presence.current_room if presence else None,
        current_room_name=presence.current_room_name if presence else None,
        devices=devices,
        device_type=device_type,
    )


@router.get("/rooms", response_model=list[RoomStateResponse])
async def list_rooms():
    """Get all room states."""
    service = get_presence_service()
    rooms = service.get_all_room_states()

    result = []
    for room_id, room in rooms.items():
        result.append(RoomStateResponse(
            room_id=room.room_id,
            room_name=room.room_name,
            occupied=room.occupied,
            confidence=room.confidence,
            last_seen=room.last_seen,
            primary_source=room.primary_source.value if room.primary_source else None,
        ))

    return result


@router.get("/rooms/{room_id}", response_model=RoomStateResponse)
async def get_room_state(room_id: str):
    """Get state for a specific room."""
    service = get_presence_service()
    room = service.get_room_state(room_id)

    if room is None:
        raise HTTPException(status_code=404, detail=f"Room {room_id} not found")

    return RoomStateResponse(
        room_id=room.room_id,
        room_name=room.room_name,
        occupied=room.occupied,
        confidence=room.confidence,
        last_seen=room.last_seen,
        primary_source=room.primary_source.value if room.primary_source else None,
    )


@router.get("/rooms/{room_id}/devices", response_model=RoomDevicesResponse)
async def get_room_devices(room_id: str):
    """Get devices in a specific room."""
    service = get_presence_service()
    room = service.get_room_state(room_id)

    if room is None:
        raise HTTPException(status_code=404, detail=f"Room {room_id} not found")

    return RoomDevicesResponse(
        room_id=room.room_id,
        room_name=room.room_name,
        lights=room.lights,
        switches=room.switches,
        media_players=room.media_players,
        ha_area=room.ha_area,
    )


@router.get("/rooms/occupied", response_model=list[RoomStateResponse])
async def list_occupied_rooms():
    """Get only occupied rooms."""
    service = get_presence_service()
    rooms = service.get_all_room_states()

    result = []
    for room_id, room in rooms.items():
        if room.occupied:
            result.append(RoomStateResponse(
                room_id=room.room_id,
                room_name=room.room_name,
                occupied=room.occupied,
                confidence=room.confidence,
                last_seen=room.last_seen,
                primary_source=room.primary_source.value if room.primary_source else None,
            ))

    return result
