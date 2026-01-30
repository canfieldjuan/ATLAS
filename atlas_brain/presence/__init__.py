"""
Presence detection and room-level location tracking.

DEPRECATED: This module now proxies to atlas_vision service.

The actual presence tracking runs in atlas_vision. This module provides
backwards-compatible interfaces that fetch data via HTTP.

For direct access to presence data, use the atlas_vision API:
    GET /presence/users/{user_id}
    GET /presence/users/{user_id}/room
    GET /presence/users/{user_id}/devices
    GET /presence/rooms
    GET /presence/rooms/{room_id}
"""

import warnings
from typing import TYPE_CHECKING

# Export proxy service as the main interface
from .proxy import (
    PresenceProxyService,
    PresenceServiceCompat,
    PresenceSource,
    RoomState,
    UserPresence,
    get_presence_service,
    get_presence_proxy,
)

# Re-export config for backwards compatibility
from .config import PresenceConfig, RoomConfig, presence_config, DEFAULT_ROOMS

__all__ = [
    # Proxy service (new)
    "PresenceProxyService",
    "PresenceServiceCompat",
    "get_presence_service",
    "get_presence_proxy",
    # Data classes
    "PresenceSource",
    "RoomState",
    "UserPresence",
    # Config
    "PresenceConfig",
    "RoomConfig",
    "presence_config",
    "DEFAULT_ROOMS",
]


# Deprecation warnings for old interfaces
def __getattr__(name: str):
    """Provide deprecation warnings for removed items."""
    deprecated = {
        "PresenceService": "Use get_presence_service() which returns a proxy to atlas_vision",
        "set_presence_service": "Presence service now runs in atlas_vision, not locally",
        "ESPresenseSubscriber": "ESPresense now runs in atlas_vision",
        "start_espresense_subscriber": "ESPresense now runs in atlas_vision",
        "stop_espresense_subscriber": "ESPresense now runs in atlas_vision",
        "CameraPresenceConsumer": "Camera presence now runs in atlas_vision",
        "start_camera_presence_consumer": "Camera presence now runs in atlas_vision",
        "get_camera_consumer": "Camera presence now runs in atlas_vision",
    }

    if name in deprecated:
        warnings.warn(
            f"{name} is deprecated: {deprecated[name]}. "
            "Use GET {atlas_vision_url}/presence/* endpoints instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Return None to allow graceful degradation
        return None

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
