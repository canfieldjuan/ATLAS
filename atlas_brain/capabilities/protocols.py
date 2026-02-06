"""
Protocol definitions for the capability/device system.

Capabilities are abstractions for anything controllable or observable:
devices, integrations, composite actions, etc.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


class CapabilityType(str, Enum):
    """Classification of capability types."""
    LIGHT = "light"
    SWITCH = "switch"
    SENSOR = "sensor"
    CAMERA = "camera"
    LOCK = "lock"
    THERMOSTAT = "thermostat"
    MEDIA_PLAYER = "media_player"
    ACTUATOR = "actuator"
    CUSTOM = "custom"


@dataclass
class CapabilityState:
    """Base state for all capabilities."""
    online: bool = True
    last_updated: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "online": self.online,
            "last_updated": self.last_updated,
            "attributes": self.attributes,
        }


@dataclass
class ActionResult:
    """Result of executing an action on a capability."""
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "error": self.error,
        }


@runtime_checkable
class Capability(Protocol):
    """
    Protocol for all capabilities in the Atlas system.

    A capability represents anything that can be observed, controlled,
    or interacted with - devices, integrations, composite actions, etc.
    """

    @property
    def id(self) -> str:
        """Unique identifier for this capability."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name."""
        ...

    @property
    def capability_type(self) -> CapabilityType:
        """Type classification for this capability."""
        ...

    @property
    def supported_actions(self) -> list[str]:
        """List of action names this capability supports."""
        ...

    async def get_state(self) -> CapabilityState:
        """Retrieve the current state of this capability."""
        ...

    async def execute_action(self, action: str, params: dict[str, Any]) -> ActionResult:
        """
        Execute an action on this capability.

        Args:
            action: The action name (must be in supported_actions)
            params: Action-specific parameters

        Returns:
            ActionResult with success/failure and any return data
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize capability metadata for API responses."""
        ...


# Device-specific state classes

@dataclass
class LightState(CapabilityState):
    """State for light devices."""
    is_on: bool = False
    brightness: Optional[int] = None  # 0-255
    color_temp: Optional[int] = None  # Kelvin
    rgb_color: Optional[tuple[int, int, int]] = None


@dataclass
class SwitchState(CapabilityState):
    """State for binary switch/relay devices."""
    is_on: bool = False


@dataclass
class SensorState(CapabilityState):
    """State for sensor devices."""
    value: Any = None
    unit: Optional[str] = None


@dataclass
class CameraState(CapabilityState):
    """State for camera devices."""
    is_streaming: bool = False
    motion_detected: bool = False


@dataclass
class MediaPlayerState(CapabilityState):
    """State for media player devices."""
    is_on: bool = False
    volume_level: Optional[float] = None     # 0.0-1.0
    is_volume_muted: bool = False
    source: Optional[str] = None             # Current input source
    media_title: Optional[str] = None
    app_name: Optional[str] = None
