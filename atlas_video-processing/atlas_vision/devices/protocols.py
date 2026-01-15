"""
Device capability protocols.

Mirrors atlas_brain/capabilities/protocols.py pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable

from ..core.constants import DeviceType, DeviceStatus


@runtime_checkable
class DeviceCapability(Protocol):
    """Protocol for all device capabilities."""

    @property
    def device_id(self) -> str:
        """Unique device identifier."""
        ...

    @property
    def device_type(self) -> DeviceType:
        """Type of device."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name."""
        ...

    async def get_status(self) -> dict[str, Any]:
        """Get current device status."""
        ...


@runtime_checkable
class CameraCapability(DeviceCapability, Protocol):
    """Protocol for camera devices."""

    @property
    def location(self) -> str:
        """Camera location."""
        ...

    async def get_current_detections(self) -> list:
        """Get current detections from this camera."""
        ...

    async def get_motion_events(self, since: Optional[Any] = None) -> list:
        """Get motion events from this camera."""
        ...

    async def start_recording(self) -> bool:
        """Start recording."""
        ...

    async def stop_recording(self) -> bool:
        """Stop recording."""
        ...
