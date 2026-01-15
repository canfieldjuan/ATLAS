"""
Base camera capability implementation.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from ...core.constants import DeviceType, DeviceStatus
from ...core.protocols import CameraInfo, Detection, MotionEvent


class BaseCameraCapability(ABC):
    """Base class for camera implementations."""

    device_type = DeviceType.CAMERA

    def __init__(
        self,
        device_id: str,
        name: str,
        location: str,
    ):
        self._device_id = device_id
        self._name = name
        self._location = location
        self._status = DeviceStatus.OFFLINE
        self._is_recording = False
        self._last_motion: Optional[datetime] = None

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def location(self) -> str:
        return self._location

    async def get_status(self) -> dict[str, Any]:
        """Get camera status."""
        return CameraInfo(
            camera_id=self._device_id,
            name=self._name,
            location=self._location,
            status=self._status,
            last_motion=self._last_motion,
            is_recording=self._is_recording,
        ).to_dict()

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the camera."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the camera."""
        ...

    async def start_recording(self) -> bool:
        """Start recording (override in subclass)."""
        self._is_recording = True
        return True

    async def stop_recording(self) -> bool:
        """Stop recording (override in subclass)."""
        self._is_recording = False
        return True

    async def get_current_detections(self) -> list[Detection]:
        """Get current detections (override in subclass)."""
        return []

    async def get_motion_events(self, since: Optional[datetime] = None) -> list[MotionEvent]:
        """Get motion events (override in subclass)."""
        return []
