"""
Protocol definitions and dataclasses for Atlas Vision.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .constants import DeviceStatus, DetectionType, SecurityZoneStatus


@dataclass
class CameraInfo:
    """Camera information."""

    camera_id: str
    name: str
    location: str
    status: DeviceStatus = DeviceStatus.UNKNOWN
    last_motion: Optional[datetime] = None
    is_recording: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.camera_id,
            "name": self.name,
            "location": self.location,
            "status": self.status.value,
            "last_motion": self.last_motion.isoformat() if self.last_motion else None,
            "is_recording": self.is_recording,
        }


@dataclass
class Detection:
    """Object detection result."""

    camera_id: str
    timestamp: datetime
    detection_type: DetectionType
    confidence: float
    label: Optional[str] = None
    bbox: Optional[tuple[int, int, int, int]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
            "type": self.detection_type.value,
            "confidence": self.confidence,
            "label": self.label,
            "bbox": self.bbox,
        }


@dataclass
class MotionEvent:
    """Motion detection event."""

    camera_id: str
    timestamp: datetime
    zone: Optional[str] = None
    confidence: float = 1.0
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
            "type": "motion",
            "zone": self.zone,
            "confidence": self.confidence,
        }


@dataclass
class SecurityZone:
    """Security zone status."""

    zone_id: str
    name: str
    status: SecurityZoneStatus = SecurityZoneStatus.DISARMED
    cameras: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "status": self.status.value,
            "cameras": self.cameras,
        }
