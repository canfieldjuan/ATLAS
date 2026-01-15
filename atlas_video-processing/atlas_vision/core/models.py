"""
Data models for object detection and tracking.

These models represent detected objects, tracks, and events
in a format suitable for API responses and internal processing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


class TrackState(str, Enum):
    """Track lifecycle states."""

    TENTATIVE = "tentative"  # New track, not yet confirmed
    CONFIRMED = "confirmed"  # Stable track with consistent detections
    LOST = "lost"  # Track not seen recently


@dataclass
class BoundingBox:
    """Bounding box for detected object (normalized 0-1 coordinates)."""

    x1: float  # Top-left x
    y1: float  # Top-left y
    x2: float  # Bottom-right x
    y2: float  # Bottom-right y

    @property
    def center(self) -> tuple[float, float]:
        """Get center point of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        """Get width of bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Get height of bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Get area of bounding box."""
        return self.width * self.height

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "x1": round(self.x1, 4),
            "y1": round(self.y1, 4),
            "x2": round(self.x2, 4),
            "y2": round(self.y2, 4),
        }

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float, img_width: int, img_height: int) -> "BoundingBox":
        """Create from pixel coordinates, normalizing to 0-1."""
        return cls(
            x1=x1 / img_width,
            y1=y1 / img_height,
            x2=x2 / img_width,
            y2=y2 / img_height,
        )


@dataclass
class TrackPoint:
    """A single point in a track's movement history."""

    x: float  # Normalized center x
    y: float  # Normalized center y
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ObjectDetection:
    """A single object detection (no tracking)."""

    class_name: str
    confidence: float
    bbox: BoundingBox
    source_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox.to_dict(),
            "source_id": self.source_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Track:
    """A tracked object with persistent identity across frames."""

    track_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    source_id: str
    state: TrackState = TrackState.TENTATIVE
    velocity: tuple[float, float] = (0.0, 0.0)  # Normalized units per second
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    frame_count: int = 1
    path: list[TrackPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox.to_dict(),
            "source_id": self.source_id,
            "state": self.state.value,
            "velocity": [round(v, 4) for v in self.velocity],
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "frame_count": self.frame_count,
        }

    def to_dict_with_path(self) -> dict:
        """Convert to dictionary including path history."""
        data = self.to_dict()
        data["path"] = [p.to_dict() for p in self.path[-50:]]  # Last 50 points
        return data


class DetectionEventType(str, Enum):
    """Types of detection events."""

    NEW_TRACK = "new_track"  # New object started being tracked
    TRACK_LOST = "track_lost"  # Object no longer visible
    TRACK_UPDATE = "track_update"  # Regular position update
    CLASS_CHANGE = "class_change"  # Object reclassified


@dataclass
class DetectionEvent:
    """An event related to object detection/tracking."""

    event_type: DetectionEventType
    track_id: int
    class_name: str
    source_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid4())[:8])
    bbox: Optional[BoundingBox] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "track_id": self.track_id,
            "class": self.class_name,
            "source_id": self.source_id,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.bbox:
            data["bbox"] = self.bbox.to_dict()
        if self.metadata:
            data["metadata"] = self.metadata
        return data


# YOLO class ID to name mapping (COCO classes we care about)
YOLO_CLASS_MAP = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}

# Reverse mapping
YOLO_NAME_TO_ID = {v: k for k, v in YOLO_CLASS_MAP.items()}
