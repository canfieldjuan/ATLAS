"""
Vision event models for atlas_brain.

These models match the events published by atlas_vision nodes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class EventType(Enum):
    """Types of detection events."""
    NEW_TRACK = "new_track"
    TRACK_LOST = "track_lost"
    TRACK_UPDATE = "track_update"


@dataclass
class BoundingBox:
    """Normalized bounding box coordinates (0-1 range)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_dict(cls, data: dict) -> "BoundingBox":
        """Create from dictionary."""
        return cls(
            x1=data["x1"],
            y1=data["y1"],
            x2=data["x2"],
            y2=data["y2"],
        )


@dataclass
class VisionEvent:
    """A detection event from an atlas_vision node."""
    event_id: str
    event_type: EventType
    track_id: int
    class_name: str
    source_id: str  # Camera ID
    node_id: str    # Vision node ID
    timestamp: datetime
    bbox: Optional[BoundingBox] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_mqtt_payload(cls, payload: dict) -> "VisionEvent":
        """Create from MQTT payload dictionary."""
        bbox = None
        if "bbox" in payload:
            bbox = BoundingBox.from_dict(payload["bbox"])

        return cls(
            event_id=payload["event_id"],
            event_type=EventType(payload["event_type"]),
            track_id=payload["track_id"],
            class_name=payload["class"],
            source_id=payload["source_id"],
            node_id=payload["node_id"],
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            bbox=bbox,
            metadata=payload.get("metadata", {}),
        )


@dataclass
class NodeStatus:
    """Status update from an atlas_vision node."""
    node_id: str
    status: str  # "online" or "offline"
    timestamp: datetime

    @classmethod
    def from_mqtt_payload(cls, payload: dict) -> "NodeStatus":
        """Create from MQTT payload dictionary."""
        return cls(
            node_id=payload["node_id"],
            status=payload["status"],
            timestamp=datetime.fromisoformat(payload["timestamp"]),
        )
