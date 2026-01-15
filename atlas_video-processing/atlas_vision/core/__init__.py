"""
Core module - Configuration, protocols, and constants.
"""

from .config import settings, Settings
from .protocols import CameraInfo, Detection, MotionEvent, SecurityZone
from .constants import DeviceType, DeviceStatus, DetectionType
from .models import (
    BoundingBox,
    TrackPoint,
    ObjectDetection,
    Track,
    TrackState,
    DetectionEvent,
    DetectionEventType,
    YOLO_CLASS_MAP,
)

__all__ = [
    "settings",
    "Settings",
    "CameraInfo",
    "Detection",
    "MotionEvent",
    "SecurityZone",
    "DeviceType",
    "DeviceStatus",
    "DetectionType",
    # Tracking models
    "BoundingBox",
    "TrackPoint",
    "ObjectDetection",
    "Track",
    "TrackState",
    "DetectionEvent",
    "DetectionEventType",
    "YOLO_CLASS_MAP",
]
