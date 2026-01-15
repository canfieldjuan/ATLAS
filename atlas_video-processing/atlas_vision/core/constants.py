"""
Constants and enums for Atlas Vision.
"""

from enum import Enum


class DeviceType(str, Enum):
    """Types of devices supported."""

    CAMERA = "camera"
    DRONE = "drone"
    VEHICLE = "vehicle"
    SENSOR = "sensor"
    NODE = "node"


class DeviceStatus(str, Enum):
    """Device status values."""

    ONLINE = "online"
    OFFLINE = "offline"
    RECORDING = "recording"
    ERROR = "error"
    UNKNOWN = "unknown"


class DetectionType(str, Enum):
    """Types of detections."""

    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    MOTION = "motion"
    PACKAGE = "package"
    UNKNOWN = "unknown"


class SecurityZoneStatus(str, Enum):
    """Security zone status."""

    ARMED = "armed"
    DISARMED = "disarmed"
    TRIGGERED = "triggered"
