"""
Vision event handling for atlas_brain.

Receives and processes detection events from atlas_vision service via MQTT.

Note: Detection (webcam, RTSP, YOLO) has been moved to atlas_vision service.
This module only handles event subscription and alert integration.

Note: Alert functionality has been moved to atlas_brain.alerts for
centralized alert handling. Imports here are for backwards compatibility.
"""

from ..alerts import AlertManager, AlertRule, get_alert_manager
from ..alerts import setup_default_callbacks as setup_alert_callbacks
from .models import BoundingBox, EventType, NodeStatus, VisionEvent
from .subscriber import (
    VisionSubscriber,
    get_vision_subscriber,
    start_vision_subscriber,
    stop_vision_subscriber,
)

__all__ = [
    # Models
    "BoundingBox",
    "EventType",
    "NodeStatus",
    "VisionEvent",
    # Subscriber (MQTT consumer for atlas_vision events)
    "VisionSubscriber",
    "get_vision_subscriber",
    "start_vision_subscriber",
    "stop_vision_subscriber",
    # Alerts (re-exported from centralized alerts)
    "AlertManager",
    "AlertRule",
    "get_alert_manager",
    "setup_alert_callbacks",
]

# Deprecation notice for removed detector imports
def __getattr__(name):
    """Provide helpful error for deprecated detector imports."""
    deprecated = {
        "WebcamPersonDetector": "atlas_vision.devices.cameras.WebcamCamera",
        "start_webcam_detector": "atlas_vision API POST /cameras/register/webcam",
        "stop_webcam_detector": "atlas_vision API DELETE /cameras/{id}",
        "get_webcam_detector": "atlas_vision API GET /cameras/{id}",
        "RTSPPersonDetector": "atlas_vision.devices.cameras.RTSPCamera",
        "RTSPDetectorManager": "atlas_vision.devices.registry",
        "get_rtsp_manager": "atlas_vision API GET /cameras",
        "start_rtsp_cameras": "atlas_vision API POST /cameras/register",
        "stop_rtsp_cameras": "atlas_vision API DELETE /cameras/{id}",
    }
    if name in deprecated:
        raise ImportError(
            f"{name} has been moved to atlas_vision service. "
            f"Use: {deprecated[name]}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
