"""
Vision event handling for atlas_brain.

Receives and processes detection events from atlas_vision nodes.

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
    # Subscriber
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
