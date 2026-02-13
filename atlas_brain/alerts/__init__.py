"""
Centralized alert system for Atlas Brain.

Provides unified alert handling for events from multiple sources:
- Vision detection events (YOLO)
- Audio events (YAMNet)
- Home Assistant state changes
- Security events (Kafka)
- Presence transitions
"""

from .delivery import (
    NtfyDelivery,
    TTSDelivery,
    WebhookDelivery,
    log_alert_callback,
    setup_default_callbacks,
)
from .events import (
    AlertEvent,
    AudioAlertEvent,
    HAStateAlertEvent,
    PresenceAlertEvent,
    ReminderAlertEvent,
    SecurityAlertEvent,
    VisionAlertEvent,
)
from .manager import (
    AlertCallback,
    AlertManager,
    get_alert_manager,
    reset_alert_manager,
)
from .rules import (
    AlertRule,
    create_audio_rule,
    create_ha_state_rule,
    create_vision_rule,
)

__all__ = [
    # Events
    "AlertEvent",
    "VisionAlertEvent",
    "AudioAlertEvent",
    "HAStateAlertEvent",
    "ReminderAlertEvent",
    "SecurityAlertEvent",
    "PresenceAlertEvent",
    # Rules
    "AlertRule",
    "create_vision_rule",
    "create_audio_rule",
    "create_ha_state_rule",
    # Manager
    "AlertManager",
    "AlertCallback",
    "get_alert_manager",
    "reset_alert_manager",
    # Delivery
    "NtfyDelivery",
    "TTSDelivery",
    "WebhookDelivery",
    "log_alert_callback",
    "setup_default_callbacks",
]
