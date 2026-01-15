"""
Alert rules for the centralized alert system.

Defines AlertRule which supports matching events from multiple sources
with configurable conditions, cooldowns, and message templates.
"""

import fnmatch
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

from .events import AlertEvent

logger = logging.getLogger("atlas.alerts.rules")


@dataclass
class AlertRule:
    """
    Alert rule supporting multiple event types.

    Attributes:
        name: Unique rule identifier
        event_types: List of event types to match (vision, audio, ha_state, security, or *)
        source_pattern: Pattern for source_id matching (supports fnmatch wildcards)
        conditions: Event-specific field conditions for matching
        message_template: Template for alert message with {field} placeholders
        cooldown_seconds: Minimum seconds between alerts for this rule
        enabled: Whether the rule is active
        priority: Higher values = more important, checked first
    """

    name: str
    event_types: list[str]
    source_pattern: str
    conditions: dict[str, Any] = field(default_factory=dict)
    message_template: str = "{event_type} event at {source_id}"
    cooldown_seconds: int = 30
    enabled: bool = True
    priority: int = 1

    def matches(self, event: AlertEvent) -> bool:
        """Check if an event matches this rule."""
        if not self.enabled:
            return False

        if not self._matches_event_type(event.event_type):
            return False

        if not self._matches_source(event.source_id):
            return False

        if not self._matches_conditions(event):
            return False

        return True

    def _matches_event_type(self, event_type: str) -> bool:
        """Check if event type matches."""
        if "*" in self.event_types:
            return True
        return event_type in self.event_types

    def _matches_source(self, source_id: str) -> bool:
        """Check if source_id matches the pattern."""
        if self.source_pattern == "*":
            return True
        return fnmatch.fnmatch(source_id.lower(), self.source_pattern.lower())

    def _matches_conditions(self, event: AlertEvent) -> bool:
        """Check if event fields match all conditions."""
        for field_name, expected in self.conditions.items():
            actual = event.get_field(field_name)
            if not self._check_condition(actual, expected):
                return False
        return True

    def _check_condition(self, actual: Any, expected: Any) -> bool:
        """Check a single condition value."""
        if actual is None:
            return False

        if isinstance(expected, str):
            if expected == "*":
                return True
            if isinstance(actual, str):
                return fnmatch.fnmatch(actual.lower(), expected.lower())
            return str(actual).lower() == expected.lower()

        if isinstance(expected, dict):
            return self._check_operator_condition(actual, expected)

        return actual == expected

    def _check_operator_condition(self, actual: Any, condition: dict) -> bool:
        """Check condition with operators like $gt, $lt, $in."""
        for op, value in condition.items():
            if op == "$gt" and not (actual > value):
                return False
            if op == "$gte" and not (actual >= value):
                return False
            if op == "$lt" and not (actual < value):
                return False
            if op == "$lte" and not (actual <= value):
                return False
            if op == "$in" and actual not in value:
                return False
            if op == "$nin" and actual in value:
                return False
            if op == "$ne" and actual == value:
                return False
        return True

    def format_message(self, event: AlertEvent) -> str:
        """Format the alert message with event data."""
        source_name = self._source_to_name(event.source_id)

        format_data = {
            "event_type": event.event_type,
            "source_id": event.source_id,
            "source": source_name,
            "time": event.timestamp.strftime("%H:%M"),
            "rule_name": self.name,
        }

        for key in ["class_name", "detection_type", "sound_class", "new_state",
                    "old_state", "domain", "label", "priority", "confidence",
                    "message", "reminder_id", "repeat_pattern"]:
            value = event.get_field(key)
            if value is not None:
                format_data[key] = value

        try:
            return self.message_template.format(**format_data)
        except KeyError as e:
            logger.warning("Missing field in message template: %s", e)
            return f"Alert from {source_name}"

    def _source_to_name(self, source_id: str) -> str:
        """Convert source_id to friendly name."""
        name_map = {
            "cam_front_door": "front door",
            "front_door": "front door",
            "cam_back_door": "back door",
            "cam_backyard": "backyard",
            "cam_garage": "garage",
            "cam_driveway": "driveway",
            "cam_living_room": "living room",
            "cam_kitchen": "kitchen",
        }
        if source_id in name_map:
            return name_map[source_id]

        cleaned = source_id.replace("cam_", "").replace("_", " ")
        if "." in cleaned:
            cleaned = cleaned.split(".")[-1].replace("_", " ")
        return cleaned


def create_vision_rule(
    name: str,
    source_pattern: str,
    class_name: str,
    detection_type: str = "new_track",
    message_template: str = "{class_name} detected at {source}.",
    cooldown_seconds: int = 30,
    priority: int = 5,
) -> AlertRule:
    """Helper to create a vision-specific alert rule."""
    return AlertRule(
        name=name,
        event_types=["vision"],
        source_pattern=source_pattern,
        conditions={
            "class_name": class_name,
            "detection_type": detection_type,
        },
        message_template=message_template,
        cooldown_seconds=cooldown_seconds,
        priority=priority,
    )


def create_audio_rule(
    name: str,
    source_pattern: str,
    sound_class: str,
    min_confidence: float = 0.5,
    message_template: str = "{sound_class} detected.",
    cooldown_seconds: int = 30,
    priority: int = 5,
) -> AlertRule:
    """Helper to create an audio-specific alert rule."""
    return AlertRule(
        name=name,
        event_types=["audio"],
        source_pattern=source_pattern,
        conditions={
            "sound_class": sound_class,
            "confidence": {"$gte": min_confidence},
        },
        message_template=message_template,
        cooldown_seconds=cooldown_seconds,
        priority=priority,
    )


def create_ha_state_rule(
    name: str,
    source_pattern: str,
    new_state: str,
    domain: Optional[str] = None,
    message_template: str = "{source} is now {new_state}.",
    cooldown_seconds: int = 60,
    priority: int = 3,
) -> AlertRule:
    """Helper to create a Home Assistant state-specific alert rule."""
    conditions = {"new_state": new_state}
    if domain:
        conditions["domain"] = domain
    return AlertRule(
        name=name,
        event_types=["ha_state"],
        source_pattern=source_pattern,
        conditions=conditions,
        message_template=message_template,
        cooldown_seconds=cooldown_seconds,
        priority=priority,
    )
