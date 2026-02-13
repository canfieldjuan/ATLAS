"""
Alert event types for the centralized alert system.

Defines the AlertEvent protocol and concrete event types for each
event source (vision, audio, HA state, security).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class AlertEvent(Protocol):
    """Base protocol for all alertable events."""

    event_type: str
    source_id: str
    timestamp: datetime
    metadata: dict[str, Any]

    def get_field(self, name: str, default: Any = None) -> Any:
        """Get event-specific field for rule matching."""
        ...


@dataclass
class VisionAlertEvent:
    """Vision detection event for alert processing."""

    source_id: str
    timestamp: datetime
    class_name: str
    detection_type: str
    track_id: int
    node_id: str
    event_type: str = "vision"
    event_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_field(self, name: str, default: Any = None) -> Any:
        """Get event-specific field for rule matching."""
        field_map = {
            "class_name": self.class_name,
            "detection_type": self.detection_type,
            "track_id": self.track_id,
            "node_id": self.node_id,
            "event_id": self.event_id,
        }
        return field_map.get(name, self.metadata.get(name, default))

    @classmethod
    def from_vision_event(cls, event: Any) -> "VisionAlertEvent":
        """Create from a VisionEvent object."""
        return cls(
            source_id=event.source_id,
            timestamp=event.timestamp,
            class_name=event.class_name,
            detection_type=event.event_type.value,
            track_id=event.track_id,
            node_id=event.node_id,
            event_id=event.event_id,
            metadata=event.metadata,
        )


@dataclass
class AudioAlertEvent:
    """Audio detection event for alert processing."""

    source_id: str
    timestamp: datetime
    sound_class: str
    confidence: float
    priority: str
    event_type: str = "audio"
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_field(self, name: str, default: Any = None) -> Any:
        """Get event-specific field for rule matching."""
        field_map = {
            "sound_class": self.sound_class,
            "confidence": self.confidence,
            "priority": self.priority,
        }
        return field_map.get(name, self.metadata.get(name, default))

    @classmethod
    def from_monitored_event(cls, event: Any, source_id: str = "default") -> "AudioAlertEvent":
        """Create from a MonitoredEvent object."""
        return cls(
            source_id=source_id if source_id else (event.location or "default"),
            timestamp=event.timestamp,
            sound_class=event.event.label,
            confidence=event.event.confidence,
            priority=event.priority,
            metadata={},
        )


@dataclass
class HAStateAlertEvent:
    """Home Assistant state change for alert processing."""

    source_id: str
    timestamp: datetime
    old_state: str
    new_state: str
    domain: str
    event_type: str = "ha_state"
    attributes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_field(self, name: str, default: Any = None) -> Any:
        """Get event-specific field for rule matching."""
        field_map = {
            "old_state": self.old_state,
            "new_state": self.new_state,
            "domain": self.domain,
        }
        if name in field_map:
            return field_map[name]
        if name in self.attributes:
            return self.attributes[name]
        return self.metadata.get(name, default)

    @classmethod
    def from_ha_event(cls, event_data: dict[str, Any]) -> "HAStateAlertEvent":
        """Create from Home Assistant state_changed event data."""
        entity_id = event_data.get("entity_id", "unknown")
        domain = entity_id.split(".")[0] if "." in entity_id else "unknown"

        old_state_obj = event_data.get("old_state", {}) or {}
        new_state_obj = event_data.get("new_state", {}) or {}

        return cls(
            source_id=entity_id,
            timestamp=datetime.utcnow(),
            old_state=old_state_obj.get("state", "unknown") if old_state_obj else "unknown",
            new_state=new_state_obj.get("state", "unknown") if new_state_obj else "unknown",
            domain=domain,
            attributes=new_state_obj.get("attributes", {}) if new_state_obj else {},
            metadata={},
        )


@dataclass
class ReminderAlertEvent:
    """Reminder due event for alert processing."""

    source_id: str
    timestamp: datetime
    message: str
    reminder_id: str
    event_type: str = "reminder"
    repeat_pattern: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_field(self, name: str, default: Any = None) -> Any:
        """Get event-specific field for rule matching."""
        field_map = {
            "message": self.message,
            "reminder_id": self.reminder_id,
            "repeat_pattern": self.repeat_pattern,
        }
        return field_map.get(name, self.metadata.get(name, default))

    @classmethod
    def from_reminder(cls, reminder: Any) -> "ReminderAlertEvent":
        """Create from a Reminder object."""
        return cls(
            source_id=f"reminder_{reminder.id}",
            timestamp=datetime.now(timezone.utc),
            message=reminder.message,
            reminder_id=str(reminder.id),
            repeat_pattern=reminder.repeat_pattern,
            metadata=reminder.metadata if reminder.metadata else {},
        )


@dataclass
class SecurityAlertEvent:
    """Security system event for alert processing."""

    source_id: str
    timestamp: datetime
    detection_type: str
    event_type: str = "security"
    label: Optional[str] = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_field(self, name: str, default: Any = None) -> Any:
        """Get event-specific field for rule matching."""
        field_map = {
            "detection_type": self.detection_type,
            "label": self.label,
            "confidence": self.confidence,
        }
        return field_map.get(name, self.metadata.get(name, default))

    @classmethod
    def from_kafka_event(cls, event: dict[str, Any]) -> "SecurityAlertEvent":
        """Create from Kafka security event."""
        return cls(
            source_id=event.get("camera_id", "unknown"),
            timestamp=datetime.utcnow(),
            detection_type=event.get("type", "unknown"),
            label=event.get("label"),
            confidence=event.get("confidence", 0.0),
            metadata=event,
        )


@dataclass
class PresenceAlertEvent:
    """Presence state transition event for alert processing."""

    source_id: str
    timestamp: datetime
    transition: str  # "arrival", "departure"
    occupancy_state: str  # "empty", "occupied", "identified"
    event_type: str = "presence"
    person_name: Optional[str] = None
    occupants: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_field(self, name: str, default: Any = None) -> Any:
        """Get event-specific field for rule matching."""
        field_map = {
            "transition": self.transition,
            "occupancy_state": self.occupancy_state,
            "person_name": self.person_name,
            "occupants": self.occupants,
        }
        return field_map.get(name, self.metadata.get(name, default))

    @classmethod
    def from_presence_state(
        cls,
        transition: str,
        state_value: str,
        occupants: list[str],
        person: Optional[str] = None,
        source_id: str = "presence_tracker",
    ) -> "PresenceAlertEvent":
        """Create from PresenceTracker transition data."""
        return cls(
            source_id=source_id,
            timestamp=datetime.utcnow(),
            transition=transition,
            occupancy_state=state_value,
            person_name=person,
            occupants=list(occupants),
            metadata={
                "transition": transition,
                "occupancy_state": state_value,
                "occupants": occupants,
                "person_name": person,
            },
        )
