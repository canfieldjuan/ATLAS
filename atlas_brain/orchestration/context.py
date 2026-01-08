"""
Context aggregator for Atlas Brain.

Tracks and aggregates contextual information:
- Who's in the room (face IDs, speaker IDs)
- What's visible (objects, scenes)
- Recent audio events
- Device states
- Conversation history
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger("atlas.orchestration.context")


@dataclass
class PersonContext:
    """Information about a detected person."""

    id: str  # Face ID or speaker ID
    name: Optional[str] = None  # Resolved name if known
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    location: Optional[str] = None  # Room/area
    confidence: float = 0.0

    def update_seen(self) -> None:
        """Update last seen timestamp."""
        self.last_seen = datetime.now()


@dataclass
class ObjectContext:
    """Information about a detected object."""

    label: str
    confidence: float
    location: Optional[str] = None
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    bounding_box: Optional[tuple[int, int, int, int]] = None  # x, y, w, h

    def update_seen(self, confidence: float) -> None:
        """Update last seen timestamp and confidence."""
        self.last_seen = datetime.now()
        self.confidence = confidence


@dataclass
class AudioEvent:
    """An audio event detected by YAMNet or similar."""

    label: str  # "doorbell", "dog_bark", "glass_break", etc.
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    location: Optional[str] = None


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    speaker_id: Optional[str] = None
    intent: Optional[str] = None


@dataclass
class DeviceState:
    """State of a controlled device."""

    device_id: str
    name: str
    state: dict[str, Any]
    last_updated: datetime = field(default_factory=datetime.now)


class ContextAggregator:
    """
    Aggregates and manages contextual information.

    Provides a unified view of:
    - Present people
    - Visible objects
    - Recent audio events
    - Device states
    - Conversation history
    """

    def __init__(
        self,
        person_timeout_seconds: int = 300,  # 5 minutes
        object_timeout_seconds: int = 60,
        event_history_seconds: int = 300,
        max_conversation_turns: int = 20,
    ):
        self._person_timeout = timedelta(seconds=person_timeout_seconds)
        self._object_timeout = timedelta(seconds=object_timeout_seconds)
        self._event_history = timedelta(seconds=event_history_seconds)
        self._max_conversation_turns = max_conversation_turns

        # State
        self._people: dict[str, PersonContext] = {}
        self._objects: dict[str, ObjectContext] = {}
        self._audio_events: list[AudioEvent] = []
        self._devices: dict[str, DeviceState] = {}
        self._conversation: list[ConversationTurn] = []
        self._current_room: Optional[str] = None
        self._current_time: datetime = datetime.now()

    def update_time(self) -> None:
        """Update current time and prune stale data."""
        self._current_time = datetime.now()
        self._prune_stale()

    def _prune_stale(self) -> None:
        """Remove stale entries."""
        now = self._current_time

        # Prune people
        stale_people = [
            pid for pid, p in self._people.items()
            if now - p.last_seen > self._person_timeout
        ]
        for pid in stale_people:
            del self._people[pid]

        # Prune objects
        stale_objects = [
            oid for oid, o in self._objects.items()
            if now - o.last_seen > self._object_timeout
        ]
        for oid in stale_objects:
            del self._objects[oid]

        # Prune audio events
        self._audio_events = [
            e for e in self._audio_events
            if now - e.timestamp <= self._event_history
        ]

    # People tracking

    def update_person(
        self,
        person_id: str,
        name: Optional[str] = None,
        location: Optional[str] = None,
        confidence: float = 0.0,
    ) -> PersonContext:
        """Update or add a person detection."""
        if person_id in self._people:
            person = self._people[person_id]
            person.update_seen()
            if name:
                person.name = name
            if location:
                person.location = location
            if confidence:
                person.confidence = confidence
        else:
            person = PersonContext(
                id=person_id,
                name=name,
                location=location,
                confidence=confidence,
            )
            self._people[person_id] = person
            logger.info("New person detected: %s (%s)", person_id, name or "unknown")

        return person

    def get_present_people(self) -> list[PersonContext]:
        """Get all currently present people."""
        self.update_time()
        return list(self._people.values())

    def get_person_by_name(self, name: str) -> Optional[PersonContext]:
        """Find a person by name."""
        for person in self._people.values():
            if person.name and person.name.lower() == name.lower():
                return person
        return None

    # Object tracking

    def update_object(
        self,
        label: str,
        confidence: float,
        location: Optional[str] = None,
        bounding_box: Optional[tuple[int, int, int, int]] = None,
    ) -> ObjectContext:
        """Update or add an object detection."""
        obj_key = f"{label}:{location or 'default'}"

        if obj_key in self._objects:
            obj = self._objects[obj_key]
            obj.update_seen(confidence)
        else:
            obj = ObjectContext(
                label=label,
                confidence=confidence,
                location=location,
                bounding_box=bounding_box,
            )
            self._objects[obj_key] = obj

        return obj

    def get_visible_objects(self) -> list[ObjectContext]:
        """Get all currently visible objects."""
        self.update_time()
        return list(self._objects.values())

    def is_object_visible(self, label: str) -> bool:
        """Check if an object type is currently visible."""
        self.update_time()
        return any(o.label.lower() == label.lower() for o in self._objects.values())

    # Audio events

    def add_audio_event(
        self,
        label: str,
        confidence: float,
        location: Optional[str] = None,
    ) -> AudioEvent:
        """Record an audio event."""
        event = AudioEvent(
            label=label,
            confidence=confidence,
            location=location,
        )
        self._audio_events.append(event)
        logger.info("Audio event: %s (%.2f)", label, confidence)
        return event

    def get_recent_events(self, seconds: int = 60) -> list[AudioEvent]:
        """Get audio events from the last N seconds."""
        self.update_time()
        cutoff = self._current_time - timedelta(seconds=seconds)
        return [e for e in self._audio_events if e.timestamp >= cutoff]

    # Device state

    def update_device(
        self,
        device_id: str,
        name: str,
        state: dict[str, Any],
    ) -> DeviceState:
        """Update device state."""
        device = DeviceState(
            device_id=device_id,
            name=name,
            state=state,
        )
        self._devices[device_id] = device
        return device

    def get_device_state(self, device_id: str) -> Optional[DeviceState]:
        """Get state of a specific device."""
        return self._devices.get(device_id)

    def get_all_devices(self) -> list[DeviceState]:
        """Get all device states."""
        return list(self._devices.values())

    # Conversation

    def add_conversation_turn(
        self,
        role: str,
        content: str,
        speaker_id: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> ConversationTurn:
        """Add a turn to the conversation history."""
        turn = ConversationTurn(
            role=role,
            content=content,
            speaker_id=speaker_id,
            intent=intent,
        )
        self._conversation.append(turn)

        # Trim to max length
        if len(self._conversation) > self._max_conversation_turns:
            self._conversation = self._conversation[-self._max_conversation_turns:]

        return turn

    def get_conversation_history(self, last_n: int = 10) -> list[ConversationTurn]:
        """Get recent conversation history."""
        return self._conversation[-last_n:]

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation.clear()

    # Context building

    def build_context_string(self) -> str:
        """
        Build a natural language context string for LLM input.

        Returns a formatted string describing the current context.
        """
        self.update_time()
        lines = []

        # Time
        lines.append(f"Current time: {self._current_time.strftime('%I:%M %p')}")

        # Location
        if self._current_room:
            lines.append(f"Location: {self._current_room}")

        # People
        people = self.get_present_people()
        if people:
            names = [p.name or f"Person_{p.id[:6]}" for p in people]
            lines.append(f"People present: {', '.join(names)}")

        # Objects
        objects = self.get_visible_objects()
        if objects:
            obj_labels = list(set(o.label for o in objects))
            lines.append(f"Visible objects: {', '.join(obj_labels)}")

        # Recent events
        events = self.get_recent_events(seconds=60)
        if events:
            event_labels = [e.label for e in events]
            lines.append(f"Recent sounds: {', '.join(event_labels)}")

        # Devices
        devices = self.get_all_devices()
        if devices:
            device_states = []
            for d in devices[:5]:  # Limit to 5 devices
                state_str = ", ".join(f"{k}={v}" for k, v in d.state.items())
                device_states.append(f"{d.name}: {state_str}")
            lines.append(f"Devices: {'; '.join(device_states)}")

        return "\n".join(lines)

    def build_context_dict(self) -> dict[str, Any]:
        """
        Build a structured context dictionary.

        Returns a dict suitable for LLM context injection.
        """
        self.update_time()

        return {
            "timestamp": self._current_time.isoformat(),
            "location": self._current_room,
            "people": [
                {"id": p.id, "name": p.name, "confidence": p.confidence}
                for p in self.get_present_people()
            ],
            "objects": [
                {"label": o.label, "confidence": o.confidence}
                for o in self.get_visible_objects()
            ],
            "recent_events": [
                {"label": e.label, "confidence": e.confidence, "seconds_ago": (self._current_time - e.timestamp).total_seconds()}
                for e in self.get_recent_events()
            ],
            "devices": [
                {"id": d.device_id, "name": d.name, "state": d.state}
                for d in self.get_all_devices()
            ],
            "conversation": [
                {"role": t.role, "content": t.content}
                for t in self.get_conversation_history()
            ],
        }

    def set_room(self, room: str) -> None:
        """Set the current room/location."""
        self._current_room = room


# Global context instance
_context: Optional[ContextAggregator] = None


def get_context() -> ContextAggregator:
    """Get or create the global context aggregator."""
    global _context
    if _context is None:
        _context = ContextAggregator()
    return _context
