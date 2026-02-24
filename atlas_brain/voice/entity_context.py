"""
Entity context extraction for voice pipeline turns.

Extracts structured entity references from turn metadata and formats
them for injection into the next turn's system prompt.
No LLM calls -- purely rule-based extraction from structured sources.
"""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class EntityRef:
    type: str       # "device" | "person" | "location" | "topic"
    name: str
    action: str = ""
    source: str = ""  # "command" | "speaker" | "tool" | "text"


def extract_location_from_text(text: str) -> Optional[str]:
    """Extract a location name from natural language using simple regex patterns."""
    patterns = [
        r"\bweather in ([A-Z][a-zA-Z\s]+?)(?:\s+is|\s+today|\s+right|\.|,|$)",
        r"\bin ([A-Z][a-zA-Z\s]+?)(?:\s+right now|\s+today|\s+is|\.|,|$)",
        r"\bat ([A-Z][a-zA-Z\s]{2,30}?)(?:\s+the|\s+is|\.|,|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            location = m.group(1).strip()
            if 2 < len(location) < 40:
                return location
    return None


# Ordered list of (topic_name, trigger_keywords) for streaming transcript matching.
# Keywords are matched case-insensitively as substrings. Order matters: first match wins.
_TOPIC_KEYWORDS: list[tuple[str, list[str]]] = [
    ("weather",   ["weather", "temperature", "forecast", "rain", "sunny", "cloudy", "snow", "humidity"]),
    ("reminder",  ["remind me", "reminder", "don't forget", "remember to", "set a reminder"]),
    ("calendar",  ["calendar", "schedule", "appointment", "meeting", "event", "on my schedule"]),
    ("traffic",   ["traffic", "commute", "how long to drive", "road conditions"]),
    ("email",     ["send an email", "send a message", "email to", "write to"]),
    ("booking",   ["book a", "make a reservation", "reserve a"]),
    ("call",      ["call someone", "make a call", "phone", "dial", "ring", "give them a call"]),
    ("time",      ["what time is it", "current time", "what's the time"]),
    ("camera",    ["show the camera", "camera feed", "show me the feed"]),
    ("location",  ["where am i", "what's my location", "current location"]),
]


def extract_topic_from_text(text: str) -> Optional[str]:
    """
    Extract a topic name from natural language using keyword matching.

    Checks user transcript for known tool trigger phrases. Used in the streaming
    path where no intent object is available. Returns the first matching topic name.
    """
    lower = text.lower()
    for topic, keywords in _TOPIC_KEYWORDS:
        for kw in keywords:
            if kw in lower:
                return topic
    return None


def collect_recent_entities(entity_dicts: list[dict], limit: int = 3) -> list[EntityRef]:
    """
    Collect deduplicated entities from a flat list of stored entity dicts.

    Deduplication: keep the first occurrence of each (type, name.lower()) pair.
    """
    seen: set[tuple[str, str]] = set()
    result: list[EntityRef] = []
    for e in entity_dicts:
        name = e.get("name", "")
        etype = e.get("type", "unknown")
        if not name:
            continue
        key = (etype, name.lower())
        if key not in seen:
            seen.add(key)
            result.append(EntityRef(
                type=etype,
                name=name,
                action=e.get("action", ""),
                source=e.get("source", ""),
            ))
        if len(result) >= limit * 4:  # cap: max 4 entities per turn * 3 turns
            break
    return result


def format_entity_context(entities: list[EntityRef]) -> Optional[str]:
    """Format entity list as a compact system prompt section. Returns None if empty."""
    if not entities:
        return None

    by_type: dict[str, list[str]] = {}
    for e in entities:
        label = e.name
        if e.action:
            label += f" ({e.action})"
        by_type.setdefault(e.type, []).append(label)

    lines = ["Recently mentioned:"]
    for etype in ("device", "person", "location", "topic"):
        if etype in by_type:
            lines.append(f"- {etype}: {', '.join(by_type[etype])}")

    return "\n".join(lines) if len(lines) > 1 else None
