"""Reasoning Agent -- cross-domain event-driven intelligence layer."""

from .config import ReasoningConfig
from .events import AtlasEvent, EventType, emit_event
from .event_bus import EventBus
from .entity_locks import EntityLockManager

__all__ = [
    "ReasoningConfig",
    "AtlasEvent",
    "EventType",
    "emit_event",
    "EventBus",
    "EntityLockManager",
]
