"""
Device implementations for the capability system.

Import implementations here to make them available.
"""

from .lights import HomeAssistantLight, MQTTLight
from .switches import HomeAssistantSwitch, MQTTSwitch

__all__ = [
    "MQTTLight",
    "HomeAssistantLight",
    "MQTTSwitch",
    "HomeAssistantSwitch",
]
