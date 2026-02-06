"""
Device implementations for the capability system.

Import implementations here to make them available.
"""

from .lights import HomeAssistantLight, MQTTLight
from .media import HomeAssistantMediaPlayer
from .mock import MockLight, MockSwitch, register_test_devices
from .switches import HomeAssistantSwitch, MQTTSwitch

__all__ = [
    "MQTTLight",
    "HomeAssistantLight",
    "HomeAssistantMediaPlayer",
    "MQTTSwitch",
    "HomeAssistantSwitch",
    "MockLight",
    "MockSwitch",
    "register_test_devices",
]
