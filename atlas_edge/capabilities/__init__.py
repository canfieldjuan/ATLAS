"""
Device control capabilities for edge devices.

Includes Home Assistant REST client and local action dispatcher.
"""

from .homeassistant import EdgeHomeAssistant
from .dispatcher import LocalActionDispatcher

__all__ = [
    "EdgeHomeAssistant",
    "LocalActionDispatcher",
]
