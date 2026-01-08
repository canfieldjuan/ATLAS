"""
Communication backends for device capabilities.

Backends handle the actual communication with devices/services
(MQTT, HTTP, Home Assistant API, GPIO, etc.)
"""

from .base import Backend

__all__ = ["Backend"]
