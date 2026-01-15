"""
Device registry for Atlas Vision.

Manages registration and lookup of local devices.
"""

import logging
from typing import Optional

from ..core.constants import DeviceType
from .protocols import DeviceCapability

logger = logging.getLogger("atlas.vision.devices.registry")


class DeviceRegistry:
    """Registry for local node devices."""

    _instance: Optional["DeviceRegistry"] = None

    def __init__(self) -> None:
        self._devices: dict[str, DeviceCapability] = {}

    @classmethod
    def get_instance(cls) -> "DeviceRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, device: DeviceCapability) -> None:
        """Register a device."""
        if device.device_id in self._devices:
            logger.warning("Device %s already registered, overwriting", device.device_id)

        self._devices[device.device_id] = device
        logger.info("Registered device: %s (%s)", device.device_id, device.device_type.value)

    def unregister(self, device_id: str) -> bool:
        """Unregister a device."""
        if device_id in self._devices:
            del self._devices[device_id]
            logger.info("Unregistered device: %s", device_id)
            return True
        return False

    def get(self, device_id: str) -> Optional[DeviceCapability]:
        """Get a device by ID."""
        return self._devices.get(device_id)

    def list_all(self) -> list[DeviceCapability]:
        """List all registered devices."""
        return list(self._devices.values())

    def list_by_type(self, device_type: DeviceType) -> list[DeviceCapability]:
        """List devices by type."""
        return [d for d in self._devices.values() if d.device_type == device_type]

    def clear(self) -> None:
        """Clear all devices."""
        self._devices.clear()
        logger.info("Device registry cleared")


# Global registry instance
device_registry = DeviceRegistry.get_instance()
