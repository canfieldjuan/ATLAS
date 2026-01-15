"""
Devices module - Device capabilities and registry.
"""

from .protocols import DeviceCapability
from .registry import DeviceRegistry, device_registry

__all__ = [
    "DeviceCapability",
    "DeviceRegistry",
    "device_registry",
]
