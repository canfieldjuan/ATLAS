"""
Capability system for device and integration management.

This module provides:
- Protocol definitions for capabilities and devices
- Registry for capability lifecycle management
- Action dispatch framework for executing actions
"""

from .actions import ActionDispatcher, ActionRequest, Intent, action_dispatcher
from .protocols import (
    ActionResult,
    CameraState,
    Capability,
    CapabilityState,
    CapabilityType,
    LightState,
    MediaPlayerState,
    SensorState,
    SwitchState,
)
from .registry import CapabilityRegistry, capability_registry, register_capability

__all__ = [
    # Protocols
    "Capability",
    "CapabilityType",
    "CapabilityState",
    "ActionResult",
    # State types
    "LightState",
    "SwitchState",
    "SensorState",
    "CameraState",
    "MediaPlayerState",
    # Registry
    "CapabilityRegistry",
    "capability_registry",
    "register_capability",
    # Actions
    "ActionDispatcher",
    "ActionRequest",
    "Intent",
    "action_dispatcher",
]
