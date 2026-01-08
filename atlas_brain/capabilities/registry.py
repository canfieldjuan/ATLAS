"""
Registry for managing capabilities (devices, integrations, etc.).

Provides centralized registration, lookup, and lifecycle management.
"""

import logging
from typing import Callable, Optional, Type

from .protocols import Capability, CapabilityType

logger = logging.getLogger("atlas.capabilities.registry")


class CapabilityRegistry:
    """
    Central registry for all capabilities in the system.

    Supports:
    - Direct instance registration
    - Factory-based lazy instantiation
    - Lookup by ID or type
    """

    _instance: Optional["CapabilityRegistry"] = None

    def __new__(cls) -> "CapabilityRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._capabilities: dict[str, Capability] = {}
            cls._instance._factories: dict[str, Callable[[], Capability]] = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> "CapabilityRegistry":
        """Get the singleton registry instance."""
        return cls()

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (mainly for testing)."""
        if cls._instance is not None:
            cls._instance._capabilities.clear()
            cls._instance._factories.clear()

    def register(self, capability: Capability) -> None:
        """Register a capability instance."""
        if capability.id in self._capabilities:
            logger.warning("Overwriting capability: %s", capability.id)
        self._capabilities[capability.id] = capability
        logger.info(
            "Registered capability: %s (%s)",
            capability.id,
            capability.capability_type.value,
        )

    def register_factory(
        self,
        capability_id: str,
        factory: Callable[[], Capability],
    ) -> None:
        """Register a factory for lazy instantiation."""
        self._factories[capability_id] = factory
        logger.info("Registered capability factory: %s", capability_id)

    def get(self, capability_id: str) -> Optional[Capability]:
        """
        Get a capability by ID.

        If the capability hasn't been instantiated but has a factory,
        the factory will be called to create it.
        """
        if capability_id not in self._capabilities and capability_id in self._factories:
            logger.info("Instantiating capability from factory: %s", capability_id)
            self._capabilities[capability_id] = self._factories[capability_id]()
        return self._capabilities.get(capability_id)

    def list_all(self) -> list[Capability]:
        """Return all registered capabilities."""
        return list(self._capabilities.values())

    def list_by_type(self, cap_type: CapabilityType) -> list[Capability]:
        """Return capabilities of a specific type."""
        return [
            c for c in self._capabilities.values()
            if c.capability_type == cap_type
        ]

    def list_ids(self) -> list[str]:
        """Return all registered capability IDs."""
        return list(self._capabilities.keys())

    def unregister(self, capability_id: str) -> bool:
        """Remove a capability from the registry."""
        if capability_id in self._capabilities:
            del self._capabilities[capability_id]
            logger.info("Unregistered capability: %s", capability_id)
            return True
        return False

    def clear(self) -> None:
        """Remove all registered capabilities."""
        self._capabilities.clear()
        logger.info("Cleared all capabilities")


# Global registry instance
capability_registry = CapabilityRegistry.get_instance()


def register_capability(capability_id: str) -> Callable[[Type[Capability]], Type[Capability]]:
    """
    Decorator to register a capability class.

    Usage:
        @register_capability("living_room_light")
        class LivingRoomLight(LightCapability):
            ...
    """
    def decorator(cls: Type[Capability]) -> Type[Capability]:
        def factory() -> Capability:
            return cls()
        capability_registry.register_factory(capability_id, factory)
        return cls
    return decorator
