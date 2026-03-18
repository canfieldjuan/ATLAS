"""
Service registry for managing AI model lifecycle and hot-swapping.

Provides thread-safe registration, activation, and deactivation of
model implementations at runtime.
"""

import logging
from threading import Lock
from typing import Any, Callable, Generic, Optional, Type, TypeVar

from .protocols import LLMService, ModelInfo

logger = logging.getLogger("atlas.registry")

T = TypeVar("T", bound=LLMService)


class ServiceRegistry(Generic[T]):
    """
    Thread-safe registry for managing service instances.

    Supports:
    - Registration of implementation classes
    - Runtime activation/deactivation (hot-swapping)
    - Lazy instantiation via factories
    """

    def __init__(self, service_type: str):
        self._service_type = service_type
        self._implementations: dict[str, Type[T]] = {}
        self._factories: dict[str, Callable[..., T]] = {}
        self._active: Optional[T] = None
        self._active_name: Optional[str] = None
        self._slots: dict[str, T] = {}
        self._lock = Lock()

    def register(self, name: str, implementation: Type[T]) -> None:
        """Register an implementation class by name."""
        self._implementations[name] = implementation
        logger.info("Registered %s implementation: %s", self._service_type, name)

    def register_factory(self, name: str, factory: Callable[..., T]) -> None:
        """Register a factory function for creating instances."""
        self._factories[name] = factory
        logger.info("Registered %s factory: %s", self._service_type, name)

    def list_available(self) -> list[str]:
        """Return names of all registered implementations."""
        return list(set(self._implementations.keys()) | set(self._factories.keys()))

    def get_active(self) -> Optional[T]:
        """Return the currently active service instance."""
        return self._active

    def get_active_name(self) -> Optional[str]:
        """Return the name of the currently active service."""
        return self._active_name

    def get_active_info(self) -> Optional[ModelInfo]:
        """Return info about the active service."""
        if self._active is not None:
            return self._active.model_info
        return None

    def activate(self, name: str, **kwargs: Any) -> T:
        """
        Activate a registered implementation.

        Unloads any currently active service first to free resources.

        Args:
            name: Name of the registered implementation
            **kwargs: Additional arguments passed to the constructor/factory

        Returns:
            The activated service instance

        Raises:
            ValueError: If the implementation name is not registered
        """
        if name not in self._implementations and name not in self._factories:
            available = self.list_available()
            raise ValueError(
                f"Unknown {self._service_type}: '{name}'. Available: {available}"
            )

        with self._lock:
            # Unload current if exists
            if self._active is not None:
                logger.info("Unloading %s: %s", self._service_type, self._active_name)
                self._active.unload()
                self._active = None
                self._active_name = None

            # Instantiate new service
            logger.info("Activating %s: %s", self._service_type, name)
            if name in self._factories:
                instance = self._factories[name](**kwargs)
            else:
                impl_class = self._implementations[name]
                instance = impl_class(**kwargs)

            # Load the model
            instance.load()
            self._active = instance
            self._active_name = name

            logger.info(
                "%s activated: %s (device=%s)",
                self._service_type,
                name,
                instance.model_info.device,
            )
            return instance

    def deactivate(self) -> None:
        """Unload the currently active service to free resources."""
        with self._lock:
            if self._active is not None:
                logger.info("Deactivating %s: %s", self._service_type, self._active_name)
                self._active.unload()
                self._active = None
                self._active_name = None

    def is_active(self, name: str) -> bool:
        """Check if a specific implementation is currently active."""
        return self._active_name == name

    # ------------------------------------------------------------------
    # Named slots -- concurrent auxiliary instances that do not evict
    # the primary active service.  Used by reasoning tiering to hold
    # heavy + light models simultaneously.
    # ------------------------------------------------------------------

    def activate_slot(self, slot_name: str, impl_name: str, **kwargs: Any) -> T:
        """Create or replace a named auxiliary instance.

        Unlike ``activate()``, this does NOT touch the primary slot.
        Multiple named slots can coexist with each other and with the
        primary active service.

        Args:
            slot_name:  Logical name for the slot (e.g. ``"reasoning_heavy"``).
            impl_name:  Registered implementation to instantiate.
            **kwargs:   Passed to the constructor / factory.

        Returns:
            The loaded service instance held in the slot.
        """
        if impl_name not in self._implementations and impl_name not in self._factories:
            available = self.list_available()
            raise ValueError(
                f"Unknown {self._service_type}: '{impl_name}'. Available: {available}"
            )

        with self._lock:
            old = self._slots.get(slot_name)
            if old is not None:
                logger.info("Replacing %s slot '%s'", self._service_type, slot_name)
                old.unload()

            if impl_name in self._factories:
                instance = self._factories[impl_name](**kwargs)
            else:
                impl_class = self._implementations[impl_name]
                instance = impl_class(**kwargs)

            instance.load()
            self._slots[slot_name] = instance
            logger.info(
                "%s slot '%s' activated: %s (model=%s)",
                self._service_type, slot_name, impl_name,
                getattr(instance, "model", getattr(instance, "model_id", "?")),
            )
            return instance

    def get_slot(self, slot_name: str) -> Optional[T]:
        """Return the instance in a named slot, or None."""
        return self._slots.get(slot_name)

    def release_slot(self, slot_name: str) -> None:
        """Unload and remove a named slot."""
        with self._lock:
            instance = self._slots.pop(slot_name, None)
            if instance is not None:
                logger.info("Releasing %s slot '%s'", self._service_type, slot_name)
                instance.unload()

    def release_all_slots(self) -> None:
        """Unload and remove all named slots."""
        with self._lock:
            for name, instance in self._slots.items():
                logger.info("Releasing %s slot '%s'", self._service_type, name)
                try:
                    instance.unload()
                except Exception:
                    pass
            self._slots.clear()


# Global registry for LLM services
llm_registry: ServiceRegistry[LLMService] = ServiceRegistry("LLM")


def register_llm(name: str) -> Callable[[Type[LLMService]], Type[LLMService]]:
    """Decorator to register an LLM implementation."""
    def decorator(cls: Type[LLMService]) -> Type[LLMService]:
        llm_registry.register(name, cls)
        return cls
    return decorator
