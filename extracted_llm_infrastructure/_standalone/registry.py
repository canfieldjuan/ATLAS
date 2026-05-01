"""Standalone ``ServiceRegistry`` and ``llm_registry`` singleton for the
LLM-infrastructure package.

Mirrors ``atlas_brain.services.registry`` but lives entirely inside the
extracted package. Providers register via ``@register_llm("name")`` at
module-import time; consumers call ``llm_registry.activate("name")`` to
swap implementations at runtime.

Pure Python -- no third-party deps.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any, Callable, Generic, Optional, Type, TypeVar

from .protocols import LLMService, ModelInfo

logger = logging.getLogger("atlas.registry")

T = TypeVar("T", bound=LLMService)


class ServiceRegistry(Generic[T]):
    """Thread-safe registry for managing service instances.

    Supports:
      - Registration of implementation classes
      - Runtime activation/deactivation (hot-swapping)
      - Lazy instantiation via factories
      - Named slots for concurrent auxiliary instances
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
        self._implementations[name] = implementation
        logger.info("Registered %s implementation: %s", self._service_type, name)

    def register_factory(self, name: str, factory: Callable[..., T]) -> None:
        self._factories[name] = factory
        logger.info("Registered %s factory: %s", self._service_type, name)

    def list_available(self) -> list[str]:
        return list(set(self._implementations.keys()) | set(self._factories.keys()))

    def get_active(self) -> Optional[T]:
        return self._active

    def get_active_name(self) -> Optional[str]:
        return self._active_name

    def get_active_info(self) -> Optional[ModelInfo]:
        if self._active is not None:
            return self._active.model_info
        return None

    def activate(self, name: str, **kwargs: Any) -> T:
        if name not in self._implementations and name not in self._factories:
            available = self.list_available()
            raise ValueError(
                f"Unknown {self._service_type}: '{name}'. Available: {available}"
            )

        with self._lock:
            if self._active is not None:
                logger.info("Unloading %s: %s", self._service_type, self._active_name)
                self._active.unload()
                self._active = None
                self._active_name = None

            logger.info("Activating %s: %s", self._service_type, name)
            if name in self._factories:
                instance = self._factories[name](**kwargs)
            else:
                impl_class = self._implementations[name]
                instance = impl_class(**kwargs)

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
        with self._lock:
            if self._active is not None:
                logger.info(
                    "Deactivating %s: %s", self._service_type, self._active_name
                )
                self._active.unload()
                self._active = None
                self._active_name = None

    def is_active(self, name: str) -> bool:
        return self._active_name == name

    # ------------------------------------------------------------------
    # Named slots
    # ------------------------------------------------------------------

    def activate_slot(self, slot_name: str, impl_name: str, **kwargs: Any) -> T:
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
                self._service_type,
                slot_name,
                impl_name,
                getattr(instance, "model", getattr(instance, "model_id", "?")),
            )
            return instance

    def get_slot(self, slot_name: str) -> Optional[T]:
        return self._slots.get(slot_name)

    def release_slot(self, slot_name: str) -> None:
        with self._lock:
            instance = self._slots.pop(slot_name, None)
            if instance is not None:
                logger.info("Releasing %s slot '%s'", self._service_type, slot_name)
                instance.unload()

    def release_all_slots(self) -> None:
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
