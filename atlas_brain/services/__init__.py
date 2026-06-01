"""
AI model services for Atlas Brain.

This module provides:
- Protocol definitions for LLM services
- Service registries for runtime model management
- Concrete implementations (ollama, etc.)

Note: VLM (moondream) and VOS were removed — they are not used.
"""

from importlib import import_module
from typing import Any

from .protocols import (
    InferenceMetrics,
    LLMService,
    Message,
    ModelInfo,
)
from .registry import (
    llm_registry,
    register_llm,
)

# Import LLM implementations to trigger registration
from . import llm  # noqa: F401

# New services
from .embedding import SentenceTransformerEmbedding

__all__ = [
    # Protocols
    "LLMService",
    "ModelInfo",
    "InferenceMetrics",
    "Message",
    # Registries
    "llm_registry",
    # Decorators
    "register_llm",
    # Embedding
    "SentenceTransformerEmbedding",
    # Reminders
    "ReminderService",
    "get_reminder_service",
]


def __getattr__(name: str) -> Any:
    if name in {"ReminderService", "get_reminder_service"}:
        module = import_module(".reminders", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
