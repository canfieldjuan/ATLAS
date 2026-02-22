"""
AI model services for Atlas Brain.

This module provides:
- Protocol definitions for LLM services
- Service registries for runtime model management
- Concrete implementations (ollama, etc.)

Note: VLM (moondream) and VOS were removed — they are not used.
"""

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
from .reminders import ReminderService, get_reminder_service

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
