"""
AI model services for Atlas Brain.

This module provides:
- Protocol definitions for VLM and LLM services
- Service registries for runtime model management
- Concrete implementations (moondream, ollama, etc.)
"""

from .protocols import (
    InferenceMetrics,
    LLMService,
    Message,
    ModelInfo,
    SegmentationResult,
    VLMService,
    VOSService,
)
from .registry import (
    llm_registry,
    register_llm,
    register_vlm,
    register_vos,
    vlm_registry,
    vos_registry,
)

# Import implementations to trigger registration
from . import vlm  # noqa: F401
from . import llm  # noqa: F401

# New services
from .embedding import SentenceTransformerEmbedding
from .reminders import ReminderService, get_reminder_service

__all__ = [
    # Protocols
    "VLMService",
    "LLMService",
    "VOSService",
    "SegmentationResult",
    "ModelInfo",
    "InferenceMetrics",
    "Message",
    # Registries
    "vlm_registry",
    "llm_registry",
    "vos_registry",
    # Decorators
    "register_vlm",
    "register_llm",
    "register_vos",
    # Embedding
    "SentenceTransformerEmbedding",
    # Reminders
    "ReminderService",
    "get_reminder_service",
]
