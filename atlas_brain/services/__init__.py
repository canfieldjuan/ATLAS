"""
AI model services for Atlas Brain.

This module provides:
- Protocol definitions for VLM and LLM services
- Service registries for runtime model management
- Concrete implementations (ollama, etc.)

Note: The moondream VLM implementation was removed as it is no longer in use.
      Drop a new file in atlas_brain/services/vlm/ to register a future VLM.
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

# Import LLM implementations to trigger registration
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
