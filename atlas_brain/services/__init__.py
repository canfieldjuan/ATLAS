"""
AI model services for Atlas Brain.

This module provides:
- Protocol definitions for VLM and STT services
- Service registries for runtime model management
- Concrete implementations (moondream, faster-whisper)
"""

from .protocols import InferenceMetrics, ModelInfo, STTService, VLMService
from .registry import register_stt, register_vlm, stt_registry, vlm_registry

# Import implementations to trigger registration
from . import vlm  # noqa: F401
from . import stt  # noqa: F401

__all__ = [
    # Protocols
    "VLMService",
    "STTService",
    "ModelInfo",
    "InferenceMetrics",
    # Registries
    "vlm_registry",
    "stt_registry",
    # Decorators
    "register_vlm",
    "register_stt",
]
