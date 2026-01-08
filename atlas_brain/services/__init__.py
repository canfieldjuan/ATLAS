"""
AI model services for Atlas Brain.

This module provides:
- Protocol definitions for VLM, STT, LLM, and TTS services
- Service registries for runtime model management
- Concrete implementations (moondream, faster-whisper, etc.)
"""

from .protocols import (
    InferenceMetrics,
    LLMService,
    Message,
    ModelInfo,
    STTService,
    TTSService,
    VLMService,
)
from .registry import (
    llm_registry,
    register_llm,
    register_stt,
    register_tts,
    register_vlm,
    stt_registry,
    tts_registry,
    vlm_registry,
)

# Import implementations to trigger registration
from . import vlm  # noqa: F401
from . import stt  # noqa: F401

# LLM and TTS implementations loaded on-demand
# from . import llm  # noqa: F401
# from . import tts  # noqa: F401

__all__ = [
    # Protocols
    "VLMService",
    "STTService",
    "LLMService",
    "TTSService",
    "ModelInfo",
    "InferenceMetrics",
    "Message",
    # Registries
    "vlm_registry",
    "stt_registry",
    "llm_registry",
    "tts_registry",
    # Decorators
    "register_vlm",
    "register_stt",
    "register_llm",
    "register_tts",
]
