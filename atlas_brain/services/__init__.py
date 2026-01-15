"""
AI model services for Atlas Brain.

This module provides:
- Protocol definitions for VLM, STT, LLM, TTS, and AudioEvent services
- Service registries for runtime model management
- Concrete implementations (moondream, faster-whisper, yamnet, etc.)
"""

from .protocols import (
    AudioEvent,
    AudioEventService,
    InferenceMetrics,
    LLMService,
    Message,
    ModelInfo,
    SegmentationResult,
    SpeakerIDService,
    SpeakerInfo,
    SpeakerMatch,
    STTService,
    TTSService,
    VLMService,
    VOSService,
)
from .registry import (
    audio_events_registry,
    llm_registry,
    register_audio_events,
    register_llm,
    register_speaker_id,
    register_stt,
    register_tts,
    register_vlm,
    register_vos,
    speaker_id_registry,
    stt_registry,
    tts_registry,
    vlm_registry,
    vos_registry,
)

# Import implementations to trigger registration
from . import vlm  # noqa: F401
from . import stt  # noqa: F401
from . import audio_events  # noqa: F401
from . import speaker_id  # noqa: F401

from . import llm  # noqa: F401
from . import tts  # noqa: F401
from . import vos  # noqa: F401

# New services
from .embedding import SentenceTransformerEmbedding
from .memory import MemoryClient, get_memory_client
from .reminders import ReminderService, get_reminder_service

__all__ = [
    # Protocols
    "VLMService",
    "STTService",
    "LLMService",
    "TTSService",
    "AudioEventService",
    "SpeakerIDService",
    "VOSService",
    "AudioEvent",
    "SpeakerInfo",
    "SpeakerMatch",
    "SegmentationResult",
    "ModelInfo",
    "InferenceMetrics",
    "Message",
    # Registries
    "vlm_registry",
    "stt_registry",
    "llm_registry",
    "tts_registry",
    "audio_events_registry",
    "speaker_id_registry",
    "vos_registry",
    # Decorators
    "register_vlm",
    "register_stt",
    "register_llm",
    "register_tts",
    "register_audio_events",
    "register_speaker_id",
    "register_vos",
    # Embedding and Memory
    "SentenceTransformerEmbedding",
    "get_embedding_service",
    "MemoryClient",
    "get_memory_client",
    # Reminders
    "ReminderService",
    "get_reminder_service",
]
