"""
Speaker identification service.

Provides voice-based speaker enrollment and verification.
"""

from .embedder import VoiceEmbedder, get_voice_embedder
from .service import (
    SpeakerIDService,
    SpeakerMatch,
    get_speaker_id_service,
    initialize_speaker_id,
)

__all__ = [
    "VoiceEmbedder",
    "get_voice_embedder",
    "SpeakerIDService",
    "SpeakerMatch",
    "get_speaker_id_service",
    "initialize_speaker_id",
]
