"""
Voice pipeline for Atlas Brain.

Provides local voice-to-voice capabilities:
- Wake word detection (OpenWakeWord)
- Voice Activity Detection (WebRTC VAD)
- Audio capture and playback
- Integration with Atlas agents
"""

from typing import TYPE_CHECKING, Any

from .audio_capture import AudioCapture
from .segmenter import CommandSegmenter
from .frame_processor import FrameProcessor
from .playback import PlaybackController, SpeechEngine
from .command_executor import CommandExecutor

if TYPE_CHECKING:
    from .pipeline import VoicePipeline


def pcm_to_wav_bytes(*args, **kwargs):
    from .pipeline import pcm_to_wav_bytes as _pcm_to_wav_bytes

    return _pcm_to_wav_bytes(*args, **kwargs)


def create_voice_pipeline(*args, **kwargs):
    from .launcher import create_voice_pipeline as _create_voice_pipeline

    return _create_voice_pipeline(*args, **kwargs)


def start_voice_pipeline(*args, **kwargs):
    from .launcher import start_voice_pipeline as _start_voice_pipeline

    return _start_voice_pipeline(*args, **kwargs)


def stop_voice_pipeline(*args, **kwargs):
    from .launcher import stop_voice_pipeline as _stop_voice_pipeline

    return _stop_voice_pipeline(*args, **kwargs)


def get_voice_pipeline(*args, **kwargs):
    from .launcher import get_voice_pipeline as _get_voice_pipeline

    return _get_voice_pipeline(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "VoicePipeline":
        from .pipeline import VoicePipeline as _VoicePipeline

        return _VoicePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AudioCapture",
    "CommandSegmenter",
    "FrameProcessor",
    "PlaybackController",
    "SpeechEngine",
    "CommandExecutor",
    "VoicePipeline",
    "pcm_to_wav_bytes",
    "create_voice_pipeline",
    "start_voice_pipeline",
    "stop_voice_pipeline",
    "get_voice_pipeline",
]
