"""
Voice pipeline for Atlas Brain.

Provides local voice-to-voice capabilities:
- Wake word detection (OpenWakeWord)
- Voice Activity Detection (WebRTC VAD)
- Audio capture and playback
- Integration with Atlas agents
"""

from .audio_capture import AudioCapture
from .segmenter import CommandSegmenter
from .frame_processor import FrameProcessor
from .playback import PlaybackController, SpeechEngine
from .command_executor import CommandExecutor
from .pipeline import VoicePipeline, pcm_to_wav_bytes
from .launcher import (
    create_voice_pipeline,
    start_voice_pipeline,
    stop_voice_pipeline,
    get_voice_pipeline,
)

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
