"""
Wake word detection service using OpenWakeWord.

Provides low-latency wake word detection that runs BEFORE
sending audio to Whisper for transcription.
"""

from .detector import WakeWordDetector, get_wakeword_detector

__all__ = ["WakeWordDetector", "get_wakeword_detector"]
