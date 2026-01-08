"""
STT service implementations.

Import implementations here to trigger registration via decorators.
"""

from .faster_whisper import FasterWhisperSTT

__all__ = ["FasterWhisperSTT"]
