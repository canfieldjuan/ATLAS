"""
STT service implementations.

Import implementations here to trigger registration via decorators.
"""

from .faster_whisper import FasterWhisperSTT
from .nemotron import NemotronSTT

__all__ = ["FasterWhisperSTT", "NemotronSTT"]
