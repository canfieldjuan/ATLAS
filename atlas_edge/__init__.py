"""
Atlas Edge - Edge device package for local voice processing.

This package enables Jetson edge devices to function independently
when the brain server is offline, handling device commands locally
while escalating complex queries to the brain when available.

Architecture:
- pipeline/: Voice pipeline with STT (Parakeet), TTS (Piper)
- intent/: Intent classification (DistilBERT) and pattern parsing
- capabilities/: Home Assistant integration and action dispatch
- brain/: Brain server connectivity and escalation
- responses/: Response templates and offline messages
"""

__version__ = "0.1.0"

from .config import settings, EdgeConfig
from .main import EdgeApplication, main

__all__ = [
    "settings",
    "EdgeConfig",
    "EdgeApplication",
    "main",
]
