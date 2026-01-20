"""
PersonaPlex speech-to-speech service for phone call handling.

PersonaPlex is NVIDIA's 7B speech-to-speech conversational AI model
that provides natural conversation with low latency.
"""

from .config import PersonaPlexConfig, get_personaplex_config
from .service import PersonaPlexService

__all__ = [
    "PersonaPlexConfig",
    "PersonaPlexService",
    "get_personaplex_config",
]
