"""
PersonaPlex configuration.

Configuration is loaded from environment variables with ATLAS_PERSONAPLEX_ prefix.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass
class PersonaPlexConfig:
    """Configuration for PersonaPlex speech-to-speech service."""

    host: str = "localhost"
    port: int = 8998
    use_ssl: bool = True
    voice_prompt: str = "NATM0"
    text_prompt: str = ""
    seed: int = 0
    connect_timeout: float = 120.0  # Server takes ~80s to initialize prompts
    read_timeout: float = 30.0


def _str_to_bool(value: str) -> bool:
    """Convert string to boolean."""
    return value.lower() in ("true", "1", "yes", "on")


import logging

logger = logging.getLogger("atlas.personaplex.config")


@lru_cache(maxsize=1)
def get_personaplex_config() -> PersonaPlexConfig:
    """Load PersonaPlex configuration from environment variables."""
    config = PersonaPlexConfig(
        host=os.environ.get("ATLAS_PERSONAPLEX_HOST", "localhost"),
        port=int(os.environ.get("ATLAS_PERSONAPLEX_PORT", "8998")),
        use_ssl=_str_to_bool(
            os.environ.get("ATLAS_PERSONAPLEX_USE_SSL", "true")
        ),
        voice_prompt=os.environ.get("ATLAS_PERSONAPLEX_VOICE", "NATM0"),
        text_prompt=os.environ.get("ATLAS_PERSONAPLEX_TEXT_PROMPT", ""),
        seed=int(os.environ.get("ATLAS_PERSONAPLEX_SEED", "0")),
        connect_timeout=float(
            os.environ.get("ATLAS_PERSONAPLEX_CONNECT_TIMEOUT", "120.0")
        ),
        read_timeout=float(
            os.environ.get("ATLAS_PERSONAPLEX_READ_TIMEOUT", "30.0")
        ),
    )
    logger.info(
        "PersonaPlex config: host=%s port=%d ssl=%s",
        config.host,
        config.port,
        config.use_ssl,
    )
    return config
