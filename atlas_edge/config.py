"""
Configuration management for Atlas Edge devices.

Uses Pydantic Settings for environment variable parsing.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BrainConfig(BaseSettings):
    """Brain server connection configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_BRAIN_")

    url: str = Field(
        default="ws://localhost:8000",
        description="Brain server WebSocket URL",
    )
    enabled: bool = Field(
        default=True,
        description="Enable brain connectivity",
    )
    reconnect_interval: int = Field(
        default=5,
        description="Seconds between reconnection attempts",
    )
    health_check_interval: int = Field(
        default=30,
        description="Seconds between health checks",
    )
    escalation_timeout: float = Field(
        default=5.0,
        description="Seconds to wait for brain response before fallback",
    )


class HomeAssistantConfig(BaseSettings):
    """Home Assistant connection configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_HA_")

    url: str = Field(
        default="http://homeassistant.local:8123",
        description="Home Assistant URL",
    )
    token: Optional[str] = Field(
        default=None,
        description="Long-lived access token",
    )
    enabled: bool = Field(
        default=True,
        description="Enable Home Assistant integration",
    )


class STTConfig(BaseSettings):
    """Speech-to-text configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_EDGE_STT_")

    model_name: str = Field(
        default="nvidia/parakeet-tdt-0.6b",
        description="Parakeet model name for STT",
    )
    device: str = Field(
        default="cuda",
        description="Device for inference: cuda, cpu",
    )
    compute_type: str = Field(
        default="float16",
        description="Compute type for inference",
    )
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate",
    )


class TTSConfig(BaseSettings):
    """Text-to-speech configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_EDGE_TTS_")

    voice: str = Field(
        default="en_US-ryan-medium",
        description="Piper voice model",
    )
    speed: float = Field(
        default=1.0,
        description="Speech speed (1.0 = normal)",
    )
    model_path: Optional[Path] = Field(
        default=None,
        description="Path to Piper voice model (auto-downloaded if None)",
    )


class IntentConfig(BaseSettings):
    """Intent classification configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_EDGE_INTENT_")

    model_id: str = Field(
        default="qanastek/XLMRoberta-Alexa-Intents-Classification",
        description="DistilBERT model for intent classification",
    )
    device: str = Field(
        default="cuda",
        description="Device for inference: cuda, cpu, auto",
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence for local handling",
    )


class VADConfig(BaseSettings):
    """Voice activity detection configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_EDGE_VAD_")

    aggressiveness: int = Field(
        default=1,
        description="VAD aggressiveness (0-3, higher = faster)",
    )
    silence_duration_ms: int = Field(
        default=800,
        description="Silence duration to end utterance (ms)",
    )
    frame_duration_ms: int = Field(
        default=30,
        description="Frame duration for VAD (10, 20, or 30 ms)",
    )


class WakeWordConfig(BaseSettings):
    """Wake word detection configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_EDGE_WAKEWORD_")

    enabled: bool = Field(
        default=True,
        description="Enable wake word detection",
    )
    keyword: str = Field(
        default="hey_atlas",
        description="Wake word model name",
    )
    threshold: float = Field(
        default=0.5,
        description="Wake word detection threshold",
    )


class EdgeConfig(BaseSettings):
    """Main edge device configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_EDGE_",
        env_file=".env",
        extra="ignore",
    )

    location_id: str = Field(
        default="default",
        description="Location identifier for this edge device",
    )
    location_name: str = Field(
        default="Home",
        description="Human-readable location name",
    )

    # Sub-configs
    brain: BrainConfig = Field(default_factory=BrainConfig)
    homeassistant: HomeAssistantConfig = Field(default_factory=HomeAssistantConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    intent: IntentConfig = Field(default_factory=IntentConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    wakeword: WakeWordConfig = Field(default_factory=WakeWordConfig)


# Global settings instance
_settings: Optional[EdgeConfig] = None


def get_settings() -> EdgeConfig:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = EdgeConfig()
    return _settings


settings = get_settings()
