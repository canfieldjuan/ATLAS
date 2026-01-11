"""
Centralized configuration management using Pydantic Settings.

Configuration is loaded from environment variables with sensible defaults.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLMConfig(BaseSettings):
    """VLM-specific configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_VLM_")

    default_model: str = Field(default="moondream", description="Default VLM to load on startup")
    moondream_cache: Path = Field(default=Path("models/moondream"), description="Cache path for moondream model")


class STTConfig(BaseSettings):
    """STT-specific configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_STT_")

    default_model: str = Field(default="faster-whisper", description="Default STT to load on startup")
    whisper_model_size: str = Field(default="small.en", description="Whisper model size")


class MQTTConfig(BaseSettings):
    """MQTT backend configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_MQTT_")

    enabled: bool = Field(default=False, description="Enable MQTT backend")
    host: str = Field(default="localhost", description="MQTT broker host")
    port: int = Field(default=1883, description="MQTT broker port")
    username: Optional[str] = Field(default=None, description="MQTT username")
    password: Optional[str] = Field(default=None, description="MQTT password")


class HomeAssistantConfig(BaseSettings):
    """Home Assistant backend configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_HA_")

    enabled: bool = Field(default=False, description="Enable Home Assistant backend")
    url: str = Field(default="http://homeassistant.local:8123", description="Home Assistant URL")
    token: Optional[str] = Field(default=None, description="Long-lived access token")
    entity_filter: list[str] = Field(
        default=["light.", "switch.", "sensor."],
        description="Entity prefixes to auto-discover",
    )


class LLMConfig(BaseSettings):
    """LLM (reasoning model) configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_LLM_")

    default_model: str = Field(default="llama-cpp", description="Default LLM backend")
    model_path: Optional[str] = Field(default=None, description="Path to GGUF model file")
    n_ctx: int = Field(default=4096, description="Context window size")
    n_gpu_layers: int = Field(default=-1, description="GPU layers (-1 = all)")


class TTSConfig(BaseSettings):
    """TTS configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_TTS_")

    default_model: str = Field(default="piper", description="Default TTS backend")
    voice: str = Field(default="en_US-ryan-medium", description="Voice model")


class OrchestrationConfig(BaseSettings):
    """Voice pipeline orchestration configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_ORCH_")

    # Wake word
    wake_word_enabled: bool = Field(default=False, description="Enable wake word detection")
    require_wake_word: bool = Field(default=False, description="Require wake word before processing")
    wake_words: list[str] = Field(default=["hey_jarvis"], description="Wake words to detect")

    # VAD
    vad_aggressiveness: int = Field(default=2, description="VAD aggressiveness (0-3)")
    silence_duration_ms: int = Field(default=1500, description="Silence to end utterance")

    # Behavior
    auto_execute: bool = Field(default=True, description="Auto-execute device actions")

    # Timeouts
    recording_timeout_ms: int = Field(default=30000, description="Max recording duration")
    processing_timeout_ms: int = Field(default=10000, description="Max processing time")


class Settings(BaseSettings):
    """Application-wide settings."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # General
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: Path = Field(default=Path("logs"), description="Log directory")
    models_dir: Path = Field(default=Path("models"), description="Models cache directory")

    # Startup behavior
    load_vlm_on_startup: bool = Field(default=True, description="Load VLM on startup")
    load_stt_on_startup: bool = Field(default=False, description="Load STT on startup (lazy load if False)")
    load_tts_on_startup: bool = Field(default=False, description="Load TTS on startup (lazy load if False)")
    load_llm_on_startup: bool = Field(default=True, description="Load LLM on startup")

    # Nested configs
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    homeassistant: HomeAssistantConfig = Field(default_factory=HomeAssistantConfig)


# Singleton settings instance
settings = Settings()
