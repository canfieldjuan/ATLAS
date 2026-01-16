"""
Centralized configuration management using Pydantic Settings.

Configuration is loaded from environment variables with sensible defaults.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
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

    # WebSocket settings for real-time state
    websocket_enabled: bool = Field(
        default=True,
        description="Enable WebSocket for real-time state updates",
    )
    websocket_reconnect_interval: int = Field(
        default=5,
        description="Seconds between WebSocket reconnection attempts",
    )
    state_cache_ttl: int = Field(
        default=300,
        description="Seconds to cache entity state before considering stale",
    )


class RokuConfig(BaseSettings):
    """Roku device configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_ROKU_")

    enabled: bool = Field(default=False, description="Enable Roku backend")
    devices: list[dict] = Field(
        default=[],
        description="List of Roku devices [{host: str, name: str}]",
    )


class LLMConfig(BaseSettings):
    """LLM (reasoning model) configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_LLM_")

    # Backend selection: "llama-cpp", "transformers-flash", or "ollama"
    default_model: str = Field(default="llama-cpp", description="Default LLM backend")

    # llama-cpp settings (GGUF models)
    model_path: Optional[str] = Field(default=None, description="Path to GGUF model file")
    n_ctx: int = Field(default=4096, description="Context window size")
    n_gpu_layers: int = Field(default=-1, description="GPU layers (-1 = all)")

    # ollama settings (Ollama API backend)
    ollama_model: str = Field(default="qwen3-coder:30b", description="Ollama model name")
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama API URL")

    # transformers-flash settings (HuggingFace models)
    hf_model_id: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="HuggingFace model ID for transformers backend"
    )
    torch_dtype: str = Field(
        default="bfloat16",
        description="Torch dtype: bfloat16, float16, or auto"
    )
    use_flash_attention: bool = Field(
        default=True,
        description="Use Flash Attention 2 if available"
    )
    max_memory_gb: Optional[float] = Field(
        default=None,
        description="Max GPU memory in GB (None = no limit)"
    )


class TTSConfig(BaseSettings):
    """TTS configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_TTS_", env_file=".env", extra="ignore")

    default_model: str = Field(default="piper", description="Default TTS backend")
    voice: str = Field(default="en_US-ryan-medium", description="Voice model")
    speed: float = Field(default=1.0, description="Speech speed (1.0 = normal)")


class SpeakerIDConfig(BaseSettings):
    """Speaker identification configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_SPEAKER_ID_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable speaker identification")
    default_model: str = Field(default="resemblyzer", description="Speaker ID backend")
    require_known_speaker: bool = Field(
        default=False,
        description="Only respond to enrolled speakers"
    )
    unknown_speaker_response: str = Field(
        default="I don't recognize your voice. Please ask the owner to enroll you.",
        description="Response when unknown speaker detected"
    )
    confidence_threshold: float = Field(
        default=0.75,
        description="Minimum confidence for speaker match (0.0-1.0)"
    )


class VOSConfig(BaseSettings):
    """VOS (Video Object Segmentation) configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_VOS_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable VOS service")
    default_model: str = Field(default="sam3", description="Default VOS model")
    device: str = Field(default="cuda", description="Device for inference")
    dtype: str = Field(default="float16", description="Model dtype")
    bpe_path: Optional[str] = Field(
        default=None,
        description="Path to BPE vocab file (auto-detected if None)"
    )
    load_from_hf: bool = Field(
        default=True,
        description="Load model from HuggingFace"
    )


class OrchestrationConfig(BaseSettings):
    """Voice pipeline orchestration configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_ORCH_",
        env_file=".env",
        extra="ignore",
    )

    # VAD - Lower values = faster response, but may cut off speech
    vad_aggressiveness: int = Field(default=3, description="VAD aggressiveness (0-3, higher=faster)")
    silence_duration_ms: int = Field(default=800, description="Silence to end utterance (ms)")

    # Keyword detection - only respond when keyword is in transcript
    keyword_enabled: bool = Field(default=True, description="Enable keyword detection")
    keyword: str = Field(default="atlas", description="Keyword to listen for (case-insensitive)")

    # Behavior
    auto_execute: bool = Field(default=True, description="Auto-execute device actions")

    # Follow-up mode: stay "hot" after response for quick follow-up commands
    follow_up_enabled: bool = Field(default=True, description="Enable follow-up mode (no wake word after response)")
    follow_up_duration_ms: int = Field(default=20000, description="Follow-up window duration (ms)")
    
    # Progressive prompting: prefill LLM with partial transcripts during speech
    progressive_prompting_enabled: bool = Field(
        default=True,
        description="Enable progressive prompting (prefill LLM during speech)"
    )
    progressive_interval_ms: int = Field(
        default=500,
        description="Interval between interim transcriptions (ms)"
    )

    # Timeouts
    recording_timeout_ms: int = Field(default=30000, description="Max recording duration")
    processing_timeout_ms: int = Field(default=10000, description="Max processing time")


class DiscoveryConfig(BaseSettings):
    """Network device discovery configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_DISCOVERY_")

    enabled: bool = Field(default=True, description="Enable device discovery")
    scan_on_startup: bool = Field(default=True, description="Scan network on startup")
    scan_interval_seconds: int = Field(default=300, description="Periodic scan interval (0=disabled)")
    ssdp_enabled: bool = Field(default=True, description="Enable SSDP scanning")
    mdns_enabled: bool = Field(default=False, description="Enable mDNS scanning (future)")
    auto_register: bool = Field(default=True, description="Auto-register discovered devices")
    persist_devices: bool = Field(default=True, description="Save devices to database")
    scan_timeout: float = Field(default=5.0, description="Scan timeout in seconds")


class MemoryConfig(BaseSettings):
    """Long-term memory configuration (atlas-memory integration)."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_MEMORY_")

    enabled: bool = Field(default=True, description="Enable memory service")
    base_url: str = Field(
        default="http://localhost:8001",
        description="URL of the atlas-memory (graphiti-wrapper) service",
    )
    group_id: str = Field(
        default="atlas-conversations",
        description="Default group ID for conversation storage",
    )
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    store_conversations: bool = Field(
        default=True,
        description="Store conversation turns in knowledge graph",
    )
    retrieve_context: bool = Field(
        default=True,
        description="Retrieve relevant context before LLM calls",
    )
    context_results: int = Field(
        default=3,
        description="Number of context results to retrieve",
    )

    # Nightly sync settings - batch processing for long-term memory
    nightly_sync_enabled: bool = Field(
        default=True,
        description="Enable nightly batch sync of conversations to GraphRAG",
    )
    purge_days: int = Field(
        default=30,
        description="Purge PostgreSQL messages older than N days",
    )
    similarity_threshold: float = Field(
        default=0.85,
        description="Skip facts with embedding similarity > this threshold (deduplication)",
    )


class ToolsConfig(BaseSettings):
    """Configuration for Atlas tools (weather, traffic, etc.)."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_TOOLS_", env_file=".env", extra="ignore")

    enabled: bool = Field(default=True, description="Enable tools system")

    # Weather tool (Open-Meteo)
    weather_enabled: bool = Field(default=True, description="Enable weather tool")
    weather_default_lat: float = Field(default=32.78, description="Default latitude")
    weather_default_lon: float = Field(default=-96.80, description="Default longitude")
    weather_units: str = Field(default="fahrenheit", description="Temperature units")

    # Traffic tool (TomTom)
    traffic_enabled: bool = Field(default=False, description="Enable traffic tool")
    traffic_api_key: str | None = Field(default=None, description="TomTom API key")

    # Calendar tool (Google Calendar)
    calendar_enabled: bool = Field(default=False, description="Enable calendar tool")
    calendar_client_id: str | None = Field(default=None, description="Google OAuth client ID")
    calendar_client_secret: str | None = Field(default=None, description="Google OAuth client secret")
    calendar_refresh_token: str | None = Field(default=None, description="Google OAuth refresh token")
    calendar_id: str = Field(default="primary", description="Calendar ID to query")
    calendar_cache_ttl: float = Field(default=300.0, description="Cache TTL in seconds")


class IntentConfig(BaseSettings):
    """Intent parsing configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_INTENT_")

    # LLM settings for intent parsing
    temperature: float = Field(default=0.1, description="LLM temperature for intent parsing")
    max_tokens: int = Field(default=80, description="Max tokens for intent response")

    # Device cache settings
    device_cache_ttl: int = Field(default=60, description="Device list cache TTL in seconds")

    # Available tools (can be extended via config)
    available_tools: list[str] = Field(
        default=["time", "weather", "traffic", "location"],
        description="List of available tool names",
    )


class AlertsConfig(BaseSettings):
    """Centralized alerts configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_ALERTS_", env_file=".env", extra="ignore")

    enabled: bool = Field(default=True, description="Enable centralized alert system")
    default_cooldown_seconds: int = Field(default=30, description="Default cooldown between alerts")
    tts_enabled: bool = Field(default=True, description="Enable TTS announcements for alerts")
    persist_alerts: bool = Field(default=True, description="Persist alerts to database")
    ntfy_enabled: bool = Field(default=False, description="Enable ntfy push notifications")
    ntfy_url: str = Field(default="http://localhost:8090", description="ntfy server URL")
    ntfy_topic: str = Field(default="atlas-alerts", description="ntfy topic for alerts")


class ReminderConfig(BaseSettings):
    """Reminder system configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_REMINDER_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable reminder system")
    default_timezone: str = Field(default="America/Chicago", description="Default timezone for parsing")
    max_reminders_per_user: int = Field(default=100, ge=1, le=1000, description="Max active reminders per user")
    scheduler_check_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=3600.0,
        description="How often to check for due reminders (0.1s - 1hr)"
    )

    @field_validator("default_timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate that the timezone string is a valid IANA timezone."""
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(v)
            return v
        except Exception:
            # Try pytz as fallback (if installed)
            try:
                import pytz
                pytz.timezone(v)
                return v
            except Exception:
                pass
            raise ValueError(
                f"Invalid timezone: '{v}'. Use IANA timezone names like "
                "'America/New_York', 'Europe/London', 'UTC'"
            )


class VoiceClientConfig(BaseSettings):
    """Voice client configuration - unified audio capture and playback."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_VOICE_")

    enabled: bool = Field(default=True, description="Enable voice client on startup")
    input_device: int | None = Field(default=None, description="Audio input device index")
    output_device: int | None = Field(default=None, description="Audio output device index")
    sample_rate: int = Field(default=16000, description="Audio sample rate")


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

    # Startup behavior - speaker ID
    load_speaker_id_on_startup: bool = Field(
        default=False, description="Load speaker ID on startup"
    )

    # Startup behavior - VOS
    load_vos_on_startup: bool = Field(
        default=False, description="Load VOS on startup"
    )

    # Nested configs
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    speaker_id: SpeakerIDConfig = Field(default_factory=SpeakerIDConfig)
    vos: VOSConfig = Field(default_factory=VOSConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    homeassistant: HomeAssistantConfig = Field(default_factory=HomeAssistantConfig)
    roku: RokuConfig = Field(default_factory=RokuConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    voice: VoiceClientConfig = Field(default_factory=VoiceClientConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    intent: IntentConfig = Field(default_factory=IntentConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    reminder: ReminderConfig = Field(default_factory=ReminderConfig)

    # Presence tracking - imported from presence module
    @property
    def presence_enabled(self) -> bool:
        """Check if presence tracking is enabled."""
        try:
            from .presence.config import presence_config
            return presence_config.enabled
        except ImportError:
            return False


# Singleton settings instance
settings = Settings()
