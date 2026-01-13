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

    # Nemotron streaming settings
    nemotron_frame_len_ms: int = Field(default=160, description="Streaming frame length in milliseconds")
    nemotron_buffer_sec: float = Field(default=2.0, description="Total audio buffer in seconds")
    nemotron_tokens_per_chunk: int = Field(default=8, description="Tokens per chunk for RNNT decoding")
    nemotron_decoding_delay: int = Field(default=8, description="Decoding delay for latency/accuracy tradeoff")


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

    # Timeouts
    recording_timeout_ms: int = Field(default=30000, description="Max recording duration")
    processing_timeout_ms: int = Field(default=10000, description="Max processing time")


class ModelTierConfig(BaseSettings):
    """Configuration for a single model tier."""

    model_config = SettingsConfigDict(extra="ignore")

    name: str = Field(default="", description="Model identifier")
    model_path: str = Field(default="", description="Path to GGUF model file")
    complexity_threshold: float = Field(default=0.5, description="Complexity score threshold")
    max_tokens: int = Field(default=512, description="Max tokens for generation")
    temperature: float = Field(default=0.7, description="Sampling temperature")


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


class ModelRoutingConfig(BaseSettings):
    """Intelligent model routing configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_ROUTING_")

    enabled: bool = Field(default=False, description="Enable intelligent routing")

    simple_model_name: str = Field(
        default="llama-1b",
        description="Model name for simple queries"
    )
    simple_model_path: str = Field(
        default="models/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        description="Path to simple tier model"
    )
    simple_threshold: float = Field(default=0.3, description="Max complexity for simple tier")

    medium_model_name: str = Field(
        default="hermes-8b",
        description="Model name for medium queries"
    )
    medium_model_path: str = Field(
        default="models/Hermes-3-Llama-3.1-8B-GGUF/Hermes-3-Llama-3.1-8B-Q4_K_M.gguf",
        description="Path to medium tier model"
    )
    medium_threshold: float = Field(default=0.7, description="Max complexity for medium tier")

    complex_model_name: str = Field(
        default="blacksheep-24b",
        description="Model name for complex queries"
    )
    complex_model_path: str = Field(
        default="models/BlackSheep-24B-GGUF/BlackSheep-24B-Q4_K_M.gguf",
        description="Path to complex tier model"
    )

    cache_duration_seconds: int = Field(default=300, description="Keep model loaded for N seconds")


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

    model_config = SettingsConfigDict(env_prefix="ATLAS_TOOLS_")

    enabled: bool = Field(default=True, description="Enable tools system")

    # Weather tool (Open-Meteo)
    weather_enabled: bool = Field(default=True, description="Enable weather tool")
    weather_default_lat: float = Field(default=32.78, description="Default latitude")
    weather_default_lon: float = Field(default=-96.80, description="Default longitude")
    weather_units: str = Field(default="fahrenheit", description="Temperature units")

    # Traffic tool (TomTom)
    traffic_enabled: bool = Field(default=False, description="Enable traffic tool")
    traffic_api_key: str | None = Field(default=None, description="TomTom API key")


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
    routing: ModelRoutingConfig = Field(default_factory=ModelRoutingConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    voice: VoiceClientConfig = Field(default_factory=VoiceClientConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)


# Singleton settings instance
settings = Settings()
