"""
Centralized configuration management using Pydantic Settings.

Configuration is loaded from environment variables with sensible defaults.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLMConfig(BaseSettings):
    """VLM-specific configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_VLM_")

    default_model: str = Field(default="moondream", description="Default VLM to load on startup")
    moondream_cache: Path = Field(default=Path("models/moondream"), description="Cache path for moondream model")


class STTConfig(BaseSettings):
    """STT-specific configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_STT_")

    default_model: str = Field(default="nemotron", description="Default STT to load on startup")


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
        default=["light.", "switch.", "sensor.", "media_player."],
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


class LLMConfig(BaseSettings):
    """LLM (reasoning model) configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_LLM_")

    # Backend selection: "llama-cpp", "transformers-flash", "ollama", "together", "cloud", or "hybrid"
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

    # together settings (Together AI cloud API)
    together_model: str = Field(
        default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        description="Together AI model name"
    )
    together_api_key: Optional[str] = Field(
        default=None,
        description="Together AI API key (or set TOGETHER_API_KEY env var)"
    )

    # groq settings (Groq cloud API - primary for low latency)
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model name"
    )
    groq_api_key: Optional[str] = Field(
        default=None,
        description="Groq API key (or set GROQ_API_KEY env var)"
    )

    # Cloud LLM (Ollama cloud model, runs alongside local for business workflows)
    cloud_enabled: bool = Field(
        default=False,
        description="Enable cloud LLM alongside local for business workflows (booking, email)",
    )
    cloud_ollama_model: str = Field(
        default="glm-5:cloud",
        description="Ollama cloud model for business workflows (e.g., glm-5:cloud)",
    )


class TTSConfig(BaseSettings):
    """TTS configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_TTS_", env_file=".env", extra="ignore")

    default_model: str = Field(default="piper", description="Default TTS backend")
    voice: str = Field(default="en_US-ryan-medium", description="Voice model")
    speed: float = Field(default=1.0, description="Speech speed (1.0 = normal)")
    device: str | None = Field(default=None, description="Device for TTS: 'cuda', 'cpu', or None for auto")
    kokoro_lang: str = Field(default="en-us", description="Kokoro language code (en-us, en-gb, ja, zh, es, fr, hi, it, pt-br)")


class OmniConfig(BaseSettings):
    """Omni (unified speech-to-speech) configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_OMNI_", env_file=".env", extra="ignore")

    enabled: bool = Field(default=False, description="Enable unified omni mode (Qwen-Omni)")
    default_model: str = Field(default="qwen-omni", description="Default omni model")
    max_new_tokens: int = Field(default=256, description="Max tokens for response generation")
    temperature: float = Field(default=0.7, description="Sampling temperature")


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
    min_enrollment_samples: int = Field(
        default=3,
        description="Minimum voice samples needed for enrollment"
    )


class RecognitionConfig(BaseSettings):
    """Face and gait recognition configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_RECOGNITION_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable recognition services")
    face_threshold: float = Field(
        default=0.6,
        description="Face match similarity threshold (0.0-1.0)"
    )
    gait_threshold: float = Field(
        default=0.5,
        description="Gait match similarity threshold (0.0-1.0)"
    )
    use_averaged: bool = Field(
        default=True,
        description="Use averaged centroid embeddings for matching"
    )
    auto_enroll_unknown: bool = Field(
        default=True,
        description="Auto-create profiles for unknown faces"
    )
    insightface_model: str = Field(
        default="buffalo_l",
        description="InsightFace model name"
    )
    gait_sequence_length: int = Field(
        default=60,
        description="Number of frames for gait analysis"
    )
    mediapipe_detection_confidence: float = Field(
        default=0.5,
        description="MediaPipe pose detection confidence"
    )
    mediapipe_tracking_confidence: float = Field(
        default=0.5,
        description="MediaPipe pose tracking confidence"
    )
    cache_ttl: float = Field(
        default=5.0,
        description="Recognition result cache TTL in seconds"
    )
    recognition_interval: float = Field(
        default=0.5,
        description="Interval between recognition attempts in seconds"
    )
    max_tracked_persons: int = Field(
        default=10,
        description="Maximum concurrent persons to track for gait"
    )
    track_timeout: float = Field(
        default=30.0,
        description="Seconds before inactive track buffer is cleared"
    )
    iou_threshold: float = Field(
        default=0.3,
        description="Min IoU to associate pose with track bounding box"
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
    vad_aggressiveness: int = Field(default=1, description="VAD aggressiveness (0-3, higher=faster)")
    silence_duration_ms: int = Field(default=800, description="Silence to end utterance (ms)")

    # Wake word detection using OpenWakeWord
    wakeword_enabled: bool = Field(default=False, description="Enable OpenWakeWord detection")
    wakeword_threshold: float = Field(default=0.5, description="Wake word detection threshold")

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

    # Audio quality - filter ambient noise
    min_audio_rms: int = Field(
        default=200,
        description="Min audio RMS to process (filters ambient noise hallucinations)"
    )


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
        default=False,
        description="Store conversation turns in knowledge graph (deprecated: use nightly sync)",
    )
    retrieve_context: bool = Field(
        default=True,
        description="Retrieve relevant context before LLM calls",
    )
    context_results: int = Field(
        default=3,
        description="Number of context results to retrieve",
    )
    context_timeout: float = Field(
        default=3.0,
        description="Timeout in seconds for in-graph memory context retrieval",
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

    # Google OAuth token file (shared by Calendar + Gmail)
    google_token_file: str = Field(
        default="data/google_tokens.json",
        description="Path to persistent Google OAuth token file",
    )

    # Calendar tool (Google Calendar)
    calendar_enabled: bool = Field(default=False, description="Enable calendar tool")
    calendar_client_id: str | None = Field(default=None, description="Google OAuth client ID")
    calendar_client_secret: str | None = Field(default=None, description="Google OAuth client secret")
    calendar_refresh_token: str | None = Field(default=None, description="Google OAuth refresh token")
    calendar_id: str = Field(default="primary", description="Calendar ID to query")
    calendar_cache_ttl: float = Field(default=300.0, description="Cache TTL in seconds")

    # Gmail digest
    gmail_enabled: bool = Field(default=False, description="Enable Gmail digest")
    gmail_client_id: str | None = Field(default=None, description="Google OAuth client ID for Gmail")
    gmail_client_secret: str | None = Field(default=None, description="Google OAuth client secret for Gmail")
    gmail_refresh_token: str | None = Field(default=None, description="Gmail OAuth refresh token")
    gmail_max_results: int = Field(default=20, ge=1, le=50, description="Max emails per digest")
    gmail_query: str = Field(default="is:unread newer_than:1d", description="Gmail search query")


class IntentConfig(BaseSettings):
    """Intent parsing configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_INTENT_")

    # LLM settings for intent parsing
    temperature: float = Field(default=0.1, description="LLM temperature for intent parsing")
    max_tokens: int = Field(default=256, description="Max tokens for intent response")

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


class EmailConfig(BaseSettings):
    """Email tool configuration (Resend API + Gmail)."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_EMAIL_", env_file=".env", extra="ignore")

    enabled: bool = Field(default=False, description="Enable email tool")
    gmail_send_enabled: bool = Field(
        default=True,
        description="Prefer Gmail API for sending (falls back to Resend if unavailable)",
    )
    api_key: str | None = Field(default=None, description="Resend API key")
    default_from: str | None = Field(default=None, description="Default sender email address")
    timeout: int = Field(default=10, description="API timeout in seconds")
    max_recipients: int = Field(default=50, description="Maximum recipients per email")

    # Attachment settings
    proposals_dirs: list[str] = Field(
        default=[],
        description="Directories containing proposal PDFs for auto-attach (searched in order)"
    )
    attachment_whitelist_dirs: list[str] = Field(
        default=[],
        description="Directories allowed for email attachments"
    )
    max_attachment_size_mb: int = Field(
        default=10,
        description="Maximum attachment size in MB"
    )


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
    """Voice client configuration - local voice pipeline with wake word detection."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_VOICE_", env_file=".env", extra="ignore")

    enabled: bool = Field(default=True, description="Enable voice pipeline on startup")

    # Audio capture settings
    sample_rate: int = Field(default=16000, description="Audio sample rate for pipeline")
    block_size: int = Field(default=1280, description="Audio block size (80ms at 16kHz)")
    use_arecord: bool = Field(default=False, description="Use arecord instead of PortAudio")
    arecord_device: str = Field(default="default", description="ALSA device for arecord")
    input_device: str | None = Field(
        default=None,
        description="PortAudio input device index or name (e.g., 'sysdefault:CARD=SoloCast' or '13')",
    )
    audio_gain: float = Field(default=1.0, description="Software gain for microphone")

    # Wake word settings
    wakeword_model_paths: list[str] = Field(
        default=[],
        description="Paths to OpenWakeWord model files"
    )
    wake_threshold: float = Field(default=0.25, description="Wake word detection threshold")

    # ASR settings (HTTP batch mode)
    asr_url: str | None = Field(default="http://localhost:8081", description="Nemotron ASR HTTP endpoint URL")
    asr_api_key: str | None = Field(default=None, description="ASR API key if required")
    asr_timeout: int = Field(default=30, description="ASR request timeout in seconds")
    asr_model: str = Field(
        default="nvidia/nemotron-speech-streaming-en-0.6b",
        description="ASR model name or path for auto-started server",
    )
    asr_device: str = Field(default="cuda", description="Torch device for ASR server (cuda, cuda:0, cpu)")

    # ASR streaming settings (WebSocket mode)
    asr_streaming_enabled: bool = Field(
        default=True,
        description="Use WebSocket streaming ASR instead of HTTP batch mode"
    )
    asr_ws_url: str | None = Field(
        default="ws://localhost:8081/v1/asr/stream",
        description="Nemotron ASR WebSocket URL (e.g., ws://localhost:8080/v1/asr/stream)"
    )

    # TTS settings (Piper)
    piper_binary: str | None = Field(default=None, description="Path to Piper binary")
    piper_model: str | None = Field(default=None, description="Path to Piper ONNX model")
    piper_speaker: int | None = Field(default=None, description="Piper speaker ID for multi-speaker models")
    piper_length_scale: float = Field(default=1.0, description="Piper speech rate")
    piper_noise_scale: float = Field(default=0.667, description="Piper noise scale")
    piper_noise_w: float = Field(default=0.8, description="Piper noise width")
    piper_sample_rate: int = Field(default=16000, description="Piper output sample rate (from model config)")
    streaming_llm_enabled: bool = Field(
        default=True,
        description="Enable streaming LLM to TTS (speak sentences as generated)"
    )
    streaming_max_tokens: int = Field(
        default=256,
        description="Max tokens for streaming LLM responses (prevents truncation)"
    )

    # VAD and segmentation settings
    vad_aggressiveness: int = Field(default=2, description="WebRTC VAD aggressiveness (0-3)")
    silence_ms: int = Field(default=500, description="Silence duration to end utterance")
    hangover_ms: int = Field(default=300, description="Hangover time before finalizing")
    max_command_seconds: int = Field(default=5, description="Maximum command duration")
    min_command_ms: int = Field(default=1500, description="Minimum recording time before silence can finalize (grace period)")
    min_speech_frames: int = Field(default=3, description="Minimum VAD speech frames required before silence can finalize")
    wake_buffer_frames: int = Field(default=5, description="Pre-roll buffer size in frames for wake word mode (captures audio before wake word)")

    # Conversation-mode segmentation -- sliding window approach
    # The sliding window of VAD probabilities prevents premature cutoff from
    # VAD flicker during continuous speech, so silence_ms can be short (~500ms)
    # for responsive finalization once the user actually stops talking.
    conversation_silence_ms: int = Field(default=500, description="Confirmation silence duration in conversation mode")
    conversation_hangover_ms: int = Field(default=300, description="Hangover in conversation mode")
    conversation_max_command_seconds: int = Field(default=120, description="Max recording in conversation mode")
    conversation_window_frames: int = Field(default=20, description="Sliding window size for speech ratio (0=disabled)")
    conversation_silence_ratio: float = Field(default=0.15, description="Speech ratio below which silence counter engages")
    conversation_asr_holdoff_ms: int = Field(default=1000, description="Suppress finalization for N ms after last ASR partial")

    # Workflow-aware segmentation (wider patience when awaiting user input)
    workflow_silence_ms: int = Field(default=1500, description="Silence duration during active workflow")
    workflow_hangover_ms: int = Field(default=500, description="Hangover time during active workflow")
    workflow_max_command_seconds: int = Field(default=15, description="Max command duration during active workflow")
    workflow_conversation_timeout_ms: int = Field(default=120000, description="Conversation timeout during active workflow (how long to wait for user to start speaking)")

    # Interrupt settings
    stop_hotkey: bool = Field(default=True, description="Enable 's' hotkey to stop TTS")
    allow_wake_barge_in: bool = Field(default=False, description="Allow wake word to interrupt TTS")
    interrupt_on_speech: bool = Field(default=False, description="Interrupt TTS on detected speech")
    interrupt_speech_frames: int = Field(default=5, description="Frames of speech to trigger interrupt")
    interrupt_rms_threshold: float = Field(default=0.05, description="RMS threshold for speech interrupt")
    interrupt_wake_models: list[str] = Field(
        default=[],
        description="Paths to interrupt wake word models"
    )
    interrupt_wake_threshold: float = Field(default=0.5, description="Interrupt wake word threshold")

    # Processing settings
    command_workers: int = Field(default=2, description="Thread pool size for command processing")
    agent_timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for agent processing (LLM + tool execution)"
    )
    prefill_timeout: float = Field(
        default=10.0,
        description="Timeout in seconds for LLM prefill on wake word"
    )
    prefill_cache_ttl: float = Field(
        default=60.0,
        description="Seconds after last LLM call during which prefill is skipped (KV cache still warm)"
    )
    speaker_id_timeout: float = Field(
        default=5.0,
        description="Timeout in seconds for speaker identification"
    )

    # Filler phrases for slow responses
    filler_enabled: bool = Field(
        default=True,
        description="Speak a filler phrase when agent processing exceeds filler_delay_ms"
    )
    filler_delay_ms: int = Field(
        default=800,
        description="Milliseconds to wait before speaking a filler phrase"
    )
    filler_phrases: list[str] = Field(
        default=[
            "Please hold.",
            "I'll get right on that, big guy.",
            "Yes sir.",
            "Be right back.",
            "Alright super chief.",
            "Here's what I got.",
            "Let me check on that.",
            "Just a sec.",
        ],
        description="Phrases randomly chosen when agent processing is slow"
    )

    # Error recovery TTS phrases
    error_asr_empty: str = Field(
        default="Sorry, I didn't catch that.",
        description="TTS phrase when ASR returns empty transcript"
    )
    error_agent_timeout: str = Field(
        default="Sorry, that took too long. Try again.",
        description="TTS phrase when agent processing times out"
    )
    error_agent_failed: str = Field(
        default="Something went wrong. Try again.",
        description="TTS phrase when agent processing fails"
    )
    error_workflow_expired: str = Field(
        default="That session timed out. Let's start over.",
        description="TTS phrase when a multi-turn workflow expires due to inactivity"
    )

    # Tool execution
    tool_execution_timeout: float = Field(
        default=15.0,
        description="Timeout in seconds for individual tool execution"
    )

    # Debug logging
    debug_logging: bool = Field(
        default=False,
        description="Enable verbose debug logging for voice pipeline troubleshooting"
    )
    log_interval_frames: int = Field(
        default=160,
        description="Log audio stats every N frames (160 frames = ~10 seconds at 16kHz/1280 block)"
    )

    # Conversation mode settings - allow follow-ups without wake word
    conversation_mode_enabled: bool = Field(
        default=False,
        description="Enable multi-turn conversation mode (no wake word for follow-ups)"
    )
    conversation_timeout_ms: int = Field(
        default=8000,
        description="Timeout in ms to stay in conversation mode after TTS completes"
    )
    conversation_start_delay_ms: int = Field(
        default=500,
        description="Delay in ms after TTS ends before entering conversation mode (prevents echo detection)"
    )
    conversation_speech_frames: int = Field(
        default=3,
        description="Consecutive VAD speech frames required to trigger recording in conversation mode"
    )
    conversation_speech_tolerance: int = Field(
        default=2,
        description="Silence frames to tolerate before resetting speech counter (handles brief pauses)"
    )
    conversation_rms_threshold: float = Field(
        default=0.002,
        description="Minimum RMS energy to count as speech in conversation mode (lower than wake-word RMS to be more permissive)"
    )
    conversation_goodbye_phrases: list[str] = Field(
        default=["goodbye", "bye", "that's all", "thanks that's it", "nevermind"],
        description="Phrases that explicitly end conversation mode"
    )

    # Node identification for distributed deployments
    node_id: str = Field(
        default="local",
        description="Unique identifier for this voice node (e.g., 'kitchen', 'office')"
    )
    node_name: str | None = Field(
        default=None,
        description="Human-readable name for this voice node"
    )


class WebcamConfig(BaseSettings):
    """
    DEPRECATED: Webcam detection moved to atlas_vision service.

    Configure webcams via atlas_vision API instead:
        POST /cameras/register/webcam

    This config is kept for backwards compatibility but is no longer used.
    Detection now runs in atlas_vision to avoid GPU contention with voice pipeline.
    """

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_WEBCAM_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="DEPRECATED - detection moved to atlas_vision")
    device_index: int = Field(default=0, description="DEPRECATED")
    device_name: str | None = Field(default=None, description="DEPRECATED")
    source_id: str = Field(default="webcam_office", description="DEPRECATED")
    fps: int = Field(default=30, description="DEPRECATED")


class RTSPCameraConfig(BaseSettings):
    """
    DEPRECATED: RTSP camera config moved to atlas_vision service.

    Configure RTSP cameras via atlas_vision API instead:
        POST /cameras/register
    """

    camera_id: str = Field(description="DEPRECATED")
    rtsp_url: str = Field(description="DEPRECATED")
    source_id: str = Field(description="DEPRECATED")
    fps: int = Field(default=10, description="DEPRECATED")


class RTSPConfig(BaseSettings):
    """
    DEPRECATED: RTSP detection moved to atlas_vision service.

    Configure RTSP cameras via atlas_vision API instead:
        POST /cameras/register

    This config is kept for backwards compatibility but is no longer used.
    Detection now runs in atlas_vision to avoid GPU contention with voice pipeline.
    """

    model_config = SettingsConfigDict(env_prefix="ATLAS_RTSP_")

    enabled: bool = Field(default=False, description="DEPRECATED - detection moved to atlas_vision")
    wyze_bridge_host: str = Field(default="localhost", description="DEPRECATED")
    wyze_bridge_port: int = Field(default=8554, description="DEPRECATED")
    fps: int = Field(default=10, description="DEPRECATED")
    cameras_json: str = Field(default="", description="DEPRECATED")


class SecurityConfig(BaseSettings):
    """Security system configuration (video processing, cameras, zones)."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_SECURITY_", env_file=".env", extra="ignore")

    enabled: bool = Field(default=True, description="Enable security tools")
    video_processing_url: str = Field(
        default="http://localhost:5002",
        description="Video processing API URL"
    )
    timeout: float = Field(default=10.0, description="API request timeout in seconds")
    camera_aliases: dict[str, str] = Field(
        default={
            "front door": "cam_front_door",
            "front": "cam_front_door",
            "back door": "cam_back_door",
            "back": "cam_back_door",
            "backyard": "cam_backyard",
            "garage": "cam_garage",
            "driveway": "cam_driveway",
            "living room": "cam_living_room",
            "kitchen": "cam_kitchen",
        },
        description="Camera name aliases to camera IDs"
    )
    
    network_monitor_enabled: bool = Field(
        default=False, 
        description="Enable network security monitoring"
    )
    wireless_interface: str = Field(
        default="wlan0mon",
        description="WiFi interface for monitor mode"
    )
    wireless_channels: list[int] = Field(
        default=[1, 6, 11],
        description="WiFi channels to monitor"
    )
    channel_hop_interval: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Seconds between channel hops"
    )
    known_ap_bssids: list[str] = Field(
        default=[],
        description="List of legitimate AP BSSIDs"
    )
    known_ssids: list[str] = Field(
        default=[],
        description="List of legitimate SSIDs"
    )
    deauth_threshold: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Deauth frames per 10s to trigger alert"
    )
    alert_voice_enabled: bool = Field(
        default=True,
        description="Enable voice alerts for security threats"
    )
    pcap_enabled: bool = Field(
        default=True,
        description="Enable packet capture for evidence"
    )
    pcap_directory: str = Field(
        default="/var/log/atlas/security/pcap",
        description="Directory for pcap files"
    )
    pcap_max_size_mb: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Max pcap storage in MB"
    )
    
    network_ids_enabled: bool = Field(
        default=False,
        description="Enable network intrusion detection"
    )
    network_interface: str = Field(
        default="eth0",
        description="Network interface to monitor"
    )
    packet_buffer_size: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Packet buffer size"
    )
    protocols_to_monitor: list[str] = Field(
        default=["TCP", "UDP", "ICMP", "ARP"],
        description="Protocols to monitor"
    )
    port_scan_threshold: int = Field(
        default=20,
        ge=5,
        le=200,
        description="Unique ports to trigger port scan alert"
    )
    port_scan_window: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Time window for port scan detection seconds"
    )
    whitelist_ips: list[str] = Field(
        default=[],
        description="IPs to whitelist from port scan detection"
    )
    arp_monitor_enabled: bool = Field(
        default=True,
        description="Enable ARP poisoning detection"
    )
    arp_change_threshold: int = Field(
        default=3,
        ge=1,
        le=10,
        description="ARP changes to trigger alert"
    )
    known_gateways: list[str] = Field(
        default=[],
        description="Legitimate gateway IP addresses"
    )
    static_arp_entries: dict[str, str] = Field(
        default={},
        description="Trusted IP to MAC mappings"
    )
    traffic_analysis_enabled: bool = Field(
        default=True,
        description="Enable traffic anomaly detection"
    )
    baseline_period_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours to establish traffic baseline"
    )
    anomaly_threshold_sigma: float = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="Standard deviations for anomaly alert"
    )
    bandwidth_spike_multiplier: float = Field(
        default=3.0,
        ge=2.0,
        le=10.0,
        description="Multiplier for bandwidth spike detection"
    )
    asset_tracking_enabled: bool = Field(
        default=False,
        description="Enable security asset tracking"
    )
    drone_tracking_enabled: bool = Field(
        default=True,
        description="Enable drone asset tracking"
    )
    vehicle_tracking_enabled: bool = Field(
        default=True,
        description="Enable vehicle asset tracking"
    )
    sensor_tracking_enabled: bool = Field(
        default=True,
        description="Enable sensor asset tracking"
    )
    asset_stale_after_seconds: int = Field(
        default=300,
        ge=30,
        le=86400,
        description="Seconds before unseen assets become stale"
    )
    asset_max_tracked: int = Field(
        default=500,
        ge=50,
        le=50000,
        description="Maximum tracked assets per asset type"
    )



class ModeManagerConfig(BaseSettings):
    """Mode manager configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_MODE_", env_file=".env", extra="ignore")

    timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=3600,
        description="Inactivity timeout before falling back to HOME mode (seconds)"
    )
    default_mode: str = Field(
        default="home",
        description="Default mode to start in and fall back to"
    )


class IntentRouterConfig(BaseSettings):
    """Intent router configuration for semantic + LLM fallback classification."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_INTENT_ROUTER_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable intent router for fast classification")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for semantic embeddings",
    )
    confidence_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for semantic match",
    )
    llm_fallback_enabled: bool = Field(
        default=True,
        description="Use LLM when semantic confidence is too low",
    )
    llm_fallback_timeout: float = Field(
        default=2.0,
        description="Timeout in seconds for LLM fallback classification",
    )
    llm_fallback_temperature: float = Field(
        default=0.0,
        description="Temperature for LLM fallback (0.0 = deterministic)",
    )
    llm_fallback_max_tokens: int = Field(
        default=64,
        ge=16,
        le=256,
        description="Max tokens for LLM fallback classification response",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for sentence-transformer embeddings (cpu or cuda)",
    )
    conversation_confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Min confidence to skip LLM parse for conversation queries",
    )


class DeviceResolverConfig(BaseSettings):
    """Device resolver for embedding-based device name matching."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_DEVICE_RESOLVER_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable embedding-based device resolver")
    confidence_threshold: float = Field(
        default=0.45, ge=0.0, le=1.0,
        description="Min cosine similarity for device match",
    )
    ambiguity_gap: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Min score gap between top-2 matches to avoid ambiguity",
    )


class VoiceFilterConfig(BaseSettings):
    """Multi-layer voice filtering configuration for conversation mode.

    Implements a 5-layer filtering stack to reduce false triggers:
    1. Silero VAD - More accurate speech detection
    2. RMS Energy - Proximity/loudness check
    3. Speaker Continuity - Same speaker as wake word (optional)
    4. Intent Gating - Gate conversation continuation on intent confidence
    5. Turn Limit - Require wake word after N turns
    """

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_VOICE_FILTER_",
        env_file=".env",
        extra="ignore",
    )

    # Master enable
    enabled: bool = Field(default=True, description="Enable multi-layer voice filtering")

    # Layer 1: VAD backend selection
    vad_backend: str = Field(
        default="silero",
        description="VAD backend: 'silero' (accurate, recommended) or 'webrtc' (lightweight)"
    )
    silero_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Silero VAD speech probability threshold"
    )

    # Layer 2: RMS energy filtering
    rms_min_threshold: float = Field(
        default=0.004,
        ge=0.0,
        description="Minimum RMS for speech detection (filters distant conversations)"
    )
    rms_adaptive: bool = Field(
        default=False,
        description="Enable adaptive RMS threshold based on ambient noise"
    )
    rms_above_ambient_factor: float = Field(
        default=3.0,
        ge=1.0,
        description="Speech must be this factor above ambient noise floor"
    )

    # Layer 3: Speaker continuity (optional, disabled by default)
    speaker_continuity_enabled: bool = Field(
        default=False,
        description="Only accept follow-ups from same speaker as wake word"
    )
    speaker_continuity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum speaker embedding similarity for continuity"
    )

    # Layer 4: Intent gating
    intent_gating_enabled: bool = Field(
        default=True,
        description="Exit conversation mode on low intent confidence"
    )
    intent_continuation_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum intent confidence to continue conversation"
    )
    intent_categories_continue: list[str] = Field(
        default=["conversation", "tool_use", "device_command"],
        description="Intent categories that allow conversation continuation"
    )

    # Layer 5: Turn limiting (disabled by default - use other filters instead)
    turn_limit_enabled: bool = Field(
        default=False,
        description="Require wake word after max turns (not recommended)"
    )
    max_conversation_turns: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum turns before requiring wake word (if enabled)"
    )


class OpenAICompatConfig(BaseModel):
    """OpenAI-compatible endpoint configuration."""

    api_key: str = ""  # If empty, no auth required


class FTLTracingConfig(BaseModel):
    """Fine-Tune Labs tracing configuration."""

    enabled: bool = True
    base_url: str = "http://localhost:3000"
    api_key: str = ""  # wak_... key for FTL API
    user_id: str = ""  # FTL user ID for trace ownership


class PersonaConfig(BaseSettings):
    """Atlas persona and system prompt configuration.

    Single source of truth for Atlas's identity and behavior.
    All LLM-facing system prompts pull from here.
    """

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_PERSONA_",
        env_file=".env",
        extra="ignore",
    )

    system_prompt: str = Field(
        default=(
            "You are Atlas, a sharp and dependable personal assistant. "
            "You work for Juan. You know his home, his devices, his schedule, and his preferences. "
            "Be warm but direct -- no filler, no fluff, no 'Sure! I'd be happy to help.' "
            "Get to the point. Add useful context when you have it. "
            "Keep responses to 1-2 sentences unless more detail is genuinely needed."
        ),
        description="Core system prompt sent to LLM for all conversations",
    )

    name: str = Field(default="Atlas", description="Assistant name")


class AgentConfig(BaseSettings):
    """Agent system configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_AGENT_")

    backend: str = Field(
        default="langgraph",
        description="Agent backend: 'langgraph' (default)",
    )


class WorkflowConfig(BaseSettings):
    """Workflow tool backend configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_WORKFLOW_",
        env_file=".env",
        extra="ignore",
    )

    use_real_tools: bool = Field(
        default=True,
        description="Use real tool backends in workflows (false = mock responses)",
    )
    timeout_minutes: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Minutes before an inactive workflow expires and is cleared",
    )


class OrchestratedConfig(BaseSettings):
    """Orchestrated voice WebSocket endpoint configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_ORCHESTRATED_",
        env_file=".env",
        extra="ignore",
    )

    max_concurrent_sessions: int = Field(
        default=10, ge=1, le=50,
        description="Max concurrent orchestrated voice sessions",
    )
    asr_connect_timeout: float = Field(
        default=10.0, ge=1.0, le=60.0,
        description="Timeout for connecting to ASR server (seconds)",
    )
    asr_finalize_timeout: float = Field(
        default=15.0, ge=1.0, le=60.0,
        description="Timeout waiting for ASR final transcript (seconds)",
    )
    agent_timeout: float = Field(
        default=30.0, ge=5.0, le=120.0,
        description="Timeout for agent processing (seconds)",
    )
    tts_timeout: float = Field(
        default=30.0, ge=5.0, le=120.0,
        description="Timeout for TTS synthesis (seconds)",
    )


class EdgeConfig(BaseSettings):
    """Edge device WebSocket protocol configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_EDGE_",
        env_file=".env",
        extra="ignore",
    )

    max_concurrent_llm: int = Field(
        default=2, ge=1, le=10,
        description="Max concurrent LLM requests per edge connection",
    )
    token_batch_interval_ms: int = Field(
        default=50, ge=10, le=500,
        description="Token batching flush interval in milliseconds",
    )
    token_batch_max_size: int = Field(
        default=10, ge=1, le=100,
        description="Max tokens to buffer before flushing",
    )
    compression_threshold: int = Field(
        default=512, ge=0, le=65536,
        description="Min payload size in bytes before applying zlib compression (0 = always compress)",
    )
    compression_level: int = Field(
        default=1, ge=1, le=9,
        description="Zlib compression level (1=fastest, 9=smallest)",
    )


class AutonomousConfig(BaseSettings):
    """Autonomous task scheduler configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_AUTONOMOUS_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable autonomous task scheduler")
    default_agent_type: str = Field(default="atlas", description="Default agent type for headless tasks")
    default_session_prefix: str = Field(default="autonomous", description="Session ID prefix for task runs")
    max_concurrent_tasks: int = Field(default=2, ge=1, le=10, description="Max concurrent task executions")
    task_timeout_seconds: int = Field(default=120, ge=10, le=600, description="Default task timeout")
    task_history_retention_days: int = Field(default=30, ge=1, le=365, description="Execution history retention")
    hooks_enabled: bool = Field(default=True, description="Enable alert-driven hook processing")
    hook_cooldown_seconds: int = Field(default=30, ge=0, le=300, description="Min seconds between duplicate hook executions")
    default_timezone: str = Field(default="America/Chicago", description="Default timezone for scheduled tasks")

    # Event queue (Phase 3)
    event_queue_enabled: bool = Field(default=True, description="Enable event queue for debounced hook dispatch")
    event_queue_debounce_seconds: float = Field(default=5.0, ge=0.5, le=60.0, description="Debounce window before flushing queued events")
    event_queue_max_batch_size: int = Field(default=50, ge=1, le=500, description="Max events per batch flush")
    event_queue_max_age_seconds: float = Field(default=30.0, ge=1.0, le=300.0, description="Max time to hold events before forced flush")

    # Presence tracking (Phase 3)
    presence_enabled: bool = Field(default=True, description="Enable presence/occupancy state tracking")
    presence_empty_delay_seconds: int = Field(default=300, ge=30, le=1800, description="Seconds after last person_left before declaring empty")
    presence_arrival_cooldown_seconds: int = Field(default=300, ge=60, le=3600, description="Cooldown between arrival transition fires")

    # Auto-disable (Phase 4)
    auto_disable_after_failures: int = Field(default=5, ge=0, le=50, description="Disable task after N consecutive failures (0=never)")

    # LLM synthesis for builtin task results (Phase 5)
    synthesis_enabled: bool = Field(
        default=True,
        description="Enable LLM synthesis of builtin task results when synthesis_skill is set in task metadata",
    )
    synthesis_max_tokens: int = Field(
        default=1024, ge=64, le=4096,
        description="Max tokens for LLM synthesis responses",
    )
    synthesis_temperature: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="LLM temperature for synthesis (lower = more deterministic)",
    )


class EscalationConfig(BaseSettings):
    """Edge-local narration + brain-side escalation configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_ESCALATION_", env_file=".env", extra="ignore")

    enabled: bool = Field(default=True, description="Enable escalation evaluation for security events")
    unknown_empty_enabled: bool = Field(default=True, description="Escalate unknown face when house is empty")
    rapid_unknowns_threshold: int = Field(default=3, ge=2, le=10, description="Unknown face count to trigger rapid-unknowns escalation")
    rapid_unknowns_window_seconds: int = Field(default=60, ge=10, le=300, description="Sliding window for rapid unknown face detection")
    synthesis_skill: str = Field(default="security/escalation_narration", description="Skill for LLM escalation synthesis")
    synthesis_max_tokens: int = Field(default=128, ge=32, le=512, description="Max tokens for escalation narration (keep short for TTS)")
    synthesis_temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="LLM temperature for escalation")
    narration_hint_enabled: bool = Field(default=True, description="Include narration hints in security_ack")
    broadcast_occupancy: bool = Field(default=True, description="Broadcast occupancy state changes to edge nodes")


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
    load_stt_on_startup: bool = Field(default=True, description="Load STT on startup")
    load_tts_on_startup: bool = Field(default=True, description="Load TTS on startup")
    load_llm_on_startup: bool = Field(default=True, description="Load LLM on startup")

    # Startup behavior - speaker ID
    load_speaker_id_on_startup: bool = Field(
        default=False, description="Load speaker ID on startup"
    )

    # Startup behavior - VOS
    load_vos_on_startup: bool = Field(
        default=False, description="Load VOS on startup"
    )

    # Startup behavior - Omni (unified speech-to-speech)
    load_omni_on_startup: bool = Field(
        default=False, description="Load Omni (unified voice) on startup"
    )

    # Nested configs
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    omni: OmniConfig = Field(default_factory=OmniConfig)
    speaker_id: SpeakerIDConfig = Field(default_factory=SpeakerIDConfig)
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)
    vos: VOSConfig = Field(default_factory=VOSConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    homeassistant: HomeAssistantConfig = Field(default_factory=HomeAssistantConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    voice: VoiceClientConfig = Field(default_factory=VoiceClientConfig)
    webcam: WebcamConfig = Field(default_factory=WebcamConfig)
    rtsp: RTSPConfig = Field(default_factory=RTSPConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    intent: IntentConfig = Field(default_factory=IntentConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    reminder: ReminderConfig = Field(default_factory=ReminderConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    modes: ModeManagerConfig = Field(default_factory=ModeManagerConfig)
    intent_router: IntentRouterConfig = Field(default_factory=IntentRouterConfig)
    device_resolver: DeviceResolverConfig = Field(default_factory=DeviceResolverConfig)
    voice_filter: VoiceFilterConfig = Field(default_factory=VoiceFilterConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    workflows: WorkflowConfig = Field(default_factory=WorkflowConfig)
    orchestrated: OrchestratedConfig = Field(default_factory=OrchestratedConfig)
    edge: EdgeConfig = Field(default_factory=EdgeConfig)
    autonomous: AutonomousConfig = Field(default_factory=AutonomousConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)
    ftl_tracing: FTLTracingConfig = Field(default_factory=FTLTracingConfig)

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
