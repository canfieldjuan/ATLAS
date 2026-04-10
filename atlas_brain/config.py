"""
Centralized configuration management using Pydantic Settings.

Configuration is loaded from environment variables with sensible defaults.
"""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILES = (".env", ".env.local")


class SaaSAuthConfig(BaseSettings):
    """SaaS authentication and billing configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_SAAS_", env_file=ENV_FILES, extra="ignore")

    enabled: bool = Field(default=False, description="Enable SaaS auth (when off, dashboard works without auth)")
    jwt_secret: str = Field(default="change-me-in-production", description="JWT signing secret")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiry_hours: int = Field(default=24, description="Access token expiry in hours")
    jwt_refresh_expiry_days: int = Field(default=30, description="Refresh token expiry in days")
    trial_days: int = Field(default=14, description="Trial period length in days")
    frontend_base_url: str = Field(
        default="http://localhost:5173",
        description="Frontend base URL used in SaaS auth emails",
    )
    email_product_name: str = Field(
        default="Churn Signals",
        description="Display product name used in SaaS auth emails",
    )
    email_company_name: str = Field(
        default="Atlas Intelligence",
        description="Display company name used in SaaS auth emails",
    )
    b2b_welcome_product_name: str = Field(
        default="Atlas B2B Intelligence",
        description="Display product name used in B2B welcome email subjects",
    )
    b2b_welcome_heading: str = Field(
        default="B2B Churn Intelligence",
        description="Heading used in B2B welcome emails",
    )
    consumer_welcome_product_name: str = Field(
        default="Amazon Seller Intelligence",
        description="Display product name used in consumer welcome emails",
    )

    # CORS -- extra origins to allow (comma-separated), e.g. "https://my-app.vercel.app"
    cors_origins: str = Field(default="", description="Extra CORS origins (comma-separated)")

    # Stripe
    stripe_secret_key: str = Field(default="", description="Stripe secret API key")
    stripe_webhook_secret: str = Field(default="", description="Stripe webhook signing secret")
    stripe_starter_price_id: str = Field(default="", description="Stripe Price ID for Starter plan")
    stripe_growth_price_id: str = Field(default="", description="Stripe Price ID for Growth plan")
    stripe_pro_price_id: str = Field(default="", description="Stripe Price ID for Pro plan")
    stripe_b2b_starter_price_id: str = Field(default="", description="Stripe Price ID for B2B Starter plan")
    stripe_b2b_growth_price_id: str = Field(default="", description="Stripe Price ID for B2B Growth plan")
    stripe_b2b_pro_price_id: str = Field(default="", description="Stripe Price ID for B2B Pro plan")
    stripe_vendor_standard_price_id: str = Field(default="", description="Stripe Price ID for Vendor Standard ($499/mo)")
    stripe_vendor_pro_price_id: str = Field(default="", description="Stripe Price ID for Vendor Pro ($1,499/mo)")

    @model_validator(mode="after")
    def _validate_secrets(self):
        if self.enabled and self.jwt_secret == "change-me-in-production":
            raise ValueError("ATLAS_SAAS_JWT_SECRET must be set when SaaS auth is enabled")
        if self.stripe_secret_key and not self.stripe_webhook_secret:
            import warnings
            warnings.warn(
                "ATLAS_SAAS_STRIPE_WEBHOOK_SECRET is empty while Stripe is configured. "
                "Stripe webhook endpoint will return 503 until this is set.",
                stacklevel=2,
            )
        return self


class STTConfig(BaseSettings):
    """STT-specific configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_STT_", env_file=ENV_FILES, extra="ignore")

    default_model: str = Field(default="nemotron", description="Default STT to load on startup")


class MQTTConfig(BaseSettings):
    """MQTT backend configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_MQTT_", env_file=ENV_FILES, extra="ignore")

    enabled: bool = Field(default=False, description="Enable MQTT backend")
    host: str = Field(default="localhost", description="MQTT broker host")
    port: int = Field(default=1883, description="MQTT broker port")
    username: Optional[str] = Field(default=None, description="MQTT username")
    password: Optional[str] = Field(default=None, description="MQTT password")


class HomeAssistantConfig(BaseSettings):
    """Home Assistant backend configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_HA_", env_file=ENV_FILES, extra="ignore")

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

    model_config = SettingsConfigDict(env_prefix="ATLAS_LLM_", env_file=ENV_FILES, extra="ignore")

    # Backend selection: "llama-cpp", "transformers-flash", "ollama", "together", "cloud", or "hybrid"
    default_model: str = Field(default="llama-cpp", description="Default LLM backend")

    # llama-cpp settings (GGUF models)
    model_path: Optional[str] = Field(default=None, description="Path to GGUF model file")
    n_ctx: int = Field(default=4096, description="Context window size")
    n_gpu_layers: int = Field(default=-1, description="GPU layers (-1 = all)")

    # ollama settings (Ollama API backend)
    ollama_model: str = Field(default="qwen3:14b", description="Ollama model name")
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama API URL (override: ATLAS_LLM__OLLAMA_URL)")
    ollama_timeout: int = Field(default=120, description="Ollama HTTP timeout in seconds (increase for cloud relay models)")

    # vllm settings (OpenAI-compatible vLLM server)
    vllm_model: str = Field(default="Qwen/Qwen3-14B-AWQ", description="vLLM model name")
    vllm_url: str = Field(default="http://localhost:8082", description="vLLM API base URL")
    vllm_guided_json_enabled: bool = Field(
        default=True,
        description=(
            "Enable guided/structured JSON decoding for vLLM requests. "
            "Disable to avoid xgrammar/nanobind leak issues in affected vLLM builds."
        ),
    )

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

    # OpenRouter reasoning model (synthesis/reasoning workloads)
    openrouter_reasoning_model: str = Field(
        default="openai/gpt-oss-120b",
        description=(
            "OpenRouter model for synthesis/reasoning workloads. "
            "Set via ATLAS_LLM__OPENROUTER_REASONING_MODEL "
            "(or ATLAS_LLM_OPENROUTER_REASONING_MODEL)."
        ),
    )
    openrouter_reasoning_strict: bool = Field(
        default=False,
        description=(
            "When true, synthesis/reasoning workloads fail closed if the "
            "configured OpenRouter reasoning model is unavailable instead of "
            "falling back to Anthropic, triage, or local models."
        ),
    )

    # Anthropic settings (email draft generation)
    anthropic_model: str = Field(
        default="claude-3-5-haiku-latest",
        description="Anthropic model name for email drafting",
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )

    # Cloud LLM (Ollama cloud model, runs alongside local for business workflows)
    cloud_enabled: bool = Field(
        default=False,
        description="Enable cloud LLM alongside local for business workflows (booking, email)",
    )
    cloud_ollama_model: str = Field(
        default="minimax-m2:cloud",
        description="Ollama cloud model for business workflows (e.g., minimax-m2:cloud)",
    )

    # Automatic day/night model swap (single 24GB GPU)
    model_swap_enabled: bool = Field(
        default=False,
        description=(
            "Enable automatic Ollama model swapping. "
            "At midnight: unloads day_model to free VRAM for graphiti-wrapper. "
            "At 7:30 AM: unloads night_model, pre-warms day_model for the day."
        ),
    )
    day_model: str = Field(
        default="qwen3:14b",
        description=(
            "Ollama model used during day hours. "
            "Should match ollama_model. Pre-warmed at 7:30 AM."
        ),
    )
    night_model: str = Field(
        default="qwen3:32b",
        description=(
            "Ollama model used by graphiti-wrapper at night for email graph extraction. "
            "Unloaded from VRAM at midnight; graphiti-wrapper loads it on demand at 1 AM."
        ),
    )
    model_swap_day_cron: str = Field(
        default="30 7 * * *",
        description=(
            "Cron expression for day model swap: unload night_model, pre-warm day_model. "
            "Default: 7:30 AM daily. Env: ATLAS_LLM__MODEL_SWAP_DAY_CRON."
        ),
    )
    model_swap_night_cron: str = Field(
        default="0 0 * * *",
        description=(
            "Cron expression for night model swap: unload day_model to free VRAM. "
            "Default: midnight daily. Env: ATLAS_LLM__MODEL_SWAP_NIGHT_CRON."
        ),
    )


class TTSConfig(BaseSettings):
    """TTS configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_TTS_", env_file=ENV_FILES, extra="ignore")

    default_model: str = Field(default="piper", description="Default TTS backend")
    voice: str = Field(default="en_US-ryan-medium", description="Voice model")
    speed: float = Field(default=1.0, description="Speech speed (1.0 = normal)")
    device: str | None = Field(default=None, description="Device for TTS: 'cuda', 'cpu', or None for auto")
    kokoro_lang: str = Field(default="en-us", description="Kokoro language code (en-us, en-gb, ja, zh, es, fr, hi, it, pt-br)")


class OmniConfig(BaseSettings):
    """Omni (unified speech-to-speech) configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_OMNI_", env_file=ENV_FILES, extra="ignore")

    enabled: bool = Field(default=False, description="Enable unified omni mode (Qwen-Omni)")
    default_model: str = Field(default="qwen-omni", description="Default omni model")
    max_new_tokens: int = Field(default=256, description="Max tokens for response generation")
    temperature: float = Field(default=0.7, description="Sampling temperature")


class SpeakerIDConfig(BaseSettings):
    """Speaker identification configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_SPEAKER_ID_",
        env_file=ENV_FILES,
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
        env_file=ENV_FILES,
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


class OrchestrationConfig(BaseSettings):
    """Voice pipeline orchestration configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_ORCH_",
        env_file=ENV_FILES,
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

    model_config = SettingsConfigDict(env_prefix="ATLAS_DISCOVERY_", env_file=ENV_FILES, extra="ignore")

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

    model_config = SettingsConfigDict(env_prefix="ATLAS_MEMORY_", env_file=ENV_FILES, extra="ignore")

    enabled: bool = Field(default=True, description="Enable memory service")
    base_url: str = Field(
        default="http://localhost:8001",
        description="URL of the atlas-memory (graphiti-wrapper) service (override: ATLAS_MEMORY__BASE_URL)",
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
    max_entity_edges: int = Field(
        default=20,
        description="Maximum entity edges to return from graph traversal",
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

    # Email graph sync settings - extract facts from emails for knowledge graph
    email_graph_sync_enabled: bool = Field(
        default=True,
        description="Enable daily email-to-graph extraction job",
    )
    email_graph_model: str = Field(
        default="qwen3:14b",
        description="Ollama model for Stage 1 email fact extraction (Atlas-side). Graphiti Stage 2 uses qwen3:32b nightly via graphiti-wrapper.",
    )
    email_graph_group_id: str = Field(
        default="",
        description="Group ID for email graph data (empty = use default group_id)",
    )
    email_graph_priorities: list[str] = Field(
        default=["action_required"],
        description="Email priorities to sync to graph",
    )
    email_graph_max_per_run: int = Field(
        default=20,
        description="Max emails to process per sync run",
    )


class ToolsConfig(BaseSettings):
    """Configuration for Atlas tools (weather, traffic, etc.)."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_TOOLS_", env_file=ENV_FILES, extra="ignore")

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

    # CalDAV calendar (provider-agnostic alternative to Google Calendar)
    # Works with Nextcloud, Apple Calendar, Fastmail, Proton Calendar, SOGo, Baikal, etc.
    caldav_url: str | None = Field(
        default=None,
        description="CalDAV server URL (e.g. https://nextcloud.example.com/remote.php/dav)",
    )
    caldav_username: str | None = Field(default=None, description="CalDAV username")
    caldav_password: str | None = Field(default=None, description="CalDAV password or app password")
    caldav_calendar_url: str | None = Field(
        default=None,
        description="Specific calendar collection URL (auto-discovered via PROPFIND if not set)",
    )

    # Gmail digest
    gmail_enabled: bool = Field(default=False, description="Enable Gmail digest")
    gmail_client_id: str | None = Field(default=None, description="Google OAuth client ID for Gmail")
    gmail_client_secret: str | None = Field(default=None, description="Google OAuth client secret for Gmail")
    gmail_refresh_token: str | None = Field(default=None, description="Gmail OAuth refresh token")
    gmail_max_results: int = Field(default=20, ge=1, le=50, description="Max emails per digest")
    gmail_query: str = Field(default="is:unread newer_than:1d", description="Gmail search query")
    gmail_body_max_chars: int = Field(
        default=2000, ge=500, le=8000,
        description="Max characters of email body text to include per message for LLM context",
    )
    gmail_dedup_retention_days: int = Field(
        default=90, ge=7, le=365,
        description="Days to keep processed email records before cleanup",
    )
    gmail_domain_map_file: str = Field(
        default="data/email_domains.json",
        description="Optional JSON file mapping sender domains to email categories (overrides built-in defaults)",
    )


class IntentConfig(BaseSettings):
    """Intent parsing configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_INTENT_", env_file=ENV_FILES, extra="ignore")

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

    model_config = SettingsConfigDict(env_prefix="ATLAS_ALERTS_", env_file=ENV_FILES, extra="ignore")

    enabled: bool = Field(default=True, description="Enable centralized alert system")
    default_cooldown_seconds: int = Field(default=30, description="Default cooldown between alerts")
    tts_enabled: bool = Field(default=True, description="Enable TTS announcements for alerts")
    persist_alerts: bool = Field(default=True, description="Persist alerts to database")
    ntfy_enabled: bool = Field(default=False, description="Enable ntfy push notifications")
    ntfy_url: str = Field(default="http://localhost:8090", description="ntfy server URL (override: ATLAS_ALERTS__NTFY_URL)")
    ntfy_topic: str = Field(default="atlas-alerts", description="ntfy topic for alerts")


class EmailConfig(BaseSettings):
    """Email tool configuration (Resend API + Gmail + IMAP)."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_EMAIL_", env_file=ENV_FILES, extra="ignore")

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

    # IMAP settings (provider-agnostic inbox reading)
    imap_host: str = Field(default="", description="IMAP server host (e.g. imap.gmail.com, outlook.office365.com)")
    imap_port: int = Field(default=993, description="IMAP server port (993=SSL, 143=STARTTLS)")
    imap_username: str = Field(default="", description="IMAP username (usually your email address)")
    imap_password: str = Field(default="", description="IMAP password or app-specific password")
    imap_ssl: bool = Field(default=True, description="Use SSL/TLS for IMAP connection")
    imap_mailbox: str = Field(default="INBOX", description="Default IMAP mailbox to read from")


class EmailDraftConfig(BaseSettings):
    """Email draft generation configuration (Anthropic LLM for reply drafting)."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_EMAIL_DRAFT_", env_file=ENV_FILES, extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable email draft generation")
    auto_draft_enabled: bool = Field(
        default=False,
        description="Enable automatic scheduled draft generation (False = user-initiated via ntfy)",
    )
    model_provider: str = Field(default="anthropic", description="LLM provider for drafting")
    model_name: str = Field(default="claude-sonnet-4-5-20250929", description="Model name for drafting")
    max_tokens: int = Field(default=1024, description="Max tokens for draft generation")
    temperature: float = Field(default=0.4, description="Sampling temperature for drafts")
    auto_draft_priorities: list[str] = Field(
        default=["action_required"],
        description="Email priorities that trigger auto-drafting",
    )
    draft_expiry_hours: int = Field(default=24, ge=1, le=168, description="Hours before pending drafts expire")
    notify_drafts: bool = Field(default=True, description="Send ntfy notification for new drafts")
    atlas_api_url: str = Field(
        default="http://localhost:8001",
        description="Atlas API URL for ntfy action buttons",
    )
    schedule_interval_seconds: int = Field(
        default=1800,
        ge=300,
        description="How often (seconds) the email draft task runs (default 30 min)",
    )
    triage_enabled: bool = Field(
        default=True,
        description="Use LLM to classify ambiguous emails as replyable/not",
    )
    triage_model: str = Field(
        default="claude-3-5-haiku-latest",
        description="Anthropic model for replyable triage (cheap/fast)",
    )
    triage_max_tokens: int = Field(
        default=32,
        description="Max tokens for triage response (yes/no answer)",
    )
    auto_approve_enabled: bool = Field(
        default=False, description="Auto-approve and send drafts above confidence threshold"
    )
    auto_approve_delay_seconds: int = Field(
        default=300, ge=60, le=1800,
        description="Delay before auto-sending (owner can cancel via ntfy during this window)",
    )
    auto_approve_min_confidence: float = Field(
        default=0.85, ge=0.5, le=1.0,
        description="Min intent confidence to auto-approve draft",
    )
    auto_approve_intents: list[str] = Field(
        default=["info_admin", "estimate_request", "reschedule"],
        description="Intents eligible for auto-approval (complaint always excluded)",
    )


class EmailIntakeConfig(BaseSettings):
    """Near-real-time email intake polling configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_EMAIL_INTAKE_", env_file=ENV_FILES, extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable 10-min email polling")
    interval_seconds: int = Field(default=600, ge=300, description="Poll interval in seconds")
    crm_enabled: bool = Field(default=True, description="Cross-reference emails against CRM")
    action_plan_enabled: bool = Field(
        default=True, description="Generate LLM action plans for CRM matches"
    )
    max_action_plans_per_cycle: int = Field(
        default=100, ge=1, le=200, description="Cap LLM calls per polling cycle"
    )
    auto_execute_enabled: bool = Field(
        default=False, description="Auto-execute intent actions above confidence threshold"
    )
    auto_execute_min_confidence: float = Field(
        default=0.85, ge=0.5, le=1.0, description="Min confidence to auto-execute"
    )
    auto_execute_intents: list[str] = Field(
        default=["estimate_request", "reschedule", "info_admin"],
        description="Intents eligible for auto-execution (complaint always excluded)",
    )
    inbox_rules_enabled: bool = Field(
        default=False, description="Evaluate user-defined inbox rules before LLM"
    )
    followup_auto_action_enabled: bool = Field(
        default=False, description="Auto-draft replies to follow-up emails"
    )
    followup_auto_action_intents: list[str] = Field(
        default=["estimate_request", "reschedule", "info_admin"],
        description="Original intents eligible for follow-up auto-actions (complaint always excluded)",
    )


class EmailStaleCheckConfig(BaseSettings):
    """Stale email re-engagement: detect stale drafts, unactioned emails, unanswered estimates."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_EMAIL_STALE_CHECK_", env_file=ENV_FILES, extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable stale email re-engagement checks")
    interval_seconds: int = Field(
        default=7200, ge=1800, le=86400, description="Check interval in seconds (default 2h)"
    )

    # Scenario 1: stale pending drafts
    stale_draft_hours: int = Field(
        default=12, ge=1, le=72, description="Hours before a pending draft is considered stale"
    )

    # Scenario 2: unactioned high-priority emails
    unactioned_hours: int = Field(
        default=24, ge=4, le=168, description="Hours before an unactioned email triggers escalation"
    )

    # Scenario 3: unanswered estimate replies
    unanswered_days: int = Field(
        default=3, ge=1, le=14, description="Days before generating a follow-up for unanswered sent replies"
    )
    unanswered_intents: list[str] = Field(
        default=["estimate_request"],
        description="Intents eligible for follow-up generation",
    )
    max_followups_per_cycle: int = Field(
        default=3, ge=1, le=10, description="Max follow-up drafts to generate per cycle"
    )

    # Anti-spam
    max_reminders: int = Field(
        default=3, ge=1, le=10, description="Max stale reminders per draft/email before giving up"
    )


class ReminderConfig(BaseSettings):
    """Reminder system configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_REMINDER_",
        env_file=ENV_FILES,
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

    model_config = SettingsConfigDict(env_prefix="ATLAS_VOICE_", env_file=ENV_FILES, extra="ignore")

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
    wake_confirmation_enabled: bool = Field(
        default=False,
        description="Play a short tone when wake word is detected"
    )
    wake_confirmation_freq: int = Field(
        default=880,
        description="Frequency in Hz for wake word confirmation tone"
    )
    wake_confirmation_duration_ms: int = Field(
        default=80,
        description="Duration in ms for wake word confirmation tone"
    )

    # ASR settings (HTTP batch mode)
    auto_start_asr: bool = Field(default=True, description="Auto-start ASR subprocess on brain startup")
    asr_url: str | None = Field(default="http://localhost:8081", description="Nemotron ASR HTTP endpoint URL (override: ATLAS_VOICE__ASR_URL)")
    asr_api_key: str | None = Field(default=None, description="ASR API key if required")
    asr_timeout: int = Field(default=30, description="ASR request timeout in seconds")
    asr_model: str = Field(
        default="nvidia/nemotron-speech-streaming-en-0.6b",
        description="ASR model name or path for auto-started server",
    )
    asr_device: str = Field(default="cuda", description="Torch device for ASR server (cuda, cuda:0, cpu)")
    asr_startup_timeout: int = Field(default=120, ge=10, le=600, description="Seconds to wait for ASR server startup")

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
    conversation_agent_enabled: bool = Field(
        default=True,
        description=(
            "Use the full agent path (with tool access) for conversation mode "
            "instead of streaming. Enables natural tool use mid-conversation "
            "at the cost of ~500ms higher time-to-first-audio."
        ),
    )
    conversation_agent_max_tokens: int = Field(
        default=512,
        description="Max tokens for agent-path responses in conversation mode",
    )

    # VAD and segmentation settings
    vad_aggressiveness: int = Field(default=2, description="WebRTC VAD aggressiveness (0-3)")
    silence_ms: int = Field(default=700, description="Silence duration to end utterance")
    hangover_ms: int = Field(default=400, description="Hangover time before finalizing")
    max_command_seconds: int = Field(default=30, description="Safety-valve max recording duration (silence detection ends utterances naturally)")
    min_command_ms: int = Field(default=1500, description="Minimum recording time before silence can finalize (grace period)")
    min_speech_frames: int = Field(default=3, description="Minimum VAD speech frames required before silence can finalize")
    wake_buffer_frames: int = Field(default=5, description="Pre-roll buffer size in frames for wake word mode (captures audio before wake word)")

    # Conversation-mode segmentation -- sliding window approach
    # In conversation mode, users pause naturally between thoughts (~0.5-1s).
    # 800ms silence + 300ms hangover = ~1040ms before finalization, which avoids
    # cutting off mid-sentence pauses while staying responsive.
    conversation_silence_ms: int = Field(default=2000, description="Confirmation silence duration in conversation mode")
    conversation_hangover_ms: int = Field(default=500, description="Hangover in conversation mode")
    conversation_early_silence_ms: int = Field(
        default=600,
        description="Silence duration to trigger early LLM preparation in conversation mode (ms)"
    )
    conversation_max_command_seconds: int = Field(default=120, description="Max recording in conversation mode")
    conversation_window_frames: int = Field(default=20, description="Sliding window size for speech ratio (0=disabled)")
    conversation_silence_ratio: float = Field(default=0.15, description="Speech ratio below which silence counter engages")
    conversation_asr_holdoff_ms: int = Field(default=500, description="Suppress finalization for N ms after last ASR partial")
    asr_quiet_limit: int = Field(
        default=5,
        description="Max frames with no new ASR partial before stopping audio feed (~80ms/frame)"
    )

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
        default=500,
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

    filler_followup_delay_ms: int = Field(
        default=5000,
        description="Milliseconds before speaking a follow-up filler phrase"
    )
    filler_followup_phrases: list[str] = Field(
        default=["Still working on that.", "Almost there.", "Hang tight."],
        description="Second-tier filler phrases for very slow agent responses"
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
    conversation_turn_limit_phrase: str = Field(
        default="Say Hey Atlas to continue.",
        description="Phrase spoken when conversation mode ends due to turn limit"
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
        env_file=ENV_FILES,
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

    model_config = SettingsConfigDict(env_prefix="ATLAS_RTSP_", env_file=ENV_FILES, extra="ignore")

    enabled: bool = Field(default=False, description="DEPRECATED - detection moved to atlas_vision")
    wyze_bridge_host: str = Field(default="localhost", description="DEPRECATED")
    wyze_bridge_port: int = Field(default=8554, description="DEPRECATED")
    fps: int = Field(default=10, description="DEPRECATED")
    cameras_json: str = Field(default="", description="DEPRECATED")


class SecurityConfig(BaseSettings):
    """Security system configuration (video processing, cameras, zones)."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_SECURITY_", env_file=ENV_FILES, extra="ignore")

    enabled: bool = Field(default=True, description="Enable security tools")
    video_processing_url: str = Field(
        default="http://localhost:5002",
        description="Video processing API URL (override: ATLAS_SECURITY__VIDEO_PROCESSING_URL)",
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

    model_config = SettingsConfigDict(env_prefix="ATLAS_MODE_", env_file=ENV_FILES, extra="ignore")

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
        env_file=ENV_FILES,
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
        default=3.0,
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
    llm_fallback_model: str = Field(
        default="phi3:mini",
        description="Ollama model for LLM fallback classification (lighter than main model)",
    )
    llm_fallback_log: str = Field(
        default="data/intent_fallback.jsonl",
        description="JSONL log path for queries that trigger LLM fallback (for fine-tuning data)",
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
    conversation_workflow_confidence_floor: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Lower confidence floor for workflow routes when in conversation mode. "
            "Prevents borderline workflow intents from falling through to the "
            "general LLM tool path during active conversation."
        ),
    )


class DeviceResolverConfig(BaseSettings):
    """Device resolver for embedding-based device name matching."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_DEVICE_RESOLVER_",
        env_file=ENV_FILES,
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


class EntityContextConfig(BaseSettings):
    """Configuration for voice entity context between turns."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_ENTITY_CONTEXT__",
        env_file=ENV_FILES,
        extra="ignore",
    )

    max_age_s: float = Field(
        default=600.0,
        ge=0.0,
        description=(
            "Maximum age in seconds for entity context entries. "
            "Entities from turns older than this are dropped at read time. "
            "0 = no expiry."
        ),
    )

    cross_session_max_age_s: float = Field(
        default=86400.0,
        ge=0.0,
        description=(
            "Maximum age in seconds for cross-session entity fallback. "
            "When the current session has no recent entities, Atlas looks back "
            "across all sessions up to this window. 0 = no expiry. Default 86400 = 24h."
        ),
    )

    graph_entity_fallback: bool = Field(
        default=False,
        description=(
            "When True, fall back to GraphRAG for entity hints when both "
            "current-session and cross-session entity lookups return nothing. "
            "Only applies when ATLAS_MEMORY__ENABLED=true."
        ),
    )


class FreeModeConfig(BaseSettings):
    """Free Conversation Mode - always-on listening when conditions are met.

    When enabled, Atlas stays in conversation mode continuously (no wake word
    needed) as long as a known speaker is active in the room. Exits automatically
    after the speaker has been silent/absent for speaker_id_expiry_s seconds.
    """

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_FREE_MODE_",
        env_file=ENV_FILES,
        extra="ignore",
    )

    enabled: bool = Field(
        default=False,
        description="Enable free conversation mode (always-on listening)",
    )
    require_known_speaker: bool = Field(
        default=True,
        description="Only activate when a known speaker is confirmed (requires speaker_id.enabled=True)",
    )
    min_speaker_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum speaker ID confidence to activate free mode",
    )
    ambient_rms_max: float = Field(
        default=0.02,
        ge=0.0,
        description="Max ambient RMS to enter/stay in free mode (only meaningful with rms_adaptive=True)",
    )
    poll_interval_s: float = Field(
        default=10.0,
        ge=1.0,
        description="How often (seconds) to re-evaluate entry/exit conditions",
    )
    extended_timeout_ms: int = Field(
        default=30000,
        ge=5000,
        description="Conversation silence timeout in free mode (longer than normal)",
    )
    speaker_id_expiry_s: float = Field(
        default=90.0,
        ge=10.0,
        description="Seconds after last speaker confirmation before exiting free mode",
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
        env_file=ENV_FILES,
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


class ModelPricingConfig(BaseModel):
    """Per-model pricing in USD per 1M tokens.

    Keys are ``provider/model`` slugs (case-insensitive lookup).
    Local models (ollama, vllm) default to $0 since they run on-prem.
    """

    # Anthropic (per 1M tokens)
    anthropic_sonnet_input: float = 3.00
    anthropic_sonnet_output: float = 15.00
    anthropic_sonnet_cache_read_input: float = 0.30
    anthropic_sonnet_cache_write_input: float = 3.75
    anthropic_haiku_input: float = 0.25
    anthropic_haiku_output: float = 1.25
    anthropic_haiku_cache_read_input: float = 0.03
    anthropic_haiku_cache_write_input: float = 0.30

    # Groq (per 1M tokens -- hosted inference)
    groq_llama70b_input: float = 0.59
    groq_llama70b_output: float = 0.79

    # OpenRouter (varies -- default baseline pricing)
    openrouter_default_input: float = 1.10
    openrouter_default_output: float = 4.40

    # Together AI (default to Llama 70B pricing)
    together_default_input: float = 0.88
    together_default_output: float = 0.88

    # Local models (free -- GPU electricity only)
    local_input: float = 0.0
    local_output: float = 0.0

    def cost_usd(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
        billable_input_tokens: int | None = None,
    ) -> float:
        """Calculate cost in USD for a given call."""
        p = (provider or "").lower()
        m = (model or "").lower()
        cache_read = max(int(cached_tokens or 0), 0)
        cache_write = max(int(cache_write_tokens or 0), 0)
        base_input = (
            max(int(billable_input_tokens), 0)
            if billable_input_tokens is not None
            else max(int(input_tokens or 0), 0)
        )
        if p in ("ollama", "vllm", "transformers-flash", "llama-cpp") or "local" in p:
            return 0.0
        if p == "anthropic" or "claude" in m:
            if "haiku" in m:
                return (
                    base_input * self.anthropic_haiku_input
                    + cache_read * self.anthropic_haiku_cache_read_input
                    + cache_write * self.anthropic_haiku_cache_write_input
                    + output_tokens * self.anthropic_haiku_output
                ) / 1_000_000
            return (
                base_input * self.anthropic_sonnet_input
                + cache_read * self.anthropic_sonnet_cache_read_input
                + cache_write * self.anthropic_sonnet_cache_write_input
                + output_tokens * self.anthropic_sonnet_output
            ) / 1_000_000
        if p == "groq":
            return (input_tokens * self.groq_llama70b_input + output_tokens * self.groq_llama70b_output) / 1_000_000
        if p == "openrouter":
            return (input_tokens * self.openrouter_default_input + output_tokens * self.openrouter_default_output) / 1_000_000
        if p in ("together", "cloud", "hybrid"):
            return (input_tokens * self.together_default_input + output_tokens * self.together_default_output) / 1_000_000
        return 0.0


class FTLTracingConfig(BaseModel):
    """Fine-Tune Labs tracing configuration."""

    enabled: bool = True
    base_url: str = "https://finetunelab.ai"
    api_key: str = ""  # wak_... key for FTL API
    user_id: str = ""  # FTL user ID for trace ownership
    capture_business_context: bool = Field(
        default=True,
        description="Attach business-intelligence context to FTL traces",
    )
    capture_reasoning_summaries: bool = Field(
        default=True,
        description="Attach structured reasoning summaries to FTL traces",
    )
    capture_raw_reasoning: bool = Field(
        default=False,
        description="Include truncated raw reasoning output when available",
    )
    max_reasoning_chars: int = Field(
        default=1200,
        ge=0,
        le=10000,
        description="Maximum characters stored for reasoning previews",
    )
    pricing: ModelPricingConfig = Field(default_factory=ModelPricingConfig)


class PersonaConfig(BaseSettings):
    """Atlas persona and system prompt configuration.

    Single source of truth for Atlas's identity and behavior.
    All LLM-facing system prompts pull from here.
    """

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_PERSONA_",
        env_file=ENV_FILES,
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
    owner_name: str = Field(default="Juan", description="Owner/user name for email sign-offs and personalization")


class HomeAgentConfig(BaseSettings):
    """Home agent LLM generation parameters."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_HOME_AGENT_", env_file=ENV_FILES, extra="ignore")

    max_tokens: int = Field(default=150, ge=50, le=1024, description="Max tokens for LLM response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature for conversation")
    tool_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature for tool calling")
    max_history: int = Field(default=4, ge=0, le=20, description="Max conversation history turns to include")


class AgentConfig(BaseSettings):
    """Agent system configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_AGENT_", env_file=ENV_FILES, extra="ignore")

    backend: str = Field(
        default="langgraph",
        description="Agent backend: 'langgraph' (default)",
    )


class WorkflowConfig(BaseSettings):
    """Workflow tool backend configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_WORKFLOW_",
        env_file=ENV_FILES,
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
        env_file=ENV_FILES,
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
        env_file=ENV_FILES,
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
        env_file=ENV_FILES,
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable autonomous task scheduler")
    default_agent_type: str = Field(default="atlas", description="Default agent type for headless tasks")
    default_session_prefix: str = Field(default="autonomous", description="Session ID prefix for task runs")
    max_concurrent_tasks: int = Field(default=2, ge=1, le=10, description="Max concurrent task executions")
    task_timeout_seconds: int = Field(default=120, ge=10, le=3600, description="Default task timeout")
    task_history_retention_days: int = Field(default=30, ge=1, le=365, description="Execution history retention")
    hooks_enabled: bool = Field(default=True, description="Enable alert-driven hook processing")
    hook_cooldown_seconds: int = Field(default=30, ge=0, le=300, description="Min seconds between duplicate hook executions")
    default_timezone: str = Field(default="America/Chicago", description="Default timezone for scheduled tasks")
    misfire_recovery_seconds: int = Field(default=600, ge=0, le=3600, description="On startup, recover cron tasks that missed their window within this many seconds")

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

    # Push notifications for task results
    notify_results: bool = Field(
        default=True,
        description="Send ntfy push notification when a synthesized task result is produced",
    )
    notify_priority: str = Field(
        default="default",
        description="Default ntfy priority for autonomous task notifications",
    )

    # TTS broadcast for task results
    announce_results: bool = Field(
        default=False,
        description="Broadcast synthesized task results to edge nodes via TTS",
    )

    # --- Task-specific defaults (overridable per-task via task.metadata) ---

    # device_health
    device_health_battery_threshold: int = Field(
        default=20, description="Low battery % threshold for device health check",
    )
    device_health_stale_hours: int = Field(
        default=24, description="Hours before an entity is considered stale",
    )

    # morning_briefing
    morning_briefing_calendar_hours: int = Field(
        default=12, description="Hours ahead for calendar events in morning briefing",
    )
    morning_briefing_security_hours: int = Field(
        default=8, description="Lookback hours for overnight security summary",
    )

    # calendar_reminder
    calendar_reminder_lead_minutes: int = Field(
        default=30, description="Max minutes before event to send reminder",
    )
    calendar_reminder_min_minutes: int = Field(
        default=15, description="Min minutes before event to send reminder",
    )

    # security_summary
    security_summary_hours: int = Field(
        default=24, description="Lookback window in hours for security summary",
    )

    # action_escalation
    action_escalation_stale_days: int = Field(
        default=7, description="Days before a pending action is considered stale",
    )
    action_escalation_overdue_days: int = Field(
        default=3, description="Days before a pending action is considered overdue",
    )

    # anomaly_detection
    anomaly_detection_deviation_threshold: float = Field(
        default=2.0, description="Standard-deviation multiplier for anomaly detection",
    )
    anomaly_detection_min_samples: int = Field(
        default=3, description="Minimum pattern samples before anomaly comparison",
    )

    # pattern_learning
    pattern_learning_lookback_days: int = Field(
        default=30, description="Days of history for temporal pattern learning",
    )

    # preference_learning
    preference_learning_lookback_days: int = Field(
        default=7, description="Days of conversation history for preference learning",
    )
    preference_learning_min_turns: int = Field(
        default=20, description="Minimum user turns before updating preferences",
    )

    # proactive_actions
    proactive_actions_lookback_hours: int = Field(
        default=24, description="Hours of conversation history to scan for action items",
    )

    # nightly_memory_sync
    nightly_sync_max_turns: int = Field(
        default=50, description="Max conversation turns to sync per nightly run",
    )
    nightly_sync_max_session_turns: int = Field(
        default=30, description="Max turns per session batch sent to Graphiti (larger sessions are chunked)",
    )
    nightly_sync_container_name: str = Field(
        default="atlas-graphiti-wrapper",
        description="Docker container name for the Graphiti wrapper service",
    )

    # email_graph_sync
    email_graph_sync_max_emails: int = Field(
        default=20, description="Max emails to process per email-graph sync run",
    )

    # gmail_digest
    gmail_digest_batch_size: int = Field(
        default=10, description="Concurrent batch size for Gmail message fetching",
    )


class EscalationConfig(BaseSettings):
    """Edge-local narration + brain-side escalation configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_ESCALATION_", env_file=ENV_FILES, extra="ignore")

    enabled: bool = Field(default=True, description="Enable escalation evaluation for security events")
    unknown_empty_enabled: bool = Field(default=True, description="Escalate unknown face when house is empty")
    rapid_unknowns_threshold: int = Field(default=3, ge=2, le=10, description="Unknown face count to trigger rapid-unknowns escalation")
    rapid_unknowns_window_seconds: int = Field(default=60, ge=10, le=300, description="Sliding window for rapid unknown face detection")
    synthesis_skill: str = Field(default="security/escalation_narration", description="Skill for LLM escalation synthesis")
    synthesis_max_tokens: int = Field(default=128, ge=32, le=512, description="Max tokens for escalation narration (keep short for TTS)")
    synthesis_temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="LLM temperature for escalation")
    narration_hint_enabled: bool = Field(default=True, description="Include narration hints in security_ack")
    broadcast_occupancy: bool = Field(default=True, description="Broadcast occupancy state changes to edge nodes")


class CallIntelligenceConfig(BaseSettings):
    """Post-call transcription and data extraction."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_CALL_INTELLIGENCE_", env_file=ENV_FILES, extra="ignore"
    )
    enabled: bool = Field(default=False, description="Enable call recording and processing")
    min_duration_seconds: int = Field(default=10, ge=0, description="Skip calls shorter than this")
    asr_url: str = Field(default="http://localhost:8081/v1/asr", description="ASR batch endpoint")
    asr_timeout: int = Field(default=60, ge=10, le=300, description="ASR request timeout")
    llm_max_tokens: int = Field(default=1024, ge=128, le=4096)
    llm_temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    llm_timeout: float = Field(default=30.0, ge=5.0, le=120.0, description="LLM call timeout in seconds")
    notify_enabled: bool = Field(default=True, description="Push ntfy after processing")


class SMSIntelligenceConfig(BaseSettings):
    """SMS classification, extraction, and notification."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_SMS_INTELLIGENCE_", env_file=ENV_FILES, extra="ignore"
    )
    enabled: bool = Field(default=True, description="Enable SMS intelligence pipeline")
    llm_max_tokens: int = Field(default=512, ge=128, le=2048)
    llm_temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    llm_timeout: float = Field(default=20.0, ge=5.0, le=60.0, description="LLM call timeout in seconds")
    notify_enabled: bool = Field(default=True, description="Push ntfy after processing")
    auto_reply_timeout: float = Field(default=10.0, ge=3.0, le=30.0, description="Auto-reply LLM timeout")


class InvoicingConfig(BaseSettings):
    """Invoicing and payment tracking configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_INVOICING_", env_file=ENV_FILES, extra="ignore"
    )
    enabled: bool = Field(default=False, description="Enable invoicing system")
    default_payment_terms_days: int = Field(default=30, ge=1, le=365, description="Default days until due")
    default_tax_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Default tax rate (0.0-1.0)")
    reminder_max_count: int = Field(default=3, ge=0, le=10, description="Max payment reminders per invoice")
    reminder_interval_days: int = Field(default=7, ge=1, le=90, description="Days between reminders")
    notify_enabled: bool = Field(default=True, description="Push ntfy for invoice events")
    invoice_number_prefix: str = Field(default="INV", description="Prefix for invoice numbers")
    auto_invoice_enabled: bool = Field(default=True, description="Monthly auto-invoice generation")
    auto_invoice_send_email: bool = Field(default=True, description="Auto-send invoices via email")
    auto_invoice_due_days: int = Field(default=30, ge=1, le=365, description="Payment terms for auto-invoices")
    auto_invoice_calendar_id: str = Field(default="", description="Google Calendar ID for commercial cleaning events")
    auto_invoice_review_mode: bool = Field(default=True, description="Hold invoices as draft for review instead of auto-sending")
    auto_invoice_save_path: str = Field(default="~/Desktop/Atlas-Invoices", description="Base path for saving invoice PDFs")


class ExternalDataConfig(BaseSettings):
    """External data producers: news feeds and financial markets."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_EXTERNAL_DATA_", env_file=ENV_FILES, extra="ignore"
    )

    enabled: bool = Field(default=False, description="Master switch for external data producers")
    # News
    news_enabled: bool = Field(default=True, description="Enable news intake (requires enabled=True)")
    news_interval_seconds: int = Field(default=900, description="News polling interval (15 min)")
    news_api_provider: str = Field(default="mediastack", description="mediastack | google_rss")
    news_api_key: Optional[str] = Field(default=None, description="API key (Mediastack access_key)")
    news_max_articles_per_poll: int = Field(default=20, description="Max articles per poll cycle")
    news_significance_threshold: float = Field(default=0.6, description="LLM significance score cutoff (0-1)")
    # Markets
    market_enabled: bool = Field(default=True, description="Enable market intake (requires enabled=True)")
    market_interval_seconds: int = Field(default=300, description="Market polling interval (5 min)")
    market_api_provider: str = Field(default="alpha_vantage", description="alpha_vantage | finnhub")
    market_api_key: Optional[str] = Field(default=None, description="API key (Alpha Vantage or Finnhub)")
    market_default_threshold_pct: float = Field(default=5.0, description="Default price move % to trigger alert")
    market_hours_only: bool = Field(default=False, description="Only poll during US market hours (9:30-16:00 ET)")
    # Context / reasoning windows
    context_lookback_hours: int = Field(default=24, description="Hours of market/news data to include in reasoning context")
    correlation_news_lookback_hours: int = Field(default=48, description="Hours to look back for news in correlation detection")
    correlation_market_window_hours: int = Field(default=24, description="Hours after news to check for correlated market moves")
    retention_days: int = Field(default=30, description="Days to retain dedup entries and market snapshots")
    # Daily intelligence
    intelligence_enabled: bool = Field(default=True, description="Enable daily intelligence analysis task")
    intelligence_cron: str = Field(default="0 20 * * *", description="Cron for daily intelligence (default 8 PM)")
    intelligence_analysis_window_days: int = Field(default=7, description="Days of data to include in analysis")
    intelligence_max_prior_sessions: int = Field(default=5, description="Max prior reasoning journal entries to include")
    intelligence_max_tokens: int = Field(default=4096, description="Max tokens for intelligence LLM call")
    intelligence_journal_retention_days: int = Field(default=90, description="Days to retain reasoning journal entries")
    intelligence_news_retention_days: int = Field(default=30, description="Days to retain news articles")
    intelligence_temperature: float = Field(default=0.4, description="LLM temperature for daily intelligence analysis")
    # Intelligence reports (on-demand)
    report_full_max_tokens: int = Field(default=1500, description="Max tokens for full intelligence reports")
    report_executive_max_tokens: int = Field(default=500, description="Max tokens for executive summary reports")
    report_builder_max_tokens: int = Field(default=2000, description="Max tokens for report package builder")
    report_temperature: float = Field(default=0.3, description="LLM temperature for intelligence reports")
    # Intervention pipeline
    intervention_stage1_max_tokens: int = Field(default=1200, description="Max tokens for intervention stage 1 (playbook)")
    intervention_stage2_max_tokens: int = Field(default=1500, description="Max tokens for intervention stage 2 (simulation)")
    intervention_stage3_max_tokens: int = Field(default=1500, description="Max tokens for intervention stage 3 (narrative)")
    intervention_temperature: float = Field(default=0.3, description="LLM temperature for intervention pipeline")
    # Temporal correlation
    temporal_correlation_window_hours: float = Field(default=4.0, description="Time window (hours) for article/market move correlation")
    market_move_threshold_pct: float = Field(default=2.0, description="Min % price change to flag as abnormal market move")
    # API tuning
    news_max_keywords_per_query: int = Field(default=10, description="Max keywords per NewsAPI query")
    news_max_rss_feeds: int = Field(default=5, description="Max Google RSS feeds to poll per cycle")
    news_keyword_min_length: int = Field(default=3, description="Min word length for market symbol cross-ref matching")
    api_timeout_seconds: float = Field(default=20.0, description="HTTP timeout for external API calls")
    # Article enrichment
    enrichment_enabled: bool = Field(default=True, description="Enable article content enrichment")
    enrichment_interval_seconds: int = Field(default=600, description="Article enrichment polling interval (10 min)")
    enrichment_max_per_batch: int = Field(default=10, description="Max articles to enrich per batch")
    enrichment_max_attempts: int = Field(default=3, description="Max fetch attempts before marking failed")
    enrichment_content_max_chars: int = Field(default=10000, description="Max chars to store from article body")
    enrichment_fetch_timeout: float = Field(default=15.0, description="HTTP timeout for article content fetch")
    enrichment_classification_max_tokens: int = Field(
        default=1200,
        description="Max LLM output tokens for article SORAM classification",
    )
    # Pressure scoring
    pressure_enabled: bool = Field(default=True, description="Enable pressure signal detection")
    pressure_baseline_window_days: int = Field(default=180, description="Rolling window for baseline (6 months)")
    pressure_alert_threshold: float = Field(default=7.0, description="Pressure score alert threshold (0-10)")
    pressure_drift_alert_threshold: float = Field(default=2.0, description="Sentiment drift alert threshold")
    pressure_max_delta_per_day: float = Field(default=2.0, description="Max pressure score change per day without sensor support")
    pressure_sensor_supported_delta: float = Field(default=5.0, description="Max pressure score change when sensor composite is HIGH/CRITICAL")
    # Safety gate
    safety_auto_approve_max_risk: str = Field(default="MEDIUM", description="Max risk level for auto-approval (LOW, MEDIUM, HIGH, CRITICAL)")
    safety_approval_expiry_hours: int = Field(default=72, description="Hours before pending approval requests expire")
    # Complaint mining
    complaint_mining_enabled: bool = Field(default=True, description="Enable complaint mining pipeline")
    complaint_enrichment_interval_seconds: int = Field(default=300, description="Complaint enrichment polling interval (5 min)")
    complaint_enrichment_max_per_batch: int = Field(default=20, description="Max reviews to enrich per batch")
    complaint_enrichment_max_attempts: int = Field(default=3, description="Max enrichment attempts before marking failed")
    complaint_enrichment_local_only: bool = Field(default=False, description="Force local LLM for enrichment (skip Claude)")
    complaint_enrichment_reviews_per_call: int = Field(default=1, description="Reviews per LLM call (batch mode when >1, max 10)")
    complaint_analysis_enabled: bool = Field(default=True, description="Enable daily complaint analysis task")
    complaint_analysis_cron: str = Field(default="0 21 * * *", description="Cron for complaint analysis (default 9 PM)")
    complaint_analysis_window_days: int = Field(default=7, description="Days of enriched reviews to include in analysis")
    complaint_analysis_max_tokens: int = Field(default=1500, description="Max output tokens for analysis LLM call")
    complaint_retention_days: int = Field(default=365, description="Days to retain complaint reports")
    # Content generation (Claude-powered)
    complaint_content_enabled: bool = Field(default=True, description="Enable complaint content generation")
    complaint_content_cron: str = Field(default="0 22 * * *", description="Cron for content generation (default 10 PM)")
    complaint_content_max_per_run: int = Field(default=5, description="Max content pieces to generate per run")
    complaint_content_max_tokens: int = Field(default=2048, description="Max tokens per content generation call")
    # Deep enrichment (second-pass extraction)
    deep_enrichment_interval_seconds: int = Field(default=600, description="Deep enrichment polling interval (10 min)")
    deep_enrichment_max_per_batch: int = Field(default=5, description="Max reviews to deep-enrich per autonomous batch")
    deep_enrichment_max_attempts: int = Field(default=3, description="Max attempts before marking deep_failed")
    deep_enrichment_max_tokens: int = Field(default=1024, description="Max LLM output tokens for deep extraction (32 fields)")
    deep_enrichment_blast_workers: int = Field(default=80, description="Concurrent workers for blast_deep_enrichment script")
    deep_enrichment_blast_batch_size: int = Field(default=30, description="Reviews claimed per worker per round in blast script")
    # Competitive intelligence (cross-brand analysis from deep_extraction)
    competitive_intelligence_enabled: bool = Field(default=True, description="Enable competitive intelligence analysis")
    competitive_intelligence_cron: str = Field(default="30 21 * * *", description="Cron for competitive intelligence (default 9:30 PM)")
    competitive_intelligence_max_tokens: int = Field(default=1500, description="Max output tokens for competitive intelligence LLM call")
    competitive_intelligence_min_deep_enriched: int = Field(default=100, description="Min deep-enriched reviews required to run")
    # Blog post generation (data-backed articles with charts)
    blog_post_enabled: bool = Field(default=True, description="Enable blog post generation from review data")
    blog_post_cron: str = Field(default="0 23 * * 0", description="Cron for blog post generation (weekly, Sunday 11 PM)")
    blog_post_max_tokens: int = Field(default=8192, description="Max tokens per blog post LLM call (reasoning models need extra budget)")
    blog_post_max_per_run: int = Field(default=1, description="Max blog posts to generate per run")
    blog_post_ui_path: str = Field(default="", description="Path to atlas-intel-ui/src/content/blog/ (empty = DB only)")
    blog_base_url: str = Field(default="https://atlas-intel-ui-two.vercel.app", description="Base URL for consumer blog (full URLs in campaign emails)")
    amazon_associate_tag: str = Field(default="", description="Amazon Associates tag for consumer affiliate links")
    blog_post_openrouter_model: str = Field(
        default="openai/gpt-oss-120b",
        description="OpenRouter model for blog post generation",
    )
    # Blog auto-deploy (git push + Vercel deploy hook)
    blog_auto_deploy_enabled: bool = Field(default=False, description="Auto git-push + Vercel deploy after blog publish")
    blog_auto_deploy_branch: str = Field(default="dev", description="Git branch to push blog commits to")
    blog_auto_deploy_hook_url: str = Field(default="", description="Vercel deploy hook URL (POST triggers rebuild)")


class B2BChurnConfig(BaseSettings):
    """B2B software churn prediction pipeline configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_B2B_CHURN_", env_file=ENV_FILES, extra="ignore"
    )

    enabled: bool = Field(default=False, description="Enable B2B churn prediction pipeline")

    # MCP tool gating
    mcp_tool_groups: str = Field(
        default="all",
        description=(
            "Comma-separated tool groups to expose via MCP. "
            "'all' = register every group (default, backward compatible). "
            "Groups: read_signals, read_reports, read_pools, reasoning, admin, "
            "calibration, webhooks, crm_events, content, write_intelligence, campaigns"
        ),
    )

    # Enrichment
    enrichment_interval_seconds: int = Field(default=300, description="Enrichment polling interval")
    enrichment_max_per_batch: int = Field(default=10, description="Max reviews to enrich per batch")
    enrichment_max_rounds_per_run: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Max enrichment claim rounds to process per scheduled run",
    )
    enrichment_concurrency: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Concurrent review enrichments allowed within a single batch",
    )
    enrichment_inter_batch_delay_seconds: float = Field(
        default=2.0,
        ge=0.0,
        le=60.0,
        description="Delay between enrichment claim rounds within a single run",
    )
    enrichment_priority_sources: str = Field(
        default="",
        description=(
            "Comma-separated enrichment source priority queue. "
            "When set, these sources are claimed first in each enrichment batch."
        ),
    )
    enrichment_skip_sources: str = Field(
        default="",
        description=(
            "Comma-separated sources to skip for churn enrichment. "
            "These rows are marked not_applicable because they are unsupported for the review extractor."
        ),
    )
    enrichment_max_attempts: int = Field(default=3, description="Max enrichment attempts")
    enrichment_full_extraction_timeout_seconds: float = Field(
        default=120.0,
        ge=30.0,
        le=1800.0,
        description="Per-review timeout for full enrichment extraction stages",
    )
    enrichment_auto_requeue_parser_upgrades: bool = Field(
        default=False,
        description=(
            "Automatically reset enriched/no-signal reviews to pending when a parser_version changes. "
            "Disabled by default to avoid expensive mass re-enrichment during normal testing."
        ),
    )
    enrichment_auto_requeue_model_upgrades: bool = Field(
        default=False,
        description=(
            "Automatically reset enriched reviews to pending when the enrichment model changes. "
            "Disabled by default to avoid mass re-enrichment on model swaps."
        ),
    )
    enrichment_max_tokens: int = Field(default=2048, description="Max LLM output tokens")
    enrichment_local_only: bool = Field(default=False, description="Force local LLM only")
    enrichment_openrouter_model: str = Field(
        default="anthropic/claude-haiku-4-5",
        description=(
            "OpenRouter model for B2B enrichment (structured extraction)."
        ),
    )

    # Hybrid two-pass enrichment (Tier 1 local + Tier 2 local)
    enrichment_schema_version: int = Field(
        default=3,
        description="Current enrichment schema version (1 = original LLM inference, 2 = extract plus Tier 2 LLM, 3 = Tier 1 extract plus conditional Tier 2 plus deterministic compute)",
    )
    evidence_map_path: str = Field(
        default="",
        description="Path to evidence_map.yaml (empty = use default at atlas_brain/reasoning/evidence_map.yaml)",
    )
    enrichment_tier1_vllm_url: str = Field(
        default="http://localhost:8082",
        description="vLLM server URL for Tier 1 deterministic extraction",
    )
    enrichment_tier1_model: str = Field(
        default="stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ",
        description="Model name on vLLM server for Tier 1 extraction",
    )
    enrichment_tier1_max_tokens: int = Field(
        default=1024,
        description="Max output tokens for Tier 1 vLLM extraction",
    )
    enrichment_tier1_timeout_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="HTTP timeout for Tier 1 vLLM extraction requests",
    )
    enrichment_tier2_vllm_url: str = Field(
        default="",
        description="vLLM server URL for Tier 2 extraction (empty = reuse Tier 1 URL)",
    )
    enrichment_tier2_model: str = Field(
        default="",
        description="Model for Tier 2 vLLM extraction (empty = reuse Tier 1 model)",
    )
    enrichment_tier2_openrouter_model: str = Field(
        default="",
        description="OpenRouter model for Tier 2 extraction (empty = reuse enrichment_openrouter_model)",
    )
    enrichment_tier2_max_tokens: int = Field(
        default=1536,
        description="Max output tokens for Tier 2 extraction",
    )
    enrichment_tier2_strict_sources: str = Field(
        default="gartner,peerspot",
        description=(
            "Comma-separated sources that require stronger Tier 1 evidence before Tier 2 classification fires"
        ),
    )
    enrichment_tier2_strict_min_complaints: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Minimum Tier 1 complaint count that qualifies strict sources for Tier 2",
    )
    enrichment_tier2_strict_min_quotes: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Minimum Tier 1 quote count that helps qualify strict sources for Tier 2",
    )
    enrichment_tier2_timeout_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="HTTP timeout for Tier 2 extraction requests",
    )
    enrichment_anthropic_batch_enabled: bool = Field(
        default=True,
        description="Allow B2B enrichment extraction to use Anthropic Message Batches when available",
    )
    enrichment_anthropic_batch_min_items: int = Field(
        default=2,
        ge=1,
        le=10000,
        description="Minimum number of enrichment rows required before using Anthropic batching",
    )
    enrichment_tier1_connect_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="HTTP connect timeout for Tier 1 vLLM extraction requests",
    )
    # Account resolution
    account_resolution_batch_size: int = Field(
        default=1000,
        ge=1,
        le=5000,
        description="Max reviews to resolve per batch in account resolution task",
    )
    account_resolution_backfill_min_confidence: str = Field(
        default="medium",
        description="Minimum confidence label to backfill reviewer_company (high/medium/low)",
    )
    account_resolution_eligible_statuses: list[str] = Field(
        default=["enriched", "no_signal", "quarantined"],
        description=(
            "Review enrichment statuses eligible for deterministic account resolution"
        ),
    )
    account_resolution_source_priority: list[str] = Field(
        default=["g2", "gartner", "capterra", "software_advice", "trustpilot"],
        description=(
            "Preferred source order when draining deterministic account-resolution candidates"
        ),
    )
    account_resolution_excluded_sources: list[str] = Field(
        default=["capterra", "software_advice", "trustpilot", "trustradius"],
        description=(
            "Sources excluded from deterministic account resolution because they lack reliable identity signals"
        ),
    )
    account_resolution_interval_seconds: int = Field(
        default=600,
        description="Account resolution task polling interval (seconds)",
    )
    account_resolution_max_profile_fetches: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Max profile fetches per batch (HN + GitHub combined). GitHub free tier allows 60 req/hr.",
    )
    account_resolution_profile_fetch_concurrency: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max concurrent HN/GitHub profile fetches. 10 concurrent x 5s timeout ~= 25s for 50 profiles.",
    )
    account_resolution_profile_fetch_timeout: float = Field(
        default=5.0,
        ge=0.5,
        le=30.0,
        description="Per-request timeout (seconds) for HN and GitHub profile fetches.",
    )

    enrichment_repair_enabled: bool = Field(
        default=False,
        description="Enable strategic adjudication pass for already-enriched B2B reviews",
    )
    enrichment_repair_interval_seconds: int = Field(
        default=900,
        description="Polling interval for weak or high-salience enriched reviews",
    )
    enrichment_repair_max_per_batch: int = Field(
        default=25,
        description="Max enriched reviews to repair per batch",
    )
    enrichment_repair_max_rounds_per_run: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Max repair claim rounds to process per scheduled run",
    )
    enrichment_repair_concurrency: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Concurrent review repairs allowed within a single batch",
    )
    enrichment_repair_max_attempts: int = Field(
        default=2,
        description="Max adjudication attempts per enriched review",
    )
    enrichment_repair_model: str = Field(
        default="",
        description="Local vLLM model for strategic adjudication pass",
    )
    enrichment_repair_max_tokens: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Max completion tokens for the narrow field-repair extraction pass",
    )
    enrichment_repair_anthropic_batch_enabled: bool = Field(
        default=True,
        description="Allow B2B enrichment repair extraction to use Anthropic Message Batches when available",
    )
    enrichment_repair_anthropic_batch_min_items: int = Field(
        default=2,
        ge=1,
        le=10000,
        description="Minimum number of repair rows required before using Anthropic batching",
    )
    enrichment_repair_strict_discussion_sources: str = Field(
        default="reddit",
        description=(
            "Comma-separated community discussion sources that require stronger commercial signal before repair"
        ),
    )
    enrichment_repair_strict_discussion_content_types: list[str] = Field(
        default=["community_discussion", "insider_account", "comment"],
        description=(
            "Content types subject to strict discussion-source repair gating"
        ),
    )
    enrichment_repair_strict_discussion_skip_limit: int = Field(
        default=500,
        ge=1,
        le=5000,
        description=(
            "Max low-signal strict-discussion reviews to terminally skip per repair run"
        ),
    )
    enrichment_repair_min_urgency: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Minimum urgency score for repair unless leave/eval pressure is already present",
    )
    enrichment_repair_orphan_timeout_minutes: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Minutes before a row stuck in repairing status is recovered as orphaned",
    )
    enrichment_repair_failure_rate_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Fraction of failed rows in a single round that triggers the circuit breaker",
    )
    enrichment_repair_no_progress_max_rounds: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Consecutive rounds with zero promotions before circuit breaker trips",
    )
    evidence_default_analysis_window_days: int = Field(
        default=30,
        ge=7,
        le=365,
        description="Default analysis window in days for evidence explorer witness queries",
    )
    enrichment_low_fidelity_enabled: bool = Field(
        default=True,
        description="Enable deterministic quarantine of low-fidelity enriched rows",
    )
    enrichment_low_fidelity_noisy_sources: str = Field(
        default="hackernews,quora,twitter,github,stackoverflow",
        description="Comma-separated noisy sources eligible for low-fidelity quarantine heuristics",
    )

    # Intelligence aggregation
    intelligence_enabled: bool = Field(default=True, description="Enable churn intelligence aggregation")
    intelligence_cron: str = Field(default="0 21 * * *", description="Daily churn intelligence (9 PM)")
    intelligence_max_tokens: int = Field(default=4096, description="Max output tokens for churn intelligence LLM call")
    intelligence_window_days: int = Field(default=30, description="Days of enriched reviews to analyze")
    intelligence_recent_window_days: int = Field(default=30, description="Recent window for evidence vault trend/recency calculations")
    intelligence_evidence_vault_supporting_review_limit: int = Field(
        default=5,
        description="Max supporting review IDs stored per evidence-vault evidence row",
    )
    intelligence_evidence_vault_segment_limit: int = Field(
        default=3,
        description="Max affected company-size segments stored per evidence-vault evidence row",
    )
    intelligence_evidence_vault_role_limit: int = Field(
        default=3,
        description="Max affected roles stored per evidence-vault evidence row",
    )
    intelligence_evidence_vault_trend_accelerating_ratio: float = Field(
        default=1.25,
        description="Recent/prior ratio needed to mark evidence-vault trend direction as accelerating",
    )
    intelligence_evidence_vault_trend_declining_ratio: float = Field(
        default=0.75,
        description="Recent/prior ratio at or below which evidence-vault trend is marked declining",
    )
    intelligence_evidence_vault_trend_new_min_recent: int = Field(
        default=2,
        description="Minimum recent mentions required to classify evidence-vault trend direction as new",
    )
    intelligence_min_reviews: int = Field(default=3, description="Min reviews per vendor to include")
    intelligence_source_allowlist: str = Field(
        default=(
            "g2,gartner,peerspot,"
            "getapp,reddit,hackernews,github,stackoverflow,slashdot"
        ),
        description="Sources allowed in churn intelligence aggregation (comma-separated)",
    )
    intelligence_executive_sources: str = Field(
        default="g2,gartner,peerspot,getapp",
        description="High-signal sources for executive-facing outputs (weekly_churn_feed, displacement, timeline)",
    )
    deprecated_review_sources: str = Field(
        default="capterra,software_advice,trustpilot,trustradius",
        description="Sources deprecated from churn intelligence, blogs, and related downstream B2B review selection",
    )
    intelligence_llm_backend: str = Field(
        default="vllm",
        description=(
            "LLM backend for intelligence synthesis. "
            "'vllm' = local vLLM (primary) with Anthropic fallback. "
            "'anthropic' = Anthropic Sonnet only. "
            "'auto' = use the default synthesis workload routing."
        ),
    )
    intelligence_exploratory_enabled: bool = Field(
        default=True,
        description="Enable exploratory_overview generation when the LLM payload fits the context budget",
    )
    intelligence_exploratory_max_tokens: int = Field(
        default=4096,
        description="Max output tokens for exploratory_overview LLM generation",
    )
    intelligence_exploratory_char_budget: int = Field(
        default=100000,
        description="Approximate JSON character budget for exploratory_overview input payload trimming",
    )
    intelligence_exploratory_vendor_limit: int = Field(
        default=18,
        description="Max vendor score rows to include in exploratory_overview payload",
    )
    temporal_analysis_vendor_limit: int = Field(
        default=100,
        description="Max vendors for temporal analysis per intelligence run (independent of LLM payload trimming)",
    )
    intelligence_exploratory_high_intent_limit: int = Field(
        default=8,
        description="Max high-intent company rows to include in exploratory_overview payload",
    )
    intelligence_exploratory_generic_limit: int = Field(
        default=12,
        description="Max rows per generic exploratory dataset such as pain, displacement, and feature gaps",
    )
    intelligence_exploratory_quote_vendor_limit: int = Field(
        default=10,
        description="Max vendor quote bundles to include in exploratory_overview payload",
    )
    intelligence_exploratory_quotes_per_vendor: int = Field(
        default=2,
        description="Max quotes to keep per vendor in exploratory_overview payload",
    )
    intelligence_exploratory_use_case_limit: int = Field(
        default=8,
        description="Max rows per use-case distribution subtype in exploratory_overview payload",
    )
    intelligence_exploratory_company_limit: int = Field(
        default=5,
        description="Max company entries to keep per vendor in exploratory_overview payload",
    )
    intelligence_exploratory_prior_report_limit: int = Field(
        default=1,
        description="Max prior reports to include in exploratory_overview payload",
    )

    # Churn thresholds
    high_churn_urgency_threshold: int = Field(default=7, description="Urgency score >= this = high churn risk")
    enterprise_only: bool = Field(default=False, description="Only include enterprise-segment reviews")

    # Aggregation thresholds
    negative_review_threshold: float = Field(default=0.5, description="Rating ratio below this is negative")
    feature_gap_min_mentions: int = Field(default=2, description="Min mentions to include a feature gap")
    quotable_phrase_min_urgency: float = Field(default=4.5, description="Min urgency for quotable phrases (4.5 = moderate+ with indicator-based scoring)")
    timeline_signals_limit: int = Field(default=50, description="Max timeline signal rows per run")
    prior_reports_limit: int = Field(default=4, description="Prior reports for trend comparison")

    # Enrichment tuning
    review_truncate_length: int = Field(default=3000, description="Max review text length before truncation")

    # Customer context enrichment
    context_enrichment_enabled: bool = Field(
        default=True,
        description="Include B2B churn signals in customer context lookups",
    )

    # Keyword search volume signals (Google Trends)
    keyword_signal_enabled: bool = Field(default=False, description="Enable Google Trends keyword signal collection")
    keyword_signal_cron: str = Field(default="0 6 * * 1", description="Keyword signal schedule (Monday 6 AM)")
    keyword_spike_threshold_pct: float = Field(default=50.0, description="Volume change % to flag as spike")
    keyword_query_delay_seconds: float = Field(default=15.0, description="Delay between vendor queries (rate limit)")
    keyword_max_vendors_per_run: int = Field(default=20, description="Max vendors to query per run")
    keyword_geo: str = Field(default="US", description="Google Trends geo region")
    keyword_retention_days: int = Field(default=364, description="Days to retain keyword signal snapshots")

    # Product profiles
    product_profile_enabled: bool = Field(default=True, description="Enable product profile generation")
    product_profile_cron: str = Field(default="30 21 * * *", description="Product profile schedule (9:30 PM)")
    product_profile_min_reviews: int = Field(default=5, description="Min enriched reviews to generate a profile")
    product_profile_max_tokens: int = Field(default=1024, description="Max LLM output tokens for profile synthesis")
    product_profile_vllm_url: str = Field(
        default="http://localhost:8082",
        description="vLLM server URL for profile synthesis",
    )
    product_profile_vllm_model: str = Field(
        default="stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ",
        description="vLLM model for profile synthesis",
    )
    product_profile_llm_backend: str = Field(
        default="vllm",
        description="LLM backend for product profiles: 'vllm' (local) or 'openrouter'",
    )
    product_profile_openrouter_model: str = Field(
        default="openai/gpt-oss-120b",
        description="OpenRouter model for product profile synthesis",
    )
    product_profile_anthropic_batch_enabled: bool = Field(
        default=True,
        description="Allow B2B product profile synthesis to use Anthropic Message Batches when available",
    )
    product_profile_anthropic_batch_min_items: int = Field(
        default=2,
        ge=1,
        le=10000,
        description="Minimum number of vendor profiles required before using Anthropic batching",
    )
    product_profile_cache_trace_enabled: bool = Field(
        default=False,
        description="Emit info-level exact-cache fingerprint logs for product profile synthesis debugging",
    )

    # Blog post generation
    blog_post_enabled: bool = Field(default=False, description="Enable B2B blog post generation")
    blog_post_cron: str = Field(default="0 23 * * 1-5", description="Blog post schedule (weekdays 11 PM, 5x/week)")
    blog_post_max_tokens: int = Field(default=16384, description="Max LLM output tokens for blog post (reasoning models need extra budget)")
    blog_post_max_per_run: int = Field(default=1, description="Max blog posts per run")
    blog_post_timeout_seconds: int = Field(default=1800, description="Task timeout for blog post generation")
    blog_post_ui_path: str = Field(default="", description="Path to atlas-churn-ui blog content dir")
    blog_base_url: str = Field(default="https://churnsignals.co", description="Base URL for B2B blog (full URLs in campaign emails)")
    blog_post_openrouter_model: str = Field(default="openai/gpt-oss-120b", description="OpenRouter model for blog generation")
    blog_post_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for B2B blog post generation",
    )
    blog_post_anthropic_batch_enabled: bool = Field(
        default=True,
        description="Allow first-pass B2B blog generation to use Anthropic Message Batches when Anthropic is available",
    )
    blog_post_anthropic_batch_min_items: int = Field(
        default=2,
        ge=1,
        le=10000,
        description="Minimum number of blog posts required before using Anthropic batching for first-pass generation",
    )
    blog_quality_pass_score: int = Field(
        default=70,
        ge=0,
        le=100,
        description="Minimum blog quality score required to pass the deterministic quality gate",
    )
    blog_specificity_require_anchor_support: bool = Field(
        default=True,
        description="Require blog drafts to use witness-backed anchors when concrete reasoning anchors are available",
    )
    blog_specificity_min_anchor_hits: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Minimum number of witness-backed specificity signal groups a blog draft must hit when anchors are available",
    )
    blog_specificity_require_timing_or_numeric_when_available: bool = Field(
        default=True,
        description="Require blog drafts to mention a timing or numeric anchor when one is available in the reasoning packet",
    )
    blog_evidence_anchor_low_signal_labels: list[str] = Field(
        default_factory=lambda: [
            "ux",
            "ui",
            "review_context",
            "named_org",
            "positive_anchor",
            "complaint",
            "overall_dissatisfaction",
        ],
        description="Low-signal witness labels that should not appear as customer-facing blog evidence anchors",
    )
    blog_publish_revalidate_enabled: bool = Field(
        default=True,
        description="Re-run the deterministic blog quality and specificity gate before admin publish",
    )
    blog_quality_backfill_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Default lookback window in days for recent blog quality backfills",
    )
    # Blog auto-deploy (git push + Vercel deploy hook)
    blog_auto_deploy_enabled: bool = Field(default=False, description="Auto git-push + Vercel deploy after B2B blog publish")
    blog_auto_deploy_branch: str = Field(default="main", description="Git branch to push B2B blog commits to")
    blog_auto_deploy_hook_url: str = Field(default="", description="Vercel deploy hook URL for B2B blog")
    # Blog source filtering
    blog_source_allowlist: str = Field(
        default=(
            "g2,gartner,peerspot,"
            "getapp,reddit,hackernews,github,stackoverflow,slashdot"
        ),
        description="Sources to include in blog data queries",
    )
    # Regeneration mode -- re-process existing drafts through fixed pipeline
    blog_post_regenerate_mode: bool = Field(default=False, description="When True, regenerate existing draft posts instead of selecting new topics")
    blog_post_min_words_default: int = Field(
        default=1800,
        ge=500,
        le=5000,
        description="Default minimum word count required for a blog draft to pass the deterministic quality gate",
    )
    blog_post_target_words_default: int = Field(
        default=2300,
        ge=500,
        le=5000,
        description="Default SEO target word count used as a warning threshold for blog drafts",
    )
    blog_post_min_words_by_topic: dict[str, int] = Field(
        default_factory=lambda: {
            "vendor_showdown": 2000,
            "market_landscape": 1900,
            "best_fit_guide": 1900,
            "vendor_deep_dive": 1800,
            "vendor_alternative": 1700,
            "churn_report": 1700,
            "pain_point_roundup": 1700,
            "pricing_reality_check": 1600,
            "migration_guide": 1500,
            "switching_story": 1500,
        },
        description="Per-topic minimum word counts for the blog quality gate",
    )
    blog_post_target_words_by_topic: dict[str, int] = Field(
        default_factory=lambda: {
            "vendor_showdown": 2600,
            "market_landscape": 2600,
            "best_fit_guide": 2500,
            "vendor_deep_dive": 2400,
            "vendor_alternative": 2300,
            "churn_report": 2300,
            "pain_point_roundup": 2300,
            "pricing_reality_check": 2200,
            "migration_guide": 2100,
            "switching_story": 2100,
        },
        description="Per-topic SEO target word counts used as warning thresholds for blog drafts",
    )
    blog_post_max_rejection_retries: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Max rejected reruns allowed after an initial rejection before permanent block",
    )
    blog_post_rejection_cooldown_hours: int = Field(
        default=24,
        ge=0,
        le=720,
        description="Minimum hours before an autonomously rejected blog slug can be attempted again",
    )
    blog_post_borderline_shortfall_max_words: int = Field(
        default=60,
        ge=0,
        le=500,
        description="Maximum word-count shortfall eligible for deterministic coverage-snapshot repair during blog quality cleanup",
    )
    blog_min_quotes_by_topic: dict[str, int] = Field(
        default_factory=lambda: {
            "vendor_showdown": 4,
            "vendor_deep_dive": 3,
            "market_landscape": 4,
            "best_fit_guide": 4,
            "migration_guide": 3,
        },
        description="Per-topic minimum quotable-review counts required before blog generation is allowed",
    )
    blog_min_vendor_profiles_by_topic: dict[str, int] = Field(
        default_factory=lambda: {
            "market_landscape": 4,
            "best_fit_guide": 4,
        },
        description="Per-topic minimum vendor-profile breadth required before multi-vendor blog generation is allowed",
    )
    blog_min_sections_by_topic: dict[str, int] = Field(
        default_factory=lambda: {
            "vendor_showdown": 6,
            "vendor_deep_dive": 6,
            "market_landscape": 7,
            "best_fit_guide": 7,
            "migration_guide": 5,
        },
        description="Per-topic minimum blueprint section counts required before blog generation is allowed",
    )

    # Historical snapshots
    snapshot_enabled: bool = Field(default=True, description="Enable daily vendor health snapshots")
    snapshot_retention_days: int = Field(default=365, description="Days to retain vendor snapshots")
    change_detection_enabled: bool = Field(default=True, description="Enable structural change event detection")
    change_event_retention_days: int = Field(default=365, description="Days to retain change events")

    # Vendor intelligence briefings
    vendor_briefing_enabled: bool = Field(default=True, description="Enable vendor intelligence briefing emails")
    vendor_briefing_booking_url: str = Field(default="https://churnsignals.co", description="Booking URL for briefing CTA button")
    vendor_briefing_sender_name: str = Field(default="Atlas Intelligence", description="Display name for briefing sender")
    vendor_briefing_cooldown_days: int = Field(default=7, description="Min days between briefings to same vendor")
    vendor_briefing_max_per_batch: int = Field(default=10, description="Max briefings per batch send run")
    vendor_briefing_timeout_seconds: int = Field(default=1800, description="Task timeout for vendor briefing batch")
    vendor_briefing_scheduled_analyst_enrichment_enabled: bool = Field(
        default=False,
        description="Allow scheduled vendor briefing batches to call analyst LLM enrichment",
    )
    vendor_briefing_scheduled_account_cards_reasoning_depth: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Reasoning depth for account cards in scheduled vendor briefing batches",
    )
    vendor_briefing_account_cards_enabled: bool = Field(default=True, description="Generate account cards in briefings")
    vendor_briefing_account_cards_max: int = Field(default=3, description="Max account cards per briefing")
    vendor_briefing_account_cards_reasoning_depth: int = Field(default=2, description="Reasoning depth for card enrichment (0=baseline, 1=CoT, 2=multi-pass)")
    vendor_briefing_account_cards_adaptive_depth: bool = Field(default=True, description="Adaptively select reasoning depth per account based on urgency and data richness")

    # Analyst enrichment (OpenRouter)
    openrouter_api_key: str = Field(default="", description="OpenRouter API key for analyst enrichment")
    briefing_analyst_model: str = Field(default="openai/gpt-oss-120b", description="OpenRouter model for briefing analyst summary")
    scorecard_narrative_max_tokens: int = Field(
        default=300,
        ge=128,
        le=4096,
        description="Default max output tokens for vendor scorecard narrative generation",
    )
    scorecard_narrative_gpt_oss_max_tokens: int = Field(
        default=1600,
        ge=256,
        le=8192,
        description="Max output tokens for vendor scorecard narratives when the synthesis model is gpt-oss",
    )
    scorecard_narrative_deepseek_max_tokens: int = Field(
        default=1200,
        ge=256,
        le=8192,
        description="Max output tokens for vendor scorecard narratives when the synthesis model is DeepSeek",
    )
    llm_exact_cache_enabled: bool = Field(
        default=False,
        description="Enable exact-match Postgres caching for B2B/reporting LLM calls",
    )
    anthropic_batch_enabled: bool = Field(
        default=False,
        description="Enable Anthropic Message Batches for eligible B2B workloads",
    )
    anthropic_batch_poll_interval_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Polling interval for Anthropic Message Batch reconciliation",
    )
    anthropic_batch_timeout_seconds: float = Field(
        default=900.0,
        ge=30.0,
        le=86400.0,
        description="Max time to wait for an Anthropic Message Batch before falling back",
    )
    scorecard_anthropic_batch_enabled: bool = Field(
        default=True,
        description="Allow vendor scorecard narratives to use Anthropic Message Batches when available",
    )
    scorecard_anthropic_batch_min_items: int = Field(
        default=2,
        ge=1,
        le=10000,
        description="Minimum uncached scorecard narratives required before using Anthropic batching",
    )
    reasoning_synthesis_anthropic_batch_enabled: bool = Field(
        default=True,
        description="Allow vendor reasoning synthesis to use Anthropic Message Batches when available",
    )
    reasoning_synthesis_anthropic_batch_min_items: int = Field(
        default=2,
        ge=1,
        le=10000,
        description="Minimum vendor reasoning items required before using Anthropic batching",
    )
    cross_vendor_anthropic_batch_enabled: bool = Field(
        default=True,
        description="Allow cross-vendor reasoning synthesis to use Anthropic Message Batches when available",
    )
    cross_vendor_anthropic_batch_min_items: int = Field(
        default=2,
        ge=1,
        le=10000,
        description="Minimum cross-vendor reasoning items required before using Anthropic batching",
    )
    tenant_report_anthropic_batch_enabled: bool = Field(
        default=True,
        description="Allow tenant report synthesis chunks to use Anthropic Message Batches when available",
    )
    tenant_report_anthropic_batch_min_items: int = Field(
        default=2,
        ge=1,
        le=10000,
        description="Minimum tenant report synthesis chunks required before using Anthropic batching",
    )

    # Briefing gate (email capture for full report)
    vendor_briefing_gate_base_url: str = Field(default="https://churnsignals.co/report", description="Base URL for briefing gate landing page")
    vendor_briefing_gate_expiry_days: int = Field(default=7, description="Gate token expiry in days")

    reasoning_synthesis_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Max generation attempts per vendor reasoning synthesis, including validation-repair retries",
    )
    reasoning_synthesis_retry_delay_seconds: float = Field(
        default=0.5,
        ge=0.0,
        le=30.0,
        description="Delay between vendor reasoning synthesis retry attempts",
    )
    reasoning_synthesis_timeout_seconds: float = Field(
        default=180.0,
        ge=1.0,
        le=3600.0,
        description="Timeout for each vendor or cross-vendor reasoning LLM call",
    )
    reasoning_synthesis_max_stale_days: int = Field(
        default=3,
        ge=0,
        le=90,
        description="Classify unchanged vendor reasoning rows newer than this as fresh reuse and older rows as stale reuse",
    )
    reasoning_synthesis_max_input_tokens: int = Field(
        default=20000,
        ge=512,
        le=50000,
        description="Approximate max input tokens allowed for a single vendor reasoning synthesis prompt before lean fallback or rejection",
    )
    reasoning_synthesis_max_items_per_pool: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Max scored items per pool in the default vendor reasoning payload",
    )
    reasoning_synthesis_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Max completion tokens for vendor or cross-vendor reasoning synthesis calls",
    )
    reasoning_synthesis_model: str = Field(
        default="",
        description=(
            "OpenRouter model override for vendor reasoning synthesis. "
            "Empty = prefer settings.llm.openrouter_reasoning_model before "
            "falling back to the legacy reasoning-model defaults."
        ),
    )
    reasoning_synthesis_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Temperature for vendor and cross-vendor reasoning synthesis calls",
    )
    reasoning_synthesis_feedback_limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Max validator issues to feed back into synthesis repair retries",
    )
    reasoning_retry_escalation_window_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Rolling window for escalating repeated recovered synthesis validation retries",
    )
    reasoning_retry_repeat_rule_threshold: int = Field(
        default=3,
        ge=2,
        le=20,
        description="Escalate when the same vendor and validation rule hit this many recovered retries within the escalation window",
    )
    reasoning_retry_cost_min_retries: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Minimum recovered retry count before token-cost escalation can trigger",
    )
    reasoning_retry_cost_tokens_threshold: int = Field(
        default=80000,
        ge=1000,
        le=1000000,
        description="Escalate recovered retry churn when retry-token spend crosses this threshold within the escalation window",
    )
    reasoning_witness_max_witnesses: int = Field(
        default=12,
        ge=1,
        le=50,
        description="Max witnesses included in each vendor reasoning packet",
    )
    reasoning_synthesis_lean_max_items_per_pool: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Max scored items per pool when vendor reasoning falls back to lean prompt mode",
    )
    reasoning_synthesis_lean_max_witnesses: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Max witnesses included when vendor reasoning falls back to lean prompt mode",
    )
    reasoning_synthesis_segment_candidate_limit: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max segment shortlist candidates included in witness-first section packets",
    )
    reasoning_synthesis_temporal_candidate_limit: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max temporal trigger candidates included in witness-first section packets",
    )
    reasoning_synthesis_displacement_candidate_limit: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max displacement destination candidates included in witness-first section packets",
    )
    reasoning_synthesis_account_candidate_limit: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max account shortlist candidates included in witness-first section packets",
    )
    reasoning_synthesis_category_candidate_limit: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Max category regime candidates included in witness-first section packets",
    )
    reasoning_synthesis_retention_candidate_limit: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max retention strength candidates included in witness-first section packets",
    )
    reasoning_synthesis_rerun_if_missing_packet_artifacts: bool = Field(
        default=True,
        description="Rerun vendor reasoning when the latest unchanged row is missing packet artifacts",
    )
    reasoning_synthesis_rerun_if_missing_reference_ids: bool = Field(
        default=True,
        description="Rerun vendor reasoning when the latest unchanged row is missing canonical reference ids",
    )
    reasoning_witness_highlight_limit: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Max witness highlights surfaced to downstream render consumers",
    )
    witness_specificity_min_score: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description="Minimum deterministic specificity score required before a witness can enter the normal witness pack",
    )
    witness_specificity_fallback_min_witnesses: int = Field(
        default=4,
        ge=0,
        le=20,
        description="Minimum witness count to recover with generic fallback when the specific witness pool is too thin",
    )
    witness_specificity_generic_patterns: list[str] = Field(
        default_factory=lambda: [
            "great tool",
            "great platform",
            "good tool",
            "good platform",
            "easy to use",
            "easy-to-use",
            "works well",
            "working well",
            "very helpful",
            "super helpful",
        ],
        description="Boilerplate phrases that reduce witness specificity",
    )
    witness_specificity_concrete_patterns: list[str] = Field(
        default_factory=lambda: [
            "pricing",
            "renewal",
            "seat",
            "integration",
            "workflow",
            "contract",
            "budget",
            "support",
            "security",
            "migration",
            "implementation",
            "downtime",
            "latency",
        ],
        description="Concrete pain or workflow anchors that increase witness specificity",
    )
    witness_specificity_short_excerpt_chars: int = Field(
        default=55,
        ge=1,
        le=500,
        description="Excerpt length below which anchor-free witnesses incur a specificity penalty",
    )
    witness_specificity_long_excerpt_chars: int = Field(
        default=80,
        ge=1,
        le=1000,
        description="Excerpt length above which a witness receives a small specificity bonus",
    )
    witness_specificity_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "currency": 2.0,
            "number": 1.0,
            "timing": 1.0,
            "competitor": 1.0,
            "reviewer_company": 1.0,
            "pain_category": 0.75,
            "replacement_mode": 0.75,
            "operating_model_shift": 0.75,
            "productivity_delta_claim": 0.75,
            "signal_type": 0.5,
            "long_excerpt": 0.5,
            "concrete_pattern": 0.5,
            "generic_phrase_penalty": 2.5,
            "short_excerpt_penalty": 1.5,
        },
        description="Deterministic witness specificity scoring weights and penalties",
    )
    reasoning_synthesis_enabled: bool = Field(
        default=True,
        description=(
            "Enable vendor reasoning synthesis as the canonical post-core reasoning pass. "
            "When true, legacy stratified vendor reasoning inside b2b_churn_intelligence "
            "is skipped to avoid duplicate LLM spend."
        ),
    )
    executive_summary_llm_enabled: bool = Field(default=False, description="Use LLM-synthesized executive summaries instead of deterministic templates")

    cross_vendor_max_battles: int = Field(default=5, ge=0, le=20, description="Max pairwise battle reasoning calls per run")
    cross_vendor_max_categories: int = Field(default=3, ge=0, le=10, description="Max category council reasoning calls per run")
    cross_vendor_max_asymmetry: int = Field(default=3, ge=0, le=10, description="Max resource-asymmetry reasoning calls per run")
    cross_vendor_battle_min_context_score: float = Field(default=2.0, ge=0.0, le=10.0, description="Min deterministic overlap score before a displacement pair gets battle reasoning")
    cross_vendor_category_min_vendors: int = Field(default=3, ge=1, le=20, description="Min reasoned vendors in a category before council reasoning")
    cross_vendor_category_min_context_vendors: int = Field(default=2, ge=1, le=20, description="Min context-rich vendors in a category before council reasoning")
    cross_vendor_category_min_displacement_intensity: float = Field(default=1.0, ge=0.0, le=100.0, description="Min displacement intensity before category council reasoning")
    cross_vendor_asymmetry_pressure_delta_max: float = Field(default=1.5, ge=0.0, le=10.0, description="Max avg urgency gap allowed when selecting asymmetry pairs")
    cross_vendor_asymmetry_review_ratio_min: float = Field(default=3.0, ge=1.0, le=100.0, description="Min review-count ratio that qualifies as resource divergence")
    cross_vendor_asymmetry_segment_divergence_bonus: float = Field(default=5.0, ge=0.0, le=20.0, description="Divergence bonus when vendors tilt toward different company-size segments")
    cross_vendor_asymmetry_min_divergence_score: float = Field(default=2.0, ge=0.0, le=50.0, description="Min divergence score before asymmetry reasoning")
    cross_vendor_asymmetry_min_context_score: float = Field(default=2.0, ge=0.0, le=10.0, description="Min deterministic overlap score before asymmetry reasoning")

    # Cross-vendor synthesis (runs in b2b_reasoning_synthesis after vendor synthesis)
    cross_vendor_synthesis_enabled: bool = Field(default=True, description="Enable cross-vendor synthesis (battles, councils, asymmetry) in the reasoning synthesis task")
    cross_vendor_synthesis_concurrency: int = Field(default=3, ge=1, le=10, description="Max concurrent cross-vendor synthesis LLM calls")
    cross_vendor_synthesis_attempts: int = Field(default=2, ge=1, le=5, description="Max generation attempts per cross-vendor packet")
    cross_vendor_synthesis_feedback_limit: int = Field(default=5, ge=1, le=10, description="Max validator issues to feed back per retry")
    cross_vendor_llm_max_input_tokens: int = Field(
        default=12000,
        ge=512,
        le=50000,
        description="Approximate max input tokens allowed for a single cross-vendor synthesis prompt before the run is rejected to control spend",
    )
    cross_vendor_category_vendor_summary_limit: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Max vendor summaries included in a category-council packet",
    )
    cross_vendor_category_flow_limit: int = Field(
        default=3,
        ge=0,
        le=50,
        description="Max displacement flows included in a category-council packet",
    )
    competitive_set_max_competitors: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max competitors allowed in one scoped competitive set",
    )
    competitive_set_refresh_interval_seconds: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="How often the scheduled competitive-set refresher scans for due scoped synthesis runs",
    )
    competitive_set_refresh_batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max competitive sets processed in one scheduled synthesis scanner run",
    )
    competitive_set_preview_lookback_days: int = Field(
        default=14,
        ge=1,
        le=90,
        description="Lookback window for competitive-set run cost previews based on recent synthesis history",
    )
    competitive_set_changed_vendors_only_default: bool = Field(
        default=True,
        description=(
            "Default changed-vendors-only policy for scoped competitive-set runs "
            "when the caller does not explicitly choose full refresh behavior"
        ),
    )
    reasoning_synthesis_scheduled_scope_strategy: str = Field(
        default="competitive_sets",
        description=(
            "Runtime strategy for scheduled b2b_reasoning_synthesis runs: "
            "'competitive_sets' scans due scoped sets, 'full_universe' keeps the legacy global run path"
        ),
    )

    scorecard_narrative_concurrency: int = Field(default=6, description="Max concurrent LLM calls during scorecard narrative generation")

    battle_card_llm_concurrency: int = Field(default=3, description="Max concurrent battle card sales copy LLM calls")
    battle_card_llm_attempts: int = Field(default=2, ge=1, le=5, description="Max generation attempts per battle card, including repair retries")
    battle_card_llm_retry_delay_seconds: float = Field(default=1.0, ge=0.0, le=30.0, description="Delay between battle card LLM attempts")
    battle_card_llm_feedback_limit: int = Field(default=5, ge=1, le=10, description="Max validator issues to feed back into battle card repair attempts")
    battle_card_anthropic_batch_enabled: bool = Field(
        default=True,
        description="Allow battle-card sales copy to use Anthropic Message Batches when the battle-card backend is set to anthropic",
    )
    battle_card_anthropic_batch_min_items: int = Field(
        default=2,
        ge=1,
        le=10000,
        description="Minimum battle-card sales-copy items required before using Anthropic batching",
    )
    battle_card_llm_backend: str = Field(
        default="auto",
        description=(
            "LLM backend for battle-card sales copy. "
            "'auto' = default synthesis routing, "
            "'anthropic' = Anthropic Sonnet only, "
            "'openrouter' = OpenRouter only."
        ),
    )
    battle_card_openrouter_model: str = Field(
        default="",
        description=(
            "OpenRouter model override for battle-card sales copy. "
            "Empty = inherit ATLAS_LLM__OPENROUTER_REASONING_MODEL."
        ),
    )
    battle_card_llm_max_tokens: int = Field(default=16384, ge=256, le=32768, description="Max output tokens for battle card sales copy (reasoning models need extra budget)")
    battle_card_llm_max_input_tokens: int = Field(
        default=25000,
        ge=512,
        le=50000,
        description="Approximate max input tokens allowed for a single battle-card sales-copy prompt before the LLM step falls back deterministically",
    )
    battle_card_render_anchor_examples_per_bucket: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Max witness-backed anchor examples to surface per anchor bucket in battle-card sales-copy prompts",
    )
    battle_card_render_witness_highlights_limit: int = Field(
        default=4,
        ge=0,
        le=12,
        description="Max witness highlights to include in battle-card sales-copy prompts",
    )
    battle_card_render_reference_ids_limit: int = Field(
        default=12,
        ge=0,
        le=40,
        description="Max metric or witness reference ids to include per list in battle-card sales-copy prompts",
    )
    battle_card_render_cross_vendor_battles_limit: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Max cross-vendor battle summaries to include in battle-card sales-copy prompts",
    )
    battle_card_render_high_intent_companies_limit: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Max high-intent accounts to include in battle-card sales-copy prompts",
    )
    battle_card_render_reframes_limit: int = Field(
        default=3,
        ge=0,
        le=8,
        description="Max competitive reframes to include in compact displacement reasoning for battle-card sales-copy prompts",
    )
    battle_card_render_priority_segments_limit: int = Field(
        default=3,
        ge=0,
        le=8,
        description="Max priority segments to include in compact vendor reasoning for battle-card sales-copy prompts",
    )
    battle_card_render_strengths_limit: int = Field(
        default=3,
        ge=0,
        le=8,
        description="Max incumbent strengths to include in compact battle-card retention context",
    )
    battle_card_render_data_gaps_limit: int = Field(
        default=4,
        ge=0,
        le=12,
        description="Max data gaps or confidence-limit notes to include per compact battle-card reasoning section",
    )
    battle_card_llm_temperature: float = Field(default=0.5, ge=0.0, le=1.5, description="Sampling temperature for battle card sales copy generation")
    battle_card_llm_timeout_seconds: float = Field(default=90.0, ge=5.0, le=300.0, description="Timeout for a single battle card LLM generation attempt")
    battle_card_cache_confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Confidence assigned to validated battle card sales copy cache entries")
    battle_card_high_priority_score_min: float = Field(default=60.0, ge=0.0, le=100.0, description="Min churn pressure score required before battle-card copy can use high-priority language")
    battle_card_high_priority_urgency_min: float = Field(default=5.0, ge=0.0, le=10.0, description="Min average urgency required before battle-card copy can use high-priority language")
    battle_card_feature_gap_headline_min_mentions: int = Field(default=5, ge=1, le=100, description="Min feature-gap mention count before a battle-card headline can elevate that gap directly")
    battle_card_quality_max_stale_days: int = Field(
        default=2,
        ge=0,
        le=30,
        description="Max allowed staleness (days) before battle-card quality gate hard-blocks publishing",
    )
    battle_card_quality_eval_divergence_warn_delta: int = Field(
        default=25,
        ge=1,
        le=1000,
        description="Warn when active-evaluation metrics differ by at least this absolute amount",
    )
    battle_card_quality_min_high_intent_urgency: float = Field(
        default=7.0,
        ge=0.0,
        le=10.0,
        description="Min urgency required for a high-intent account to qualify for strict battle-card readiness",
    )
    battle_card_quality_required_stages: list[str] = Field(
        default=["evaluation", "renewal_decision"],
        description="Allowed buying stages for high-intent accounts in strict battle-card readiness",
    )
    battle_card_quality_allow_global_eval_fallback: bool = Field(
        default=True,
        description="Allow strict readiness to pass account-stage requirement when global active-eval evidence exists",
    )
    battle_card_quality_min_total_plays: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum recommended plays required for final battle-card readiness",
    )
    battle_card_quality_min_actionable_plays: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum recommended plays that include targeting, timing, and CTA language",
    )
    synthesis_reference_confidence_min: float = Field(default=0.6, ge=0.0, le=1.0, description="Min reasoning or cross-vendor confidence before synthesis may reference a structured conclusion directly")
    synthesis_expert_take_max_words: int = Field(default=80, ge=20, le=200, description="Max word count for synthesized scorecard expert_take narratives")
    vendor_scorecard_limit: int = Field(
        default=15,
        ge=1,
        le=500,
        description="Default max vendors included in the unscoped vendor_scorecard report artifact",
    )
    battle_card_leaving_patterns: list[str] = Field(
        default=[
            "customers are leaving",
            "customer are leaving",
            "are leaving for",
            "capturing defectors",
            "capture defectors",
            "defectors",
        ],
        description="Battle-card phrases that imply explicit switching and require switch-count evidence",
    )

    # Accounts in motion
    company_signal_skip_deprecated_sources: bool = Field(
        default=True,
        description="Exclude globally deprecated review sources from canonical company-signal and named-account products",
    )
    company_signal_low_trust_sources: list[str] = Field(
        default=["reddit"],
        description="Low-trust sources that require a higher confidence threshold before becoming canonical named-account signals",
    )
    company_signal_low_trust_min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum unit-confidence required for low-trust company signals to enter canonical named-account products",
    )
    high_intent_require_signal_evidence: bool = Field(
        default=True,
        description="Require explicit churn/evaluation/renewal signal evidence before reviewer-company rows enter named-account high-intent products",
    )
    accounts_in_motion_cron: str = Field(default="35 21 * * *", description="Cron for accounts-in-motion prospecting lists")
    accounts_in_motion_max_per_vendor: int = Field(default=25, ge=1, le=100, description="Max accounts per vendor in accounts_in_motion report")
    accounts_in_motion_feed_max_total: int = Field(default=100, ge=1, le=200, description="Max total tenant feed rows returned by the aggregated accounts_in_motion endpoint")
    accounts_in_motion_min_urgency: float = Field(default=5.0, ge=0, le=10, description="Min urgency to include an account in motion")
    accounts_in_motion_signal_metadata_min_confidence: float = Field(default=6.0, ge=0, le=10, description="Minimum normalized confidence required for company-signal metadata fallback rows to seed accounts_in_motion")
    accounts_in_motion_reddit_insider_min_confidence: float = Field(default=6.0, ge=0, le=10, description="Minimum normalized confidence required for reddit insider_account company signals to seed accounts_in_motion")
    accounts_in_motion_repeat_evidence_bonus: int = Field(default=3, ge=0, le=20, description="Bonus points added per extra supporting review for an account in motion")
    accounts_in_motion_repeat_evidence_bonus_max: int = Field(default=6, ge=0, le=30, description="Max total repeat-evidence bonus for an account in motion")
    accounts_in_motion_low_confidence_threshold: float = Field(default=6.0, ge=0, le=10, description="Confidence below this threshold incurs a quality penalty in accounts_in_motion scoring")
    accounts_in_motion_low_confidence_penalty: int = Field(default=6, ge=0, le=30, description="Penalty applied when confidence is below the configured threshold")
    accounts_in_motion_missing_domain_penalty: int = Field(default=8, ge=0, le=30, description="Penalty for accounts without a matched company domain")
    accounts_in_motion_missing_title_penalty: int = Field(default=4, ge=0, le=30, description="Penalty for accounts without a matched buyer title")
    accounts_in_motion_missing_quote_penalty: int = Field(default=4, ge=0, le=30, description="Penalty for accounts without a company-matched quote")
    accounts_in_motion_invalid_alternative_terms: list[str] = Field(default=["bare metal"], description="Configured non-vendor alternative terms to drop from accounts_in_motion alternatives")

    # Challenger brief
    challenger_brief_cron: str = Field(default="40 21 * * *", description="Cron for challenger brief report (runs after all other follow-ups)")
    challenger_brief_min_displacement_mentions: int = Field(default=3, ge=1, le=100, description="Min displacement mentions to generate a challenger brief")
    challenger_brief_max_pairs_per_incumbent: int = Field(default=5, ge=1, le=20, description="Top N challengers per incumbent to generate briefs for")
    challenger_brief_max_target_accounts: int = Field(default=15, ge=1, le=100, description="Max accounts in target_accounts section of challenger brief")
    challenger_brief_report_fallback_days: int = Field(default=7, ge=1, le=30, description="Max age of persisted battle-card or accounts-in-motion artifacts that challenger briefs may reuse")
    challenger_brief_quote_fallback_limit: int = Field(default=5, ge=1, le=20, description="Max review-sourced pain quotes to include when a battle card has no quotes")
    challenger_brief_quote_candidate_limit: int = Field(default=25, ge=5, le=100, description="Max enriched review rows to inspect when assembling challenger-brief quote fallbacks")
    challenger_brief_quote_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Jaccard similarity threshold used to deduplicate review-sourced quote fallbacks")

    # Follow-up task scheduling (staggered after core)
    reports_cron: str = Field(default="30 21 * * *", description="Cron for churn reports follow-up task")
    battle_cards_cron: str = Field(default="30 21 * * *", description="Cron for battle cards follow-up task")
    article_correlation_cron: str = Field(default="35 21 * * *", description="Cron for article correlation follow-up task")


class B2BAlertConfig(BaseSettings):
    """B2B churn signal spike alert configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_B2B_ALERT_", env_file=ENV_FILES, extra="ignore"
    )

    enabled: bool = Field(default=False, description="Enable churn signal spike alerts")
    email_enabled: bool = Field(default=True, description="Send churn alerts to tenant owners via email when configured")
    sender_name: str = Field(default="Atlas Intelligence", description="Display name for churn alert emails")
    dashboard_base_url: str = Field(default="", description="Optional dashboard URL included in churn alert emails")
    signal_count_threshold: int = Field(default=3, description="New signals to trigger alert")
    urgency_spike_threshold: float = Field(default=1.5, description="Avg urgency increase to trigger alert")
    min_reviews_for_urgency: int = Field(default=10, ge=1, le=100, description="Min reviews in 7-day window before avg_urgency can trigger alerts")
    cooldown_hours: int = Field(default=24, description="Min hours between alerts for same vendor")
    interval_seconds: int = Field(default=3600, description="Alert check interval (1 hour)")


class B2BWatchlistDeliveryConfig(BaseSettings):
    """Recurring email delivery for saved-view watchlist alerts."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_B2B_WATCHLIST_DELIVERY_", env_file=ENV_FILES, extra="ignore"
    )

    enabled: bool = Field(default=False, description="Enable recurring saved-view watchlist alert delivery")
    interval_seconds: int = Field(default=3600, ge=60, le=86400, description="How often to check for due watchlist alert deliveries")
    max_views_per_run: int = Field(default=25, ge=1, le=250, description="Max saved views processed in a single watchlist alert delivery run")
    stale_claim_seconds: int = Field(default=900, ge=60, le=86400, description="How long a scheduled watchlist delivery claim can remain processing before another worker may reclaim it")
    failed_retry_seconds: int = Field(default=3600, ge=60, le=86400 * 7, description="Delay before retrying a failed scheduled watchlist delivery")


class B2BReportDeliveryConfig(BaseSettings):
    """Recurring report-subscription delivery configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_B2B_REPORT_DELIVERY_", env_file=ENV_FILES, extra="ignore"
    )

    enabled: bool = Field(default=False, description="Enable recurring report-subscription delivery")
    sender_name: str = Field(default="Atlas Intelligence", description="Display name for recurring report emails")
    dashboard_base_url: str = Field(default="", description="Optional frontend base URL used in report-delivery links")
    interval_seconds: int = Field(default=3600, ge=60, le=86400, description="How often to check for due report subscriptions")
    max_subscriptions_per_run: int = Field(default=25, ge=1, le=250, description="Max due subscriptions processed in a single task run")
    max_reports_per_delivery: int = Field(default=6, ge=1, le=20, description="Max persisted artifacts included in one recurring library delivery")
    stale_claim_seconds: int = Field(default=900, ge=60, le=86400, description="How long an in-flight delivery claim can sit before another worker may reclaim it")
    fresh_hours: int = Field(default=72, ge=1, le=24 * 30, description="Artifacts newer than this are treated as fresh for delivery-policy checks")
    monitor_hours: int = Field(default=24 * 7, ge=1, le=24 * 90, description="Artifacts newer than this but older than fresh_hours are treated as monitor state")
    require_sales_ready_for_competitive: bool = Field(
        default=True,
        description="Require sales_ready quality status for battle-card style deliverables when quality metadata exists",
    )
    max_blocker_count: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Maximum blocker_count allowed for a deliverable to remain eligible",
    )
    max_open_review_count: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Maximum unresolved operator-review count allowed for a deliverable to remain eligible",
    )
    report_scope_overrides_library: bool = Field(
        default=True,
        description="Exclude artifacts from library deliveries when an enabled report-scoped subscription exists for the same artifact",
    )
    dry_run: bool = Field(
        default=False,
        description="Resolve due subscriptions and record delivery outcomes without sending email or advancing schedules",
    )
    canary_account_ids: str = Field(
        default="",
        description="Comma-separated or JSON list of account IDs allowed for live recurring delivery sends",
    )
    suppress_unchanged_deliveries: bool = Field(
        default=True,
        description="Skip recurring sends when the eligible artifact package has not materially changed since the last completed delivery",
    )
    max_send_attempts_per_recipient: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum send attempts per recipient before the delivery is marked partial or failed",
    )
    retry_backoff_seconds: int = Field(
        default=3,
        ge=0,
        le=300,
        description="Delay between recipient send retries",
    )


class B2BWebhookConfig(BaseSettings):
    """B2B outbound webhook delivery configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_B2B_WEBHOOK_", env_file=ENV_FILES, extra="ignore"
    )

    enabled: bool = Field(default=False, description="Enable outbound webhook delivery")
    timeout_seconds: int = Field(default=10, ge=1, le=30, description="HTTP timeout per delivery")
    max_retries: int = Field(default=3, ge=0, le=5, description="Max retry attempts on failure")
    retry_delay_seconds: int = Field(default=5, ge=1, le=600, description="Delay between retries")
    max_payload_bytes: int = Field(default=65536, description="Max payload size in bytes")
    delivery_log_retention_days: int = Field(default=30, ge=1, le=365, description="Days to keep delivery logs")
    min_change_event_severity: str = Field(
        default="moderate",
        description="Minimum severity level for change event webhook dispatch (low/moderate/high/critical). Events below threshold are still persisted.",
    )


class CRMEventConfig(BaseSettings):
    """Inbound CRM event ingestion configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_CRM_EVENT_", env_file=ENV_FILES, extra="ignore"
    )

    enabled: bool = Field(default=False, description="Enable CRM event ingestion")
    batch_size: int = Field(default=50, ge=1, le=500, description="Events processed per run")
    # Mapping from CRM deal stages to campaign outcomes
    stage_outcome_map: dict[str, str] = Field(
        default={
            "closed_won": "deal_won",
            "closed_lost": "deal_lost",
            "demo_scheduled": "meeting_booked",
            "meeting_booked": "meeting_booked",
            "proposal_sent": "deal_opened",
            "negotiation": "deal_opened",
            "qualified": "deal_opened",
            "disqualified": "disqualified",
        },
        description="Maps CRM deal stage names to campaign outcome values",
    )


class UniversalScrapeConfig(BaseSettings):
    """Universal web scraper configuration (data-agnostic LLM extraction)."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_UNIVERSAL_SCRAPE_", env_file=ENV_FILES, extra="ignore"
    )

    enabled: bool = Field(default=True, description="Enable universal scraper")
    default_concurrency: int = Field(default=3, ge=1, le=10, description="Default concurrent targets per job")
    default_llm_workload: str = Field(default="triage", description="Default LLM workload tier for extraction")
    max_pages_limit: int = Field(default=100, ge=1, description="Hard ceiling on pages per target")
    html_max_chars: int = Field(default=30000, description="Max chars of cleaned HTML sent to LLM")
    config_dir: str = Field(default="scrape_configs", description="Directory for JSON config files on disk")


class B2BScrapeConfig(BaseSettings):
    """B2B review scraping pipeline configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_B2B_SCRAPE_", env_file=ENV_FILES, extra="ignore"
    )

    enabled: bool = Field(default=False, description="Enable B2B review scraping pipeline")

    # Schedule
    intake_interval_seconds: int = Field(default=3600, description="Scrape polling interval (1 hour)")
    enrichment_on_scrape: bool = Field(default=True, description="Fire enrichment immediately after scraping new reviews (disable to save credits when enrichment model is failing)")
    max_targets_per_run: int = Field(default=0, description="Max targets to scrape per run (0 = unlimited)")
    source_allowlist: str = Field(
        default=(
            "g2,gartner,peerspot,"
            "getapp,reddit,hackernews,github,stackoverflow,slashdot"
        ),
        description="Sources allowed for automated scrape intake (comma-separated)",
    )
    deprecated_sources: str = Field(
        default="capterra,software_advice,trustpilot,trustradius",
        description="Sources deprecated from automated scrape intake and target planning",
    )
    source_fit_filter_enabled: bool = Field(
        default=True,
        description="Skip scrape targets whose source is a poor fit for the target product category",
    )
    source_fit_allow_conditional: bool = Field(
        default=True,
        description="Allow conditionally useful source/category pairs while blocking poor-fit pairs",
    )
    source_fit_allow_probation: bool = Field(
        default=True,
        description="Allow targets marked as source-fit probation when the fit is conditional",
    )
    source_fit_probation_priority: int = Field(
        default=3,
        description="Priority cap for conditionally seeded probation targets",
    )
    source_fit_probation_max_pages: int = Field(
        default=3,
        description="Max pages cap for conditionally seeded probation targets",
    )
    source_fit_probation_scrape_interval_hours: int = Field(
        default=72,
        description="Minimum scrape interval for conditionally seeded probation targets",
    )
    source_fit_probation_telemetry_lookback_days: int = Field(
        default=30,
        description="Lookback window for probation-target telemetry summaries",
    )
    source_fit_probation_actionable_urgency_min: float = Field(
        default=7.0,
        description="Urgency threshold used when counting actionable enriched reviews for probation telemetry",
    )
    source_low_yield_pruning_enabled: bool = Field(
        default=False,
        description="Enable automated disabling of scrape targets that repeatedly produce no inserted reviews",
    )
    source_low_yield_pruning_source: str = Field(
        default="twitter",
        description="Source slug to evaluate for low-yield pruning",
    )
    source_low_yield_pruning_lookback_runs: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of most recent runs to evaluate per target",
    )
    source_low_yield_pruning_min_runs: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Minimum observed runs required before a target can be disabled",
    )
    source_low_yield_pruning_max_inserted_total: int = Field(
        default=0,
        ge=0,
        le=10000,
        description="Disable targets whose inserted review sum across lookback runs is at or below this value",
    )
    source_low_yield_pruning_max_disable_per_run: int = Field(
        default=25,
        ge=1,
        le=1000,
        description="Safety cap on number of targets disabled per pruning run",
    )
    source_low_yield_pruning_interval_seconds: int = Field(
        default=21600,
        ge=300,
        le=604800,
        description="Scheduler interval for low-yield pruning task",
    )
    source_low_yield_pruning_dry_run: bool = Field(
        default=False,
        description="Run low-yield pruning in dry-run mode even when scheduled",
    )

    # Proxies (comma-separated URLs)
    proxy_datacenter_urls: str = Field(default="", description="Datacenter proxy URLs (comma-separated)")
    proxy_residential_urls: str = Field(default="", description="Residential proxy URLs (comma-separated)")
    proxy_residential_geo: str = Field(default="US", description="Residential proxy geo code")

    # Per-domain rate limits (RPM)
    g2_rpm: int = Field(default=6, description="G2 requests per minute")
    capterra_rpm: int = Field(default=8, description="Capterra requests per minute")
    trustradius_rpm: int = Field(default=10, description="TrustRadius requests per minute")
    reddit_rpm: int = Field(default=30, description="Reddit requests per minute")

    # Phase 1 API sources
    hackernews_rpm: int = Field(default=100, description="HN Algolia requests per minute")
    github_rpm: int = Field(default=25, description="GitHub API requests per minute")
    rss_rpm: int = Field(default=10, description="RSS feed requests per minute")
    github_token: str = Field(default="", description="GitHub personal access token for higher rate limits")

    # Phase 2 sources
    gartner_rpm: int = Field(default=4, description="Gartner Peer Insights requests per minute")
    trustpilot_rpm: int = Field(default=6, description="TrustPilot requests per minute")
    getapp_rpm: int = Field(default=8, description="GetApp requests per minute")
    getapp_protection_page_stop_threshold: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Consecutive protection-like GetApp fallback pages before aborting the transport",
    )
    twitter_rpm: int = Field(default=10, description="Twitter/X requests per minute")
    producthunt_rpm: int = Field(default=20, description="ProductHunt requests per minute")
    producthunt_api_token: str = Field(default="", description="ProductHunt API bearer token")
    youtube_api_key: str = Field(default="", description="YouTube Data API v3 key")
    youtube_rpm: int = Field(default=50, description="YouTube API requests per minute")
    quora_rpm: int = Field(default=4, description="Quora requests per minute")
    stackoverflow_rpm: int = Field(default=25, description="Stack Exchange API requests per minute")
    stackoverflow_api_key: str = Field(default="", description="Stack Exchange API key (10k req/day vs 300)")
    peerspot_rpm: int = Field(default=4, description="PeerSpot requests per minute")
    software_advice_rpm: int = Field(default=8, description="Software Advice requests per minute")
    sourceforge_rpm: int = Field(default=12, description="SourceForge requests per minute")
    slashdot_rpm: int = Field(default=8, description="Slashdot Software requests per minute")

    # Behavioral delays
    min_delay_seconds: float = Field(default=2.0, description="Min delay between requests")
    max_delay_seconds: float = Field(default=8.0, description="Max delay between requests")

    # Reddit
    reddit_client_id: str = Field(default="", description="Reddit API OAuth2 client ID")
    reddit_client_secret: str = Field(default="", description="Reddit API OAuth2 client secret")
    reddit_default_subreddits: str = Field(
        default="sysadmin,salesforce,aws,ITManagers,devops,msp",
        description="Default subreddits for Reddit scraping (comma-separated)",
    )

    # Firecrawl (JS-rendered scraping for TrustRadius etc.)
    firecrawl_api_key: str = Field(default="", description="Firecrawl API key for JS-rendered page scraping")

    # Resilience
    max_retries: int = Field(default=2, description="Max retries per request")
    blocked_cooldown_hours: int = Field(default=24, description="Hours to cool down after blocked")
    scrape_log_retention_days: int = Field(default=30, description="Days to retain scrape logs")

    # CAPTCHA solving
    captcha_enabled: bool = Field(default=False, description="Enable CAPTCHA solving for protected sites")
    captcha_provider: str = Field(default="capsolver", description="CAPTCHA solver provider (capsolver or 2captcha)")
    captcha_api_key: str = Field(default="", description="CAPTCHA solver API key")
    captcha_domains: str = Field(default="g2.com,capterra.com,gartner.com,getapp.com", description="Domains with CAPTCHA solving enabled (comma-separated)")
    captcha_proxy_url: str = Field(default="", description="Sticky/static proxy URL for CAPTCHA solving (same IP for solve + retry)")
    captcha_2captcha_api_key: str = Field(default="", description="2Captcha API key (used as fallback or per-domain override)")
    captcha_2captcha_domains: str = Field(
        default="getapp.com,gartner.com",
        description="Domains that should use 2Captcha instead of primary provider (comma-separated)",
    )

    # Relevance filtering (social media noise reduction)
    relevance_filter_enabled: bool = Field(default=True, description="Enable relevance filtering for social media sources")
    relevance_threshold: float = Field(default=0.55, description="Min relevance score (0.0-1.0) for social media posts")
    source_quality_gate_enabled: bool = Field(
        default=True,
        description="Enable source-specific pre-insert quality gates for noisy sources",
    )
    source_quality_gate_sources: str = Field(
        default="quora,twitter,capterra",
        description="Comma-separated sources with pre-insert quality gating",
    )
    source_quality_twitter_require_intent: bool = Field(
        default=True,
        description="Require churn/comparison intent language for Twitter/X rows",
    )
    source_quality_twitter_drop_vendor_self_posts: bool = Field(
        default=True,
        description="Drop Twitter/X rows authored by vendor-owned accounts",
    )
    source_quality_drop_capterra_aggregates: bool = Field(
        default=True,
        description="Drop Capterra JSON-LD aggregate pages that are not real reviews",
    )
    cross_source_dedup_enabled: bool = Field(
        default=True,
        description="Detect and suppress duplicate B2B reviews syndicated across multiple sources",
    )
    cross_source_dedup_similarity_threshold: float = Field(
        default=0.82,
        ge=0.5,
        le=1.0,
        description="Minimum normalized text similarity for reviewer/date cross-source duplicate matches",
    )
    cross_source_dedup_max_candidates: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Max existing vendor review candidates to compare when checking cross-source duplicates",
    )
    cross_source_dedup_loose_similarity_threshold: float = Field(
        default=0.9,
        ge=0.5,
        le=1.0,
        description="Minimum normalized text similarity for reviewer-prefix/date-tolerant duplicate matches",
    )
    cross_source_dedup_review_date_tolerance_days: int = Field(
        default=1,
        ge=0,
        le=7,
        description="Allowed review-date drift when matching syndicated duplicates across sources",
    )
    cross_source_dedup_rating_tolerance: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="Allowed rating delta when matching syndicated duplicates across sources",
    )
    cross_source_dedup_reviewer_stem_length: int = Field(
        default=5,
        ge=2,
        le=12,
        description="Reviewer-name prefix length used for loose cross-source duplicate candidate matching",
    )

    # Exhaustive scrape mode
    exhaustive_lookback_days: int = Field(default=365, description="Date cutoff for exhaustive mode (days)")
    exhaustive_max_pages_default: int = Field(default=100, description="Safety cap for exhaustive pagination")
    exhaustive_inter_vendor_delay: float = Field(default=4.0, description="Seconds between vendors in exhaustive mode")

    # Playwright stealth browser (DataDome/Cloudflare bypass)
    playwright_enabled: bool = Field(default=False, description="Enable Playwright stealth browser for protected sites")
    playwright_headless: bool = Field(default=True, description="Run Chromium in headless mode")
    playwright_timeout_ms: int = Field(default=30000, description="Page navigation timeout (ms)")
    playwright_max_concurrent: int = Field(default=1, description="Max concurrent browser contexts")

    # Bright Data Web Unlocker (DataDome / heavy anti-bot bypass)
    web_unlocker_url: str = Field(
        default="",
        description="Bright Data Web Unlocker proxy URL (bypasses DataDome/Cloudflare automatically)",
    )
    web_unlocker_domains: str = Field(
        default="g2.com,capterra.com,gartner.com,getapp.com,peerspot.com,softwareadvice.com,quora.com",
        description="Domains to route through Web Unlocker (comma-separated)",
    )

    # Bright Data Scraping Browser (cloud Chromium with CAPTCHA solving)
    scraping_browser_ws_url: str = Field(
        default="",
        description="Bright Data Scraping Browser WebSocket URL (wss://...@brd.superproxy.io:9222)",
    )
    scraping_browser_domains: str = Field(
        default="getapp.com,x.com,slashdot.org",
        description="Domains to route through Scraping Browser instead of Web Unlocker (comma-separated)",
    )

    # Bright Data SERP API (for discovering URLs on blocked sites via Google)
    serp_api_token: str = Field(
        default="",
        description="Bright Data SERP API bearer token",
    )
    serp_api_zone: str = Field(
        default="serp_api1",
        description="Bright Data SERP API zone name",
    )
    serp_api_url: str = Field(
        default="https://api.brightdata.com/request",
        description="Bright Data SERP API endpoint URL",
    )


class B2BCampaignConfig(BaseSettings):
    """B2B ABM campaign generation configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_B2B_CAMPAIGN_", env_file=ENV_FILES, extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable B2B campaign engine")
    min_opportunity_score: int = Field(default=70, ge=0, le=100, description="Min opportunity score to target")
    require_decision_maker: bool = Field(default=False, description="Require decision-maker flag on reviews (False = full buying committee)")
    max_campaigns_per_run: int = Field(default=20, ge=1, description="Max campaigns per generation run")
    channels: list[str] = Field(
        default=["email_cold", "linkedin", "email_followup"],
        description="Channels to generate content for",
    )
    schedule_cron: str = Field(default="0 22 * * *", description="Campaign generation schedule (daily 10 PM)")
    dedup_days: int = Field(default=7, ge=1, description="Days before re-targeting same company")
    retention_days: int = Field(default=90, ge=1, description="Days to retain expired/sent campaigns before cleanup")
    concurrency: int = Field(default=8, description="Max concurrent LLM calls during campaign generation")
    max_tokens: int = Field(default=2048, description="Max tokens per LLM generation call")
    llm_timeout_seconds: float = Field(
        default=120.0,
        ge=5.0,
        le=300.0,
        description="Timeout for a single campaign LLM generation call",
    )
    anthropic_batch_enabled: bool = Field(
        default=True,
        description="Allow campaign generation to use Anthropic Message Batches when eligible",
    )
    anthropic_batch_detached_enabled: bool = Field(
        default=False,
        description="Submit campaign batches for later reconciliation instead of waiting inline for completion",
    )
    anthropic_batch_stale_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Minutes before a detached campaign batch job or claim is considered stale",
    )
    anthropic_batch_min_items: int = Field(
        default=2,
        ge=1,
        le=10000,
        description="Minimum uncached campaign items required before using Anthropic batching",
    )
    word_limits: dict[str, dict[str, list[int]]] = Field(
        default_factory=lambda: {
            "default": {
                "email_cold": [50, 150],
                "email_followup": [75, 150],
                "linkedin": [0, 100],
            },
            "vendor_retention": {
                "email_cold": [50, 125],
                "email_followup": [75, 150],
            },
            "challenger_intel": {
                "email_cold": [50, 125],
                "email_followup": [75, 150],
            },
            "churning_company": {
                "email_cold": [75, 150],
                "email_followup": [75, 125],
                "linkedin": [0, 100],
            },
        },
        description="Per-target-mode campaign word limits by channel as [min_words, max_words]",
    )
    temperature: float = Field(default=0.7, description="LLM sampling temperature")
    default_sender_name: str = Field(default="", description="Sender name for outreach")
    default_sender_title: str = Field(default="", description="Sender title for outreach")
    default_sender_company: str = Field(default="", description="Sender company name for outreach")
    default_booking_url: str = Field(default="", description="Default booking/calendar URL for outreach CTAs")
    require_display_safe_company: bool = Field(
        default=True,
        description="Require churning-company outreach targets to be display-safe named companies",
    )
    require_primary_blog_post: bool = Field(
        default=True,
        description="Require a matched blog post before generating churning-company outreach",
    )
    min_pain_categories: int = Field(
        default=1,
        ge=1,
        description="Minimum number of distinct pain categories required for churning-company outreach",
    )
    review_queue_min_score: int = Field(
        default=55,
        ge=0,
        le=100,
        description="Lower bound for the analyst review queue score band",
    )
    review_queue_max_score: int = Field(
        default=69,
        ge=0,
        le=100,
        description="Upper bound for the analyst review queue score band",
    )
    target_mode: str = Field(
        default="vendor_retention",
        description="Campaign target mode: vendor_retention | challenger_intel | churning_company",
    )
    personas: list[str] = Field(
        default=["executive", "technical", "operations", "evaluator", "champion"],
        description="Persona types to generate campaigns for (buying committee coverage)",
    )
    specificity_require_anchor_support: bool = Field(
        default=True,
        description="Require campaign drafts to use witness-backed anchors when briefing-backed concrete anchors are available",
    )
    specificity_min_anchor_hits: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Minimum number of witness-backed specificity signal groups a campaign draft must hit when anchors are available",
    )
    specificity_require_timing_or_numeric_when_available: bool = Field(
        default=True,
        description="Require campaign drafts to mention a timing or numeric anchor when one is available in briefing-backed witnesses",
    )
    specificity_revision_term_limit: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Max exact witness-backed proof terms to surface in campaign prompt retries",
    )
    revalidate_before_manual_approval: bool = Field(
        default=True,
        description="Re-run deterministic witness-backed specificity checks before manual campaign approval",
    )
    revalidate_before_queue_send: bool = Field(
        default=True,
        description="Re-run deterministic witness-backed specificity checks before queueing a campaign for send",
    )
    revalidate_before_send: bool = Field(
        default=True,
        description="Re-run deterministic witness-backed specificity checks immediately before campaign auto-send",
    )


class CampaignSequenceConfig(BaseSettings):
    """B2B campaign email sequence configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_CAMPAIGN_SEQ_", env_file=ENV_FILES, extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable stateful campaign sequences")
    max_steps: int = Field(default=4, ge=2, le=8, description="Max emails per sequence")
    progression_batch_limit: int = Field(
        default=10, ge=1, le=250,
        description="Max due sequences to progress in one task run",
    )
    onboarding_max_steps: int = Field(
        default=4, ge=1, le=8,
        description="Max emails in the onboarding sequence",
    )
    onboarding_sender_name: str = Field(
        default="Atlas Intel",
        description="Sender name used in onboarding sequences",
    )
    onboarding_sender_company: str = Field(
        default="Atlas Intelligence",
        description="Sender company used in onboarding sequences",
    )
    onboarding_product_name: str = Field(
        default="Atlas Intel",
        description="Display product name used in onboarding sequence prompts",
    )
    step_delay_days: list[int] = Field(
        default=[3, 5, 7],
        description="Days between steps 1->2, 2->3, 3->4",
    )
    auto_send_enabled: bool = Field(
        default=False, description="Auto-send queued campaign emails after cancel window"
    )
    auto_send_delay_seconds: int = Field(
        default=300, ge=60, description="Cancel window before auto-send (seconds)"
    )
    check_interval_seconds: int = Field(
        default=3600, ge=600, description="How often to check for due sequence steps"
    )
    prompt_max_tokens: int = Field(
        default=512, ge=128, le=2048,
        description="Max completion tokens for next-step sequence generation",
    )
    prompt_list_limit: int = Field(
        default=5, ge=1, le=20,
        description="Max list items to keep when compacting sequence context for storage and prompts",
    )
    prompt_quote_limit: int = Field(
        default=3, ge=1, le=10,
        description="Max quotes or short signal items to keep in compact sequence context",
    )
    prompt_blog_post_limit: int = Field(
        default=3, ge=1, le=10,
        description="Max blog posts to keep in compact sequence selling context",
    )
    prompt_email_body_preview_chars: int = Field(
        default=220, ge=80, le=1000,
        description="Max plain-text characters to keep per previous-email preview in sequence prompts",
    )
    resend_api_key: str = Field(default="", description="Resend ESP API key")
    resend_from_email: str = Field(default="", description="Resend sender email address")
    resend_webhook_signing_secret: str = Field(
        default="", description="Resend webhook signature verification secret"
    )

    # ESP selection + SES
    sender_type: str = Field(default="resend", description="Campaign ESP: 'resend' or 'ses'")
    ses_region: str = Field(default="us-east-1", description="AWS region for SES")
    ses_access_key_id: str = Field(default="", description="AWS access key (blank = use env/instance role)")
    ses_secret_access_key: str = Field(default="", description="AWS secret key")
    ses_configuration_set: str = Field(default="", description="SES Configuration Set name for tracking")
    ses_from_email: str = Field(default="", description="Verified sender for SES")

    # CAN-SPAM compliance
    unsubscribe_base_url: str = Field(
        default="", description="Base URL for one-click unsubscribe endpoint",
    )
    company_address: str = Field(
        default="", description="Physical mailing address for CAN-SPAM footer",
    )

    # Smart send scheduling
    send_window_start: str = Field(default="09:00", description="Earliest send time (HH:MM)")
    send_window_end: str = Field(default="17:00", description="Latest send time (HH:MM)")
    send_timezone: str = Field(default="America/Chicago", description="Timezone for send window")
    skip_weekends: bool = Field(default=True, description="Don't send on Sat/Sun")
    max_sends_per_contact: int = Field(
        default=8, ge=1,
        description="Max total emails sent to a single recipient across all sequences before fatigue suppression",
    )

    @field_validator("sender_type", mode="after")
    @classmethod
    def _validate_sender_type(cls, v: str) -> str:
        allowed = ("resend", "ses")
        if v not in allowed:
            raise ValueError(f"sender_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("send_window_start", "send_window_end", mode="after")
    @classmethod
    def _validate_hhmm(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{2}:\d{2}$", v):
            raise ValueError(f"Must be HH:MM format, got '{v}'")
        h, m = int(v[:2]), int(v[3:])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError(f"Invalid time '{v}': hour 0-23, minute 0-59")
        return v

    @field_validator("send_timezone", mode="after")
    @classmethod
    def _validate_tz(cls, v: str) -> str:
        from zoneinfo import ZoneInfo
        ZoneInfo(v)
        return v


class AmazonSellerCampaignConfig(BaseSettings):
    """Amazon Seller Intelligence campaign outreach configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_SELLER_CAMPAIGN_", env_file=ENV_FILES, extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable Amazon seller campaign engine")
    max_campaigns_per_run: int = Field(default=10, ge=1, description="Max categories to generate campaigns for per run")
    channels: list[str] = Field(
        default=["email_cold", "email_followup"],
        description="Channels to generate content for",
    )
    schedule_cron: str = Field(default="0 21 * * *", description="Campaign generation schedule (daily 9 PM)")
    dedup_days: int = Field(default=14, ge=1, description="Days before re-targeting same seller+category")
    min_reviews_per_category: int = Field(default=50, ge=10, description="Min reviews in category before generating campaigns")
    max_tokens: int = Field(default=2048, description="Max tokens per LLM generation call")
    temperature: float = Field(default=0.7, description="LLM sampling temperature")
    default_sender_name: str = Field(default="", description="Sender name for outreach")
    default_sender_title: str = Field(default="", description="Sender title for outreach")
    product_name: str = Field(default="Atlas Seller Intelligence", description="Product name used in outreach")
    landing_url: str = Field(default="", description="Dashboard landing page URL")
    free_report_url: str = Field(default="", description="Free category report URL template")


class SubcategoryIntelligenceConfig(BaseSettings):
    """Subcategory-level intelligence report configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_SUBCATEGORY_INTEL_", env_file=ENV_FILES, extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable subcategory intelligence reports")
    schedule_cron: str = Field(default="30 22 * * *", description="Report generation schedule (daily 10:30 PM)")
    min_products: int = Field(default=5, ge=1, description="Min products to qualify a subcategory")
    min_reviews: int = Field(default=100, ge=10, description="Min enriched reviews to qualify")
    max_subcategories_per_run: int = Field(default=10, ge=1, description="Max subcategories per run")
    max_tokens: int = Field(default=4096, description="Max tokens per LLM generation call")
    temperature: float = Field(default=0.4, description="LLM sampling temperature")
    target_subcategories: list[str] = Field(default=[], description="Explicit subcategories (empty = auto-discover)")
    dedup_days: int = Field(default=1, ge=1, description="Days before regenerating same subcategory")


class ComparisonNormalizationConfig(BaseSettings):
    """Competitive flow brand normalization configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_COMPARISON_NORMALIZATION_", env_file=ENV_FILES, extra="ignore",
    )

    known_brand_max_words: int = Field(default=4, ge=1, le=8, description="Max words allowed for trusted canonical brands")
    known_brand_max_length: int = Field(default=40, ge=4, le=120, description="Max character length allowed for trusted canonical brands")
    suspicious_singleton_max_products: int = Field(default=1, ge=0, le=10, description="Apply suspicious singleton filtering to canonical brands with product counts at or below this threshold")
    invalid_known_brands: str = Field(default="", description="Comma-separated exact brand values to reject from canonical and comparison normalization")
    suspicious_singleton_terms: str = Field(default="", description="Comma-separated product-type terms that indicate a singleton canonical brand is likely dirty")
    suspicious_singleton_chars: str = Field(default=",!?:;", description="Characters that make singleton canonical brands suspicious")


class ApolloConfig(BaseSettings):
    """Apollo.io prospect enrichment configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_APOLLO_", env_file=ENV_FILES, extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable Apollo.io prospect pipeline")
    api_key: str = Field(default="", description="Apollo.io API key")
    max_prospects_per_company: int = Field(default=3, ge=1, le=25, description="Max people to reveal per company (1 credit each)")
    target_seniorities: list[str] = Field(
        default=["c_suite", "owner", "founder", "vp", "head", "director"],
        description="Apollo seniority levels to target (decision makers only)",
    )
    min_urgency_score: float = Field(default=3.0, ge=0, le=10, description="Min review urgency score to count as a churn signal")
    min_churn_signals: int = Field(default=5, ge=1, description="Min churn signals a vendor must have to trigger enrichment")
    org_cache_days: int = Field(default=30, ge=1, description="Days before re-enriching a cached org")
    max_credits_per_run: int = Field(default=5000, ge=1, description="Max Apollo credits per enrichment run")
    rate_limit_per_minute: int = Field(default=50, ge=1, description="API calls per minute")
    accepted_email_statuses: list[str] = Field(
        default=["verified", "probabilistic"],
        description="Email statuses considered usable",
    )
    manual_review_block_sources: list[str] = Field(
        default_factory=list,
        description="Discovery sources blocked by prospect_org_cache manual_review rows",
    )
    company_enrichment_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Company-specific Apollo alias/domain overrides keyed by raw company name",
    )
    enrichment_cron: str = Field(default="0 20 * * *", description="Prospect enrichment schedule (daily 8 PM)")
    matching_interval_seconds: int = Field(default=3600, ge=300, description="Prospect-to-sequence matching interval")
    max_vendor_credits_per_run: int = Field(default=2000, ge=1, description="Max Apollo credits per vendor target enrichment run")
    vendor_enrichment_cron: str = Field(default="30 19 * * *", description="Vendor target enrichment schedule (daily 7:30 PM)")

    @field_validator("manual_review_block_sources", mode="after")
    @classmethod
    def _validate_manual_review_block_sources(cls, v: list[str]) -> list[str]:
        allowed = {"reviewer_company", "vendor_name", "campaign_sequence", "vendor_target"}
        invalid = [source for source in v if source not in allowed]
        if invalid:
            raise ValueError(f"manual_review_block_sources contains invalid sources: {invalid}")
        return v


class TemporalPatternConfig(BaseSettings):
    """Temporal pattern context configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_TEMPORAL_", env_file=ENV_FILES, extra="ignore")

    enabled: bool = Field(default=True, description="Enable temporal pattern context in LLM prompts")
    min_samples: int = Field(default=5, ge=1, le=100, description="Minimum sample count per pattern row")
    failure_cooldown: float = Field(default=60.0, ge=0.0, le=600.0, description="Seconds to suppress DB retries after failure")



class MCPConfig(BaseSettings):
    """MCP server configuration.

    Both the CRM and Email MCP servers default to stdio transport (Claude
    Desktop / Cursor compatible).  Set transport='sse' to expose them as HTTP
    endpoints instead.
    """

    model_config = SettingsConfigDict(env_prefix="ATLAS_MCP_", env_file=ENV_FILES, extra="ignore")

    client_enabled: bool = Field(default=True, description="Enable Atlas as MCP client")
    crm_enabled: bool = Field(default=True, description="Enable CRM MCP server")
    email_enabled: bool = Field(default=True, description="Enable Email MCP server")
    calendar_enabled: bool = Field(default=True, description="Enable Calendar MCP server")
    twilio_enabled: bool = Field(default=True, description="Enable Twilio MCP server")
    invoicing_enabled: bool = Field(default=True, description="Enable Invoicing MCP server")
    intelligence_enabled: bool = Field(default=True, description="Enable Intelligence MCP server")
    b2b_churn_enabled: bool = Field(default=True, description="Enable B2B Churn Intelligence MCP server")
    auth_token: str = Field(default="", description="Bearer token for SSE transport auth (empty = no auth)")
    transport: str = Field(default="stdio", description="MCP transport: stdio or sse")
    host: str = Field(default="0.0.0.0", description="Bind host for SSE transport")
    crm_port: int = Field(default=8056, description="Port for CRM MCP server (SSE transport)")
    email_port: int = Field(default=8057, description="Port for Email MCP server (SSE transport)")
    twilio_port: int = Field(default=8058, description="Port for Twilio MCP server (SSE transport)")
    calendar_port: int = Field(default=8059, description="Port for Calendar MCP server (SSE transport)")
    invoicing_port: int = Field(default=8060, description="Port for Invoicing MCP server (SSE transport)")
    intelligence_port: int = Field(default=8061, description="Port for Intelligence MCP server (SSE transport)")
    b2b_churn_port: int = Field(default=8062, description="Port for B2B Churn Intelligence MCP server (SSE transport)")
    scraper_enabled: bool = Field(default=True, description="Enable Universal Scraper MCP server")
    scraper_port: int = Field(default=8063, description="Port for Universal Scraper MCP server (SSE transport)")
    memory_enabled: bool = Field(default=True, description="Enable Memory MCP server")
    memory_port: int = Field(default=8064, description="Port for Memory MCP server (SSE transport)")


class AlertMonitorConfig(BaseSettings):
    """Proactive weather and traffic alert monitoring configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_ALERT_MONITOR__", env_file=ENV_FILES, extra="ignore")

    enabled: bool = Field(default=False, description="Enable proactive weather/traffic alerts")
    check_interval_seconds: int = Field(default=600, description="Poll interval (10 min default)")

    # Location (defaults from weather tool config at runtime)
    home_lat: float | None = Field(default=None, description="Home latitude (falls back to weather_default_lat)")
    home_lon: float | None = Field(default=None, description="Home longitude (falls back to weather_default_lon)")
    work_lat: float | None = Field(default=None, description="Work/office latitude for commute monitoring")
    work_lon: float | None = Field(default=None, description="Work/office longitude for commute monitoring")

    # Weather alert filtering
    nws_severities: str = Field(
        default="Extreme,Severe",
        description="Comma-separated NWS severity levels to alert on (Extreme, Severe, Moderate, Minor)",
    )

    # Traffic thresholds
    traffic_radius_miles: float = Field(default=15, description="Radius around home for traffic incidents")
    commute_delay_threshold_minutes: int = Field(default=10, description="Commute delay (minutes) before alerting")
    traffic_min_severity: int = Field(default=2, description="Min TomTom incident magnitude (0-4) for area alerts")

    # TTS voice announcements for urgent alerts
    tts_on_urgent: bool = Field(default=True, description="TTS broadcast for tornado/flash flood/severe storm alerts")


class NewsIntelligenceConfig(BaseSettings):
    """News intelligence configuration -- entity-level pressure signal detection.

    Monitors a watchlist of entities (public companies, sports teams, markets,
    crypto, or custom topics) via the NewsAPI and identifies which ones are
    building *pre-movement pressure* before the story breaks as mainstream news.

    The core insight: article velocity, sentiment shifts, and source diversity
    all tend to accelerate *before* a meaningful price, odds, or sentiment
    movement is publicly reported.
    """

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_NEWS_", env_file=ENV_FILES, extra="ignore"
    )

    enabled: bool = Field(default=False, description="Enable daily news intelligence analysis")
    api_key: str | None = Field(default=None, description="NewsAPI.org API key (newsapi.org/register)")

    # Watchlist -- primary entity configuration
    watchlist: str = Field(
        default=(
            '[{"name":"Apple Inc","type":"company","query":"Apple AAPL supply chain earnings","ticker":"AAPL"},'
            '{"name":"Bitcoin","type":"crypto","query":"Bitcoin BTC price regulation","ticker":"BTC"},'
            '{"name":"S&P 500","type":"market","query":"S&P 500 SPX market outlook","ticker":"SPX"}]'
        ),
        description=(
            "JSON array of watched entities. Each entry: "
            '{"name": "...", "type": "company|sports_team|market|crypto|custom", '
            '"query": "NewsAPI search query", "ticker": "optional ticker/symbol"}'
        ),
    )

    # Legacy simple-mode fallback
    topics: str = Field(
        default="",
        description=(
            "Comma-separated plain-text topics (simple mode -- used only when watchlist is empty). "
            "Prefer the watchlist for entity-specific tracking."
        ),
    )
    regions: str = Field(
        default="US",
        description="Comma-separated regions prepended to simple-mode topic queries",
    )
    languages: str = Field(
        default="en",
        description="Comma-separated language codes for article filtering (e.g. en,es)",
    )

    # Pressure signal detection
    lookback_days: int = Field(
        default=7, ge=2, le=30,
        description="Days of history used to establish the baseline article volume for each entity",
    )
    pressure_velocity_threshold: float = Field(
        default=1.5, ge=1.0, le=10.0,
        description=(
            "Minimum volume growth multiplier to flag a pressure signal -- "
            "1.5 means 50% more articles than the recent daily average"
        ),
    )
    signal_min_articles: int = Field(
        default=3, ge=1, le=20,
        description="Minimum articles in the most-recent day to confirm a signal (filters single-source noise)",
    )

    # Multi-dimensional scoring
    sentiment_enabled: bool = Field(
        default=True,
        description=(
            "Score sentiment shift per entity -- a sudden increase in negative (or positive) "
            "tone often precedes a meaningful movement"
        ),
    )
    source_diversity_enabled: bool = Field(
        default=True,
        description=(
            "Score source diversity -- when a story spreads from niche outlets to mainstream "
            "the diversity score rises, strengthening the pressure signal"
        ),
    )
    composite_score_threshold: float = Field(
        default=1.3, ge=1.0, le=10.0,
        description=(
            "Minimum composite pressure score to flag a signal. "
            "Composite = velocity * sentiment_factor * diversity_factor * linguistic_factor. "
            "Lower than velocity_threshold because extra dimensions add confirmation."
        ),
    )

    # Linguistic pre-indicator analysis (behavioral stacking)
    linguistic_analysis_enabled: bool = Field(
        default=True,
        description=(
            "Enable linguistic pre-indicator pattern analysis. "
            "Detects language patterns that statistically appear before major movements -- "
            "hedging, deflection, insider sourcing, and escalation language."
        ),
    )
    linguistic_hedge_enabled: bool = Field(
        default=True,
        description=(
            "Detect hedging/uncertainty language ('reportedly', 'could', 'may', 'sources say') -- "
            "builds before unconfirmed information goes mainstream"
        ),
    )
    linguistic_deflection_enabled: bool = Field(
        default=True,
        description=(
            "Detect deflection/denial language ('denies', 'dismisses', 'refuses to comment') -- "
            "denial clusters often appear immediately before a story breaks"
        ),
    )
    linguistic_insider_enabled: bool = Field(
        default=True,
        description=(
            "Detect insider/source language ('people familiar with the matter', 'anonymous sources') -- "
            "indicates information leakage before official disclosure"
        ),
    )
    linguistic_escalation_enabled: bool = Field(
        default=True,
        description=(
            "Detect escalation/urgency language ('breaking', 'crisis', 'urgent', 'imminent') -- "
            "urgency words in trade press before mainstream indicates accelerating pressure"
        ),
    )
    linguistic_permission_enabled: bool = Field(
        default=True,
        description=(
            "Detect moral permission language ('must be stopped', 'for the greater good', "
            "'no option but') -- grants readers permission to act against prior values, "
            "often appears before coordinated pressure campaigns"
        ),
    )
    linguistic_certainty_enabled: bool = Field(
        default=True,
        description=(
            "Detect certainty/moral panic language ('undeniable', 'settled', 'always', 'never') -- "
            "absolute language combined with emotional triggers precedes coordinated narratives"
        ),
    )
    linguistic_dissociation_enabled: bool = Field(
        default=True,
        description=(
            "Detect we/us -> they/them language shifts and label-based framing ('these people', "
            "'their kind', 'outsiders') -- group identity dissociation builds before major events"
        ),
    )

    # SORAM Framework (Chase Hughes) -- 5 societal pressure channels
    soram_enabled: bool = Field(
        default=True,
        description=(
            "Enable SORAM framework analysis. Scores five societal pressure channels "
            "(Societal / Operational / Regulatory / Alignment / Media Novelty) that "
            "Hughes identifies as the levers pulled simultaneously before major events."
        ),
    )
    soram_societal_enabled: bool = Field(
        default=True,
        description=(
            "SORAM Societal: detect coordinated threat/fear framing across outlets -- "
            "a sudden obsession with a specific 'threat' or 'misinformation' topic across "
            "unrelated platforms signals a coordinated pressure campaign"
        ),
    )
    soram_operational_enabled: bool = Field(
        default=True,
        description=(
            "SORAM Operational: detect drills, simulations, and readiness exercises in coverage -- "
            "an increase in 'exercise' and 'preparedness' language often precedes actual events"
        ),
    )
    soram_regulatory_enabled: bool = Field(
        default=True,
        description=(
            "SORAM Regulatory: detect new emergency powers, executive orders, or rule changes -- "
            "quietly introduced regulations that are only useful if a certain crisis occurs"
        ),
    )
    soram_alignment_enabled: bool = Field(
        default=True,
        description=(
            "SORAM Alignment: detect scripted consensus -- when government, media, and tech "
            "begin using the exact same phrasing simultaneously (coordinated messaging)"
        ),
    )
    soram_media_novelty_enabled: bool = Field(
        default=True,
        description=(
            "SORAM Media Novelty: detect novelty hijacking -- a constant stream of 'breaking' "
            "and unrelated urgent news keeps the brain in high suggestibility, often preceding "
            "a major coordinated narrative push"
        ),
    )

    # Alternative data sources
    sec_edgar_enabled: bool = Field(
        default=False,
        description=(
            "Fetch recent SEC 8-K filings for company/crypto entities via EDGAR free API. "
            "Elevated 8-K activity indicates undisclosed material events. Requires no API key."
        ),
    )
    usaspending_enabled: bool = Field(
        default=False,
        description=(
            "Fetch recent USAspending.gov contract awards for watched entities. "
            "Sudden government contracts indicate business momentum or regulatory attention. "
            "No API key required."
        ),
    )
    state_sos_enabled: bool = Field(
        default=False,
        description=(
            "Monitor State Secretary of State filings for new business formations near watched entities. "
            "Requires custom regional integration -- see docs for setup."
        ),
    )
    county_recorder_enabled: bool = Field(
        default=False,
        description=(
            "Monitor county recorder / building permit data for commercial development signals. "
            "Requires custom regional integration -- see docs for setup."
        ),
    )
    bls_enabled: bool = Field(
        default=False,
        description=(
            "Fetch BLS/Census employment and industry trend data for watched sectors. "
            "Useful for macro context on company and market entities."
        ),
    )

    # Signal streak and cross-entity correlation
    signal_streak_enabled: bool = Field(
        default=True,
        description=(
            "Track consecutive days with elevated signals per entity. "
            "A streak of N days is significantly more predictive than a single-day spike."
        ),
    )
    signal_streak_threshold: int = Field(
        default=3, ge=2, le=14,
        description=(
            "Number of consecutive elevated-signal days that triggers a 'building pressure' alert. "
            "Streaks indicate sustained pre-movement accumulation, not noise."
        ),
    )
    cross_entity_correlation_enabled: bool = Field(
        default=True,
        description=(
            "Detect macro signals when multiple entities of the same type spike simultaneously. "
            "Correlated spikes across companies/markets indicate sector-wide events."
        ),
    )
    cross_entity_min_signals: int = Field(
        default=3, ge=2, le=10,
        description=(
            "Minimum number of same-type entities that must signal simultaneously "
            "to flag a cross-entity macro correlation."
        ),
    )

    # Operations
    max_articles_per_topic: int = Field(
        default=20, ge=5, le=100,
        description="Maximum articles fetched per entity per run (controls NewsAPI quota usage)",
    )
    llm_model: str = Field(
        default="qwen3:14b",
        description="Ollama model used to synthesise the intelligence briefing",
    )
    schedule_hour: int = Field(
        default=5, ge=0, le=23,
        description="Hour of day (0-23, local time) to run the daily intelligence analysis",
    )

    # Output
    notify_on_signal: bool = Field(
        default=True,
        description="Send a push notification when new pressure signals are detected",
    )
    notify_all_runs: bool = Field(
        default=False,
        description="Send a push notification after every run, even when no new signals are found",
    )
    include_in_morning_briefing: bool = Field(
        default=True,
        description="Include active pressure signals in the morning briefing summary",
    )


class ProviderCostConfig(BaseSettings):
    """Provider billing sync configuration for cost reconciliation."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_PROVIDER_COST_", env_file=ENV_FILES, extra="ignore"
    )

    enabled: bool = Field(
        default=False,
        description="Enable periodic sync of provider billing totals for reconciliation",
    )
    interval_seconds: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="How often to sync provider billing totals",
    )
    sync_timeout_seconds: int = Field(
        default=20,
        ge=5,
        le=300,
        description="HTTP timeout for provider billing sync requests",
    )
    snapshot_retention_days: int = Field(
        default=90,
        ge=7,
        le=730,
        description="Days to retain cumulative provider usage snapshots",
    )
    daily_retention_days: int = Field(
        default=365,
        ge=30,
        le=1825,
        description="Days to retain imported provider daily cost rows",
    )
    openrouter_enabled: bool = Field(
        default=True,
        description="Sync OpenRouter cumulative usage snapshots when a management key is available",
    )
    openrouter_api_key: str = Field(
        default="",
        description="Optional OpenRouter management key override for credits snapshots",
    )
    anthropic_enabled: bool = Field(
        default=False,
        description="Sync Anthropic admin daily cost reports when an admin key is available",
    )
    anthropic_admin_api_key: str = Field(
        default="",
        description="Anthropic Admin API key for usage and cost reporting",
    )
    anthropic_lookback_days: int = Field(
        default=7,
        ge=1,
        le=31,
        description="Lookback window for Anthropic daily cost report sync",
    )


class Settings(BaseSettings):
    """Application-wide settings."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_",
        env_file=ENV_FILES,
        env_nested_delimiter="__",
        extra="ignore",
    )

    # General
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: Path = Field(default=Path("logs"), description="Log directory")
    models_dir: Path = Field(default=Path("models"), description="Models cache directory")

    # Startup behavior
    load_stt_on_startup: bool = Field(default=True, description="Load STT on startup")
    load_tts_on_startup: bool = Field(default=True, description="Load TTS on startup")
    load_llm_on_startup: bool = Field(default=True, description="Load LLM on startup")

    # Startup behavior - speaker ID
    load_speaker_id_on_startup: bool = Field(
        default=False, description="Load speaker ID on startup"
    )

    # Startup behavior - Omni (unified speech-to-speech)
    load_omni_on_startup: bool = Field(
        default=False, description="Load Omni (unified voice) on startup"
    )

    # Nested configs
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    omni: OmniConfig = Field(default_factory=OmniConfig)
    speaker_id: SpeakerIDConfig = Field(default_factory=SpeakerIDConfig)
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)
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
    free_mode: FreeModeConfig = Field(default_factory=FreeModeConfig)
    entity_context: EntityContextConfig = Field(default_factory=EntityContextConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    home_agent: HomeAgentConfig = Field(default_factory=HomeAgentConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    workflows: WorkflowConfig = Field(default_factory=WorkflowConfig)
    orchestrated: OrchestratedConfig = Field(default_factory=OrchestratedConfig)
    edge: EdgeConfig = Field(default_factory=EdgeConfig)
    autonomous: AutonomousConfig = Field(default_factory=AutonomousConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    email_draft: EmailDraftConfig = Field(default_factory=EmailDraftConfig)
    email_intake: EmailIntakeConfig = Field(default_factory=EmailIntakeConfig)
    email_stale_check: EmailStaleCheckConfig = Field(default_factory=EmailStaleCheckConfig)
    temporal: TemporalPatternConfig = Field(default_factory=TemporalPatternConfig)
    call_intelligence: CallIntelligenceConfig = Field(default_factory=CallIntelligenceConfig)
    sms_intelligence: SMSIntelligenceConfig = Field(default_factory=SMSIntelligenceConfig)
    invoicing: InvoicingConfig = Field(default_factory=InvoicingConfig)
    external_data: ExternalDataConfig = Field(default_factory=ExternalDataConfig)
    b2b_churn: B2BChurnConfig = Field(default_factory=B2BChurnConfig)
    b2b_alert: B2BAlertConfig = Field(default_factory=B2BAlertConfig)
    b2b_watchlist_delivery: B2BWatchlistDeliveryConfig = Field(default_factory=B2BWatchlistDeliveryConfig)
    b2b_report_delivery: B2BReportDeliveryConfig = Field(default_factory=B2BReportDeliveryConfig)
    b2b_webhook: B2BWebhookConfig = Field(default_factory=B2BWebhookConfig)
    crm_event: CRMEventConfig = Field(default_factory=CRMEventConfig)
    b2b_scrape: B2BScrapeConfig = Field(default_factory=B2BScrapeConfig)
    universal_scrape: UniversalScrapeConfig = Field(default_factory=UniversalScrapeConfig)
    b2b_campaign: B2BCampaignConfig = Field(default_factory=B2BCampaignConfig)
    campaign_sequence: CampaignSequenceConfig = Field(default_factory=CampaignSequenceConfig)
    seller_campaign: AmazonSellerCampaignConfig = Field(default_factory=AmazonSellerCampaignConfig)
    subcategory_intelligence: SubcategoryIntelligenceConfig = Field(default_factory=SubcategoryIntelligenceConfig)
    comparison_normalization: ComparisonNormalizationConfig = Field(default_factory=ComparisonNormalizationConfig)
    apollo: ApolloConfig = Field(default_factory=ApolloConfig)
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)
    ftl_tracing: FTLTracingConfig = Field(default_factory=FTLTracingConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    alert_monitor: AlertMonitorConfig = Field(default_factory=AlertMonitorConfig)
    news_intel: NewsIntelligenceConfig = Field(default_factory=NewsIntelligenceConfig)
    provider_cost: ProviderCostConfig = Field(default_factory=ProviderCostConfig)
    saas_auth: SaaSAuthConfig = Field(default_factory=SaaSAuthConfig)

    # Reasoning agent (cross-domain event-driven intelligence)
    @staticmethod
    def _reasoning_factory():
        from .reasoning.config import ReasoningConfig
        return ReasoningConfig()
    reasoning: Any = Field(default_factory=lambda: Settings._reasoning_factory())

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
