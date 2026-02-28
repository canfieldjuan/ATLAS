"""
Centralized configuration management using Pydantic Settings.

Configuration is loaded from environment variables with sensible defaults.
"""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    ollama_model: str = Field(default="qwen3:14b", description="Ollama model name")
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama API URL (override: ATLAS_LLM__OLLAMA_URL)")
    ollama_timeout: int = Field(default=120, description="Ollama HTTP timeout in seconds (increase for cloud relay models)")

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

    # Anthropic settings (email draft generation)
    anthropic_model: str = Field(
        default="claude-sonnet-4-5-20250929",
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
    ntfy_url: str = Field(default="http://localhost:8090", description="ntfy server URL (override: ATLAS_ALERTS__NTFY_URL)")
    ntfy_topic: str = Field(default="atlas-alerts", description="ntfy topic for alerts")


class EmailConfig(BaseSettings):
    """Email tool configuration (Resend API + Gmail + IMAP)."""

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
        env_prefix="ATLAS_EMAIL_DRAFT_", env_file=".env", extra="ignore",
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
        default="claude-haiku-4-5-20251001",
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
        env_prefix="ATLAS_EMAIL_INTAKE_", env_file=".env", extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable 10-min email polling")
    interval_seconds: int = Field(default=600, ge=300, description="Poll interval in seconds")
    crm_enabled: bool = Field(default=True, description="Cross-reference emails against CRM")
    action_plan_enabled: bool = Field(
        default=True, description="Generate LLM action plans for CRM matches"
    )
    max_action_plans_per_cycle: int = Field(
        default=5, ge=1, le=20, description="Cap LLM calls per polling cycle"
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
        env_prefix="ATLAS_EMAIL_STALE_CHECK_", env_file=".env", extra="ignore",
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


class EntityContextConfig(BaseSettings):
    """Configuration for voice entity context between turns."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_ENTITY_CONTEXT__",
        env_file=".env",
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
        env_file=".env",
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
    owner_name: str = Field(default="Juan", description="Owner/user name for email sign-offs and personalization")


class HomeAgentConfig(BaseSettings):
    """Home agent LLM generation parameters."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_HOME_AGENT_")

    max_tokens: int = Field(default=150, ge=50, le=1024, description="Max tokens for LLM response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature for conversation")
    tool_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature for tool calling")
    max_history: int = Field(default=4, ge=0, le=20, description="Max conversation history turns to include")


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
        default=200, description="Max conversation turns to sync per nightly run",
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


class CallIntelligenceConfig(BaseSettings):
    """Post-call transcription and data extraction."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_CALL_INTELLIGENCE_", env_file=".env", extra="ignore"
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
        env_prefix="ATLAS_SMS_INTELLIGENCE_", env_file=".env", extra="ignore"
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
        env_prefix="ATLAS_INVOICING_", env_file=".env", extra="ignore"
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


class ExternalDataConfig(BaseSettings):
    """External data producers: news feeds and financial markets."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_EXTERNAL_DATA_", env_file=".env", extra="ignore"
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
    intelligence_max_tokens: int = Field(default=16384, description="Max tokens for intelligence LLM call")
    intelligence_journal_retention_days: int = Field(default=90, description="Days to retain reasoning journal entries")
    intelligence_news_retention_days: int = Field(default=30, description="Days to retain news articles")
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
    # Pressure scoring
    pressure_enabled: bool = Field(default=True, description="Enable pressure signal detection")
    pressure_baseline_window_days: int = Field(default=180, description="Rolling window for baseline (6 months)")
    pressure_alert_threshold: float = Field(default=7.0, description="Pressure score alert threshold (0-10)")
    pressure_drift_alert_threshold: float = Field(default=2.0, description="Sentiment drift alert threshold")
    # Complaint mining
    complaint_mining_enabled: bool = Field(default=False, description="Enable complaint mining pipeline")
    complaint_enrichment_interval_seconds: int = Field(default=300, description="Complaint enrichment polling interval (5 min)")
    complaint_enrichment_max_per_batch: int = Field(default=20, description="Max reviews to enrich per batch")
    complaint_enrichment_max_attempts: int = Field(default=3, description="Max enrichment attempts before marking failed")
    complaint_enrichment_local_only: bool = Field(default=False, description="Force local LLM for enrichment (skip Claude)")
    complaint_enrichment_reviews_per_call: int = Field(default=1, description="Reviews per LLM call (batch mode when >1, max 10)")
    complaint_analysis_enabled: bool = Field(default=True, description="Enable daily complaint analysis task")
    complaint_analysis_cron: str = Field(default="0 21 * * *", description="Cron for complaint analysis (default 9 PM)")
    complaint_analysis_window_days: int = Field(default=7, description="Days of enriched reviews to include in analysis")
    complaint_analysis_max_tokens: int = Field(default=16384, description="Max tokens for analysis LLM call")
    complaint_retention_days: int = Field(default=365, description="Days to retain complaint reports")
    # Content generation (Claude-powered)
    complaint_content_enabled: bool = Field(default=True, description="Enable complaint content generation")
    complaint_content_cron: str = Field(default="0 22 * * *", description="Cron for content generation (default 10 PM)")
    complaint_content_max_per_run: int = Field(default=5, description="Max content pieces to generate per run")
    complaint_content_max_tokens: int = Field(default=4096, description="Max tokens per content generation call")
    # Deep enrichment (second-pass extraction)
    deep_enrichment_interval_seconds: int = Field(default=600, description="Deep enrichment polling interval (10 min)")
    deep_enrichment_max_per_batch: int = Field(default=5, description="Max reviews to deep-enrich per autonomous batch")
    deep_enrichment_max_attempts: int = Field(default=3, description="Max attempts before marking deep_failed")
    deep_enrichment_max_tokens: int = Field(default=1024, description="Max LLM output tokens for deep extraction")
    # Competitive intelligence (cross-brand analysis from deep_extraction)
    competitive_intelligence_enabled: bool = Field(default=True, description="Enable competitive intelligence analysis")
    competitive_intelligence_cron: str = Field(default="30 21 * * *", description="Cron for competitive intelligence (default 9:30 PM)")
    competitive_intelligence_max_tokens: int = Field(default=16384, description="Max tokens for competitive intelligence LLM call")
    competitive_intelligence_min_deep_enriched: int = Field(default=100, description="Min deep-enriched reviews required to run")


class B2BChurnConfig(BaseSettings):
    """B2B software churn prediction pipeline configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_B2B_CHURN_", env_file=".env", extra="ignore"
    )

    enabled: bool = Field(default=False, description="Enable B2B churn prediction pipeline")

    # Enrichment
    enrichment_interval_seconds: int = Field(default=300, description="Enrichment polling interval")
    enrichment_max_per_batch: int = Field(default=10, description="Max reviews to enrich per batch")
    enrichment_max_attempts: int = Field(default=3, description="Max enrichment attempts")
    enrichment_max_tokens: int = Field(default=1024, description="Max LLM output tokens")
    enrichment_local_only: bool = Field(default=False, description="Force local LLM only")

    # Intelligence aggregation
    intelligence_enabled: bool = Field(default=True, description="Enable churn intelligence aggregation")
    intelligence_cron: str = Field(default="0 21 * * 0", description="Weekly churn intelligence (Sunday 9 PM)")
    intelligence_max_tokens: int = Field(default=16384, description="Max tokens for intelligence LLM call")
    intelligence_window_days: int = Field(default=30, description="Days of enriched reviews to analyze")
    intelligence_min_reviews: int = Field(default=3, description="Min reviews per vendor to include")

    # Churn thresholds
    high_churn_urgency_threshold: int = Field(default=7, description="Urgency score >= this = high churn risk")
    enterprise_only: bool = Field(default=False, description="Only include enterprise-segment reviews")


class B2BScrapeConfig(BaseSettings):
    """B2B review scraping pipeline configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_B2B_SCRAPE_", env_file=".env", extra="ignore"
    )

    enabled: bool = Field(default=False, description="Enable B2B review scraping pipeline")

    # Schedule
    intake_interval_seconds: int = Field(default=3600, description="Scrape polling interval (1 hour)")
    max_targets_per_run: int = Field(default=5, description="Max targets to scrape per run")

    # Proxies (comma-separated URLs)
    proxy_datacenter_urls: str = Field(default="", description="Datacenter proxy URLs (comma-separated)")
    proxy_residential_urls: str = Field(default="", description="Residential proxy URLs (comma-separated)")
    proxy_residential_geo: str = Field(default="US", description="Residential proxy geo code")

    # Per-domain rate limits (RPM)
    g2_rpm: int = Field(default=6, description="G2 requests per minute")
    capterra_rpm: int = Field(default=8, description="Capterra requests per minute")
    trustradius_rpm: int = Field(default=10, description="TrustRadius requests per minute")
    reddit_rpm: int = Field(default=30, description="Reddit requests per minute")

    # Behavioral delays
    min_delay_seconds: float = Field(default=2.0, description="Min delay between requests")
    max_delay_seconds: float = Field(default=8.0, description="Max delay between requests")

    # Reddit
    reddit_default_subreddits: str = Field(
        default="sysadmin,salesforce,aws,ITManagers,devops,msp",
        description="Default subreddits for Reddit scraping (comma-separated)",
    )

    # Resilience
    max_retries: int = Field(default=2, description="Max retries per request")
    blocked_cooldown_hours: int = Field(default=24, description="Hours to cool down after blocked")
    scrape_log_retention_days: int = Field(default=30, description="Days to retain scrape logs")

    # CAPTCHA solving
    captcha_enabled: bool = Field(default=False, description="Enable CAPTCHA solving for protected sites")
    captcha_provider: str = Field(default="capsolver", description="CAPTCHA solver provider (capsolver or 2captcha)")
    captcha_api_key: str = Field(default="", description="CAPTCHA solver API key")
    captcha_domains: str = Field(default="g2.com,capterra.com", description="Domains with CAPTCHA solving enabled (comma-separated)")
    captcha_proxy_url: str = Field(default="", description="Sticky/static proxy URL for CAPTCHA solving (same IP for solve + retry)")
    captcha_2captcha_api_key: str = Field(default="", description="2Captcha API key (used as fallback or per-domain override)")
    captcha_2captcha_domains: str = Field(default="", description="Domains that should use 2Captcha instead of primary provider (comma-separated)")


class TemporalPatternConfig(BaseSettings):
    """Temporal pattern context configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_TEMPORAL_", env_file=".env", extra="ignore")

    enabled: bool = Field(default=True, description="Enable temporal pattern context in LLM prompts")
    min_samples: int = Field(default=5, ge=1, le=100, description="Minimum sample count per pattern row")
    failure_cooldown: float = Field(default=60.0, ge=0.0, le=600.0, description="Seconds to suppress DB retries after failure")



class MCPConfig(BaseSettings):
    """MCP server configuration.

    Both the CRM and Email MCP servers default to stdio transport (Claude
    Desktop / Cursor compatible).  Set transport='sse' to expose them as HTTP
    endpoints instead.
    """

    model_config = SettingsConfigDict(env_prefix="ATLAS_MCP_", env_file=".env", extra="ignore")

    client_enabled: bool = Field(default=True, description="Enable Atlas as MCP client")
    crm_enabled: bool = Field(default=True, description="Enable CRM MCP server")
    email_enabled: bool = Field(default=True, description="Enable Email MCP server")
    calendar_enabled: bool = Field(default=True, description="Enable Calendar MCP server")
    twilio_enabled: bool = Field(default=True, description="Enable Twilio MCP server")
    invoicing_enabled: bool = Field(default=True, description="Enable Invoicing MCP server")
    auth_token: str = Field(default="", description="Bearer token for SSE transport auth (empty = no auth)")
    transport: str = Field(default="stdio", description="MCP transport: stdio or sse")
    host: str = Field(default="0.0.0.0", description="Bind host for SSE transport")
    crm_port: int = Field(default=8056, description="Port for CRM MCP server (SSE transport)")
    email_port: int = Field(default=8057, description="Port for Email MCP server (SSE transport)")
    twilio_port: int = Field(default=8058, description="Port for Twilio MCP server (SSE transport)")
    calendar_port: int = Field(default=8059, description="Port for Calendar MCP server (SSE transport)")
    invoicing_port: int = Field(default=8060, description="Port for Invoicing MCP server (SSE transport)")


class AlertMonitorConfig(BaseSettings):
    """Proactive weather and traffic alert monitoring configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_ALERT_MONITOR__", env_file=".env", extra="ignore")

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
    b2b_scrape: B2BScrapeConfig = Field(default_factory=B2BScrapeConfig)
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)
    ftl_tracing: FTLTracingConfig = Field(default_factory=FTLTracingConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    alert_monitor: AlertMonitorConfig = Field(default_factory=AlertMonitorConfig)

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
