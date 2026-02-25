"""
Voice pipeline settings API.

GET  /settings/voice  — read current user-configurable voice settings
PATCH /settings/voice — update settings (applies in-memory + persists to .env.local)
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..config import settings

logger = logging.getLogger("atlas.api.settings")

router = APIRouter(prefix="/settings", tags=["Settings"])

# .env.local lives at the project root (same directory that main.py loads it from)
_ENV_LOCAL_PATH = Path(__file__).parent.parent.parent / ".env.local"

# Serialise all .env.local writes to prevent concurrent corruption
_env_write_lock = asyncio.Lock()

# Mapping from Python field name → ATLAS_VOICE_* environment variable key
_VOICE_ENV_MAP: dict[str, str] = {
    "enabled": "ATLAS_VOICE_ENABLED",
    "input_device": "ATLAS_VOICE_INPUT_DEVICE",
    "audio_gain": "ATLAS_VOICE_AUDIO_GAIN",
    "use_arecord": "ATLAS_VOICE_USE_ARECORD",
    "arecord_device": "ATLAS_VOICE_ARECORD_DEVICE",
    "wake_threshold": "ATLAS_VOICE_WAKE_THRESHOLD",
    "wake_confirmation_enabled": "ATLAS_VOICE_WAKE_CONFIRMATION_ENABLED",
    "asr_url": "ATLAS_VOICE_ASR_URL",
    "asr_timeout": "ATLAS_VOICE_ASR_TIMEOUT",
    "asr_streaming_enabled": "ATLAS_VOICE_ASR_STREAMING_ENABLED",
    "asr_ws_url": "ATLAS_VOICE_ASR_WS_URL",
    "piper_length_scale": "ATLAS_VOICE_PIPER_LENGTH_SCALE",
    "vad_aggressiveness": "ATLAS_VOICE_VAD_AGGRESSIVENESS",
    "silence_ms": "ATLAS_VOICE_SILENCE_MS",
    "conversation_mode_enabled": "ATLAS_VOICE_CONVERSATION_MODE_ENABLED",
    "conversation_timeout_ms": "ATLAS_VOICE_CONVERSATION_TIMEOUT_MS",
    "filler_enabled": "ATLAS_VOICE_FILLER_ENABLED",
    "filler_delay_ms": "ATLAS_VOICE_FILLER_DELAY_MS",
    "agent_timeout": "ATLAS_VOICE_AGENT_TIMEOUT",
    "streaming_llm_enabled": "ATLAS_VOICE_STREAMING_LLM_ENABLED",
    "debug_logging": "ATLAS_VOICE_DEBUG_LOGGING",
}


class VoiceSettings(BaseModel):
    """User-configurable voice pipeline settings returned by the API."""

    # Pipeline
    enabled: bool = Field(description="Enable the voice pipeline on startup")
    streaming_llm_enabled: bool = Field(description="Stream LLM tokens to TTS as they generate")
    debug_logging: bool = Field(description="Verbose debug logging for the voice pipeline")

    # Microphone / audio capture
    input_device: Optional[str] = Field(description="PortAudio device name or index (null = system default)")
    audio_gain: float = Field(description="Software microphone gain multiplier (1.0 = unity)")
    use_arecord: bool = Field(description="Use ALSA arecord instead of PortAudio")
    arecord_device: str = Field(description="ALSA device string when use_arecord is true")

    # Wake word
    wake_threshold: float = Field(description="OpenWakeWord detection threshold (0.0–1.0)")
    wake_confirmation_enabled: bool = Field(description="Play a tone when the wake word is detected")

    # ASR (speech-to-text)
    asr_url: Optional[str] = Field(description="Nemotron ASR HTTP endpoint URL")
    asr_timeout: int = Field(description="ASR HTTP request timeout (seconds)")
    asr_streaming_enabled: bool = Field(description="Use WebSocket streaming ASR instead of HTTP batch mode")
    asr_ws_url: Optional[str] = Field(description="Nemotron ASR WebSocket URL")

    # TTS
    piper_length_scale: float = Field(description="Piper speech rate — lower = faster (0.5–2.0)")

    # VAD & segmentation
    vad_aggressiveness: int = Field(description="WebRTC VAD aggressiveness 0–3 (higher = more aggressive)")
    silence_ms: int = Field(description="Silence duration that finalises an utterance (ms)")

    # Conversation mode
    conversation_mode_enabled: bool = Field(description="Stay in conversation mode after each response (no wake word)")
    conversation_timeout_ms: int = Field(description="Milliseconds to wait for follow-up before exiting conversation mode")

    # Filler phrases
    filler_enabled: bool = Field(description="Speak a filler phrase when the agent is slow to respond")
    filler_delay_ms: int = Field(description="Milliseconds before the first filler phrase is spoken")

    # Timeouts
    agent_timeout: float = Field(description="Max seconds to wait for LLM + tool execution")


class VoiceSettingsUpdate(BaseModel):
    """Partial update payload for voice pipeline settings (all fields optional)."""

    enabled: Optional[bool] = None
    streaming_llm_enabled: Optional[bool] = None
    debug_logging: Optional[bool] = None

    input_device: Optional[str] = None
    audio_gain: Optional[float] = None
    use_arecord: Optional[bool] = None
    arecord_device: Optional[str] = None

    wake_threshold: Optional[float] = None
    wake_confirmation_enabled: Optional[bool] = None

    asr_url: Optional[str] = None
    asr_timeout: Optional[int] = None
    asr_streaming_enabled: Optional[bool] = None
    asr_ws_url: Optional[str] = None

    piper_length_scale: Optional[float] = None

    vad_aggressiveness: Optional[int] = None
    silence_ms: Optional[int] = None

    conversation_mode_enabled: Optional[bool] = None
    conversation_timeout_ms: Optional[int] = None

    filler_enabled: Optional[bool] = None
    filler_delay_ms: Optional[int] = None

    agent_timeout: Optional[float] = None


def _quote_env_value(value: str) -> str:
    """Quote a .env value that contains characters python-dotenv would misparse.

    python-dotenv treats unquoted values as ending at any `` # `` (space-hash)
    and strips leading/trailing whitespace.  Embedded newlines would also split
    the entry across multiple lines, corrupting the file.

    When quoting is needed the value is wrapped in double quotes and the
    standard escape sequences (``\\n``, ``\\r``, ``\\"`` , ``\\\\``) are
    applied.  python-dotenv restores these escape sequences back to their
    original characters when reading a double-quoted value, so the round-trip
    is lossless.
    """
    if "\n" not in value and "\r" not in value and " #" not in value:
        return value
    escaped = (
        value
        .replace("\\", "\\\\")  # escape backslashes first
        .replace('"', '\\"')    # escape embedded double quotes
        .replace("\n", "\\n")   # escape newlines
        .replace("\r", "\\r")   # escape carriage returns
    )
    return f'"{escaped}"'


def _write_env_local(updates: dict[str, str]) -> None:
    """Upsert key=value pairs in .env.local, preserving unrelated lines.

    Values that contain ``=`` are handled correctly because we only split on
    the *first* ``=`` when extracting existing keys.  Values with embedded
    newlines or comment markers are automatically quoted (see
    :func:`_quote_env_value`).
    """
    env_path = _ENV_LOCAL_PATH

    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text().splitlines()

    # Index existing entries by key so we can replace them in-place
    key_to_line_idx: dict[str, int] = {}
    for idx, line in enumerate(existing_lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            key_to_line_idx[key] = idx

    for key, value in updates.items():
        new_line = f"{key}={_quote_env_value(value)}"
        if key in key_to_line_idx:
            existing_lines[key_to_line_idx[key]] = new_line
        else:
            existing_lines.append(new_line)

    env_path.write_text("\n".join(existing_lines) + "\n")


def _current_voice_settings() -> VoiceSettings:
    """Build a VoiceSettings snapshot from the live settings object."""
    v = settings.voice
    return VoiceSettings(
        enabled=v.enabled,
        streaming_llm_enabled=v.streaming_llm_enabled,
        debug_logging=v.debug_logging,
        input_device=v.input_device,
        audio_gain=v.audio_gain,
        use_arecord=v.use_arecord,
        arecord_device=v.arecord_device,
        wake_threshold=v.wake_threshold,
        wake_confirmation_enabled=v.wake_confirmation_enabled,
        asr_url=v.asr_url,
        asr_timeout=v.asr_timeout,
        asr_streaming_enabled=v.asr_streaming_enabled,
        asr_ws_url=v.asr_ws_url,
        piper_length_scale=v.piper_length_scale,
        vad_aggressiveness=v.vad_aggressiveness,
        silence_ms=v.silence_ms,
        conversation_mode_enabled=v.conversation_mode_enabled,
        conversation_timeout_ms=v.conversation_timeout_ms,
        filler_enabled=v.filler_enabled,
        filler_delay_ms=v.filler_delay_ms,
        agent_timeout=v.agent_timeout,
    )


@router.get("/voice", response_model=VoiceSettings)
async def get_voice_settings() -> VoiceSettings:
    """Return the current user-configurable voice pipeline settings."""
    return _current_voice_settings()


@router.patch("/voice", response_model=VoiceSettings)
async def update_voice_settings(updates: VoiceSettingsUpdate) -> VoiceSettings:
    """
    Update voice pipeline settings.

    Changes are applied to the in-memory settings object immediately
    and persisted to .env.local so they survive a server restart.
    Note: settings that control hardware initialisation (e.g. input_device,
    use_arecord) only take full effect after a pipeline restart.
    """
    update_dict = updates.model_dump(exclude_none=True)

    if not update_dict:
        return _current_voice_settings()

    # Apply in-memory (takes effect immediately for runtime-checked settings)
    for field, value in update_dict.items():
        if hasattr(settings.voice, field):
            setattr(settings.voice, field, value)
            logger.info("Voice setting updated in-memory: %s = %r", field, value)

    # Persist to .env.local (serialised to avoid concurrent file corruption)
    env_updates: dict[str, str] = {}
    for field, value in update_dict.items():
        env_key = _VOICE_ENV_MAP.get(field)
        if env_key is not None:
            env_updates[env_key] = str(value)

    if env_updates:
        async with _env_write_lock:
            try:
                _write_env_local(env_updates)
                logger.info("Voice settings persisted to .env.local: %s", list(env_updates.keys()))
            except OSError as exc:
                logger.warning("Failed to persist voice settings to .env.local: %s", exc)

    return _current_voice_settings()


# ---------------------------------------------------------------------------
# Email settings
# ---------------------------------------------------------------------------

# Flat field-name → environment variable for all three email-related configs.
# Fields are prefixed (intake_*, draft_*) to namespace them within one model.
_EMAIL_ENV_MAP: dict[str, str] = {
    # EmailConfig  (ATLAS_EMAIL_*)
    "enabled": "ATLAS_EMAIL_ENABLED",
    "default_from": "ATLAS_EMAIL_DEFAULT_FROM",
    "gmail_send_enabled": "ATLAS_EMAIL_GMAIL_SEND_ENABLED",
    "timeout": "ATLAS_EMAIL_TIMEOUT",
    "imap_host": "ATLAS_EMAIL_IMAP_HOST",
    "imap_port": "ATLAS_EMAIL_IMAP_PORT",
    "imap_username": "ATLAS_EMAIL_IMAP_USERNAME",
    "imap_ssl": "ATLAS_EMAIL_IMAP_SSL",
    "imap_mailbox": "ATLAS_EMAIL_IMAP_MAILBOX",
    # ToolsConfig  (ATLAS_TOOLS_*)
    "gmail_query": "ATLAS_TOOLS_GMAIL_QUERY",
    "gmail_max_results": "ATLAS_TOOLS_GMAIL_MAX_RESULTS",
    # EmailIntakeConfig  (ATLAS_EMAIL_INTAKE_*)
    "intake_enabled": "ATLAS_EMAIL_INTAKE_ENABLED",
    "intake_interval_seconds": "ATLAS_EMAIL_INTAKE_INTERVAL_SECONDS",
    "intake_crm_enabled": "ATLAS_EMAIL_INTAKE_CRM_ENABLED",
    "intake_action_plan_enabled": "ATLAS_EMAIL_INTAKE_ACTION_PLAN_ENABLED",
    "intake_max_action_plans_per_cycle": "ATLAS_EMAIL_INTAKE_MAX_ACTION_PLANS_PER_CYCLE",
    # EmailDraftConfig  (ATLAS_EMAIL_DRAFT_*)
    # Note: EmailDraftConfig has env_prefix="ATLAS_EMAIL_DRAFT_" and the field is
    # named "draft_expiry_hours", so pydantic-settings resolves the env var as
    # ATLAS_EMAIL_DRAFT_ + DRAFT_EXPIRY_HOURS = ATLAS_EMAIL_DRAFT_DRAFT_EXPIRY_HOURS.
    # The double-DRAFT is correct and intentional.
    "draft_enabled": "ATLAS_EMAIL_DRAFT_ENABLED",
    "draft_auto_draft_enabled": "ATLAS_EMAIL_DRAFT_AUTO_DRAFT_ENABLED",
    "draft_model_name": "ATLAS_EMAIL_DRAFT_MODEL_NAME",
    "draft_temperature": "ATLAS_EMAIL_DRAFT_TEMPERATURE",
    "draft_expiry_hours": "ATLAS_EMAIL_DRAFT_DRAFT_EXPIRY_HOURS",
    "draft_notify_drafts": "ATLAS_EMAIL_DRAFT_NOTIFY_DRAFTS",
    "draft_schedule_interval_seconds": "ATLAS_EMAIL_DRAFT_SCHEDULE_INTERVAL_SECONDS",
    "draft_triage_enabled": "ATLAS_EMAIL_DRAFT_TRIAGE_ENABLED",
}


class EmailSettings(BaseModel):
    """User-configurable email settings returned by the API.

    Combines EmailConfig, ToolsConfig (gmail_*), EmailIntakeConfig, and
    EmailDraftConfig into a single flat model.  Credentials (api_key,
    imap_password, OAuth tokens) are intentionally omitted — manage those
    via environment variables or the .env file.
    """

    # General
    enabled: bool = Field(description="Enable the email system (sending + reading)")
    default_from: Optional[str] = Field(description="Default sender address (e.g. you@gmail.com)")
    gmail_send_enabled: bool = Field(description="Prefer Gmail API for sending; falls back to Resend when false")
    timeout: int = Field(description="HTTP request timeout for email API calls (seconds)")

    # IMAP inbox reading
    imap_host: str = Field(description="IMAP server host (e.g. imap.gmail.com, outlook.office365.com)")
    imap_port: int = Field(description="IMAP port — 993 (SSL) or 143 (STARTTLS)")
    imap_username: str = Field(description="IMAP login username (usually your email address)")
    imap_ssl: bool = Field(description="Use SSL/TLS for the IMAP connection")
    imap_mailbox: str = Field(description="Default IMAP mailbox / folder to read from")
    gmail_query: str = Field(description="Default inbox search query (IMAP/Gmail syntax)")
    gmail_max_results: int = Field(description="Maximum emails to fetch per inbox read")

    # Autonomous intake polling
    intake_enabled: bool = Field(description="Enable background email polling")
    intake_interval_seconds: int = Field(description="How often to poll for new emails (minimum 300 s)")
    intake_crm_enabled: bool = Field(description="Cross-reference incoming emails against CRM contacts")
    intake_action_plan_enabled: bool = Field(description="Generate AI action plans for CRM-matched emails")
    intake_max_action_plans_per_cycle: int = Field(description="Max LLM calls per polling cycle (cost control)")

    # AI draft generation
    draft_enabled: bool = Field(description="Enable AI-powered reply draft generation")
    draft_auto_draft_enabled: bool = Field(
        description=(
            "Automatically generate drafts on a schedule — when off, drafts are only "
            "generated on demand via the ntfy Draft Reply button"
        )
    )
    draft_model_name: str = Field(description="LLM model used to write reply drafts")
    draft_temperature: float = Field(description="Draft creativity — lower = more conservative, higher = more varied")
    draft_expiry_hours: int = Field(description="Hours a pending draft stays available before it expires")
    draft_notify_drafts: bool = Field(description="Send a push notification when a new draft is ready for review")
    draft_schedule_interval_seconds: int = Field(
        description="How often the draft generation task runs when auto-draft is enabled (seconds)"
    )
    draft_triage_enabled: bool = Field(
        description="Use a fast LLM to classify ambiguous emails as replyable/non-replyable before drafting"
    )


class EmailSettingsUpdate(BaseModel):
    """Partial update payload for email settings (all fields optional)."""

    # General
    enabled: Optional[bool] = None
    default_from: Optional[str] = None
    gmail_send_enabled: Optional[bool] = None
    timeout: Optional[int] = None

    # IMAP
    imap_host: Optional[str] = None
    imap_port: Optional[int] = None
    imap_username: Optional[str] = None
    imap_ssl: Optional[bool] = None
    imap_mailbox: Optional[str] = None
    gmail_query: Optional[str] = None
    gmail_max_results: Optional[int] = None

    # Intake
    intake_enabled: Optional[bool] = None
    intake_interval_seconds: Optional[int] = None
    intake_crm_enabled: Optional[bool] = None
    intake_action_plan_enabled: Optional[bool] = None
    intake_max_action_plans_per_cycle: Optional[int] = None

    # Draft
    draft_enabled: Optional[bool] = None
    draft_auto_draft_enabled: Optional[bool] = None
    draft_model_name: Optional[str] = None
    draft_temperature: Optional[float] = None
    draft_expiry_hours: Optional[int] = None
    draft_notify_drafts: Optional[bool] = None
    draft_schedule_interval_seconds: Optional[int] = None
    draft_triage_enabled: Optional[bool] = None


def _current_email_settings() -> EmailSettings:
    """Build an EmailSettings snapshot from the live settings objects."""
    e = settings.email
    t = settings.tools
    ei = settings.email_intake
    ed = settings.email_draft
    return EmailSettings(
        enabled=e.enabled,
        default_from=e.default_from,
        gmail_send_enabled=e.gmail_send_enabled,
        timeout=e.timeout,
        imap_host=e.imap_host,
        imap_port=e.imap_port,
        imap_username=e.imap_username,
        imap_ssl=e.imap_ssl,
        imap_mailbox=e.imap_mailbox,
        gmail_query=t.gmail_query,
        gmail_max_results=t.gmail_max_results,
        intake_enabled=ei.enabled,
        intake_interval_seconds=ei.interval_seconds,
        intake_crm_enabled=ei.crm_enabled,
        intake_action_plan_enabled=ei.action_plan_enabled,
        intake_max_action_plans_per_cycle=ei.max_action_plans_per_cycle,
        draft_enabled=ed.enabled,
        draft_auto_draft_enabled=ed.auto_draft_enabled,
        draft_model_name=ed.model_name,
        draft_temperature=ed.temperature,
        draft_expiry_hours=ed.draft_expiry_hours,
        draft_notify_drafts=ed.notify_drafts,
        draft_schedule_interval_seconds=ed.schedule_interval_seconds,
        draft_triage_enabled=ed.triage_enabled,
    )


# Fields prefixed with "intake_" live in settings.email_intake (strip prefix)
_INTAKE_PREFIX = "intake_"
# Fields prefixed with "draft_" live in settings.email_draft (strip prefix)
_DRAFT_PREFIX = "draft_"
# "gmail_*" fields live in settings.tools
_TOOLS_FIELDS = {"gmail_query", "gmail_max_results"}


@router.get("/email", response_model=EmailSettings)
async def get_email_settings() -> EmailSettings:
    """Return current user-configurable email settings."""
    return _current_email_settings()


@router.patch("/email", response_model=EmailSettings)
async def update_email_settings(updates: EmailSettingsUpdate) -> EmailSettings:
    """
    Update email settings.

    Changes are applied to the in-memory settings objects immediately and
    persisted to .env.local.  Credentials (IMAP password, API keys, OAuth
    tokens) are not managed here — set them in your .env file directly.
    """
    update_dict = updates.model_dump(exclude_none=True)

    if not update_dict:
        return _current_email_settings()

    for field, value in update_dict.items():
        if field.startswith(_INTAKE_PREFIX):
            real_field = field[len(_INTAKE_PREFIX):]
            if hasattr(settings.email_intake, real_field):
                setattr(settings.email_intake, real_field, value)
                logger.info("Email intake setting updated: %s = %r", real_field, value)
        elif field.startswith(_DRAFT_PREFIX):
            real_field = field[len(_DRAFT_PREFIX):]
            if hasattr(settings.email_draft, real_field):
                setattr(settings.email_draft, real_field, value)
                logger.info("Email draft setting updated: %s = %r", real_field, value)
        elif field in _TOOLS_FIELDS:
            if hasattr(settings.tools, field):
                setattr(settings.tools, field, value)
                logger.info("Tools setting updated: %s = %r", field, value)
        else:
            if hasattr(settings.email, field):
                setattr(settings.email, field, value)
                logger.info("Email setting updated: %s = %r", field, value)

    # Persist to .env.local
    env_updates: dict[str, str] = {}
    for field, value in update_dict.items():
        env_key = _EMAIL_ENV_MAP.get(field)
        if env_key is not None:
            env_updates[env_key] = str(value)

    if env_updates:
        async with _env_write_lock:
            try:
                _write_env_local(env_updates)
                logger.info("Email settings persisted to .env.local: %s", list(env_updates.keys()))
            except OSError as exc:
                logger.warning("Failed to persist email settings to .env.local: %s", exc)

    return _current_email_settings()


# ---------------------------------------------------------------------------
# Daily intelligence / nightly reasoning settings
# ---------------------------------------------------------------------------

_DAILY_ENV_MAP: dict[str, str] = {
    # PersonaConfig (ATLAS_PERSONA_*)
    "persona_name": "ATLAS_PERSONA_NAME",
    "persona_owner_name": "ATLAS_PERSONA_OWNER_NAME",
    "persona_system_prompt": "ATLAS_PERSONA_SYSTEM_PROMPT",
    # HomeAgentConfig (ATLAS_HOME_AGENT_*)
    "llm_temperature": "ATLAS_HOME_AGENT_TEMPERATURE",
    "llm_max_tokens": "ATLAS_HOME_AGENT_MAX_TOKENS",
    "llm_max_history": "ATLAS_HOME_AGENT_MAX_HISTORY",
    # AutonomousConfig (ATLAS_AUTONOMOUS_*)
    "autonomous_enabled": "ATLAS_AUTONOMOUS_ENABLED",
    "autonomous_timezone": "ATLAS_AUTONOMOUS_DEFAULT_TIMEZONE",
    "notify_results": "ATLAS_AUTONOMOUS_NOTIFY_RESULTS",
    "announce_results": "ATLAS_AUTONOMOUS_ANNOUNCE_RESULTS",
    "synthesis_enabled": "ATLAS_AUTONOMOUS_SYNTHESIS_ENABLED",
    "synthesis_temperature": "ATLAS_AUTONOMOUS_SYNTHESIS_TEMPERATURE",
    # Morning briefing (ATLAS_AUTONOMOUS_*)
    "briefing_calendar_hours": "ATLAS_AUTONOMOUS_MORNING_BRIEFING_CALENDAR_HOURS",
    "briefing_security_hours": "ATLAS_AUTONOMOUS_MORNING_BRIEFING_SECURITY_HOURS",
    # Nightly memory sync (ATLAS_AUTONOMOUS_* + ATLAS_MEMORY_*)
    "nightly_sync_enabled": "ATLAS_MEMORY_NIGHTLY_SYNC_ENABLED",
    "nightly_sync_max_turns": "ATLAS_AUTONOMOUS_NIGHTLY_SYNC_MAX_TURNS",
    "memory_purge_days": "ATLAS_MEMORY_PURGE_DAYS",
    # Pattern & preference learning (ATLAS_AUTONOMOUS_*)
    "pattern_learning_lookback_days": "ATLAS_AUTONOMOUS_PATTERN_LEARNING_LOOKBACK_DAYS",
    "preference_learning_lookback_days": "ATLAS_AUTONOMOUS_PREFERENCE_LEARNING_LOOKBACK_DAYS",
    "preference_learning_min_turns": "ATLAS_AUTONOMOUS_PREFERENCE_LEARNING_MIN_TURNS",
    # Proactive intelligence (ATLAS_AUTONOMOUS_*)
    "proactive_lookback_hours": "ATLAS_AUTONOMOUS_PROACTIVE_ACTIONS_LOOKBACK_HOURS",
    "action_escalation_stale_days": "ATLAS_AUTONOMOUS_ACTION_ESCALATION_STALE_DAYS",
    "action_escalation_overdue_days": "ATLAS_AUTONOMOUS_ACTION_ESCALATION_OVERDUE_DAYS",
    # Device & security (ATLAS_AUTONOMOUS_*)
    "device_health_battery_threshold": "ATLAS_AUTONOMOUS_DEVICE_HEALTH_BATTERY_THRESHOLD",
    "device_health_stale_hours": "ATLAS_AUTONOMOUS_DEVICE_HEALTH_STALE_HOURS",
    "security_summary_hours": "ATLAS_AUTONOMOUS_SECURITY_SUMMARY_HOURS",
}

# Sub-config routing for write path
_DAILY_PERSONA_FIELDS = {"persona_name", "persona_owner_name", "persona_system_prompt"}
_DAILY_HOME_AGENT_FIELDS = {"llm_temperature", "llm_max_tokens", "llm_max_history"}
_DAILY_MEMORY_FIELDS = {"nightly_sync_enabled", "memory_purge_days"}
# Everything else → settings.autonomous


class DailySettings(BaseModel):
    """User-configurable settings for Atlas's daily reasoning and nightly intelligence cycle."""

    # ── Atlas Identity ────────────────────────────────────────────────────────
    persona_name: str = Field(description="Atlas's name (used in greetings and responses)")
    persona_owner_name: str = Field(description="Your name — used in emails, sign-offs, and personalization")
    persona_system_prompt: str = Field(description="Core personality and instructions sent to the LLM for every conversation")

    # ── LLM Response Behaviour ────────────────────────────────────────────────
    llm_temperature: float = Field(description="Conversational temperature — higher = more creative, lower = more precise")
    llm_max_tokens: int = Field(description="Maximum tokens Atlas generates per conversational response")
    llm_max_history: int = Field(description="Conversation turns to include as context (higher = better memory, more tokens)")

    # ── Autonomous Scheduler ──────────────────────────────────────────────────
    autonomous_enabled: bool = Field(description="Enable the daily/nightly autonomous task scheduler")
    autonomous_timezone: str = Field(description="Timezone used for scheduling all nightly and daily tasks")
    notify_results: bool = Field(description="Push a notification (via ntfy) when a nightly task produces a result")
    announce_results: bool = Field(description="Speak nightly task results aloud via TTS on edge nodes")
    synthesis_enabled: bool = Field(description="Use LLM to synthesise nightly task results into plain-language summaries")
    synthesis_temperature: float = Field(description="LLM temperature for nightly synthesis (lower = more consistent summaries)")

    # ── Daily Briefing (7:00 AM) ──────────────────────────────────────────────
    briefing_calendar_hours: int = Field(description="Hours ahead to include calendar events in the morning briefing")
    briefing_security_hours: int = Field(description="Hours to look back for overnight security events in the morning briefing")

    # ── Nightly Memory Sync (3:00 AM) ─────────────────────────────────────────
    nightly_sync_enabled: bool = Field(description="Sync today's conversations into the knowledge graph each night")
    nightly_sync_max_turns: int = Field(description="Max conversation turns to process per nightly sync run (spread across nights if > cap)")
    memory_purge_days: int = Field(description="Delete raw conversation turns from the database after this many days (knowledge graph keeps the facts)")

    # ── Pattern & Preference Learning ─────────────────────────────────────────
    pattern_learning_lookback_days: int = Field(description="Days of history used to learn your temporal patterns (presence, device usage)")
    preference_learning_lookback_days: int = Field(description="Days of conversation history to analyse for user preferences")
    preference_learning_min_turns: int = Field(description="Minimum user messages needed before Atlas updates preferences (prevents premature changes)")

    # ── Proactive Intelligence ────────────────────────────────────────────────
    proactive_lookback_hours: int = Field(description="Hours of recent conversation Atlas scans to extract pending action items")
    action_escalation_stale_days: int = Field(description="Flag a pending action as stale after this many days of inactivity")
    action_escalation_overdue_days: int = Field(description="Flag a pending action as overdue after this many days past its deadline")

    # ── Device & Security ────────────────────────────────────────────────────
    device_health_battery_threshold: int = Field(description="Battery percentage below which a device is flagged as low in the daily health check")
    device_health_stale_hours: int = Field(description="Hours without a state update before a device is considered stale")
    security_summary_hours: int = Field(description="Lookback window for security event summaries (runs every 6 hours)")


class DailySettingsUpdate(BaseModel):
    """Partial update payload for daily intelligence settings."""

    # Persona
    persona_name: Optional[str] = None
    persona_owner_name: Optional[str] = None
    persona_system_prompt: Optional[str] = None

    # LLM
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None
    llm_max_history: Optional[int] = None

    # Autonomous
    autonomous_enabled: Optional[bool] = None
    autonomous_timezone: Optional[str] = None
    notify_results: Optional[bool] = None
    announce_results: Optional[bool] = None
    synthesis_enabled: Optional[bool] = None
    synthesis_temperature: Optional[float] = None

    # Briefing
    briefing_calendar_hours: Optional[int] = None
    briefing_security_hours: Optional[int] = None

    # Nightly sync
    nightly_sync_enabled: Optional[bool] = None
    nightly_sync_max_turns: Optional[int] = None
    memory_purge_days: Optional[int] = None

    # Learning
    pattern_learning_lookback_days: Optional[int] = None
    preference_learning_lookback_days: Optional[int] = None
    preference_learning_min_turns: Optional[int] = None

    # Proactive
    proactive_lookback_hours: Optional[int] = None
    action_escalation_stale_days: Optional[int] = None
    action_escalation_overdue_days: Optional[int] = None

    # Device & security
    device_health_battery_threshold: Optional[int] = None
    device_health_stale_hours: Optional[int] = None
    security_summary_hours: Optional[int] = None


def _current_daily_settings() -> DailySettings:
    """Build a DailySettings snapshot from the live settings objects."""
    p = settings.persona
    ha = settings.home_agent
    a = settings.autonomous
    m = settings.memory
    return DailySettings(
        persona_name=p.name,
        persona_owner_name=p.owner_name,
        persona_system_prompt=p.system_prompt,
        llm_temperature=ha.temperature,
        llm_max_tokens=ha.max_tokens,
        llm_max_history=ha.max_history,
        autonomous_enabled=a.enabled,
        autonomous_timezone=a.default_timezone,
        notify_results=a.notify_results,
        announce_results=a.announce_results,
        synthesis_enabled=a.synthesis_enabled,
        synthesis_temperature=a.synthesis_temperature,
        briefing_calendar_hours=a.morning_briefing_calendar_hours,
        briefing_security_hours=a.morning_briefing_security_hours,
        nightly_sync_enabled=m.nightly_sync_enabled,
        nightly_sync_max_turns=a.nightly_sync_max_turns,
        memory_purge_days=m.purge_days,
        pattern_learning_lookback_days=a.pattern_learning_lookback_days,
        preference_learning_lookback_days=a.preference_learning_lookback_days,
        preference_learning_min_turns=a.preference_learning_min_turns,
        proactive_lookback_hours=a.proactive_actions_lookback_hours,
        action_escalation_stale_days=a.action_escalation_stale_days,
        action_escalation_overdue_days=a.action_escalation_overdue_days,
        device_health_battery_threshold=a.device_health_battery_threshold,
        device_health_stale_hours=a.device_health_stale_hours,
        security_summary_hours=a.security_summary_hours,
    )


# Persona field name → actual PersonaConfig attribute
_PERSONA_ATTR_MAP = {
    "persona_name": "name",
    "persona_owner_name": "owner_name",
    "persona_system_prompt": "system_prompt",
}
# HomeAgentConfig attribute map
_HOME_AGENT_ATTR_MAP = {
    "llm_temperature": "temperature",
    "llm_max_tokens": "max_tokens",
    "llm_max_history": "max_history",
}
# MemoryConfig attribute map
_MEMORY_ATTR_MAP = {
    "nightly_sync_enabled": "nightly_sync_enabled",
    "memory_purge_days": "purge_days",
}
# AutonomousConfig attribute map (strip prefix from API field name)
_AUTONOMOUS_ATTR_MAP = {
    "autonomous_enabled": "enabled",
    "autonomous_timezone": "default_timezone",
    "notify_results": "notify_results",
    "announce_results": "announce_results",
    "synthesis_enabled": "synthesis_enabled",
    "synthesis_temperature": "synthesis_temperature",
    "briefing_calendar_hours": "morning_briefing_calendar_hours",
    "briefing_security_hours": "morning_briefing_security_hours",
    "nightly_sync_max_turns": "nightly_sync_max_turns",
    "pattern_learning_lookback_days": "pattern_learning_lookback_days",
    "preference_learning_lookback_days": "preference_learning_lookback_days",
    "preference_learning_min_turns": "preference_learning_min_turns",
    "proactive_lookback_hours": "proactive_actions_lookback_hours",
    "action_escalation_stale_days": "action_escalation_stale_days",
    "action_escalation_overdue_days": "action_escalation_overdue_days",
    "device_health_battery_threshold": "device_health_battery_threshold",
    "device_health_stale_hours": "device_health_stale_hours",
    "security_summary_hours": "security_summary_hours",
}


@router.get("/daily", response_model=DailySettings)
async def get_daily_settings() -> DailySettings:
    """Return current user-configurable daily intelligence / nightly reasoning settings."""
    return _current_daily_settings()


@router.patch("/daily", response_model=DailySettings)
async def update_daily_settings(updates: DailySettingsUpdate) -> DailySettings:
    """
    Update daily intelligence settings.

    Changes apply in-memory immediately and are persisted to .env.local.
    Persona changes (name, system_prompt) take effect on the next LLM call.
    Schedule timing changes take effect on the next scheduler cycle.
    """
    update_dict = updates.model_dump(exclude_none=True)

    if not update_dict:
        return _current_daily_settings()

    for field, value in update_dict.items():
        if field in _PERSONA_ATTR_MAP:
            attr = _PERSONA_ATTR_MAP[field]
            if hasattr(settings.persona, attr):
                setattr(settings.persona, attr, value)
                logger.info("Persona setting updated: %s = %r", attr, value)
        elif field in _HOME_AGENT_ATTR_MAP:
            attr = _HOME_AGENT_ATTR_MAP[field]
            if hasattr(settings.home_agent, attr):
                setattr(settings.home_agent, attr, value)
                logger.info("Home agent setting updated: %s = %r", attr, value)
        elif field in _MEMORY_ATTR_MAP:
            attr = _MEMORY_ATTR_MAP[field]
            if hasattr(settings.memory, attr):
                setattr(settings.memory, attr, value)
                logger.info("Memory setting updated: %s = %r", attr, value)
        elif field in _AUTONOMOUS_ATTR_MAP:
            attr = _AUTONOMOUS_ATTR_MAP[field]
            if hasattr(settings.autonomous, attr):
                setattr(settings.autonomous, attr, value)
                logger.info("Autonomous setting updated: %s = %r", attr, value)

    # Persist to .env.local
    env_updates: dict[str, str] = {}
    for field, value in update_dict.items():
        env_key = _DAILY_ENV_MAP.get(field)
        if env_key is not None:
            env_updates[env_key] = str(value)

    if env_updates:
        async with _env_write_lock:
            try:
                _write_env_local(env_updates)
                logger.info("Daily settings persisted to .env.local: %s", list(env_updates.keys()))
            except OSError as exc:
                logger.warning("Failed to persist daily settings to .env.local: %s", exc)

    return _current_daily_settings()


# ---------------------------------------------------------------------------
# News intelligence settings
# ---------------------------------------------------------------------------

_INTEL_ENV_MAP: dict[str, str] = {
    "enabled": "ATLAS_NEWS_ENABLED",
    "watchlist": "ATLAS_NEWS_WATCHLIST",
    "topics": "ATLAS_NEWS_TOPICS",
    "regions": "ATLAS_NEWS_REGIONS",
    "languages": "ATLAS_NEWS_LANGUAGES",
    "lookback_days": "ATLAS_NEWS_LOOKBACK_DAYS",
    "pressure_velocity_threshold": "ATLAS_NEWS_PRESSURE_VELOCITY_THRESHOLD",
    "signal_min_articles": "ATLAS_NEWS_SIGNAL_MIN_ARTICLES",
    "sentiment_enabled": "ATLAS_NEWS_SENTIMENT_ENABLED",
    "source_diversity_enabled": "ATLAS_NEWS_SOURCE_DIVERSITY_ENABLED",
    "composite_score_threshold": "ATLAS_NEWS_COMPOSITE_SCORE_THRESHOLD",
    "max_articles_per_topic": "ATLAS_NEWS_MAX_ARTICLES_PER_TOPIC",
    "llm_model": "ATLAS_NEWS_LLM_MODEL",
    "schedule_hour": "ATLAS_NEWS_SCHEDULE_HOUR",
    "notify_on_signal": "ATLAS_NEWS_NOTIFY_ON_SIGNAL",
    "notify_all_runs": "ATLAS_NEWS_NOTIFY_ALL_RUNS",
    "include_in_morning_briefing": "ATLAS_NEWS_INCLUDE_IN_MORNING_BRIEFING",
    # Linguistic pre-indicator patterns
    "linguistic_analysis_enabled": "ATLAS_NEWS_LINGUISTIC_ANALYSIS_ENABLED",
    "linguistic_hedge_enabled": "ATLAS_NEWS_LINGUISTIC_HEDGE_ENABLED",
    "linguistic_deflection_enabled": "ATLAS_NEWS_LINGUISTIC_DEFLECTION_ENABLED",
    "linguistic_insider_enabled": "ATLAS_NEWS_LINGUISTIC_INSIDER_ENABLED",
    "linguistic_escalation_enabled": "ATLAS_NEWS_LINGUISTIC_ESCALATION_ENABLED",
    "linguistic_permission_enabled": "ATLAS_NEWS_LINGUISTIC_PERMISSION_ENABLED",
    "linguistic_certainty_enabled": "ATLAS_NEWS_LINGUISTIC_CERTAINTY_ENABLED",
    "linguistic_dissociation_enabled": "ATLAS_NEWS_LINGUISTIC_DISSOCIATION_ENABLED",
    # SORAM Framework
    "soram_enabled": "ATLAS_NEWS_SORAM_ENABLED",
    "soram_societal_enabled": "ATLAS_NEWS_SORAM_SOCIETAL_ENABLED",
    "soram_operational_enabled": "ATLAS_NEWS_SORAM_OPERATIONAL_ENABLED",
    "soram_regulatory_enabled": "ATLAS_NEWS_SORAM_REGULATORY_ENABLED",
    "soram_alignment_enabled": "ATLAS_NEWS_SORAM_ALIGNMENT_ENABLED",
    "soram_media_novelty_enabled": "ATLAS_NEWS_SORAM_MEDIA_NOVELTY_ENABLED",
    # Alternative data sources
    "sec_edgar_enabled": "ATLAS_NEWS_SEC_EDGAR_ENABLED",
    "usaspending_enabled": "ATLAS_NEWS_USASPENDING_ENABLED",
    "state_sos_enabled": "ATLAS_NEWS_STATE_SOS_ENABLED",
    "county_recorder_enabled": "ATLAS_NEWS_COUNTY_RECORDER_ENABLED",
    "bls_enabled": "ATLAS_NEWS_BLS_ENABLED",
    # Signal streak / correlation
    "signal_streak_enabled": "ATLAS_NEWS_SIGNAL_STREAK_ENABLED",
    "signal_streak_threshold": "ATLAS_NEWS_SIGNAL_STREAK_THRESHOLD",
    "cross_entity_correlation_enabled": "ATLAS_NEWS_CROSS_ENTITY_CORRELATION_ENABLED",
    "cross_entity_min_signals": "ATLAS_NEWS_CROSS_ENTITY_MIN_SIGNALS",
}


class IntelligenceSettings(BaseModel):
    """User-configurable news intelligence / pressure signal settings."""

    # General
    enabled: bool = Field(description="Enable daily news intelligence analysis")

    # Watchlist
    watchlist: str = Field(
        description=(
            'JSON array of watched entities — each with name, type, query, ticker. '
            'Example: [{"name":"Apple","type":"company","query":"Apple AAPL","ticker":"AAPL"}]'
        )
    )

    # Simple-mode fallback
    topics: str = Field(
        description="Comma-separated plain topics (simple mode, used only when watchlist is empty)"
    )
    regions: str = Field(
        description="Comma-separated geographic focus areas prepended to simple-mode queries"
    )
    languages: str = Field(description="Comma-separated language codes for article filtering")

    # Pressure signal detection
    lookback_days: int = Field(
        description="Days of history used to establish the baseline article volume for each entity"
    )
    pressure_velocity_threshold: float = Field(
        description="Minimum volume growth multiplier to flag a volume-only signal"
    )
    signal_min_articles: int = Field(
        description="Minimum articles today to confirm a signal — prevents single-source noise"
    )

    # Multi-dimensional scoring
    sentiment_enabled: bool = Field(
        description="Score sentiment shift — tone change often precedes a movement"
    )
    source_diversity_enabled: bool = Field(
        description="Score source diversity — story spreading to new outlets strengthens the signal"
    )
    composite_score_threshold: float = Field(
        description="Minimum composite pressure score to flag a signal"
    )

    # Linguistic pre-indicator patterns
    linguistic_analysis_enabled: bool = Field(description="Enable linguistic pre-indicator analysis")
    linguistic_hedge_enabled: bool = Field(description="Detect hedging/uncertainty language")
    linguistic_deflection_enabled: bool = Field(description="Detect deflection/denial language")
    linguistic_insider_enabled: bool = Field(description="Detect insider/source language")
    linguistic_escalation_enabled: bool = Field(description="Detect escalation/urgency language")
    linguistic_permission_enabled: bool = Field(description="Detect moral permission language")
    linguistic_certainty_enabled: bool = Field(description="Detect certainty/moral panic language")
    linguistic_dissociation_enabled: bool = Field(description="Detect we/us → they/them dissociation language")

    # SORAM Framework
    soram_enabled: bool = Field(description="Enable SORAM framework channel analysis")
    soram_societal_enabled: bool = Field(description="SORAM Societal: coordinated threat/fear framing")
    soram_operational_enabled: bool = Field(description="SORAM Operational: drills and readiness exercises")
    soram_regulatory_enabled: bool = Field(description="SORAM Regulatory: new emergency powers/laws")
    soram_alignment_enabled: bool = Field(description="SORAM Alignment: scripted consensus messaging")
    soram_media_novelty_enabled: bool = Field(description="SORAM Media Novelty: breaking-news hijacking")

    # Alternative data sources
    sec_edgar_enabled: bool = Field(description="Fetch SEC 8-K filings for company entities (free EDGAR API)")
    usaspending_enabled: bool = Field(description="Fetch USAspending.gov contract awards (free API)")
    state_sos_enabled: bool = Field(description="Monitor State SoS filings (requires custom regional setup)")
    county_recorder_enabled: bool = Field(description="Monitor county recorder/permit data (requires custom setup)")
    bls_enabled: bool = Field(description="Fetch BLS/Census employment trend data (free API)")

    # Signal streak / correlation
    signal_streak_enabled: bool = Field(description="Track consecutive-day signal streaks per entity")
    signal_streak_threshold: int = Field(description="Consecutive elevated-signal days to trigger a streak alert")
    cross_entity_correlation_enabled: bool = Field(description="Detect macro signals across multiple entities")
    cross_entity_min_signals: int = Field(description="Min same-type entities signalling simultaneously for a macro alert")

    # Operations
    max_articles_per_topic: int = Field(
        description="Max articles fetched per entity per run (controls NewsAPI quota usage)"
    )
    llm_model: str = Field(description="Ollama model used to synthesise the intelligence briefing")
    schedule_hour: int = Field(description="Hour of day (0–23) to run the daily analysis")

    # Output
    notify_on_signal: bool = Field(description="Push a notification when pressure signals are detected")
    notify_all_runs: bool = Field(description="Push a notification after every run, even with no new signals")
    include_in_morning_briefing: bool = Field(
        description="Include active pressure signals in the morning briefing"
    )


class IntelligenceSettingsUpdate(BaseModel):
    """Partial update payload for news intelligence settings."""

    enabled: Optional[bool] = None
    watchlist: Optional[str] = None
    topics: Optional[str] = None
    regions: Optional[str] = None
    languages: Optional[str] = None
    lookback_days: Optional[int] = None
    pressure_velocity_threshold: Optional[float] = None
    signal_min_articles: Optional[int] = None
    sentiment_enabled: Optional[bool] = None
    source_diversity_enabled: Optional[bool] = None
    composite_score_threshold: Optional[float] = None
    max_articles_per_topic: Optional[int] = None
    llm_model: Optional[str] = None
    schedule_hour: Optional[int] = None
    notify_on_signal: Optional[bool] = None
    notify_all_runs: Optional[bool] = None
    include_in_morning_briefing: Optional[bool] = None
    # Linguistic
    linguistic_analysis_enabled: Optional[bool] = None
    linguistic_hedge_enabled: Optional[bool] = None
    linguistic_deflection_enabled: Optional[bool] = None
    linguistic_insider_enabled: Optional[bool] = None
    linguistic_escalation_enabled: Optional[bool] = None
    linguistic_permission_enabled: Optional[bool] = None
    linguistic_certainty_enabled: Optional[bool] = None
    linguistic_dissociation_enabled: Optional[bool] = None
    # SORAM
    soram_enabled: Optional[bool] = None
    soram_societal_enabled: Optional[bool] = None
    soram_operational_enabled: Optional[bool] = None
    soram_regulatory_enabled: Optional[bool] = None
    soram_alignment_enabled: Optional[bool] = None
    soram_media_novelty_enabled: Optional[bool] = None
    # Data sources
    sec_edgar_enabled: Optional[bool] = None
    usaspending_enabled: Optional[bool] = None
    state_sos_enabled: Optional[bool] = None
    county_recorder_enabled: Optional[bool] = None
    bls_enabled: Optional[bool] = None
    # Streak / correlation
    signal_streak_enabled: Optional[bool] = None
    signal_streak_threshold: Optional[int] = None
    cross_entity_correlation_enabled: Optional[bool] = None
    cross_entity_min_signals: Optional[int] = None


def _current_intelligence_settings() -> IntelligenceSettings:
    """Build an IntelligenceSettings snapshot from the live settings object."""
    n = settings.news_intel
    return IntelligenceSettings(
        enabled=n.enabled,
        watchlist=n.watchlist,
        topics=n.topics,
        regions=n.regions,
        languages=n.languages,
        lookback_days=n.lookback_days,
        pressure_velocity_threshold=n.pressure_velocity_threshold,
        signal_min_articles=n.signal_min_articles,
        sentiment_enabled=n.sentiment_enabled,
        source_diversity_enabled=n.source_diversity_enabled,
        composite_score_threshold=n.composite_score_threshold,
        max_articles_per_topic=n.max_articles_per_topic,
        llm_model=n.llm_model,
        schedule_hour=n.schedule_hour,
        notify_on_signal=n.notify_on_signal,
        notify_all_runs=n.notify_all_runs,
        include_in_morning_briefing=n.include_in_morning_briefing,
        # Linguistic
        linguistic_analysis_enabled=n.linguistic_analysis_enabled,
        linguistic_hedge_enabled=n.linguistic_hedge_enabled,
        linguistic_deflection_enabled=n.linguistic_deflection_enabled,
        linguistic_insider_enabled=n.linguistic_insider_enabled,
        linguistic_escalation_enabled=n.linguistic_escalation_enabled,
        linguistic_permission_enabled=n.linguistic_permission_enabled,
        linguistic_certainty_enabled=n.linguistic_certainty_enabled,
        linguistic_dissociation_enabled=n.linguistic_dissociation_enabled,
        # SORAM
        soram_enabled=n.soram_enabled,
        soram_societal_enabled=n.soram_societal_enabled,
        soram_operational_enabled=n.soram_operational_enabled,
        soram_regulatory_enabled=n.soram_regulatory_enabled,
        soram_alignment_enabled=n.soram_alignment_enabled,
        soram_media_novelty_enabled=n.soram_media_novelty_enabled,
        # Data sources
        sec_edgar_enabled=n.sec_edgar_enabled,
        usaspending_enabled=n.usaspending_enabled,
        state_sos_enabled=n.state_sos_enabled,
        county_recorder_enabled=n.county_recorder_enabled,
        bls_enabled=n.bls_enabled,
        # Streak / correlation
        signal_streak_enabled=n.signal_streak_enabled,
        signal_streak_threshold=n.signal_streak_threshold,
        cross_entity_correlation_enabled=n.cross_entity_correlation_enabled,
        cross_entity_min_signals=n.cross_entity_min_signals,
    )


@router.get("/intelligence", response_model=IntelligenceSettings)
async def get_intelligence_settings() -> IntelligenceSettings:
    """Return current user-configurable news intelligence settings."""
    return _current_intelligence_settings()


@router.patch("/intelligence", response_model=IntelligenceSettings)
async def update_intelligence_settings(updates: IntelligenceSettingsUpdate) -> IntelligenceSettings:
    """
    Update news intelligence settings.

    Changes are applied in-memory immediately and persisted to .env.local.
    Note: ATLAS_NEWS_API_KEY is a credential — set it in your .env file directly,
    not via this endpoint.
    """
    update_dict = updates.model_dump(exclude_none=True)

    if not update_dict:
        return _current_intelligence_settings()

    for field, value in update_dict.items():
        if hasattr(settings.news_intel, field):
            setattr(settings.news_intel, field, value)
            logger.info("Intelligence setting updated: %s = %r", field, value)

    env_updates: dict[str, str] = {}
    for field, value in update_dict.items():
        env_key = _INTEL_ENV_MAP.get(field)
        if env_key is not None:
            env_updates[env_key] = str(value)

    if env_updates:
        async with _env_write_lock:
            try:
                _write_env_local(env_updates)
                logger.info("Intelligence settings persisted to .env.local: %s", list(env_updates.keys()))
            except OSError as exc:
                logger.warning("Failed to persist intelligence settings to .env.local: %s", exc)

    return _current_intelligence_settings()


# ---------------------------------------------------------------------------
# LLM / AI Model settings
# ---------------------------------------------------------------------------

_LLM_ENV_MAP: dict[str, str] = {
    # LLMConfig (ATLAS_LLM_*)
    "backend": "ATLAS_LLM_DEFAULT_MODEL",
    "ollama_model": "ATLAS_LLM_OLLAMA_MODEL",
    "ollama_url": "ATLAS_LLM_OLLAMA_URL",
    "groq_model": "ATLAS_LLM_GROQ_MODEL",
    "together_model": "ATLAS_LLM_TOGETHER_MODEL",
    "cloud_enabled": "ATLAS_LLM_CLOUD_ENABLED",
    "cloud_ollama_model": "ATLAS_LLM_CLOUD_OLLAMA_MODEL",
    "model_swap_enabled": "ATLAS_LLM_MODEL_SWAP_ENABLED",
    "day_model": "ATLAS_LLM_DAY_MODEL",
    "night_model": "ATLAS_LLM_NIGHT_MODEL",
    "model_swap_day_cron": "ATLAS_LLM_MODEL_SWAP_DAY_CRON",
    "model_swap_night_cron": "ATLAS_LLM_MODEL_SWAP_NIGHT_CRON",
}


class LLMSettings(BaseModel):
    """User-configurable LLM / AI model settings.

    Credentials (groq_api_key, together_api_key, anthropic_api_key) are
    intentionally omitted -- manage those via environment variables.
    """

    # Backend selection
    backend: str = Field(
        description=(
            "Active LLM backend: ollama (local Ollama), groq (Groq cloud), "
            "together (Together AI), llama-cpp (GGUF file), or hybrid"
        )
    )

    # Ollama (local)
    ollama_model: str = Field(description="Ollama model tag used for all conversations (e.g. qwen3:14b)")
    ollama_url: str = Field(description="Ollama API base URL (default: http://localhost:11434)")

    # Groq (cloud -- low latency)
    groq_model: str = Field(description="Groq model name when backend=groq (e.g. llama-3.3-70b-versatile)")

    # Together AI (cloud)
    together_model: str = Field(description="Together AI model name when backend=together")

    # Cloud / hybrid routing
    cloud_enabled: bool = Field(
        description=(
            "Enable a second cloud LLM (via Ollama cloud relay) for business workflows "
            "like email drafting and booking -- runs alongside the local model"
        )
    )
    cloud_ollama_model: str = Field(
        description="Ollama cloud relay model tag for business workflows (e.g. minimax-m2:cloud)"
    )

    # Automatic day/night model swap
    model_swap_enabled: bool = Field(
        description=(
            "Swap Ollama models on a schedule to optimise VRAM -- "
            "loads a lighter model during the day, heavier model at night for background tasks"
        )
    )
    day_model: str = Field(
        description="Ollama model loaded during daytime hours (pre-warmed for conversations)"
    )
    night_model: str = Field(
        description="Ollama model loaded at night for background processing (e.g. memory graph extraction)"
    )
    model_swap_day_cron: str = Field(
        description="Cron expression for the day-model swap (default: 30 7 * * * = 7:30 AM)"
    )
    model_swap_night_cron: str = Field(
        description="Cron expression for the night-model swap (default: 0 0 * * * = midnight)"
    )


class LLMSettingsUpdate(BaseModel):
    """Partial update payload for LLM settings (all fields optional)."""

    backend: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_url: Optional[str] = None
    groq_model: Optional[str] = None
    together_model: Optional[str] = None
    cloud_enabled: Optional[bool] = None
    cloud_ollama_model: Optional[str] = None
    model_swap_enabled: Optional[bool] = None
    day_model: Optional[str] = None
    night_model: Optional[str] = None
    model_swap_day_cron: Optional[str] = None
    model_swap_night_cron: Optional[str] = None


def _current_llm_settings() -> LLMSettings:
    """Build an LLMSettings snapshot from the live settings object."""
    lc = settings.llm
    return LLMSettings(
        backend=lc.default_model,
        ollama_model=lc.ollama_model,
        ollama_url=lc.ollama_url,
        groq_model=lc.groq_model,
        together_model=lc.together_model,
        cloud_enabled=lc.cloud_enabled,
        cloud_ollama_model=lc.cloud_ollama_model,
        model_swap_enabled=lc.model_swap_enabled,
        day_model=lc.day_model,
        night_model=lc.night_model,
        model_swap_day_cron=lc.model_swap_day_cron,
        model_swap_night_cron=lc.model_swap_night_cron,
    )


# LLM API field -> LLMConfig attribute
_LLM_ATTR_MAP: dict[str, str] = {
    "backend": "default_model",
    "ollama_model": "ollama_model",
    "ollama_url": "ollama_url",
    "groq_model": "groq_model",
    "together_model": "together_model",
    "cloud_enabled": "cloud_enabled",
    "cloud_ollama_model": "cloud_ollama_model",
    "model_swap_enabled": "model_swap_enabled",
    "day_model": "day_model",
    "night_model": "night_model",
    "model_swap_day_cron": "model_swap_day_cron",
    "model_swap_night_cron": "model_swap_night_cron",
}


@router.get("/llm", response_model=LLMSettings)
async def get_llm_settings() -> LLMSettings:
    """Return current user-configurable LLM / AI model settings."""
    return _current_llm_settings()


@router.patch("/llm", response_model=LLMSettings)
async def update_llm_settings(updates: LLMSettingsUpdate) -> LLMSettings:
    """
    Update LLM / AI model settings.

    Changes apply in-memory immediately and are persisted to .env.local.
    API keys (GROQ_API_KEY, TOGETHER_API_KEY, ANTHROPIC_API_KEY) are credentials
    and must be set in your .env file directly.
    Note: backend and model_swap changes require a server restart to take full effect.
    """
    update_dict = updates.model_dump(exclude_none=True)

    if not update_dict:
        return _current_llm_settings()

    for field, value in update_dict.items():
        attr = _LLM_ATTR_MAP.get(field, field)
        if hasattr(settings.llm, attr):
            setattr(settings.llm, attr, value)
            logger.info("LLM setting updated: %s = %r", attr, value)

    env_updates: dict[str, str] = {}
    for field, value in update_dict.items():
        env_key = _LLM_ENV_MAP.get(field)
        if env_key is not None:
            env_updates[env_key] = str(value)

    if env_updates:
        async with _env_write_lock:
            try:
                _write_env_local(env_updates)
                logger.info("LLM settings persisted to .env.local: %s", list(env_updates.keys()))
            except OSError as exc:
                logger.warning("Failed to persist LLM settings to .env.local: %s", exc)

    return _current_llm_settings()


# ---------------------------------------------------------------------------
# Notifications settings (alerts, reminders, call intelligence)
# ---------------------------------------------------------------------------

_NOTIFY_ENV_MAP: dict[str, str] = {
    # AlertsConfig (ATLAS_ALERTS_*)
    "alerts_enabled": "ATLAS_ALERTS_ENABLED",
    "alerts_cooldown_seconds": "ATLAS_ALERTS_DEFAULT_COOLDOWN_SECONDS",
    "alerts_tts_enabled": "ATLAS_ALERTS_TTS_ENABLED",
    "alerts_persist": "ATLAS_ALERTS_PERSIST_ALERTS",
    "ntfy_enabled": "ATLAS_ALERTS_NTFY_ENABLED",
    "ntfy_url": "ATLAS_ALERTS_NTFY_URL",
    "ntfy_topic": "ATLAS_ALERTS_NTFY_TOPIC",
    # ReminderConfig (ATLAS_REMINDER_*)
    "reminders_enabled": "ATLAS_REMINDER_ENABLED",
    "reminder_timezone": "ATLAS_REMINDER_DEFAULT_TIMEZONE",
    "reminder_max_per_user": "ATLAS_REMINDER_MAX_REMINDERS_PER_USER",
    # CallIntelligenceConfig (ATLAS_CALL_INTELLIGENCE_*)
    "call_intel_enabled": "ATLAS_CALL_INTELLIGENCE_ENABLED",
    "call_min_duration_seconds": "ATLAS_CALL_INTELLIGENCE_MIN_DURATION_SECONDS",
    "call_notify_enabled": "ATLAS_CALL_INTELLIGENCE_NOTIFY_ENABLED",
}

_NOTIFY_ALERTS_FIELDS = {
    "alerts_enabled", "alerts_cooldown_seconds", "alerts_tts_enabled", "alerts_persist",
    "ntfy_enabled", "ntfy_url", "ntfy_topic",
}
_NOTIFY_REMINDER_FIELDS = {"reminders_enabled", "reminder_timezone", "reminder_max_per_user"}
_NOTIFY_CALL_FIELDS = {"call_intel_enabled", "call_min_duration_seconds", "call_notify_enabled"}

_NOTIFY_ALERTS_ATTR_MAP: dict[str, str] = {
    "alerts_enabled": "enabled",
    "alerts_cooldown_seconds": "default_cooldown_seconds",
    "alerts_tts_enabled": "tts_enabled",
    "alerts_persist": "persist_alerts",
    "ntfy_enabled": "ntfy_enabled",
    "ntfy_url": "ntfy_url",
    "ntfy_topic": "ntfy_topic",
}
_NOTIFY_REMINDER_ATTR_MAP: dict[str, str] = {
    "reminders_enabled": "enabled",
    "reminder_timezone": "default_timezone",
    "reminder_max_per_user": "max_reminders_per_user",
}
_NOTIFY_CALL_ATTR_MAP: dict[str, str] = {
    "call_intel_enabled": "enabled",
    "call_min_duration_seconds": "min_duration_seconds",
    "call_notify_enabled": "notify_enabled",
}


class NotificationSettings(BaseModel):
    """User-configurable notification settings (alerts, ntfy, reminders, call intelligence)."""

    # Alert system
    alerts_enabled: bool = Field(description="Enable the centralised alert system")
    alerts_cooldown_seconds: int = Field(
        description="Minimum seconds between repeated alerts of the same type"
    )
    alerts_tts_enabled: bool = Field(
        description="Speak alert messages aloud via TTS on edge nodes"
    )
    alerts_persist: bool = Field(
        description="Persist every alert to the database for history and reporting"
    )

    # ntfy push notifications
    ntfy_enabled: bool = Field(
        description="Enable ntfy push notifications -- receives alerts on your phone or browser"
    )
    ntfy_url: str = Field(description="ntfy server URL (self-hosted or ntfy.sh)")
    ntfy_topic: str = Field(
        description="ntfy topic to publish to -- subscribe to the same topic in the ntfy app"
    )

    # Reminders
    reminders_enabled: bool = Field(description="Enable the reminder system")
    reminder_timezone: str = Field(
        description="Default IANA timezone for reminder scheduling (e.g. America/Chicago)"
    )
    reminder_max_per_user: int = Field(
        description="Maximum active reminders allowed per user"
    )

    # Call intelligence
    call_intel_enabled: bool = Field(
        description="Automatically transcribe and extract data from Twilio call recordings after a call ends"
    )
    call_min_duration_seconds: int = Field(
        description="Skip post-call processing for calls shorter than this duration (seconds)"
    )
    call_notify_enabled: bool = Field(
        description="Push an ntfy notification with the call summary after processing"
    )


class NotificationSettingsUpdate(BaseModel):
    """Partial update payload for notification settings (all fields optional)."""

    alerts_enabled: Optional[bool] = None
    alerts_cooldown_seconds: Optional[int] = None
    alerts_tts_enabled: Optional[bool] = None
    alerts_persist: Optional[bool] = None
    ntfy_enabled: Optional[bool] = None
    ntfy_url: Optional[str] = None
    ntfy_topic: Optional[str] = None
    reminders_enabled: Optional[bool] = None
    reminder_timezone: Optional[str] = None
    reminder_max_per_user: Optional[int] = None
    call_intel_enabled: Optional[bool] = None
    call_min_duration_seconds: Optional[int] = None
    call_notify_enabled: Optional[bool] = None


def _current_notification_settings() -> NotificationSettings:
    """Build a NotificationSettings snapshot from the live settings objects."""
    al = settings.alerts
    re = settings.reminder
    ci = settings.call_intelligence
    return NotificationSettings(
        alerts_enabled=al.enabled,
        alerts_cooldown_seconds=al.default_cooldown_seconds,
        alerts_tts_enabled=al.tts_enabled,
        alerts_persist=al.persist_alerts,
        ntfy_enabled=al.ntfy_enabled,
        ntfy_url=al.ntfy_url,
        ntfy_topic=al.ntfy_topic,
        reminders_enabled=re.enabled,
        reminder_timezone=re.default_timezone,
        reminder_max_per_user=re.max_reminders_per_user,
        call_intel_enabled=ci.enabled,
        call_min_duration_seconds=ci.min_duration_seconds,
        call_notify_enabled=ci.notify_enabled,
    )


@router.get("/notifications", response_model=NotificationSettings)
async def get_notification_settings() -> NotificationSettings:
    """Return current user-configurable notification settings."""
    return _current_notification_settings()


@router.patch("/notifications", response_model=NotificationSettings)
async def update_notification_settings(updates: NotificationSettingsUpdate) -> NotificationSettings:
    """
    Update notification settings.

    Changes apply in-memory immediately and are persisted to .env.local.
    """
    update_dict = updates.model_dump(exclude_none=True)

    if not update_dict:
        return _current_notification_settings()

    for field, value in update_dict.items():
        if field in _NOTIFY_ALERTS_FIELDS:
            attr = _NOTIFY_ALERTS_ATTR_MAP.get(field, field)
            if hasattr(settings.alerts, attr):
                setattr(settings.alerts, attr, value)
                logger.info("Alerts setting updated: %s = %r", attr, value)
        elif field in _NOTIFY_REMINDER_FIELDS:
            attr = _NOTIFY_REMINDER_ATTR_MAP.get(field, field)
            if hasattr(settings.reminder, attr):
                setattr(settings.reminder, attr, value)
                logger.info("Reminder setting updated: %s = %r", attr, value)
        elif field in _NOTIFY_CALL_FIELDS:
            attr = _NOTIFY_CALL_ATTR_MAP.get(field, field)
            if hasattr(settings.call_intelligence, attr):
                setattr(settings.call_intelligence, attr, value)
                logger.info("Call intelligence setting updated: %s = %r", attr, value)

    env_updates: dict[str, str] = {}
    for field, value in update_dict.items():
        env_key = _NOTIFY_ENV_MAP.get(field)
        if env_key is not None:
            env_updates[env_key] = str(value)

    if env_updates:
        async with _env_write_lock:
            try:
                _write_env_local(env_updates)
                logger.info("Notification settings persisted to .env.local: %s", list(env_updates.keys()))
            except OSError as exc:
                logger.warning("Failed to persist notification settings to .env.local: %s", exc)

    return _current_notification_settings()


# ---------------------------------------------------------------------------
# Integrations settings (Home Assistant, MQTT, MCP servers)
# ---------------------------------------------------------------------------

_INTEG_ENV_MAP: dict[str, str] = {
    # HomeAssistantConfig (ATLAS_HA_*)
    "ha_enabled": "ATLAS_HA_ENABLED",
    "ha_url": "ATLAS_HA_URL",
    "ha_entity_filter": "ATLAS_HA_ENTITY_FILTER",
    "ha_websocket_enabled": "ATLAS_HA_WEBSOCKET_ENABLED",
    "ha_websocket_reconnect_interval": "ATLAS_HA_WEBSOCKET_RECONNECT_INTERVAL",
    "ha_state_cache_ttl": "ATLAS_HA_STATE_CACHE_TTL",
    # MQTTConfig (ATLAS_MQTT_*)
    "mqtt_enabled": "ATLAS_MQTT_ENABLED",
    "mqtt_host": "ATLAS_MQTT_HOST",
    "mqtt_port": "ATLAS_MQTT_PORT",
    "mqtt_username": "ATLAS_MQTT_USERNAME",
    # MCPConfig (ATLAS_MCP_*)
    "mcp_crm_enabled": "ATLAS_MCP_CRM_ENABLED",
    "mcp_email_enabled": "ATLAS_MCP_EMAIL_ENABLED",
    "mcp_calendar_enabled": "ATLAS_MCP_CALENDAR_ENABLED",
    "mcp_twilio_enabled": "ATLAS_MCP_TWILIO_ENABLED",
    "mcp_transport": "ATLAS_MCP_TRANSPORT",
}

_INTEG_HA_FIELDS = {
    "ha_enabled", "ha_url", "ha_entity_filter", "ha_websocket_enabled",
    "ha_websocket_reconnect_interval", "ha_state_cache_ttl",
}
_INTEG_MQTT_FIELDS = {"mqtt_enabled", "mqtt_host", "mqtt_port", "mqtt_username"}
_INTEG_MCP_FIELDS = {
    "mcp_crm_enabled", "mcp_email_enabled", "mcp_calendar_enabled",
    "mcp_twilio_enabled", "mcp_transport",
}

_INTEG_HA_ATTR_MAP: dict[str, str] = {
    "ha_enabled": "enabled",
    "ha_url": "url",
    "ha_entity_filter": "entity_filter",
    "ha_websocket_enabled": "websocket_enabled",
    "ha_websocket_reconnect_interval": "websocket_reconnect_interval",
    "ha_state_cache_ttl": "state_cache_ttl",
}
_INTEG_MQTT_ATTR_MAP: dict[str, str] = {
    "mqtt_enabled": "enabled",
    "mqtt_host": "host",
    "mqtt_port": "port",
    "mqtt_username": "username",
}
_INTEG_MCP_ATTR_MAP: dict[str, str] = {
    "mcp_crm_enabled": "crm_enabled",
    "mcp_email_enabled": "email_enabled",
    "mcp_calendar_enabled": "calendar_enabled",
    "mcp_twilio_enabled": "twilio_enabled",
    "mcp_transport": "transport",
}


class IntegrationSettings(BaseModel):
    """User-configurable integration settings (Home Assistant, MQTT, MCP)."""

    # Home Assistant
    ha_enabled: bool = Field(description="Enable Home Assistant backend for device control")
    ha_url: str = Field(description="Home Assistant base URL (e.g. http://homeassistant.local:8123)")
    ha_entity_filter: str = Field(
        description=(
            'JSON array of entity-prefix strings to auto-discover '
            '(e.g. ["light.","switch.","media_player."])'
        )
    )
    ha_websocket_enabled: bool = Field(
        description="Enable WebSocket for real-time HA state updates (recommended)"
    )
    ha_websocket_reconnect_interval: int = Field(
        description="Seconds between WebSocket reconnection attempts"
    )
    ha_state_cache_ttl: int = Field(
        description="Seconds to cache HA entity state before treating it as stale"
    )

    # MQTT
    mqtt_enabled: bool = Field(description="Enable MQTT backend for direct device messaging")
    mqtt_host: str = Field(description="MQTT broker hostname or IP address")
    mqtt_port: int = Field(description="MQTT broker port (default 1883)")
    mqtt_username: str = Field(
        description="MQTT broker username (leave empty if no authentication required)"
    )

    # MCP servers (Claude Desktop / Cursor tool integration)
    mcp_crm_enabled: bool = Field(description="Enable CRM MCP server (exposes contacts tools to AI clients)")
    mcp_email_enabled: bool = Field(description="Enable Email MCP server (exposes email tools to AI clients)")
    mcp_calendar_enabled: bool = Field(
        description="Enable Calendar MCP server (exposes scheduling tools to AI clients)"
    )
    mcp_twilio_enabled: bool = Field(
        description="Enable Twilio MCP server (exposes call and SMS tools to AI clients)"
    )
    mcp_transport: str = Field(
        description="MCP transport mode: stdio (Claude Desktop/Cursor) or sse (HTTP endpoint)"
    )


class IntegrationSettingsUpdate(BaseModel):
    """Partial update payload for integration settings (all fields optional)."""

    # Home Assistant
    ha_enabled: Optional[bool] = None
    ha_url: Optional[str] = None
    ha_entity_filter: Optional[str] = None
    ha_websocket_enabled: Optional[bool] = None
    ha_websocket_reconnect_interval: Optional[int] = None
    ha_state_cache_ttl: Optional[int] = None

    # MQTT
    mqtt_enabled: Optional[bool] = None
    mqtt_host: Optional[str] = None
    mqtt_port: Optional[int] = None
    mqtt_username: Optional[str] = None

    # MCP
    mcp_crm_enabled: Optional[bool] = None
    mcp_email_enabled: Optional[bool] = None
    mcp_calendar_enabled: Optional[bool] = None
    mcp_twilio_enabled: Optional[bool] = None
    mcp_transport: Optional[str] = None


def _current_integration_settings() -> IntegrationSettings:
    """Build an IntegrationSettings snapshot from the live settings objects."""
    import json
    ha = settings.homeassistant
    mq = settings.mqtt
    mc = settings.mcp
    return IntegrationSettings(
        ha_enabled=ha.enabled,
        ha_url=ha.url,
        ha_entity_filter=json.dumps(ha.entity_filter),
        ha_websocket_enabled=ha.websocket_enabled,
        ha_websocket_reconnect_interval=ha.websocket_reconnect_interval,
        ha_state_cache_ttl=ha.state_cache_ttl,
        mqtt_enabled=mq.enabled,
        mqtt_host=mq.host,
        mqtt_port=mq.port,
        mqtt_username=mq.username or "",
        mcp_crm_enabled=mc.crm_enabled,
        mcp_email_enabled=mc.email_enabled,
        mcp_calendar_enabled=mc.calendar_enabled,
        mcp_twilio_enabled=mc.twilio_enabled,
        mcp_transport=mc.transport,
    )


@router.get("/integrations", response_model=IntegrationSettings)
async def get_integration_settings() -> IntegrationSettings:
    """Return current user-configurable integration settings."""
    return _current_integration_settings()


@router.patch("/integrations", response_model=IntegrationSettings)
async def update_integration_settings(updates: IntegrationSettingsUpdate) -> IntegrationSettings:
    """
    Update integration settings.

    Changes apply in-memory immediately and are persisted to .env.local.
    Credentials (HA token, MQTT password) must be set in your .env file directly.
    Note: enabling/disabling HA or MQTT backends requires a server restart.
    """
    import json
    update_dict = updates.model_dump(exclude_none=True)

    if not update_dict:
        return _current_integration_settings()

    for field, value in update_dict.items():
        if field in _INTEG_HA_FIELDS:
            attr = _INTEG_HA_ATTR_MAP.get(field, field)
            if attr == "entity_filter":
                try:
                    value = json.loads(value) if isinstance(value, str) else value
                except (ValueError, TypeError):
                    pass
            if hasattr(settings.homeassistant, attr):
                setattr(settings.homeassistant, attr, value)
                logger.info("Home Assistant setting updated: %s = %r", attr, value)
        elif field in _INTEG_MQTT_FIELDS:
            attr = _INTEG_MQTT_ATTR_MAP.get(field, field)
            if hasattr(settings.mqtt, attr):
                setattr(settings.mqtt, attr, value)
                logger.info("MQTT setting updated: %s = %r", attr, value)
        elif field in _INTEG_MCP_FIELDS:
            attr = _INTEG_MCP_ATTR_MAP.get(field, field)
            if hasattr(settings.mcp, attr):
                setattr(settings.mcp, attr, value)
                logger.info("MCP setting updated: %s = %r", attr, value)

    env_updates: dict[str, str] = {}
    for field, value in update_dict.items():
        env_key = _INTEG_ENV_MAP.get(field)
        if env_key is not None:
            env_updates[env_key] = str(value)

    if env_updates:
        async with _env_write_lock:
            try:
                _write_env_local(env_updates)
                logger.info("Integration settings persisted to .env.local: %s", list(env_updates.keys()))
            except OSError as exc:
                logger.warning("Failed to persist integration settings to .env.local: %s", exc)

    return _current_integration_settings()
