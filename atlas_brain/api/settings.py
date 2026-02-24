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


def _write_env_local(updates: dict[str, str]) -> None:
    """Upsert key=value pairs in .env.local, preserving unrelated lines.

    Values that contain ``=`` are handled correctly because we only split on
    the *first* ``=`` when extracting existing keys.
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
        new_line = f"{key}={value}"
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
