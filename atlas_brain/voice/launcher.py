"""
Voice pipeline launcher for Atlas Brain.

Integrates the voice pipeline with AtlasAgent.
"""

import asyncio
import logging
import signal
import sys
import threading
from typing import Any, Dict, Optional

from ..agents.atlas import get_atlas_agent
from ..agents.protocols import AgentContext
from ..config import settings
from .pipeline import NemotronAsrHttpClient, PiperTTS, VoicePipeline

logger = logging.getLogger("atlas.voice.launcher")

_voice_pipeline: Optional[VoicePipeline] = None
_voice_thread: Optional[threading.Thread] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _create_agent_runner():
    """Create a sync wrapper for the async AtlasAgent."""
    agent = get_atlas_agent()

    def runner(transcript: str, context_dict: Dict[str, Any]) -> str:
        """Run the agent synchronously."""
        if _event_loop is None:
            logger.error("No event loop available for agent runner")
            return ""

        ctx = AgentContext(
            input_text=transcript,
            session_id=context_dict.get("session_id"),
        )

        try:
            future = asyncio.run_coroutine_threadsafe(
                agent.run(ctx),
                _event_loop,
            )
            result = future.result(timeout=30.0)
            response = result.response_text or ""
            logger.info("Agent runner response: %s", response[:100] if response else "(empty)")
            return response
        except Exception as e:
            logger.error("Agent runner failed: %s", e, exc_info=True)
            return ""

    return runner


def create_voice_pipeline() -> Optional[VoicePipeline]:
    """Create the voice pipeline from config."""
    cfg = settings.voice

    if not cfg.enabled:
        logger.info("Voice pipeline disabled in config")
        return None

    if not cfg.wakeword_model_paths:
        logger.warning("No wake word models configured")

    if not cfg.asr_url:
        logger.warning("No ASR URL configured")

    if not cfg.piper_binary or not cfg.piper_model:
        logger.warning("Piper TTS not fully configured")

    asr_client = NemotronAsrHttpClient(
        url=cfg.asr_url,
        api_key=cfg.asr_api_key,
        timeout=cfg.asr_timeout,
    )

    tts = PiperTTS(
        binary_path=cfg.piper_binary,
        model_path=cfg.piper_model,
        speaker=cfg.piper_speaker,
        length_scale=cfg.piper_length_scale,
        noise_scale=cfg.piper_noise_scale,
        noise_w=cfg.piper_noise_w,
    )

    agent_runner = _create_agent_runner()

    pipeline = VoicePipeline(
        wakeword_model_paths=cfg.wakeword_model_paths,
        wake_threshold=cfg.wake_threshold,
        asr_client=asr_client,
        tts=tts,
        agent_runner=agent_runner,
        sample_rate=cfg.sample_rate,
        block_size=cfg.block_size,
        silence_ms=cfg.silence_ms,
        max_command_seconds=cfg.max_command_seconds,
        vad_aggressiveness=cfg.vad_aggressiveness,
        hangover_ms=cfg.hangover_ms,
        use_arecord=cfg.use_arecord,
        arecord_device=cfg.arecord_device,
        stop_hotkey=cfg.stop_hotkey,
        allow_wake_barge_in=cfg.allow_wake_barge_in,
        interrupt_on_speech=cfg.interrupt_on_speech,
        interrupt_speech_frames=cfg.interrupt_speech_frames,
        interrupt_rms_threshold=cfg.interrupt_rms_threshold,
        interrupt_wake_models=cfg.interrupt_wake_models,
        interrupt_wake_threshold=cfg.interrupt_wake_threshold,
        command_workers=cfg.command_workers,
        audio_gain=cfg.audio_gain,
    )

    return pipeline


def start_voice_pipeline(loop: asyncio.AbstractEventLoop) -> bool:
    """
    Start the voice pipeline in a background thread.

    Args:
        loop: The main asyncio event loop for agent calls

    Returns:
        True if started successfully
    """
    global _voice_pipeline, _voice_thread, _event_loop

    if _voice_thread is not None and _voice_thread.is_alive():
        logger.warning("Voice pipeline already running")
        return True

    _event_loop = loop

    try:
        _voice_pipeline = create_voice_pipeline()
        if _voice_pipeline is None:
            return False
    except Exception as e:
        logger.error("Failed to create voice pipeline: %s", e)
        return False

    def run_pipeline():
        """Run the pipeline (blocking)."""
        try:
            _voice_pipeline.start()
        except Exception as e:
            logger.error("Voice pipeline crashed: %s", e)

    _voice_thread = threading.Thread(
        target=run_pipeline,
        name="voice-pipeline",
        daemon=True,
    )
    _voice_thread.start()

    logger.info("Voice pipeline started in background thread")
    return True


def stop_voice_pipeline() -> None:
    """Stop the voice pipeline."""
    global _voice_pipeline, _voice_thread

    if _voice_pipeline is not None:
        try:
            _voice_pipeline.playback.stop()
            _voice_pipeline.command_executor.shutdown()
        except Exception as e:
            logger.warning("Error stopping voice pipeline: %s", e)

    _voice_pipeline = None
    _voice_thread = None

    logger.info("Voice pipeline stopped")


def get_voice_pipeline() -> Optional[VoicePipeline]:
    """Get the active voice pipeline instance."""
    return _voice_pipeline
