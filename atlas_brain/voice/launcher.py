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
from .pipeline import (
    NemotronAsrHttpClient,
    NemotronAsrStreamingClient,
    PiperTTS,
    VoicePipeline,
)

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


# Static system prompt for prefill (matches AtlasAgent._generate_llm_response)
_PREFILL_SYSTEM_PROMPT = (
    "You are Atlas, a capable personal assistant. "
    "You can control smart home devices, answer questions, have conversations, and help with various tasks. "
    "Be conversational, helpful, and concise. Keep responses to 1-3 sentences unless more detail is needed."
)


def _create_prefill_runner():
    """Create a prefill runner to warm up LLM KV cache on wake word detection."""
    from ..services import llm_registry
    from ..services.protocols import Message
    import time

    def runner() -> None:
        """Prefill the LLM system prompt."""
        start_time = time.perf_counter()
        logger.info("LLM prefill STARTING...")

        if _event_loop is None:
            logger.warning("No event loop available for prefill")
            return

        llm = llm_registry.get_active()
        if llm is None:
            logger.warning("No active LLM for prefill")
            return

        # Check if LLM supports prefill
        if not hasattr(llm, "prefill_async"):
            logger.warning("LLM does not support prefill_async")
            return

        messages = [Message(role="system", content=_PREFILL_SYSTEM_PROMPT)]
        logger.info("Prefill sending system prompt (%d chars)", len(_PREFILL_SYSTEM_PROMPT))

        try:
            future = asyncio.run_coroutine_threadsafe(
                llm.prefill_async(messages),
                _event_loop,
            )
            result = future.result(timeout=10.0)
            prefill_ms = result.get("prefill_time_ms", 0)
            prompt_tokens = result.get("prompt_tokens", 0)
            total_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "LLM prefill DONE: ollama_time=%.1fms, total=%.1fms, tokens=%d",
                prefill_ms, total_ms, prompt_tokens
            )
        except Exception as e:
            logger.warning("Prefill failed after %.1fms: %s",
                          (time.perf_counter() - start_time) * 1000, e)

    return runner


def create_voice_pipeline() -> Optional[VoicePipeline]:
    """Create the voice pipeline from config."""
    cfg = settings.voice

    if not cfg.enabled:
        logger.info("Voice pipeline disabled in config")
        return None

    # Log full configuration at startup for debugging
    logger.info("=== Voice Pipeline Configuration ===")
    logger.info("  enabled=%s", cfg.enabled)
    logger.info("  sample_rate=%d, block_size=%d", cfg.sample_rate, cfg.block_size)
    logger.info("  use_arecord=%s, arecord_device=%s", cfg.use_arecord, cfg.arecord_device)
    logger.info("  input_device=%s", cfg.input_device)
    logger.info("  audio_gain=%.2f", cfg.audio_gain)
    logger.info("  wake_threshold=%.3f", cfg.wake_threshold)
    logger.info("  wakeword_model_paths=%s", cfg.wakeword_model_paths)
    logger.info("  asr_url=%s", cfg.asr_url)
    logger.info("  asr_streaming_enabled=%s, asr_ws_url=%s", cfg.asr_streaming_enabled, cfg.asr_ws_url)
    logger.info("  piper_binary=%s", cfg.piper_binary)
    logger.info("  piper_model=%s", cfg.piper_model)
    logger.info("  vad_aggressiveness=%d", cfg.vad_aggressiveness)
    logger.info("  silence_ms=%d, hangover_ms=%d", cfg.silence_ms, cfg.hangover_ms)
    logger.info("  debug_logging=%s, log_interval_frames=%d", cfg.debug_logging, cfg.log_interval_frames)
    logger.info("  conversation_mode=%s, timeout=%dms", cfg.conversation_mode_enabled, cfg.conversation_timeout_ms)
    logger.info("====================================")

    if not cfg.wakeword_model_paths:
        logger.warning("No wake word models configured")

    if not cfg.piper_binary or not cfg.piper_model:
        logger.warning("Piper TTS not fully configured")

    # Create ASR client (streaming or HTTP based on config)
    if cfg.asr_streaming_enabled and cfg.asr_ws_url:
        logger.info("Using streaming ASR: %s", cfg.asr_ws_url)
        try:
            asr_client = NemotronAsrStreamingClient(
                url=cfg.asr_ws_url,
                timeout=cfg.asr_timeout,
                sample_rate=cfg.sample_rate,
            )
        except ImportError as e:
            logger.warning("Streaming ASR unavailable (%s), falling back to HTTP", e)
            asr_client = NemotronAsrHttpClient(
                url=cfg.asr_url,
                api_key=cfg.asr_api_key,
                timeout=cfg.asr_timeout,
            )
    else:
        if not cfg.asr_url:
            logger.warning("No ASR URL configured")
        logger.info("Using HTTP batch ASR: %s", cfg.asr_url)
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
    prefill_runner = _create_prefill_runner()

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
        min_command_ms=cfg.min_command_ms,
        vad_aggressiveness=cfg.vad_aggressiveness,
        hangover_ms=cfg.hangover_ms,
        use_arecord=cfg.use_arecord,
        arecord_device=cfg.arecord_device,
        input_device=cfg.input_device,
        stop_hotkey=cfg.stop_hotkey,
        allow_wake_barge_in=cfg.allow_wake_barge_in,
        interrupt_on_speech=cfg.interrupt_on_speech,
        interrupt_speech_frames=cfg.interrupt_speech_frames,
        interrupt_rms_threshold=cfg.interrupt_rms_threshold,
        interrupt_wake_models=cfg.interrupt_wake_models,
        interrupt_wake_threshold=cfg.interrupt_wake_threshold,
        command_workers=cfg.command_workers,
        audio_gain=cfg.audio_gain,
        prefill_runner=prefill_runner,
        debug_logging=cfg.debug_logging,
        log_interval_frames=cfg.log_interval_frames,
        conversation_mode_enabled=cfg.conversation_mode_enabled,
        conversation_timeout_ms=cfg.conversation_timeout_ms,
        conversation_start_delay_ms=cfg.conversation_start_delay_ms,
        conversation_speech_frames=cfg.conversation_speech_frames,
        conversation_speech_tolerance=cfg.conversation_speech_tolerance,
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
