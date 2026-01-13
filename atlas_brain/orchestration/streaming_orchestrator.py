"""
Streaming orchestrator for low-latency voice interaction.

Optimizes the pipeline for human-like response times:
- Multi-model routing (1B, 8B, Cloud loaded simultaneously)
- Intent-based tier selection (no model swapping)
- Streaming LLM responses
- Sentence-level TTS chunking for faster first response
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Optional

from .audio_buffer import AudioBuffer, VADConfig
from .states import PipelineContext, PipelineEvent, PipelineState, PipelineStateMachine
from .streaming_intent import IntentCategory, StreamingIntentDetector, streaming_intent_detector

logger = logging.getLogger("atlas.orchestration.streaming")


@dataclass
class StreamingConfig:
    """Configuration for streaming orchestrator."""

    # VAD
    vad_config: VADConfig = field(default_factory=VADConfig)

    # Streaming STT
    streaming_stt_enabled: bool = True  # Use streaming STT for early intent detection
    early_warmup_enabled: bool = True  # Warm up LLM when conversation detected early

    # Streaming behavior
    stream_tts: bool = True  # Stream TTS as response generates
    min_tts_chunk_chars: int = 50  # Min characters before TTS
    sentence_delimiters: str = ".!?;"  # Split response at these

    # Behavior
    auto_execute: bool = True

    # Timeouts
    recording_timeout_ms: int = 30000
    processing_timeout_ms: int = 10000

    @classmethod
    def from_settings(cls) -> "StreamingConfig":
        """Create config from environment settings."""
        from ..config import settings

        orch = settings.orchestration

        vad_config = VADConfig(
            aggressiveness=orch.vad_aggressiveness,
            silence_duration_ms=orch.silence_duration_ms,
        )

        return cls(
            vad_config=vad_config,
            auto_execute=orch.auto_execute,
            recording_timeout_ms=orch.recording_timeout_ms,
            processing_timeout_ms=orch.processing_timeout_ms,
        )


@dataclass
class StreamingResult:
    """Result with streaming support."""

    success: bool = True
    transcript: Optional[str] = None
    intent: Optional[Any] = None
    action_results: list[dict] = field(default_factory=list)

    # Streaming response
    response_chunks: list[str] = field(default_factory=list)
    audio_chunks: list[bytes] = field(default_factory=list)

    # Timing
    first_audio_ms: float = 0.0  # Time to first audio chunk
    total_ms: float = 0.0

    error: Optional[str] = None


class StreamingOrchestrator:
    """
    Low-latency streaming voice orchestrator.

    Key optimizations:
    1. Multi-model pool (1B + 8B loaded simultaneously)
    2. Intent-based routing (no model swapping)
    3. Streaming LLM responses
    4. Sentence-level TTS chunking
    5. Play audio while still generating
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        agent: Optional[Any] = None,
    ):
        self.config = config or StreamingConfig.from_settings()

        # Agent for delegated reasoning (optional)
        # Note: Agent is used for intent/action only; streaming LLM is handled separately
        self._agent = agent

        # Components
        self._state_machine = PipelineStateMachine()
        self._audio_buffer = AudioBuffer(self.config.vad_config)

        # Service references (lazy loaded)
        self._stt = None
        self._tts = None
        self._intent_parser = None
        self._action_dispatcher = None

        # Multi-model routing (new)
        self._model_pool = None
        self._intent_router = None

        # Streaming intent detection
        self._streaming_intent_detector = streaming_intent_detector
        self._warmup_task: Optional[asyncio.Task] = None
        self._warmup_triggered = False

        # Streaming callbacks
        self._on_transcript: Optional[Callable] = None
        self._on_response_chunk: Optional[Callable] = None
        self._on_audio_chunk: Optional[Callable] = None

        logger.info(
            "Streaming orchestrator initialized (agent=%s)",
            "enabled" if self._agent else "disabled",
        )

    @property
    def agent(self) -> Optional[Any]:
        """Get the agent instance (if any)."""
        return self._agent

    @agent.setter
    def agent(self, value: Any) -> None:
        """Set the agent instance."""
        self._agent = value
        logger.info("Streaming orchestrator agent %s", "enabled" if value else "disabled")

    def _get_stt(self):
        if self._stt is None:
            from ..services import stt_registry
            self._stt = stt_registry.get_active()
        return self._stt

    def _get_model_pool(self):
        """Get the multi-model pool."""
        if self._model_pool is None:
            from ..services.model_pool import get_model_pool
            self._model_pool = get_model_pool()
        return self._model_pool

    def _get_intent_router(self):
        """Get the intent router."""
        if self._intent_router is None:
            from ..services.intent_router import get_intent_router
            self._intent_router = get_intent_router()
        return self._intent_router

    def _get_tts(self):
        if self._tts is None:
            from ..services import tts_registry
            self._tts = tts_registry.get_active()
        return self._tts

    def _get_intent_parser(self):
        if self._intent_parser is None:
            from ..capabilities.intent_parser import intent_parser
            self._intent_parser = intent_parser
        return self._intent_parser

    def _get_action_dispatcher(self):
        if self._action_dispatcher is None:
            from ..capabilities.actions import action_dispatcher
            self._action_dispatcher = action_dispatcher
        return self._action_dispatcher

    def set_callbacks(
        self,
        on_transcript: Optional[Callable] = None,
        on_response_chunk: Optional[Callable] = None,
        on_audio_chunk: Optional[Callable] = None,
    ):
        """Set streaming callbacks."""
        self._on_transcript = on_transcript
        self._on_response_chunk = on_response_chunk
        self._on_audio_chunk = on_audio_chunk

    async def _trigger_llm_warmup(self) -> None:
        """Trigger LLM warmup in the background."""
        if self._warmup_triggered:
            return

        self._warmup_triggered = True
        pool = self._get_model_pool()

        if pool is None or not pool._initialized:
            logger.debug("Model pool not initialized, skipping warmup")
            return

        from ..services.model_pool import ModelTier

        # Warm up the FAST tier with the system prompt
        system_prompt = (
            "You are Atlas, a helpful voice assistant. "
            "Respond naturally and concisely in 1-2 sentences."
        )

        try:
            await pool.warmup_context(ModelTier.FAST, system_prompt)
            logger.info("LLM warmup completed (early intent detection)")
        except Exception as e:
            logger.warning("LLM warmup failed: %s", e)

    def _reset_warmup_state(self) -> None:
        """Reset warmup state for new utterance."""
        self._warmup_triggered = False
        if self._warmup_task and not self._warmup_task.done():
            self._warmup_task.cancel()
        self._warmup_task = None

    async def _process_streaming_stt(self, stt, audio_bytes: bytes) -> str:
        """
        Process audio through streaming STT with early intent detection.

        Args:
            stt: STT service with transcribe_streaming method
            audio_bytes: Complete audio bytes to process

        Returns:
            Final transcript
        """
        import numpy as np
        from ..config import settings

        # Load audio from bytes
        try:
            from ..services.stt.nemotron import _load_audio_from_bytes, SAMPLE_RATE
            audio, orig_sr = _load_audio_from_bytes(audio_bytes)

            # Resample if needed
            if orig_sr != SAMPLE_RATE:
                from ..services.stt.nemotron import _resample_audio
                audio = _resample_audio(audio, orig_sr, SAMPLE_RATE)
        except Exception as e:
            logger.warning("Failed to load audio for streaming: %s", e)
            # Fall back to batch transcription
            result = await stt.transcribe(audio_bytes)
            return result.get("transcript", "")

        # Get chunk size from config
        frame_len_ms = settings.stt.nemotron_frame_len_ms
        chunk_samples = int(SAMPLE_RATE * frame_len_ms / 1000)

        # Clear STT buffer for new utterance
        if hasattr(stt, "clear_audio_buffer"):
            stt.clear_audio_buffer()

        # Process audio in chunks
        last_transcript = ""
        last_category = None

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]

            # Pad last chunk if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

            # Get streaming transcription result
            result = await stt.transcribe_streaming(chunk)
            transcript = result.get("transcript", "")

            # Skip if no transcript or unchanged
            if not transcript or transcript == last_transcript:
                continue

            last_transcript = transcript

            # Classify partial transcript for early intent detection
            if self.config.early_warmup_enabled:
                streaming_intent = self._streaming_intent_detector.classify(transcript)

                # Log category change
                if streaming_intent.category != last_category:
                    last_category = streaming_intent.category
                    logger.debug(
                        "Partial intent: '%s...' -> %s (conf=%.2f)",
                        transcript[:30],
                        streaming_intent.category.value,
                        streaming_intent.confidence,
                    )

                # Trigger warmup for conversation/question intents
                if streaming_intent.should_warmup_llm():
                    logger.info("Early intent detected: %s - triggering LLM warmup",
                               streaming_intent.category.value)
                    asyncio.create_task(self._trigger_llm_warmup())

        # Get final transcript from buffer
        if hasattr(stt, "transcribe_buffer"):
            final_result = await stt.transcribe_buffer()
            return final_result.get("transcript", last_transcript)

        return last_transcript

    async def process_utterance_streaming(
        self,
        audio_bytes: bytes,
    ) -> AsyncIterator[dict]:
        """
        Process utterance with streaming responses.

        Yields events as they happen:
        - {"type": "partial_transcript", "text": "...", "category": "..."}
        - {"type": "transcript", "text": "..."}
        - {"type": "intent", "action": "...", ...}
        - {"type": "response_chunk", "text": "..."}
        - {"type": "audio_chunk", "data": bytes}
        - {"type": "complete", "total_ms": ...}
        """
        start_time = datetime.now()
        first_audio_time = None

        # Reset warmup state for this utterance
        self._reset_warmup_state()

        # Phase 1: STT with streaming intent detection
        stt = self._get_stt()
        if stt is None:
            yield {"type": "error", "message": "STT not available"}
            return

        transcript = ""
        try:
            # Check if streaming STT is available and enabled
            has_streaming = hasattr(stt, "transcribe_streaming")

            if self.config.streaming_stt_enabled and has_streaming:
                # Use streaming STT with early intent detection
                transcript = await self._process_streaming_stt(stt, audio_bytes)
            else:
                # Fall back to batch transcription
                transcription = await stt.transcribe(audio_bytes)
                transcript = transcription.get("transcript", "")

            yield {"type": "transcript", "text": transcript}

            if self._on_transcript:
                await self._on_transcript(transcript)

        except Exception as e:
            yield {"type": "error", "message": f"STT failed: {e}"}
            return

        # Phase 2: Intent parsing (fast - regex)
        intent_parser = self._get_intent_parser()
        intent = None
        if intent_parser:
            try:
                intent = await intent_parser.parse(transcript)
                if intent:
                    yield {
                        "type": "intent",
                        "action": intent.action,
                        "target_type": intent.target_type,
                        "target_name": intent.target_name,
                        "confidence": intent.confidence,
                    }
            except Exception as e:
                logger.warning("Intent parsing failed: %s", e)

        # Phase 3: Action execution OR LLM response
        if intent and self.config.auto_execute:
            # Device command - execute and respond
            dispatcher = self._get_action_dispatcher()
            if dispatcher:
                try:
                    action_result = await dispatcher.dispatch_intent(intent)
                    yield {
                        "type": "action_result",
                        "success": action_result.success,
                        "message": action_result.message,
                    }

                    # Generate simple response
                    response_text = action_result.message or "Done."
                    yield {"type": "response_chunk", "text": response_text}

                    # TTS for action response
                    async for audio_event in self._stream_tts(response_text):
                        if first_audio_time is None:
                            first_audio_time = datetime.now()
                        yield audio_event

                except Exception as e:
                    logger.warning("Action failed: %s", e)
                    yield {"type": "error", "message": str(e)}
        else:
            # Conversational - stream LLM response
            async for event in self._stream_llm_response(transcript):
                if event["type"] == "audio_chunk" and first_audio_time is None:
                    first_audio_time = datetime.now()
                yield event

        # Calculate timing
        total_ms = (datetime.now() - start_time).total_seconds() * 1000
        first_audio_ms = 0.0
        if first_audio_time:
            first_audio_ms = (first_audio_time - start_time).total_seconds() * 1000

        yield {
            "type": "complete",
            "total_ms": total_ms,
            "first_audio_ms": first_audio_ms,
        }

    async def _stream_llm_response(self, transcript: str) -> AsyncIterator[dict]:
        """
        Stream LLM response with multi-model routing and sentence-level TTS.

        Key optimizations:
        1. Intent router picks the right model tier instantly
        2. Multiple models loaded = no swap overhead
        3. TTS starts as soon as we have a complete sentence
        """
        from ..services.model_pool import ModelTier

        # Route to appropriate model tier
        router = self._get_intent_router()
        pool = self._get_model_pool()

        routing = router.route_with_fallback(
            transcript,
            pool.get_available_tiers(),
        )

        yield {
            "type": "routing",
            "tier": routing.tier.name,
            "confidence": routing.confidence,
            "reason": routing.reason,
        }

        logger.info(
            "Routed '%s...' to %s (confidence=%.2f, reason=%s)",
            transcript[:30], routing.tier.name, routing.confidence, routing.reason
        )

        # ACTION tier means device command - should have been handled already
        if routing.tier == ModelTier.ACTION:
            yield {"type": "response_chunk", "text": "I'll handle that command."}
            return

        # Build messages
        messages = [
            {
                "role": "system",
                "content": "You are Atlas, a helpful voice assistant. Respond naturally and concisely in 1-2 sentences.",
            },
            {"role": "user", "content": transcript},
        ]

        # Determine max tokens based on tier
        max_tokens = {
            ModelTier.FAST: 100,
            ModelTier.BALANCED: 256,
            ModelTier.POWERFUL: 512,
            ModelTier.CLOUD: 512,
        }.get(routing.tier, 150)

        try:
            # Check if pool is initialized
            if not pool._initialized:
                logger.warning("Model pool not initialized, falling back to registry")
                yield {"type": "response_chunk", "text": f"I heard: {transcript}"}
                return

            # Stream response from the selected tier
            buffer = ""
            async for token in pool.chat_stream(
                routing.tier,
                messages,
                max_tokens=max_tokens,
            ):
                buffer += token
                yield {"type": "token", "text": token}

                # Check for sentence boundary
                sentence, remaining = self._extract_sentence(buffer)
                if sentence:
                    buffer = remaining
                    yield {"type": "response_chunk", "text": sentence}

                    # TTS for this sentence
                    async for audio_event in self._stream_tts(sentence):
                        yield audio_event

            # Handle remaining text
            if buffer.strip():
                yield {"type": "response_chunk", "text": buffer}
                async for audio_event in self._stream_tts(buffer):
                    yield audio_event

        except Exception as e:
            logger.error("Model pool chat failed: %s", e)
            # Fallback to old single-model approach
            try:
                from ..services import llm_registry
                llm = llm_registry.get_active()
                if llm:
                    from ..services.protocols import Message
                    msgs = [Message(role=m["role"], content=m["content"]) for m in messages]
                    result = llm.chat(messages=msgs, max_tokens=150)
                    response = result.get("response", f"I heard: {transcript}")
                    yield {"type": "response_chunk", "text": response}
                    async for audio_event in self._stream_tts(response):
                        yield audio_event
                else:
                    yield {"type": "response_chunk", "text": f"I heard: {transcript}"}
            except Exception as e2:
                logger.error("Fallback also failed: %s", e2)
                yield {"type": "response_chunk", "text": f"I heard: {transcript}"}

    def _extract_sentence(self, text: str) -> tuple[str, str]:
        """
        Extract first complete sentence from text.

        Returns (sentence, remaining) or ("", text) if no complete sentence.
        """
        for delim in self.config.sentence_delimiters:
            idx = text.find(delim)
            if idx != -1:
                sentence = text[: idx + 1].strip()
                remaining = text[idx + 1 :].strip()
                if len(sentence) >= self.config.min_tts_chunk_chars:
                    return sentence, remaining
        return "", text

    async def _stream_tts(self, text: str) -> AsyncIterator[dict]:
        """
        Stream TTS audio.

        For now, generates full audio then yields. Future: true streaming.
        """
        tts = self._get_tts()
        if tts is None:
            return

        try:
            audio_bytes = await tts.synthesize(text)
            yield {"type": "audio_chunk", "data": audio_bytes}

            if self._on_audio_chunk:
                await self._on_audio_chunk(audio_bytes)

        except Exception as e:
            logger.warning("TTS failed: %s", e)

    async def process_text_streaming(self, text: str) -> AsyncIterator[dict]:
        """
        Process text input with streaming (bypass STT).

        Useful for testing or text interfaces.
        """
        start_time = datetime.now()
        first_audio_time = None

        yield {"type": "transcript", "text": text}

        # Intent parsing
        intent_parser = self._get_intent_parser()
        intent = None
        if intent_parser:
            try:
                intent = await intent_parser.parse(text)
                if intent:
                    yield {
                        "type": "intent",
                        "action": intent.action,
                        "target_type": intent.target_type,
                        "target_name": intent.target_name,
                        "confidence": intent.confidence,
                    }
            except Exception as e:
                logger.warning("Intent parsing failed: %s", e)

        # Action or LLM
        if intent and self.config.auto_execute:
            dispatcher = self._get_action_dispatcher()
            if dispatcher:
                try:
                    action_result = await dispatcher.dispatch_intent(intent)
                    yield {
                        "type": "action_result",
                        "success": action_result.success,
                        "message": action_result.message,
                    }

                    response_text = action_result.message or "Done."
                    yield {"type": "response_chunk", "text": response_text}

                    async for audio_event in self._stream_tts(response_text):
                        if first_audio_time is None:
                            first_audio_time = datetime.now()
                        yield audio_event

                except Exception as e:
                    yield {"type": "error", "message": str(e)}
        else:
            async for event in self._stream_llm_response(text):
                if event["type"] == "audio_chunk" and first_audio_time is None:
                    first_audio_time = datetime.now()
                yield event

        total_ms = (datetime.now() - start_time).total_seconds() * 1000
        first_audio_ms = 0.0
        if first_audio_time:
            first_audio_ms = (first_audio_time - start_time).total_seconds() * 1000

        yield {
            "type": "complete",
            "total_ms": total_ms,
            "first_audio_ms": first_audio_ms,
        }


# Global instance
_streaming_orchestrator: Optional[StreamingOrchestrator] = None


def get_streaming_orchestrator() -> StreamingOrchestrator:
    """Get or create the streaming orchestrator."""
    global _streaming_orchestrator
    if _streaming_orchestrator is None:
        _streaming_orchestrator = StreamingOrchestrator()
    return _streaming_orchestrator
