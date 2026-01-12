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
from .wake_word import WakeWordConfig, WakeWordManager

logger = logging.getLogger("atlas.orchestration.streaming")


@dataclass
class StreamingConfig:
    """Configuration for streaming orchestrator."""

    # Wake word
    wake_word_enabled: bool = True
    wake_word_config: WakeWordConfig = field(default_factory=WakeWordConfig)
    require_wake_word: bool = False

    # VAD
    vad_config: VADConfig = field(default_factory=VADConfig)

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

        wake_word_config = WakeWordConfig(
            threshold=orch.wake_word_threshold,
            wake_words=orch.wake_words,
        )

        vad_config = VADConfig(
            aggressiveness=orch.vad_aggressiveness,
            silence_duration_ms=orch.silence_duration_ms,
        )

        return cls(
            wake_word_enabled=orch.wake_word_enabled,
            wake_word_config=wake_word_config,
            require_wake_word=orch.require_wake_word,
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

    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig.from_settings()

        # Components
        self._state_machine = PipelineStateMachine()
        self._audio_buffer = AudioBuffer(self.config.vad_config)
        self._wake_word: Optional[WakeWordManager] = None

        # Service references (lazy loaded)
        self._stt = None
        self._tts = None
        self._intent_parser = None
        self._action_dispatcher = None

        # Multi-model routing (new)
        self._model_pool = None
        self._intent_router = None

        # Streaming callbacks
        self._on_transcript: Optional[Callable] = None
        self._on_response_chunk: Optional[Callable] = None
        self._on_audio_chunk: Optional[Callable] = None

        if self.config.wake_word_enabled:
            self._wake_word = WakeWordManager(self.config.wake_word_config)

        logger.info("Streaming orchestrator initialized")

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

    async def process_utterance_streaming(
        self,
        audio_bytes: bytes,
    ) -> AsyncIterator[dict]:
        """
        Process utterance with streaming responses.

        Yields events as they happen:
        - {"type": "transcript", "text": "..."}
        - {"type": "intent", "action": "...", ...}
        - {"type": "response_chunk", "text": "..."}
        - {"type": "audio_chunk", "data": bytes}
        - {"type": "complete", "total_ms": ...}
        """
        start_time = datetime.now()
        first_audio_time = None

        # Phase 1: STT (can't parallelize - need transcript first)
        stt = self._get_stt()
        if stt is None:
            yield {"type": "error", "message": "STT not available"}
            return

        try:
            transcription = await stt.transcribe(audio_bytes)
            transcript = transcription.get("transcript", "")

            yield {"type": "transcript", "text": transcript}

            if self._on_transcript:
                await self._on_transcript(transcript)

        except Exception as e:
            yield {"type": "error", "message": f"STT failed: {e}"}
            return

        # Phase 2: Intent parsing (fast - VLM)
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
