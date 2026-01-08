"""
Central orchestrator for the Atlas audio pipeline.

Coordinates the flow:
Wake Word → VAD → STT → Intent/LLM → Action → TTS
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Optional

from .audio_buffer import AudioBuffer, VADConfig
from .states import (
    PipelineContext,
    PipelineEvent,
    PipelineState,
    PipelineStateMachine,
)
from .wake_word import WakeWordConfig, WakeWordManager

logger = logging.getLogger("atlas.orchestration")


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    # Wake word settings
    wake_word_enabled: bool = True
    wake_word_config: WakeWordConfig = field(default_factory=WakeWordConfig)

    # VAD settings
    vad_config: VADConfig = field(default_factory=VADConfig)

    # Behavior
    auto_execute: bool = True  # Auto-execute actions or return for confirmation
    require_wake_word: bool = False  # If False, any speech triggers pipeline

    # Timeouts
    wake_word_timeout_ms: int = 30000  # Max time waiting for wake word
    recording_timeout_ms: int = 30000  # Max recording duration
    processing_timeout_ms: int = 10000  # Max time for LLM processing


@dataclass
class OrchestratorResult:
    """Result of processing a user request."""

    success: bool
    transcript: Optional[str] = None
    intent: Optional[Any] = None
    action_results: list[dict] = field(default_factory=list)
    response_text: Optional[str] = None
    response_audio: Optional[bytes] = None
    error: Optional[str] = None
    latency_ms: float = 0.0

    # Speaker identification
    speaker_id: Optional[str] = None
    speaker_confidence: float = 0.0

    # Detailed timing
    transcription_ms: float = 0.0
    speaker_id_ms: float = 0.0
    processing_ms: float = 0.0
    action_ms: float = 0.0
    tts_ms: float = 0.0


class Orchestrator:
    """
    Central coordinator for the Atlas voice pipeline.

    Manages the complete flow from audio input to action execution.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()

        # Components
        self._state_machine = PipelineStateMachine()
        self._audio_buffer = AudioBuffer(self.config.vad_config)
        self._wake_word: Optional[WakeWordManager] = None

        # Service references (lazy loaded)
        self._stt = None
        self._vlm = None
        self._llm = None
        self._tts = None
        self._speaker_id = None
        self._intent_parser = None
        self._action_dispatcher = None

        # Event callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_transcript: Optional[Callable] = None
        self._on_response: Optional[Callable] = None

        # Initialize wake word if enabled
        if self.config.wake_word_enabled:
            self._wake_word = WakeWordManager(self.config.wake_word_config)

        logger.info(
            "Orchestrator initialized (wake_word=%s, require_wake=%s)",
            self.config.wake_word_enabled,
            self.config.require_wake_word,
        )

    @property
    def state(self) -> PipelineState:
        """Current pipeline state."""
        return self._state_machine.state

    @property
    def context(self) -> PipelineContext:
        """Current pipeline context."""
        return self._state_machine.context

    def _get_stt(self):
        """Lazy load STT service."""
        if self._stt is None:
            from ..services import stt_registry
            self._stt = stt_registry.get_active()
        return self._stt

    def _get_vlm(self):
        """Lazy load VLM service."""
        if self._vlm is None:
            from ..services import vlm_registry
            self._vlm = vlm_registry.get_active()
        return self._vlm

    def _get_speaker_id(self):
        """Lazy load speaker ID service."""
        if self._speaker_id is None:
            from ..services import speaker_id_registry
            self._speaker_id = speaker_id_registry.get_active()
        return self._speaker_id

    def _get_tts(self):
        """Lazy load TTS service."""
        if self._tts is None:
            from ..services import tts_registry
            self._tts = tts_registry.get_active()
        return self._tts

    def _get_llm(self):
        """Lazy load LLM service for reasoning."""
        if self._llm is None:
            from ..services import llm_registry
            self._llm = llm_registry.get_active()
        return self._llm

    def _get_intent_parser(self):
        """Lazy load intent parser."""
        if self._intent_parser is None:
            from ..capabilities.intent_parser import intent_parser
            self._intent_parser = intent_parser
        return self._intent_parser

    def _get_action_dispatcher(self):
        """Lazy load action dispatcher."""
        if self._action_dispatcher is None:
            from ..capabilities.actions import action_dispatcher
            self._action_dispatcher = action_dispatcher
        return self._action_dispatcher

    def set_callbacks(
        self,
        on_state_change: Optional[Callable] = None,
        on_transcript: Optional[Callable] = None,
        on_response: Optional[Callable] = None,
    ) -> None:
        """Set event callbacks for real-time updates."""
        self._on_state_change = on_state_change
        self._on_transcript = on_transcript
        self._on_response = on_response

        # Wire up state machine listener
        if on_state_change:
            self._state_machine.add_listener(
                lambda event, old, new, ctx: on_state_change(event, old, new, ctx)
            )

    def reset(self) -> None:
        """Reset the orchestrator to idle state."""
        self._state_machine.reset()
        self._audio_buffer.reset()
        if self._wake_word:
            self._wake_word.reset()
        logger.debug("Orchestrator reset")

    async def process_audio_stream(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> OrchestratorResult:
        """
        Process a stream of audio data through the full pipeline.

        This is the main entry point for continuous audio processing.

        Args:
            audio_stream: Async iterator yielding audio chunks (16-bit, 16kHz, mono)

        Returns:
            OrchestratorResult with transcript, actions, and response
        """
        start_time = datetime.now()
        self.reset()

        # Start in listening state
        if self.config.require_wake_word:
            self._state_machine._state = PipelineState.LISTENING
        else:
            self._state_machine._state = PipelineState.IDLE

        try:
            async for audio_chunk in audio_stream:
                result = await self._process_audio_chunk(audio_chunk)
                if result is not None:
                    result.latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                    return result

            # Stream ended without completing
            return OrchestratorResult(
                success=False,
                error="Audio stream ended without detecting speech",
            )

        except asyncio.TimeoutError:
            return OrchestratorResult(
                success=False,
                error="Pipeline timeout",
            )
        except Exception as e:
            logger.exception("Pipeline error")
            return OrchestratorResult(
                success=False,
                error=str(e),
            )

    async def _process_audio_chunk(
        self,
        audio_chunk: bytes,
    ) -> Optional[OrchestratorResult]:
        """
        Process a single audio chunk.

        Returns OrchestratorResult when pipeline completes, None otherwise.
        """
        state = self._state_machine.state

        # Check for wake word if required
        if state in (PipelineState.IDLE, PipelineState.LISTENING):
            if self.config.require_wake_word and self._wake_word:
                detected, confidence, wake_word = self._wake_word.detect(audio_chunk)
                if detected:
                    logger.info("Wake word detected: %s (%.2f)", wake_word, confidence)
                    self._state_machine.context.wake_detected_at = datetime.now()
                    self._state_machine.transition(PipelineEvent.WAKE_WORD_DETECTED)

        # Process audio through VAD
        if state in (
            PipelineState.IDLE,
            PipelineState.WAKE_DETECTED,
            PipelineState.RECORDING,
        ):
            event = self._audio_buffer.add_audio(audio_chunk)

            if event == "speech_start":
                if not self.config.require_wake_word or state == PipelineState.WAKE_DETECTED:
                    self._state_machine.context.recording_started_at = datetime.now()
                    self._state_machine.transition(PipelineEvent.SPEECH_START)
                    logger.debug("Speech started")

            elif event in ("speech_end", "max_duration"):
                self._state_machine.context.recording_ended_at = datetime.now()
                self._state_machine.transition(PipelineEvent.SPEECH_END)
                logger.debug("Speech ended")

                # Get the recorded audio
                audio_bytes = self._audio_buffer.get_utterance()
                if audio_bytes:
                    self._state_machine.context.audio_bytes = audio_bytes
                    return await self._process_utterance()

        return None

    async def _process_utterance(self) -> OrchestratorResult:
        """Process a complete utterance through STT → Intent → Action → TTS."""
        ctx = self._state_machine.context
        result = OrchestratorResult(success=True)

        # Speaker identification (run in parallel with transcription conceptually)
        speaker_id_service = self._get_speaker_id()
        if speaker_id_service is not None:
            try:
                sid_start = datetime.now()
                match = speaker_id_service.identify(ctx.audio_bytes)
                ctx.speaker_id = match.name
                ctx.speaker_confidence = match.confidence
                result.speaker_id = match.name
                result.speaker_confidence = match.confidence
                result.speaker_id_ms = (datetime.now() - sid_start).total_seconds() * 1000

                if match.is_known:
                    logger.info(
                        "Speaker identified: %s (%.2f, %.0fms)",
                        match.name, match.confidence, result.speaker_id_ms
                    )
                else:
                    logger.debug("Unknown speaker (best match: %.2f)", match.confidence)
            except Exception as e:
                logger.warning("Speaker identification failed: %s", e)

        # Transcription
        trans_start = datetime.now()
        self._state_machine.transition(PipelineEvent.TRANSCRIPTION_COMPLETE)  # Move to PROCESSING

        stt = self._get_stt()
        if stt is None:
            return OrchestratorResult(success=False, error="STT service not available")

        try:
            transcription = await stt.transcribe(ctx.audio_bytes)
            ctx.transcript = transcription.get("text", "")
            ctx.transcript_confidence = transcription.get("confidence", 0.0)
            result.transcript = ctx.transcript
            result.transcription_ms = (datetime.now() - trans_start).total_seconds() * 1000

            logger.info("Transcription: '%s' (%.0fms)", ctx.transcript, result.transcription_ms)

            if self._on_transcript:
                await self._on_transcript(ctx.transcript)

        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return OrchestratorResult(success=False, error=f"Transcription failed: {e}")

        # Intent parsing / LLM processing
        proc_start = datetime.now()

        intent_parser = self._get_intent_parser()
        if intent_parser:
            try:
                intent = await intent_parser.parse(ctx.transcript)
                ctx.intent = intent
                result.intent = intent
                logger.info("Intent: %s", intent)
            except Exception as e:
                logger.warning("Intent parsing failed: %s", e)

        result.processing_ms = (datetime.now() - proc_start).total_seconds() * 1000
        self._state_machine.transition(PipelineEvent.INTENT_PARSED)

        # Action execution
        if self.config.auto_execute and ctx.intent:
            action_start = datetime.now()

            dispatcher = self._get_action_dispatcher()
            if dispatcher:
                try:
                    action_result = await dispatcher.dispatch_intent(ctx.intent)
                    ctx.action_results.append(action_result)
                    result.action_results = ctx.action_results
                except Exception as e:
                    logger.warning("Action execution failed: %s", e)

            result.action_ms = (datetime.now() - action_start).total_seconds() * 1000
            self._state_machine.transition(PipelineEvent.ACTION_COMPLETE)

        # Generate response text
        result.response_text = self._generate_response_text(ctx)
        ctx.response_text = result.response_text

        # TTS (if available)
        tts = self._get_tts()
        if tts is not None and result.response_text:
            try:
                tts_start = datetime.now()
                result.response_audio = await tts.synthesize(result.response_text)
                ctx.response_audio = result.response_audio
                result.tts_ms = (datetime.now() - tts_start).total_seconds() * 1000
                logger.info(
                    "TTS: %d bytes in %.0fms",
                    len(result.response_audio),
                    result.tts_ms,
                )
            except Exception as e:
                logger.warning("TTS synthesis failed: %s", e)

        self._state_machine.transition(PipelineEvent.RESPONSE_READY)

        if self._on_response:
            await self._on_response(result)

        return result

    def _generate_response_text(self, ctx: PipelineContext) -> str:
        """Generate a natural language response based on the action results."""
        if ctx.error:
            return f"Sorry, there was an error: {ctx.error}"

        if not ctx.action_results:
            if ctx.intent:
                return f"I understood '{ctx.intent.action}' but couldn't execute it."
            # No intent - use LLM for conversational response
            return self._generate_llm_response(ctx)

        # Summarize action results with LLM
        # ActionResult can be object or dict depending on context
        def is_success(r):
            if hasattr(r, "success"):
                return r.success
            return r.get("success", False) if isinstance(r, dict) else False

        successes = [r for r in ctx.action_results if is_success(r)]
        if successes:
            return self._generate_action_response(ctx, successes)
        else:
            return "I couldn't complete that action."

    def _generate_llm_response(self, ctx: PipelineContext) -> str:
        """Generate a conversational response using the LLM."""
        llm = self._get_llm()
        if llm is None:
            return f"I heard: {ctx.transcript}"

        # Build context for the LLM
        speaker = f"Speaker: {ctx.speaker_id}" if ctx.speaker_id != "unknown" else ""

        prompt = f"""You are Atlas, a helpful home automation assistant. Respond briefly and naturally to the user.

{speaker}
User said: "{ctx.transcript}"

Respond in 1-2 sentences:"""

        try:
            result = llm.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.7,
            )
            response = result.get("response", "").strip()
            if response:
                return response
        except Exception as e:
            logger.warning("LLM response generation failed: %s", e)

        return f"I heard: {ctx.transcript}"

    def _generate_action_response(self, ctx: PipelineContext, successes: list) -> str:
        """
        Generate a response describing completed actions.

        Uses the action result messages directly - no LLM needed for simple confirmations.
        This keeps response time fast (~500ms total instead of ~9s with LLM).
        """
        messages = []
        for r in successes:
            msg = r.message if hasattr(r, "message") else r.get("message", "Done")
            messages.append(msg)
        return " ".join(messages) if messages else "Done."

    async def process_text(self, text: str) -> OrchestratorResult:
        """
        Process a text command directly (skipping STT).

        Useful for testing or text-based interfaces.
        """
        start_time = datetime.now()

        # Reset context for each new text query
        self._state_machine.reset()
        ctx = self._state_machine.context
        ctx.transcript = text

        result = OrchestratorResult(success=True, transcript=text)

        # Intent parsing
        proc_start = datetime.now()
        intent_parser = self._get_intent_parser()
        if intent_parser:
            try:
                intent = await intent_parser.parse(text)
                ctx.intent = intent
                result.intent = intent
            except Exception as e:
                logger.warning("Intent parsing failed: %s", e)

        result.processing_ms = (datetime.now() - proc_start).total_seconds() * 1000

        # Action execution
        if self.config.auto_execute and ctx.intent:
            action_start = datetime.now()
            dispatcher = self._get_action_dispatcher()
            if dispatcher:
                try:
                    action_result = await dispatcher.dispatch_intent(ctx.intent)
                    ctx.action_results = [action_result]
                    result.action_results = [action_result]
                except Exception as e:
                    logger.warning("Action execution failed: %s", e)
            result.action_ms = (datetime.now() - action_start).total_seconds() * 1000

        result.response_text = self._generate_response_text(ctx)
        result.latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        return result


# Global orchestrator instance
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def reset_orchestrator() -> None:
    """Reset the global orchestrator."""
    global _orchestrator
    if _orchestrator is not None:
        _orchestrator.reset()
