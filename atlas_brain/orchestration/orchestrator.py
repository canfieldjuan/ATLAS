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

logger = logging.getLogger("atlas.orchestration")

# Global lock to prevent concurrent CUDA operations (STT + LLM conflict)
_cuda_lock = asyncio.Lock()


def calculate_audio_rms(wav_bytes: bytes) -> float:
    """
    Calculate RMS (root mean square) energy of WAV audio.

    Args:
        wav_bytes: WAV file bytes with header

    Returns:
        RMS energy value (0-32768 for 16-bit audio)
    """
    import struct
    import wave
    import io

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            n_frames = wav_file.getnframes()
            if n_frames == 0:
                return 0.0
            audio_data = wav_file.readframes(n_frames)

        # Convert to 16-bit samples
        samples = struct.unpack(f"<{len(audio_data)//2}h", audio_data)
        if not samples:
            return 0.0

        # Calculate RMS
        sum_squares = sum(s * s for s in samples)
        rms = (sum_squares / len(samples)) ** 0.5
        return rms
    except Exception as e:
        logger.warning("Failed to calculate audio RMS: %s", e)
        return 0.0


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    # VAD settings
    vad_config: VADConfig = field(default_factory=VADConfig)

    # Keyword detection - only respond when keyword is in transcript
    keyword_enabled: bool = True
    keyword: str = "atlas"

    # Behavior
    auto_execute: bool = True  # Auto-execute actions or return for confirmation

    # Follow-up mode: After a response, stay "hot" for follow-up commands
    follow_up_enabled: bool = True
    follow_up_duration_ms: int = 20000  # 20 seconds of hot mic after response

    # Timeouts
    recording_timeout_ms: int = 30000  # Max recording duration
    processing_timeout_ms: int = 10000  # Max time for LLM processing

    # Audio quality threshold - skip STT if audio RMS is below this
    min_audio_rms: int = 0  # Disabled for testing

    @classmethod
    def from_settings(cls) -> "OrchestratorConfig":
        """Create config from environment settings."""
        from ..config import settings

        orch = settings.orchestration

        # Build VAD config from settings
        vad_config = VADConfig(
            aggressiveness=orch.vad_aggressiveness,
            silence_duration_ms=orch.silence_duration_ms,
        )

        return cls(
            vad_config=vad_config,
            keyword_enabled=getattr(orch, "keyword_enabled", True),
            keyword=getattr(orch, "keyword", "atlas"),
            auto_execute=orch.auto_execute,
            follow_up_enabled=getattr(orch, "follow_up_enabled", True),
            follow_up_duration_ms=getattr(orch, "follow_up_duration_ms", 20000),
            recording_timeout_ms=orch.recording_timeout_ms,
            processing_timeout_ms=orch.processing_timeout_ms,
        )


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

    # Detailed timing (all in milliseconds)
    transcription_ms: float = 0.0
    speaker_id_ms: float = 0.0
    processing_ms: float = 0.0  # Intent parsing
    action_ms: float = 0.0
    llm_ms: float = 0.0  # LLM response generation
    memory_ms: float = 0.0  # RAG/memory retrieval
    tools_ms: float = 0.0  # Tool execution (weather, traffic, location)
    tts_ms: float = 0.0
    storage_ms: float = 0.0  # Database persistence

    def timing_breakdown(self) -> dict[str, float]:
        """Get timing breakdown as a dictionary."""
        return {
            "total": self.latency_ms,
            "transcription": self.transcription_ms,
            "speaker_id": self.speaker_id_ms,
            "intent_parsing": self.processing_ms,
            "action": self.action_ms,
            "llm": self.llm_ms,
            "memory": self.memory_ms,
            "tools": self.tools_ms,
            "tts": self.tts_ms,
            "storage": self.storage_ms,
        }


class Orchestrator:
    """
    Central coordinator for the Atlas voice pipeline.

    Manages the complete flow from audio input to action execution.

    Can optionally delegate reasoning to an Agent instance.
    When an agent is provided, the orchestrator handles only audio I/O
    (STT, TTS) and delegates intent parsing, action execution, and
    response generation to the agent.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        session_id: Optional[str] = None,
        agent: Optional[Any] = None,
    ):
        self.config = config or OrchestratorConfig.from_settings()

        # Agent for delegated reasoning (optional)
        self._agent = agent

        # Components
        self._state_machine = PipelineStateMachine()
        self._audio_buffer = AudioBuffer(self.config.vad_config)

        # Service references (lazy loaded)
        self._stt = None
        self._vlm = None
        self._llm = None
        self._tts = None
        self._speaker_id = None
        self._intent_parser = None
        self._action_dispatcher = None
        self._model_router = None
        self._memory_client = None

        # Session management for persistence
        self._session_id: Optional[str] = session_id

        # Event callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_transcript: Optional[Callable] = None
        self._on_response: Optional[Callable] = None

        # Follow-up mode tracking
        self._last_response_time: Optional[datetime] = None

        logger.info(
            "Orchestrator initialized (keyword=%s/%s, follow_up=%s/%dms, agent=%s)",
            self.config.keyword_enabled,
            self.config.keyword if self.config.keyword_enabled else "disabled",
            self.config.follow_up_enabled,
            self.config.follow_up_duration_ms,
            "enabled" if self._agent else "disabled",
        )

    @property
    def state(self) -> PipelineState:
        """Current pipeline state."""
        return self._state_machine.state

    @property
    def context(self) -> PipelineContext:
        """Current pipeline context."""
        return self._state_machine.context

    @property
    def in_follow_up_mode(self) -> bool:
        """Check if we're in follow-up mode (no wake word required)."""
        if not self.config.follow_up_enabled:
            return False
        if self._last_response_time is None:
            return False
        elapsed = (datetime.now() - self._last_response_time).total_seconds() * 1000
        return elapsed < self.config.follow_up_duration_ms

    @property
    def agent(self) -> Optional[Any]:
        """Get the agent instance (if any)."""
        return self._agent

    @agent.setter
    def agent(self, value: Any) -> None:
        """Set the agent instance."""
        self._agent = value
        logger.info("Agent %s", "enabled" if value else "disabled")

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

    def _get_model_router(self):
        """Lazy load model router."""
        if self._model_router is None:
            from .model_router import get_model_router
            self._model_router = get_model_router()
        return self._model_router

    def _get_memory_client(self):
        """Lazy load memory client for atlas-memory."""
        if self._memory_client is None:
            from ..services.memory import get_memory_client
            self._memory_client = get_memory_client()
        return self._memory_client

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
        logger.debug("Orchestrator reset")

    def _log_latency_summary(
        self,
        result: OrchestratorResult,
        pipeline_type: str = "audio",
    ) -> None:
        """
        Log a comprehensive latency summary for the pipeline.

        Args:
            result: The OrchestratorResult with timing data
            pipeline_type: "audio" or "text" to indicate pipeline type
        """
        # Determine if this was a command or conversation
        query_type = "command" if result.intent else "conversation"

        # Build timing breakdown string (only non-zero values)
        timing_parts = []
        if result.transcription_ms > 0:
            timing_parts.append(f"stt={result.transcription_ms:.0f}")
        if result.speaker_id_ms > 0:
            timing_parts.append(f"spkr={result.speaker_id_ms:.0f}")
        if result.processing_ms > 0:
            timing_parts.append(f"intent={result.processing_ms:.0f}")
        if result.action_ms > 0:
            timing_parts.append(f"action={result.action_ms:.0f}")
        if result.memory_ms > 0:
            timing_parts.append(f"rag={result.memory_ms:.0f}")
        if result.llm_ms > 0:
            timing_parts.append(f"llm={result.llm_ms:.0f}")
        if result.tts_ms > 0:
            timing_parts.append(f"tts={result.tts_ms:.0f}")
        if result.storage_ms > 0:
            timing_parts.append(f"db={result.storage_ms:.0f}")

        timing_str = " | ".join(timing_parts) if timing_parts else "no breakdown"

        # Log summary
        logger.info(
            "Pipeline complete [%s/%s] total=%.0fms (%s)",
            pipeline_type,
            query_type,
            result.latency_ms,
            timing_str,
        )

        # Log detailed breakdown at debug level
        logger.debug(
            "Latency breakdown: %s",
            result.timing_breakdown(),
        )

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

        # Start in idle state (always listening mode)
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
        # Debug: log that we're receiving audio
        if not hasattr(self, "_audio_chunk_count"):
            self._audio_chunk_count = 0
        self._audio_chunk_count += 1
        if self._audio_chunk_count % 100 == 1:  # Log every 100 chunks (~3 seconds)
            logger.info("AUDIO: chunk #%d (%d bytes)", self._audio_chunk_count, len(audio_chunk))

        state = self._state_machine.state

        # Process audio through VAD (always listening mode)
        if state in (PipelineState.IDLE, PipelineState.RECORDING):
            event = self._audio_buffer.add_audio(audio_chunk)

            if event == "speech_start":
                if self.in_follow_up_mode:
                    logger.info("Follow-up speech detected")
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

        # Clear any stale data from previous processing
        ctx.intent = None
        ctx.action_results = []

        # Speaker identification (run in parallel with transcription conceptually)
        speaker_id_service = self._get_speaker_id()
        if speaker_id_service is not None:
            try:
                sid_start = datetime.now()
                from ..config import settings
                threshold = settings.speaker_id.confidence_threshold
                match = speaker_id_service.identify(ctx.audio_bytes, threshold=threshold)
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
                    logger.info("Unknown speaker (best match: %.2f)", match.confidence)
                    # Reject unknown speakers if required
                    if settings.speaker_id.require_known_speaker:
                        logger.warning("Rejecting unknown speaker")
                        result.success = True
                        result.response_text = settings.speaker_id.unknown_speaker_response
                        # Generate TTS for rejection message
                        tts = self._get_tts()
                        if tts:
                            try:
                                audio = await tts.synthesize(result.response_text)
                                result.response_audio = audio
                            except Exception:
                                pass
                        return result
            except Exception as e:
                logger.warning("Speaker identification failed: %s", e)

        # Transcription
        trans_start = datetime.now()
        self._state_machine.transition(PipelineEvent.TRANSCRIPTION_COMPLETE)  # Move to PROCESSING

        stt = self._get_stt()
        if stt is None:
            return OrchestratorResult(success=False, error="STT service not available")

        # Check audio quality before STT to filter noise/echo
        audio_rms = calculate_audio_rms(ctx.audio_bytes)
        logger.info("Audio RMS: %.1f (threshold: %d)", audio_rms, self.config.min_audio_rms)
        if audio_rms < self.config.min_audio_rms:
            logger.info("Audio below threshold, skipping STT")
            result.success = True
            return result

        try:
            async with _cuda_lock:
                transcription = await stt.transcribe(ctx.audio_bytes)
            ctx.transcript = transcription.get("transcript", "")
            ctx.transcript_confidence = transcription.get("confidence", 0.0)
            result.transcript = ctx.transcript
            result.transcription_ms = (datetime.now() - trans_start).total_seconds() * 1000

            logger.info("Transcription: '%s' (%.0fms)", ctx.transcript, result.transcription_ms)

            if self._on_transcript:
                await self._on_transcript(ctx.transcript)

            # Skip processing if transcription is empty (no speech detected)
            if not ctx.transcript or not ctx.transcript.strip():
                logger.info("Empty transcription, skipping response generation")
                result.success = True
                return result

            # Keyword detection - only respond if keyword is in transcript
            # Skip keyword check if in follow-up mode (already in conversation)
            if self.config.keyword_enabled and not self.in_follow_up_mode:
                keyword_lower = self.config.keyword.lower()
                transcript_lower = ctx.transcript.lower()
                if keyword_lower not in transcript_lower:
                    logger.info(
                        "Keyword '%s' not found in transcript, ignoring",
                        self.config.keyword
                    )
                    result.success = True
                    return result
                logger.info("Keyword '%s' detected, processing command", self.config.keyword)
            elif self.in_follow_up_mode:
                logger.info("Follow-up mode active, skipping keyword check")

        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return OrchestratorResult(success=False, error=f"Transcription failed: {e}")

        # Delegate to agent if available
        if self._agent is not None:
            return await self._process_with_agent(ctx, result)

        # Intent parsing / LLM processing (legacy path when no agent)
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

        # Generate response text (includes LLM and memory timing)
        response_start = datetime.now()
        result.response_text = await self._generate_response_text(ctx, result)
        ctx.response_text = result.response_text
        logger.info("Response generated: intent=%s, action_results=%d, text='%s'",
                   ctx.intent.action if ctx.intent else None,
                   len(ctx.action_results),
                   result.response_text[:100] if result.response_text else None)

        # TTS (if available)
        tts = self._get_tts()
        logger.info("TTS check: tts=%s, response_text='%s'",
                   tts is not None, result.response_text[:50] if result.response_text else None)
        if tts is not None and result.response_text:
            try:
                tts_start = datetime.now()
                result.response_audio = await tts.synthesize(result.response_text)
                ctx.response_audio = result.response_audio
                result.tts_ms = (datetime.now() - tts_start).total_seconds() * 1000
                logger.info("TTS synthesized: %d bytes in %.0fms",
                           len(result.response_audio) if result.response_audio else 0, result.tts_ms)
            except Exception as e:
                logger.warning("TTS synthesis failed: %s", e)
        else:
            logger.warning("TTS skipped: tts=%s, has_text=%s", tts is not None, bool(result.response_text))

        self._state_machine.transition(PipelineEvent.RESPONSE_READY)

        if self._on_response:
            await self._on_response(result)

        # Enter follow-up mode for subsequent commands without wake word
        if self.config.follow_up_enabled and result.success:
            self._last_response_time = datetime.now()
            logger.info("Entering follow-up mode for %dms", self.config.follow_up_duration_ms)

        # Store conversation in memory
        storage_start = datetime.now()
        await self._store_conversation(ctx, result)
        result.storage_ms = (datetime.now() - storage_start).total_seconds() * 1000

        # Calculate total latency
        result.latency_ms = ctx.elapsed_ms()

        # Log latency summary
        self._log_latency_summary(result, pipeline_type="audio")

        return result

    async def _process_with_agent(
        self,
        ctx: PipelineContext,
        result: OrchestratorResult,
    ) -> OrchestratorResult:
        """
        Process utterance using the Agent for reasoning.

        The agent handles:
        - Intent parsing
        - Action execution
        - Response generation
        - Conversation storage

        This method handles:
        - Building agent context from pipeline context
        - TTS synthesis of agent response
        - Follow-up mode management
        """
        from ..agents import AgentContext

        logger.info("Delegating to agent for processing")

        # Build agent context from pipeline context
        agent_context = AgentContext(
            input_text=ctx.transcript,
            input_type="voice",
            session_id=self._session_id,
            speaker_id=ctx.speaker_id,
            speaker_confidence=ctx.speaker_confidence,
        )

        # Load conversation history into context
        if self._session_id:
            try:
                history = await self.load_session_history(limit=6)
                agent_context.conversation_history = [
                    {"role": turn.role, "content": turn.content}
                    for turn in history
                ]
            except Exception as e:
                logger.warning("Failed to load history for agent: %s", e)

        # Call agent
        try:
            agent_result = await self._agent.run(agent_context)

            # Transfer results to orchestrator result
            result.success = agent_result.success
            result.response_text = agent_result.response_text
            result.intent = agent_result.intent
            result.action_results = [
                r if isinstance(r, dict) else r.to_dict() if hasattr(r, "to_dict") else {"result": str(r)}
                for r in agent_result.action_results
            ]
            result.processing_ms = agent_result.think_ms
            result.action_ms = agent_result.act_ms
            result.llm_ms = agent_result.llm_ms
            result.memory_ms = agent_result.memory_ms
            result.tools_ms = agent_result.tools_ms
            result.storage_ms = agent_result.storage_ms
            result.error = agent_result.error

            # Update pipeline context
            ctx.intent = agent_result.intent
            ctx.response_text = agent_result.response_text

            logger.info(
                "Agent result: success=%s, action_type=%s, response=%.50s...",
                agent_result.success,
                agent_result.action_type,
                agent_result.response_text or "",
            )

        except Exception as e:
            logger.exception("Agent processing failed: %s", e)
            result.success = False
            result.error = str(e)
            result.response_text = f"I'm sorry, I encountered an error: {e}"

        # TTS (if available)
        tts = self._get_tts()
        if tts is not None and result.response_text:
            try:
                tts_start = datetime.now()
                result.response_audio = await tts.synthesize(result.response_text)
                ctx.response_audio = result.response_audio
                result.tts_ms = (datetime.now() - tts_start).total_seconds() * 1000
                logger.info(
                    "TTS synthesized: %d bytes in %.0fms",
                    len(result.response_audio) if result.response_audio else 0,
                    result.tts_ms,
                )
            except Exception as e:
                logger.warning("TTS synthesis failed: %s", e)

        self._state_machine.transition(PipelineEvent.RESPONSE_READY)

        if self._on_response:
            await self._on_response(result)

        # Enter follow-up mode for subsequent commands without wake word
        if self.config.follow_up_enabled and result.success:
            self._last_response_time = datetime.now()
            logger.info("Entering follow-up mode for %dms", self.config.follow_up_duration_ms)

        # Calculate total latency
        result.latency_ms = ctx.elapsed_ms()

        # Log latency summary
        self._log_latency_summary(result, pipeline_type="audio-agent")

        return result

    async def _store_conversation(
        self,
        ctx: PipelineContext,
        result: OrchestratorResult,
    ) -> None:
        """Store conversation turns in PostgreSQL (primary).

        GraphRAG ingestion is handled by nightly batch job, not real-time.
        This keeps daytime operations fast while still building long-term memory.
        """
        # Store in PostgreSQL (primary storage for session continuity)
        await self._store_to_postgresql(ctx, result)

        # NOTE: GraphRAG writes disabled for real-time operations.
        # Nightly batch job processes daily sessions → extracts facts → embeds → stores
        # See: atlas_brain/jobs/nightly_memory_sync.py

    async def _store_to_postgresql(
        self,
        ctx: PipelineContext,
        result: OrchestratorResult,
    ) -> None:
        """Store conversation turns in PostgreSQL for session persistence."""
        from ..storage import db_settings
        from ..storage.database import get_db_pool
        from ..storage.repositories.conversation import get_conversation_repo
        from uuid import UUID

        if not self._session_id or not db_settings.enabled:
            return

        pool = get_db_pool()
        if not pool.is_initialized:
            return

        try:
            conv_repo = get_conversation_repo()
            session_uuid = UUID(self._session_id)

            # Determine turn_type based on intent
            # Commands (device actions) are tracked separately from conversations
            turn_type = "command" if ctx.intent else "conversation"

            # Store user turn
            if ctx.transcript:
                intent_str = None
                if ctx.intent and hasattr(ctx.intent, "action"):
                    intent_str = ctx.intent.action
                await conv_repo.add_turn(
                    session_id=session_uuid,
                    role="user",
                    content=ctx.transcript,
                    speaker_id=ctx.speaker_id,
                    intent=intent_str,
                    turn_type=turn_type,
                )

            # Store assistant turn
            if result.response_text:
                await conv_repo.add_turn(
                    session_id=session_uuid,
                    role="assistant",
                    content=result.response_text,
                    turn_type=turn_type,
                )

            logger.debug("Persisted %s to PostgreSQL session %s", turn_type, self._session_id)

        except Exception as e:
            logger.warning("Failed to persist to PostgreSQL: %s", e)

    async def _store_to_graphrag(
        self,
        ctx: PipelineContext,
        result: OrchestratorResult,
    ) -> None:
        """Store conversation turns in GraphRAG for semantic memory."""
        # Skip device commands - only store actual conversations for semantic memory
        if ctx.intent:
            logger.debug("Skipping GraphRAG storage for device command")
            return

        try:
            memory_client = self._get_memory_client()
            # Use PostgreSQL session_id if available for correlation
            graphrag_session_id = self._session_id or str(id(ctx))

            if ctx.transcript:
                await memory_client.add_conversation_turn(
                    role="user",
                    content=ctx.transcript,
                    session_id=graphrag_session_id,
                    speaker_name=ctx.speaker_id,
                )

            if result.response_text:
                await memory_client.add_conversation_turn(
                    role="assistant",
                    content=result.response_text,
                    session_id=graphrag_session_id,
                )

            logger.debug("Stored conversation in GraphRAG")

        except Exception as e:
            logger.warning("Failed to store in GraphRAG: %s", e)

    async def load_session_history(
        self,
        limit: int = 10,
        include_commands: bool = False,
    ) -> list:
        """
        Load conversation history from PostgreSQL for this session.

        Args:
            limit: Maximum number of turns to load
            include_commands: If False, only load conversations (not device commands)

        Returns:
            List of ConversationTurn objects, oldest first
        """
        if not self._session_id:
            return []

        from ..storage import db_settings
        from ..storage.database import get_db_pool
        from ..storage.repositories.conversation import get_conversation_repo
        from uuid import UUID

        if not db_settings.enabled:
            return []

        pool = get_db_pool()
        if not pool.is_initialized:
            return []

        try:
            conv_repo = get_conversation_repo()
            session_uuid = UUID(self._session_id)
            # Filter to conversations only by default - commands don't provide context
            turn_type = None if include_commands else "conversation"
            history = await conv_repo.get_history(
                session_uuid,
                limit=limit,
                turn_type=turn_type,
            )
            logger.debug(
                "Loaded %d conversation turns from session %s",
                len(history), self._session_id
            )
            return history
        except Exception as e:
            logger.warning("Failed to load session history: %s", e)
            return []

    async def _generate_response_text(
        self,
        ctx: PipelineContext,
        result: Optional[OrchestratorResult] = None,
    ) -> str:
        """Generate a natural language response based on the action results."""
        if ctx.error:
            return f"Sorry, there was an error: {ctx.error}"

        # No intent means this is a conversational query - use LLM
        if not ctx.intent:
            return await self._generate_llm_response(ctx, result)

        # We had an intent - check if actions were executed
        if not ctx.action_results:
            return f"I understood '{ctx.intent.action}' but couldn't execute it."

        # Summarize action results
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

    async def _execute_tools_for_query(
        self,
        transcript: str,
        result: Optional[OrchestratorResult] = None,
    ) -> dict[str, str]:
        """Execute relevant tools based on query keywords."""
        from ..tools import tool_registry
        from datetime import datetime

        tool_results = {}
        transcript_lower = transcript.lower()
        tools_start = datetime.now()

        # Weather tool detection
        weather_keywords = ["weather", "temperature", "forecast", "rain", "sunny"]
        if any(kw in transcript_lower for kw in weather_keywords):
            try:
                loc_result = await tool_registry.execute("get_location", {})
                params = {}
                if loc_result.success and loc_result.data:
                    params["latitude"] = loc_result.data.get("latitude")
                    params["longitude"] = loc_result.data.get("longitude")
                weather_result = await tool_registry.execute("get_weather", params)
                if weather_result.success:
                    tool_results["weather"] = weather_result.message
                    logger.info("Weather tool: %s", weather_result.message)
            except Exception as e:
                logger.warning("Weather tool failed: %s", e)

        # Traffic tool detection
        traffic_keywords = ["traffic", "commute", "drive", "driving", "route"]
        if any(kw in transcript_lower for kw in traffic_keywords):
            try:
                loc_result = await tool_registry.execute("get_location", {})
                if loc_result.success and loc_result.data:
                    params = {
                        "latitude": loc_result.data.get("latitude"),
                        "longitude": loc_result.data.get("longitude"),
                    }
                    traffic_result = await tool_registry.execute("get_traffic", params)
                    if traffic_result.success:
                        tool_results["traffic"] = traffic_result.message
                        logger.info("Traffic tool: %s", traffic_result.message)
            except Exception as e:
                logger.warning("Traffic tool failed: %s", e)

        # Location tool detection
        location_keywords = ["where am i", "my location", "current location", "gps"]
        if any(kw in transcript_lower for kw in location_keywords):
            try:
                loc_result = await tool_registry.execute("get_location", {})
                if loc_result.success:
                    tool_results["location"] = loc_result.message
                    logger.info("Location tool: %s", loc_result.message)
            except Exception as e:
                logger.warning("Location tool failed: %s", e)

        # Time tool detection
        time_keywords = ["what time", "current time", "what day", "what date", "today"]
        if any(kw in transcript_lower for kw in time_keywords):
            try:
                time_result = await tool_registry.execute("get_time", {})
                if time_result.success:
                    tool_results["time"] = time_result.message
                    logger.info("Time tool: %s", time_result.message)
            except Exception as e:
                logger.warning("Time tool failed: %s", e)

        if result and tool_results:
            result.tools_ms = (datetime.now() - tools_start).total_seconds() * 1000

        return tool_results

    async def _generate_llm_response(
        self,
        ctx: PipelineContext,
        result: Optional[OrchestratorResult] = None,
    ) -> str:
        """Generate a conversational response using the LLM."""
        from ..config import settings
        from ..services.protocols import Message

        # Intelligent model routing when enabled
        if settings.routing.enabled:
            router = self._get_model_router()
            tier = router.select_tier(ctx.transcript, ctx)
            swapped = router.ensure_model_loaded(tier)
            if swapped:
                self._llm = None  # Clear cached LLM reference
                logger.info(
                    "Model routed: %s (complexity=%.2f)",
                    tier.name,
                    tier.score,
                )

        llm = self._get_llm()
        if llm is None:
            return f"I heard: {ctx.transcript}"

        # Retrieve context from memory if enabled (with timeout to avoid blocking)
        memory_context = ""
        if settings.memory.enabled and settings.memory.retrieve_context:
            try:
                memory_start = datetime.now()
                memory_client = self._get_memory_client()
                # 2 second timeout - fail fast if GraphRAG is slow
                memory_context = await asyncio.wait_for(
                    memory_client.get_context_for_query(
                        ctx.transcript,
                        num_results=settings.memory.context_results,
                    ),
                    timeout=2.0
                )
                memory_elapsed = (datetime.now() - memory_start).total_seconds() * 1000
                if result:
                    result.memory_ms = memory_elapsed
                if memory_context:
                    logger.info("Retrieved memory context: %d chars in %.0fms", len(memory_context), memory_elapsed)
            except asyncio.TimeoutError:
                logger.warning("Memory retrieval timed out (>2s), skipping")
            except Exception as e:
                logger.warning("Failed to retrieve memory context: %s", e)

        # Execute tools if query matches tool keywords
        tool_results = await self._execute_tools_for_query(ctx.transcript, result)
        tool_context = ""
        if tool_results:
            tool_context = "IMPORTANT - Use this exact information in your response:\n"
            for tool_name, tool_msg in tool_results.items():
                tool_context += f"- {tool_msg}\n"

        system_msg = "You are Atlas, a helpful home automation assistant. Respond briefly and naturally in 1-2 sentences."
        if tool_context:
            system_msg += " Use ONLY the provided information below. Do NOT make up facts."
        if ctx.speaker_id != "unknown":
            system_msg += f" The speaker is {ctx.speaker_id}."
        if memory_context:
            system_msg += f"\n\n{memory_context}"
        if tool_context:
            system_msg += f"\n\n{tool_context}"

        messages = [Message(role="system", content=system_msg)]

        # Include conversation history from session if available
        if self._session_id:
            try:
                history = await self.load_session_history(limit=6)  # Last 3 exchanges
                for turn in history:
                    messages.append(Message(role=turn.role, content=turn.content))
                if history:
                    logger.info("Added %d history turns to LLM context", len(history))
            except Exception as e:
                logger.warning("Failed to load history for LLM: %s", e)

        # Add current user message
        messages.append(Message(role="user", content=ctx.transcript))

        try:
            llm_start = datetime.now()
            async with _cuda_lock:
                llm_result = llm.chat(
                    messages=messages,
                    max_tokens=100,
                    temperature=0.7,
                )
            llm_elapsed = (datetime.now() - llm_start).total_seconds() * 1000
            if result:
                result.llm_ms = llm_elapsed
            response = llm_result.get("response", "").strip()
            if response:
                logger.info("LLM response: %d chars in %.0fms", len(response), llm_elapsed)
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

        # Delegate to agent if available
        if self._agent is not None:
            return await self._process_text_with_agent(text, result, start_time)

        # Intent parsing (legacy path when no agent)
        proc_start = datetime.now()
        intent_parser = self._get_intent_parser()
        if intent_parser:
            try:
                intent = await intent_parser.parse(text)
                ctx.intent = intent
                result.intent = intent
                if intent:
                    logger.info("Intent parsed: %s (%.0fms)", intent.action, (datetime.now() - proc_start).total_seconds() * 1000)
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

        # Generate response (includes LLM and memory timing)
        result.response_text = await self._generate_response_text(ctx, result)
        ctx.response_text = result.response_text

        # Store conversation in memory
        storage_start = datetime.now()
        await self._store_conversation(ctx, result)
        result.storage_ms = (datetime.now() - storage_start).total_seconds() * 1000

        # Calculate total latency
        result.latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Log latency summary
        self._log_latency_summary(result, pipeline_type="text")

        return result

    async def _process_text_with_agent(
        self,
        text: str,
        result: OrchestratorResult,
        start_time: datetime,
    ) -> OrchestratorResult:
        """
        Process text input using the Agent for reasoning.

        Similar to _process_with_agent but for text-only input.
        """
        from ..agents import AgentContext

        logger.info("Delegating text to agent for processing")

        # Build agent context
        agent_context = AgentContext(
            input_text=text,
            input_type="text",
            session_id=self._session_id,
        )

        # Load conversation history into context
        if self._session_id:
            try:
                history = await self.load_session_history(limit=6)
                agent_context.conversation_history = [
                    {"role": turn.role, "content": turn.content}
                    for turn in history
                ]
            except Exception as e:
                logger.warning("Failed to load history for agent: %s", e)

        # Call agent
        try:
            agent_result = await self._agent.run(agent_context)

            # Transfer results to orchestrator result
            result.success = agent_result.success
            result.response_text = agent_result.response_text
            result.intent = agent_result.intent
            result.action_results = [
                r if isinstance(r, dict) else r.to_dict() if hasattr(r, "to_dict") else {"result": str(r)}
                for r in agent_result.action_results
            ]
            result.processing_ms = agent_result.think_ms
            result.action_ms = agent_result.act_ms
            result.llm_ms = agent_result.llm_ms
            result.memory_ms = agent_result.memory_ms
            result.tools_ms = agent_result.tools_ms
            result.storage_ms = agent_result.storage_ms
            result.error = agent_result.error

            logger.info(
                "Agent result: success=%s, action_type=%s, response=%.50s...",
                agent_result.success,
                agent_result.action_type,
                agent_result.response_text or "",
            )

        except Exception as e:
            logger.exception("Agent processing failed: %s", e)
            result.success = False
            result.error = str(e)
            result.response_text = f"I'm sorry, I encountered an error: {e}"

        # Calculate total latency
        result.latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Log latency summary
        self._log_latency_summary(result, pipeline_type="text-agent")

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
