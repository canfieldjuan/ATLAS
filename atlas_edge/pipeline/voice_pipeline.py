"""
Edge Voice Pipeline.

Main voice-to-voice pipeline for edge devices that integrates:
- Wake word detection
- Voice Activity Detection (VAD)
- Speech-to-text (Parakeet)
- Intent classification and pattern parsing
- Local action dispatch or brain escalation
- Text-to-speech (Piper)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("atlas.edge.pipeline.voice")


@dataclass
class PipelineResult:
    """Result from voice pipeline processing."""

    success: bool
    transcript: str = ""
    response_text: str = ""
    action_type: str = "none"
    handled_locally: bool = False
    escalated: bool = False
    total_ms: float = 0.0
    stt_ms: float = 0.0
    intent_ms: float = 0.0
    action_ms: float = 0.0
    tts_ms: float = 0.0


class EdgeVoicePipeline:
    """
    Edge voice pipeline for Jetson devices.

    Handles the full voice-to-voice loop:
    1. Wake word detection (optional)
    2. VAD-based speech segmentation
    3. Speech-to-text transcription
    4. Intent classification
    5. Local handling or brain escalation
    6. Text-to-speech response
    """

    def __init__(
        self,
        enable_wake_word: bool = True,
        enable_vad: bool = True,
        sample_rate: int = 16000,
    ):
        """
        Initialize the edge voice pipeline.

        Args:
            enable_wake_word: Enable wake word detection
            enable_vad: Enable VAD for speech segmentation
            sample_rate: Audio sample rate
        """
        self._enable_wake_word = enable_wake_word
        self._enable_vad = enable_vad
        self._sample_rate = sample_rate

        # Component instances (lazy loaded)
        self._stt = None
        self._tts = None
        self._classifier = None
        self._pattern_parser = None
        self._dispatcher = None
        self._escalation = None
        self._vad = None
        self._skill_router = None
        self._skills_initialized = False
        self._skills_prefer_local = True

        # Callbacks
        self._on_wake_word: Optional[Callable[[], None]] = None
        self._on_transcript: Optional[Callable[[str], None]] = None
        self._on_response: Optional[Callable[[str], None]] = None

        # State
        self._is_listening = False
        self._is_processing = False

    def on_wake_word(self, callback: Callable[[], None]) -> None:
        """Set callback for wake word detection."""
        self._on_wake_word = callback

    def on_transcript(self, callback: Callable[[str], None]) -> None:
        """Set callback for transcript ready."""
        self._on_transcript = callback

    def on_response(self, callback: Callable[[str], None]) -> None:
        """Set callback for response ready."""
        self._on_response = callback

    async def _ensure_components(self) -> None:
        """Ensure all components are loaded."""
        from .stt import get_stt_service
        from .tts import get_tts_service
        from ..intent.classifier import get_intent_classifier
        from ..intent.patterns import get_pattern_parser
        from ..capabilities.dispatcher import get_dispatcher
        from ..brain.escalation import get_escalation

        if self._stt is None:
            self._stt = get_stt_service()
            await self._stt.load()

        if self._tts is None:
            self._tts = get_tts_service()
            await self._tts.load()

        if self._classifier is None:
            self._classifier = get_intent_classifier()
            await self._classifier.load()

        if self._pattern_parser is None:
            self._pattern_parser = get_pattern_parser()

        if self._dispatcher is None:
            self._dispatcher = get_dispatcher()

        if self._escalation is None:
            self._escalation = await get_escalation()

        if not self._skills_initialized:
            from ..config import settings as edge_settings
            if edge_settings.skills.enabled:
                from ..skills import get_skill_router
                self._skill_router = get_skill_router()
            self._skills_prefer_local = edge_settings.skills.prefer_local
            self._skills_initialized = True

        if self._enable_vad and self._vad is None:
            import webrtcvad

            self._vad = webrtcvad.Vad(1)  # Aggressiveness 1

    async def _try_skill(self, query: str):
        """Try matching a query against the skill router. Returns None if skills disabled."""
        if self._skill_router is None:
            return None
        return await self._skill_router.execute(query)

    async def process_audio(
        self,
        audio: np.ndarray,
        session_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process audio through the complete pipeline.

        Args:
            audio: Audio samples (int16 or float32)
            session_id: Session identifier
            speaker_id: Speaker identifier

        Returns:
            PipelineResult with transcript, response, and timing
        """
        start_time = time.perf_counter()
        await self._ensure_components()

        result = PipelineResult(success=False)

        # Step 1: Speech-to-text
        stt_start = time.perf_counter()
        stt_result = await self._stt.transcribe(audio, self._sample_rate)
        result.stt_ms = (time.perf_counter() - stt_start) * 1000
        result.transcript = stt_result.text

        if not stt_result.text.strip():
            logger.debug("Empty transcript, skipping")
            return result

        transcript = stt_result.text.strip()
        logger.info("Transcript: '%s'", transcript)

        if self._on_transcript:
            self._on_transcript(transcript)

        # Step 2: Intent classification
        intent_start = time.perf_counter()
        classification = await self._classifier.classify(transcript)
        result.intent_ms = (time.perf_counter() - intent_start) * 1000

        # Step 3: Handle locally or escalate
        action_start = time.perf_counter()

        if self._classifier.should_handle_locally(classification):
            # Try pattern parsing for device commands
            parsed = self._pattern_parser.parse(transcript)

            if parsed:
                # Local device command
                from ..intent.actions import ActionIntent

                intent = ActionIntent(
                    action=parsed.action,
                    target_type=parsed.target_type,
                    target_name=parsed.target_name,
                    parameters=parsed.parameters,
                    confidence=parsed.confidence,
                    raw_query=transcript,
                    source="pattern",
                )

                action_result = await self._dispatcher.execute(intent)

                result.response_text = action_result.message
                result.action_type = "device_command"
                result.handled_locally = True
                result.success = action_result.success

            else:
                # Try skills before escalating
                skill_result = await self._try_skill(transcript)
                if skill_result and skill_result.success:
                    result.response_text = skill_result.response_text
                    result.action_type = skill_result.action_type
                    result.handled_locally = True
                    result.success = True
                else:
                    result = await self._escalate_to_brain(
                        transcript, session_id, speaker_id, result
                    )

        else:
            # Try skills as fast path before brain round-trip
            if self._skills_prefer_local:
                skill_result = await self._try_skill(transcript)
                if skill_result and skill_result.success:
                    result.response_text = skill_result.response_text
                    result.action_type = skill_result.action_type
                    result.handled_locally = True
                    result.success = True
                else:
                    result = await self._escalate_to_brain(
                        transcript, session_id, speaker_id, result
                    )
            else:
                result = await self._escalate_to_brain(
                    transcript, session_id, speaker_id, result
                )

        result.action_ms = (time.perf_counter() - action_start) * 1000

        # Step 4: Text-to-speech
        if result.response_text:
            tts_start = time.perf_counter()
            await self._tts.speak(result.response_text)
            result.tts_ms = (time.perf_counter() - tts_start) * 1000

            if self._on_response:
                self._on_response(result.response_text)

        result.total_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Pipeline complete: %s (total=%.0fms, stt=%.0fms, intent=%.0fms, action=%.0fms, tts=%.0fms)",
            "local" if result.handled_locally else "escalated",
            result.total_ms,
            result.stt_ms,
            result.intent_ms,
            result.action_ms,
            result.tts_ms,
        )

        return result

    async def _escalate_to_brain(
        self,
        query: str,
        session_id: Optional[str],
        speaker_id: Optional[str],
        result: PipelineResult,
    ) -> PipelineResult:
        """Escalate query to brain server."""
        escalation_result = await self._escalation.escalate(
            query,
            session_id=session_id,
            speaker_id=speaker_id,
        )

        result.response_text = escalation_result.response_text
        result.action_type = escalation_result.action_type
        result.escalated = True
        result.handled_locally = False
        result.success = escalation_result.success

        if escalation_result.was_offline:
            # Try skills before falling back to static message
            skill_result = await self._try_skill(query)
            if skill_result and skill_result.success:
                result.response_text = skill_result.response_text
                result.action_type = skill_result.action_type
                result.handled_locally = True
                result.success = True
                return result
            logger.warning("Brain offline, using fallback response")

        return result

    async def process_text(
        self,
        text: str,
        session_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        speak_response: bool = True,
    ) -> PipelineResult:
        """
        Process text input (skip STT).

        Args:
            text: Input text
            session_id: Session identifier
            speaker_id: Speaker identifier
            speak_response: Whether to speak the response

        Returns:
            PipelineResult
        """
        start_time = time.perf_counter()
        await self._ensure_components()

        result = PipelineResult(success=False, transcript=text)

        if self._on_transcript:
            self._on_transcript(text)

        # Intent classification
        intent_start = time.perf_counter()
        classification = await self._classifier.classify(text)
        result.intent_ms = (time.perf_counter() - intent_start) * 1000

        # Handle locally or escalate
        action_start = time.perf_counter()

        if self._classifier.should_handle_locally(classification):
            parsed = self._pattern_parser.parse(text)

            if parsed:
                from ..intent.actions import ActionIntent

                intent = ActionIntent(
                    action=parsed.action,
                    target_type=parsed.target_type,
                    target_name=parsed.target_name,
                    parameters=parsed.parameters,
                    confidence=parsed.confidence,
                    raw_query=text,
                    source="pattern",
                )

                action_result = await self._dispatcher.execute(intent)

                result.response_text = action_result.message
                result.action_type = "device_command"
                result.handled_locally = True
                result.success = action_result.success
            else:
                # Try skills before escalating
                skill_result = await self._try_skill(text)
                if skill_result and skill_result.success:
                    result.response_text = skill_result.response_text
                    result.action_type = skill_result.action_type
                    result.handled_locally = True
                    result.success = True
                else:
                    result = await self._escalate_to_brain(
                        text, session_id, speaker_id, result
                    )
        else:
            # Try skills as fast path before brain round-trip
            if self._skills_prefer_local:
                skill_result = await self._try_skill(text)
                if skill_result and skill_result.success:
                    result.response_text = skill_result.response_text
                    result.action_type = skill_result.action_type
                    result.handled_locally = True
                    result.success = True
                else:
                    result = await self._escalate_to_brain(
                        text, session_id, speaker_id, result
                    )
            else:
                result = await self._escalate_to_brain(
                    text, session_id, speaker_id, result
                )

        result.action_ms = (time.perf_counter() - action_start) * 1000

        # TTS
        if speak_response and result.response_text:
            tts_start = time.perf_counter()
            await self._tts.speak(result.response_text)
            result.tts_ms = (time.perf_counter() - tts_start) * 1000

            if self._on_response:
                self._on_response(result.response_text)

        result.total_ms = (time.perf_counter() - start_time) * 1000
        return result


# Singleton instance
_pipeline: Optional[EdgeVoicePipeline] = None


def get_voice_pipeline() -> EdgeVoicePipeline:
    """Get or create global voice pipeline."""
    global _pipeline
    if _pipeline is None:
        from ..config import settings

        _pipeline = EdgeVoicePipeline(
            enable_wake_word=settings.wakeword.enabled,
            enable_vad=True,
            sample_rate=settings.stt.sample_rate,
        )
    return _pipeline
