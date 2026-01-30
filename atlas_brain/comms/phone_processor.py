"""
Phone call audio processor.

Handles bidirectional audio streaming for phone calls:
- Receives mulaw 8kHz audio from SignalWire
- Converts to PCM 16kHz for STT
- Processes with ReceptionistAgent
- Generates TTS response
- Converts back to mulaw 8kHz for SignalWire
"""

import asyncio
import audioop
import base64
import logging
import struct
import wave
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("atlas.comms.phone_processor")


@dataclass
class PhoneCallState:
    """State for an active phone call."""

    call_sid: str
    from_number: str
    to_number: str
    context_id: str
    stream_sid: Optional[str] = None

    # Audio buffers
    audio_buffer: bytes = b""
    silence_frames: int = 0

    # Conversation state
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    is_speaking: bool = False
    last_speech_time: Optional[datetime] = None

    # Timing
    started_at: datetime = field(default_factory=datetime.now)


class PhoneCallProcessor:
    """
    Processes audio for phone calls using ReceptionistAgent.

    Audio flow:
    1. Receive mulaw 8kHz chunks from SignalWire WebSocket
    2. Buffer until silence detected (VAD)
    3. Convert to PCM 16kHz WAV for STT
    4. Transcribe with STT service
    5. Process with ReceptionistAgent
    6. Synthesize response with TTS
    7. Convert to mulaw 8kHz for SignalWire
    8. Send back via WebSocket
    """

    # Audio format constants
    MULAW_SAMPLE_RATE = 8000
    PCM_SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 500
    SILENCE_FRAMES_REQUIRED = 24  # ~300ms at 8kHz with 100 samples/frame

    def __init__(self, call_state: PhoneCallState, business_context: Any):
        """
        Initialize phone call processor.

        Args:
            call_state: PhoneCallState with call info
            business_context: BusinessContext for ReceptionistAgent
        """
        self._state = call_state
        self._business_context = business_context
        self._agent = None
        self._stt = None
        self._tts = None
        self._processing_lock = asyncio.Lock()

    def _get_agent(self):
        """Lazy load ReceptionistAgent via unified interface."""
        if self._agent is None:
            from ..agents.interface import get_agent
            self._agent = get_agent(
                agent_type="receptionist",
                session_id=self._state.call_sid,
                business_context=self._business_context,
            )
        return self._agent

    def _get_stt(self):
        """Lazy load STT service.

        NOTE: Voice services removed. Phone STT/TTS to be re-integrated
        with V3 voice pipeline.
        """
        logger.warning("STT service not available - voice services removed")
        return None

    def _get_tts(self):
        """Lazy load TTS service.

        NOTE: Voice services removed. Phone STT/TTS to be re-integrated
        with V3 voice pipeline.
        """
        logger.warning("TTS service not available - voice services removed")
        return None

    async def synthesize_greeting(self, text: str) -> Optional[str]:
        """
        Synthesize greeting text to mulaw audio for SignalWire.

        Args:
            text: Greeting text to synthesize

        Returns:
            Base64-encoded mulaw audio, or None on failure
        """
        try:
            tts = self._get_tts()
            if tts is None:
                logger.warning("TTS not available for greeting")
                return None

            # Synthesize with Atlas TTS (Kokoro)
            tts_audio = await tts.synthesize(text)
            if tts_audio is None:
                logger.warning("TTS returned no audio for greeting")
                return None

            # Use the same conversion as responses - handles any sample rate
            return self._convert_tts_to_mulaw(tts_audio)

        except Exception as e:
            logger.error("Failed to synthesize greeting: %s", e)
            return None

    def mulaw_to_pcm(self, mulaw_bytes: bytes) -> bytes:
        """Convert mulaw audio to PCM."""
        return audioop.ulaw2lin(mulaw_bytes, 2)

    def pcm_to_mulaw(self, pcm_bytes: bytes) -> bytes:
        """Convert PCM audio to mulaw."""
        return audioop.lin2ulaw(pcm_bytes, 2)

    def resample_8k_to_16k(self, audio_8k: bytes) -> bytes:
        """Resample 8kHz audio to 16kHz."""
        audio_16k, _ = audioop.ratecv(
            audio_8k, 2, 1,
            self.MULAW_SAMPLE_RATE,
            self.PCM_SAMPLE_RATE,
            None,
        )
        return audio_16k

    def resample_16k_to_8k(self, audio_16k: bytes) -> bytes:
        """Resample 16kHz audio to 8kHz."""
        audio_8k, _ = audioop.ratecv(
            audio_16k, 2, 1,
            self.PCM_SAMPLE_RATE,
            self.MULAW_SAMPLE_RATE,
            None,
        )
        return audio_8k

    def pcm_to_wav(self, pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
        """Wrap PCM bytes in WAV header."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm_bytes)
        return buffer.getvalue()

    def wav_to_pcm(self, wav_bytes: bytes) -> bytes:
        """Extract PCM from WAV file."""
        buffer = io.BytesIO(wav_bytes)
        with wave.open(buffer, "rb") as wav:
            return wav.readframes(wav.getnframes())

    def calculate_energy(self, pcm_bytes: bytes) -> int:
        """Calculate RMS energy of PCM audio."""
        if len(pcm_bytes) < 2:
            return 0
        return audioop.rms(pcm_bytes, 2)

    async def process_audio_chunk(self, mulaw_b64: str) -> Optional[str]:
        """
        Process incoming audio chunk.

        Args:
            mulaw_b64: Base64-encoded mulaw audio from SignalWire

        Returns:
            Base64-encoded mulaw audio response, or None if still listening
        """
        # Decode base64 mulaw
        mulaw_bytes = base64.b64decode(mulaw_b64)

        # Convert to PCM for analysis
        pcm_bytes = self.mulaw_to_pcm(mulaw_bytes)

        # Calculate energy for VAD
        energy = self.calculate_energy(pcm_bytes)

        if energy > self.SILENCE_THRESHOLD:
            # Speech detected
            self._state.is_speaking = True
            self._state.silence_frames = 0
            self._state.last_speech_time = datetime.now()

            # Resample to 16kHz and buffer
            pcm_16k = self.resample_8k_to_16k(pcm_bytes)
            self._state.audio_buffer += pcm_16k

        elif self._state.is_speaking:
            # Silence after speech
            self._state.silence_frames += 1

            if self._state.silence_frames >= self.SILENCE_FRAMES_REQUIRED:
                # End of speech - process the utterance
                self._state.is_speaking = False
                response = await self._process_utterance()
                self._state.audio_buffer = b""
                self._state.silence_frames = 0
                return response

        return None

    async def _process_utterance(self) -> Optional[str]:
        """Process buffered audio through STT -> Agent -> TTS."""
        async with self._processing_lock:
            if len(self._state.audio_buffer) < 3200:  # < 100ms at 16kHz
                return None

            try:
                # Convert buffer to WAV
                wav_bytes = self.pcm_to_wav(self._state.audio_buffer)

                # Transcribe
                stt = self._get_stt()
                if stt is None:
                    logger.warning("STT not available")
                    return None

                transcription = await stt.transcribe(wav_bytes)
                transcript = transcription.get("transcript", "").strip()

                if not transcript:
                    logger.debug("Empty transcript, skipping")
                    return None

                logger.info("Phone transcript: %s", transcript)

                # Process with agent via unified interface
                agent = self._get_agent()
                agent_result = await agent.process(
                    input_text=transcript,
                    input_type="voice",
                    session_id=self._state.call_sid,
                    runtime_context={
                        "conversation_history": self._state.conversation_history,
                    },
                )

                if not agent_result.response_text:
                    return None

                logger.info("Phone response: %s", agent_result.response_text)

                # Store in conversation history
                self._state.conversation_history.append({
                    "role": "user",
                    "content": transcript,
                })
                self._state.conversation_history.append({
                    "role": "assistant",
                    "content": agent_result.response_text,
                })

                # Synthesize TTS
                tts = self._get_tts()
                if tts is None:
                    logger.warning("TTS not available")
                    return None

                logger.info("Synthesizing TTS for response...")
                tts_audio = await tts.synthesize(agent_result.response_text)

                if tts_audio is None:
                    logger.warning("TTS returned None")
                    return None

                logger.info("TTS audio received: %d bytes", len(tts_audio))

                # Convert to phone format
                mulaw_result = self._convert_tts_to_mulaw(tts_audio)
                if mulaw_result:
                    logger.info("Converted to mulaw: %d bytes", len(mulaw_result))
                else:
                    logger.warning("Mulaw conversion returned empty")
                return mulaw_result

            except Exception as e:
                logger.exception("Error processing phone utterance: %s", e)
                return None

    def _convert_tts_to_mulaw(self, tts_wav: bytes) -> str:
        """Convert TTS WAV output to base64 mulaw for SignalWire."""
        try:
            # TTS outputs 24kHz WAV, need to convert to 8kHz mulaw
            buffer = io.BytesIO(tts_wav)
            with wave.open(buffer, "rb") as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                pcm_bytes = wav.readframes(wav.getnframes())
                logger.debug(
                    "TTS WAV: %dHz, %d channels, %d bytes, %d sample width",
                    sample_rate, channels, len(pcm_bytes), sample_width
                )

            # Resample to 8kHz
            pcm_8k, _ = audioop.ratecv(
                pcm_bytes, 2, 1,
                sample_rate,
                self.MULAW_SAMPLE_RATE,
                None,
            )
            logger.debug("Resampled to 8kHz: %d bytes", len(pcm_8k))

            # Convert to mulaw
            mulaw_bytes = self.pcm_to_mulaw(pcm_8k)
            logger.debug("Converted to mulaw: %d bytes", len(mulaw_bytes))

            # Base64 encode
            return base64.b64encode(mulaw_bytes).decode("ascii")

        except Exception as e:
            logger.exception("Error converting TTS to mulaw: %s", e)
            return ""


# Active call processors
_active_calls: dict[str, PhoneCallProcessor] = {}

# Greeting audio cache (context_id -> base64 mulaw audio)
_greeting_cache: dict[str, str] = {}


def get_call_processor(call_sid: str) -> Optional[PhoneCallProcessor]:
    """Get processor for an active call."""
    return _active_calls.get(call_sid)


def create_call_processor(
    call_sid: str,
    from_number: str,
    to_number: str,
    context_id: str,
    business_context: Any,
) -> PhoneCallProcessor:
    """Create a new call processor."""
    state = PhoneCallState(
        call_sid=call_sid,
        from_number=from_number,
        to_number=to_number,
        context_id=context_id,
    )
    processor = PhoneCallProcessor(state, business_context)
    _active_calls[call_sid] = processor
    logger.info("Created call processor for %s", call_sid)
    return processor


def remove_call_processor(call_sid: str) -> None:
    """Remove call processor when call ends."""
    if call_sid in _active_calls:
        del _active_calls[call_sid]
        logger.info("Removed call processor for %s", call_sid)


def get_cached_greeting(context_id: str) -> Optional[str]:
    """Get cached greeting audio for a context."""
    return _greeting_cache.get(context_id)


async def warm_greeting_cache(contexts: list[Any]) -> None:
    """
    Pre-synthesize greetings for all contexts at startup.

    NOTE: Voice services removed. Phone TTS to be re-integrated
    with V3 voice pipeline.

    Args:
        contexts: List of BusinessContext objects
    """
    logger.warning("TTS not available for greeting cache warmup - voice services removed")
    return

    # Create a temporary processor for audio conversion
    temp_state = PhoneCallState(
        call_sid="warmup",
        from_number="",
        to_number="",
        context_id="warmup",
    )
    converter = PhoneCallProcessor(temp_state, None)

    for ctx in contexts:
        if not ctx.greeting:
            continue

        try:
            logger.info("Pre-caching greeting for context: %s", ctx.id)
            tts_audio = await tts.synthesize(ctx.greeting)
            if tts_audio:
                mulaw_b64 = converter._convert_tts_to_mulaw(tts_audio)
                if mulaw_b64:
                    _greeting_cache[ctx.id] = mulaw_b64
                    logger.info(
                        "Cached greeting for %s (%d bytes)",
                        ctx.id, len(mulaw_b64)
                    )
        except Exception as e:
            logger.error("Failed to cache greeting for %s: %s", ctx.id, e)
