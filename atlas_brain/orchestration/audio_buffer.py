"""
Audio buffer with Voice Activity Detection (VAD).

Manages streaming audio input, detects speech segments,
and extracts complete utterances based on silence detection.
"""

import io
import logging
import struct
import wave
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("atlas.orchestration.audio")


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""

    # VAD sensitivity (0-3, higher = more aggressive filtering)
    aggressiveness: int = 2

    # Audio format
    sample_rate: int = 16000
    frame_duration_ms: int = 30  # 10, 20, or 30 ms for webrtcvad

    # Speech detection
    speech_threshold: float = 0.5  # Ratio of voiced frames to trigger speech
    silence_threshold: float = 0.3  # Ratio below which to end speech

    # Timing
    min_speech_duration_ms: int = 200  # Minimum speech to be valid
    max_speech_duration_ms: int = 30000  # Maximum recording length
    silence_duration_ms: int = 1500  # Silence duration to end utterance
    pre_speech_buffer_ms: int = 300  # Audio to keep before speech starts
    
    # Progressive streaming
    interim_interval_ms: int = 500  # Emit interim audio every N ms during speech


class AudioBuffer:
    """
    Manages streaming audio with VAD-based utterance detection.

    Accumulates audio frames, detects speech start/end,
    and extracts complete utterances for transcription.
    """

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self._vad = None
        self._init_vad()

        # Frame size in bytes (16-bit mono)
        self._frame_size = int(
            self.config.sample_rate * self.config.frame_duration_ms / 1000 * 2
        )

        # Buffers
        self._pre_buffer: deque[bytes] = deque(
            maxlen=self._frames_for_ms(self.config.pre_speech_buffer_ms)
        )
        self._speech_buffer: list[bytes] = []
        self._partial_frame: bytes = b""

        # State
        self._is_speaking = False
        self._voiced_frames = 0
        self._unvoiced_frames = 0
        self._total_speech_frames = 0
        
        # Progressive streaming state
        self._last_interim_frame = 0  # Track when last interim was emitted
        self._interim_interval_frames = self._frames_for_ms(self.config.interim_interval_ms)

    def _init_vad(self) -> None:
        """Initialize the VAD engine."""
        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(self.config.aggressiveness)
            logger.info("WebRTC VAD initialized (aggressiveness=%d)", self.config.aggressiveness)
        except ImportError:
            logger.warning("webrtcvad not installed, using energy-based VAD fallback")
            self._vad = None

    def _frames_for_ms(self, ms: int) -> int:
        """Calculate number of frames for a given duration."""
        return max(1, ms // self.config.frame_duration_ms)

    def reset(self) -> None:
        """Clear all buffers and reset state."""
        self._pre_buffer.clear()
        self._speech_buffer.clear()
        self._partial_frame = b""
        self._is_speaking = False
        self._voiced_frames = 0
        self._unvoiced_frames = 0
        self._total_speech_frames = 0
        self._last_interim_frame = 0

    @property
    def is_speaking(self) -> bool:
        """Whether speech is currently detected."""
        return self._is_speaking

    @property
    def speech_duration_ms(self) -> int:
        """Current speech duration in milliseconds."""
        return self._total_speech_frames * self.config.frame_duration_ms

    def add_audio(self, audio_bytes: bytes) -> Optional[str]:
        """
        Add audio data to the buffer.

        Returns:
            - "speech_start" when speech begins
            - "speech_interim" periodically during speech (for progressive transcription)
            - "speech_end" when speech ends (utterance ready)
            - "max_duration" when max recording length reached
            - None otherwise
        """
        # Combine with any partial frame from last call
        audio_bytes = self._partial_frame + audio_bytes
        self._partial_frame = b""

        # Process complete frames
        offset = 0
        event = None

        while offset + self._frame_size <= len(audio_bytes):
            frame = audio_bytes[offset : offset + self._frame_size]
            frame_event = self._process_frame(frame)
            if frame_event:
                event = frame_event
            offset += self._frame_size

        # Save incomplete frame for next call
        if offset < len(audio_bytes):
            self._partial_frame = audio_bytes[offset:]

        return event

    def _process_frame(self, frame: bytes) -> Optional[str]:
        """Process a single audio frame."""
        is_voiced = self._is_voiced(frame)

        if not self._is_speaking:
            # Not speaking yet - buffer and check for speech start
            self._pre_buffer.append(frame)

            if is_voiced:
                self._voiced_frames += 1
            else:
                self._voiced_frames = max(0, self._voiced_frames - 1)

            # Check if enough voiced frames to start speech
            threshold_frames = self._frames_for_ms(100)  # 100ms of voice to start
            if self._voiced_frames >= threshold_frames:
                self._start_speech()
                return "speech_start"

        else:
            # Currently speaking - record and check for speech end
            self._speech_buffer.append(frame)
            self._total_speech_frames += 1

            if is_voiced:
                self._unvoiced_frames = 0
            else:
                self._unvoiced_frames += 1

            # Check for max duration
            if self.speech_duration_ms >= self.config.max_speech_duration_ms:
                return "max_duration"

            # Check for interim event FIRST (progressive streaming)
            # This must come before silence check so interim events fire during speech
            frames_since_last_interim = self._total_speech_frames - self._last_interim_frame
            if frames_since_last_interim >= self._interim_interval_frames:
                self._last_interim_frame = self._total_speech_frames
                return "speech_interim"

            # Check for silence (speech end) AFTER interim
            silence_frames = self._frames_for_ms(self.config.silence_duration_ms)
            if self._unvoiced_frames >= silence_frames:
                return "speech_end"

        return None

    def _is_voiced(self, frame: bytes) -> bool:
        """Determine if a frame contains voice."""
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame, self.config.sample_rate)
            except Exception:
                pass

        # Fallback: energy-based detection
        return self._energy_vad(frame)

    def _energy_vad(self, frame: bytes) -> bool:
        """Simple energy-based voice activity detection."""
        # Convert bytes to 16-bit samples
        samples = struct.unpack(f"<{len(frame)//2}h", frame)

        # Calculate RMS energy
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5

        # Threshold (tune based on your mic)
        return rms > 500

    def _start_speech(self) -> None:
        """Mark speech as started, include pre-buffer."""
        self._is_speaking = True
        self._speech_buffer = list(self._pre_buffer)
        self._total_speech_frames = len(self._speech_buffer)
        self._unvoiced_frames = 0
        logger.debug("Speech started (pre-buffer: %d frames)", len(self._speech_buffer))

    def get_utterance(self) -> Optional[bytes]:
        """
        Get the complete utterance audio.

        Returns WAV-formatted audio bytes if speech was detected,
        None if no valid speech.
        """
        if not self._speech_buffer:
            return None

        # Check minimum duration
        if self.speech_duration_ms < self.config.min_speech_duration_ms:
            logger.debug(
                "Speech too short (%d ms < %d ms), discarding",
                self.speech_duration_ms,
                self.config.min_speech_duration_ms,
            )
            self.reset()
            return None

        # Combine all frames
        audio_data = b"".join(self._speech_buffer)

        # Convert to WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(audio_data)

        wav_bytes = wav_buffer.getvalue()

        logger.info(
            "Utterance captured: %d ms, %d bytes",
            self.speech_duration_ms,
            len(wav_bytes),
        )

        # Reset for next utterance
        self.reset()

        return wav_bytes

    def get_raw_audio(self) -> Optional[bytes]:
        """Get raw PCM audio without WAV header."""
        if not self._speech_buffer:
            return None

        if self.speech_duration_ms < self.config.min_speech_duration_ms:
            self.reset()
            return None

        audio_data = b"".join(self._speech_buffer)
        self.reset()
        return audio_data

    def get_interim_audio(self) -> Optional[bytes]:
        """
        Get current accumulated audio as WAV for interim transcription.
        
        Unlike get_utterance(), this does NOT reset the buffer - it returns
        a snapshot of the current speech for progressive transcription.
        
        Returns:
            WAV-formatted audio bytes of current speech, or None if insufficient data.
        """
        if not self._speech_buffer:
            return None
        
        # Require minimum duration for meaningful transcription
        min_interim_ms = 300  # 300ms minimum for interim
        if self.speech_duration_ms < min_interim_ms:
            return None
        
        # Combine all frames (snapshot, don't clear)
        audio_data = b"".join(self._speech_buffer)
        
        # Convert to WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(audio_data)
        
        wav_bytes = wav_buffer.getvalue()
        
        logger.debug(
            "Interim audio snapshot: %d ms, %d bytes",
            self.speech_duration_ms,
            len(wav_bytes),
        )
        
        return wav_bytes
