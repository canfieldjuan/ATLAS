"""
Frame processor for voice pipeline.

Encapsulates wake word, VAD, and interrupt logic over incoming frames.
"""

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np

from .segmenter import CommandSegmenter

logger = logging.getLogger("atlas.voice.frame_processor")


class FrameProcessor:
    """Encapsulates wake word, VAD, and interrupt logic over incoming frames."""

    def __init__(
        self,
        wake_predict: Callable[[np.ndarray], Dict[str, float]],
        wake_threshold: float,
        segmenter: CommandSegmenter,
        vad: Any,
        allow_wake_barge_in: bool,
        interrupt_predict: Optional[Callable[[np.ndarray], Dict[str, float]]] = None,
        interrupt_threshold: float = 0.5,
        interrupt_on_speech: bool = False,
        interrupt_speech_frames: int = 5,
        interrupt_rms_threshold: float = 0.05,
        audio_gain: float = 1.0,
        wake_reset: Optional[Callable[[], None]] = None,
    ):
        self.wake_predict = wake_predict
        self.wake_threshold = wake_threshold
        self.segmenter = segmenter
        self.vad = vad
        self.allow_wake_barge_in = allow_wake_barge_in
        self.interrupt_predict = interrupt_predict
        self.interrupt_threshold = interrupt_threshold
        self.interrupt_on_speech = interrupt_on_speech
        self.interrupt_speech_frames = max(1, interrupt_speech_frames)
        self.interrupt_rms_threshold = interrupt_rms_threshold
        self.audio_gain = audio_gain
        self.wake_reset = wake_reset

        self.state = "listening"
        self.interrupt_speech_counter = 0
        self._frame_count = 0

    def reset(self):
        """Reset processor to listening state."""
        self.segmenter.reset()
        if self.wake_reset is not None:
            self.wake_reset()
        self.state = "listening"
        self.interrupt_speech_counter = 0

    def process_frame(
        self,
        frame_bytes: bytes,
        is_speaking: bool,
        current_allow_barge_in: bool,
        stop_playback: Callable[[], None],
        on_finalize: Callable[[bytes], None],
    ):
        """
        Process an audio frame.

        Args:
            frame_bytes: Raw audio frame
            is_speaking: Whether TTS is currently playing
            current_allow_barge_in: Whether barge-in is allowed
            stop_playback: Callback to stop TTS
            on_finalize: Callback when command is ready
        """
        mono = np.frombuffer(frame_bytes, dtype=np.int16)
        audio_float = mono.astype(np.float32) / 32768.0

        # Apply audio gain for wake word detection
        if self.audio_gain != 1.0:
            audio_float = np.clip(audio_float * self.audio_gain, -1.0, 1.0)

        wake_scores = self.wake_predict(audio_float)
        detected = (
            any(score > self.wake_threshold for score in wake_scores.values())
            if wake_scores
            else False
        )

        self._frame_count += 1
        if self._frame_count == 1:
            logger.info(
                "FrameProcessor: First frame, state=%s, audio_gain=%.1f, wake_scores=%s",
                self.state,
                self.audio_gain,
                wake_scores,
            )
        if self._frame_count % 160 == 0:
            rms = self._rms(frame_bytes)
            amplified_rms = float(np.sqrt(np.mean(audio_float * audio_float)))
            max_score = max(wake_scores.values()) if wake_scores else 0.0
            logger.info(
                "FrameProcessor: frames=%d state=%s rms=%.4f "
                "amp_rms=%.4f wake_score=%.3f threshold=%.2f",
                self._frame_count,
                self.state,
                rms,
                amplified_rms,
                max_score,
                self.wake_threshold,
            )

        # Handle interrupts during TTS playback
        if is_speaking:
            if self._handle_speaking_interrupts(
                audio_float,
                frame_bytes,
                detected,
                current_allow_barge_in,
                stop_playback,
            ):
                return
            return

        # Wake word detection
        if self.state == "listening" and detected:
            logger.info("Wake word detected.")
            self.state = "recording"
            self.segmenter.reset()
            return

        # Recording state
        if self.state == "recording":
            finalize = self.segmenter.add_frame(
                frame_bytes, self._is_speech(frame_bytes)
            )
            if finalize:
                audio_bytes = self.segmenter.consume_audio()
                on_finalize(audio_bytes)
                self.state = "listening"
                if self.wake_reset is not None:
                    self.wake_reset()

    def _handle_speaking_interrupts(
        self,
        audio_float: np.ndarray,
        frame_bytes: bytes,
        wake_detected: bool,
        current_allow_barge_in: bool,
        stop_playback: Callable[[], None],
    ) -> bool:
        """Handle interrupt conditions during TTS playback."""
        # Check interrupt wake word
        if self.interrupt_predict is not None:
            intr_scores = self.interrupt_predict(audio_float)
            if intr_scores and any(
                val > self.interrupt_threshold for val in intr_scores.values()
            ):
                logger.info("Interrupt wake word detected during TTS.")
                stop_playback()
                self.reset()
                return True

        # Check normal wake word barge-in
        if self.allow_wake_barge_in and current_allow_barge_in and wake_detected:
            logger.info("Wake word detected during TTS; stopping playback.")
            stop_playback()
            self.reset()
            return True

        # Check speech-based interrupt
        if self.interrupt_on_speech:
            energy = self._rms(frame_bytes)
            vad_hit = self._is_speech(frame_bytes)
            if vad_hit and energy > self.interrupt_rms_threshold:
                self.interrupt_speech_counter += 1
                if self.interrupt_speech_counter >= self.interrupt_speech_frames:
                    logger.info(
                        "Speech detected during TTS; stopping playback "
                        "(energy=%.4f vad=%s).",
                        energy,
                        vad_hit,
                    )
                    stop_playback()
                    self.reset()
                    return True
            else:
                self.interrupt_speech_counter = 0

        return False

    def _is_speech(self, frame_bytes: bytes) -> bool:
        """Check if frame contains speech using VAD."""
        try:
            return self.vad.is_speech(frame_bytes, self.segmenter.sample_rate)
        except Exception:
            return True

    @staticmethod
    def _rms(frame_bytes: bytes) -> float:
        """Calculate RMS energy of audio frame."""
        arr = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(arr * arr)))
