"""
Frame processor for voice pipeline.

Encapsulates wake word, VAD, and interrupt logic over incoming frames.
Supports optional streaming ASR for reduced latency.
"""

import logging
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from .segmenter import CommandSegmenter

logger = logging.getLogger("atlas.voice.frame_processor")


class FrameProcessor:
    """Encapsulates wake word, VAD, and interrupt logic over incoming frames.

    Supports two ASR modes:
    - Batch mode: Accumulates audio, sends complete recording on finalize
    - Streaming mode: Sends audio frames during recording, gets transcript on finalize
    """

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
        on_wake_detected: Optional[Callable[[], None]] = None,
        streaming_asr_client: Optional[Any] = None,
        debug_logging: bool = False,
        log_interval_frames: int = 160,
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
        self.on_wake_detected = on_wake_detected
        self.streaming_asr_client = streaming_asr_client
        self.debug_logging = debug_logging
        self.log_interval_frames = max(1, log_interval_frames)

        self.state = "listening"
        self.interrupt_speech_counter = 0
        self._frame_count = 0
        self._streaming_active = False
        self._max_wake_score = 0.0
        self._state_transitions = 0
        self._last_partial = ""

        # Warn if gain is too high (causes clipping that destroys wake word patterns)
        if audio_gain > 5.0:
            logger.warning(
                "audio_gain=%.1f is very high! Values >5.0 cause hard clipping "
                "that destroys wake word patterns. Consider using 1.0-3.0.",
                audio_gain
            )

        logger.info("=== FrameProcessor Initialized ===")
        logger.info("  wake_threshold=%.3f, audio_gain=%.2f", wake_threshold, audio_gain)
        logger.info("  allow_wake_barge_in=%s, interrupt_on_speech=%s",
                    allow_wake_barge_in, interrupt_on_speech)
        logger.info("  streaming_asr=%s", streaming_asr_client is not None)
        logger.info("  debug_logging=%s, log_interval=%d", debug_logging, log_interval_frames)

    def reset(self):
        """Reset processor to listening state."""
        logger.info("FrameProcessor.reset() called, previous state=%s", self.state)
        self.segmenter.reset()
        if self.wake_reset is not None:
            try:
                logger.info("Calling wake_reset() to reset wake word model")
                self.wake_reset()
                logger.info("Wake word model reset successful")
            except Exception as e:
                logger.error("Error resetting wake word model: %s", e, exc_info=True)
        # Disconnect streaming client if active
        if self._streaming_active and self.streaming_asr_client is not None:
            try:
                self.streaming_asr_client.disconnect()
            except Exception as e:
                logger.warning("Error disconnecting streaming ASR: %s", e)
        self._streaming_active = False
        self.state = "listening"
        self.interrupt_speech_counter = 0

    def process_frame(
        self,
        frame_bytes: bytes,
        is_speaking: bool,
        current_allow_barge_in: bool,
        stop_playback: Callable[[], None],
        on_finalize: Callable[[bytes], None],
        on_streaming_finalize: Optional[Callable[[str], None]] = None,
    ):
        """
        Process an audio frame.

        Args:
            frame_bytes: Raw audio frame
            is_speaking: Whether TTS is currently playing
            current_allow_barge_in: Whether barge-in is allowed
            stop_playback: Callback to stop TTS
            on_finalize: Callback when command is ready (batch mode - receives audio bytes)
            on_streaming_finalize: Callback for streaming mode (receives transcript directly)
        """
        # OpenWakeWord expects int16 audio directly (not float32)
        # See: https://github.com/dscripka/openWakeWord/blob/main/examples/detect_from_microphone.py
        mono_int16 = np.frombuffer(frame_bytes, dtype=np.int16)

        # For VAD and other processing, use float32 normalized audio
        audio_float = mono_int16.astype(np.float32) / 32768.0

        try:
            # Pass int16 directly to wake word model (critical for detection!)
            wake_scores = self.wake_predict(mono_int16)
            max_score = max(wake_scores.values()) if wake_scores else 0.0
            detected = max_score > self.wake_threshold if wake_scores else False
        except Exception as e:
            logger.error("Error predicting wake word: %s", e, exc_info=True)
            wake_scores = {}
            max_score = 0.0
            detected = False

        # Track max wake score seen for debugging
        if max_score > self._max_wake_score:
            self._max_wake_score = max_score

        self._frame_count += 1

        # First frame logging
        if self._frame_count == 1:
            logger.info(
                "FrameProcessor: First frame state=%s gain=%.1f scores=%s",
                self.state, self.audio_gain, wake_scores,
            )

        # Periodic logging based on config
        if self._frame_count % self.log_interval_frames == 0:
            rms = self._rms(frame_bytes)
            vad_speech = self._is_speech(frame_bytes)
            logger.info(
                "FrameProcessor: frames=%d state=%s rms=%.6f "
                "wake=%.4f/%.2f vad=%s",
                self._frame_count, self.state, rms,
                max_score, self.wake_threshold, vad_speech,
            )

        # Handle interrupts during TTS playback
        if is_speaking:
            if self._handle_speaking_interrupts(
                mono_int16,
                frame_bytes,
                detected,
                current_allow_barge_in,
                stop_playback,
            ):
                return
            return

        # Wake word detection
        if self.state == "listening" and detected:
            self._state_transitions += 1
            logger.info(
                "WAKE WORD DETECTED! score=%.4f threshold=%.2f transition=%d",
                max_score, self.wake_threshold, self._state_transitions,
            )
            self.state = "recording"
            self.segmenter.reset()

            # Connect streaming ASR if available
            if self.streaming_asr_client is not None:
                try:
                    if self.streaming_asr_client.connect():
                        self._streaming_active = True
                        logger.info("Streaming ASR connected for recording")
                    else:
                        self._streaming_active = False
                        logger.warning("Streaming ASR connection failed, using batch mode")
                except Exception as e:
                    logger.warning("Error connecting streaming ASR: %s", e)
                    self._streaming_active = False

            # Trigger LLM prefill in background while recording
            if self.on_wake_detected is not None:
                logger.info("Triggering LLM prefill callback")
                self.on_wake_detected()
            return

        # Recording state
        if self.state == "recording":
            # Stream audio to ASR if streaming mode is active
            if self._streaming_active and self.streaming_asr_client is not None:
                try:
                    partial = self.streaming_asr_client.send_audio(frame_bytes)
                    # Only log when partial changes to reduce noise
                    if partial and partial != self._last_partial:
                        logger.info("Streaming ASR partial: %s", partial[:80] if partial else "")
                        self._last_partial = partial
                except Exception as e:
                    logger.warning("Error streaming audio: %s", e)

            is_speech = self._is_speech(frame_bytes)
            finalize = self.segmenter.add_frame(frame_bytes, is_speech)

            # Log recording progress periodically
            if self.debug_logging and len(self.segmenter.frames) % 20 == 0:
                logger.info(
                    "Recording: frames=%d silence=%d/%d speech=%s",
                    len(self.segmenter.frames),
                    self.segmenter.silence_counter,
                    self.segmenter.silence_limit_frames,
                    is_speech,
                )

            if finalize:
                audio_len_ms = (len(self.segmenter.frames) * self.segmenter.block_size
                                * 1000 // self.segmenter.sample_rate)
                logger.info(
                    "RECORDING FINALIZED: frames=%d duration=%dms streaming=%s",
                    len(self.segmenter.frames), audio_len_ms, self._streaming_active,
                )

                if self._streaming_active and self.streaming_asr_client is not None:
                    # Streaming mode: get final transcript directly
                    try:
                        logger.info("Finalizing streaming ASR...")
                        transcript = self.streaming_asr_client.finalize()
                        self.streaming_asr_client.disconnect()
                        self._streaming_active = False
                        self._last_partial = ""  # Reset partial tracking
                        if transcript and on_streaming_finalize is not None:
                            logger.info("Streaming transcript: %s", transcript[:100])
                            on_streaming_finalize(transcript)
                        elif transcript:
                            logger.warning("No streaming handler, transcript: %s", transcript[:50])
                    except Exception as e:
                        logger.error("Error finalizing streaming ASR: %s", e)
                        try:
                            self.streaming_asr_client.disconnect()
                        except Exception:
                            pass
                        self._streaming_active = False
                        # Fallback to batch mode
                        audio_bytes = self.segmenter.consume_audio()
                        logger.info("Falling back to batch ASR, audio=%d bytes", len(audio_bytes))
                        on_finalize(audio_bytes)
                else:
                    # Batch mode: send accumulated audio
                    audio_bytes = self.segmenter.consume_audio()
                    logger.info("Batch ASR: sending %d bytes", len(audio_bytes))
                    on_finalize(audio_bytes)

                self._state_transitions += 1
                logger.info("State -> listening (transition %d)", self._state_transitions)
                self.segmenter.reset()
                self.state = "listening"
                if self.wake_reset is not None:
                    try:
                        logger.info("Resetting wake word model after recording")
                        self.wake_reset()
                        logger.info("Wake word model reset complete")
                    except Exception as e:
                        logger.error("Error resetting wake word model: %s", e, exc_info=True)

    def _handle_speaking_interrupts(
        self,
        audio_int16: np.ndarray,
        frame_bytes: bytes,
        wake_detected: bool,
        current_allow_barge_in: bool,
        stop_playback: Callable[[], None],
    ) -> bool:
        """Handle interrupt conditions during TTS playback."""
        # Check interrupt wake word (expects int16 like main wake model)
        if self.interrupt_predict is not None:
            intr_scores = self.interrupt_predict(audio_int16)
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
        """Check if frame contains speech using VAD.
        
        webrtcvad only supports 10ms, 20ms, or 30ms frames at 16kHz.
        Our 80ms frames (1280 samples) must be split into 30ms chunks (480 samples).
        Returns True if ANY chunk contains speech.
        """
        sample_rate = self.segmenter.sample_rate
        # 30ms at 16kHz = 480 samples = 960 bytes
        chunk_bytes = 960
        
        try:
            # Process 80ms frame in 30ms chunks (we get 2 full chunks + remainder)
            for i in range(0, len(frame_bytes) - chunk_bytes + 1, chunk_bytes):
                chunk = frame_bytes[i:i + chunk_bytes]
                if self.vad.is_speech(chunk, sample_rate):
                    return True
            return False
        except Exception:
            # Fallback: use RMS-based detection if VAD fails
            rms = self._rms(frame_bytes)
            return rms > 0.01

    @staticmethod
    def _rms(frame_bytes: bytes) -> float:
        """Calculate RMS energy of audio frame."""
        arr = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(arr * arr)))
