"""
Frame processor for voice pipeline.

Encapsulates wake word, VAD, and interrupt logic over incoming frames.
Supports optional streaming ASR for reduced latency.
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

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
        conversation_mode_enabled: bool = False,
        conversation_timeout_ms: int = 8000,
        conversation_speech_frames: int = 3,
        conversation_speech_tolerance: int = 2,
        conversation_rms_threshold: float = 0.002,
        on_conversation_timeout: Optional[Callable[[], None]] = None,
        # Voice filter settings
        rms_min_threshold: float = 0.008,
        rms_adaptive: bool = False,
        rms_above_ambient_factor: float = 3.0,
        turn_limit_enabled: bool = True,
        max_conversation_turns: int = 3,
        speaker_continuity_enabled: bool = False,
        speaker_continuity_threshold: float = 0.7,
        on_turn_limit_reached: Optional[Callable[[], None]] = None,
        on_asr_error: Optional[Callable[[], None]] = None,
        # Conversation-mode recording thresholds
        conversation_silence_ms: int = 500,
        conversation_hangover_ms: int = 300,
        conversation_max_command_seconds: int = 30,
        # Sliding window segmentation for conversation mode
        conversation_window_frames: int = 20,
        conversation_silence_ratio: float = 0.15,
        conversation_asr_holdoff_ms: int = 1000,
        wake_buffer_frames: int = 5,
        asr_quiet_limit: int = 5,
        on_early_silence: Optional[Callable[[str], None]] = None,
        conversation_early_silence_ms: int = 600,
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
        self._state_transitions = 0
        self._last_partial = ""
        # Stop feeding audio to ASR after this many frames with no new partial.
        # Prevents trailing silence from causing transcript oscillation.
        self._asr_quiet_frames = 0
        self._asr_quiet_limit = asr_quiet_limit

        # Conversation mode settings
        self.conversation_mode_enabled = conversation_mode_enabled
        self.conversation_timeout_ms = conversation_timeout_ms
        self.conversation_speech_frames = max(1, conversation_speech_frames)
        self.conversation_speech_tolerance = max(1, conversation_speech_tolerance)
        self.conversation_rms_threshold = conversation_rms_threshold
        self.on_conversation_timeout = on_conversation_timeout
        self._conversation_timer: Optional[threading.Timer] = None
        self._conversation_speech_counter = 0
        self._conversation_silence_counter = 0
        self._came_from_conversation = False  # Track if recording started from conversation mode
        self._state_lock = threading.Lock()  # Protects state transitions from timer thread
        # Buffer to capture frames before speech is confirmed (prevents first word cutoff)
        self._conversation_buffer: List[bytes] = []
        self._conversation_buffer_max = conversation_speech_frames + 5  # Extra margin
        # Pre-roll buffer for wake-word mode (captures audio around wake word)
        self._wake_buffer: List[bytes] = []
        self._wake_buffer_max = wake_buffer_frames

        # Voice filter: RMS settings
        self.rms_min_threshold = rms_min_threshold
        self.rms_adaptive = rms_adaptive
        self.rms_above_ambient_factor = rms_above_ambient_factor
        self._ambient_rms = 0.002  # Initial estimate of ambient noise floor
        self._ambient_rms_samples = 0  # Number of samples for running average

        # Voice filter: Turn limit settings
        self.turn_limit_enabled = turn_limit_enabled
        self.max_conversation_turns = max_conversation_turns
        self._turn_count = 0
        self.on_turn_limit_reached = on_turn_limit_reached
        self.on_asr_error = on_asr_error

        # Voice filter: Speaker continuity settings
        self.speaker_continuity_enabled = speaker_continuity_enabled
        self.speaker_continuity_threshold = speaker_continuity_threshold
        self._wake_speaker_embedding: Optional[np.ndarray] = None

        # Conversation recording thresholds (wider than wake-word commands)
        self._conversation_silence_ms = conversation_silence_ms
        self._conversation_hangover_ms = conversation_hangover_ms
        self._conversation_max_command_seconds = conversation_max_command_seconds
        # Sliding window params for conversation mode
        self._conversation_window_frames = conversation_window_frames
        self._conversation_silence_ratio = conversation_silence_ratio
        self._conversation_asr_holdoff_ms = conversation_asr_holdoff_ms
        # Store originals so we can restore after conversation recording
        self._orig_max_frames = segmenter.max_frames
        self._orig_silence_limit = segmenter.silence_limit_frames
        self._orig_hangover_frames = segmenter.hangover_frames
        self._orig_window_frames = segmenter._window_frames
        self._orig_silence_ratio = segmenter._silence_ratio
        self._orig_asr_holdoff_ms = segmenter._asr_holdoff_ms
        self._orig_min_speech_frames = segmenter._min_speech_frames

        # Early silence detection for conversation mode (overlapped prefill)
        # NOTE: We maintain our own silence counter (_recording_silence_frames)
        # because the segmenter's silence_counter is reset by the sliding window
        # gate in _should_finalize(), making it unreliable for early detection.
        frame_ms = 1000 * segmenter.block_size / segmenter.sample_rate  # ~80ms
        self._early_silence_frames = max(1, int(conversation_early_silence_ms / frame_ms))
        self._early_silence_fired = False
        self._recording_silence_frames = 0
        self.on_early_silence = on_early_silence

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
        logger.info("  conversation_mode=%s, timeout=%dms, speech_frames=%d, rms_thresh=%.3f",
                    conversation_mode_enabled, conversation_timeout_ms,
                    self.conversation_speech_frames, self.conversation_rms_threshold)
        logger.info("  voice_filter: rms_min=%.4f, rms_adaptive=%s, above_ambient=%.1fx",
                    rms_min_threshold, rms_adaptive, rms_above_ambient_factor)
        logger.info("  voice_filter: turn_limit=%s (max=%d), speaker_continuity=%s",
                    turn_limit_enabled, max_conversation_turns, speaker_continuity_enabled)
        logger.info("  early_silence: frames=%d (%.0fms), callback=%s",
                    self._early_silence_frames,
                    self._early_silence_frames * (1000 * segmenter.block_size / segmenter.sample_rate),
                    self.on_early_silence is not None)

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
        # Cancel conversation timer if active
        self._cancel_conversation_timer()
        self.state = "listening"
        self.interrupt_speech_counter = 0
        self._conversation_speech_counter = 0
        self._conversation_silence_counter = 0
        self._conversation_buffer.clear()
        self._wake_buffer.clear()
        # Reset turn count on full reset (wake word re-detection)
        self._turn_count = 0
        # Clear speaker embedding on reset
        self._wake_speaker_embedding = None
        # Reset early silence state
        self._early_silence_fired = False
        self._recording_silence_frames = 0

    def _connect_streaming_asr(self, context: str = "") -> bool:
        """Connect streaming ASR client if available.

        Args:
            context: Optional context string for logging (e.g., "for recording")

        Returns:
            True if connected, False otherwise
        """
        if self.streaming_asr_client is None:
            return False
        try:
            if self.streaming_asr_client.connect():
                self._streaming_active = True
                self._asr_quiet_frames = 0
                if context:
                    logger.info("Streaming ASR connected %s", context)
                return True
            else:
                self._streaming_active = False
                logger.warning("Streaming ASR connection failed, using batch mode")
                return False
        except Exception as e:
            logger.warning("Error connecting streaming ASR: %s", e)
            self._streaming_active = False
            return False

    def process_frame(
        self,
        frame_bytes: bytes,
        is_speaking: bool,
        current_allow_barge_in: bool,
        stop_playback: Callable[[], None],
        on_finalize: Callable[[bytes], None],
        on_streaming_finalize: Optional[Callable[[str, bytes], None]] = None,
    ):
        """
        Process an audio frame.

        Args:
            frame_bytes: Raw audio frame
            is_speaking: Whether TTS is currently playing
            current_allow_barge_in: Whether barge-in is allowed
            stop_playback: Callback to stop TTS
            on_finalize: Callback when command is ready (batch mode - receives audio bytes)
            on_streaming_finalize: Callback for streaming mode (receives transcript and audio)
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

        # Buffer recent frames while listening (pre-roll for wake word)
        if self.state == "listening":
            self._wake_buffer.append(frame_bytes)
            if len(self._wake_buffer) > self._wake_buffer_max:
                self._wake_buffer.pop(0)

        # Wake word detection
        if self.state == "listening" and detected:
            self._state_transitions += 1
            logger.info(
                "WAKE WORD DETECTED! score=%.4f threshold=%.2f transition=%d buffered=%d",
                max_score, self.wake_threshold, self._state_transitions,
                len(self._wake_buffer),
            )
            self.state = "recording"
            self._came_from_conversation = False
            self.segmenter.reset()

            # Connect streaming ASR if available
            self._connect_streaming_asr("for recording")

            # Feed buffered pre-roll frames (captures speech overlapping wake word)
            buffered = self._wake_buffer[:]
            self._wake_buffer.clear()
            for buf_frame in buffered:
                self.segmenter.add_frame(buf_frame, speech_prob=1.0, is_preroll=True)
                if self._streaming_active and self.streaming_asr_client is not None:
                    try:
                        self.streaming_asr_client.send_audio(buf_frame)
                    except Exception as e:
                        logger.warning("Error streaming wake pre-roll frame: %s", e)

            # Notify pipeline of wake word (plays confirmation sound)
            if self.on_wake_detected is not None:
                logger.info("Wake word callback fired")
                self.on_wake_detected()
            return

        # Conversation mode - accept speech without wake word
        if self.state == "conversing":
            rms = self._rms(frame_bytes)
            is_speech = self._is_speech(frame_bytes)
            # Require BOTH VAD and RMS to filter ambient conversations
            # VAD (Silero) distinguishes speech from noise
            # RMS uses a lower conversation-specific threshold (more permissive
            # than wake-word mode since the user just spoke to Atlas)
            rms_ok = rms > self.conversation_rms_threshold
            speech_detected = is_speech and rms_ok

            # Always buffer recent frames to prevent first word cutoff
            self._conversation_buffer.append(frame_bytes)
            if len(self._conversation_buffer) > self._conversation_buffer_max:
                self._conversation_buffer.pop(0)

            if speech_detected:
                # Log when potential speech detected for debugging
                logger.info(
                    "Conversation speech: vad=%s rms=%.4f (min=%.4f)",
                    is_speech, rms, self.conversation_rms_threshold,
                )
                self._conversation_speech_counter += 1
                self._conversation_silence_counter = 0  # Reset silence on speech
                # Reset timeout on any speech - ensures timeout is from last speech, not entry
                self._start_conversation_timer()
                if self._conversation_speech_counter >= self.conversation_speech_frames:
                    # Lock state transition to prevent race with timeout callback
                    with self._state_lock:
                        if self.state != "conversing":
                            # Timeout fired while we were processing, abort transition
                            return
                        self._cancel_conversation_timer()
                        self._state_transitions += 1
                        logger.info(
                            "SPEECH DETECTED in conversation mode after %d frames, "
                            "rms=%.4f, recording (transition=%d), buffered=%d",
                            self._conversation_speech_counter, rms,
                            self._state_transitions, len(self._conversation_buffer),
                        )
                        self._conversation_speech_counter = 0
                        self._conversation_silence_counter = 0
                        self.state = "recording"
                        self._came_from_conversation = True
                        self._early_silence_fired = False
                        self._recording_silence_frames = 0
                    self.segmenter.reset()
                    # Widen thresholds for conversation recording.
                    # min_speech_frames=0: VAD already confirmed speech above,
                    # so don't require additional non-preroll speech frames
                    # (preroll frames are is_preroll=True and don't count).
                    self.segmenter.update_thresholds(
                        max_command_seconds=self._conversation_max_command_seconds,
                        silence_ms=self._conversation_silence_ms,
                        hangover_ms=self._conversation_hangover_ms,
                        window_frames=self._conversation_window_frames,
                        silence_ratio=self._conversation_silence_ratio,
                        asr_holdoff_ms=self._conversation_asr_holdoff_ms,
                        min_speech_frames=0,
                    )

                    # Connect streaming ASR if available
                    self._connect_streaming_asr("for conversation follow-up")

                    # Include buffered frames (captures first word)
                    buffered = self._conversation_buffer[:]
                    self._conversation_buffer.clear()
                    for buf_frame in buffered:
                        self.segmenter.add_frame(buf_frame, speech_prob=1.0, is_preroll=True)
                        if self._streaming_active and self.streaming_asr_client is not None:
                            try:
                                self.streaming_asr_client.send_audio(buf_frame)
                            except Exception as e:
                                logger.warning("Error streaming buffered frame: %s", e)
                    return
            else:
                # Tolerate brief silences - only reset after N consecutive silence frames
                if self._conversation_speech_counter > 0:
                    self._conversation_silence_counter += 1
                    if self._conversation_silence_counter >= self.conversation_speech_tolerance:
                        self._conversation_speech_counter = 0
                        self._conversation_silence_counter = 0

            # Also allow wake word to re-engage during conversation
            if detected:
                with self._state_lock:
                    if self.state != "conversing":
                        # Timeout fired while we were processing, abort transition
                        return
                    self._cancel_conversation_timer()
                    self._state_transitions += 1
                    logger.info(
                        "WAKE WORD re-engaged during conversation (transition=%d)",
                        self._state_transitions,
                    )
                    self._conversation_speech_counter = 0
                    self._conversation_silence_counter = 0
                    self.state = "recording"
                    self._came_from_conversation = True
                    self._early_silence_fired = False
                    self._recording_silence_frames = 0
                self.segmenter.reset()
                # Widen thresholds for conversation recording.
                # min_speech_frames=0: wake word itself is evidence of speech
                # intent — don't gate on non-preroll speech frames (preroll
                # frames don't count, causing infinite hang on false wakes).
                self.segmenter.update_thresholds(
                    max_command_seconds=self._conversation_max_command_seconds,
                    silence_ms=self._conversation_silence_ms,
                    hangover_ms=self._conversation_hangover_ms,
                    window_frames=self._conversation_window_frames,
                    silence_ratio=self._conversation_silence_ratio,
                    asr_holdoff_ms=self._conversation_asr_holdoff_ms,
                    min_speech_frames=0,
                )
                self._connect_streaming_asr("for wake word re-engagement")

                # Include buffered frames (captures speech before wake word)
                buffered = self._conversation_buffer[:]
                self._conversation_buffer.clear()
                for buf_frame in buffered:
                    self.segmenter.add_frame(buf_frame, speech_prob=1.0, is_preroll=True)
                    if self._streaming_active and self.streaming_asr_client is not None:
                        try:
                            self.streaming_asr_client.send_audio(buf_frame)
                        except Exception as e:
                            logger.warning("Error streaming buffered frame: %s", e)
                return

            # No speech or wake word - stay in conversing, timer continues
            return

        # Recording state
        if self.state == "recording":
            # Stream audio to ASR if streaming mode is active.
            # Stop feeding frames after _asr_quiet_limit frames with no new
            # partial -- trailing silence causes transcript oscillation as the
            # model re-evaluates its hypothesis on an ever-growing silent tail.
            # Also stop when VAD is confident we're in silence (silence_counter
            # >= 3) to prevent HVAC/background noise from being transcribed.
            asr_active = False
            if self._streaming_active and self.streaming_asr_client is not None:
                vad_says_silence = self.segmenter.silence_counter >= 3
                if self._asr_quiet_frames < self._asr_quiet_limit and not vad_says_silence:
                    try:
                        partial = self.streaming_asr_client.send_audio(frame_bytes)
                        # Only log when partial changes to reduce noise.
                        # Strip trailing punctuation before comparing to avoid
                        # oscillation (e.g. "funny." vs "funny") keeping asr_active
                        if partial and partial.rstrip(".,!?") != self._last_partial.rstrip(".,!?"):
                            logger.info("Streaming ASR partial: %s", partial[:80] if partial else "")
                            asr_active = True
                            self._asr_quiet_frames = 0
                        elif not partial:
                            self._asr_quiet_frames += 1
                        if partial:
                            self._last_partial = partial
                    except Exception as e:
                        logger.warning("Error streaming audio: %s", e)

            speech_prob = self._get_speech_prob(frame_bytes)
            finalize = self.segmenter.add_frame(frame_bytes, speech_prob, asr_active)

            # Early silence detection -- start preparation during conversation silence.
            # Track raw silence ourselves (segmenter's counter is reset by sliding window).
            if self._came_from_conversation and self.on_early_silence is not None:
                if speech_prob <= self.segmenter._speech_threshold and not asr_active:
                    self._recording_silence_frames += 1
                else:
                    self._recording_silence_frames = 0

                if (not self._early_silence_fired
                        and self._recording_silence_frames >= self._early_silence_frames):
                    self._early_silence_fired = True
                    partial = self._last_partial
                    logger.info(
                        "Early silence at %d frames, triggering prep (partial: %s)",
                        self._recording_silence_frames,
                        partial[:50] if partial else "(none)",
                    )
                    self.on_early_silence(partial)

            # Log recording progress periodically
            if self.debug_logging and len(self.segmenter.frames) % 20 == 0:
                logger.info(
                    "Recording: frames=%d silence=%d/%d prob=%.2f asr_active=%s",
                    len(self.segmenter.frames),
                    self.segmenter.silence_counter,
                    self.segmenter.silence_limit_frames,
                    speech_prob,
                    asr_active,
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
                        t0 = time.perf_counter()
                        transcript = self.streaming_asr_client.finalize()
                        finalize_ms = (time.perf_counter() - t0) * 1000
                        logger.info("ASR finalize: %.0fms", finalize_ms)
                        self.streaming_asr_client.reset()
                        self._streaming_active = False
                        last_partial = self._last_partial
                        self._last_partial = ""  # Reset partial tracking
                        # Get audio bytes for speaker verification
                        audio_bytes = self.segmenter.consume_audio()
                        if transcript and on_streaming_finalize is not None:
                            logger.info("Streaming transcript: %s", transcript[:100])
                            on_streaming_finalize(transcript, audio_bytes)
                        elif last_partial and on_streaming_finalize is not None:
                            # Use last partial as fallback
                            logger.info("Using last partial as transcript: %s", last_partial[:100])
                            on_streaming_finalize(last_partial, audio_bytes)
                        elif transcript:
                            logger.warning("No streaming handler, transcript: %s", transcript[:50])
                        else:
                            # No transcript at all - fall back to batch ASR
                            # Don't continue conversation mode since we got no valid speech
                            logger.warning("Streaming ASR empty, falling back to batch")
                            self._came_from_conversation = False
                            if audio_bytes and on_finalize is not None:
                                on_finalize(audio_bytes)
                            elif self.on_asr_error is not None:
                                self.on_asr_error()
                    except Exception as e:
                        logger.error("Error finalizing streaming ASR: %s", e)
                        try:
                            self.streaming_asr_client.disconnect()
                        except Exception as disc_err:
                            logger.debug("Disconnect after error failed: %s", disc_err)
                        self._streaming_active = False
                        self._last_partial = ""
                        # Fallback to batch mode - don't continue conversation since streaming failed
                        self._came_from_conversation = False
                        audio_bytes = self.segmenter.consume_audio()
                        if len(audio_bytes) >= self.segmenter.block_size * 2:
                            logger.info("Falling back to batch ASR, audio=%d bytes", len(audio_bytes))
                            on_finalize(audio_bytes)
                        else:
                            logger.warning("Streaming ASR failed and audio too short (%d bytes), dropping", len(audio_bytes))
                            if self.on_asr_error is not None:
                                self.on_asr_error()
                else:
                    # Batch mode: send accumulated audio
                    audio_bytes = self.segmenter.consume_audio()
                    logger.info("Batch ASR: sending %d bytes", len(audio_bytes))
                    on_finalize(audio_bytes)

                self._state_transitions += 1
                self.segmenter.reset()

                # Restore original segmenter thresholds after conversation recording
                if self._came_from_conversation:
                    self.segmenter.max_frames = self._orig_max_frames
                    self.segmenter.silence_limit_frames = self._orig_silence_limit
                    self.segmenter.hangover_frames = self._orig_hangover_frames
                    self.segmenter._window_frames = self._orig_window_frames
                    self.segmenter._silence_ratio = self._orig_silence_ratio
                    self.segmenter._asr_holdoff_ms = self._orig_asr_holdoff_ms
                    self.segmenter._min_speech_frames = self._orig_min_speech_frames

                # Return to conversation mode if we came from it, otherwise go to listening
                if self._came_from_conversation and self.conversation_mode_enabled:
                    logger.info("State -> conversing (transition %d, returning to conversation)", self._state_transitions)
                    self.state = "conversing"
                    self._start_conversation_timer()
                else:
                    logger.info("State -> listening (transition %d)", self._state_transitions)
                    self.state = "listening"

                self._came_from_conversation = False

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

        Supports both webrtcvad and Silero VAD backends.
        """
        sample_rate = self.segmenter.sample_rate

        try:
            # Check if this is Silero VAD (has reset_states method)
            if hasattr(self.vad, 'reset_states'):
                # Silero VAD - pass entire frame, it handles chunking internally
                return self.vad.is_speech(frame_bytes, sample_rate)
            else:
                # webrtcvad - needs specific frame sizes (10ms, 20ms, or 30ms)
                # Split 80ms frame into 30ms + 30ms + 20ms chunks
                chunk_30ms = 960  # 30ms at 16kHz = 480 samples = 960 bytes
                chunk_20ms = 640  # 20ms at 16kHz = 320 samples = 640 bytes

                offset = 0
                while offset + chunk_30ms <= len(frame_bytes):
                    chunk = frame_bytes[offset:offset + chunk_30ms]
                    if self.vad.is_speech(chunk, sample_rate):
                        return True
                    offset += chunk_30ms

                remaining = len(frame_bytes) - offset
                if remaining >= chunk_20ms:
                    chunk = frame_bytes[offset:offset + chunk_20ms]
                    if self.vad.is_speech(chunk, sample_rate):
                        return True

                return False
        except Exception as e:
            logger.warning("VAD error: %s, falling back to RMS", e)
            rms = self._rms(frame_bytes)
            return rms > 0.01

    def _get_speech_prob(self, frame_bytes: bytes) -> float:
        """Get speech probability from VAD (0.0-1.0).

        Uses Silero VAD's get_speech_prob() if available, otherwise
        falls back to binary is_speech() returning 1.0 or 0.0.
        """
        if hasattr(self.vad, 'get_speech_prob'):
            return self.vad.get_speech_prob(frame_bytes, self.segmenter.sample_rate)
        return 1.0 if self._is_speech(frame_bytes) else 0.0

    @staticmethod
    def _rms(frame_bytes: bytes) -> float:
        """Calculate RMS energy of audio frame."""
        arr = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(arr * arr)))

    # --- Conversation mode methods ---

    def _start_conversation_timer(self) -> None:
        """Start or reset the conversation timeout timer."""
        self._cancel_conversation_timer()
        self._conversation_timer = threading.Timer(
            self.conversation_timeout_ms / 1000.0,
            self._on_conversation_timeout_internal
        )
        self._conversation_timer.daemon = True
        self._conversation_timer.start()
        logger.debug("Conversation timer started: %dms", self.conversation_timeout_ms)

    def _cancel_conversation_timer(self) -> None:
        """Cancel any active conversation timer."""
        if self._conversation_timer is not None:
            self._conversation_timer.cancel()
            self._conversation_timer = None

    def _on_conversation_timeout_internal(self) -> None:
        """Handle conversation timeout - return to listening state."""
        with self._state_lock:
            # Only transition if still in conversing state (avoid race with process_frame)
            if self.state != "conversing":
                logger.debug("Timeout fired but state is %s, ignoring", self.state)
                return
            logger.info("Conversation timeout, returning to listening state")
            self._conversation_timer = None
            self.state = "listening"
        if self.on_conversation_timeout is not None:
            try:
                self.on_conversation_timeout()
            except Exception as e:
                logger.warning("Error in conversation timeout callback: %s", e)

    def set_conversation_timeout(self, timeout_ms: int) -> None:
        """Dynamically update conversation timeout. Restarts timer if active."""
        self.conversation_timeout_ms = timeout_ms
        if self._conversation_timer is not None:
            self._start_conversation_timer()
        logger.info("Conversation timeout updated to %dms", timeout_ms)

    def pause_conversation_mode(self) -> None:
        """Pause conversation mode timer during TTS playback. Called by pipeline."""
        self._cancel_conversation_timer()
        logger.debug("Conversation timer paused for TTS playback")

    def enter_conversation_mode(self) -> None:
        """Enter or resume conversation mode after TTS completes. Called by pipeline."""
        if not self.conversation_mode_enabled:
            return
        with self._state_lock:
            if self.state == "conversing":
                # Already in conversation mode (follow-up response), just restart timer
                self._conversation_speech_counter = 0
                self._conversation_silence_counter = 0
                self._conversation_buffer.clear()
                self._start_conversation_timer()
                logger.info("Resumed conversation mode (timeout=%dms)", self.conversation_timeout_ms)
                return
            if self.state != "listening":
                logger.warning("enter_conversation_mode called in state=%s, ignoring", self.state)
                return
            self.state = "conversing"
            self._conversation_speech_counter = 0
            self._conversation_silence_counter = 0
            self._conversation_buffer.clear()
            self._start_conversation_timer()
            logger.info("Entered conversation mode (timeout=%dms, turn=%d/%d)",
                       self.conversation_timeout_ms, self._turn_count, self.max_conversation_turns)

    # --- Turn limit methods ---

    def increment_turn_count(self) -> bool:
        """Increment conversation turn count.

        Called after each successful response in conversation mode.

        Returns:
            True if turn limit reached (conversation should end)
        """
        self._turn_count += 1
        logger.info("Conversation turn %d/%d", self._turn_count, self.max_conversation_turns)

        if self.turn_limit_enabled and self._turn_count >= self.max_conversation_turns:
            logger.info("Turn limit reached (%d), ending conversation mode", self._turn_count)
            return True
        return False

    def reset_turn_count(self) -> None:
        """Reset turn count to 0. Called on wake word detection."""
        if self._turn_count > 0:
            logger.info("Resetting turn count from %d to 0", self._turn_count)
        self._turn_count = 0

    def get_turn_count(self) -> int:
        """Get current turn count."""
        return self._turn_count

    def exit_conversation_mode(self, reason: str = "manual") -> None:
        """Exit conversation mode and return to listening.

        Called when:
        - Turn limit reached
        - Intent confidence too low
        - User says goodbye phrase

        Args:
            reason: Reason for exiting (for logging)
        """
        with self._state_lock:
            if self.state != "conversing":
                return
            logger.info("Exiting conversation mode (reason=%s, turns=%d)",
                       reason, self._turn_count)
            self._cancel_conversation_timer()
            self.state = "listening"
            self._conversation_speech_counter = 0
            self._conversation_silence_counter = 0
            self._conversation_buffer.clear()

        if self.on_turn_limit_reached is not None and reason == "turn_limit":
            try:
                self.on_turn_limit_reached()
            except Exception as e:
                logger.warning("Error in turn limit callback: %s", e)

    # --- RMS filter methods ---

    def _passes_rms_filter(self, frame_bytes: bytes) -> bool:
        """Check if audio frame passes RMS energy filter.

        Returns True if the frame has sufficient energy to be considered
        potential speech (not ambient noise).

        Args:
            frame_bytes: Raw PCM audio bytes

        Returns:
            True if frame passes RMS filter
        """
        rms = self._rms(frame_bytes)

        # Update ambient noise estimate (exponential moving average of low-energy frames)
        if self.rms_adaptive and rms < self.rms_min_threshold:
            # This is likely ambient noise, update estimate
            alpha = 0.01  # Slow adaptation
            self._ambient_rms = (1 - alpha) * self._ambient_rms + alpha * rms
            self._ambient_rms_samples += 1

        # Check minimum threshold
        if rms < self.rms_min_threshold:
            return False

        # Check above-ambient factor (if adaptive mode enabled)
        if self.rms_adaptive and self._ambient_rms_samples > 100:
            required_rms = self._ambient_rms * self.rms_above_ambient_factor
            if rms < required_rms:
                return False

        return True

    def get_ambient_rms(self) -> float:
        """Get current ambient noise floor estimate."""
        return self._ambient_rms

    # --- Speaker continuity methods ---

    def set_wake_speaker_embedding(self, embedding: np.ndarray) -> None:
        """Store speaker embedding from wake word audio.

        Called after wake word detection with embedding extracted from
        the audio segment containing the wake word.

        Args:
            embedding: Speaker embedding vector (numpy array)
        """
        self._wake_speaker_embedding = embedding
        logger.info("Wake speaker embedding stored (shape=%s)", embedding.shape)

    def clear_wake_speaker_embedding(self) -> None:
        """Clear stored wake speaker embedding."""
        self._wake_speaker_embedding = None
        logger.debug("Wake speaker embedding cleared")

    def get_wake_speaker_embedding(self) -> Optional[np.ndarray]:
        """Get stored wake speaker embedding."""
        return self._wake_speaker_embedding

    def compare_speaker_embedding(self, embedding: np.ndarray) -> float:
        """Compare embedding against stored wake speaker embedding.

        Args:
            embedding: Speaker embedding to compare

        Returns:
            Similarity score (0.0-1.0), or 1.0 if no wake embedding stored
        """
        if self._wake_speaker_embedding is None:
            return 1.0  # No reference, allow all

        # Cosine similarity
        dot = np.dot(self._wake_speaker_embedding, embedding)
        norm1 = np.linalg.norm(self._wake_speaker_embedding)
        norm2 = np.linalg.norm(embedding)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = dot / (norm1 * norm2)
        return float(max(0.0, similarity))
