"""
Command segmenter for voice pipeline.

Tracks audio frames to decide when to finalize a command recording.
Supports an optional sliding window of speech probabilities to gate
the silence counter, preventing premature cutoff from VAD flicker
during continuous speech.
"""

import logging
import time
from collections import deque
from typing import List

logger = logging.getLogger("atlas.voice.segmenter")


class CommandSegmenter:
    """Tracks audio frames to decide when to finalize a command recording.

    When ``window_frames > 0``, a sliding window of VAD speech probabilities
    gates the silence counter.  Only when the speech ratio inside the window
    drops below ``silence_ratio`` does the existing consecutive-silence logic
    engage.  This makes the segmenter immune to brief VAD flicker (breaths,
    consonants) while remaining responsive when the user actually stops.
    """

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        silence_ms: int,
        hangover_ms: int,
        max_command_seconds: int,
        min_command_ms: int = 1500,
        min_speech_frames: int = 0,
        # Sliding window params (disabled by default for wake-word commands)
        speech_threshold: float = 0.5,
        window_frames: int = 0,
        silence_ratio: float = 0.15,
        asr_holdoff_ms: int = 0,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        frame_ms = 1000 * block_size / sample_rate
        self.silence_limit_frames = max(1, int(silence_ms / frame_ms))
        self.hangover_frames = max(0, int(hangover_ms / frame_ms))
        self.max_frames = int((sample_rate * max_command_seconds) / block_size)
        self.min_frames = max(1, int(min_command_ms / frame_ms))
        self._min_speech_frames = max(0, min_speech_frames)

        # Sliding window configuration
        self._speech_threshold = speech_threshold
        self._window_frames = window_frames
        self._silence_ratio = silence_ratio
        self._asr_holdoff_ms = asr_holdoff_ms

        logger.info("=== CommandSegmenter Initialized ===")
        logger.info("  sample_rate=%d, block_size=%d", sample_rate, block_size)
        logger.info("  frame_ms=%.1f, silence_limit=%d frames", frame_ms, self.silence_limit_frames)
        logger.info("  hangover=%d frames, max_frames=%d", self.hangover_frames, self.max_frames)
        logger.info("  min_frames=%d (grace period before silence finalize), min_speech_frames=%d",
                    self.min_frames, self._min_speech_frames)
        if window_frames > 0:
            logger.info("  sliding_window=%d frames, speech_thresh=%.2f, silence_ratio=%.2f, asr_holdoff=%dms",
                        window_frames, speech_threshold, silence_ratio, asr_holdoff_ms)

        self.reset()

    def update_thresholds(
        self,
        silence_ms: int | None = None,
        hangover_ms: int | None = None,
        max_command_seconds: int | None = None,
        window_frames: int | None = None,
        silence_ratio: float | None = None,
        asr_holdoff_ms: int | None = None,
        min_speech_frames: int | None = None,
    ) -> None:
        """Update segmenter thresholds at runtime (e.g. for conversation/workflow mode)."""
        frame_ms = 1000 * self.block_size / self.sample_rate
        if silence_ms is not None:
            self.silence_limit_frames = max(1, int(silence_ms / frame_ms))
        if hangover_ms is not None:
            self.hangover_frames = max(0, int(hangover_ms / frame_ms))
        if max_command_seconds is not None:
            self.max_frames = int((self.sample_rate * max_command_seconds) / self.block_size)
        if window_frames is not None:
            self._window_frames = window_frames
        if silence_ratio is not None:
            self._silence_ratio = silence_ratio
        if asr_holdoff_ms is not None:
            self._asr_holdoff_ms = asr_holdoff_ms
        if min_speech_frames is not None:
            self._min_speech_frames = max(0, min_speech_frames)
        logger.info(
            "Thresholds updated: silence=%d frames, hangover=%d frames, max=%d frames, window=%d, ratio=%.2f",
            self.silence_limit_frames, self.hangover_frames, self.max_frames,
            self._window_frames, self._silence_ratio,
        )

    def reset(self):
        """Reset the segmenter state."""
        self.frames: List[bytes] = []
        self.silence_counter = 0
        self.hangover_counter = 0
        self._speech_frame_count = 0
        # Sliding window state
        self._speech_window: deque[float] = deque(maxlen=max(1, self._window_frames) if self._window_frames > 0 else 1)
        self._asr_active_until: float = 0.0

    def add_frame(
        self,
        frame_bytes: bytes,
        speech_prob: float,
        asr_active: bool = False,
        is_preroll: bool = False,
    ) -> bool:
        """
        Add a frame and check if recording should finalize.

        Args:
            frame_bytes: Audio frame data
            speech_prob: VAD speech probability (0.0-1.0)
            asr_active: Whether ASR produced new words this frame
            is_preroll: Whether this is a buffered pre-roll frame

        Returns:
            True if command should be finalized
        """
        self.frames.append(frame_bytes)

        # Feed sliding window (if enabled)
        if self._window_frames > 0:
            self._speech_window.append(speech_prob)

        # Track ASR activity holdoff
        if asr_active and self._asr_holdoff_ms > 0:
            self._asr_active_until = time.monotonic() + self._asr_holdoff_ms / 1000.0

        # Consecutive silence counter (same as before, using threshold).
        # Trust ASR as speech evidence only when VAD shows SOME activity
        # (prob > 0.10). This catches quiet/clipped speech (VAD 0.1-0.5)
        # that ASR confirms, but ignores ASR oscillation during pure
        # silence (VAD 0.0-0.04) which would reset the counter forever.
        is_speech = speech_prob > self._speech_threshold or (asr_active and speech_prob > 0.10)
        if is_speech:
            self.silence_counter = 0
            self.hangover_counter = 0
            if not is_preroll:
                self._speech_frame_count += 1
        else:
            self.silence_counter += 1

        return self._should_finalize()

    def consume_audio(self) -> bytes:
        """Get collected audio bytes. Does NOT reset state - call reset() separately."""
        audio = b"".join(self.frames)
        return audio

    def _should_finalize(self) -> bool:
        """Check if we should finalize the recording."""
        # Safety cap: finalize at max_frames to prevent runaway recordings.
        # With generous max_command_seconds (30s+), this is a last resort;
        # silence detection should finalize naturally well before this.
        if len(self.frames) >= self.max_frames:
            logger.warning(
                "Hard safety cap reached (%d frames), force finalizing",
                len(self.frames),
            )
            return True

        # Don't finalize until we have minimum frames (grace period after wake word)
        if len(self.frames) < self.min_frames:
            return False

        # Don't finalize until we've seen real speech (prevents empty recordings)
        if self._min_speech_frames > 0 and self._speech_frame_count < self._min_speech_frames:
            return False

        # Sliding window gate (if enabled)
        if self._window_frames > 0 and len(self._speech_window) > 0:
            ratio = sum(1 for p in self._speech_window if p > self._speech_threshold) / len(self._speech_window)
            asr_still_active = time.monotonic() < self._asr_active_until

            if ratio >= self._silence_ratio or asr_still_active:
                # Window says still speaking -- reset silence counter
                self.silence_counter = 0
                self.hangover_counter = 0
                return False

        # Silence detected long enough -> finalize (VAD says user stopped talking)
        if self.silence_counter >= self.silence_limit_frames:
            if self.hangover_frames > 0:
                self.hangover_counter += 1
                if self.hangover_counter < self.hangover_frames:
                    return False
            return True
        return False
