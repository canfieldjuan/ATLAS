"""
Command segmenter for voice pipeline.

Tracks audio frames to decide when to finalize a command recording.
"""

import logging
from typing import List

logger = logging.getLogger("atlas.voice.segmenter")


class CommandSegmenter:
    """Tracks audio frames to decide when to finalize a command recording."""

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        silence_ms: int,
        hangover_ms: int,
        max_command_seconds: int,
        min_command_ms: int = 1500,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        frame_ms = 1000 * block_size / sample_rate
        self.silence_limit_frames = max(1, int(silence_ms / frame_ms))
        self.hangover_frames = max(0, int(hangover_ms / frame_ms))
        self.max_frames = int((sample_rate * max_command_seconds) / block_size)
        self.min_frames = max(1, int(min_command_ms / frame_ms))

        logger.info("=== CommandSegmenter Initialized ===")
        logger.info("  sample_rate=%d, block_size=%d", sample_rate, block_size)
        logger.info("  frame_ms=%.1f, silence_limit=%d frames", frame_ms, self.silence_limit_frames)
        logger.info("  hangover=%d frames, max_frames=%d", self.hangover_frames, self.max_frames)
        logger.info("  min_frames=%d (grace period before silence finalize)", self.min_frames)

        self.reset()

    def reset(self):
        """Reset the segmenter state."""
        self.frames: List[bytes] = []
        self.silence_counter = 0
        self.hangover_counter = 0

    def add_frame(self, frame_bytes: bytes, is_speech: bool) -> bool:
        """
        Add a frame and check if recording should finalize.

        Args:
            frame_bytes: Audio frame data
            is_speech: Whether VAD detected speech

        Returns:
            True if command should be finalized
        """
        self.frames.append(frame_bytes)
        if is_speech:
            self.silence_counter = 0
            self.hangover_counter = 0
        else:
            self.silence_counter += 1
        return self._should_finalize()

    def consume_audio(self) -> bytes:
        """Get collected audio and reset."""
        audio = b"".join(self.frames)
        return audio

    def _should_finalize(self) -> bool:
        """Check if we should finalize the recording."""
        # Don't finalize until we have minimum frames (grace period after wake word)
        if len(self.frames) < self.min_frames:
            return False
        if self.silence_counter >= self.silence_limit_frames:
            if self.hangover_frames > 0:
                self.hangover_counter += 1
                if self.hangover_counter < self.hangover_frames:
                    return False
            return True
        return len(self.frames) >= self.max_frames
