"""Silero VAD wrapper for Atlas voice pipeline.

Uses the official Silero VAD model via torch hub for accurate speech detection.
Provides a webrtcvad-compatible interface for drop-in replacement.

Usage:
    vad = SileroVAD(threshold=0.5)
    is_speech = vad.is_speech(frame_bytes, sample_rate=16000)
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("atlas.voice.vad.silero")

SILERO_WINDOW_SIZE = 512  # Required by Silero: 512 samples at 16kHz (32ms)


class SileroVAD:
    """Silero VAD with webrtcvad-compatible interface.

    Uses the official Silero OnnxWrapper from torch hub for reliable inference.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize Silero VAD.

        Args:
            threshold: Speech probability threshold (0.0-1.0).
        """
        self.threshold = threshold
        self._model = None
        logger.info("SileroVAD initialized with threshold=%.2f", threshold)

    def _load_model(self) -> None:
        """Load the Silero VAD model from torch hub."""
        if self._model is not None:
            return

        logger.info("Loading Silero VAD from torch hub...")
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
        )
        logger.info("Silero VAD loaded successfully")

    def is_speech(self, frame_bytes: bytes, sample_rate: int = 16000) -> bool:
        """Check if audio frame contains speech.

        Args:
            frame_bytes: Raw PCM audio bytes (int16, mono)
            sample_rate: Audio sample rate (16000 or 8000)

        Returns:
            True if speech probability exceeds threshold
        """
        if self._model is None:
            self._load_model()

        audio = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)

        # Process in 512-sample chunks, return True if any chunk is speech
        for i in range(0, len(tensor), SILERO_WINDOW_SIZE):
            chunk = tensor[i:i + SILERO_WINDOW_SIZE]
            if len(chunk) < SILERO_WINDOW_SIZE:
                chunk = torch.nn.functional.pad(chunk, (0, SILERO_WINDOW_SIZE - len(chunk)))

            prob = self._model(chunk, sample_rate).item()
            if prob > self.threshold:
                return True

        return False

    def get_speech_prob(self, frame_bytes: bytes, sample_rate: int = 16000) -> float:
        """Get max speech probability across all chunks in frame.

        Args:
            frame_bytes: Raw PCM audio bytes (int16, mono)
            sample_rate: Audio sample rate

        Returns:
            Maximum speech probability (0.0-1.0)
        """
        if self._model is None:
            self._load_model()

        audio = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)

        max_prob = 0.0
        for i in range(0, len(tensor), SILERO_WINDOW_SIZE):
            chunk = tensor[i:i + SILERO_WINDOW_SIZE]
            if len(chunk) < SILERO_WINDOW_SIZE:
                chunk = torch.nn.functional.pad(chunk, (0, SILERO_WINDOW_SIZE - len(chunk)))

            prob = self._model(chunk, sample_rate).item()
            if prob > max_prob:
                max_prob = prob

        return max_prob

    def reset_states(self) -> None:
        """Reset internal VAD states."""
        if self._model is not None:
            self._model.reset_states()
        logger.debug("Silero VAD states reset")

    def preload(self) -> None:
        """Preload model to avoid first-inference latency."""
        logger.info("Preloading Silero VAD model...")
        self._load_model()
        logger.info("Silero VAD model preloaded")
