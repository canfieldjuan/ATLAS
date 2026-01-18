"""
Wake word detector using OpenWakeWord.

Fast, accurate wake word detection that runs continuously on audio
and triggers when "Atlas" or "Hey Atlas" is detected.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger("atlas.wakeword")


@dataclass
class WakeWordResult:
    """Result from wake word detection."""
    detected: bool
    wake_word: Optional[str] = None
    confidence: float = 0.0


class WakeWordDetector:
    """
    OpenWakeWord-based wake word detector.

    Processes audio chunks and detects wake words like "Atlas" or "Hey Atlas".
    Much faster and more accurate than using Whisper for keyword detection.

    Usage:
        detector = WakeWordDetector()
        detector.load()

        # Process audio chunks (16kHz, 16-bit, mono)
        result = detector.process_audio(audio_chunk)
        if result.detected:
            print(f"Wake word detected: {result.wake_word}")
    """

    # OpenWakeWord expects 80ms chunks at 16kHz = 1280 samples
    CHUNK_SAMPLES = 1280
    SAMPLE_RATE = 16000

    def __init__(
        self,
        threshold: float = 0.5,
        custom_model_path: Optional[Path] = None,
    ):
        """
        Initialize wake word detector.

        Args:
            threshold: Detection threshold (0.0-1.0), higher = stricter
            custom_model_path: Path to custom trained model for "Atlas"
        """
        self.threshold = threshold
        self.custom_model_path = custom_model_path
        self._model = None
        self._is_loaded = False

        # Buffer for accumulating audio until we have enough for a chunk
        self._audio_buffer = np.array([], dtype=np.int16)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        """Load the wake word model."""
        if self._is_loaded:
            logger.info("Wake word model already loaded")
            return

        try:
            from openwakeword.model import Model

            # Load models - use custom if available, otherwise use defaults
            if self.custom_model_path and self.custom_model_path.exists():
                logger.info("Loading custom wake word model: %s", self.custom_model_path)
                self._model = Model(
                    wakeword_model_paths=[str(self.custom_model_path)],
                )
            else:
                # Use default models (alexa, hey_mycroft, hey_jarvis, etc.)
                # hey_jarvis sounds similar to "hey atlas" as fallback
                logger.info("Loading default wake word models")
                self._model = Model()

            self._is_loaded = True
            logger.info("Wake word detector loaded (threshold=%.2f)", self.threshold)
            logger.info("Available wake words: %s", list(self._model.models.keys()))

        except Exception as e:
            logger.error("Failed to load wake word model: %s", e)
            raise

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            self._model = None
            self._is_loaded = False
            self._audio_buffer = np.array([], dtype=np.int16)
            logger.info("Wake word detector unloaded")

    def process_audio(self, audio_bytes: bytes) -> WakeWordResult:
        """
        Process audio chunk and check for wake word.

        Args:
            audio_bytes: Raw audio bytes (16kHz, 16-bit, mono)

        Returns:
            WakeWordResult with detection status
        """
        if not self._is_loaded:
            raise RuntimeError("Wake word detector not loaded")

        # Convert bytes to int16 samples
        audio_samples = np.frombuffer(audio_bytes, dtype=np.int16)

        # Debug: log audio stats periodically
        if not hasattr(self, '_audio_stats_counter'):
            self._audio_stats_counter = 0
        self._audio_stats_counter += 1
        if self._audio_stats_counter % 100 == 1:
            logger.info("Audio chunk: %d bytes -> %d samples, range=[%d, %d], rms=%.1f",
                       len(audio_bytes), len(audio_samples),
                       audio_samples.min() if len(audio_samples) > 0 else 0,
                       audio_samples.max() if len(audio_samples) > 0 else 0,
                       np.sqrt(np.mean(audio_samples.astype(np.float32)**2)) if len(audio_samples) > 0 else 0)

        # Add to buffer
        self._audio_buffer = np.concatenate([self._audio_buffer, audio_samples])

        # Process complete chunks
        result = WakeWordResult(detected=False)

        while len(self._audio_buffer) >= self.CHUNK_SAMPLES:
            chunk = self._audio_buffer[:self.CHUNK_SAMPLES]
            self._audio_buffer = self._audio_buffer[self.CHUNK_SAMPLES:]

            # OpenWakeWord expects int16 audio directly (not float32!)
            # Pass the raw int16 samples as per the official example
            prediction = self._model.predict(chunk)

            # Debug: log raw prediction output once
            if not hasattr(self, '_logged_raw'):
                self._logged_raw = True
                logger.info("Raw prediction type: %s, value: %s", type(prediction), prediction)
                logger.info("Audio chunk int16 stats: min=%d, max=%d, samples=%d",
                           chunk.min(), chunk.max(), len(chunk))

            # Debug: log scores periodically (every ~2 seconds = 25 chunks)
            if not hasattr(self, '_debug_counter'):
                self._debug_counter = 0
            self._debug_counter += 1
            if self._debug_counter % 25 == 0:
                # Log hey_jarvis score with high precision
                hj_score = prediction.get('hey_jarvis', 0)
                hm_score = prediction.get('hey_mycroft', 0)
                logger.info("Wake word: hey_jarvis=%.8f, hey_mycroft=%.8f", hj_score, hm_score)

            # Check each wake word model
            for wake_word, score in prediction.items():
                if score >= self.threshold:
                    result = WakeWordResult(
                        detected=True,
                        wake_word=wake_word,
                        confidence=float(score),
                    )
                    logger.info("Wake word detected: %s (confidence=%.3f)", wake_word, score)
                    # Reset buffer on detection to avoid re-triggering
                    self._audio_buffer = np.array([], dtype=np.int16)
                    return result

        return result

    def reset(self) -> None:
        """Reset the detector state (clear buffers)."""
        self._audio_buffer = np.array([], dtype=np.int16)
        if self._model is not None:
            # Reset model state if it has any
            self._model.reset()


# Global detector instance
_detector: Optional[WakeWordDetector] = None


def get_wakeword_detector(
    threshold: float = 0.5,
    custom_model_path: Optional[Path] = None,
) -> WakeWordDetector:
    """
    Get or create the global wake word detector.

    Args:
        threshold: Detection threshold
        custom_model_path: Path to custom model

    Returns:
        WakeWordDetector instance
    """
    global _detector
    if _detector is None:
        _detector = WakeWordDetector(
            threshold=threshold,
            custom_model_path=custom_model_path,
        )
    return _detector


def reset_wakeword_detector() -> None:
    """Reset the global wake word detector."""
    global _detector
    if _detector is not None:
        _detector.unload()
    _detector = None
