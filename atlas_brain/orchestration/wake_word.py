"""
Wake word detection for Atlas.

Supports multiple backends:
- OpenWakeWord (recommended, open source)
- Porcupine (requires API key)
- Simple energy-based fallback
"""

import logging
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("atlas.orchestration.wake_word")


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""

    # Detection settings
    threshold: float = 0.5  # Confidence threshold (0-1)
    sample_rate: int = 16000

    # Model settings
    model_path: Optional[Path] = None  # Custom model path
    wake_words: list[str] = None  # Wake words to detect

    def __post_init__(self):
        if self.wake_words is None:
            self.wake_words = ["hey_jarvis", "alexa"]  # Default wake words


class WakeWordDetector(ABC):
    """Abstract base class for wake word detectors."""

    @abstractmethod
    def detect(self, audio_frame: bytes) -> tuple[bool, float, Optional[str]]:
        """
        Process an audio frame for wake word detection.

        Args:
            audio_frame: Raw PCM audio (16-bit, mono, 16kHz)

        Returns:
            Tuple of (detected, confidence, wake_word_name)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""
        pass


class OpenWakeWordDetector(WakeWordDetector):
    """
    Wake word detection using OpenWakeWord.

    OpenWakeWord is an open-source wake word detection library
    that runs efficiently on CPU.
    """

    def __init__(self, config: Optional[WakeWordConfig] = None):
        self.config = config or WakeWordConfig()
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the OpenWakeWord model."""
        try:
            from openwakeword.model import Model

            # Load pre-trained models for specified wake words
            self._model = Model(
                wakeword_models=self.config.wake_words,
                inference_framework="onnx",
            )
            logger.info(
                "OpenWakeWord initialized with wake words: %s",
                self.config.wake_words,
            )
        except ImportError:
            logger.warning(
                "openwakeword not installed. Install with: pip install openwakeword"
            )
            self._model = None
        except Exception as e:
            logger.error("Failed to initialize OpenWakeWord: %s", e)
            self._model = None

    def detect(self, audio_frame: bytes) -> tuple[bool, float, Optional[str]]:
        """Process audio frame for wake word."""
        if self._model is None:
            return False, 0.0, None

        try:
            import numpy as np

            # Convert bytes to numpy array (16-bit signed int to float)
            audio_array = np.frombuffer(audio_frame, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / 32768.0  # Normalize to [-1, 1]

            # Run prediction
            predictions = self._model.predict(audio_array)

            # Check each wake word
            for wake_word, scores in predictions.items():
                if scores:
                    score = scores[-1] if isinstance(scores, list) else scores
                    if score >= self.config.threshold:
                        logger.info(
                            "Wake word detected: %s (confidence: %.2f)",
                            wake_word,
                            score,
                        )
                        return True, float(score), wake_word

            return False, 0.0, None

        except Exception as e:
            logger.error("Wake word detection error: %s", e)
            return False, 0.0, None

    def reset(self) -> None:
        """Reset the model's internal state."""
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:
                pass


class EnergyWakeWordDetector(WakeWordDetector):
    """
    Simple energy-based "wake word" detection.

    Not a real wake word detector - just detects loud sounds.
    Use as fallback when no proper wake word library is available.
    """

    def __init__(self, config: Optional[WakeWordConfig] = None):
        self.config = config or WakeWordConfig()
        self._energy_threshold = 2000  # RMS threshold
        self._consecutive_frames = 0
        self._required_frames = 3  # Number of consecutive high-energy frames

    def detect(self, audio_frame: bytes) -> tuple[bool, float, Optional[str]]:
        """Detect based on audio energy."""
        # Calculate RMS energy
        samples = struct.unpack(f"<{len(audio_frame)//2}h", audio_frame)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5

        # Normalize to 0-1 range (roughly)
        confidence = min(1.0, rms / 10000)

        if rms > self._energy_threshold:
            self._consecutive_frames += 1
            if self._consecutive_frames >= self._required_frames:
                self._consecutive_frames = 0
                return True, confidence, "energy_trigger"
        else:
            self._consecutive_frames = 0

        return False, confidence, None

    def reset(self) -> None:
        """Reset state."""
        self._consecutive_frames = 0


class WakeWordManager:
    """
    Manages wake word detection with fallback support.

    Tries OpenWakeWord first, falls back to energy-based detection.
    """

    def __init__(self, config: Optional[WakeWordConfig] = None):
        self.config = config or WakeWordConfig()
        self._detector: Optional[WakeWordDetector] = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initialize the best available detector."""
        # Try OpenWakeWord first
        try:
            detector = OpenWakeWordDetector(self.config)
            if detector._model is not None:
                self._detector = detector
                logger.info("Using OpenWakeWord for wake word detection")
                return
        except Exception:
            pass

        # Fall back to energy-based
        logger.warning("Falling back to energy-based wake detection")
        self._detector = EnergyWakeWordDetector(self.config)

    @property
    def is_available(self) -> bool:
        """Whether wake word detection is available."""
        return self._detector is not None

    @property
    def detector_type(self) -> str:
        """Get the type of detector in use."""
        if isinstance(self._detector, OpenWakeWordDetector):
            return "openwakeword"
        elif isinstance(self._detector, EnergyWakeWordDetector):
            return "energy"
        return "none"

    def detect(self, audio_frame: bytes) -> tuple[bool, float, Optional[str]]:
        """
        Process audio frame for wake word detection.

        Args:
            audio_frame: Raw PCM audio (16-bit, mono, 16kHz)

        Returns:
            Tuple of (detected, confidence, wake_word_name)
        """
        if self._detector is None:
            return False, 0.0, None
        return self._detector.detect(audio_frame)

    def reset(self) -> None:
        """Reset detector state."""
        if self._detector is not None:
            self._detector.reset()
