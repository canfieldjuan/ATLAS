"""
Piper TTS Service for Edge Devices.

Uses Piper for fast, local text-to-speech synthesis
optimized for Jetson devices.
"""

import asyncio
import io
import logging
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("atlas.edge.pipeline.tts")


@dataclass
class SynthesisResult:
    """Result from text-to-speech synthesis."""

    audio: np.ndarray
    sample_rate: int = 22050
    duration_sec: float = 0.0
    processing_ms: float = 0.0


class PiperTTSService:
    """
    Piper TTS service for edge devices.

    Uses Piper for fast, local TTS synthesis with
    high-quality neural voices.
    """

    def __init__(
        self,
        voice: str = "en_US-ryan-medium",
        speed: float = 1.0,
        model_path: Optional[Path] = None,
    ):
        """
        Initialize Piper TTS service.

        Args:
            voice: Voice model name
            speed: Speech speed (1.0 = normal)
            model_path: Path to voice model (auto-downloaded if None)
        """
        self._voice = voice
        self._speed = speed
        self._model_path = model_path
        self._piper = None
        self._loaded = False
        self._sample_rate = 22050  # Piper default

    @property
    def is_loaded(self) -> bool:
        """Check if voice is loaded."""
        return self._loaded

    @property
    def sample_rate(self) -> int:
        """Get audio sample rate."""
        return self._sample_rate

    async def load(self) -> None:
        """Load the Piper voice model."""
        if self._loaded:
            return

        logger.info("Loading Piper TTS voice: %s", self._voice)
        start = time.time()

        loop = asyncio.get_event_loop()

        def _load_voice():
            try:
                from piper import PiperVoice

                # Download or load voice model
                if self._model_path:
                    voice = PiperVoice.load(str(self._model_path))
                else:
                    # Use piper's built-in download
                    from piper.download import ensure_voice_exists, find_voice

                    model_path, config_path = find_voice(self._voice)
                    if not model_path.exists():
                        ensure_voice_exists(
                            self._voice,
                            data_dir=Path.home() / ".local/share/piper",
                            download_dir=Path.home() / ".local/share/piper",
                        )
                        model_path, config_path = find_voice(self._voice)

                    voice = PiperVoice.load(str(model_path), config_path=str(config_path))

                return voice

            except ImportError:
                logger.error("Piper not installed. Install with: pip install piper-tts")
                raise
            except Exception as e:
                logger.error("Failed to load Piper voice: %s", e)
                raise

        self._piper = await loop.run_in_executor(None, _load_voice)
        self._loaded = True

        elapsed = time.time() - start
        logger.info("Piper TTS loaded in %.2fs", elapsed)

    def unload(self) -> None:
        """Unload the voice to free memory."""
        if self._piper is not None:
            del self._piper
            self._piper = None
            self._loaded = False
            logger.info("Piper TTS unloaded")

    async def synthesize(
        self,
        text: str,
    ) -> SynthesisResult:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize

        Returns:
            SynthesisResult with audio and metadata
        """
        if not self._loaded:
            await self.load()

        if not text.strip():
            return SynthesisResult(
                audio=np.array([], dtype=np.int16),
                sample_rate=self._sample_rate,
            )

        start_time = time.perf_counter()

        loop = asyncio.get_event_loop()

        def _synthesize():
            # Collect audio samples
            audio_buffer = io.BytesIO()

            with wave.open(audio_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self._sample_rate)

                for audio_bytes in self._piper.synthesize_stream_raw(
                    text,
                    length_scale=1.0 / self._speed,
                ):
                    wav_file.writeframes(audio_bytes)

            # Extract PCM
            audio_buffer.seek(0)
            with wave.open(audio_buffer, "rb") as wav_file:
                pcm_bytes = wav_file.readframes(wav_file.getnframes())

            return np.frombuffer(pcm_bytes, dtype=np.int16)

        try:
            audio = await loop.run_in_executor(None, _synthesize)
        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            audio = np.array([], dtype=np.int16)

        processing_ms = (time.perf_counter() - start_time) * 1000
        duration_sec = len(audio) / self._sample_rate if len(audio) > 0 else 0.0

        logger.info(
            "Synthesized '%s' in %.0fms (%.2fs audio)",
            text[:30] if text else "",
            processing_ms,
            duration_sec,
        )

        return SynthesisResult(
            audio=audio,
            sample_rate=self._sample_rate,
            duration_sec=duration_sec,
            processing_ms=processing_ms,
        )

    async def synthesize_to_wav(
        self,
        text: str,
    ) -> bytes:
        """
        Synthesize text to WAV bytes.

        Args:
            text: Text to synthesize

        Returns:
            WAV file bytes
        """
        result = await self.synthesize(text)

        # Convert to WAV
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(result.sample_rate)
            wav_file.writeframes(result.audio.tobytes())

        return buffer.getvalue()

    async def speak(
        self,
        text: str,
        play_callback: Optional[callable] = None,
    ) -> None:
        """
        Synthesize and play text.

        Args:
            text: Text to speak
            play_callback: Optional callback to play audio (audio, sample_rate)
        """
        result = await self.synthesize(text)

        if len(result.audio) == 0:
            return

        if play_callback:
            play_callback(result.audio, result.sample_rate)
        else:
            # Default playback using sounddevice
            try:
                import sounddevice as sd

                sd.play(result.audio, result.sample_rate)
                sd.wait()
            except Exception as e:
                logger.error("Audio playback failed: %s", e)


# Singleton instance
_tts_service: Optional[PiperTTSService] = None


def get_tts_service() -> PiperTTSService:
    """Get or create global TTS service."""
    global _tts_service
    if _tts_service is None:
        from ..config import settings

        _tts_service = PiperTTSService(
            voice=settings.tts.voice,
            speed=settings.tts.speed,
            model_path=settings.tts.model_path,
        )
    return _tts_service
