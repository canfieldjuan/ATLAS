"""
TTS implementation using Kokoro.

Kokoro is an 82M parameter TTS model that produces very natural-sounding speech.
"""

import io
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..base import BaseModelService, InferenceTimer
from ..protocols import ModelInfo
from ..registry import register_tts


def _verbalize_phone_numbers(text: str) -> str:
    """
    Convert phone numbers to spoken form for clearer TTS output.

    Converts "555-4321" to "5 5 5, 4 3 2 1" with digit spacing.
    This helps STT accurately transcribe phone numbers.
    """
    import re

    digit_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    }

    def verbalize(match: re.Match) -> str:
        phone = match.group(0)
        digits = re.sub(r'[^\d]', '', phone)
        if len(digits) == 7:
            # Format: XXX-XXXX -> "X X X, X X X X"
            part1 = ' '.join(digit_words[d] for d in digits[:3])
            part2 = ' '.join(digit_words[d] for d in digits[3:])
            return f"{part1}, {part2}"
        elif len(digits) == 10:
            # Format: XXX-XXX-XXXX -> "X X X, X X X, X X X X"
            part1 = ' '.join(digit_words[d] for d in digits[:3])
            part2 = ' '.join(digit_words[d] for d in digits[3:6])
            part3 = ' '.join(digit_words[d] for d in digits[6:])
            return f"{part1}, {part2}, {part3}"
        return phone

    # Match phone patterns: 555-4321 (7 digits) or 555-123-4567 (10 digits)
    pattern = r'\b(\d{3}[-.\s]?\d{4}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b'
    return re.sub(pattern, verbalize, text)


@register_tts("kokoro")
class KokoroTTS(BaseModelService):
    """
    TTS implementation using Kokoro-82M.

    Produces natural-sounding speech with multiple voice options.
    """

    CAPABILITIES = ["tts"]

    # Available voices (American English)
    VOICES = {
        "af_heart": "Heart (female, warm)",
        "af_bella": "Bella (female, clear)",
        "af_sarah": "Sarah (female, professional)",
        "am_adam": "Adam (male, deep)",
        "am_michael": "Michael (male, natural)",
    }

    DEFAULT_VOICE = "af_heart"

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        lang_code: str = "a",  # 'a' for American English
        sample_rate: int = 24000,
        device: str | None = None,  # 'cuda', 'cpu', or None for auto
        **kwargs: Any,
    ):
        super().__init__(
            name="kokoro",
            model_id=f"kokoro-82m-{voice}",
            cache_path=Path("models/kokoro"),
        )
        self._voice = voice
        self._speed = speed
        self._lang_code = lang_code
        self._sample_rate = sample_rate
        self._device = device
        self._pipeline = None
        self._loaded = False

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            model_id=self.model_id,
            is_loaded=self.is_loaded,
            device="cuda" if self._use_cuda() else "cpu",
            capabilities=self.CAPABILITIES,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._pipeline is not None

    def _use_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def load(self) -> None:
        """Load the Kokoro pipeline."""
        if self._loaded:
            self.logger.info("Kokoro already loaded")
            return

        try:
            from kokoro import KPipeline

            device_str = self._device or ("cuda" if self._use_cuda() else "cpu")
            self.logger.info("Loading Kokoro pipeline (lang=%s, voice=%s, device=%s)...",
                           self._lang_code, self._voice, device_str)

            self._pipeline = KPipeline(lang_code=self._lang_code, device=self._device)
            self._loaded = True

            self.logger.info("Kokoro TTS loaded successfully on %s", device_str)

        except Exception as e:
            self.logger.error("Failed to load Kokoro: %s", e)
            raise

    def unload(self) -> None:
        """Unload the Kokoro pipeline."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        self._loaded = False
        self.logger.info("Kokoro TTS unloaded")

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice to use (optional, uses default if not specified)
            **kwargs: Additional options

        Returns:
            WAV audio bytes
        """
        import asyncio

        if not self.is_loaded:
            raise RuntimeError("Kokoro TTS not loaded. Call load() first.")

        voice = voice or self._voice

        # Run synthesis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._synthesize_sync(text, voice)
        )

    def _synthesize_sync(self, text: str, voice: str) -> bytes:
        """Synchronous synthesis implementation."""
        # Verbalize phone numbers for clearer speech
        text = _verbalize_phone_numbers(text)

        with InferenceTimer() as timer:
            # Generate audio using Kokoro
            generator = self._pipeline(text, voice=voice, speed=self._speed)

            # Collect all audio chunks
            audio_chunks = []
            for gs, ps, audio in generator:
                # Convert torch tensor to numpy if needed
                if hasattr(audio, 'numpy'):
                    audio = audio.numpy()
                audio_chunks.append(audio)

            # Concatenate all chunks
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
            else:
                full_audio = np.array([], dtype=np.float32)

        # Convert to WAV bytes
        wav_bytes = self._audio_to_wav(full_audio)

        metrics = self.gather_metrics(timer.duration)
        self.logger.info(
            "Synthesized %d samples in %.0fms (voice=%s)",
            len(full_audio),
            metrics.duration_ms,
            voice,
        )

        return wav_bytes

    def _audio_to_wav(self, audio: np.ndarray) -> bytes:
        """Convert audio array to WAV bytes."""
        # Normalize to int16
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = (audio * 32767).astype(np.int16)

        # Create WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(audio.tobytes())

        return buffer.getvalue()

    async def synthesize_async(
        self,
        text: str,
        voice: Optional[str] = None,
        **kwargs: Any,
    ) -> bytes:
        """Async version of synthesize (runs sync in thread pool)."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.synthesize(text, voice, **kwargs)
        )
