"""
Parakeet STT Service for Edge Devices.

Uses NVIDIA Parakeet-TDT for fast, accurate speech recognition
optimized for Jetson devices.
"""

import asyncio
import io
import logging
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("atlas.edge.pipeline.stt")


@dataclass
class TranscriptionResult:
    """Result from speech transcription."""

    text: str
    confidence: float = 1.0
    duration_sec: float = 0.0
    processing_ms: float = 0.0


class ParakeetSTTService:
    """
    Parakeet STT service for edge devices.

    Uses NVIDIA NeMo's Parakeet-TDT model for fast transcription
    with support for both batch and streaming modes.
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        """
        Initialize Parakeet STT service.

        Args:
            model_name: NeMo model name or path
            device: Device for inference (cuda, cpu)
            compute_type: Compute type (float16, float32)
        """
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    async def load(self) -> None:
        """Load the Parakeet model."""
        if self._loaded:
            return

        logger.info("Loading Parakeet STT model: %s", self._model_name)
        start = time.time()

        loop = asyncio.get_event_loop()

        def _load_model():
            try:
                import nemo.collections.asr as nemo_asr

                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self._model_name
                )

                if self._device == "cuda":
                    model = model.cuda()

                if self._compute_type == "float16" and self._device == "cuda":
                    model = model.half()

                model.eval()
                return model

            except ImportError:
                logger.error(
                    "NeMo not installed. Install with: pip install nemo_toolkit[asr]"
                )
                raise
            except Exception as e:
                logger.error("Failed to load Parakeet model: %s", e)
                raise

        self._model = await loop.run_in_executor(None, _load_model)
        self._loaded = True

        elapsed = time.time() - start
        logger.info("Parakeet STT loaded in %.2fs on %s", elapsed, self._device)

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Parakeet STT unloaded")

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples as numpy array (float32 or int16)
            sample_rate: Sample rate of the audio

        Returns:
            TranscriptionResult with text and metadata
        """
        if not self._loaded:
            await self.load()

        start_time = time.perf_counter()

        # Normalize audio to float32 [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Calculate duration
        duration_sec = len(audio) / sample_rate

        loop = asyncio.get_event_loop()

        def _transcribe():
            import torch

            with torch.no_grad():
                # Create temporary file for NeMo
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                    # Write audio to temp file
                    import soundfile as sf

                    sf.write(f.name, audio, sample_rate)

                    # Transcribe
                    transcriptions = self._model.transcribe([f.name])
                    return transcriptions[0] if transcriptions else ""

        try:
            text = await loop.run_in_executor(None, _transcribe)
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            text = ""

        processing_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Transcribed %.2fs audio in %.0fms: '%s'",
            duration_sec,
            processing_ms,
            text[:50] if text else "(empty)",
        )

        return TranscriptionResult(
            text=text,
            confidence=1.0,
            duration_sec=duration_sec,
            processing_ms=processing_ms,
        )

    async def transcribe_wav(
        self,
        wav_bytes: bytes,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """
        Transcribe WAV audio bytes.

        Args:
            wav_bytes: WAV file bytes
            sample_rate: Expected sample rate

        Returns:
            TranscriptionResult with text and metadata
        """
        # Extract PCM from WAV
        try:
            buffer = io.BytesIO(wav_bytes)
            with wave.open(buffer, "rb") as wf:
                actual_rate = wf.getframerate()
                pcm_bytes = wf.readframes(wf.getnframes())
                audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        except Exception as e:
            logger.error("Failed to parse WAV: %s", e)
            return TranscriptionResult(text="", confidence=0.0)

        return await self.transcribe(audio, actual_rate)

    async def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """
        Transcribe raw PCM audio bytes.

        Args:
            pcm_bytes: Raw PCM bytes (int16, mono)
            sample_rate: Sample rate

        Returns:
            TranscriptionResult with text and metadata
        """
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        return await self.transcribe(audio, sample_rate)


# Singleton instance
_stt_service: Optional[ParakeetSTTService] = None


def get_stt_service() -> ParakeetSTTService:
    """Get or create global STT service."""
    global _stt_service
    if _stt_service is None:
        from ..config import settings

        _stt_service = ParakeetSTTService(
            model_name=settings.stt.model_name,
            device=settings.stt.device,
            compute_type=settings.stt.compute_type,
        )
    return _stt_service
