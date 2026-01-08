"""
Faster-Whisper STT implementation.

A fast speech-to-text implementation using CTranslate2.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Optional

from faster_whisper import WhisperModel

from ..base import BaseModelService, InferenceTimer
from ..protocols import ModelInfo
from ..registry import register_stt


@register_stt("faster-whisper")
class FasterWhisperSTT(BaseModelService):
    """Faster-Whisper implementation of STTService."""

    CAPABILITIES = ["transcription"]

    def __init__(
        self,
        model_size: str = "small.en",
        cache_path: Optional[Path] = None,
    ):
        super().__init__(
            name="faster-whisper",
            model_id=model_size,
            cache_path=cache_path,
            log_file=Path("logs/atlas_stt.log"),
        )
        self.model_size = model_size

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            model_id=self.model_size,
            is_loaded=self.is_loaded,
            device=self.device,
            capabilities=self.CAPABILITIES,
        )

    def load(self) -> None:
        """Load the Whisper model."""
        if self._model is not None:
            self.logger.info("Model already loaded")
            return

        compute_type = "float16" if self.device == "cuda" else "int8"
        self.logger.info(
            "Loading STT model '%s' on %s (compute_type=%s)",
            self.model_size,
            self.device,
            compute_type,
        )

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=compute_type,
        )
        self.logger.info("STT model loaded successfully")

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            self.logger.info("Unloading model: %s", self.name)
            del self._model
            self._model = None
            self._clear_gpu_memory()
            self.logger.info("Model unloaded")

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
    ) -> dict[str, Any]:
        """Transcribe audio bytes to text."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not audio_bytes:
            self.logger.warning("Empty audio payload received")
            return {"error": "Empty audio payload."}

        self.logger.info("Transcription request: %s (%d bytes)", filename, len(audio_bytes))

        loop = asyncio.get_event_loop()

        def _transcribe() -> tuple[str, list[dict], Any]:
            suffix = Path(filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                segments, info = self._model.transcribe(tmp.name)
                text_segments = self._serialize_segments(segments)
                transcript = " ".join(seg["text"] for seg in text_segments).strip()
                return transcript, text_segments, info

        with InferenceTimer() as timer:
            transcript, text_segments, info = await loop.run_in_executor(None, _transcribe)

        metrics = self.gather_metrics(timer.duration)
        self.logger.info(
            "Transcription completed in %.2f ms on %s (language=%s)",
            metrics.duration_ms,
            metrics.device,
            getattr(info, "language", "unknown"),
        )

        return {
            "transcript": transcript,
            "segments": text_segments,
            "language": getattr(info, "language", "unknown"),
            "confidence": getattr(info, "language_probability", None),
            "metrics": metrics.to_dict(),
        }

    @staticmethod
    def _serialize_segments(segments) -> list[dict]:
        """Convert segment objects to serializable dicts."""
        serialized = []
        for segment in segments:
            serialized.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })
        return serialized
