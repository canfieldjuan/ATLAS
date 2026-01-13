"""
Nemotron Speech Streaming STT implementation.

NVIDIA's cache-aware streaming ASR model (600M params) optimized for:
- Real-time voice assistant applications
- Native punctuation and capitalization output
- Low-latency streaming inference
"""

import asyncio
import io
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..base import BaseModelService, InferenceTimer
from ..protocols import ModelInfo
from ..registry import register_stt

logger = logging.getLogger("atlas.stt.nemotron")

# Model constants
MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
SAMPLE_RATE = 16000  # Model expects 16kHz audio


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample audio to target sample rate using scipy."""
    if orig_sr == target_sr:
        return audio

    from scipy import signal

    # Calculate new length
    new_length = int(len(audio) * target_sr / orig_sr)
    resampled = signal.resample(audio, new_length)
    return resampled.astype(np.float32)


def _load_audio_from_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Load audio from bytes, returning (audio_array, sample_rate)."""
    import soundfile as sf

    # Try reading with soundfile first (handles WAV, FLAC, OGG, etc.)
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono
        return audio, sr
    except Exception:
        pass

    # Fallback to scipy for other formats
    try:
        from scipy.io import wavfile
        sr, audio = wavfile.read(io.BytesIO(audio_bytes))
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    except Exception as e:
        raise ValueError(f"Could not load audio: {e}")


@register_stt("nemotron")
class NemotronSTT(BaseModelService):
    """
    Nemotron Speech Streaming STT implementation.

    Features:
    - Cache-aware streaming for low latency
    - Native punctuation and capitalization
    - 600M parameter FastConformer + RNN-T architecture
    - Optimized for voice assistant applications
    """

    CAPABILITIES = ["transcription", "punctuation", "streaming"]

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_path: Optional[Path] = None,
    ):
        super().__init__(
            name="nemotron",
            model_id=model_name,
            cache_path=cache_path or Path("models/nemotron"),
            log_file=Path("logs/atlas_stt.log"),
        )
        self._model_name = model_name

        # Streaming state (initialized lazily)
        self._streaming_asr = None
        self._streaming_initialized = False

        # Audio buffer for streaming transcription
        self._audio_buffer = []
        self._total_samples = 0

        # Lock to prevent concurrent CUDA access
        self._inference_lock = asyncio.Lock()

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            model_id=self._model_name,
            is_loaded=self.is_loaded,
            device=self.device,
            capabilities=self.CAPABILITIES,
        )

    def load(self) -> None:
        """Load the Nemotron ASR model."""
        if self._model is not None:
            self.logger.info("Model already loaded")
            return

        self.logger.info("Loading Nemotron model: %s", self._model_name)

        try:
            import nemo.collections.asr as nemo_asr
            import torch
        except ImportError:
            raise ImportError(
                "NeMo toolkit not installed. Install with:\n"
                "pip install 'nemo_toolkit[asr]'"
            )

        # Load model from HuggingFace
        self._model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self._model_name
        )
        self._model.eval()

        # Move to GPU if available
        if torch.cuda.is_available():
            self._model = self._model.cuda()
            self._device = "cuda"
            self.logger.info("Nemotron loaded on CUDA")
        else:
            self._device = "cpu"
            self.logger.info("Nemotron loaded on CPU")

        self.logger.info("Nemotron STT model loaded successfully")

    def unload(self) -> None:
        """Unload the model from memory."""
        # Clean up streaming state first
        if self._streaming_asr is not None:
            self._streaming_asr = None
            self._streaming_initialized = False

        if self._model is not None:
            self.logger.info("Unloading Nemotron model")
            del self._model
            self._model = None
            self._clear_gpu_memory()
            self.logger.info("Nemotron model unloaded")

    def init_streaming(self) -> None:
        """Initialize the streaming ASR wrapper for real-time transcription."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self._streaming_initialized:
            return

        from ...config import settings
        from nemo.collections.asr.parts.utils.streaming_utils import BatchedFrameASRRNNT

        # Convert frame length from ms to seconds
        frame_len_sec = settings.stt.nemotron_frame_len_ms / 1000.0

        self._streaming_asr = BatchedFrameASRRNNT(
            asr_model=self._model,
            frame_len=frame_len_sec,
            total_buffer=settings.stt.nemotron_buffer_sec,
            batch_size=1,
            stateful_decoding=True,
        )
        self._streaming_initialized = True
        self.logger.info(
            "Streaming ASR initialized (frame=%.0fms, buffer=%.1fs)",
            settings.stt.nemotron_frame_len_ms,
            settings.stt.nemotron_buffer_sec,
        )

    def reset_streaming(self) -> None:
        """Reset streaming state for a new utterance."""
        if self._streaming_asr is not None:
            self._streaming_asr.reset()

        # Clear audio buffer
        self.clear_audio_buffer()
        self.logger.debug("Streaming state reset")

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
    ) -> dict[str, Any]:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data (WAV, FLAC, etc.)
            filename: Original filename (used for format detection)

        Returns:
            Dict with transcript, punctuation info, and metrics
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not audio_bytes:
            self.logger.warning("Empty audio payload received")
            return {"error": "Empty audio payload.", "transcript": ""}

        self.logger.info("Transcription request: %s (%d bytes)", filename, len(audio_bytes))

        loop = asyncio.get_event_loop()

        def _transcribe() -> tuple[str, dict]:
            import torch

            # Load and preprocess audio
            audio, orig_sr = _load_audio_from_bytes(audio_bytes)

            # Resample to 16kHz if needed
            if orig_sr != SAMPLE_RATE:
                self.logger.debug("Resampling from %d to %d Hz", orig_sr, SAMPLE_RATE)
                audio = _resample_audio(audio, orig_sr, SAMPLE_RATE)

            # Ensure float32
            audio = audio.astype(np.float32)

            # Log audio stats
            audio_max = np.max(np.abs(audio))
            audio_rms = np.sqrt(np.mean(audio ** 2))
            duration_sec = len(audio) / SAMPLE_RATE
            self.logger.debug(
                "Audio: %.2fs, max=%.4f, rms=%.4f",
                duration_sec, audio_max, audio_rms
            )

            # Transcribe
            with torch.no_grad():
                results = self._model.transcribe([audio], batch_size=1)

            # Extract text from result
            result = results[0]
            if hasattr(result, 'text'):
                text = result.text
            else:
                text = str(result)

            # Analyze punctuation
            punct_info = {
                "has_period": "." in text,
                "has_question": "?" in text,
                "has_exclamation": "!" in text,
                "ends_with_punctuation": text.rstrip().endswith((".", "?", "!")),
            }

            return text, punct_info

        # Acquire lock to prevent concurrent CUDA access
        async with self._inference_lock:
            with InferenceTimer() as timer:
                transcript, punct_info = await loop.run_in_executor(None, _transcribe)

        metrics = self.gather_metrics(timer.duration)

        self.logger.info(
            "Transcription: '%.50s%s' (%.0fms, punct=%s)",
            transcript,
            "..." if len(transcript) > 50 else "",
            metrics.duration_ms,
            punct_info["ends_with_punctuation"],
        )

        return {
            "transcript": transcript,
            "punctuation": punct_info,
            "language": "en",
            "confidence": 1.0,  # Nemotron doesn't provide confidence scores
            "metrics": metrics.to_dict(),
        }

    def add_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
        """
        Add an audio chunk to the streaming buffer.

        Args:
            audio_chunk: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate of the audio
        """
        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            audio_chunk = _resample_audio(audio_chunk, sample_rate, SAMPLE_RATE)

        audio_chunk = audio_chunk.astype(np.float32)
        self._audio_buffer.append(audio_chunk)
        self._total_samples += len(audio_chunk)

    def get_buffer_duration_ms(self) -> float:
        """Get current buffer duration in milliseconds."""
        return (self._total_samples / SAMPLE_RATE) * 1000.0

    def clear_audio_buffer(self) -> None:
        """Clear the streaming audio buffer."""
        self._audio_buffer = []
        self._total_samples = 0

    async def transcribe_buffer(self) -> dict[str, Any]:
        """
        Transcribe the accumulated audio buffer.

        Returns partial transcript from all accumulated audio.
        Call this periodically during streaming for partial results.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not self._audio_buffer:
            return {"transcript": "", "is_final": False, "latency_ms": 0.0}

        import torch

        # Concatenate all buffered audio
        audio = np.concatenate(self._audio_buffer)

        # Acquire lock to prevent concurrent CUDA access
        async with self._inference_lock:
            with InferenceTimer() as timer:
                with torch.no_grad():
                    results = self._model.transcribe([audio], batch_size=1)

                result = results[0]
                if hasattr(result, 'text'):
                    text = result.text
                else:
                    text = str(result)

        # Check for utterance end punctuation
        is_final = text.rstrip().endswith((".", "?", "!"))

        return {
            "transcript": text,
            "is_final": is_final,
            "latency_ms": timer.duration * 1000,
            "buffer_duration_ms": self.get_buffer_duration_ms(),
        }

    async def transcribe_streaming(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> dict[str, Any]:
        """
        Process a streaming audio chunk and return partial transcript.

        Adds chunk to buffer and transcribes if enough audio accumulated.
        For real-time streaming, call this with successive audio chunks.

        Args:
            audio_chunk: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate of the audio

        Returns:
            Dict with partial transcript, is_final flag, and timing info
        """
        from ...config import settings

        # Add chunk to buffer
        self.add_audio_chunk(audio_chunk, sample_rate)

        # Get minimum transcription interval from config (frame_len_ms * 3 for stability)
        min_interval_ms = settings.stt.nemotron_frame_len_ms * 3

        # Only transcribe if we have enough audio
        buffer_ms = self.get_buffer_duration_ms()
        if buffer_ms < min_interval_ms:
            return {
                "transcript": "",
                "is_final": False,
                "latency_ms": 0.0,
                "buffer_duration_ms": buffer_ms,
                "needs_more_audio": True,
            }

        # Transcribe accumulated buffer
        result = await self.transcribe_buffer()
        result["needs_more_audio"] = False
        return result
