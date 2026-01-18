"""
Nemotron Speech-to-Text STT implementation.

NVIDIA's ASR model (600M params) optimized for:
- Real-time voice assistant applications
- Native punctuation and capitalization output
- Cache-aware streaming for live audio processing

Cache-Aware Streaming:
    The model maintains internal encoder cache states across audio chunks,
    eliminating redundant recomputation of previous context. This provides
    ~3x throughput improvement over buffered inference for streaming audio.
"""

import asyncio
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..base import BaseModelService, InferenceTimer
from ..protocols import ModelInfo
from ..registry import register_stt

logger = logging.getLogger("atlas.stt.nemotron")

# Model constants
MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
SAMPLE_RATE = 16000  # Model expects 16kHz audio

# Streaming chunk configuration
# att_context_size controls latency: [70,0]=80ms, [70,1]=160ms, [70,6]=560ms, [70,13]=1.12s
DEFAULT_ATT_CONTEXT_SIZE = [70, 13]  # 1.12s chunks - balance of latency and accuracy


@dataclass
class StreamingState:
    """Holds cache state for cache-aware streaming inference.

    The encoder maintains three cache tensors:
    - cache_last_channel: Channel-wise cache for attention layers
    - cache_last_time: Time-wise cache for convolution layers
    - cache_last_channel_len: Valid lengths of channel cache

    Plus decoder state:
    - previous_hypotheses: Partial hypotheses from previous chunks
    - previous_pred_out: Previous prediction outputs for RNN-T
    - accumulated_text: Full transcript accumulated across chunks
    """
    cache_last_channel: Optional[torch.Tensor] = None
    cache_last_time: Optional[torch.Tensor] = None
    cache_last_channel_len: Optional[torch.Tensor] = None
    previous_hypotheses: Optional[list] = None
    previous_pred_out: Optional[torch.Tensor] = None
    accumulated_text: str = ""
    chunk_count: int = 0
    total_audio_samples: int = 0
    is_initialized: bool = False


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
    Nemotron Speech-to-Text STT implementation with cache-aware streaming.

    Features:
    - Native punctuation and capitalization
    - 600M parameter FastConformer + RNN-T architecture
    - Optimized for voice assistant applications
    - Cache-aware streaming for ~3x throughput on live audio

    Usage:
        # Batch mode (full audio)
        result = await stt.transcribe(audio_bytes)

        # Streaming mode (incremental chunks)
        stt.reset_streaming()
        for chunk in audio_chunks:
            result = await stt.transcribe_chunk(chunk)
        final_text = stt.get_streaming_transcript()
    """

    CAPABILITIES = ["transcription", "punctuation", "streaming"]

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_path: Optional[Path] = None,
        att_context_size: Optional[list] = None,
    ):
        super().__init__(
            name="nemotron",
            model_id=model_name,
            cache_path=cache_path or Path("models/nemotron"),
            log_file=Path("logs/atlas_stt.log"),
        )
        self._model_name = model_name
        self._att_context_size = att_context_size or DEFAULT_ATT_CONTEXT_SIZE

        # Lock to prevent concurrent CUDA access
        self._inference_lock = asyncio.Lock()

        # Streaming state
        self._streaming_state: Optional[StreamingState] = None
        self._preprocessor = None  # Audio preprocessor for mel spectrogram

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
        """Load the Nemotron ASR model with streaming support."""
        if self._model is not None:
            self.logger.info("Model already loaded")
            return

        self.logger.info("Loading Nemotron model: %s", self._model_name)

        try:
            import nemo.collections.asr as nemo_asr
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

        # Configure streaming parameters
        self._model.encoder.setup_streaming_params(
            att_context_size=self._att_context_size
        )
        self.logger.info(
            "Streaming configured: att_context_size=%s, cfg=%s",
            self._att_context_size,
            self._model.encoder.streaming_cfg
        )

        # Store preprocessor reference for streaming
        self._preprocessor = self._model.preprocessor

        self.logger.info("Nemotron STT model loaded successfully with streaming support")

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            self.logger.info("Unloading Nemotron model")
            self._streaming_state = None
            self._preprocessor = None
            del self._model
            self._model = None
            self._clear_gpu_memory()
            self.logger.info("Nemotron model unloaded")

    # =========================================================================
    # Cache-Aware Streaming API
    # =========================================================================

    def reset_streaming(self) -> None:
        """Reset streaming state for a new utterance.

        Call this before starting to process a new audio stream.
        Initializes encoder cache tensors and clears accumulated transcript.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Initialize cache state from encoder
        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype

        cache_last_channel, cache_last_time, cache_last_channel_len = (
            self._model.encoder.get_initial_cache_state(
                batch_size=1,
                dtype=dtype,
                device=device,
            )
        )

        self._streaming_state = StreamingState(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            previous_hypotheses=None,
            previous_pred_out=None,
            accumulated_text="",
            chunk_count=0,
            total_audio_samples=0,
            is_initialized=True,
        )
        self.logger.debug("Streaming state reset")

    def get_streaming_transcript(self) -> str:
        """Get the full transcript accumulated during streaming."""
        if self._streaming_state is None:
            return ""
        return self._streaming_state.accumulated_text

    async def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
    ) -> dict[str, Any]:
        """
        Transcribe an audio chunk using cache-aware streaming.

        This method processes audio incrementally, maintaining encoder cache
        state between calls. Each call processes new audio without recomputing
        previous context, providing ~3x throughput improvement.

        Args:
            audio_chunk: Float32 audio samples at 16kHz (mono)

        Returns:
            Dict with:
            - chunk_text: New text from this chunk
            - accumulated_text: Full transcript so far
            - chunk_count: Number of chunks processed
            - is_final: False (use finalize_streaming for final)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self._streaming_state is None or not self._streaming_state.is_initialized:
            self.reset_streaming()

        loop = asyncio.get_event_loop()
        state = self._streaming_state

        def _transcribe_chunk() -> str:
            # Ensure float32 format
            audio = audio_chunk.astype(np.float32)

            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            audio_len = torch.tensor([len(audio)], dtype=torch.long)

            # Move to device
            device = next(self._model.parameters()).device
            audio_tensor = audio_tensor.to(device)
            audio_len = audio_len.to(device)

            # Preprocess audio to mel spectrogram
            with torch.no_grad():
                processed_signal, processed_signal_length = self._preprocessor(
                    input_signal=audio_tensor,
                    length=audio_len,
                )

                # Run streaming step with cache
                # Returns: (pred_out, transcribed_texts, cache_channel, cache_time, cache_len, hypotheses)
                (
                    state.previous_pred_out,
                    transcribed_texts,
                    state.cache_last_channel,
                    state.cache_last_time,
                    state.cache_last_channel_len,
                    state.previous_hypotheses,
                ) = self._model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=state.cache_last_channel,
                    cache_last_time=state.cache_last_time,
                    cache_last_channel_len=state.cache_last_channel_len,
                    previous_hypotheses=state.previous_hypotheses,
                    previous_pred_out=state.previous_pred_out,
                    return_transcription=True,
                )

            # Extract text from Hypothesis object
            # RNN-T returns list of Hypothesis objects with .text attribute
            chunk_text = ""
            if transcribed_texts and len(transcribed_texts) > 0:
                hyp = transcribed_texts[0]
                if hasattr(hyp, 'text'):
                    chunk_text = hyp.text
                elif isinstance(hyp, str):
                    chunk_text = hyp
                else:
                    chunk_text = str(hyp)

            return chunk_text

        # Acquire lock and process chunk
        async with self._inference_lock:
            with InferenceTimer() as timer:
                chunk_text = await loop.run_in_executor(None, _transcribe_chunk)

        # Update state
        state.chunk_count += 1
        state.total_audio_samples += len(audio_chunk)
        state.accumulated_text = chunk_text  # RNN-T gives full hypothesis

        metrics = self.gather_metrics(timer.duration)

        self.logger.debug(
            "Chunk %d: '%.30s%s' (%.0fms)",
            state.chunk_count,
            chunk_text,
            "..." if len(chunk_text) > 30 else "",
            metrics.duration_ms,
        )

        return {
            "chunk_text": chunk_text,
            "accumulated_text": state.accumulated_text,
            "chunk_count": state.chunk_count,
            "is_final": False,
            "metrics": metrics.to_dict(),
        }

    def finalize_streaming(self) -> dict[str, Any]:
        """Finalize streaming and return the complete transcript.

        Call this when the audio stream ends to get the final result
        and clean up streaming state.

        Returns:
            Dict with final transcript and statistics
        """
        if self._streaming_state is None:
            return {"transcript": "", "chunk_count": 0, "total_samples": 0}

        state = self._streaming_state
        result = {
            "transcript": state.accumulated_text,
            "chunk_count": state.chunk_count,
            "total_samples": state.total_audio_samples,
            "duration_seconds": state.total_audio_samples / SAMPLE_RATE,
        }

        # Analyze punctuation
        text = state.accumulated_text
        result["punctuation"] = {
            "has_period": "." in text,
            "has_question": "?" in text,
            "has_exclamation": "!" in text,
            "ends_with_punctuation": text.rstrip().endswith((".", "?", "!")),
        }

        self.logger.info(
            "Streaming finalized: '%s' (%d chunks, %.2fs audio)",
            text[:50] + "..." if len(text) > 50 else text,
            state.chunk_count,
            result["duration_seconds"],
        )

        # Clear state
        self._streaming_state = None

        return result

    # =========================================================================
    # Batch Transcription API (original)
    # =========================================================================

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
