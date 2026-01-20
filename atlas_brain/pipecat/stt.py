"""
Parakeet STT Service for Pipecat.

Uses NVIDIA Parakeet-TDT-0.6b model for fast, accurate transcription.
Based on the model that achieved 1-second latency in the Modal demo.
"""

import asyncio
import io
import logging
import time
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

logger = logging.getLogger("atlas.pipecat.stt")


class ParakeetSTTService(SegmentedSTTService):
    """
    Speech-to-text service using NVIDIA Parakeet-TDT-0.6b.

    This model is optimized for fast, streaming-compatible transcription
    and was used in the Modal demo to achieve 1-second latency.
    """

    class InputParams(BaseModel):
        """Configuration for Parakeet STT."""
        language: str = "en"

    def __init__(
        self,
        *,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cuda",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """
        Initialize Parakeet STT service.

        Args:
            model_name: HuggingFace model name or local path
            device: Device to run on ("cuda" or "cpu")
            params: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self._model_name = model_name
        self._device = device
        self._params = params or self.InputParams()
        self._model = None
        self._sample_rate = 16000  # Parakeet expects 16kHz

    async def start(self, frame: StartFrame):
        """Initialize the model on pipeline start."""
        await self._load_model()
        await super().start(frame)

    async def _load_model(self):
        """Load the Parakeet model."""
        if self._model is not None:
            return

        logger.info("Loading Parakeet model: %s", self._model_name)
        start = time.time()

        try:
            import nemo.collections.asr as nemo_asr

            # Load model (runs in thread to not block)
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self._model_name
                )
            )

            # Move to device
            self._model = self._model.to(self._device)
            self._model.eval()

            load_time = time.time() - start
            logger.info("Parakeet model loaded in %.2fs on %s", load_time, self._device)

        except Exception as e:
            logger.error("Failed to load Parakeet model: %s", e)
            raise

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Transcribe audio bytes to text.

        Args:
            audio: Raw PCM audio bytes (16-bit, 16kHz, mono)

        Yields:
            TranscriptionFrame with the transcribed text
        """
        if self._model is None:
            yield ErrorFrame(error="Parakeet model not loaded")
            return

        try:
            start_time = time.time()

            # SegmentedSTTService passes WAV-wrapped audio, need to extract raw PCM
            # WAV files start with "RIFF" header
            if audio[:4] == b'RIFF':
                import wave
                wav_io = io.BytesIO(audio)
                with wave.open(wav_io, 'rb') as wav:
                    raw_audio = wav.readframes(wav.getnframes())
                audio_array = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                logger.debug("Extracted %d samples from WAV container", len(audio_array))
            else:
                audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Run transcription in thread pool
            loop = asyncio.get_event_loop()

            def transcribe():
                with torch.no_grad():
                    # Parakeet expects audio tensor and lengths
                    audio_tensor = torch.tensor(audio_array).unsqueeze(0).to(self._device)
                    lengths = torch.tensor([len(audio_array)]).to(self._device)

                    # Transcribe
                    hypotheses = self._model.transcribe(
                        audio=audio_tensor,
                        batch_size=1,
                    )

                    if hypotheses and len(hypotheses) > 0:
                        return hypotheses[0].text if hasattr(hypotheses[0], 'text') else str(hypotheses[0])
                    return ""

            text = await loop.run_in_executor(None, transcribe)

            latency = (time.time() - start_time) * 1000
            logger.info("Parakeet transcription: '%s' (%.0fms)", text[:50] if text else "", latency)

            if text:
                yield TranscriptionFrame(
                    text=text.strip(),
                    user_id="",
                    timestamp=time_now_iso8601(),
                )

        except Exception as e:
            logger.error("Parakeet transcription error: %s", e)
            yield ErrorFrame(error=str(e))


class NemotronSTTService(SegmentedSTTService):
    """
    Alternative STT service using NVIDIA Nemotron-0.6b.

    Similar to Parakeet but uses the streaming-optimized variant.
    """

    class InputParams(BaseModel):
        language: str = "en"

    def __init__(
        self,
        *,
        model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b",
        device: str = "cuda",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._device = device
        self._params = params or self.InputParams()
        self._model = None
        self._sample_rate = 16000
        self._audio_buffer_count = 0

    async def start(self, frame: StartFrame):
        """Initialize the model on pipeline start."""
        await self._load_model()
        await super().start(frame)
        logger.info("NemotronSTTService started - listening for speech segments")

    async def _load_model(self):
        if self._model is not None:
            return

        logger.info("Loading Nemotron model: %s", self._model_name)
        start = time.time()

        try:
            import nemo.collections.asr as nemo_asr

            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self._model_name
                )
            )

            self._model = self._model.to(self._device)
            self._model.eval()

            # Try to set larger attention context for better accuracy
            # [70, 6] = 560ms lookahead, trades latency for accuracy
            if hasattr(self._model, 'encoder') and hasattr(self._model.encoder, 'set_default_att_context_size'):
                try:
                    self._model.encoder.set_default_att_context_size([70, 6])
                    logger.info("Set encoder att_context_size to [70, 6] for better accuracy")
                except Exception as e:
                    logger.debug("Could not set att_context_size: %s", e)

            logger.info("Nemotron model loaded in %.2fs", time.time() - start)

        except Exception as e:
            logger.error("Failed to load Nemotron model: %s", e)
            raise

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        logger.info("run_stt called with %d bytes of audio", len(audio) if audio else 0)

        if not audio or len(audio) < 100:
            logger.warning("run_stt: Audio too short (%d bytes), skipping", len(audio) if audio else 0)
            return

        if self._model is None:
            logger.error("run_stt: Nemotron model not loaded!")
            yield ErrorFrame(error="Nemotron model not loaded")
            return

        try:
            start_time = time.time()

            # SegmentedSTTService passes WAV-wrapped audio, need to extract raw PCM
            # WAV files start with "RIFF" header
            if audio[:4] == b'RIFF':
                import wave
                wav_io = io.BytesIO(audio)
                with wave.open(wav_io, 'rb') as wav:
                    raw_audio = wav.readframes(wav.getnframes())
                audio_array = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                logger.debug("Extracted %d samples from WAV container", len(audio_array))
            else:
                audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

            logger.info("run_stt: Processing %.2f seconds of audio", len(audio_array) / 16000)

            loop = asyncio.get_event_loop()

            def transcribe():
                with torch.no_grad():
                    hypotheses = self._model.transcribe(
                        audio=[audio_array],
                        batch_size=1,
                    )
                    if hypotheses and len(hypotheses) > 0:
                        return hypotheses[0].text if hasattr(hypotheses[0], 'text') else str(hypotheses[0])
                    return ""

            text = await loop.run_in_executor(None, transcribe)

            latency = (time.time() - start_time) * 1000
            logger.info("Nemotron transcription: '%s' (%.0fms)", text, latency)

            if text:
                yield TranscriptionFrame(
                    text=text.strip(),
                    user_id="",
                    timestamp=time_now_iso8601(),
                )

        except Exception as e:
            logger.error("Nemotron transcription error: %s", e)
            yield ErrorFrame(error=str(e))
