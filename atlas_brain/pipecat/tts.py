"""
Kokoro TTS Service for Pipecat.

Uses Kokoro-82M for fast, high-quality text-to-speech.
Same model used in the Modal demo for 1-second latency.
"""

import asyncio
import io
import logging
import time
from typing import AsyncGenerator, Optional

import numpy as np
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

logger = logging.getLogger("atlas.pipecat.tts")


class KokoroTTSService(TTSService):
    """
    Text-to-speech service using Kokoro-82M.

    Kokoro is a fast 82M parameter TTS model that supports
    streaming output and phonetic input for precise pronunciation.
    """

    class InputParams(BaseModel):
        """Configuration for Kokoro TTS."""
        voice: str = "am_michael"
        speed: float = 1.15
        language: str = "a"  # 'a' = American English

    def __init__(
        self,
        *,
        voice: str = "am_michael",
        speed: float = 1.15,
        device: str = "cuda",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """
        Initialize Kokoro TTS service.

        Args:
            voice: Voice ID (e.g., "am_michael", "af_sarah")
            speed: Speech speed multiplier
            device: Device to run on ("cuda" or "cpu")
        """
        super().__init__(**kwargs)
        self._voice = voice
        self._speed = speed
        self._device = device
        self._params = params or self.InputParams(voice=voice, speed=speed)
        self._pipeline = None
        self._sample_rate = 24000  # Kokoro outputs 24kHz

    async def start(self, frame: Frame):
        """Initialize the model on pipeline start."""
        await super().start(frame)
        await self._load_model()

    async def process_frame(self, frame: Frame, direction):
        """Process incoming frames - log to debug TTS issues."""
        from pipecat.frames.frames import TextFrame
        if isinstance(frame, TextFrame):
            logger.info("TTS received TextFrame: '%s'", frame.text[:100] if frame.text else "")
        await super().process_frame(frame, direction)

    async def _load_model(self):
        """Load the Kokoro model."""
        if self._pipeline is not None:
            return

        logger.info("Loading Kokoro TTS (voice=%s, speed=%.2f, device=%s)",
                    self._voice, self._speed, self._device)
        start = time.time()

        try:
            from kokoro import KPipeline

            # Load in thread to not block
            loop = asyncio.get_event_loop()

            def load():
                return KPipeline(
                    lang_code=self._params.language,
                    repo_id="hexgrad/Kokoro-82M",
                    device=self._device,
                )

            self._pipeline = await loop.run_in_executor(None, load)

            logger.info("Kokoro TTS loaded in %.2fs", time.time() - start)

        except Exception as e:
            logger.error("Failed to load Kokoro TTS: %s", e)
            raise

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Convert text to speech audio.

        Args:
            text: Text to synthesize

        Yields:
            TTSStartedFrame, TTSAudioRawFrame chunks, TTSStoppedFrame
        """
        logger.info("TTS run_tts called with: '%s'", text[:100] if text else "")

        if self._pipeline is None:
            logger.error("TTS pipeline not loaded!")
            yield ErrorFrame(error="Kokoro TTS not loaded")
            return

        if not text or not text.strip():
            logger.warning("TTS received empty text, skipping")
            return

        try:
            start_time = time.time()
            yield TTSStartedFrame()

            # Run synthesis in thread pool
            loop = asyncio.get_event_loop()

            def synthesize():
                # Kokoro returns generator of audio chunks
                audio_chunks = []
                for _, _, audio in self._pipeline(
                    text,
                    voice=self._voice,
                    speed=self._speed,
                ):
                    if audio is not None:
                        # Convert PyTorch tensor to numpy if needed
                        if hasattr(audio, 'cpu'):
                            audio = audio.cpu().numpy()
                        # Convert to int16
                        audio_int16 = (audio * 32767).astype(np.int16)
                        audio_chunks.append(audio_int16.tobytes())
                return audio_chunks

            audio_chunks = await loop.run_in_executor(None, synthesize)

            # Yield audio frames
            for chunk in audio_chunks:
                yield TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                )

            latency = (time.time() - start_time) * 1000
            total_audio_bytes = sum(len(c) for c in audio_chunks)
            audio_duration = total_audio_bytes / (self._sample_rate * 2)  # 16-bit = 2 bytes
            logger.info("Kokoro TTS: %.1fs audio in %.0fms", audio_duration, latency)

            yield TTSStoppedFrame()

        except Exception as e:
            logger.error("Kokoro TTS error: %s", e)
            yield ErrorFrame(error=str(e))
            yield TTSStoppedFrame()


class StreamingKokoroTTSService(TTSService):
    """
    Streaming variant of Kokoro TTS that yields audio as it's generated.

    This can reduce time-to-first-audio by streaming chunks
    before the full synthesis is complete.
    """

    class InputParams(BaseModel):
        voice: str = "am_michael"
        speed: float = 1.15
        language: str = "a"

    def __init__(
        self,
        *,
        voice: str = "am_michael",
        speed: float = 1.15,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._voice = voice
        self._speed = speed
        self._device = device
        self._pipeline = None
        self._sample_rate = 24000

    async def start(self, frame: Frame):
        await super().start(frame)
        await self._load_model()

    async def process_frame(self, frame: Frame, direction):
        """Process incoming frames - log to debug TTS issues."""
        from pipecat.frames.frames import TextFrame
        if isinstance(frame, TextFrame):
            logger.info("StreamingTTS received TextFrame: '%s'", frame.text[:100] if frame.text else "")
        await super().process_frame(frame, direction)

    async def _load_model(self):
        if self._pipeline is not None:
            return

        logger.info("Loading Streaming Kokoro TTS")
        try:
            from kokoro import KPipeline

            loop = asyncio.get_event_loop()
            self._pipeline = await loop.run_in_executor(
                None,
                lambda: KPipeline(
                    lang_code="a",
                    repo_id="hexgrad/Kokoro-82M",
                    device=self._device,
                )
            )
            logger.info("Streaming Kokoro TTS loaded")

        except Exception as e:
            logger.error("Failed to load Streaming Kokoro: %s", e)
            raise

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.info("StreamingTTS run_tts called with: '%s'", text[:100] if text else "")

        if self._pipeline is None:
            logger.error("StreamingTTS pipeline not loaded!")
            yield ErrorFrame(error="Kokoro TTS not loaded")
            return

        if not text or not text.strip():
            logger.warning("StreamingTTS received empty text, skipping")
            return

        try:
            yield TTSStartedFrame()
            first_chunk = True
            start_time = time.time()

            # Process synchronously but yield asynchronously
            loop = asyncio.get_event_loop()

            # Create a queue for streaming
            import queue
            audio_queue = queue.Queue()
            done_event = asyncio.Event()

            def synthesize_streaming():
                try:
                    for _, _, audio in self._pipeline(
                        text,
                        voice=self._voice,
                        speed=self._speed,
                    ):
                        if audio is not None:
                            # Convert PyTorch tensor to numpy if needed
                            if hasattr(audio, 'cpu'):
                                audio = audio.cpu().numpy()
                            audio_int16 = (audio * 32767).astype(np.int16)
                            audio_queue.put(audio_int16.tobytes())
                finally:
                    audio_queue.put(None)  # Signal done

            # Start synthesis in background
            synthesis_task = loop.run_in_executor(None, synthesize_streaming)

            # Yield chunks as they arrive
            while True:
                try:
                    # Non-blocking get with short timeout
                    chunk = await loop.run_in_executor(
                        None,
                        lambda: audio_queue.get(timeout=0.01)
                    )
                    if chunk is None:
                        break

                    if first_chunk:
                        ttfa = (time.time() - start_time) * 1000
                        logger.info("Kokoro time-to-first-audio: %.0fms", ttfa)
                        first_chunk = False

                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                    )
                except queue.Empty:
                    await asyncio.sleep(0.001)
                    continue

            await synthesis_task
            yield TTSStoppedFrame()

            total_time = (time.time() - start_time) * 1000
            logger.info("Kokoro streaming TTS complete: %.0fms", total_time)

        except Exception as e:
            logger.error("Kokoro streaming TTS error: %s", e)
            yield ErrorFrame(error=str(e))
            yield TTSStoppedFrame()
