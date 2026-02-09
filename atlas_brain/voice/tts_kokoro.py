"""
Kokoro TTS engine for high-quality speech synthesis.

Uses kokoro-onnx (82M params, 24kHz, 54 voices) as a drop-in
replacement for Piper TTS. Implements the SpeechEngine protocol.
"""

import logging
import os
import threading
import time

import sounddevice as sd

logger = logging.getLogger("atlas.voice.tts_kokoro")


class KokoroTTS:
    """Kokoro TTS engine using kokoro-onnx for natural speech synthesis."""

    def __init__(
        self,
        model_dir: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "en-us",
        model_file: str = "kokoro-v1.0.onnx",
        voices_file: str = "voices-v1.0.bin",
    ):
        self.model_dir = model_dir
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.model_file = model_file
        self.voices_file = voices_file
        self.stop_event = threading.Event()
        self.current_stream = None
        self._kokoro = None  # Lazy-loaded

    def _ensure_loaded(self):
        """Lazy-load Kokoro model on first speak()."""
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            model_path = os.path.join(self.model_dir, self.model_file)
            voices_path = os.path.join(self.model_dir, self.voices_file)

            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Kokoro model not found: {model_path}")
            if not os.path.isfile(voices_path):
                raise FileNotFoundError(f"Kokoro voices not found: {voices_path}")

            logger.info("Loading Kokoro model from %s", self.model_dir)
            self._kokoro = Kokoro(model_path, voices_path)
            logger.info("Kokoro model loaded successfully")

    def speak(self, text: str):
        """Synthesize and stream audio from Kokoro to sounddevice."""
        self.stop_event.clear()
        try:
            self._ensure_loaded()
        except Exception as e:
            logger.error("Kokoro model load failed: %s", e)
            return

        start_time = time.perf_counter()
        try:
            samples, sr = self._kokoro.create(
                text, voice=self.voice, speed=self.speed, lang=self.lang
            )
        except Exception as e:
            logger.error("Kokoro synthesis failed: %s", e)
            return

        synth_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Kokoro synthesis: %.0fms for %d samples (%.1fs audio)",
            synth_ms, len(samples), len(samples) / sr,
        )

        # Stream chunks to sounddevice at native 24kHz
        chunk_size = 4800  # 200ms at 24kHz
        first_chunk = True

        try:
            with sd.OutputStream(samplerate=sr, channels=1, dtype="float32") as stream:
                self.current_stream = stream
                for i in range(0, len(samples), chunk_size):
                    if self.stop_event.is_set():
                        break
                    if first_chunk:
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        logger.info("TTS first chunk latency: %.0fms", latency_ms)
                        first_chunk = False
                    stream.write(samples[i:i + chunk_size])
                try:
                    stream.stop()
                except Exception as e:
                    logger.debug("Stream stop failed: %s", e)
        except Exception as e:
            logger.error("Kokoro playback failed: %s", e)
        finally:
            self.current_stream = None

    def stop(self):
        """Stop current playback."""
        self.stop_event.set()
        try:
            if self.current_stream is not None:
                try:
                    self.current_stream.abort()
                except Exception as e:
                    logger.debug("Stream abort failed: %s", e)
            sd.stop()
        except Exception as e:
            logger.debug("sd.stop() failed: %s", e)
