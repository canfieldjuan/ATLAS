"""
Orpheus TTS engine for expressive speech synthesis.

Uses orpheus-cpp (llama-cpp-python + SNAC decoder) for high-quality,
emotionally expressive TTS. Supports emotion tags like <laugh>, <sigh>.
Implements the SpeechEngine protocol.
"""

import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd

logger = logging.getLogger("atlas.voice.tts_orpheus")


class OrpheusTTS:
    """Orpheus TTS engine using orpheus-cpp for expressive speech synthesis.

    Supports streaming playback at 24kHz via sounddevice. The model is
    lazy-loaded on first speak() call to avoid blocking startup.
    """

    def __init__(
        self,
        voice: str = "tara",
        n_gpu_layers: int = -1,
        lang: str = "en",
        temperature: float = 0.6,
        top_p: float = 0.8,
        pre_buffer_size: float = 0.5,
        n_ctx: int = 8192,
    ):
        self.voice = voice
        self.n_gpu_layers = n_gpu_layers
        self.lang = lang
        self.temperature = temperature
        self.top_p = top_p
        self.pre_buffer_size = pre_buffer_size
        self.n_ctx = n_ctx
        self.stop_event = threading.Event()
        self.current_stream = None
        self._engine = None  # Lazy-loaded

    def _ensure_loaded(self):
        """Lazy-load Orpheus engine on first speak().

        Builds the engine manually instead of using OrpheusCpp() directly
        because OrpheusCpp sets n_ctx=0 (model default 131072) which
        requires ~14GB KV cache. We override to n_ctx=8192.
        """
        if self._engine is not None:
            return
        from orpheus_cpp import OrpheusCpp

        logger.info(
            "Loading OrpheusCpp (lang=%s, n_gpu_layers=%d)",
            self.lang, self.n_gpu_layers,
        )
        # Create engine without loading model (will replace _llm)
        engine = object.__new__(OrpheusCpp)

        # Download GGUF model file
        from huggingface_hub import hf_hub_download
        repo_id = OrpheusCpp.lang_to_model[self.lang]
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=repo_id.split("/")[-1].lower().replace(
                "-gguf", ".gguf",
            ),
        )
        logger.info("GGUF model: %s", model_file)

        # Load LLM with constrained context (8192 vs default 131072)
        from llama_cpp import Llama
        engine._llm = Llama(
            model_path=model_file,
            n_ctx=self.n_ctx,
            verbose=False,
            n_gpu_layers=self.n_gpu_layers,
            batch_size=1,
        )

        # Load SNAC decoder (ONNX)
        import onnxruntime
        snac_path = hf_hub_download(
            "onnx-community/snac_24khz-ONNX",
            subfolder="onnx",
            filename="decoder_model.onnx",
        )
        engine._snac_session = onnxruntime.InferenceSession(
            snac_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        self._engine = engine
        logger.info("OrpheusCpp engine loaded successfully")

    def speak(self, text: str):
        """Synthesize and stream audio via sounddevice.

        Uses a producer-consumer pattern: a background thread generates
        audio chunks into a queue while the main thread plays them.
        This prevents stutter when generation is slower than real-time.
        """
        self.stop_event.clear()
        try:
            self._ensure_loaded()
        except Exception as e:
            logger.error("Orpheus model load failed: %s", e)
            return

        start_time = time.perf_counter()
        sample_rate = 24000
        audio_queue = queue.Queue(maxsize=64)
        options = {
            "voice_id": self.voice,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "pre_buffer_size": self.pre_buffer_size,
        }

        def producer():
            """Generate audio chunks into queue."""
            try:
                for sr, samples in self._engine.stream_tts_sync(
                    text, options=options,
                ):
                    if self.stop_event.is_set():
                        break
                    if samples is None or len(samples) == 0:
                        continue
                    audio_f32 = samples.astype(np.float32) / 32768.0
                    audio_f32 = audio_f32.ravel()
                    try:
                        audio_queue.put(audio_f32, timeout=5.0)
                    except queue.Full:
                        break
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error("Orpheus generation failed: %s", e)
            finally:
                audio_queue.put(None)  # sentinel

        gen_thread = threading.Thread(
            target=producer, daemon=True, name="orpheus-gen",
        )
        gen_thread.start()

        self._play_from_queue(
            audio_queue, sample_rate, start_time,
        )
        gen_thread.join(timeout=2.0)

    def _play_from_queue(self, audio_queue, sample_rate, start_time):
        """Consume audio chunks from queue and play via sounddevice."""
        first_chunk = True
        try:
            stream = self._open_output_stream(sample_rate)
            with stream:
                self.current_stream = stream
                while not self.stop_event.is_set():
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if chunk is None:
                        break
                    if first_chunk:
                        latency_ms = (
                            time.perf_counter() - start_time
                        ) * 1000
                        logger.info(
                            "TTS first chunk latency: %.0fms",
                            latency_ms,
                        )
                        first_chunk = False
                    chunk_size = 4800  # 200ms at 24kHz
                    for i in range(0, len(chunk), chunk_size):
                        if self.stop_event.is_set():
                            break
                        stream.write(chunk[i:i + chunk_size])
                try:
                    stream.stop()
                except Exception as e:
                    logger.debug("Stream stop failed: %s", e)
        except Exception as e:
            logger.error("Orpheus playback failed: %s", e)
        finally:
            self.current_stream = None

    def _open_output_stream(
        self, samplerate: int, retries: int = 2,
    ) -> sd.OutputStream:
        """Open an output stream with retry on ALSA errors.

        After stream.abort(), ALSA may need a moment to release resources.
        Retry with a short delay handles this race condition.
        """
        last_err = None
        for attempt in range(retries + 1):
            try:
                return sd.OutputStream(
                    samplerate=samplerate, channels=1, dtype="float32",
                )
            except Exception as e:
                last_err = e
                if attempt < retries:
                    logger.debug(
                        "OutputStream open failed (attempt %d/%d): %s, retrying...",
                        attempt + 1, retries + 1, e,
                    )
                    time.sleep(0.2)
        raise last_err

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
            # Brief pause lets ALSA release the device so the next
            # OutputStream open does not hit PaErrorCode -9999.
            time.sleep(0.1)
        except Exception as e:
            logger.debug("sd.stop() failed: %s", e)
