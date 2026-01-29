"""
Voice pipeline for Atlas Brain.

Main voice-to-voice pipeline that integrates:
- Wake word detection (OpenWakeWord)
- Voice Activity Detection (WebRTC VAD)
- Speech recognition (HTTP or WebSocket streaming ASR)
- Atlas agent for response generation
- Text-to-speech (Piper)
"""

import asyncio
import io
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import webrtcvad
from openwakeword.model import Model as WakeWordModel

try:
    import websockets
    from websockets.sync.client import connect as ws_connect
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from .audio_capture import AudioCapture
from .command_executor import CommandExecutor
from .frame_processor import FrameProcessor
from .playback import PlaybackController
from .segmenter import CommandSegmenter

logger = logging.getLogger("atlas.voice.pipeline")


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Convert PCM audio to WAV format."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buffer.getvalue()


class NemotronAsrHttpClient:
    """HTTP client for Nemotron ASR service."""

    def __init__(
        self,
        url: Optional[str],
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout

    def transcribe(self, wav_bytes: bytes, sample_rate: int) -> Optional[str]:
        """Transcribe WAV audio to text."""
        if not self.url:
            logger.error("No Nemotron ASR URL configured.")
            return None
        headers = {}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"sample_rate": sample_rate}
        try:
            response = requests.post(
                self.url,
                data=data,
                files=files,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.error("Nemotron ASR request failed: %s", exc)
            return None
        for key in ("text", "transcript"):
            if key in payload:
                return payload[key]
        logger.warning("Nemotron ASR response missing transcript: %s", payload)
        return None


class NemotronAsrStreamingClient:
    """WebSocket streaming client for Nemotron ASR service.

    Streams audio chunks to the ASR server and receives transcripts.
    Designed to reduce latency by processing audio incrementally.
    """

    def __init__(
        self,
        url: Optional[str],
        timeout: int = 30,
        sample_rate: int = 16000,
    ):
        """Initialize streaming ASR client.

        Args:
            url: WebSocket URL (e.g., ws://localhost:8080/v1/asr/stream)
            timeout: Connection and receive timeout in seconds
            sample_rate: Audio sample rate (default 16kHz)
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "websockets package required for streaming ASR. "
                "Install with: pip install websockets"
            )
        self.url = url
        self.timeout = timeout
        self.sample_rate = sample_rate
        self._ws: Optional[Any] = None
        self._lock = threading.Lock()
        self._connected = False
        self._last_partial = ""

    def connect(self) -> bool:
        """Establish WebSocket connection.

        Returns:
            True if connected successfully
        """
        if not self.url:
            logger.error("No streaming ASR URL configured")
            return False

        with self._lock:
            if self._connected and self._ws:
                return True

            try:
                self._ws = ws_connect(
                    self.url,
                    open_timeout=self.timeout,
                    close_timeout=5,
                )
                self._connected = True
                self._last_partial = ""
                logger.info("Streaming ASR connected to %s", self.url)
                return True
            except Exception as e:
                logger.error("Failed to connect to streaming ASR: %s", e)
                self._ws = None
                self._connected = False
                return False

    def disconnect(self) -> None:
        """Close WebSocket connection."""
        with self._lock:
            if self._ws:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None
            self._connected = False
            self._last_partial = ""

    def send_audio(self, pcm_bytes: bytes) -> Optional[str]:
        """Send audio chunk and return any partial transcript.

        Args:
            pcm_bytes: Raw PCM audio (int16, mono)

        Returns:
            Partial transcript if available, None otherwise
        """
        if not self._connected or not self._ws:
            return None

        try:
            # Send binary audio
            self._ws.send(pcm_bytes)

            # Check for partial transcript (native timeout, no socket mode toggle)
            try:
                response = self._ws.recv(timeout=0.001)
                data = json.loads(response)
                if data.get("type") == "partial":
                    self._last_partial = data.get("text", "")
                    return self._last_partial
            except TimeoutError:
                pass  # No data available yet

            return None
        except Exception as e:
            logger.warning("Error sending audio to streaming ASR: %s", e)
            self._connected = False
            return None

    def finalize(self) -> Optional[str]:
        """Request final transcription.

        Returns:
            Final transcript text, or None on error
        """
        if not self._connected or not self._ws:
            logger.warning("Cannot finalize: not connected to streaming ASR")
            return None

        try:
            # Send finalize command
            self._ws.send(json.dumps({"type": "finalize"}))

            # Wait for final response
            response = self._ws.recv(timeout=self.timeout)
            data = json.loads(response)

            if data.get("type") == "final":
                text = data.get("text", "")
                duration = data.get("duration_sec", 0)
                logger.info(
                    "Streaming ASR finalized: %.2fs -> '%s'",
                    duration,
                    text[:50] if text else "(empty)",
                )
                return text
            elif data.get("type") == "error":
                logger.error("Streaming ASR error: %s", data.get("message"))
                return None
            else:
                logger.warning("Unexpected response type: %s", data.get("type"))
                return data.get("text")

        except Exception as e:
            logger.error("Error finalizing streaming ASR: %s", e)
            self._connected = False
            return None

    def reset(self) -> None:
        """Reset the streaming session for a new utterance."""
        if not self._connected or not self._ws:
            return

        try:
            self._ws.send(json.dumps({"type": "reset"}))
            # Drain any response
            try:
                self._ws.recv(timeout=0.5)
            except Exception:
                pass
            self._last_partial = ""
        except Exception as e:
            logger.warning("Error resetting streaming ASR: %s", e)

    def transcribe(self, wav_bytes: bytes, sample_rate: int) -> Optional[str]:
        """Transcribe complete audio (compatibility with HTTP client).

        This method provides compatibility with NemotronAsrHttpClient.
        For streaming use, prefer connect/send_audio/finalize pattern.

        Args:
            wav_bytes: WAV audio bytes
            sample_rate: Sample rate (unused, for compatibility)

        Returns:
            Transcript text
        """
        # Extract PCM from WAV
        try:
            buffer = io.BytesIO(wav_bytes)
            with wave.open(buffer, "rb") as wf:
                pcm_bytes = wf.readframes(wf.getnframes())
        except Exception as e:
            logger.error("Failed to extract PCM from WAV: %s", e)
            return None

        # Connect if needed
        if not self._connected:
            if not self.connect():
                return None

        # Send all audio
        chunk_size = 2560  # 80ms at 16kHz, int16
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            self.send_audio(chunk)

        # Get final transcript
        return self.finalize()

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int) -> Optional[str]:
        """Transcribe PCM audio directly (avoids WAV encode/decode).

        Args:
            pcm_bytes: Raw PCM audio bytes (int16, mono)
            sample_rate: Sample rate (for logging only)

        Returns:
            Transcript text
        """
        if not self._connected:
            if not self.connect():
                return None

        chunk_size = 2560  # 80ms at 16kHz, int16
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            self.send_audio(chunk)

        return self.finalize()

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    @property
    def last_partial(self) -> str:
        """Get the last partial transcript received."""
        return self._last_partial


class SentenceBuffer:
    """Buffer that accumulates tokens and yields complete sentences."""

    SENTENCE_ENDINGS = ".!?"

    def __init__(self):
        self._tokens: list[str] = []

    def add_token(self, token: str) -> Optional[str]:
        """
        Add a token to buffer. Returns sentence if complete.

        Args:
            token: Token string from LLM

        Returns:
            Complete sentence if buffer ends with sentence punctuation, else None
        """
        self._tokens.append(token)
        combined = "".join(self._tokens).strip()
        if combined and combined[-1] in self.SENTENCE_ENDINGS:
            self._tokens.clear()
            return combined
        return None

    def flush(self) -> Optional[str]:
        """Flush remaining buffer content."""
        combined = "".join(self._tokens).strip()
        if combined:
            self._tokens.clear()
            return combined
        return None

    def clear(self):
        """Clear the buffer."""
        self._tokens.clear()


class PiperTTS:
    """Piper TTS engine for speech synthesis with streaming support."""

    def __init__(
        self,
        binary_path: Optional[str],
        model_path: Optional[str],
        speaker: Optional[int] = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        sample_rate: int = 16000,
    ):
        self.binary_path = binary_path
        self.model_path = model_path
        self.speaker = speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.sample_rate = sample_rate
        self.stop_event = threading.Event()
        self.current_stream: Optional[sd.OutputStream] = None
        self._current_process: Optional[subprocess.Popen] = None

    def speak(self, text: str):
        """Speak the given text using Piper TTS with streaming."""
        if not self.binary_path or not os.path.isfile(self.binary_path):
            logger.error("Piper binary not found: %s", self.binary_path)
            return
        if not self.model_path or not os.path.isfile(self.model_path):
            logger.error("Piper model not found: %s", self.model_path)
            return

        self.stop_event.clear()
        try:
            self._speak_streaming(text)
        except Exception as exc:
            logger.warning("Streaming TTS failed (%s), falling back to batch", exc)
            self._speak_batch(text)

    def _speak_streaming(self, text: str):
        """Stream audio directly from Piper stdout."""
        cmd = [
            self.binary_path,
            "--model",
            self.model_path,
            "--output-raw",
            "--length_scale",
            str(self.length_scale),
            "--noise_scale",
            str(self.noise_scale),
            "--noise_w",
            str(self.noise_w),
        ]
        if self.speaker is not None:
            cmd.extend(["--speaker", str(self.speaker)])

        start_time = time.perf_counter()
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._current_process = process

        try:
            process.stdin.write(text.encode("utf-8"))
            process.stdin.close()
        except BrokenPipeError:
            logger.error("Piper process died unexpectedly")
            raise

        chunk_bytes = 4096  # 2048 int16 samples = 128ms at 16kHz
        first_chunk = True

        with sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16"
        ) as stream:
            self.current_stream = stream
            while not self.stop_event.is_set():
                chunk = process.stdout.read(chunk_bytes)
                if not chunk:
                    break
                if first_chunk:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    logger.info("TTS first chunk latency: %.0fms", latency_ms)
                    first_chunk = False
                audio = np.frombuffer(chunk, dtype=np.int16)
                stream.write(audio)
            try:
                stream.stop()
            except Exception:
                pass
        self.current_stream = None
        self._current_process = None

        if self.stop_event.is_set():
            process.terminate()
            process.wait(timeout=1.0)
        else:
            process.wait(timeout=5.0)

    def _speak_batch(self, text: str):
        """Fallback: synthesize to file then play (original method)."""
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            cmd = [
                self.binary_path,
                "--model",
                self.model_path,
                "--output_file",
                wav_path,
                "--length_scale",
                str(self.length_scale),
                "--noise_scale",
                str(self.noise_scale),
                "--noise_w",
                str(self.noise_w),
            ]
            if self.speaker is not None:
                cmd.extend(["--speaker", str(self.speaker)])
            subprocess.run(cmd, input=text.encode("utf-8"), check=True)
            audio, sr = sf.read(wav_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            chunk = 2048
            with sd.OutputStream(samplerate=sr, channels=1, dtype="float32") as stream:
                self.current_stream = stream
                for start in range(0, len(audio), chunk):
                    if self.stop_event.is_set():
                        break
                    stream.write(audio[start:start + chunk])
                try:
                    stream.stop()
                except Exception:
                    pass
            self.current_stream = None
        except Exception as exc:
            logger.error("Piper batch synthesis failed: %s", exc)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    def stop(self):
        """Stop current playback and terminate Piper process."""
        self.stop_event.set()
        try:
            if self._current_process is not None:
                try:
                    self._current_process.terminate()
                except Exception:
                    pass
            if self.current_stream is not None:
                try:
                    self.current_stream.abort()
                except Exception:
                    pass
            sd.stop()
        except Exception:
            pass


class VoicePipeline:
    """Main voice-to-voice pipeline."""

    def __init__(
        self,
        wakeword_model_paths: List[str],
        wake_threshold: float,
        asr_client: NemotronAsrHttpClient,
        tts: PiperTTS,
        agent_runner: Callable[[str, Dict[str, Any]], str],
        streaming_agent_runner: Optional[Callable[[str, Dict[str, Any], Callable[[str], None]], None]] = None,
        streaming_llm_enabled: bool = False,
        sample_rate: int = 16000,
        block_size: int = 1280,
        silence_ms: int = 500,
        max_command_seconds: int = 5,
        min_command_ms: int = 1500,
        vad_aggressiveness: int = 2,
        hangover_ms: int = 300,
        use_arecord: bool = False,
        arecord_device: str = "default",
        input_device: Optional[str] = None,
        stop_hotkey: bool = True,
        allow_wake_barge_in: bool = False,
        interrupt_on_speech: bool = False,
        interrupt_speech_frames: int = 5,
        interrupt_rms_threshold: float = 0.05,
        interrupt_wake_models: Optional[List[str]] = None,
        interrupt_wake_threshold: float = 0.5,
        command_workers: int = 2,
        audio_gain: float = 1.0,
        prefill_runner: Optional[Callable[[], None]] = None,
        debug_logging: bool = False,
        log_interval_frames: int = 160,
        conversation_mode_enabled: bool = False,
        conversation_timeout_ms: int = 8000,
        conversation_start_delay_ms: int = 500,
        conversation_speech_frames: int = 3,
        conversation_speech_tolerance: int = 2,
        conversation_rms_threshold: float = 0.01,
    ):
        self.sample_rate = sample_rate
        self.conversation_mode_enabled = conversation_mode_enabled
        self.conversation_timeout_ms = conversation_timeout_ms
        self.conversation_start_delay_ms = conversation_start_delay_ms
        self.conversation_speech_tolerance = conversation_speech_tolerance
        self.block_size = block_size
        self.wake_threshold = wake_threshold
        self.asr_client = asr_client
        self.agent_runner = agent_runner
        self.streaming_agent_runner = streaming_agent_runner
        self.streaming_llm_enabled = streaming_llm_enabled
        self.prefill_runner = prefill_runner
        self._prefill_in_progress = False
        self.session_id = str(uuid.uuid4())
        self.playback = PlaybackController(tts)

        self.segmenter = CommandSegmenter(
            sample_rate=self.sample_rate,
            block_size=self.block_size,
            silence_ms=silence_ms,
            hangover_ms=hangover_ms,
            max_command_seconds=max_command_seconds,
            min_command_ms=min_command_ms,
        )
        self.vad = webrtcvad.Vad(vad_aggressiveness)

        logger.info("Initializing wake word model with paths: %s", wakeword_model_paths)
        self.model = WakeWordModel(wakeword_model_paths=wakeword_model_paths)
        logger.info("Wake word model initialized successfully")
        self.stop_hotkey = stop_hotkey
        self.stop_hotkey_thread: Optional[threading.Thread] = None
        self.allow_wake_barge_in = allow_wake_barge_in
        self.interrupt_on_speech = interrupt_on_speech
        self.interrupt_speech_frames = max(1, interrupt_speech_frames)
        self.interrupt_rms_threshold = interrupt_rms_threshold
        self.current_allow_barge_in = True
        self.current_response_metadata: Dict[str, Any] = {}
        self.interrupt_threshold = interrupt_wake_threshold

        interrupt_wake_models = interrupt_wake_models or []
        self.interrupt_model = (
            WakeWordModel(wakeword_model_paths=interrupt_wake_models)
            if interrupt_wake_models
            else None
        )

        # Check if ASR client supports streaming (has connect method)
        streaming_client = None
        if hasattr(asr_client, "connect") and hasattr(asr_client, "send_audio"):
            streaming_client = asr_client
            logger.info("Streaming ASR client detected for frame processor")

        self.frame_processor = FrameProcessor(
            wake_predict=self.model.predict,
            wake_threshold=self.wake_threshold,
            segmenter=self.segmenter,
            vad=self.vad,
            allow_wake_barge_in=self.allow_wake_barge_in,
            interrupt_predict=(
                self.interrupt_model.predict if self.interrupt_model else None
            ),
            interrupt_threshold=self.interrupt_threshold,
            interrupt_on_speech=self.interrupt_on_speech,
            interrupt_speech_frames=self.interrupt_speech_frames,
            interrupt_rms_threshold=self.interrupt_rms_threshold,
            audio_gain=audio_gain,
            wake_reset=self.model.reset,
            on_wake_detected=self._trigger_prefill,
            streaming_asr_client=streaming_client,
            debug_logging=debug_logging,
            log_interval_frames=log_interval_frames,
            conversation_mode_enabled=conversation_mode_enabled,
            conversation_timeout_ms=conversation_timeout_ms,
            conversation_speech_frames=conversation_speech_frames,
            conversation_speech_tolerance=conversation_speech_tolerance,
            conversation_rms_threshold=conversation_rms_threshold,
            on_conversation_timeout=self._on_conversation_timeout,
        )

        self.capture = AudioCapture(
            sample_rate=self.sample_rate,
            block_size=self.block_size,
            use_arecord=use_arecord,
            arecord_device=arecord_device,
            input_device=input_device,
            debug_logging=debug_logging,
            log_interval_frames=log_interval_frames,
        )

        self.command_executor = CommandExecutor(
            handler=self._handle_command,
            max_workers=command_workers,
            streaming_handler=self._handle_streaming_command,
        )

    def start(self):
        """Start the voice pipeline."""
        if self.stop_hotkey:
            self.stop_hotkey_thread = threading.Thread(
                target=self._stop_listener, daemon=True
            )
            self.stop_hotkey_thread.start()
        logger.info(
            "Starting voice pipeline at %d Hz. Waiting for wake word.",
            self.sample_rate,
        )
        self.capture.run(self._process_frame)

    def _stop_listener(self):
        """Listen for 's' + Enter to stop playback."""
        import sys
        logger.info("Stop hotkey enabled: press 's' then Enter to stop TTS.")
        for line in sys.stdin:
            if line.strip().lower() == "s":
                logger.info("Stop hotkey pressed; stopping playback.")
                self._stop_playback()

    def _process_frame(self, frame_bytes: bytes):
        """Process an audio frame."""
        self.frame_processor.process_frame(
            frame_bytes=frame_bytes,
            is_speaking=self.playback.speaking.is_set(),
            current_allow_barge_in=self.current_allow_barge_in,
            stop_playback=self._stop_playback,
            on_finalize=self.command_executor.submit,
            on_streaming_finalize=self.command_executor.submit_streaming,
        )

    def _stop_playback(self):
        """Stop TTS playback and reset state."""
        self.playback.stop()
        self.frame_processor.reset()
        self.current_allow_barge_in = True
        self.current_response_metadata = {}

    def _handle_command(self, pcm_bytes: bytes):
        """Handle a completed voice command."""
        # Use PCM directly if client supports it (avoids WAV encode/decode)
        if hasattr(self.asr_client, "transcribe_pcm"):
            transcript = self.asr_client.transcribe_pcm(pcm_bytes, self.sample_rate)
        else:
            wav_bytes = pcm_to_wav_bytes(pcm_bytes, self.sample_rate)
            transcript = self.asr_client.transcribe(wav_bytes, self.sample_rate)
        if not transcript:
            return
        logger.info("ASR: %s", transcript)

        # Use streaming LLM if enabled
        if self.streaming_llm_enabled and self.streaming_agent_runner:
            self._handle_streaming_llm_command(transcript)
            return

        context = {"session_id": self.session_id}
        reply = self.agent_runner(transcript, context)

        if not reply:
            logger.warning("Agent returned empty reply")
            return
        logger.info("Speaking reply: %s", reply[:100] if len(reply) > 100 else reply)
        self.playback.speak(
            reply,
            on_start=self._on_playback_start,
            on_done=self._on_playback_done,
        )

    def _handle_streaming_command(self, transcript: str):
        """Handle a command with transcript from streaming ASR.

        Skips batch ASR transcription since we already have the transcript.
        """
        logger.info("_handle_streaming_command called with transcript: %r", transcript)
        if not transcript:
            logger.warning("Empty transcript from streaming ASR, returning without TTS")
            return
        logger.info("Streaming ASR: %s", transcript)

        # Use streaming LLM if enabled
        if self.streaming_llm_enabled and self.streaming_agent_runner:
            self._handle_streaming_llm_command(transcript)
            return

        context = {"session_id": self.session_id}
        reply = self.agent_runner(transcript, context)

        if not reply:
            logger.warning("Agent returned empty reply")
            return
        logger.info("Speaking reply: %s", reply[:100] if len(reply) > 100 else reply)
        self.playback.speak(
            reply,
            on_start=self._on_playback_start,
            on_done=self._on_playback_done,
        )

    def _handle_streaming_llm_command(self, transcript: str):
        """Handle command with streaming LLM to TTS - speak sentences as generated."""
        if not transcript:
            logger.warning("Empty transcript for streaming LLM")
            return

        logger.info("Streaming LLM command: %s", transcript)
        context = {"session_id": self.session_id}
        sentences = []

        def on_sentence(sentence: str):
            """Callback for each complete sentence from LLM."""
            sentences.append(sentence)
            log_text = sentence[:80] if len(sentence) > 80 else sentence
            logger.info("Streaming LLM sentence %d: %s", len(sentences), log_text)

        if self.streaming_agent_runner:
            self.streaming_agent_runner(transcript, context, on_sentence)
        else:
            # Fallback to non-streaming
            reply = self.agent_runner(transcript, context)
            if reply:
                on_sentence(reply)

        # Concatenate all sentences and speak as one utterance
        if sentences:
            full_reply = " ".join(sentences)
            logger.info("Speaking concatenated reply (%d sentences): %s",
                       len(sentences), full_reply[:100] if len(full_reply) > 100 else full_reply)
            self.playback.speak(
                full_reply,
                on_start=self._on_playback_start,
                on_done=self._on_playback_done,
            )
        else:
            logger.warning("No sentences generated from streaming LLM")

    def _on_playback_start(self):
        """Called when TTS playback starts."""
        self.frame_processor.interrupt_speech_counter = 0

    def _on_playback_done(self):
        """Called when TTS playback ends."""
        try:
            logger.info("TTS playback done, resetting wake word model")
            self.model.reset()
            logger.info("Wake word model reset after TTS")
        except Exception as e:
            logger.error("Error resetting wake word model after TTS: %s", e, exc_info=True)

        # Enter conversation mode if enabled (with delay to avoid echo detection)
        if self.conversation_mode_enabled:
            delay_sec = self.conversation_start_delay_ms / 1000.0
            logger.info("Scheduling conversation mode in %.0fms", self.conversation_start_delay_ms)
            timer = threading.Timer(delay_sec, self._enter_conversation_mode_delayed)
            timer.daemon = True
            timer.start()

    def _enter_conversation_mode_delayed(self):
        """Enter conversation mode after delay (called by timer)."""
        self.frame_processor.enter_conversation_mode()

    def _on_conversation_timeout(self):
        """Called when conversation mode times out."""
        logger.info("Conversation mode ended (timeout)")
        # Could play acknowledgment sound here if desired

    def _trigger_prefill(self):
        """Trigger LLM system prompt prefill in background.

        Called when wake word is detected to warm up the LLM KV cache
        while ASR is still recording. This reduces time-to-first-token
        when the actual request is made.
        """
        if self.prefill_runner is None:
            logger.debug("No prefill_runner configured")
            return
        # Guard against concurrent prefills
        if self._prefill_in_progress:
            logger.debug("Prefill already in progress, skipping")
            return
        self._prefill_in_progress = True
        logger.info("Spawning prefill thread...")
        # Run prefill in background thread to not block audio processing
        thread = threading.Thread(
            target=self._run_prefill,
            name="llm-prefill",
            daemon=True,
        )
        thread.start()

    def _run_prefill(self):
        """Execute the prefill runner."""
        try:
            self.prefill_runner()
        except Exception as e:
            logger.warning("LLM prefill failed: %s", e)
        finally:
            self._prefill_in_progress = False
