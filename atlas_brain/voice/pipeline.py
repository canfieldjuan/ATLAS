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
import queue
from datetime import datetime
import random
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

from .vad import SileroVAD

try:
    import websockets
    from websockets.sync.client import connect as ws_connect
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from .audio_capture import AudioCapture
from .command_executor import CommandExecutor
from .frame_processor import FrameProcessor
from .playback import PlaybackController, SpeechEngine
from .segmenter import CommandSegmenter

logger = logging.getLogger("atlas.voice.pipeline")


def _log_transcript(transcript: str, source: str, wake_to_asr_ms: float) -> None:
    """Append one transcript record to data/transcripts.jsonl."""
    try:
        entry = {
            "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": source,           # "batch" | "streaming"
            "transcript": transcript,
            "wake_to_asr_ms": round(wake_to_asr_ms, 1),
        }
        path = os.path.join("data", "transcripts.jsonl")
        os.makedirs("data", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # never crash pipeline for logging


def _generate_tone(freq: int, duration_ms: int, sample_rate: int = 24000) -> np.ndarray:
    """Generate a short sine wave tone with fade-in/out."""
    t = np.linspace(0, duration_ms / 1000.0, int(sample_rate * duration_ms / 1000), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    fade = int(sample_rate * 0.01)
    if fade > 0 and len(tone) > 2 * fade:
        tone[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        tone[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return tone


class ErrorPhrase(str):
    """Marker subclass so the pipeline can distinguish error recovery
    phrases from normal LLM replies without fragile string comparison."""
    pass


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
        self._best_partial = ""  # Longest partial seen (word count)

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
                try:
                    if self._ws.socket.fileno() == -1:
                        raise ConnectionError("Socket closed")
                    return True
                except Exception as e:
                    logger.info("ASR WebSocket stale (%s), reconnecting", e)
                    try:
                        self._ws.close()
                    except Exception as close_err:
                        logger.debug("Stale WS close failed: %s", close_err)
                    self._ws = None
                    self._connected = False
                    # Fall through to reconnect

            try:
                self._ws = ws_connect(
                    self.url,
                    open_timeout=self.timeout,
                    close_timeout=5,
                    ping_interval=None,
                )
                self._connected = True
                self._last_partial = ""
                self._best_partial = ""
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
                except Exception as e:
                    logger.debug("WebSocket close failed: %s", e)
                self._ws = None
            self._connected = False
            self._last_partial = ""
            self._best_partial = ""

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

            # Check for partial transcript (non-blocking with short timeout)
            # 5ms timeout balances responsiveness with network latency tolerance
            try:
                response = self._ws.recv(timeout=0.005)
                data = json.loads(response)
                if data.get("type") == "partial":
                    self._last_partial = data.get("text", "")
                    # Track best partial by word count — RNNT models can
                    # degrade mid-stream (oscillation), so we keep the
                    # longest meaningful hypothesis as a fallback.
                    if len(self._last_partial.split()) > len(self._best_partial.split()):
                        self._best_partial = self._last_partial
                    return self._last_partial
            except TimeoutError:
                pass  # No data available yet

            return None
        except Exception as e:
            logger.warning("Error sending audio to streaming ASR: %s", e)
            self._connected = False
            try:
                self._ws.close()
            except Exception as close_err:
                logger.debug("WS close after send error: %s", close_err)
            self._ws = None
            # Reconnect so subsequent send_audio() calls don't all fail
            self.connect()
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

            # Drain partial responses until we get the final transcript.
            # The server may send buffered partials before the final result.
            for _ in range(10):
                response = self._ws.recv(timeout=self.timeout)
                data = json.loads(response)
                msg_type = data.get("type", "")

                if msg_type == "final":
                    text = data.get("text", "")
                    duration = data.get("duration_sec", 0)
                    # Guard against RNNT oscillation: if the final
                    # transcript has fewer words than the best streaming
                    # partial, the model degraded during finalization
                    # (e.g. "what time is it" → "with diamond").
                    # Prefer the best partial in that case.
                    best = self._best_partial
                    final_words = len(text.split()) if text else 0
                    best_words = len(best.split()) if best else 0
                    if best_words > final_words and best_words >= 3:
                        logger.warning(
                            "ASR oscillation detected: final '%s' (%d words) "
                            "< best partial '%s' (%d words), using best partial",
                            text[:50], final_words,
                            best[:50], best_words,
                        )
                        text = best
                    logger.info(
                        "Streaming ASR finalized: %.2fs -> '%s'",
                        duration,
                        text[:50] if text else "(empty)",
                    )
                    return text
                elif msg_type == "error":
                    logger.error("Streaming ASR error: %s", data.get("message"))
                    return None
                else:
                    # Partial or other non-final response, keep waiting
                    logger.debug("Draining %s response during finalize", msg_type)

            logger.warning("Never received final response after 10 reads")
            return None

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
            except Exception as e:
                logger.debug("Reset drain failed (expected): %s", e)
            self._last_partial = ""
            self._best_partial = ""
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
        self._warm_process: Optional[subprocess.Popen] = None
        self._warm_lock = threading.Lock()

    def _build_piper_cmd(self) -> list:
        """Build the Piper command line arguments."""
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
        return cmd

    def _spawn_process(self) -> subprocess.Popen:
        """Spawn a new Piper subprocess."""
        return subprocess.Popen(
            self._build_piper_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def warm_up(self):
        """Pre-spawn a Piper process so the next speak() call is fast.

        Thread-safe; no-op if a warm process is already alive.
        """
        with self._warm_lock:
            if self._warm_process is not None and self._warm_process.poll() is None:
                return  # Already warm and alive
            try:
                self._warm_process = self._spawn_process()
                logger.info("Piper warm process spawned (PID %d)", self._warm_process.pid)
            except Exception as e:
                logger.warning("Failed to spawn warm Piper process: %s", e)
                self._warm_process = None

    def _take_warm_process(self) -> Optional[subprocess.Popen]:
        """Atomically take the warm process if it's still alive."""
        with self._warm_lock:
            proc = self._warm_process
            if proc is not None:
                if proc.poll() is None:
                    self._warm_process = None
                    logger.info("Using warm Piper process (PID %d)", proc.pid)
                    return proc
                else:
                    logger.debug("Warm process already exited, discarding")
                    self._warm_process = None
        return None

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
        start_time = time.perf_counter()
        process = self._take_warm_process() or self._spawn_process()
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
            except Exception as e:
                logger.debug("Stream stop failed: %s", e)
        self.current_stream = None
        self._current_process = None

        if self.stop_event.is_set():
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                logger.warning("Piper did not terminate after stop, killing")
                process.kill()
                process.wait(timeout=1.0)
        else:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("Piper process timed out, killing")
                process.kill()
                process.wait(timeout=1.0)

        # Pre-spawn next warm process for subsequent calls
        threading.Thread(target=self.warm_up, daemon=True, name="piper-warmup").start()

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
                except Exception as e:
                    logger.debug("Batch stream stop failed: %s", e)
            self.current_stream = None
        except Exception as exc:
            logger.error("Piper batch synthesis failed: %s", exc)
        finally:
            try:
                os.remove(wav_path)
            except OSError as e:
                logger.debug("WAV temp file removal failed: %s", e)

    def stop(self):
        """Stop current playback and terminate Piper process."""
        self.stop_event.set()
        try:
            if self._current_process is not None:
                try:
                    self._current_process.terminate()
                except Exception as e:
                    logger.debug("Process terminate failed: %s", e)
            if self.current_stream is not None:
                try:
                    self.current_stream.abort()
                except Exception as e:
                    logger.debug("Stream abort failed: %s", e)
            sd.stop()
        except Exception as e:
            logger.debug("sd.stop() failed: %s", e)
        # Clean up warm process
        with self._warm_lock:
            if self._warm_process is not None:
                try:
                    self._warm_process.terminate()
                    self._warm_process.wait(timeout=1.0)
                except Exception as e:
                    logger.debug("Warm Piper cleanup failed: %s", e)
                self._warm_process = None


class VoicePipeline:
    """Main voice-to-voice pipeline."""

    def __init__(
        self,
        wakeword_model_paths: List[str],
        wake_threshold: float,
        asr_client: NemotronAsrHttpClient,
        tts: SpeechEngine,
        agent_runner: Callable[[str, Dict[str, Any]], str],
        streaming_agent_runner: Optional[Callable[[str, Dict[str, Any], Callable[[str], None]], None]] = None,
        streaming_llm_enabled: bool = False,
        sample_rate: int = 16000,
        block_size: int = 1280,
        silence_ms: int = 500,
        max_command_seconds: int = 5,
        min_command_ms: int = 1500,
        min_speech_frames: int = 3,
        wake_buffer_frames: int = 5,
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
        prefill_cache_ttl: float = 60.0,
        filler_enabled: bool = True,
        filler_delay_ms: int = 800,
        filler_phrases: Optional[List[str]] = None,
        filler_followup_delay_ms: int = 5000,
        filler_followup_phrases: Optional[List[str]] = None,
        debug_logging: bool = False,
        log_interval_frames: int = 160,
        conversation_mode_enabled: bool = False,
        conversation_timeout_ms: int = 8000,
        conversation_start_delay_ms: int = 500,
        conversation_speech_frames: int = 3,
        conversation_speech_tolerance: int = 2,
        conversation_rms_threshold: float = 0.002,
        speaker_id_enabled: bool = False,
        speaker_id_service: Optional[Any] = None,
        require_known_speaker: bool = False,
        unknown_speaker_response: str = "I don't recognize your voice.",
        error_asr_empty: str = "Sorry, I didn't catch that.",
        error_agent_timeout: str = "Sorry, that took too long. Try again.",
        error_agent_failed: str = "Something went wrong. Try again.",
        speaker_id_timeout: float = 5.0,
        node_id: str = "local",
        event_loop: Optional[Any] = None,
        # Voice filter config
        vad_backend: str = "webrtc",
        silero_threshold: float = 0.7,
        rms_min_threshold: float = 0.008,
        rms_adaptive: bool = False,
        rms_above_ambient_factor: float = 3.0,
        turn_limit_enabled: bool = True,
        max_conversation_turns: int = 3,
        intent_gating_enabled: bool = True,
        intent_continuation_threshold: float = 0.6,
        intent_categories_continue: Optional[List[str]] = None,
        speaker_continuity_enabled: bool = False,
        speaker_continuity_threshold: float = 0.7,
        # Conversation-mode recording thresholds
        conversation_silence_ms: int = 500,
        conversation_hangover_ms: int = 300,
        conversation_max_command_seconds: int = 30,
        # Sliding window segmentation for conversation mode
        conversation_window_frames: int = 20,
        conversation_silence_ratio: float = 0.15,
        conversation_asr_holdoff_ms: int = 1000,
        asr_quiet_limit: int = 5,
        # Workflow-aware thresholds
        workflow_silence_ms: int = 1500,
        workflow_hangover_ms: int = 500,
        workflow_max_command_seconds: int = 15,
        workflow_conversation_timeout_ms: int = 30000,
        # Wake confirmation tone
        wake_confirmation_enabled: bool = True,
        wake_confirmation_freq: int = 880,
        wake_confirmation_duration_ms: int = 80,
        # Turn limit warning
        conversation_turn_limit_phrase: str = "Say Hey Atlas to continue.",
        # Early preparation during conversation silence
        early_preparation_runner: Optional[Callable[[str, Optional[str]], None]] = None,
        conversation_early_silence_ms: int = 600,
    ):
        self.sample_rate = sample_rate
        self.event_loop = event_loop
        self.speaker_id_enabled = speaker_id_enabled
        self.speaker_id_service = speaker_id_service
        self.require_known_speaker = require_known_speaker
        self.unknown_speaker_response = unknown_speaker_response
        self.error_asr_empty = error_asr_empty
        self.error_agent_timeout = error_agent_timeout
        self.error_agent_failed = error_agent_failed
        self.speaker_id_timeout = speaker_id_timeout
        self._last_audio_buffer: Optional[bytes] = None
        self._last_speaker_match: Optional[Any] = None
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
        self._last_llm_call_time: float = 0.0
        self._wake_ts: float = 0.0          # perf_counter at last wake word detection
        self._prefill_cache_ttl = prefill_cache_ttl
        self._early_prep_runner = early_preparation_runner
        self._early_prep_in_progress = False
        self._conversation_early_silence_ms = conversation_early_silence_ms
        self._filler_enabled = filler_enabled
        self._filler_delay = filler_delay_ms / 1000.0
        self._filler_phrases = filler_phrases or [
            "Please hold.",
            "I'll get right on that, big guy.",
            "Yes sir.",
            "Be right back.",
            "Alright super chief.",
            "Here's what I got.",
            "Let me check on that.",
            "Just a sec.",
        ]
        self._filler_followup_delay = filler_followup_delay_ms / 1000.0
        self._filler_followup_phrases = filler_followup_phrases or [
            "Still working on that.",
            "Almost there.",
            "Hang tight.",
        ]
        # Turn limit warning phrase
        self._turn_limit_phrase = conversation_turn_limit_phrase

        # Wake confirmation tone (pre-generated at init)
        self._wake_tone = (
            _generate_tone(wake_confirmation_freq, wake_confirmation_duration_ms)
            if wake_confirmation_enabled else None
        )

        # Session ID stored as string for context passing, converted to UUID for database ops
        self.session_id = str(uuid.uuid4())
        self.node_id = node_id
        # Free conversation mode flag — set by FreeModeManager
        self._free_mode_active = False
        self.playback = PlaybackController(tts)
        # Monotonic counter incremented on each new command; checked before
        # speaking so that a slow command cannot overwrite a newer one's output.
        self._command_gen = 0
        self._command_gen_lock = threading.Lock()
        # Entity lock heartbeat (reasoning agent sovereignty)
        self._last_lock_heartbeat: float = 0.0

        self.segmenter = CommandSegmenter(
            sample_rate=self.sample_rate,
            block_size=self.block_size,
            silence_ms=silence_ms,
            hangover_ms=hangover_ms,
            max_command_seconds=max_command_seconds,
            min_command_ms=min_command_ms,
            min_speech_frames=min_speech_frames,
            # Match speech_threshold to the Silero VAD threshold so the
            # segmenter agrees with the VAD on what counts as speech.
            speech_threshold=silero_threshold,
            # Enable sliding window + ASR holdoff for wake-word mode too,
            # so silence detection works naturally instead of relying on a
            # hard time cap.
            window_frames=10,
            silence_ratio=0.15,
            asr_holdoff_ms=500,
        )

        # Conversation-mode recording thresholds
        self._conversation_silence_ms = conversation_silence_ms
        self._conversation_hangover_ms = conversation_hangover_ms
        self._conversation_max_command_seconds = conversation_max_command_seconds
        self._conversation_window_frames = conversation_window_frames
        self._conversation_silence_ratio = conversation_silence_ratio
        self._conversation_asr_holdoff_ms = conversation_asr_holdoff_ms

        # Workflow-aware thresholds
        self._workflow_active = False
        self._workflow_silence_ms = workflow_silence_ms
        self._workflow_hangover_ms = workflow_hangover_ms
        self._workflow_max_command_seconds = workflow_max_command_seconds
        self._workflow_conversation_timeout_ms = workflow_conversation_timeout_ms
        self._orig_silence_ms = silence_ms
        self._orig_hangover_ms = hangover_ms
        self._orig_max_command_seconds = max_command_seconds

        # Voice filter settings
        self.vad_backend = vad_backend
        self.rms_min_threshold = rms_min_threshold
        self.rms_adaptive = rms_adaptive
        self.rms_above_ambient_factor = rms_above_ambient_factor
        self.turn_limit_enabled = turn_limit_enabled
        self.max_conversation_turns = max_conversation_turns
        self.intent_gating_enabled = intent_gating_enabled
        self.intent_continuation_threshold = intent_continuation_threshold
        self.intent_categories_continue = intent_categories_continue or ["conversation", "tool_use", "device_command"]
        self.speaker_continuity_enabled = speaker_continuity_enabled
        self.speaker_continuity_threshold = speaker_continuity_threshold

        # Create VAD based on backend selection
        if vad_backend == "silero":
            logger.info("Using Silero VAD with threshold=%.2f", silero_threshold)
            self.vad = SileroVAD(threshold=silero_threshold)
        else:
            logger.info("Using WebRTC VAD with aggressiveness=%d", vad_aggressiveness)
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
            on_wake_detected=self._on_wake_detected,
            streaming_asr_client=streaming_client,
            debug_logging=debug_logging,
            log_interval_frames=log_interval_frames,
            conversation_mode_enabled=conversation_mode_enabled,
            conversation_timeout_ms=conversation_timeout_ms,
            conversation_speech_frames=conversation_speech_frames,
            conversation_speech_tolerance=conversation_speech_tolerance,
            conversation_rms_threshold=conversation_rms_threshold,
            on_conversation_timeout=self._on_conversation_timeout,
            # Voice filter settings
            rms_min_threshold=rms_min_threshold,
            rms_adaptive=rms_adaptive,
            rms_above_ambient_factor=rms_above_ambient_factor,
            turn_limit_enabled=turn_limit_enabled,
            max_conversation_turns=max_conversation_turns,
            speaker_continuity_enabled=speaker_continuity_enabled,
            speaker_continuity_threshold=speaker_continuity_threshold,
            on_turn_limit_reached=self._on_turn_limit_reached,
            on_asr_error=lambda: self._speak_error(self.error_asr_empty),
            conversation_silence_ms=conversation_silence_ms,
            conversation_hangover_ms=conversation_hangover_ms,
            conversation_max_command_seconds=conversation_max_command_seconds,
            conversation_window_frames=conversation_window_frames,
            conversation_silence_ratio=conversation_silence_ratio,
            conversation_asr_holdoff_ms=conversation_asr_holdoff_ms,
            wake_buffer_frames=wake_buffer_frames,
            asr_quiet_limit=asr_quiet_limit,
            on_early_silence=self._on_early_silence,
            conversation_early_silence_ms=conversation_early_silence_ms,
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
        self._ensure_session()
        # Set room in ContextAggregator from node config
        try:
            from ..orchestration.context import get_context
            get_context().set_room(self.node_id)
        except Exception as e:
            logger.debug("Could not set room context: %s", e)
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

    def _ensure_session(self):
        """Ensure voice pipeline session exists in database.

        Uses SessionRepository for proper session management:
        - Reuses today's active session for this node (terminal_id)
        - Creates new session via repository if none found
        - Updates self.session_id with the DB-backed session
        """
        if self.event_loop is None:
            logger.warning("No event loop, skipping session creation")
            return

        async def create_session():
            from ..storage.database import get_db_pool
            from ..storage.repositories.session import get_session_repo
            from datetime import date

            pool = get_db_pool()
            if not pool.is_initialized:
                logger.warning("Database pool not initialized, skipping session")
                return

            repo = get_session_repo()
            today = date.today()

            # Try to reuse today's active session for this voice node
            row = await pool.fetchrow(
                """SELECT id FROM sessions
                   WHERE terminal_id = $1 AND session_date = $2 AND is_active = true
                   LIMIT 1""",
                self.node_id,
                today,
            )

            if row:
                self.session_id = str(row["id"])
                await repo.touch_session(row["id"])
                logger.info("Resumed today's voice session: %s", self.session_id[:8])
                return

            # No existing session for today -- create via repository
            session = await repo.create_session(
                user_id=None,
                terminal_id=self.node_id,
                metadata={"source": "voice_pipeline", "node_id": self.node_id},
            )
            self.session_id = str(session.id)
            logger.info("Created voice session: %s", self.session_id[:8])

        try:
            import asyncio
            future = asyncio.run_coroutine_threadsafe(create_session(), self.event_loop)
            future.result(timeout=5.0)
        except Exception as e:
            logger.warning("Session creation failed: %s", e)

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
        # Heartbeat entity locks every ~30s to prevent expiry during long sessions
        now = time.monotonic()
        if now - self._last_lock_heartbeat > 30:
            self._last_lock_heartbeat = now
            if self.event_loop is not None:
                try:
                    from ..reasoning.lock_integration import heartbeat_voice_session
                    asyncio.run_coroutine_threadsafe(
                        heartbeat_voice_session(self.session_id), self.event_loop
                    )
                except Exception:
                    pass  # non-fatal; heartbeat_voice_session has its own guards

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

    def _next_command_gen(self) -> int:
        """Claim the next command generation number (thread-safe)."""
        with self._command_gen_lock:
            self._command_gen += 1
            return self._command_gen

    def _is_current_gen(self, gen: int) -> bool:
        """Return True if *gen* is still the latest command generation."""
        with self._command_gen_lock:
            return self._command_gen == gen

    @staticmethod
    def _is_error_phrase(text) -> bool:
        """Check if text is an ErrorPhrase marker (type-safe, no string comparison)."""
        return isinstance(text, ErrorPhrase)

    def _speak_error(self, phrase: str) -> None:
        """Speak an error recovery phrase.

        Uses _on_error_playback_done instead of _on_playback_done so that:
        - Wake word model is reset (ready for next detection)
        - Conversation mode resumes if it was active (user can retry)
        - Turn counter is NOT incremented (errors don't count as turns)
        """
        if not phrase:
            return
        try:
            self.playback.speak(
                phrase,
                on_start=self._on_playback_start,
                on_done=self._on_error_playback_done,
            )
        except Exception:
            logger.error("TTS error phrase failed", exc_info=True)

    def _run_agent_with_filler(self, transcript: str, context: dict) -> str:
        """Run agent_runner, speaking a filler phrase if it's slow.

        If the agent doesn't return within _filler_delay seconds,
        a random filler phrase is spoken via TTS. If still waiting
        after _filler_followup_delay, a second-tier filler is spoken.
        When the agent finishes, the caller's subsequent playback.speak()
        will naturally replace the filler.
        """
        if not self._filler_enabled:
            return self.agent_runner(transcript, context)

        result_event = threading.Event()
        result_holder = [None]
        exception_holder = [None]

        def agent_work():
            try:
                result_holder[0] = self.agent_runner(transcript, context)
            except Exception as e:
                exception_holder[0] = e
            result_event.set()

        thread = threading.Thread(target=agent_work, daemon=True, name="agent-work")
        thread.start()

        if not result_event.wait(timeout=self._filler_delay):
            # Agent is slow -- speak filler while waiting
            filler = random.choice(self._filler_phrases)
            logger.info("Agent slow (>%dms), speaking filler: %s",
                        int(self._filler_delay * 1000), filler)
            self.playback.speak(filler, on_start=self._on_playback_start)

            # Wait for follow-up filler delay
            if not result_event.wait(timeout=self._filler_followup_delay):
                followup = random.choice(self._filler_followup_phrases)
                logger.info("Agent still slow (>%dms), speaking follow-up filler: %s",
                            int((self._filler_delay + self._filler_followup_delay) * 1000), followup)
                self.playback.speak(followup, on_start=self._on_playback_start)
                result_event.wait()

        thread.join(timeout=2.0)

        if exception_holder[0] is not None:
            raise exception_holder[0]
        return result_holder[0]

    def _handle_command(self, pcm_bytes: bytes):
        """Handle a completed voice command."""
        gen = self._next_command_gen()

        # Store audio for speaker ID
        self._last_audio_buffer = pcm_bytes

        # Verify speaker if enabled
        if not self._verify_speaker(pcm_bytes):
            return

        # Use PCM directly if client supports it (avoids WAV encode/decode)
        if hasattr(self.asr_client, "transcribe_pcm"):
            transcript = self.asr_client.transcribe_pcm(pcm_bytes, self.sample_rate)
        else:
            wav_bytes = pcm_to_wav_bytes(pcm_bytes, self.sample_rate)
            transcript = self.asr_client.transcribe(wav_bytes, self.sample_rate)
        if not transcript:
            logger.warning("ASR returned empty transcript")
            self._speak_error(self.error_asr_empty)
            return
        logger.info("ASR: %s", transcript)
        _asr_ms = (time.perf_counter() - self._wake_ts) * 1000 if self._wake_ts else 0.0
        logger.info("Wake-to-ASR: %.0fms", _asr_ms)
        _log_transcript(transcript, source="batch", wake_to_asr_ms=_asr_ms)

        # Use streaming LLM if enabled
        if self.streaming_llm_enabled and self.streaming_agent_runner:
            self._handle_streaming_llm_command(transcript, gen)
            return

        context = self._build_context()
        reply = self._run_agent_with_filler(transcript, context)
        self._last_llm_call_time = time.monotonic()

        if not self._is_current_gen(gen):
            logger.info("Stale command gen=%d, discarding reply", gen)
            return
        if not reply:
            logger.warning("Agent returned empty reply")
            self._speak_error(self.error_agent_failed)
            return
        if self._is_error_phrase(reply):
            self._speak_error(reply)
            return
        logger.info("Speaking reply: %s", reply[:100] if len(reply) > 100 else reply)
        if self._wake_ts:
            logger.info(
                "Wake-to-TTS-call: %.0fms",
                (time.perf_counter() - self._wake_ts) * 1000,
            )
        self.playback.speak(
            reply,
            on_start=self._on_playback_start,
            on_done=self._on_playback_done,
        )

    def _handle_streaming_command(self, transcript: str, audio_bytes: bytes):
        """Handle a command with transcript from streaming ASR.

        Skips batch ASR transcription since we already have the transcript.

        Args:
            transcript: Final transcript from streaming ASR
            audio_bytes: Raw PCM audio bytes for speaker verification
        """
        gen = self._next_command_gen()
        logger.info("_handle_streaming_command called with transcript: %r", transcript)
        if not transcript:
            logger.warning("Empty transcript from streaming ASR")
            self._speak_error(self.error_asr_empty)
            return

        # Store audio and verify speaker
        self._last_audio_buffer = audio_bytes
        if not self._verify_speaker(audio_bytes):
            return

        logger.info("Streaming ASR: %s", transcript)
        _asr_ms = (time.perf_counter() - self._wake_ts) * 1000 if self._wake_ts else 0.0
        logger.info("Wake-to-ASR: %.0fms", _asr_ms)
        _log_transcript(transcript, source="streaming", wake_to_asr_ms=_asr_ms)

        # Use streaming LLM if enabled
        if self.streaming_llm_enabled and self.streaming_agent_runner:
            self._handle_streaming_llm_command(transcript, gen)
            return

        context = self._build_context()
        reply = self._run_agent_with_filler(transcript, context)
        self._last_llm_call_time = time.monotonic()

        if not self._is_current_gen(gen):
            logger.info("Stale command gen=%d, discarding reply", gen)
            return
        if not reply:
            logger.warning("Agent returned empty reply")
            self._speak_error(self.error_agent_failed)
            return
        if self._is_error_phrase(reply):
            self._speak_error(reply)
            return
        logger.info("Speaking reply: %s", reply[:100] if len(reply) > 100 else reply)
        if self._wake_ts:
            logger.info(
                "Wake-to-TTS-call: %.0fms",
                (time.perf_counter() - self._wake_ts) * 1000,
            )
        self.playback.speak(
            reply,
            on_start=self._on_playback_start,
            on_done=self._on_playback_done,
        )

    def _handle_streaming_llm_command(self, transcript: str, gen: int = 0):
        """Handle command with streaming LLM to TTS.

        Sentences are played as soon as they arrive from the LLM stream
        via a queue-based playback thread, so the user hears the first
        sentence immediately instead of waiting for the full response.

        If streaming fails and falls back to the regular agent, the
        fallback response is collected and spoken as a single utterance.
        """
        if not transcript:
            logger.warning("Empty transcript for streaming LLM")
            self._speak_error(self.error_asr_empty)
            return

        logger.info("Streaming LLM command: %s", transcript)
        context = self._build_context()

        sentence_queue: queue.Queue = queue.Queue()
        # Tracks whether streaming was reset (None sentinel from runner)
        # and we switched to collecting fallback output in a list.
        use_queue = [True]
        fallback_sentences: List[str] = []
        is_error = [False]
        sentence_count = [0]
        filler_timer = None
        followup_filler_timer = None

        def on_sentence(sentence):
            nonlocal filler_timer, followup_filler_timer
            # Cancel fillers on first real content
            if filler_timer is not None:
                filler_timer.cancel()
                filler_timer = None
            if followup_filler_timer is not None:
                followup_filler_timer.cancel()
                followup_filler_timer = None

            if sentence is None:
                # Streaming failed, switching to fallback agent.
                # Stop queue-based playback; fallback output goes to list.
                use_queue[0] = False
                self.playback.stop()
                return

            if isinstance(sentence, ErrorPhrase):
                is_error[0] = True

            sentence_count[0] += 1
            if sentence_count[0] == 1 and self._wake_ts:
                logger.info(
                    "Wake-to-first-sentence: %.0fms",
                    (time.perf_counter() - self._wake_ts) * 1000,
                )
            log_text = sentence[:80] if len(sentence) > 80 else sentence
            logger.info("Streaming LLM sentence %d: %s",
                        sentence_count[0], log_text)

            if use_queue[0]:
                sentence_queue.put(sentence)
            else:
                fallback_sentences.append(sentence)

        # Dynamic on_done: use error callback if only an ErrorPhrase came through
        def _on_done():
            if is_error[0]:
                self._on_error_playback_done()
            else:
                self._on_playback_done()

        # Start queue-based playback thread before the agent call
        self.playback.speak_streamed(
            sentence_queue,
            on_start=self._on_playback_start,
            on_done=_on_done,
        )

        # Filler: push to the queue so it plays naturally in sequence
        if self._filler_enabled:
            def _push_filler():
                nonlocal followup_filler_timer
                if use_queue[0] and sentence_count[0] == 0:
                    filler = random.choice(self._filler_phrases)
                    logger.info("Streaming LLM slow (>%dms), queueing filler: %s",
                                int(self._filler_delay * 1000), filler)
                    sentence_queue.put(filler)
                    # Schedule follow-up filler
                    def _push_followup():
                        if use_queue[0] and sentence_count[0] == 0:
                            followup = random.choice(self._filler_followup_phrases)
                            logger.info("Streaming LLM still slow, queueing follow-up: %s", followup)
                            sentence_queue.put(followup)
                    followup_filler_timer = threading.Timer(self._filler_followup_delay, _push_followup)
                    followup_filler_timer.daemon = True
                    followup_filler_timer.start()
            filler_timer = threading.Timer(self._filler_delay, _push_filler)
            filler_timer.daemon = True
            filler_timer.start()

        # Run the agent (blocks until all sentences are emitted)
        if self.streaming_agent_runner:
            self.streaming_agent_runner(transcript, context, on_sentence)
        else:
            reply = self.agent_runner(transcript, context)
            if reply:
                on_sentence(reply)

        # Clean up filler timers
        if filler_timer is not None:
            filler_timer.cancel()
        if followup_filler_timer is not None:
            followup_filler_timer.cancel()
        self._last_llm_call_time = time.monotonic()

        # Stale command check -- kill playback if superseded
        if gen and not self._is_current_gen(gen):
            logger.info("Stale streaming command gen=%d, discarding", gen)
            self.playback.stop()
            return

        if use_queue[0]:
            # Normal path: signal playback thread to finish
            if sentence_count[0] == 0:
                # No sentences at all -- stop playback, speak error
                self.playback.stop()
                self._speak_error(self.error_agent_failed)
            else:
                sentence_queue.put(None)  # Sentinel for clean completion
        else:
            # Fallback path: streaming was reset, speak fallback as one blob
            if fallback_sentences:
                full_reply = " ".join(fallback_sentences)
                if len(fallback_sentences) == 1 and self._is_error_phrase(fallback_sentences[0]):
                    self._speak_error(fallback_sentences[0])
                else:
                    logger.info("Speaking fallback reply: %s",
                                full_reply[:100] if len(full_reply) > 100 else full_reply)
                    self.playback.speak(
                        full_reply,
                        on_start=self._on_playback_start,
                        on_done=self._on_playback_done,
                    )
            else:
                self._speak_error(self.error_agent_failed)

    def _on_playback_start(self):
        """Called when TTS playback starts."""
        self.frame_processor.interrupt_speech_counter = 0
        # Pause conversation timer during TTS to prevent timeout during playback
        self.frame_processor.pause_conversation_mode()

    def _on_playback_done(self):
        """Called when TTS playback ends."""
        try:
            logger.info("TTS playback done, resetting wake word model")
            self.model.reset()
            logger.info("Wake word model reset after TTS")
        except Exception as e:
            logger.error("Error resetting wake word model after TTS: %s", e, exc_info=True)

        # Increment turn count and check if limit reached
        if self.conversation_mode_enabled and self.turn_limit_enabled:
            limit_reached = self.frame_processor.increment_turn_count()
            if limit_reached:
                # Turn limit reached - speak warning phrase, then exit
                if self._turn_limit_phrase:
                    def _exit_after_phrase():
                        self.frame_processor.exit_conversation_mode("turn_limit")
                    self.playback.speak(
                        self._turn_limit_phrase,
                        on_start=self._on_playback_start,
                        on_done=_exit_after_phrase,
                    )
                else:
                    self.frame_processor.exit_conversation_mode("turn_limit")
                return

        # Enter conversation mode if enabled (with delay to avoid echo detection)
        if self.conversation_mode_enabled:
            delay_sec = self.conversation_start_delay_ms / 1000.0
            logger.info("Scheduling conversation mode in %.0fms", self.conversation_start_delay_ms)
            timer = threading.Timer(delay_sec, self._enter_conversation_mode_delayed)
            timer.daemon = True
            timer.start()

    def _on_error_playback_done(self):
        """Called when error recovery TTS playback ends.

        Unlike _on_playback_done this does NOT increment the turn counter,
        because errors should not count as conversation turns.  It still
        resets the wake-word model and re-enters conversation mode (if
        enabled) so the user can immediately retry.
        """
        try:
            self.model.reset()
        except Exception as e:
            logger.error("Error resetting wake word model after error TTS: %s", e, exc_info=True)

        if self.conversation_mode_enabled:
            delay_sec = self.conversation_start_delay_ms / 1000.0
            timer = threading.Timer(delay_sec, self._enter_conversation_mode_delayed)
            timer.daemon = True
            timer.start()

    def _enter_conversation_mode_delayed(self):
        """Enter conversation mode after delay (called by timer)."""
        logger.info("_enter_conversation_mode_delayed fired, fp.state=%s", self.frame_processor.state)
        self.frame_processor.enter_conversation_mode()

    def _on_conversation_timeout(self):
        """Called when conversation mode times out."""
        logger.info("Conversation mode ended (timeout)")
        # Reset turn count when conversation ends
        self.frame_processor.reset_turn_count()
        # Free mode: immediately re-enter conversation mode so Atlas
        # keeps listening without requiring a wake word.
        if self._free_mode_active:
            logger.info("Free mode active — resuming conversation after timeout")
            delay_sec = self.conversation_start_delay_ms / 1000.0
            timer = threading.Timer(delay_sec, self._enter_conversation_mode_delayed)
            timer.daemon = True
            timer.start()

    def _on_turn_limit_reached(self):
        """Called when turn limit is reached in conversation mode."""
        logger.info("Conversation mode ended (turn limit reached)")
        # Reset turn count
        self.frame_processor.reset_turn_count()

    def _play_wake_confirmation(self):
        """Play a short confirmation tone when wake word is detected."""
        if self._wake_tone is None:
            return
        try:
            sd.play(self._wake_tone, 24000, blocking=False)
        except Exception as e:
            logger.debug("Wake confirmation tone failed: %s", e)

    def _on_wake_detected(self):
        """Called when wake word is detected. Only plays confirmation sound.

        Prefill is deferred until after intent routing so fast-path tool
        queries (get_time, get_weather, etc.) never touch the LLM.
        """
        self._wake_ts = time.perf_counter()
        self._play_wake_confirmation()

    def trigger_prefill(self):
        """Trigger LLM system prompt prefill in background.

        Called by the launcher AFTER intent routing confirms the query
        needs the conversation LLM. This avoids wasting GPU on prefill
        for fast-path tool queries.
        """
        if self.prefill_runner is None:
            logger.debug("No prefill_runner configured")
            return
        # Skip if KV cache is still warm from a recent LLM call
        elapsed = time.monotonic() - self._last_llm_call_time
        if self._last_llm_call_time > 0 and elapsed < self._prefill_cache_ttl:
            logger.info("Skipping prefill, KV cache warm (%.1fs ago)", elapsed)
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

    def _on_early_silence(self, partial: str):
        """Start background preparation when conversation silence is detected early."""
        # Start prefill (system prompt KV cache warming)
        self.trigger_prefill()
        # Start context gathering if runner provided
        if self._early_prep_runner and not self._early_prep_in_progress:
            self._early_prep_in_progress = True
            thread = threading.Thread(
                target=self._run_early_prep,
                args=(partial, self.session_id),
                daemon=True,
                name="early-prep",
            )
            thread.start()

    def _run_early_prep(self, partial: str, session_id: str):
        """Execute early preparation runner in background."""
        try:
            self._early_prep_runner(partial, session_id)
        except Exception as e:
            logger.warning("Early preparation failed: %s", e)
        finally:
            self._early_prep_in_progress = False

    def _verify_speaker(self, pcm_bytes: bytes) -> bool:
        """
        Verify speaker identity from audio.

        When require_known_speaker is False, runs verification in the
        background to enrich context without blocking the pipeline.

        Args:
            pcm_bytes: Raw PCM audio (int16, mono)

        Returns:
            True if speaker is allowed, False if rejected
        """
        if not self.speaker_id_enabled or self.speaker_id_service is None:
            return True

        if self.event_loop is None:
            logger.warning("No event loop for speaker verification, skipping")
            return True

        # Non-blocking path: run verification in background for context only
        if not self.require_known_speaker:
            self._run_speaker_id_background(pcm_bytes)
            return True

        # Blocking path: must verify speaker before allowing command
        return self._verify_speaker_blocking(pcm_bytes)

    def _verify_speaker_blocking(self, pcm_bytes: bytes) -> bool:
        """Block until speaker verification completes (require_known_speaker mode)."""
        try:
            import asyncio
            future = asyncio.run_coroutine_threadsafe(
                self.speaker_id_service.identify_speaker_from_pcm(
                    pcm_bytes, self.sample_rate
                ),
                self.event_loop,
            )
            match = future.result(timeout=self.speaker_id_timeout)
            self._last_speaker_match = match

            if match.matched:
                logger.info(
                    "Speaker identified: %s (confidence=%.2f)",
                    match.user_name, match.confidence
                )
                self._update_speaker_context(match)
                return True

            logger.warning(
                "Unknown speaker rejected (confidence=%.2f, threshold=%.2f)",
                match.confidence, self.speaker_id_service.threshold
            )
            self.playback.speak(
                self.unknown_speaker_response,
                on_start=self._on_playback_start,
                on_done=self._on_playback_done,
            )
            return False

        except Exception as e:
            logger.error("Speaker verification failed: %s", type(e).__name__)
            self._speak_error(self.error_agent_failed)
            return False

    def _run_speaker_id_background(self, pcm_bytes: bytes) -> None:
        """Run speaker identification in background thread for context enrichment."""
        def _identify():
            try:
                import asyncio
                future = asyncio.run_coroutine_threadsafe(
                    self.speaker_id_service.identify_speaker_from_pcm(
                        pcm_bytes, self.sample_rate
                    ),
                    self.event_loop,
                )
                match = future.result(timeout=self.speaker_id_timeout)
                self._last_speaker_match = match
                if match.matched:
                    logger.info(
                        "Speaker identified (bg): %s (confidence=%.2f)",
                        match.user_name, match.confidence
                    )
                    self._update_speaker_context(match)
                else:
                    logger.debug(
                        "Speaker not matched (bg): confidence=%.2f",
                        match.confidence
                    )
            except Exception as e:
                logger.debug("Background speaker ID failed: %s", type(e).__name__)

        thread = threading.Thread(target=_identify, daemon=True, name="speaker-id-bg")
        thread.start()

    def _update_speaker_context(self, match) -> None:
        """Update ContextAggregator with speaker identification result."""
        try:
            from ..orchestration.context import get_context
            ctx = get_context()
            person_id = str(match.user_id) if match.user_id else match.user_name
            ctx.update_person(
                person_id=person_id,
                name=match.user_name,
                location=self.node_id,
                confidence=match.confidence,
            )
        except Exception as e:
            logger.debug("Could not update context from speaker ID: %s", e)

        # Notify free mode evaluator so it refreshes its speaker heartbeat
        try:
            from ..voice.launcher import _free_mode_manager
            if _free_mode_manager is not None:
                _free_mode_manager.notify_speaker_confirmed(match.confidence)
        except Exception:
            pass

        # Acquire entity lock for reasoning agent sovereignty
        if match.user_id and self.session_id and self.event_loop:
            try:
                import asyncio
                from ..reasoning.lock_integration import on_voice_session_start
                asyncio.run_coroutine_threadsafe(
                    on_voice_session_start(
                        session_id=self.session_id,
                        speaker_id=str(match.user_id),
                        contact_id=str(match.user_id),
                    ),
                    self.event_loop,
                )
            except Exception:
                pass

    def _build_context(self) -> Dict[str, Any]:
        """Build context dict with session, node, and speaker info."""
        ctx = {
            "session_id": self.session_id,
            "node_id": self.node_id,
        }
        if self._last_speaker_match:
            ctx["speaker_name"] = self._last_speaker_match.user_name
            if self._last_speaker_match.user_id:
                ctx["speaker_id"] = str(self._last_speaker_match.user_id)
            ctx["speaker_confidence"] = self._last_speaker_match.confidence
        return ctx

    def enter_free_mode(self, timeout_ms: int = 30000) -> None:
        """Enter free conversation mode.

        - Disables turn limit so conversation never requires a wake word.
        - Extends conversation timeout to keep listening longer between turns.
        - Enters conversation mode immediately if currently in listening state.

        Called by FreeModeManager when entry conditions are met.
        """
        if self._free_mode_active:
            return
        if not self.conversation_mode_enabled:
            logger.warning("Free mode: conversation_mode_enabled=False, skipping")
            return
        logger.info("Free conversation mode: ACTIVATED (timeout=%dms)", timeout_ms)
        self._free_mode_active = True
        # Disable turn limit for the duration of free mode
        self.frame_processor.turn_limit_enabled = False
        # Extend conversation timeout
        self.frame_processor.set_conversation_timeout(timeout_ms)
        # Enter conversation mode immediately if currently idle
        if self.frame_processor.state == "listening":
            self.frame_processor.enter_conversation_mode()

    def exit_free_mode(self) -> None:
        """Exit free conversation mode and restore original settings.

        Called by FreeModeManager when conditions are no longer met
        (speaker absent, ambient noise too high, etc.).
        """
        if not self._free_mode_active:
            return
        logger.info("Free conversation mode: DEACTIVATED")
        self._free_mode_active = False
        # Restore original turn limit setting from config
        self.frame_processor.turn_limit_enabled = self.turn_limit_enabled
        # Restore original conversation timeout
        self.frame_processor.set_conversation_timeout(self.conversation_timeout_ms)
        # Exit conversation mode → back to wake-word listening
        if self.frame_processor.state == "conversing":
            self.frame_processor.exit_conversation_mode("free_mode_exit")

    def set_workflow_active(self) -> None:
        """Widen segmenter thresholds and conversation timeout for workflow mode."""
        if self._workflow_active:
            return
        self._workflow_active = True
        self.segmenter.update_thresholds(
            silence_ms=self._workflow_silence_ms,
            hangover_ms=self._workflow_hangover_ms,
            max_command_seconds=self._workflow_max_command_seconds,
        )
        self.frame_processor.set_conversation_timeout(self._workflow_conversation_timeout_ms)
        logger.info("Workflow mode activated: wider thresholds, conversation timeout=%dms",
                     self._workflow_conversation_timeout_ms)

    def clear_workflow_active(self) -> None:
        """Restore original segmenter thresholds and conversation timeout."""
        if not self._workflow_active:
            return
        self._workflow_active = False
        self.segmenter.update_thresholds(
            silence_ms=self._orig_silence_ms,
            hangover_ms=self._orig_hangover_ms,
            max_command_seconds=self._orig_max_command_seconds,
        )
        self.frame_processor.set_conversation_timeout(self.conversation_timeout_ms)
        logger.info("Workflow mode deactivated: normal thresholds")
