"""
Voice pipeline for Atlas Brain.

Main voice-to-voice pipeline that integrates:
- Wake word detection (OpenWakeWord)
- Voice Activity Detection (WebRTC VAD)
- Speech recognition (HTTP ASR)
- Atlas agent for response generation
- Text-to-speech (Piper)
"""

import io
import logging
import os
import subprocess
import tempfile
import threading
import uuid
import wave
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import webrtcvad
from openwakeword.model import Model as WakeWordModel

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


class PiperTTS:
    """Piper TTS engine for speech synthesis."""

    def __init__(
        self,
        binary_path: Optional[str],
        model_path: Optional[str],
        speaker: Optional[int] = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ):
        self.binary_path = binary_path
        self.model_path = model_path
        self.speaker = speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.stop_event = threading.Event()
        self.current_stream: Optional[sd.OutputStream] = None

    def speak(self, text: str):
        """Speak the given text using Piper TTS."""
        if not self.binary_path or not os.path.isfile(self.binary_path):
            logger.error("Piper binary not found: %s", self.binary_path)
            return
        if not self.model_path or not os.path.isfile(self.model_path):
            logger.error("Piper model not found: %s", self.model_path)
            return

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
            self.stop_event.clear()
            chunk = 2048
            with sd.OutputStream(
                samplerate=sr, channels=1, dtype="float32"
            ) as stream:
                self.current_stream = stream
                for start in range(0, len(audio), chunk):
                    if self.stop_event.is_set():
                        break
                    end = start + chunk
                    stream.write(audio[start:end])
                try:
                    stream.stop()
                except Exception:
                    pass
            self.current_stream = None
        except Exception as exc:
            logger.error("Piper synthesis failed: %s", exc)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    def stop(self):
        """Stop current playback."""
        self.stop_event.set()
        try:
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
        sample_rate: int = 16000,
        block_size: int = 1280,
        silence_ms: int = 500,
        max_command_seconds: int = 5,
        vad_aggressiveness: int = 2,
        hangover_ms: int = 300,
        use_arecord: bool = False,
        arecord_device: str = "default",
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
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.wake_threshold = wake_threshold
        self.asr_client = asr_client
        self.agent_runner = agent_runner
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
        )
        self.vad = webrtcvad.Vad(vad_aggressiveness)

        self.model = WakeWordModel(wakeword_model_paths=wakeword_model_paths)
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
        )

        self.capture = AudioCapture(
            sample_rate=self.sample_rate,
            block_size=self.block_size,
            use_arecord=use_arecord,
            arecord_device=arecord_device,
        )

        self.command_executor = CommandExecutor(
            handler=self._handle_command,
            max_workers=command_workers,
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
        )

    def _stop_playback(self):
        """Stop TTS playback and reset state."""
        self.playback.stop()
        self.frame_processor.reset()
        self.current_allow_barge_in = True
        self.current_response_metadata = {}

    def _handle_command(self, pcm_bytes: bytes):
        """Handle a completed voice command."""
        wav_bytes = pcm_to_wav_bytes(pcm_bytes, self.sample_rate)
        transcript = self.asr_client.transcribe(wav_bytes, self.sample_rate)
        if not transcript:
            return
        logger.info("ASR: %s", transcript)

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

    def _on_playback_start(self):
        """Called when TTS playback starts."""
        self.frame_processor.interrupt_speech_counter = 0

    def _on_playback_done(self):
        """Called when TTS playback ends."""
        self.model.reset()
        logger.info("TTS done, wake word model reset")

    def _trigger_prefill(self):
        """Trigger LLM system prompt prefill in background.

        Called when wake word is detected to warm up the LLM KV cache
        while ASR is still recording. This reduces time-to-first-token
        when the actual request is made.
        """
        if self.prefill_runner is None:
            return
        # Guard against concurrent prefills
        if self._prefill_in_progress:
            return
        self._prefill_in_progress = True
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
