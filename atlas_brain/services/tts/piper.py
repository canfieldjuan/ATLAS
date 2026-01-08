"""
TTS implementation using Piper.

Piper is a fast, local text-to-speech system that produces
natural-sounding speech. Runs efficiently on CPU.
"""

import io
import subprocess
import wave
from pathlib import Path
from typing import Any, Optional

from ..base import BaseModelService, InferenceTimer
from ..protocols import ModelInfo
from ..registry import register_tts


@register_tts("piper")
class PiperTTS(BaseModelService):
    """
    TTS implementation using Piper.

    Uses the piper-tts CLI or piper_phonemize library.
    """

    CAPABILITIES = ["tts"]

    # Default voice model
    DEFAULT_VOICE = "en_US-amy-medium"

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        model_path: Optional[Path] = None,
        cache_path: Optional[Path] = None,
        sample_rate: int = 22050,
    ):
        super().__init__(
            name="piper",
            model_id=voice,
            cache_path=cache_path or Path("models/piper"),
            log_file=Path("logs/atlas_tts.log"),
        )
        self._voice = voice
        self._model_path = model_path
        self._sample_rate = sample_rate
        self._piper_available = False
        self._use_cli = False

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            model_id=self.model_id,
            is_loaded=self.is_loaded,
            device="cpu",  # Piper runs on CPU
            capabilities=self.CAPABILITIES,
        )

    @property
    def is_loaded(self) -> bool:
        return self._piper_available

    def load(self) -> None:
        """Check for Piper availability."""
        if self._piper_available:
            self.logger.info("Piper already loaded")
            return

        # Try CLI first
        try:
            result = subprocess.run(
                ["piper", "--help"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._piper_available = True
                self._use_cli = True
                self.logger.info("Piper CLI available")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try Python library
        try:
            import piper  # noqa: F401
            self._piper_available = True
            self._use_cli = False
            self.logger.info("Piper Python library available")
            return
        except ImportError:
            pass

        self.logger.warning(
            "Piper not available. Install with: pip install piper-tts "
            "or download from https://github.com/rhasspy/piper"
        )

    def unload(self) -> None:
        """Unload Piper resources."""
        self._piper_available = False
        self.logger.info("Piper unloaded")

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> bytes:
        """
        Convert text to speech.

        Args:
            text: Text to synthesize
            voice: Voice model to use (optional, uses default)
            speed: Speech speed multiplier

        Returns:
            WAV audio bytes
        """
        if not self._piper_available:
            raise RuntimeError("Piper not available. Call load() first.")

        voice = voice or self._voice
        self.logger.info("Synthesizing: '%s...' (voice=%s)", text[:50], voice)

        with InferenceTimer() as timer:
            if self._use_cli:
                audio_bytes = await self._synthesize_cli(text, voice, speed)
            else:
                audio_bytes = await self._synthesize_python(text, voice, speed)

        metrics = self.gather_metrics(timer.duration)
        self.logger.info(
            "Synthesized %d bytes in %.0fms",
            len(audio_bytes),
            metrics.duration_ms,
        )

        return audio_bytes

    async def _synthesize_cli(
        self,
        text: str,
        voice: str,
        speed: float,
    ) -> bytes:
        """Synthesize using Piper CLI."""
        import asyncio

        # Try to find model path
        model_path = self._find_model_path(voice)

        # Build command
        cmd = [
            "piper",
            "--model", model_path,
            "--output_file", "-",  # Output to stdout
        ]

        if speed != 1.0:
            cmd.extend(["--length_scale", str(1.0 / speed)])

        # Run Piper
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate(input=text.encode())

        if proc.returncode != 0:
            raise RuntimeError(f"Piper failed: {stderr.decode()}")

        return stdout

    def _find_model_path(self, voice: str) -> str:
        """Find the path to a voice model."""
        # Check if voice is already a path
        if Path(voice).exists():
            return voice

        # Check in cache path
        model_file = self.cache_path / f"{voice}.onnx"
        if model_file.exists():
            return str(model_file)

        # Check common locations
        common_paths = [
            Path.home() / ".local/share/piper/voices" / f"{voice}.onnx",
            Path("/usr/share/piper/voices") / f"{voice}.onnx",
            Path("models/piper") / f"{voice}.onnx",
        ]

        for path in common_paths:
            if path.exists():
                return str(path)

        # Return voice name for piper to resolve
        return voice

    async def _synthesize_python(
        self,
        text: str,
        voice: str,
        speed: float,
    ) -> bytes:
        """Synthesize using Piper Python library."""
        import asyncio

        # Find model path first
        model_path = self._find_model_path(voice)

        def _synthesize():
            try:
                from piper import PiperVoice

                # Load voice
                if not Path(model_path).exists():
                    raise FileNotFoundError(f"Voice model not found: {model_path}")

                piper_voice = PiperVoice.load(model_path)

                # Synthesize
                audio_data = []
                for audio_bytes in piper_voice.synthesize_stream_raw(text):
                    audio_data.append(audio_bytes)

                # Combine and convert to WAV
                raw_audio = b"".join(audio_data)
                return self._raw_to_wav(raw_audio, piper_voice.config.sample_rate)

            except ImportError:
                raise RuntimeError(
                    "piper-tts not installed. Install with: pip install piper-tts"
                )

        return await asyncio.get_event_loop().run_in_executor(None, _synthesize)

    def _raw_to_wav(self, raw_audio: bytes, sample_rate: int) -> bytes:
        """Convert raw PCM to WAV format."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_audio)
        return wav_buffer.getvalue()


@register_tts("espeak")
class ESpeakTTS(BaseModelService):
    """
    Fallback TTS using eSpeak-ng.

    Lower quality but universally available on Linux.
    """

    CAPABILITIES = ["tts"]

    def __init__(self):
        super().__init__(
            name="espeak",
            model_id="espeak-ng",
            cache_path=Path("models/tts"),
            log_file=Path("logs/atlas_tts.log"),
        )
        self._available = False

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            model_id=self.model_id,
            is_loaded=self._available,
            device="cpu",
            capabilities=self.CAPABILITIES,
        )

    @property
    def is_loaded(self) -> bool:
        return self._available

    def load(self) -> None:
        """Check for eSpeak availability."""
        try:
            result = subprocess.run(
                ["espeak-ng", "--version"],
                capture_output=True,
                timeout=5,
            )
            self._available = result.returncode == 0
            if self._available:
                self.logger.info("eSpeak-ng available")
            else:
                self.logger.warning("eSpeak-ng not available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.logger.warning(
                "eSpeak-ng not found. Install with: sudo apt install espeak-ng"
            )

    def unload(self) -> None:
        """Unload (no-op for eSpeak)."""
        self._available = False

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> bytes:
        """Synthesize using eSpeak-ng."""
        if not self._available:
            raise RuntimeError("eSpeak-ng not available")

        import asyncio

        # Words per minute (default ~175)
        wpm = int(175 * speed)

        cmd = [
            "espeak-ng",
            "-v", voice or "en",
            "-s", str(wpm),
            "--stdout",
            text,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"eSpeak failed: {stderr.decode()}")

        return stdout
