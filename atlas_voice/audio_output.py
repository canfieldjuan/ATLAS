"""
Audio output module using sounddevice.

Provides speaker output for TTS audio playback.
"""

import io
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger("atlas.voice.output")


@dataclass
class OutputConfig:
    """Configuration for audio output."""

    device: Optional[int] = None  # None = default device


class AudioOutput:
    """
    Audio output for TTS playback.

    Handles playing WAV audio through speakers.
    """

    def __init__(self, config: Optional[OutputConfig] = None):
        self.config = config or OutputConfig()
        self._playing = False

    def play(self, audio_bytes: bytes, blocking: bool = True) -> None:
        """
        Play audio bytes through speakers.

        Args:
            audio_bytes: WAV-formatted audio data
            blocking: If True, wait for playback to complete
        """
        if not audio_bytes:
            return

        try:
            # Read WAV data
            buffer = io.BytesIO(audio_bytes)
            data, samplerate = sf.read(buffer)

            # Play through sounddevice
            self._playing = True
            sd.play(data, samplerate, device=self.config.device)

            if blocking:
                sd.wait()
                self._playing = False

            logger.debug("Played %d bytes at %d Hz", len(audio_bytes), samplerate)

        except Exception as e:
            logger.error("Audio playback error: %s", e)
            self._playing = False

    def stop(self) -> None:
        """Stop any current playback."""
        sd.stop()
        self._playing = False

    @property
    def is_playing(self) -> bool:
        """Whether audio is currently playing."""
        return self._playing


def list_devices() -> list[dict]:
    """List available audio output devices."""
    devices = sd.query_devices()
    output_devices = []

    for i, d in enumerate(devices):
        if d["max_output_channels"] > 0:
            output_devices.append({
                "index": i,
                "name": d["name"],
                "channels": d["max_output_channels"],
                "default_samplerate": d["default_samplerate"],
                "is_default": i == sd.default.device[1],
            })

    return output_devices


def get_default_device() -> Optional[int]:
    """Get the default output device index."""
    return sd.default.device[1]
