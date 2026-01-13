"""
Audio capture module using sounddevice.

Provides continuous microphone input with configurable sample rate
and frame size, suitable for streaming to the Atlas Brain WebSocket.
"""

import logging
import queue
import struct
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger("atlas.voice.capture")


@dataclass
class CaptureConfig:
    """Configuration for audio capture."""

    sample_rate: int = 16000  # Hz, required for wake word detection
    channels: int = 1  # Mono
    dtype: str = "int16"
    frame_duration_ms: int = 30  # Frame size for VAD compatibility
    device: Optional[int] = None  # None = default device

    @property
    def frame_size(self) -> int:
        """Samples per frame."""
        return int(self.sample_rate * self.frame_duration_ms / 1000)

    @property
    def frame_bytes(self) -> int:
        """Bytes per frame (16-bit = 2 bytes per sample)."""
        return self.frame_size * 2


class AudioCapture:
    """
    Continuous audio capture from microphone.

    Captures audio in small frames suitable for streaming to
    wake word detection and VAD processing.
    """

    def __init__(self, config: Optional[CaptureConfig] = None):
        self.config = config or CaptureConfig()
        self._stream: Optional[sd.InputStream] = None
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._running = False

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ):
        """Callback for audio stream - runs in separate thread."""
        if status:
            logger.warning("Audio capture status: %s", status)

        # Convert to bytes and queue
        audio_bytes = indata.tobytes()
        try:
            self._audio_queue.put_nowait(audio_bytes)
        except queue.Full:
            logger.warning("Audio queue full, dropping frame")

    def start(self) -> None:
        """Start audio capture."""
        if self._running:
            return

        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.frame_size,
            device=self.config.device,
            callback=self._audio_callback,
        )
        self._stream.start()

        logger.info(
            "Audio capture started (rate=%d, device=%s)",
            self.config.sample_rate,
            self.config.device or "default",
        )

    def stop(self) -> None:
        """Stop audio capture."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Audio capture stopped")

    def get_frame(self, timeout: float = 0.1) -> Optional[bytes]:
        """
        Get next audio frame.

        Args:
            timeout: Max seconds to wait for frame

        Returns:
            Audio bytes or None if no frame available
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        """Whether capture is active."""
        return self._running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def list_devices() -> list[dict]:
    """List available audio input devices."""
    devices = sd.query_devices()
    input_devices = []

    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            input_devices.append({
                "index": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "default_samplerate": d["default_samplerate"],
                "is_default": i == sd.default.device[0],
            })

    return input_devices


def get_default_device() -> Optional[int]:
    """Get the default input device index."""
    return sd.default.device[0]
