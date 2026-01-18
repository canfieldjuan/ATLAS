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
import scipy.signal
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
    gain: float = 1.0  # Software gain multiplier (1.0 = no boost)

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

        # Process audio (Resample + Gain + Convert)
        try:
            # 1. Convert to float32 for processing
            audio_float = indata.flatten().astype(np.float32)

            # 2. Resample if needed
            if getattr(self, '_resample_needed', False):
                # Calculate target length based on configured rate (likely 16000)
                # self.config.frame_size is the target size
                audio_float = scipy.signal.resample(audio_float, self.config.frame_size)

            # 3. Apply Gain
            if self.config.gain != 1.0:
                audio_float = audio_float * self.config.gain

            # 4. Clip and convert to int16 bytes
            # Ensure we are saving as valid int16
            audio_clipped = np.clip(audio_float, -32768, 32767)
            audio_bytes = audio_clipped.astype(np.int16).tobytes()

            self._audio_queue.put_nowait(audio_bytes)
        except queue.Full:
            logger.warning("Audio queue full, dropping frame")
        except Exception as e:
            logger.error(f"Audio processing error: {e}")

    def start(self) -> None:
        """Start audio capture."""
        if self._running:
            return

        self._running = True
        
        # Determine capture rate (auto-detect or use config)
        capture_rate = self.config.sample_rate
        self._resample_needed = False
        
        # Try configured settings first
        try:
            sd.check_input_settings(
                device=self.config.device,
                channels=self.config.channels,
                dtype=self.config.dtype,
                samplerate=capture_rate
            )
        except Exception:
            # Fallback to device default rate
            try:
                # If device is None (default), query default device
                dev_id = self.config.device
                if dev_id is None:
                    dev_id = sd.default.device[0]
                
                dev_info = sd.query_devices(dev_id, 'input')
                native_rate = int(dev_info['default_samplerate'])
                
                if native_rate != capture_rate:
                    logger.warning(
                        "Requested rate %d Hz failed. Using native %d Hz with real-time resampling.", 
                        capture_rate, native_rate
                    )
                    capture_rate = native_rate
                    self._resample_needed = True
            except Exception as e:
                logger.error("Failed to query device info: %s. Trying fallback to 48000Hz.", e)
                capture_rate = 48000
                self._resample_needed = True

        # Calculate blocksize for the actual capture rate
        # We want the duration to match the target frame_duration_ms
        capture_frame_size = int(capture_rate * self.config.frame_duration_ms / 1000)

        self._stream = sd.InputStream(
            samplerate=capture_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=capture_frame_size,
            device=self.config.device,
            callback=self._audio_callback,
        )
        self._stream.start()

        logger.info(
            "Audio capture started (rate=%d, resample=%s, device=%s)",
            capture_rate,
            self._resample_needed,
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
