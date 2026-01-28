"""
Audio capture module for voice pipeline.

Handles microphone input via PortAudio or arecord.
"""

import logging
import subprocess
import time
from typing import Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger("atlas.voice.audio_capture")


class AudioCapture:
    """Handles microphone capture via PortAudio or arecord."""

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        use_arecord: bool = False,
        arecord_device: str = "default",
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.use_arecord = use_arecord
        self.arecord_device = arecord_device or "default"
        self._frame_count = 0

    def run(self, on_frame: Callable[[bytes], None]):
        """Start capturing audio and call on_frame for each block."""
        if self.use_arecord:
            logger.info("Using arecord input device: %s", self.arecord_device)
            self._run_arecord(on_frame)
        else:
            logger.info("Using PortAudio for audio capture")
            self._run_portaudio(on_frame)

    def _run_portaudio(self, on_frame: Callable[[bytes], None]):
        """Capture audio using PortAudio/sounddevice."""
        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("Input status: %s", status)
            try:
                mono = np.array(indata[:, 0], dtype=np.int16)
                frame_bytes = mono.tobytes()
                self._frame_count += 1
                if self._frame_count == 1:
                    logger.info(
                        "AudioCapture: First frame received (%d bytes)",
                        len(frame_bytes),
                    )
                if self._frame_count % 160 == 0:
                    rms = float(
                        np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2))
                    )
                    logger.info(
                        "AudioCapture: frames=%d rms=%.4f",
                        self._frame_count,
                        rms,
                    )
                on_frame(frame_bytes)
            except Exception:
                logger.exception("Audio callback error.")

        with sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Stopping listener.")

    def _run_arecord(self, on_frame: Callable[[bytes], None]):
        """Capture audio using arecord (ALSA)."""
        block_bytes = self.block_size * 2  # int16 mono
        cmd = [
            "arecord",
            "-q",
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-c",
            "1",
            "-r",
            str(self.sample_rate),
            "-D",
            self.arecord_device,
        ]
        logger.info("Starting arecord capture: %s", " ".join(cmd))
        try:
            with subprocess.Popen(
                cmd, stdout=subprocess.PIPE, bufsize=block_bytes * 4
            ) as proc:
                while True:
                    frame_bytes = proc.stdout.read(block_bytes)
                    if not frame_bytes or len(frame_bytes) < block_bytes:
                        break
                    on_frame(frame_bytes)
        except KeyboardInterrupt:
            logger.info("Stopping arecord capture.")
        except Exception:
            logger.exception("arecord capture failed.")
