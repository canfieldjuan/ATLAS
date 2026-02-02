"""
Audio capture module for voice pipeline.

Handles microphone input via PortAudio or arecord.
"""

import logging
import os
import subprocess
import tempfile
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
        input_device: str | None = None,
        debug_logging: bool = False,
        log_interval_frames: int = 160,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.use_arecord = use_arecord
        self.arecord_device = arecord_device or "default"
        self.input_device = input_device
        self.debug_logging = debug_logging
        self.log_interval_frames = max(1, log_interval_frames)
        self._frame_count = 0
        self._max_rms_seen = 0.0
        self._min_rms_seen = 1.0

    def run(self, on_frame: Callable[[bytes], None]):
        """Start capturing audio and call on_frame for each block."""
        logger.info("=== AudioCapture Starting ===")
        logger.info("  sample_rate=%d, block_size=%d", self.sample_rate, self.block_size)
        logger.info("  use_arecord=%s", self.use_arecord)
        logger.info("  debug_logging=%s, log_interval=%d frames",
                    self.debug_logging, self.log_interval_frames)

        if self.use_arecord:
            logger.info("  arecord_device=%s", self.arecord_device)
            self._run_arecord(on_frame)
        else:
            logger.info("  input_device=%s (PortAudio)", self.input_device)
            self._run_portaudio(on_frame)

    def _run_portaudio(self, on_frame: Callable[[bytes], None]):
        """Capture audio using PortAudio/sounddevice."""
        # Log available devices for debugging
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            logger.info("PortAudio default input device index: %s", default_input)
            if self.debug_logging:
                for i, d in enumerate(devices):
                    if d.get("max_input_channels", 0) > 0:
                        logger.info("  [%d] %s (in=%d)",
                                    i, d.get("name", "?"), d.get("max_input_channels", 0))
        except Exception as e:
            logger.warning("Could not query audio devices: %s", e)

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("PortAudio callback status: %s", status)
            try:
                mono = np.array(indata[:, 0], dtype=np.int16)
                frame_bytes = mono.tobytes()
                self._frame_count += 1

                # Only calculate RMS when logging (first frame or periodic)
                should_log = (
                    self._frame_count == 1 or
                    self._frame_count % self.log_interval_frames == 0
                )
                if should_log:
                    rms = float(np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2)))
                    self._max_rms_seen = max(self._max_rms_seen, rms)
                    if rms > 0:
                        self._min_rms_seen = min(self._min_rms_seen, rms)
                    if self._frame_count == 1:
                        logger.info(
                            "AudioCapture: First frame (%d bytes), rms=%.6f",
                            len(frame_bytes), rms
                        )
                    else:
                        logger.info(
                            "AudioCapture: frames=%d rms=%.6f (min=%.6f max=%.6f)",
                            self._frame_count, rms,
                            self._min_rms_seen, self._max_rms_seen,
                        )

                on_frame(frame_bytes)
            except Exception:
                logger.exception("Audio callback error.")

        device = self.input_device
        if isinstance(device, str) and device.isdigit():
            device = int(device)

        # Log actual device being used
        try:
            if device is not None:
                dev_info = sd.query_devices(device)
                logger.info("Opening PortAudio device %s: %s", device, dev_info.get("name", "?"))
            else:
                logger.info("Opening PortAudio default input device")
        except Exception as e:
            logger.warning("Could not query device %s: %s", device, e)

        with sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="int16",
            channels=1,
            device=device,
            callback=callback,
        ):
            logger.info("PortAudio stream opened successfully")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Stopping listener.")

    def _run_arecord(self, on_frame: Callable[[bytes], None]):
        """Capture audio using arecord (ALSA) via FIFO.

        Uses a named pipe (FIFO) instead of subprocess.PIPE because
        PipeWire's ALSA plugin has buffering issues with anonymous pipes.
        """
        block_bytes = self.block_size * 2  # int16 mono

        # Create FIFO in temp directory with unique name per process
        fifo_path = os.path.join(
            tempfile.gettempdir(),
            "atlas_audio_fifo_{}".format(os.getpid())
        )
        try:
            os.unlink(fifo_path)
        except FileNotFoundError:
            pass
        os.mkfifo(fifo_path)

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
            fifo_path,
        ]
        logger.info("Starting arecord capture: %s", " ".join(cmd))

        proc = None
        fifo_fd = None
        try:
            # Start arecord writing to FIFO
            proc = subprocess.Popen(cmd)

            # Open FIFO for reading (blocks until arecord opens for write)
            fifo_fd = os.open(fifo_path, os.O_RDONLY)

            while True:
                frame_bytes = os.read(fifo_fd, block_bytes)
                if not frame_bytes or len(frame_bytes) < block_bytes:
                    if proc.poll() is not None:
                        break
                    continue
                self._frame_count += 1
                if self._frame_count == 1:
                    logger.info(
                        "AudioCapture: First frame received (%d bytes)",
                        len(frame_bytes),
                    )
                if self._frame_count % self.log_interval_frames == 0:
                    mono = np.frombuffer(frame_bytes, dtype=np.int16)
                    rms = float(
                        np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2))
                    )
                    logger.info(
                        "AudioCapture: frames=%d rms=%.4f",
                        self._frame_count,
                        rms,
                    )
                on_frame(frame_bytes)

        except KeyboardInterrupt:
            logger.info("Stopping arecord capture.")
        except Exception:
            logger.exception("arecord capture failed.")
        finally:
            if fifo_fd is not None:
                try:
                    os.close(fifo_fd)
                except Exception as e:
                    logger.debug("FIFO fd close failed: %s", e)
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    logger.warning("arecord did not terminate, killing")
                    proc.kill()
                    proc.wait(timeout=1.0)
            try:
                os.unlink(fifo_path)
            except Exception as e:
                logger.debug("FIFO unlink failed: %s", e)
