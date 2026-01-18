"""
Webcam camera implementation using OpenCV.

Captures frames from local USB/built-in webcams.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from ...core.constants import DeviceStatus
from ...core.protocols import Detection, MotionEvent
from ...processing.frame_buffer import get_frame_buffer
from .base import BaseCameraCapability

logger = logging.getLogger("atlas.vision.devices.cameras.webcam")


class WebcamCamera(BaseCameraCapability):
    """Webcam camera that captures frames via OpenCV."""

    def __init__(
        self,
        device_id: str,
        name: str,
        location: str,
        device_index: int = 0,
        fps: int = 15,
        width: int = 640,
        height: int = 480,
    ):
        """
        Initialize webcam camera.

        Args:
            device_id: Unique identifier for this camera
            name: Human-readable name
            location: Room/location name
            device_index: Video device index (0 = /dev/video0, 1 = /dev/video1, etc.)
            fps: Target frames per second
            width: Capture width
            height: Capture height
        """
        super().__init__(device_id, name, location)
        self.device_index = device_index
        self.fps = fps
        self.width = width
        self.height = height

        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0.0
        self._frame_count: int = 0
        self._motion_detector = None
        self._current_detections: list[Detection] = []

    async def connect(self) -> bool:
        """Connect to webcam and start capture loop."""
        if self._running:
            logger.warning("Camera %s already running", self.device_id)
            return True

        try:
            # Open capture in thread to avoid blocking
            self._capture = await asyncio.to_thread(
                self._create_capture, self.device_index
            )

            if not self._capture.isOpened():
                logger.error(
                    "Failed to open webcam at /dev/video%d", self.device_index
                )
                self._status = DeviceStatus.ERROR
                return False

            self._status = DeviceStatus.ONLINE
            self._running = True

            # Start capture loop
            self._capture_task = asyncio.create_task(self._capture_loop())
            logger.info(
                "Connected to webcam %s: /dev/video%d (%dx%d @ %d fps)",
                self.device_id,
                self.device_index,
                self.width,
                self.height,
                self.fps,
            )
            return True

        except Exception as e:
            logger.error("Failed to connect to webcam %s: %s", self.device_id, e)
            self._status = DeviceStatus.ERROR
            return False

    def _create_capture(self, device_index: int) -> cv2.VideoCapture:
        """Create OpenCV capture for webcam (runs in thread)."""
        cap = cv2.VideoCapture(device_index)

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Set buffer size to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Try to set FPS (may not be supported by all webcams)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        return cap

    async def disconnect(self) -> None:
        """Disconnect from webcam."""
        self._running = False

        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None

        if self._capture:
            await asyncio.to_thread(self._capture.release)
            self._capture = None

        self._status = DeviceStatus.OFFLINE
        logger.info("Disconnected from webcam %s", self.device_id)

    async def _capture_loop(self) -> None:
        """Main capture loop - runs continuously."""
        frame_interval = 1.0 / self.fps
        frame_buffer = get_frame_buffer()

        while self._running:
            try:
                start_time = time.time()

                # Read frame in thread
                ret, frame = await asyncio.to_thread(self._read_frame)

                if not ret or frame is None:
                    logger.warning(
                        "Failed to read frame from webcam %s", self.device_id
                    )
                    await asyncio.sleep(0.1)
                    continue

                # Store frame
                self._last_frame = frame
                self._last_frame_time = time.time()
                self._frame_count += 1

                # Push to frame buffer for processing
                await frame_buffer.push(self.device_id, frame)

                # Run detection if detector attached
                if self._motion_detector:
                    await self._run_detection(frame)

                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Capture error on webcam %s: %s", self.device_id, e)
                await asyncio.sleep(1.0)

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a single frame (runs in thread)."""
        if self._capture is None:
            return False, None
        return self._capture.read()

    async def _run_detection(self, frame: np.ndarray) -> None:
        """Run detection on frame."""
        try:
            detections = await asyncio.to_thread(
                self._motion_detector.detect, frame, self.device_id
            )
            if detections:
                self._last_motion = datetime.now()
                self._current_detections = detections
        except Exception as e:
            logger.warning("Detection error on webcam %s: %s", self.device_id, e)

    def set_motion_detector(self, detector) -> None:
        """Attach a detector to this camera."""
        self._motion_detector = detector

    async def get_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame."""
        return self._last_frame

    async def get_current_detections(self) -> list[Detection]:
        """Get current detections."""
        return self._current_detections

    async def get_motion_events(
        self, since: Optional[datetime] = None
    ) -> list[MotionEvent]:
        """Get motion events."""
        events = []
        if self._last_motion and (since is None or self._last_motion > since):
            events.append(
                MotionEvent(
                    camera_id=self.device_id,
                    timestamp=self._last_motion,
                    confidence=0.9,
                )
            )
        return events


def detect_webcams() -> list[int]:
    """
    Detect available webcam devices.

    Returns:
        List of available device indices
    """
    available = []
    for i in range(10):  # Check first 10 device indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available
