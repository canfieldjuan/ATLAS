"""
RTSP camera implementation using OpenCV.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from ...core.constants import DeviceStatus, DetectionType
from ...core.protocols import Detection, MotionEvent
from ...processing.frame_buffer import get_frame_buffer
from .base import BaseCameraCapability

logger = logging.getLogger("atlas.vision.devices.cameras.rtsp")


class RTSPCamera(BaseCameraCapability):
    """RTSP camera that captures frames via OpenCV."""

    def __init__(
        self,
        device_id: str,
        name: str,
        location: str,
        rtsp_url: str,
        fps: int = 10,
        reconnect_delay: float = 5.0,
    ):
        super().__init__(device_id, name, location)
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.reconnect_delay = reconnect_delay

        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0.0
        self._frame_count: int = 0
        self._motion_detector = None

    async def connect(self) -> bool:
        """Connect to RTSP stream and start capture loop."""
        if self._running:
            logger.warning("Camera %s already running", self.device_id)
            return True

        try:
            # Open capture in thread to avoid blocking
            self._capture = await asyncio.to_thread(
                self._create_capture, self.rtsp_url
            )

            if not self._capture.isOpened():
                logger.error("Failed to open RTSP stream: %s", self.rtsp_url)
                self._status = DeviceStatus.ERROR
                return False

            self._status = DeviceStatus.ONLINE
            self._running = True

            # Start capture loop
            self._capture_task = asyncio.create_task(self._capture_loop())
            logger.info("Connected to camera %s: %s", self.device_id, self.rtsp_url)
            return True

        except Exception as e:
            logger.error("Failed to connect to camera %s: %s", self.device_id, e)
            self._status = DeviceStatus.ERROR
            return False

    def _create_capture(self, url: str) -> cv2.VideoCapture:
        """Create OpenCV capture (runs in thread)."""
        cap = cv2.VideoCapture(url)
        # Set buffer size to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    async def disconnect(self) -> None:
        """Disconnect from RTSP stream."""
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
        logger.info("Disconnected from camera %s", self.device_id)

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
                    logger.warning("Failed to read frame from %s, reconnecting...", self.device_id)
                    await self._reconnect()
                    continue

                # Store frame
                self._last_frame = frame
                self._last_frame_time = time.time()
                self._frame_count += 1

                # Push to frame buffer
                await frame_buffer.push(self.device_id, frame)

                # Run motion detection if available
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
                logger.error("Capture error on %s: %s", self.device_id, e)
                await asyncio.sleep(1.0)

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a single frame (runs in thread)."""
        if self._capture is None:
            return False, None
        return self._capture.read()

    async def _reconnect(self) -> None:
        """Attempt to reconnect to the stream."""
        self._status = DeviceStatus.ERROR

        if self._capture:
            await asyncio.to_thread(self._capture.release)

        await asyncio.sleep(self.reconnect_delay)

        try:
            self._capture = await asyncio.to_thread(
                self._create_capture, self.rtsp_url
            )
            if self._capture.isOpened():
                self._status = DeviceStatus.ONLINE
                logger.info("Reconnected to camera %s", self.device_id)
            else:
                logger.warning("Reconnect failed for camera %s", self.device_id)
        except Exception as e:
            logger.error("Reconnect error for %s: %s", self.device_id, e)

    async def _run_detection(self, frame: np.ndarray) -> None:
        """Run motion detection on frame."""
        try:
            detections = await asyncio.to_thread(
                self._motion_detector.detect, frame, self.device_id
            )
            if detections:
                self._last_motion = datetime.now()
                # Store detections for API access
                self._current_detections = detections
        except Exception as e:
            logger.warning("Detection error on %s: %s", self.device_id, e)

    def set_motion_detector(self, detector) -> None:
        """Attach a motion detector to this camera."""
        self._motion_detector = detector
        self._current_detections: list[Detection] = []

    async def get_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame."""
        return self._last_frame

    async def get_current_detections(self) -> list[Detection]:
        """Get current detections from motion detector."""
        if hasattr(self, "_current_detections"):
            return self._current_detections
        return []

    async def get_motion_events(self, since: Optional[datetime] = None) -> list[MotionEvent]:
        """Get motion events (simplified - returns recent if motion detected)."""
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
