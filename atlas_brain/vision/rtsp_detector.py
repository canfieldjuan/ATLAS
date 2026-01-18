"""
RTSP camera person detection for presence tracking.

Runs YOLO on RTSP streams (e.g., from Wyze Bridge) and feeds
detections directly to PresenceService.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("atlas.vision.rtsp_detector")


class RTSPPersonDetector:
    """
    RTSP stream person detection using YOLO.

    Connects to RTSP streams (like wyze-bridge) and runs person detection,
    feeding results to PresenceService for room-aware control.
    """

    def __init__(
        self,
        rtsp_url: str,
        camera_id: str,
        camera_source_id: str,
        fps: int = 10,
        confidence_threshold: float = 0.5,
        reconnect_delay: float = 5.0,
    ):
        """
        Initialize RTSP detector.

        Args:
            rtsp_url: RTSP stream URL (e.g., rtsp://localhost:8554/camera-name)
            camera_id: Unique identifier for this camera
            camera_source_id: ID used to map to room in presence config
            fps: Detection rate (10 is good for RTSP to reduce bandwidth)
            confidence_threshold: YOLO confidence threshold
            reconnect_delay: Seconds to wait before reconnecting on failure
        """
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.camera_source_id = camera_source_id
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        self.reconnect_delay = reconnect_delay

        self._capture: Optional[cv2.VideoCapture] = None
        self._model = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Track state
        self._person_detected = False
        self._last_detection_time: Optional[datetime] = None
        self._active_tracks: set[int] = set()
        self._next_track_id = 1
        self._consecutive_failures = 0

    async def start(self) -> bool:
        """Start the detector."""
        if self._running:
            return True

        try:
            # Load YOLO model (shared across all detectors)
            from ultralytics import YOLO
            logger.info("[%s] Loading YOLO model...", self.camera_id)
            self._model = YOLO("yolov8n.pt")

            # Warm up model
            logger.info("[%s] Warming up YOLO on CUDA...", self.camera_id)
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self._model(dummy_frame, verbose=False, classes=[0])
            logger.info("[%s] YOLO model ready", self.camera_id)

            self._running = True
            self._task = asyncio.create_task(self._detection_loop())

            logger.info(
                "[%s] RTSP detector started: %s -> %s @ %d fps",
                self.camera_id,
                self.rtsp_url,
                self.camera_source_id,
                self.fps,
            )
            return True

        except Exception as e:
            logger.error("[%s] Failed to start RTSP detector: %s", self.camera_id, e)
            return False

    async def stop(self) -> None:
        """Stop the detector."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._capture:
            self._capture.release()
            self._capture = None

        logger.info("[%s] RTSP detector stopped", self.camera_id)

    def _connect(self) -> bool:
        """Connect to the RTSP stream."""
        if self._capture:
            self._capture.release()

        self._capture = cv2.VideoCapture(self.rtsp_url)

        # RTSP-specific settings
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        self._capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

        if not self._capture.isOpened():
            logger.warning("[%s] Failed to connect to %s", self.camera_id, self.rtsp_url)
            return False

        # Discard initial frames
        for _ in range(5):
            self._capture.read()

        logger.info("[%s] Connected to RTSP stream", self.camera_id)
        self._consecutive_failures = 0
        return True

    async def _detection_loop(self) -> None:
        """Main detection loop with auto-reconnect."""
        frame_interval = 1.0 / self.fps

        while self._running:
            # Connect if needed
            if self._capture is None or not self._capture.isOpened():
                if not await asyncio.to_thread(self._connect):
                    self._consecutive_failures += 1
                    wait_time = min(self.reconnect_delay * self._consecutive_failures, 60)
                    logger.info(
                        "[%s] Reconnecting in %.1fs (attempt %d)",
                        self.camera_id,
                        wait_time,
                        self._consecutive_failures,
                    )
                    await asyncio.sleep(wait_time)
                    continue

            try:
                start_time = asyncio.get_event_loop().time()

                # Read frame
                ret, frame = await asyncio.to_thread(self._capture.read)
                if not ret or frame is None:
                    self._consecutive_failures += 1
                    if self._consecutive_failures > 5:
                        logger.warning("[%s] Stream lost, reconnecting...", self.camera_id)
                        self._capture.release()
                        self._capture = None
                    await asyncio.sleep(0.1)
                    continue

                self._consecutive_failures = 0

                # Run YOLO detection in thread
                detections = await asyncio.to_thread(self._detect_persons, frame)

                # Process detections
                await self._process_detections(detections)

                # Maintain frame rate
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[%s] Detection loop error: %s", self.camera_id, e)
                await asyncio.sleep(1.0)

    def _detect_persons(self, frame: np.ndarray) -> list[dict]:
        """Run YOLO person detection on a frame."""
        if self._model is None:
            return []

        results = self._model(frame, verbose=False, classes=[0])

        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    })

        return detections

    async def _process_detections(self, detections: list[dict]) -> None:
        """Process detections and update presence service."""
        person_count = len(detections)
        was_detected = self._person_detected
        self._person_detected = person_count > 0

        # Get presence service
        try:
            from ..presence import get_presence_service
            presence_service = get_presence_service()
        except Exception as e:
            logger.debug("[%s] Presence service not available: %s", self.camera_id, e)
            return

        # Handle state changes
        if self._person_detected and not was_detected:
            track_id = self._next_track_id
            self._next_track_id += 1
            self._active_tracks.add(track_id)
            self._last_detection_time = datetime.now()

            logger.info(
                "[%s] Person detected (track %d, confidence: %.2f)",
                self.camera_id,
                track_id,
                detections[0]["confidence"] if detections else 0,
            )

            await presence_service.handle_camera_detection(
                camera_source=self.camera_source_id,
                person_detected=True,
                track_id=track_id,
                confidence=detections[0]["confidence"] if detections else 0.85,
            )

        elif not self._person_detected and was_detected:
            if self._active_tracks:
                track_id = self._active_tracks.pop()

                logger.info("[%s] Person left (track %d)", self.camera_id, track_id)

                await presence_service.handle_camera_detection(
                    camera_source=self.camera_source_id,
                    person_detected=False,
                    track_id=track_id,
                )

        if self._person_detected:
            self._last_detection_time = datetime.now()


# Manager for multiple RTSP cameras
class RTSPDetectorManager:
    """Manages multiple RTSP camera detectors."""

    def __init__(self):
        self._detectors: dict[str, RTSPPersonDetector] = {}
        self._model = None  # Shared YOLO model

    async def add_camera(
        self,
        camera_id: str,
        rtsp_url: str,
        camera_source_id: str,
        fps: int = 10,
    ) -> bool:
        """Add and start a new camera detector."""
        if camera_id in self._detectors:
            logger.warning("Camera %s already exists", camera_id)
            return False

        detector = RTSPPersonDetector(
            rtsp_url=rtsp_url,
            camera_id=camera_id,
            camera_source_id=camera_source_id,
            fps=fps,
        )

        if await detector.start():
            self._detectors[camera_id] = detector
            return True
        return False

    async def remove_camera(self, camera_id: str) -> bool:
        """Stop and remove a camera detector."""
        if camera_id not in self._detectors:
            return False

        await self._detectors[camera_id].stop()
        del self._detectors[camera_id]
        return True

    async def stop_all(self) -> None:
        """Stop all camera detectors."""
        for detector in self._detectors.values():
            await detector.stop()
        self._detectors.clear()

    def get_cameras(self) -> list[str]:
        """Get list of active camera IDs."""
        return list(self._detectors.keys())

    def get_detector(self, camera_id: str) -> Optional[RTSPPersonDetector]:
        """Get a specific detector."""
        return self._detectors.get(camera_id)


# Singleton manager
_rtsp_manager: Optional[RTSPDetectorManager] = None


def get_rtsp_manager() -> RTSPDetectorManager:
    """Get the RTSP detector manager singleton."""
    global _rtsp_manager
    if _rtsp_manager is None:
        _rtsp_manager = RTSPDetectorManager()
    return _rtsp_manager


async def start_rtsp_cameras(cameras: list[dict]) -> RTSPDetectorManager:
    """
    Start RTSP camera detectors.

    Args:
        cameras: List of camera configs, each with:
            - camera_id: Unique ID
            - rtsp_url: RTSP stream URL
            - camera_source_id: Maps to room in presence config
            - fps: Optional, default 10

    Returns:
        RTSPDetectorManager instance
    """
    manager = get_rtsp_manager()

    for cam in cameras:
        await manager.add_camera(
            camera_id=cam["camera_id"],
            rtsp_url=cam["rtsp_url"],
            camera_source_id=cam["camera_source_id"],
            fps=cam.get("fps", 10),
        )

    return manager


async def stop_rtsp_cameras() -> None:
    """Stop all RTSP camera detectors."""
    global _rtsp_manager
    if _rtsp_manager:
        await _rtsp_manager.stop_all()
        _rtsp_manager = None
