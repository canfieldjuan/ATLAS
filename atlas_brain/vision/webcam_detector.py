"""
Direct webcam person detection for presence tracking.

Runs YOLO on webcam feed and feeds detections directly to
PresenceService without requiring MQTT.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("atlas.vision.webcam_detector")


class WebcamPersonDetector:
    """
    Webcam-based person detection using YOLO.

    Feeds detections directly to PresenceService for room-aware control.
    """

    @staticmethod
    def find_device_by_name(name_substring: str) -> int | None:
        """Find video device index by name substring.

        Args:
            name_substring: Substring to search for in device name (case-insensitive)

        Returns:
            Device index if found, None otherwise
        """
        import subprocess
        try:
            # List video devices with v4l2-ctl
            result = subprocess.run(
                ["v4l2-ctl", "--list-devices"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                logger.warning("v4l2-ctl not available, falling back to index")
                return None

            lines = result.stdout.strip().split('\n')
            current_name = ""
            for line in lines:
                if not line.startswith('\t'):
                    current_name = line.strip()
                elif '/dev/video' in line and name_substring.lower() in current_name.lower():
                    # Extract device number
                    dev_path = line.strip()
                    if dev_path.startswith('/dev/video'):
                        idx = int(dev_path.replace('/dev/video', ''))
                        logger.info("Found video device '%s' at index %d: %s",
                                   name_substring, idx, current_name)
                        return idx
            logger.warning("Video device '%s' not found", name_substring)
            return None
        except Exception as e:
            logger.warning("Error finding video device: %s", e)
            return None

    def __init__(
        self,
        device_index: int = 0,
        device_name: str | None = None,
        camera_source_id: str = "webcam_office",
        fps: int = 5,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize webcam detector.

        Args:
            device_index: Video device index (deprecated, use device_name)
            device_name: Video device name substring (e.g. "C920", "Logitech")
            camera_source_id: ID used to map to room in presence config
            fps: Detection rate (lower = less CPU, 5 is usually enough)
            confidence_threshold: YOLO confidence threshold
        """
        # Resolve device name to index if provided
        if device_name:
            resolved_idx = self.find_device_by_name(device_name)
            if resolved_idx is not None:
                device_index = resolved_idx
            else:
                logger.warning("Device '%s' not found, using index %d", device_name, device_index)

        self.device_index = device_index
        self.camera_source_id = camera_source_id
        self.fps = fps
        self.confidence_threshold = confidence_threshold

        self._capture: Optional[cv2.VideoCapture] = None
        self._model = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Track state
        self._person_detected = False
        self._last_detection_time: Optional[datetime] = None
        self._active_tracks: set[int] = set()
        self._next_track_id = 1

    async def start(self) -> bool:
        """Start the detector."""
        if self._running:
            return True

        try:
            # Load YOLO model
            from ultralytics import YOLO
            logger.info("Loading YOLO model...")
            self._model = YOLO("yolov8n.pt")

            # Warm up model with dummy inference (pre-compiles CUDA kernels)
            logger.info("Warming up YOLO on CUDA...")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self._model(dummy_frame, verbose=False, classes=[0])
            logger.info("YOLO model ready (CUDA)")

            # Open webcam
            self._capture = cv2.VideoCapture(self.device_index)
            if not self._capture.isOpened():
                logger.error("Failed to open webcam at /dev/video%d", self.device_index)
                return False

            # Configure capture
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Warm up camera (discard initial black frames)
            logger.info("Warming up camera...")
            for _ in range(30):  # ~1 second at 30fps
                self._capture.read()
            logger.info("Camera ready")

            self._running = True
            self._task = asyncio.create_task(self._detection_loop())

            logger.info(
                "Webcam detector started: /dev/video%d -> %s @ %d fps",
                self.device_index,
                self.camera_source_id,
                self.fps,
            )
            return True

        except Exception as e:
            logger.error("Failed to start webcam detector: %s", e)
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

        logger.info("Webcam detector stopped")

    async def _detection_loop(self) -> None:
        """Main detection loop."""
        frame_interval = 1.0 / self.fps

        while self._running:
            try:
                start_time = asyncio.get_event_loop().time()

                # Read frame
                ret, frame = await asyncio.to_thread(self._capture.read)
                if not ret or frame is None:
                    await asyncio.sleep(0.1)
                    continue

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
                logger.error("Detection loop error: %s", e)
                await asyncio.sleep(1.0)

    def _detect_persons(self, frame: np.ndarray) -> list[dict]:
        """Run YOLO person detection on a frame (runs in thread)."""
        if self._model is None:
            return []

        # Run inference
        results = self._model(frame, verbose=False, classes=[0])  # class 0 = person

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
            logger.debug("Presence service not available: %s", e)
            return

        # Handle state changes
        if self._person_detected and not was_detected:
            # Person appeared
            track_id = self._next_track_id
            self._next_track_id += 1
            self._active_tracks.add(track_id)
            self._last_detection_time = datetime.now()

            logger.info(
                "Person detected in %s (track %d, confidence: %.2f)",
                self.camera_source_id,
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
            # Person left
            if self._active_tracks:
                track_id = self._active_tracks.pop()

                logger.info(
                    "Person left %s (track %d)",
                    self.camera_source_id,
                    track_id,
                )

                await presence_service.handle_camera_detection(
                    camera_source=self.camera_source_id,
                    person_detected=False,
                    track_id=track_id,
                )

        # Update last detection time if person present
        if self._person_detected:
            self._last_detection_time = datetime.now()


# Singleton
_webcam_detector: Optional[WebcamPersonDetector] = None


async def start_webcam_detector(
    device_index: int = 0,
    device_name: str | None = None,
    camera_source_id: str = "webcam_office",
    fps: int = 5,
) -> Optional[WebcamPersonDetector]:
    """Start the webcam detector singleton."""
    global _webcam_detector

    if _webcam_detector is None:
        _webcam_detector = WebcamPersonDetector(
            device_index=device_index,
            device_name=device_name,
            camera_source_id=camera_source_id,
            fps=fps,
        )

    if await _webcam_detector.start():
        return _webcam_detector
    return None


async def stop_webcam_detector() -> None:
    """Stop the webcam detector."""
    global _webcam_detector
    if _webcam_detector:
        await _webcam_detector.stop()
        _webcam_detector = None


def get_webcam_detector() -> Optional[WebcamPersonDetector]:
    """Get the webcam detector singleton."""
    return _webcam_detector
