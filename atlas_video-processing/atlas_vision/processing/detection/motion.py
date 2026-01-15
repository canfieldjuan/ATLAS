"""
Motion detection using background subtraction.
"""

import logging
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from ...core.constants import DetectionType
from ...core.protocols import Detection

logger = logging.getLogger("atlas.vision.processing.detection.motion")


class MotionDetector:
    """Detect motion using OpenCV background subtraction."""

    def __init__(
        self,
        sensitivity: float = 0.5,
        min_area: int = 500,
        cooldown_seconds: float = 2.0,
        learning_rate: float = 0.01,
    ):
        """
        Initialize motion detector.

        Args:
            sensitivity: Detection sensitivity 0.0-1.0 (higher = more sensitive)
            min_area: Minimum contour area to consider as motion
            cooldown_seconds: Minimum time between detections per source
            learning_rate: Background model learning rate
        """
        self.sensitivity = sensitivity
        self.min_area = min_area
        self.cooldown_seconds = cooldown_seconds
        self.learning_rate = learning_rate

        # Per-source background subtractors
        self._subtractors: dict[str, cv2.BackgroundSubtractorMOG2] = {}
        self._last_detection: dict[str, float] = {}

        # Threshold based on sensitivity (inverted: high sensitivity = low threshold)
        self._threshold = int(255 * (1.0 - sensitivity * 0.5))

    @property
    def name(self) -> str:
        return "motion"

    def _get_subtractor(self, source_id: str) -> cv2.BackgroundSubtractorMOG2:
        """Get or create background subtractor for a source."""
        if source_id not in self._subtractors:
            self._subtractors[source_id] = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=False,
            )
        return self._subtractors[source_id]

    def detect(self, frame: np.ndarray, source_id: str) -> list[Detection]:
        """
        Detect motion in a frame.

        Args:
            frame: Input frame (BGR format)
            source_id: Camera identifier

        Returns:
            List of Detection objects for motion regions
        """
        detections = []

        # Check cooldown
        now = time.time()
        last = self._last_detection.get(source_id, 0)
        if now - last < self.cooldown_seconds:
            return detections

        try:
            # Get background subtractor
            subtractor = self._get_subtractor(source_id)

            # Apply background subtraction
            fg_mask = subtractor.apply(frame, learningRate=self.learning_rate)

            # Threshold to remove shadows and noise
            _, thresh = cv2.threshold(fg_mask, self._threshold, 255, cv2.THRESH_BINARY)

            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter by area and create detections
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate confidence based on area relative to frame
                    frame_area = frame.shape[0] * frame.shape[1]
                    confidence = min(0.99, 0.5 + (area / frame_area) * 10)

                    detections.append(
                        Detection(
                            camera_id=source_id,
                            timestamp=datetime.now(),
                            detection_type=DetectionType.MOTION,
                            confidence=confidence,
                            label="motion",
                            bbox=(x, y, w, h),
                        )
                    )

            if detections:
                self._last_detection[source_id] = now
                logger.debug(
                    "Motion detected on %s: %d regions",
                    source_id, len(detections)
                )

        except Exception as e:
            logger.error("Motion detection error on %s: %s", source_id, e)

        return detections

    def reset(self, source_id: Optional[str] = None) -> None:
        """Reset background model."""
        if source_id:
            if source_id in self._subtractors:
                del self._subtractors[source_id]
                logger.info("Reset motion detector for %s", source_id)
        else:
            self._subtractors.clear()
            self._last_detection.clear()
            logger.info("Reset all motion detectors")


# Global motion detector instance
_motion_detector: Optional[MotionDetector] = None


def get_motion_detector() -> MotionDetector:
    """Get or create global motion detector."""
    global _motion_detector
    if _motion_detector is None:
        _motion_detector = MotionDetector()
    return _motion_detector
