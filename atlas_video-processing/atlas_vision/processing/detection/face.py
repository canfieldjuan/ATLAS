"""
Face detection using InsightFace for overlay rendering.

This is a lightweight wrapper that just does detection - no enrollment or recognition.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("atlas.vision.detection.face")


class FaceDetector:
    """Face detection using InsightFace."""

    def __init__(self):
        self._app = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of InsightFace model."""
        if self._initialized:
            return True

        try:
            from insightface.app import FaceAnalysis

            logger.info("Initializing InsightFace for face detection...")
            self._app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            self._initialized = True
            logger.info("InsightFace initialized successfully")
            return True
        except Exception as e:
            logger.error("Failed to initialize InsightFace: %s", e)
            return False

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect faces in frame.

        Returns list of dicts with keys:
            - bbox: [x1, y1, x2, y2] (integer pixel coordinates)
            - det_score: detection confidence
            - landmarks: facial landmarks (5 points)
        """
        if not self._ensure_initialized():
            return []

        try:
            faces = self._app.get(frame)
            results = []

            for face in faces:
                results.append({
                    "bbox": face.bbox.astype(int).tolist(),
                    "det_score": float(face.det_score),
                    "landmarks": face.kps.tolist() if face.kps is not None else None,
                })

            return results
        except Exception as e:
            logger.error("Face detection failed: %s", e)
            return []

    def draw_detections(
        self,
        frame: np.ndarray,
        faces: list[dict],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_landmarks: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw face detection boxes and landmarks on frame.

        Args:
            frame: BGR image
            faces: List of face detections from detect()
            color: Box color (BGR)
            thickness: Line thickness
            show_landmarks: Draw facial landmarks
            show_confidence: Show detection confidence

        Returns:
            Annotated frame (modifies in place)
        """
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw confidence
            if show_confidence:
                conf = face["det_score"]
                label = f"Face {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

            # Draw landmarks
            if show_landmarks and face["landmarks"]:
                for lm in face["landmarks"]:
                    cx, cy = int(lm[0]), int(lm[1])
                    cv2.circle(frame, (cx, cy), 3, (255, 0, 255), -1)

        return frame


# Singleton instance
_detector: Optional[FaceDetector] = None


def get_face_detector() -> FaceDetector:
    """Get the face detector singleton."""
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector
