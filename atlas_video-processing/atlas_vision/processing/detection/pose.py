"""
Pose detection using MediaPipe for overlay rendering.

This is a lightweight wrapper that just does pose detection and drawing.
"""

import logging
from pathlib import Path
from typing import Optional
import urllib.request

import cv2
import numpy as np

logger = logging.getLogger("atlas.vision.detection.pose")

# MediaPipe pose landmark indices for skeleton drawing
POSE_CONNECTIONS = [
    # Torso
    (11, 12),  # left shoulder - right shoulder
    (11, 23),  # left shoulder - left hip
    (12, 24),  # right shoulder - right hip
    (23, 24),  # left hip - right hip
    # Left arm
    (11, 13),  # left shoulder - left elbow
    (13, 15),  # left elbow - left wrist
    # Right arm
    (12, 14),  # right shoulder - right elbow
    (14, 16),  # right elbow - right wrist
    # Left leg
    (23, 25),  # left hip - left knee
    (25, 27),  # left knee - left ankle
    # Right leg
    (24, 26),  # right hip - right knee
    (26, 28),  # right knee - right ankle
]

# Landmark names for reference
POSE_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


class PoseDetector:
    """Pose detection using MediaPipe PoseLandmarker."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._pose = None
        self._initialized = False
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of MediaPipe PoseLandmarker."""
        if self._initialized:
            return True

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            logger.info("Initializing MediaPipe PoseLandmarker...")

            # Download model if not present
            model_path = Path("models/pose_landmarker_lite.task")
            model_path.parent.mkdir(parents=True, exist_ok=True)

            if not model_path.exists():
                logger.info("Downloading pose landmarker model...")
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
                urllib.request.urlretrieve(url, model_path)
                logger.info("Model downloaded to %s", model_path)

            # Create PoseLandmarker
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )
            self._pose = vision.PoseLandmarker.create_from_options(options)
            self._initialized = True
            logger.info("MediaPipe PoseLandmarker initialized")
            return True
        except Exception as e:
            logger.error("Failed to initialize MediaPipe: %s", e)
            return False

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect poses in frame.

        Returns list of pose dicts with keys:
            - landmarks: list of 33 landmarks with x, y, z, visibility
            - world_landmarks: 3D world coordinates (if available)
        """
        if not self._ensure_initialized():
            return []

        try:
            import mediapipe as mp

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect poses
            results = self._pose.detect(mp_image)

            if not results.pose_landmarks:
                return []

            poses = []
            for i, pose_landmarks in enumerate(results.pose_landmarks):
                landmarks = []
                for lm in pose_landmarks:
                    landmarks.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    })

                world_landmarks = None
                if results.pose_world_landmarks and i < len(results.pose_world_landmarks):
                    world_landmarks = []
                    for wlm in results.pose_world_landmarks[i]:
                        world_landmarks.append({
                            "x": wlm.x,
                            "y": wlm.y,
                            "z": wlm.z,
                            "visibility": wlm.visibility,
                        })

                poses.append({
                    "landmarks": landmarks,
                    "world_landmarks": world_landmarks,
                })

            return poses
        except Exception as e:
            logger.error("Pose detection failed: %s", e)
            return []

    def draw_detections(
        self,
        frame: np.ndarray,
        poses: list[dict],
        connection_color: tuple[int, int, int] = (0, 255, 255),
        landmark_color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
        landmark_radius: int = 4,
        visibility_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Draw pose skeleton on frame.

        Args:
            frame: BGR image
            poses: List of pose detections from detect()
            connection_color: Color for skeleton lines (BGR)
            landmark_color: Color for landmark points (BGR)
            thickness: Line thickness
            landmark_radius: Radius of landmark circles
            visibility_threshold: Min visibility to draw landmark

        Returns:
            Annotated frame (modifies in place)
        """
        h, w = frame.shape[:2]

        for pose in poses:
            landmarks = pose["landmarks"]

            # Draw connections (skeleton lines)
            for start_idx, end_idx in POSE_CONNECTIONS:
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]

                # Only draw if both landmarks are visible enough
                if (start_lm["visibility"] >= visibility_threshold and
                    end_lm["visibility"] >= visibility_threshold):
                    start_pt = (int(start_lm["x"] * w), int(start_lm["y"] * h))
                    end_pt = (int(end_lm["x"] * w), int(end_lm["y"] * h))
                    cv2.line(frame, start_pt, end_pt, connection_color, thickness)

            # Draw landmarks
            for lm in landmarks:
                if lm["visibility"] >= visibility_threshold:
                    pt = (int(lm["x"] * w), int(lm["y"] * h))
                    cv2.circle(frame, pt, landmark_radius, landmark_color, -1)

        return frame


# Singleton instance
_detector: Optional[PoseDetector] = None


def get_pose_detector() -> PoseDetector:
    """Get the pose detector singleton."""
    global _detector
    if _detector is None:
        _detector = PoseDetector()
    return _detector
