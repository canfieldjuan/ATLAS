"""
Gait recognition service using MediaPipe Pose.
"""

import logging
import math
from collections import deque
from typing import Optional
from uuid import UUID

import numpy as np

logger = logging.getLogger("atlas.recognition.gait")

# MediaPipe pose landmark indices
POSE_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


class GaitRecognitionService:
    """Gait analysis and embedding extraction using MediaPipe."""

    def __init__(self, sequence_length: Optional[int] = None):
        from ...config import settings

        self._pose = None
        self._initialized = False
        self._config = settings.recognition
        self.sequence_length = sequence_length or self._config.gait_sequence_length
        self._pose_buffer: deque = deque(maxlen=self.sequence_length)

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of MediaPipe PoseLandmarker (new tasks API)."""
        if self._initialized:
            return True

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            from pathlib import Path
            import urllib.request

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
                min_pose_detection_confidence=self._config.mediapipe_detection_confidence,
                min_tracking_confidence=self._config.mediapipe_tracking_confidence,
            )
            self._pose = vision.PoseLandmarker.create_from_options(options)
            self._initialized = True
            logger.info("MediaPipe PoseLandmarker initialized")
            return True
        except Exception as e:
            logger.error("Failed to initialize MediaPipe: %s", e)
            return False

    def extract_pose(self, frame: np.ndarray) -> Optional[dict]:
        """
        Extract pose landmarks from frame.

        Returns dict with normalized landmark coordinates or None.
        """
        if not self._ensure_initialized():
            return None

        import mediapipe as mp
        import cv2

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect pose
        results = self._pose.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        # Get first detected pose
        pose_landmarks = results.pose_landmarks[0]

        landmarks = {}
        for name, idx in POSE_LANDMARKS.items():
            lm = pose_landmarks[idx]
            landmarks[name] = {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            }

        return landmarks

    def add_pose_to_buffer(self, pose: dict) -> int:
        """Add pose to sequence buffer. Returns buffer length."""
        self._pose_buffer.append(pose)
        return len(self._pose_buffer)

    def clear_buffer(self) -> None:
        """Clear the pose sequence buffer."""
        self._pose_buffer.clear()

    def is_buffer_full(self) -> bool:
        """Check if buffer has enough frames."""
        return len(self._pose_buffer) >= self.sequence_length

    def _compute_angle(
        self,
        p1: dict,
        p2: dict,
        p3: dict,
    ) -> float:
        """Compute angle at p2 formed by p1-p2-p3."""
        v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
        v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return float(angle)

    def _extract_frame_features(self, pose: dict) -> np.ndarray:
        """Extract feature vector from single pose frame."""
        features = []

        # Joint angles
        # Left knee angle
        features.append(self._compute_angle(
            pose["left_hip"],
            pose["left_knee"],
            pose["left_ankle"],
        ))

        # Right knee angle
        features.append(self._compute_angle(
            pose["right_hip"],
            pose["right_knee"],
            pose["right_ankle"],
        ))

        # Left hip angle
        features.append(self._compute_angle(
            pose["left_shoulder"],
            pose["left_hip"],
            pose["left_knee"],
        ))

        # Right hip angle
        features.append(self._compute_angle(
            pose["right_shoulder"],
            pose["right_hip"],
            pose["right_knee"],
        ))

        # Left elbow angle
        features.append(self._compute_angle(
            pose["left_shoulder"],
            pose["left_elbow"],
            pose["left_wrist"],
        ))

        # Right elbow angle
        features.append(self._compute_angle(
            pose["right_shoulder"],
            pose["right_elbow"],
            pose["right_wrist"],
        ))

        # Body proportions (normalized)
        shoulder_width = abs(
            pose["left_shoulder"]["x"] - pose["right_shoulder"]["x"]
        )
        hip_width = abs(pose["left_hip"]["x"] - pose["right_hip"]["x"])
        torso_length = abs(
            (pose["left_shoulder"]["y"] + pose["right_shoulder"]["y"]) / 2
            - (pose["left_hip"]["y"] + pose["right_hip"]["y"]) / 2
        )
        leg_length = abs(
            (pose["left_hip"]["y"] + pose["right_hip"]["y"]) / 2
            - (pose["left_ankle"]["y"] + pose["right_ankle"]["y"]) / 2
        )

        features.extend([
            shoulder_width,
            hip_width,
            torso_length,
            leg_length,
            shoulder_width / (hip_width + 1e-6),
            torso_length / (leg_length + 1e-6),
        ])

        return np.array(features, dtype=np.float32)

    def compute_gait_embedding(self) -> Optional[np.ndarray]:
        """
        Compute gait embedding from buffered pose sequence.

        Returns 256-dim embedding or None if buffer not full.
        """
        if not self.is_buffer_full():
            logger.warning(
                "Buffer not full: %d/%d",
                len(self._pose_buffer),
                self.sequence_length,
            )
            return None

        # Extract features from each frame
        frame_features = []
        for pose in self._pose_buffer:
            features = self._extract_frame_features(pose)
            frame_features.append(features)

        frame_features = np.array(frame_features)

        # Temporal statistics
        embedding_parts = []

        # Mean features across sequence
        embedding_parts.append(np.mean(frame_features, axis=0))

        # Std features across sequence
        embedding_parts.append(np.std(frame_features, axis=0))

        # Min features
        embedding_parts.append(np.min(frame_features, axis=0))

        # Max features
        embedding_parts.append(np.max(frame_features, axis=0))

        # First derivative statistics (velocity)
        velocity = np.diff(frame_features, axis=0)
        embedding_parts.append(np.mean(velocity, axis=0))
        embedding_parts.append(np.std(velocity, axis=0))

        # Second derivative statistics (acceleration)
        acceleration = np.diff(velocity, axis=0)
        embedding_parts.append(np.mean(acceleration, axis=0))
        embedding_parts.append(np.std(acceleration, axis=0))

        # Concatenate all parts
        embedding = np.concatenate(embedding_parts)

        # Pad or truncate to 256 dimensions
        if len(embedding) < 256:
            embedding = np.pad(embedding, (0, 256 - len(embedding)))
        else:
            embedding = embedding[:256]

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    async def enroll_gait(
        self,
        person_id: UUID,
        walking_direction: Optional[str] = None,
        source: str = "enrollment",
    ) -> Optional[UUID]:
        """
        Enroll gait signature from buffered sequence.

        Args:
            person_id: UUID of the person
            walking_direction: Direction of walking (left, right, towards, away)
            source: Source of enrollment

        Returns:
            Gait embedding UUID or None if buffer not full
        """
        from .repository import get_person_repository

        embedding = self.compute_gait_embedding()
        if embedding is None:
            return None

        repo = get_person_repository()
        embedding_id = await repo.add_gait_embedding(
            person_id=person_id,
            embedding=embedding,
            capture_duration_ms=int(self.sequence_length * 33.3),  # ~30fps
            frame_count=self.sequence_length,
            walking_direction=walking_direction,
            source=source,
        )

        logger.info("Enrolled gait for person %s", person_id)
        self.clear_buffer()
        return embedding_id

    async def recognize_gait(
        self,
        threshold: float = 0.5,
        camera_source: Optional[str] = None,
        use_averaged: bool = True,
        auto_enroll_unknown: bool = False,
    ) -> Optional[dict]:
        """
        Recognize gait from buffered sequence.

        Args:
            threshold: Similarity threshold for match
            camera_source: Camera source identifier
            use_averaged: Use averaged centroids for more reliable matching
            auto_enroll_unknown: Auto-create profile for unknown gaits

        Returns:
            Dict with person_id, name, similarity, matched
        """
        from .repository import get_person_repository

        embedding = self.compute_gait_embedding()
        if embedding is None:
            return None

        repo = get_person_repository()

        if use_averaged:
            match = await repo.find_matching_gait_averaged(embedding, threshold)
        else:
            match = await repo.find_matching_gait(embedding, threshold)

        if match:
            await repo.update_last_seen(match["person_id"])
            await repo.log_recognition_event(
                person_id=match["person_id"],
                recognition_type="gait",
                confidence=match["similarity"],
                camera_source=camera_source,
                matched=True,
            )

            self.clear_buffer()
            return {
                "person_id": match["person_id"],
                "name": match["name"],
                "similarity": match["similarity"],
                "is_known": match["is_known"],
                "matched": True,
            }

        # Unknown gait - optionally auto-enroll
        if auto_enroll_unknown:
            unknown_count = await repo.get_unknown_person_count()
            unknown_name = f"unknown_{unknown_count + 1}"

            person_id = await repo.create_person(
                name=unknown_name,
                is_known=False,
                auto_created=True,
            )

            await repo.add_gait_embedding(
                person_id=person_id,
                embedding=embedding,
                capture_duration_ms=int(self.sequence_length * 33.3),
                frame_count=self.sequence_length,
                source="auto_enrollment",
            )

            await repo.log_recognition_event(
                person_id=person_id,
                recognition_type="gait",
                confidence=0.0,
                camera_source=camera_source,
                matched=False,
                metadata={"auto_enrolled": True},
            )

            logger.info("Auto-enrolled unknown gait: %s", unknown_name)

            self.clear_buffer()
            return {
                "person_id": person_id,
                "name": unknown_name,
                "similarity": 1.0,
                "is_known": False,
                "matched": False,
                "auto_enrolled": True,
            }

        self.clear_buffer()
        return None


# Singleton instance
_service: Optional[GaitRecognitionService] = None


def get_gait_service() -> GaitRecognitionService:
    """Get the gait recognition service singleton."""
    global _service
    if _service is None:
        _service = GaitRecognitionService()
    return _service
