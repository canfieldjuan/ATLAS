"""
Face recognition service using InsightFace.
"""

import logging
from typing import Optional
from uuid import UUID

import cv2
import numpy as np

logger = logging.getLogger("atlas.vision.recognition.face")


class FaceRecognitionService:
    """Face detection and embedding extraction using InsightFace."""

    def __init__(self):
        self._app = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of InsightFace model."""
        if self._initialized:
            return True

        try:
            from insightface.app import FaceAnalysis

            logger.info("Initializing InsightFace...")
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

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """
        Detect faces in frame and extract embeddings.

        Returns list of dicts with keys:
            - bbox: [x1, y1, x2, y2]
            - embedding: 512-dim numpy array
            - det_score: detection confidence
            - landmarks: facial landmarks
        """
        if not self._ensure_initialized():
            return []

        faces = self._app.get(frame)
        results = []

        for face in faces:
            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "embedding": face.embedding,
                "det_score": float(face.det_score),
                "landmarks": face.kps.tolist() if face.kps is not None else None,
            })

        return results

    def extract_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from frame (expects single face).

        Returns 512-dim embedding or None if no face detected.
        """
        faces = self.detect_faces(frame)
        if not faces:
            return None
        # Return highest confidence face
        best_face = max(faces, key=lambda f: f["det_score"])
        return best_face["embedding"]

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def crop_face(
        self,
        frame: np.ndarray,
        bbox: list[int],
        padding: float = 0.2,
    ) -> np.ndarray:
        """Crop face region from frame with padding."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Add padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        return frame[y1:y2, x1:x2]

    def encode_face_image(
        self,
        frame: np.ndarray,
        bbox: list[int],
        quality: int = 90,
    ) -> bytes:
        """Encode cropped face as JPEG bytes."""
        face_crop = self.crop_face(frame, bbox)
        _, buffer = cv2.imencode(
            ".jpg",
            face_crop,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        return buffer.tobytes()

    async def enroll_face(
        self,
        frame: np.ndarray,
        person_id: UUID,
        source: str = "enrollment",
        save_image: bool = True,
    ) -> Optional[UUID]:
        """
        Enroll a face for a person.

        Args:
            frame: Image containing the face
            person_id: UUID of the person
            source: Source of enrollment
            save_image: Whether to save reference image

        Returns:
            Face embedding UUID or None if no face detected
        """
        from .repository import get_person_repository

        faces = self.detect_faces(frame)
        if not faces:
            logger.warning("No face detected for enrollment")
            return None

        # Use highest confidence face
        best_face = max(faces, key=lambda f: f["det_score"])
        embedding = best_face["embedding"]
        quality_score = best_face["det_score"]

        # Optionally save reference image
        reference_image = None
        if save_image:
            reference_image = self.encode_face_image(frame, best_face["bbox"])

        repo = get_person_repository()
        embedding_id = await repo.add_face_embedding(
            person_id=person_id,
            embedding=embedding,
            quality_score=quality_score,
            source=source,
            reference_image=reference_image,
        )

        logger.info(
            "Enrolled face for person %s (quality: %.2f)",
            person_id,
            quality_score,
        )
        return embedding_id

    async def recognize_face(
        self,
        frame: np.ndarray,
        threshold: float = 0.6,
        camera_source: Optional[str] = None,
        auto_enroll_unknown: bool = True,
        use_averaged: bool = True,
    ) -> Optional[dict]:
        """
        Recognize a face in the frame.

        Args:
            frame: Image containing the face
            threshold: Similarity threshold for match
            camera_source: Camera source identifier
            auto_enroll_unknown: Auto-create profile for unknown faces
            use_averaged: Use averaged embeddings for more reliable matching

        Returns:
            Dict with person_id, name, similarity, is_known, matched
        """
        from .repository import get_person_repository

        faces = self.detect_faces(frame)
        if not faces:
            return None

        best_face = max(faces, key=lambda f: f["det_score"])
        embedding = best_face["embedding"]

        repo = get_person_repository()

        # Try to find matching face (averaged is more reliable with multiple samples)
        if use_averaged:
            match = await repo.find_matching_face_averaged(embedding, threshold)
        else:
            match = await repo.find_matching_face(embedding, threshold)

        if match:
            # Known face
            await repo.update_last_seen(match["person_id"])
            await repo.log_recognition_event(
                person_id=match["person_id"],
                recognition_type="face",
                confidence=match["similarity"],
                camera_source=camera_source,
                matched=True,
            )
            return {
                "person_id": match["person_id"],
                "name": match["name"],
                "similarity": match["similarity"],
                "is_known": match["is_known"],
                "matched": True,
            }

        # Unknown face
        if auto_enroll_unknown:
            # Create new unknown person
            unknown_count = await repo.get_unknown_person_count()
            unknown_name = f"unknown_{unknown_count + 1}"

            person_id = await repo.create_person(
                name=unknown_name,
                is_known=False,
                auto_created=True,
            )

            # Enroll their face
            reference_image = self.encode_face_image(frame, best_face["bbox"])
            await repo.add_face_embedding(
                person_id=person_id,
                embedding=embedding,
                quality_score=best_face["det_score"],
                source="auto_enrollment",
                reference_image=reference_image,
            )

            await repo.log_recognition_event(
                person_id=person_id,
                recognition_type="face",
                confidence=best_face["det_score"],
                camera_source=camera_source,
                matched=False,
                metadata={"auto_enrolled": True},
            )

            logger.info("Auto-enrolled unknown person: %s", unknown_name)

            return {
                "person_id": person_id,
                "name": unknown_name,
                "similarity": 1.0,
                "is_known": False,
                "matched": False,
                "auto_enrolled": True,
            }

        return None


# Singleton instance
_service: Optional[FaceRecognitionService] = None


def get_face_service() -> FaceRecognitionService:
    """Get the face recognition service singleton."""
    global _service
    if _service is None:
        _service = FaceRecognitionService()
    return _service
