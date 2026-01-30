"""
Multi-person track manager for recognition.

Associates YOLO track IDs with recognized person IDs using face and gait.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import numpy as np

logger = logging.getLogger("atlas.vision.recognition.tracker")


@dataclass
class TrackedPerson:
    """State for a tracked person."""

    track_id: int
    camera_source: str
    person_id: Optional[UUID] = None
    person_name: Optional[str] = None
    is_known: bool = False

    # Recognition state
    face_similarity: float = 0.0
    gait_similarity: float = 0.0
    combined_similarity: float = 0.0

    # Gait enrollment state
    needs_gait_enrollment: bool = False
    gait_enrolled: bool = False

    # Timestamps
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_face_match: float = 0.0
    last_gait_match: float = 0.0

    # Bounding box (pixel coordinates)
    bbox: list[int] = field(default_factory=lambda: [0, 0, 0, 0])


class TrackManager:
    """
    Manages track-to-person associations for multi-person recognition.

    Associates YOLO ByteTrack IDs with recognized person IDs using:
    - Face recognition (primary identifier)
    - Gait recognition (secondary, for verification)
    - Bounding box IoU for pose-track association
    """

    def __init__(self):
        from ..core.config import settings

        self._config = settings.recognition
        # Per-camera track state: camera_source -> track_id -> TrackedPerson
        self._tracks: dict[str, dict[int, TrackedPerson]] = {}

    def _ensure_camera(self, camera_source: str) -> None:
        """Ensure camera dict exists."""
        if camera_source not in self._tracks:
            self._tracks[camera_source] = {}

    def get_track(
        self,
        camera_source: str,
        track_id: int,
    ) -> Optional[TrackedPerson]:
        """Get tracked person state."""
        self._ensure_camera(camera_source)
        return self._tracks[camera_source].get(track_id)

    def create_or_update_track(
        self,
        camera_source: str,
        track_id: int,
        bbox: list[int],
    ) -> TrackedPerson:
        """Create or update track with new bbox."""
        self._ensure_camera(camera_source)

        if track_id not in self._tracks[camera_source]:
            self._tracks[camera_source][track_id] = TrackedPerson(
                track_id=track_id,
                camera_source=camera_source,
                bbox=bbox,
            )
        else:
            track = self._tracks[camera_source][track_id]
            track.bbox = bbox
            track.last_seen = time.time()

        return self._tracks[camera_source][track_id]

    def associate_person(
        self,
        camera_source: str,
        track_id: int,
        person_id: UUID,
        person_name: str,
        is_known: bool,
        face_similarity: float,
    ) -> TrackedPerson:
        """Associate a track with a recognized person from face."""
        track = self.get_track(camera_source, track_id)
        if not track:
            track = self.create_or_update_track(camera_source, track_id, [0, 0, 0, 0])

        track.person_id = person_id
        track.person_name = person_name
        track.is_known = is_known
        track.face_similarity = face_similarity
        track.last_face_match = time.time()

        # Update combined score
        if track.gait_similarity > 0:
            track.combined_similarity = (
                track.face_similarity + track.gait_similarity
            ) / 2
        else:
            track.combined_similarity = track.face_similarity

        return track

    def update_gait_match(
        self,
        camera_source: str,
        track_id: int,
        gait_similarity: float,
    ) -> Optional[TrackedPerson]:
        """Update gait similarity for a track."""
        track = self.get_track(camera_source, track_id)
        if not track:
            return None

        track.gait_similarity = gait_similarity
        track.last_gait_match = time.time()

        # Update combined score
        if track.face_similarity > 0:
            track.combined_similarity = (
                track.face_similarity + track.gait_similarity
            ) / 2
        else:
            track.combined_similarity = track.gait_similarity

        return track

    def mark_gait_enrolled(
        self,
        camera_source: str,
        track_id: int,
    ) -> Optional[TrackedPerson]:
        """Mark that gait has been enrolled for this track."""
        track = self.get_track(camera_source, track_id)
        if track:
            track.gait_enrolled = True
            track.needs_gait_enrollment = False
        return track

    def mark_needs_gait_enrollment(
        self,
        camera_source: str,
        track_id: int,
    ) -> Optional[TrackedPerson]:
        """Mark that this track needs gait enrollment."""
        track = self.get_track(camera_source, track_id)
        if track and not track.gait_enrolled:
            track.needs_gait_enrollment = True
        return track

    def find_track_by_bbox_iou(
        self,
        camera_source: str,
        pose_bbox: list[int],
        min_iou: Optional[float] = None,
    ) -> Optional[TrackedPerson]:
        """
        Find track with best IoU match to pose bbox.

        Args:
            camera_source: Camera identifier
            pose_bbox: [x1, y1, x2, y2] from pose detection
            min_iou: Minimum IoU threshold (default from config)

        Returns:
            Best matching TrackedPerson or None
        """
        from .gait import GaitRecognitionService

        if min_iou is None:
            min_iou = self._config.iou_threshold

        self._ensure_camera(camera_source)

        best_track = None
        best_iou = min_iou

        for track in self._tracks[camera_source].values():
            if track.bbox == [0, 0, 0, 0]:
                continue

            iou = GaitRecognitionService.compute_iou(track.bbox, pose_bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track

        return best_track

    def find_track_containing_bbox(
        self,
        camera_source: str,
        inner_bbox: list[int],
    ) -> Optional[TrackedPerson]:
        """
        Find track whose bbox contains the inner bbox (e.g., face inside body).

        Uses center point containment - checks if the center of inner_bbox
        is inside any track's bbox. This works better than IoU for faces
        which are contained inside body bboxes.

        Args:
            camera_source: Camera identifier
            inner_bbox: [x1, y1, x2, y2] of smaller bbox (e.g., face)

        Returns:
            Best matching TrackedPerson or None
        """
        self._ensure_camera(camera_source)

        # Calculate center of inner bbox
        inner_cx = (inner_bbox[0] + inner_bbox[2]) / 2
        inner_cy = (inner_bbox[1] + inner_bbox[3]) / 2

        for track in self._tracks[camera_source].values():
            if track.bbox == [0, 0, 0, 0]:
                continue

            # Check if center is inside track bbox
            tx1, ty1, tx2, ty2 = track.bbox
            if tx1 <= inner_cx <= tx2 and ty1 <= inner_cy <= ty2:
                return track

        return None

    def get_active_tracks(
        self,
        camera_source: str,
    ) -> list[TrackedPerson]:
        """Get all active tracks for a camera."""
        self._ensure_camera(camera_source)
        return list(self._tracks[camera_source].values())

    def get_identified_tracks(
        self,
        camera_source: str,
    ) -> list[TrackedPerson]:
        """Get tracks with identified persons."""
        return [
            t for t in self.get_active_tracks(camera_source)
            if t.person_id is not None
        ]

    def remove_track(
        self,
        camera_source: str,
        track_id: int,
    ) -> None:
        """Remove a track."""
        self._ensure_camera(camera_source)
        self._tracks[camera_source].pop(track_id, None)

    def cleanup_stale_tracks(
        self,
        max_age: Optional[float] = None,
    ) -> int:
        """
        Remove tracks not seen recently.

        Args:
            max_age: Max seconds since last seen (default from config)

        Returns:
            Number of tracks removed
        """
        if max_age is None:
            max_age = self._config.track_timeout

        now = time.time()
        removed = 0

        for camera_source in list(self._tracks.keys()):
            stale_ids = [
                track_id
                for track_id, track in self._tracks[camera_source].items()
                if now - track.last_seen > max_age
            ]
            for track_id in stale_ids:
                self._tracks[camera_source].pop(track_id)
                removed += 1
                logger.debug(
                    "Removed stale track %d from %s",
                    track_id,
                    camera_source,
                )

        return removed

    def get_track_summary(self, camera_source: str) -> dict:
        """Get summary of tracks for a camera."""
        tracks = self.get_active_tracks(camera_source)
        identified = [t for t in tracks if t.person_id]
        with_gait = [t for t in identified if t.gait_similarity > 0]

        return {
            "total_tracks": len(tracks),
            "identified": len(identified),
            "with_gait": len(with_gait),
            "persons": [
                {
                    "track_id": t.track_id,
                    "name": t.person_name,
                    "face": t.face_similarity,
                    "gait": t.gait_similarity,
                    "combined": t.combined_similarity,
                }
                for t in identified
            ],
        }


# Singleton instance
_manager: Optional[TrackManager] = None


def get_track_manager() -> TrackManager:
    """Get the track manager singleton."""
    global _manager
    if _manager is None:
        _manager = TrackManager()
    return _manager
