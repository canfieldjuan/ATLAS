"""
YOLO object detector with tracking support.

Uses Ultralytics YOLOv8 with built-in ByteTrack for
real-time object detection and tracking.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np

from ...core.config import settings
from ...core.models import (
    BoundingBox,
    ObjectDetection,
    Track,
    TrackPoint,
    TrackState,
    YOLO_CLASS_MAP,
)

logger = logging.getLogger("atlas.vision.detection.yolo")

# Lazy load ultralytics to avoid import overhead at startup
_YOLO = None


def _get_yolo_class():
    """Lazy load YOLO class."""
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO
    return _YOLO


class YOLODetector:
    """
    Real-time object detector using YOLOv8.

    Supports both detection-only mode and tracking mode
    with persistent object IDs across frames.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None,
        track_classes: Optional[list[str]] = None,
    ):
        """
        Initialize YOLO detector.

        Args:
            model_name: YOLO model name (e.g., yolov8n.pt, yolov8s.pt)
            confidence_threshold: Minimum confidence for detections
            device: Device to run on (auto, cuda, cpu)
            track_classes: List of class names to track
        """
        self.model_name = model_name or settings.detection.model
        self.confidence_threshold = confidence_threshold or settings.detection.confidence_threshold
        self.device = device or settings.detection.device
        self.track_classes = track_classes or settings.detection.track_classes

        self._model = None
        self._loaded = False

        # Per-source tracking state (required for multi-camera tracking)
        self._track_states: dict[str, dict[int, Track]] = {}

        # Build set of class IDs to track
        self._track_class_ids = self._build_class_ids()

        logger.info(
            "YOLODetector initialized: model=%s, device=%s, classes=%s",
            self.model_name,
            self.device,
            self.track_classes,
        )

    def _build_class_ids(self) -> set[int]:
        """Build set of YOLO class IDs to track based on configured class names."""
        class_ids = set()
        name_to_id = {v: k for k, v in YOLO_CLASS_MAP.items()}

        for class_name in self.track_classes:
            if class_name in name_to_id:
                class_ids.add(name_to_id[class_name])
            else:
                logger.warning("Unknown class name: %s", class_name)

        return class_ids

    def load(self) -> bool:
        """
        Load the YOLO model.

        Returns:
            True if loaded successfully
        """
        if self._loaded:
            return True

        try:
            YOLO = _get_yolo_class()

            # Determine device
            device = self.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Loading YOLO model %s on %s...", self.model_name, device)
            self._model = YOLO(self.model_name)
            self._model.to(device)

            self._loaded = True
            logger.info("YOLO model loaded successfully")
            return True

        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            return False

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            logger.info("YOLO model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    async def detect(
        self,
        frame: np.ndarray,
        source_id: str,
    ) -> list[ObjectDetection]:
        """
        Run object detection on a frame (no tracking).

        Args:
            frame: BGR image as numpy array
            source_id: Camera/source identifier

        Returns:
            List of detections
        """
        if not self._loaded:
            if not self.load():
                return []

        # Run inference in thread pool to avoid blocking
        results = await asyncio.to_thread(
            self._model.predict,
            frame,
            conf=self.confidence_threshold,
            verbose=False,
        )

        return self._parse_results(results, source_id, frame.shape)

    async def track(
        self,
        frame: np.ndarray,
        source_id: str,
    ) -> list[Track]:
        """
        Run object detection with tracking on a frame.

        Tracking maintains persistent object IDs across frames.

        Args:
            frame: BGR image as numpy array
            source_id: Camera/source identifier

        Returns:
            List of tracked objects with persistent IDs
        """
        if not self._loaded:
            if not self.load():
                return []

        # Initialize tracking state for this source if needed
        if source_id not in self._track_states:
            self._track_states[source_id] = {}

        # Run tracking inference in thread pool
        results = await asyncio.to_thread(
            self._model.track,
            frame,
            conf=self.confidence_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        return self._parse_tracking_results(results, source_id, frame.shape)

    def _parse_results(
        self,
        results: Any,
        source_id: str,
        frame_shape: tuple,
    ) -> list[ObjectDetection]:
        """Parse YOLO results into ObjectDetection objects."""
        detections = []
        timestamp = datetime.utcnow()
        img_height, img_width = frame_shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())

                # Skip classes we don't care about
                if class_id not in self._track_class_ids:
                    continue

                confidence = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                class_name = YOLO_CLASS_MAP.get(class_id, f"class_{class_id}")

                detection = ObjectDetection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox.from_xyxy(x1, y1, x2, y2, img_width, img_height),
                    source_id=source_id,
                    timestamp=timestamp,
                )
                detections.append(detection)

        return detections

    def _parse_tracking_results(
        self,
        results: Any,
        source_id: str,
        frame_shape: tuple,
    ) -> list[Track]:
        """Parse YOLO tracking results into Track objects."""
        tracks = []
        timestamp = datetime.utcnow()
        img_height, img_width = frame_shape[:2]
        track_state = self._track_states[source_id]

        current_track_ids = set()

        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())

                # Skip classes we don't care about
                if class_id not in self._track_class_ids:
                    continue

                track_id = int(boxes.id[i].item())
                confidence = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                class_name = YOLO_CLASS_MAP.get(class_id, f"class_{class_id}")

                bbox = BoundingBox.from_xyxy(x1, y1, x2, y2, img_width, img_height)
                center = bbox.center

                current_track_ids.add(track_id)

                # Update or create track
                if track_id in track_state:
                    # Update existing track
                    track = track_state[track_id]
                    old_center = track.bbox.center
                    dt = (timestamp - track.last_seen).total_seconds()

                    # Calculate velocity
                    if dt > 0:
                        vx = (center[0] - old_center[0]) / dt
                        vy = (center[1] - old_center[1]) / dt
                        track.velocity = (vx, vy)

                    track.bbox = bbox
                    track.confidence = confidence
                    track.last_seen = timestamp
                    track.frame_count += 1

                    # Add to path history
                    track.path.append(TrackPoint(x=center[0], y=center[1], timestamp=timestamp))
                    # Limit path history
                    if len(track.path) > settings.detection.max_track_history:
                        track.path = track.path[-settings.detection.max_track_history:]

                    # Update state
                    if track.frame_count >= 3:
                        track.state = TrackState.CONFIRMED

                else:
                    # Create new track
                    track = Track(
                        track_id=track_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        source_id=source_id,
                        state=TrackState.TENTATIVE,
                        first_seen=timestamp,
                        last_seen=timestamp,
                        frame_count=1,
                        path=[TrackPoint(x=center[0], y=center[1], timestamp=timestamp)],
                    )
                    track_state[track_id] = track

                tracks.append(track)

        # Mark lost tracks
        for track_id, track in list(track_state.items()):
            if track_id not in current_track_ids:
                time_since_seen = (timestamp - track.last_seen).total_seconds()
                if time_since_seen > settings.detection.track_timeout:
                    # Remove old track
                    del track_state[track_id]
                else:
                    # Mark as lost
                    track.state = TrackState.LOST

        return tracks

    def get_active_tracks(self, source_id: Optional[str] = None) -> list[Track]:
        """
        Get all currently active tracks.

        Args:
            source_id: Optional filter by source

        Returns:
            List of active tracks
        """
        tracks = []

        sources = [source_id] if source_id else list(self._track_states.keys())

        for src in sources:
            if src in self._track_states:
                for track in self._track_states[src].values():
                    if track.state != TrackState.LOST:
                        tracks.append(track)

        return tracks

    def get_track(self, track_id: int, source_id: Optional[str] = None) -> Optional[Track]:
        """
        Get a specific track by ID.

        Args:
            track_id: Track ID to find
            source_id: Optional source to search in

        Returns:
            Track if found, None otherwise
        """
        sources = [source_id] if source_id else list(self._track_states.keys())

        for src in sources:
            if src in self._track_states:
                if track_id in self._track_states[src]:
                    return self._track_states[src][track_id]

        return None

    def clear_tracks(self, source_id: Optional[str] = None) -> None:
        """Clear tracking state for a source (or all sources)."""
        if source_id:
            if source_id in self._track_states:
                self._track_states[source_id].clear()
        else:
            self._track_states.clear()


# Global detector instance
_detector: Optional[YOLODetector] = None


def get_yolo_detector() -> YOLODetector:
    """Get or create the global YOLO detector."""
    global _detector
    if _detector is None:
        _detector = YOLODetector()
    return _detector
