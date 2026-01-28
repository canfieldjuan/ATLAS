"""
Camera-based presence detection consumer.

Integrates directly with atlas_vision's track_store to receive person detection
events and feed them to the PresenceService.

This is different from atlas_brain's version which used VisionSubscriber (MQTT).
Here we integrate directly since we're already in atlas_vision.
"""

import logging
from typing import Optional

from .service import PresenceService, get_presence_service
from .config import presence_config

logger = logging.getLogger("atlas.vision.presence.camera")


class CameraPresenceConsumer:
    """
    Consumes track events and updates presence state.

    Registers as a callback with track_store to receive real-time
    person detection events from cameras.
    """

    def __init__(
        self,
        presence_service: Optional[PresenceService] = None,
    ):
        self.presence_service = presence_service
        self._registered = False

        # Track active persons per camera to detect "person left"
        self._active_tracks: dict[str, set[int]] = {}  # source_id -> set of track_ids

    @property
    def is_registered(self) -> bool:
        """Check if registered with track_store."""
        return self._registered

    async def register_with_track_store(self) -> bool:
        """
        Register as a callback with the track_store.

        Returns True if registration succeeded.
        """
        if self._registered:
            return True

        try:
            from ..processing.tracking import get_track_store
            track_store = get_track_store()

            if track_store is None:
                logger.warning("TrackStore not available")
                return False

            track_store.register_callback(self._handle_track_event)
            self._registered = True
            logger.info("Registered with TrackStore for presence detection")
            return True

        except ImportError as e:
            logger.warning("Could not import TrackStore: %s", e)
            return False
        except Exception as e:
            logger.error("Failed to register with TrackStore: %s", e)
            return False

    async def _handle_track_event(self, event) -> None:
        """
        Handle a track event from the track_store.

        Args:
            event: TrackEvent from atlas_vision.processing.tracking
        """
        # Only care about person detections
        if event.class_name != "person":
            return

        # Lazy load presence service
        if self.presence_service is None:
            self.presence_service = get_presence_service()

        source_id = event.source_id
        track_id = event.track_id
        event_type = event.event_type  # "new_track", "track_lost", "track_update"

        if source_id not in self._active_tracks:
            self._active_tracks[source_id] = set()

        if event_type == "new_track":
            # Person appeared in camera
            self._active_tracks[source_id].add(track_id)
            logger.debug(
                "Person detected: camera=%s, track=%d",
                source_id,
                track_id,
            )
            await self.presence_service.handle_camera_detection(
                camera_source=source_id,
                person_detected=True,
                track_id=track_id,
                confidence=event.confidence if hasattr(event, 'confidence') else 0.85,
            )

        elif event_type == "track_lost":
            # Person left camera view
            self._active_tracks[source_id].discard(track_id)
            logger.debug(
                "Person lost: camera=%s, track=%d, remaining=%d",
                source_id,
                track_id,
                len(self._active_tracks[source_id]),
            )

            # Only signal "person left" if no other tracks in this camera
            if not self._active_tracks[source_id]:
                await self.presence_service.handle_camera_detection(
                    camera_source=source_id,
                    person_detected=False,
                    track_id=track_id,
                )

        elif event_type == "track_update":
            # Person still visible - update presence timestamp
            if track_id in self._active_tracks[source_id]:
                await self.presence_service.handle_camera_detection(
                    camera_source=source_id,
                    person_detected=True,
                    track_id=track_id,
                    confidence=event.confidence if hasattr(event, 'confidence') else 0.85,
                )


# Singleton
_camera_consumer: Optional[CameraPresenceConsumer] = None


async def start_camera_presence_consumer() -> Optional[CameraPresenceConsumer]:
    """Start the camera presence consumer singleton."""
    global _camera_consumer

    if not presence_config.camera_enabled:
        logger.info("Camera presence detection disabled")
        return None

    if _camera_consumer is None:
        _camera_consumer = CameraPresenceConsumer()

    success = await _camera_consumer.register_with_track_store()
    if success:
        return _camera_consumer
    return None


def get_camera_consumer() -> Optional[CameraPresenceConsumer]:
    """Get the camera consumer singleton."""
    return _camera_consumer
