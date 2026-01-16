"""
Camera-based presence detection consumer.

Integrates with the existing VisionSubscriber to receive person detection
events and feed them to the PresenceService.
"""

import asyncio
import logging
from typing import Optional

from .service import PresenceService, get_presence_service
from .config import presence_config

logger = logging.getLogger("atlas.presence.camera")


class CameraPresenceConsumer:
    """
    Consumes vision events and updates presence state.

    Registers as a callback with VisionSubscriber to receive real-time
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

    async def register_with_vision_subscriber(self) -> bool:
        """
        Register as a callback with the VisionSubscriber.

        Returns True if registration succeeded.
        """
        if self._registered:
            return True

        try:
            from ..vision.subscriber import get_vision_subscriber
            subscriber = get_vision_subscriber()

            if subscriber is None:
                logger.warning("VisionSubscriber not available")
                return False

            subscriber.register_event_callback(self._handle_vision_event)
            self._registered = True
            logger.info("Registered with VisionSubscriber for presence detection")
            return True

        except ImportError as e:
            logger.warning("Could not import VisionSubscriber: %s", e)
            return False
        except Exception as e:
            logger.error("Failed to register with VisionSubscriber: %s", e)
            return False

    async def _handle_vision_event(self, event) -> None:
        """
        Handle a vision event from the subscriber.

        Args:
            event: VisionEvent from atlas_brain.vision.models
        """
        # Only care about person detections
        if event.class_name != "person":
            return

        # Lazy load presence service
        if self.presence_service is None:
            self.presence_service = get_presence_service()

        source_id = event.source_id
        track_id = event.track_id
        event_type = event.event_type.value  # "new_track", "track_lost"

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
                confidence=0.85,  # Default confidence for YOLO detection
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

    success = await _camera_consumer.register_with_vision_subscriber()
    if success:
        return _camera_consumer
    return None


def get_camera_consumer() -> Optional[CameraPresenceConsumer]:
    """Get the camera consumer singleton."""
    return _camera_consumer
