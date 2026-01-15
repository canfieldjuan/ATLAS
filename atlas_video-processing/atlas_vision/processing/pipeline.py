"""
Detection pipeline for real-time object tracking.

Continuously processes frames from cameras and updates
the track store with detection results.
"""

import asyncio
import logging
from typing import Optional

from ..core.config import settings
from ..devices.registry import device_registry
from ..core.constants import DeviceType
from .detection import get_yolo_detector
from .tracking import get_track_store
from .frame_buffer import get_frame_buffer

logger = logging.getLogger("atlas.vision.processing.pipeline")


class DetectionPipeline:
    """
    Real-time detection pipeline.

    Processes frames from all connected cameras and
    updates the track store with detection results.
    """

    def __init__(self):
        self._running = False
        self._tasks: dict[str, asyncio.Task] = {}
        self._detector = None
        self._store = None

    async def start(self) -> bool:
        """
        Start the detection pipeline.

        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning("Pipeline already running")
            return True

        if not settings.detection.enabled:
            logger.info("Detection disabled in settings")
            return False

        # Initialize detector
        self._detector = get_yolo_detector()
        if not self._detector.load():
            logger.error("Failed to load YOLO model")
            return False

        self._store = get_track_store()
        self._running = True

        # Start processing task for each camera
        cameras = device_registry.list_by_type(DeviceType.CAMERA)
        for camera in cameras:
            task = asyncio.create_task(self._process_camera(camera.device_id))
            self._tasks[camera.device_id] = task
            logger.info("Started detection for camera: %s", camera.device_id)

        logger.info("Detection pipeline started with %d cameras", len(cameras))
        return True

    async def stop(self) -> None:
        """Stop the detection pipeline."""
        if not self._running:
            return

        self._running = False

        # Cancel all tasks
        for device_id, task in self._tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.debug("Stopped detection for camera: %s", device_id)

        self._tasks.clear()

        # Unload model to free memory
        if self._detector:
            self._detector.unload()

        logger.info("Detection pipeline stopped")

    async def add_camera(self, device_id: str) -> None:
        """Start processing a new camera."""
        if not self._running:
            return

        if device_id in self._tasks:
            return

        task = asyncio.create_task(self._process_camera(device_id))
        self._tasks[device_id] = task
        logger.info("Added camera to pipeline: %s", device_id)

    async def remove_camera(self, device_id: str) -> None:
        """Stop processing a camera."""
        if device_id in self._tasks:
            self._tasks[device_id].cancel()
            try:
                await self._tasks[device_id]
            except asyncio.CancelledError:
                pass
            del self._tasks[device_id]
            logger.info("Removed camera from pipeline: %s", device_id)

    async def _process_camera(self, device_id: str) -> None:
        """
        Processing loop for a single camera.

        Continuously fetches frames and runs detection.
        """
        frame_buffer = get_frame_buffer()
        frame_interval = 1.0 / settings.detection.fps

        logger.info("Starting detection loop for %s (%.1f FPS)", device_id, settings.detection.fps)

        while self._running:
            try:
                # Get latest frame
                frame_obj = await frame_buffer.get_latest(device_id)

                if frame_obj is not None:
                    # Run tracking
                    tracks = await self._detector.track(frame_obj.data, device_id)

                    # Update store and generate events
                    events = await self._store.update_tracks(device_id, tracks)

                    # Broadcast updates via WebSocket
                    if tracks:
                        from ..api.tracks import broadcast_track_update
                        await broadcast_track_update(device_id, tracks)

                    # Broadcast events
                    for event in events:
                        from ..api.tracks import broadcast_event
                        await broadcast_event(event)

                # Wait for next frame interval
                await asyncio.sleep(frame_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Detection error for %s: %s", device_id, e)
                await asyncio.sleep(1.0)  # Back off on errors

        logger.debug("Detection loop ended for %s", device_id)

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running

    @property
    def active_cameras(self) -> list[str]:
        """Get list of cameras being processed."""
        return list(self._tasks.keys())


# Global pipeline instance
_pipeline: Optional[DetectionPipeline] = None


def get_detection_pipeline() -> DetectionPipeline:
    """Get or create the global detection pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = DetectionPipeline()
    return _pipeline


async def start_detection_pipeline() -> bool:
    """Start the detection pipeline (call from app startup)."""
    pipeline = get_detection_pipeline()
    return await pipeline.start()


async def stop_detection_pipeline() -> None:
    """Stop the detection pipeline (call from app shutdown)."""
    pipeline = get_detection_pipeline()
    await pipeline.stop()
