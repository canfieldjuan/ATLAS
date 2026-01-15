"""
FastAPI application for Atlas Vision.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..core.config import settings

logger = logging.getLogger("atlas.vision.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Atlas Vision starting on %s:%d", settings.server.host, settings.server.port)

    # Initialize device registry with mock cameras
    from ..devices.registry import device_registry
    from ..devices.cameras.mock import create_mock_cameras

    for camera in create_mock_cameras():
        device_registry.register(camera)
        logger.info("Registered camera: %s", camera.name)

    # Start mDNS announcer for discovery
    announcer = None
    if settings.discovery.enabled:
        from ..communication import get_node_announcer
        announcer = get_node_announcer()
        if await announcer.start():
            logger.info("mDNS discovery announcer started")
        else:
            logger.warning("Failed to start mDNS announcer")

    # Start detection pipeline
    detection_pipeline = None
    if settings.detection.enabled:
        from ..processing.pipeline import get_detection_pipeline
        detection_pipeline = get_detection_pipeline()
        if await detection_pipeline.start():
            logger.info("Detection pipeline started")
        else:
            logger.warning("Detection pipeline failed to start (model may not be available)")

    # Start MQTT publisher for events
    mqtt_publisher = None
    if settings.mqtt.enabled:
        from ..communication import get_mqtt_publisher
        from ..processing.tracking import get_track_store

        mqtt_publisher = get_mqtt_publisher()
        if await mqtt_publisher.connect():
            logger.info("MQTT publisher connected to %s:%d", settings.mqtt.host, settings.mqtt.port)

            # Register MQTT publisher as event callback
            track_store = get_track_store()
            track_store.register_callback(mqtt_publisher.publish_event)
            logger.info("MQTT event publishing enabled")
        else:
            logger.warning("Failed to connect MQTT publisher")
            mqtt_publisher = None

    yield

    # Shutdown
    logger.info("Atlas Vision shutting down")

    # Stop MQTT publisher
    if mqtt_publisher and mqtt_publisher.is_connected:
        await mqtt_publisher.disconnect()
        logger.info("MQTT publisher disconnected")

    # Stop detection pipeline
    if detection_pipeline and detection_pipeline.is_running:
        await detection_pipeline.stop()
        logger.info("Detection pipeline stopped")

    # Stop mDNS announcer
    if announcer and announcer.is_running:
        await announcer.stop()
        logger.info("mDNS announcer stopped")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    application = FastAPI(
        title="Atlas Vision",
        description="Video processing and node management for Atlas",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Include routers
    from .health import router as health_router
    from .cameras import router as cameras_router
    from .detections import router as detections_router
    from .security import router as security_router
    from .tracks import router as tracks_router

    application.include_router(health_router, tags=["health"])
    application.include_router(cameras_router, prefix="/cameras", tags=["cameras"])
    application.include_router(detections_router, tags=["detections"])
    application.include_router(security_router, prefix="/security", tags=["security"])
    application.include_router(tracks_router, prefix="/tracks", tags=["tracks"])

    return application


# Create app instance
app = create_app()
