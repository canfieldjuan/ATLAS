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

    # Initialize database for recognition
    from ..storage import db_settings, init_database, close_database
    if db_settings.enabled:
        try:
            await init_database()
            logger.info("Database initialized")
        except Exception as e:
            logger.warning("Database initialization failed: %s", e)

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

    # Initialize presence service
    presence_service = None
    espresense_subscriber = None
    camera_consumer = None
    if settings.presence.enabled:
        from ..presence import (
            get_presence_service,
            start_espresense_subscriber,
            stop_espresense_subscriber,
            start_camera_presence_consumer,
            presence_config,
        )

        presence_service = get_presence_service()
        await presence_service.start()
        logger.info("Presence service started")

        # Start ESPresense subscriber if MQTT enabled
        if presence_config.espresense_enabled and settings.mqtt.enabled:
            espresense_subscriber = await start_espresense_subscriber(
                mqtt_host=settings.mqtt.host,
                mqtt_port=settings.mqtt.port,
                mqtt_username=settings.mqtt.username,
                mqtt_password=settings.mqtt.password,
            )
            if espresense_subscriber:
                logger.info("ESPresense BLE tracking enabled")

        # Start camera presence consumer if detection enabled
        if presence_config.camera_enabled and settings.detection.enabled:
            camera_consumer = await start_camera_presence_consumer()
            if camera_consumer:
                logger.info("Camera presence detection enabled")

    yield

    # Shutdown
    logger.info("Atlas Vision shutting down")

    # Stop presence service
    if presence_service and presence_service.is_running:
        if espresense_subscriber:
            from ..presence import stop_espresense_subscriber
            await stop_espresense_subscriber()
            logger.info("ESPresense subscriber stopped")
        await presence_service.stop()
        logger.info("Presence service stopped")

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

    # Close database
    if db_settings.enabled:
        try:
            await close_database()
            logger.info("Database closed")
        except Exception as e:
            logger.warning("Database close error: %s", e)


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
    from .recognition import router as recognition_router
    from .presence import router as presence_router

    application.include_router(health_router, tags=["health"])
    application.include_router(cameras_router, prefix="/cameras", tags=["cameras"])
    application.include_router(detections_router, tags=["detections"])
    application.include_router(security_router, prefix="/security", tags=["security"])
    application.include_router(tracks_router, prefix="/tracks", tags=["tracks"])
    application.include_router(recognition_router, prefix="/recognition", tags=["recognition"])
    application.include_router(presence_router, prefix="/presence", tags=["presence"])

    return application


# Create app instance
app = create_app()
