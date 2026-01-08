"""
Atlas Brain - Central Intelligence Server

The main FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api import router as api_router
from .config import settings
from .services import vlm_registry, stt_registry

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("atlas.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Handles startup (model loading) and shutdown (cleanup).
    """
    # --- Startup ---
    logger.info("Atlas Brain starting up...")

    # Load default VLM if configured
    if settings.load_vlm_on_startup:
        try:
            logger.info("Loading default VLM: %s", settings.vlm.default_model)
            vlm_registry.activate(settings.vlm.default_model)
        except Exception as e:
            logger.error("Failed to load default VLM: %s", e)

    # Load default STT if configured
    if settings.load_stt_on_startup:
        try:
            logger.info("Loading default STT: %s", settings.stt.default_model)
            stt_registry.activate(
                settings.stt.default_model,
                model_size=settings.stt.whisper_model_size,
            )
        except Exception as e:
            logger.error("Failed to load default STT: %s", e)

    # Register test devices for development
    try:
        from .capabilities.devices import register_test_devices
        device_ids = register_test_devices()
        logger.info("Registered test devices: %s", device_ids)
    except Exception as e:
        logger.error("Failed to register test devices: %s", e)

    # Initialize Home Assistant backend if enabled
    try:
        from .capabilities.homeassistant import init_homeassistant
        ha_devices = await init_homeassistant()
        if ha_devices:
            logger.info("Registered Home Assistant devices: %s", ha_devices)
    except Exception as e:
        logger.error("Failed to initialize Home Assistant: %s", e)

    logger.info("Atlas Brain startup complete")

    yield  # Application runs here

    # --- Shutdown ---
    logger.info("Atlas Brain shutting down...")

    # Disconnect Home Assistant backend
    try:
        from .capabilities.homeassistant import shutdown_homeassistant
        await shutdown_homeassistant()
    except Exception as e:
        logger.error("Error shutting down Home Assistant: %s", e)

    # Unload models to free resources
    vlm_registry.deactivate()
    stt_registry.deactivate()

    logger.info("Atlas Brain shutdown complete")


# Create the FastAPI application
app = FastAPI(
    title="Atlas Brain",
    description="The central intelligence server for the Atlas project.",
    version="0.2.0",
    lifespan=lifespan,
)

# Include API routers with /api/v1 prefix
app.include_router(api_router, prefix="/api/v1")
