"""
Atlas Brain - Central Intelligence Server

The main FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from .api import router as api_router
from .config import settings
from .services import vlm_registry, stt_registry, llm_registry, tts_registry
from .storage import db_settings
from .storage.database import init_database, close_database

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

    # Initialize database connection pool
    if db_settings.enabled:
        try:
            await init_database()
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error("Failed to initialize database: %s", e)
            # Continue without database - service can still function
            # but conversation persistence will be unavailable

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

    # Load default TTS if configured
    if settings.load_tts_on_startup:
        try:
            logger.info("Loading default TTS: %s (voice=%s)", settings.tts.default_model, settings.tts.voice)
            tts_registry.activate(settings.tts.default_model, voice=settings.tts.voice)
        except Exception as e:
            logger.error("Failed to load default TTS: %s", e)

    # Load default LLM if configured
    if settings.load_llm_on_startup:
        try:
            logger.info("Loading default LLM: %s", settings.llm.default_model)
            kwargs = {}
            if settings.llm.model_path:
                kwargs["model_path"] = Path(settings.llm.model_path)
            if settings.llm.n_ctx:
                kwargs["n_ctx"] = settings.llm.n_ctx
            if settings.llm.n_gpu_layers is not None:
                kwargs["n_gpu_layers"] = settings.llm.n_gpu_layers
            llm_registry.activate(settings.llm.default_model, **kwargs)
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.error("Failed to load default LLM: %s", e)

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

    # Close database connection pool
    if db_settings.enabled:
        try:
            await close_database()
            logger.info("Database connection pool closed")
        except Exception as e:
            logger.error("Error closing database: %s", e)

    # Unload models to free resources
    vlm_registry.deactivate()
    stt_registry.deactivate()
    tts_registry.deactivate()
    llm_registry.deactivate()

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
