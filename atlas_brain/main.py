"""
Atlas Brain - Central Intelligence Server

The main FastAPI application entry point.
"""

# Load .env file FIRST, before any other imports
from pathlib import Path
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path, override=True)

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from .api import router as api_router
from .config import settings
from .services import vlm_registry, llm_registry, vos_registry
from .storage import db_settings
from .storage.database import init_database, close_database

# Voice pipeline managed by voice.launcher module

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

    # Note: STT/TTS registries not implemented - voice uses Piper TTS directly
    # via voice/pipeline.py. These can be added later if centralized
    # STT/TTS management is needed.

    # Load default LLM if configured
    if settings.load_llm_on_startup:
        try:
            backend = settings.llm.default_model
            logger.info("Loading default LLM backend: %s", backend)

            if backend == "transformers-flash":
                # Transformers with Flash Attention
                kwargs = {
                    "model_id": settings.llm.hf_model_id,
                    "torch_dtype": settings.llm.torch_dtype,
                    "use_flash_attention": settings.llm.use_flash_attention,
                }
                if settings.llm.max_memory_gb:
                    kwargs["max_memory_gb"] = settings.llm.max_memory_gb
                logger.info("HF Model: %s, dtype: %s, flash_attn: %s",
                           settings.llm.hf_model_id,
                           settings.llm.torch_dtype,
                           settings.llm.use_flash_attention)
            elif backend == "ollama":
                # Ollama API backend
                kwargs = {
                    "model": settings.llm.ollama_model,
                    "base_url": settings.llm.ollama_url,
                }
                logger.info("Ollama model: %s, url: %s",
                           settings.llm.ollama_model,
                           settings.llm.ollama_url)
            elif backend == "together":
                # Together AI cloud backend
                kwargs = {
                    "model": settings.llm.together_model,
                }
                if settings.llm.together_api_key:
                    kwargs["api_key"] = settings.llm.together_api_key
                logger.info("Together AI model: %s", settings.llm.together_model)
            else:
                # llama-cpp (GGUF models)
                kwargs = {}
                if settings.llm.model_path:
                    kwargs["model_path"] = Path(settings.llm.model_path)
                if settings.llm.n_ctx:
                    kwargs["n_ctx"] = settings.llm.n_ctx
                if settings.llm.n_gpu_layers is not None:
                    kwargs["n_gpu_layers"] = settings.llm.n_gpu_layers

            llm_registry.activate(backend, **kwargs)
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.error("Failed to load default LLM: %s", e)

    # Note: Speaker ID loaded lazily via get_speaker_id_service() when voice
    # pipeline starts. No registry needed - single Resemblyzer implementation.

    # Load VOS if enabled
    if settings.load_vos_on_startup or settings.vos.enabled:
        try:
            logger.info("Loading VOS: %s", settings.vos.default_model)
            vos_registry.activate(
                settings.vos.default_model,
                device=settings.vos.device,
                dtype=settings.vos.dtype,
            )
            logger.info("VOS loaded successfully")
        except Exception as e:
            logger.error("Failed to load VOS: %s", e)

    # Note: Omni (speech-to-speech) not yet implemented as registry.
    # Future: Add omni_registry for Qwen2-Audio or similar models.

    # Note: FunctionGemma tool router was in pipecat module (now removed).
    # Tool routing handled by services/intent_router.py instead.

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

    # Initialize Roku devices if enabled
    try:
        from .capabilities.roku import init_roku
        roku_devices = await init_roku()
        if roku_devices:
            logger.info("Registered Roku devices: %s", roku_devices)
    except Exception as e:
        logger.error("Failed to initialize Roku: %s", e)

    # Initialize device discovery service
    if settings.discovery.enabled:
        try:
            from .discovery import init_discovery, run_discovery_scan, get_discovery_service
            await init_discovery()

            # Run initial scan if configured
            if settings.discovery.scan_on_startup:
                discovered = await run_discovery_scan(timeout=settings.discovery.scan_timeout)
                logger.info(
                    "Discovery scan complete: found %d devices",
                    len(discovered),
                )
                for device in discovered:
                    logger.info(
                        "  - %s: %s (%s) at %s",
                        device.device_type,
                        device.name,
                        device.device_id,
                        device.host,
                    )

            # Start periodic scanning if interval > 0
            if settings.discovery.scan_interval_seconds > 0:
                service = get_discovery_service()
                await service.start_periodic_scan()

        except Exception as e:
            logger.error("Failed to initialize discovery service: %s", e)

    # Initialize centralized alert system
    alert_manager = None
    if settings.alerts.enabled:
        try:
            from .alerts import get_alert_manager, setup_default_callbacks, NtfyDelivery

            alert_manager = get_alert_manager()
            setup_default_callbacks(alert_manager)

            # Note: TTS delivery via WebSocket removed - use Pipecat voice pipeline instead
            if settings.alerts.tts_enabled:
                logger.info("TTS alerts will be delivered via Pipecat voice pipeline")

            if settings.alerts.ntfy_enabled:
                ntfy_delivery = NtfyDelivery(
                    base_url=settings.alerts.ntfy_url,
                    topic=settings.alerts.ntfy_topic,
                )
                alert_manager.register_callback(ntfy_delivery.deliver)
                logger.info("ntfy push notifications enabled (%s/%s)",
                           settings.alerts.ntfy_url, settings.alerts.ntfy_topic)

            logger.info("Centralized alerts enabled with %d rules", len(alert_manager.list_rules()))
        except Exception as e:
            logger.error("Failed to initialize alert system: %s", e)

    # Initialize reminder service
    reminder_service = None
    if settings.reminder.enabled:
        try:
            from .services.reminders import initialize_reminder_service

            reminder_service = await initialize_reminder_service()

            # Simple reminder callback - logs reminder (TTS delivery via Pipecat)
            async def reminder_alert_callback(reminder):
                """Log reminder - TTS delivery handled by Pipecat voice pipeline."""
                message = f"Reminder: {reminder.message}"
                logger.info("REMINDER: %s", message)
                # Note: For TTS delivery, use Pipecat voice pipeline
                # The reminder will be announced when user interacts with Atlas

            reminder_service.set_alert_callback(reminder_alert_callback)
            logger.info("Reminder service initialized with %d pending", reminder_service.pending_count)
        except Exception as e:
            logger.error("Failed to initialize reminder service: %s", e)

    # Start vision event subscriber if MQTT is enabled
    vision_subscriber = None
    if settings.mqtt.enabled:
        try:
            from .vision import get_vision_subscriber

            vision_subscriber = get_vision_subscriber()
            if await vision_subscriber.start():
                logger.info("Vision subscriber started, listening for detection events")
            else:
                logger.warning("Failed to start vision subscriber")
                vision_subscriber = None
        except Exception as e:
            logger.error("Failed to initialize vision subscriber: %s", e)
            vision_subscriber = None

    # NOTE: Presence service moved to atlas_vision
    # The presence module now proxies to atlas_vision API
    # Configure atlas_vision with ATLAS_VISION_PRESENCE_ENABLED=true
    if settings.presence_enabled:
        logger.info("Presence tracking enabled via atlas_vision proxy")

    # Initialize communications service if enabled
    comms_service = None
    try:
        from .comms import comms_settings, init_comms_service
        if comms_settings.enabled:
            comms_service = await init_comms_service()
            if comms_service:
                logger.info("Communications service initialized with provider: %s",
                           comms_service.provider.name if comms_service.provider else "none")
            else:
                logger.warning("Communications service failed to initialize")
    except Exception as e:
        logger.error("Failed to initialize communications service: %s", e)

    # Start voice pipeline if enabled
    if settings.voice.enabled:
        try:
            from .voice.launcher import start_voice_pipeline
            loop = asyncio.get_event_loop()
            if start_voice_pipeline(loop):
                logger.info("Voice pipeline started")
            else:
                logger.warning("Voice pipeline failed to start")
        except ImportError as e:
            logger.warning("Voice pipeline not available: %s", e)
        except Exception as e:
            logger.error("Failed to start voice pipeline: %s", e)

    # NOTE: Webcam and RTSP detection moved to atlas_vision service
    # Detection events are received via MQTT subscriber (vision/subscriber.py)
    # Configure atlas_vision with ATLAS_VISION_MQTT_ENABLED=true

    logger.info("Atlas Brain startup complete")

    yield  # Application runs here

    # --- Shutdown ---
    logger.info("Atlas Brain shutting down...")

    # NOTE: Presence service runs in atlas_vision, no local shutdown needed

    # Stop vision subscriber
    if vision_subscriber and vision_subscriber.is_running:
        try:
            await vision_subscriber.stop()
            logger.info("Vision subscriber stopped")
        except Exception as e:
            logger.error("Error stopping vision subscriber: %s", e)

    # Stop voice pipeline
    if settings.voice.enabled:
        try:
            from .voice.launcher import stop_voice_pipeline
            stop_voice_pipeline()
            logger.info("Voice pipeline stopped")
        except Exception as e:
            logger.error("Error stopping voice pipeline: %s", e)

    # Shutdown discovery service
    if settings.discovery.enabled:
        try:
            from .discovery import shutdown_discovery
            await shutdown_discovery()
            logger.info("Discovery service shutdown complete")
        except Exception as e:
            logger.error("Error shutting down discovery service: %s", e)

    # Shutdown reminder service
    if reminder_service:
        try:
            from .services.reminders import shutdown_reminder_service
            await shutdown_reminder_service()
            logger.info("Reminder service shutdown complete")
        except Exception as e:
            logger.error("Error shutting down reminder service: %s", e)

    # Disconnect Home Assistant backend
    try:
        from .capabilities.homeassistant import shutdown_homeassistant
        await shutdown_homeassistant()
    except Exception as e:
        logger.error("Error shutting down Home Assistant: %s", e)

    # Disconnect Roku devices
    try:
        from .capabilities.roku import shutdown_roku
        await shutdown_roku()
    except Exception as e:
        logger.error("Error shutting down Roku: %s", e)

    # Shutdown communications service
    if comms_service:
        try:
            from .comms import shutdown_comms_service
            await shutdown_comms_service()
            logger.info("Communications service shutdown complete")
        except Exception as e:
            logger.error("Error shutting down communications: %s", e)

    # Close database connection pool
    if db_settings.enabled:
        try:
            await close_database()
            logger.info("Database connection pool closed")
        except Exception as e:
            logger.error("Error closing database: %s", e)

    # Unload models to free resources
    vos_registry.deactivate()
    vlm_registry.deactivate()
    llm_registry.deactivate()

    # Force garbage collection to clean up semaphores from NeMo/PyTorch
    import gc
    gc.collect()

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
