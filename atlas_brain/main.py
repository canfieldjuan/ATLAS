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
            elif backend == "cloud":
                # Cloud LLM (Groq primary + Together fallback)
                kwargs = {}
                logger.info("Cloud LLM: Groq primary, Together fallback")
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

    # Load Gorilla tool router if enabled (for fast local tool routing)
    if settings.load_tool_router_on_startup:
        try:
            from .services.tool_router import get_tool_router
            logger.info("Loading Gorilla tool router...")
            await get_tool_router()
            logger.info("Gorilla tool router loaded successfully")
        except Exception as e:
            logger.error("Failed to load tool router: %s", e)

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

            # Note: TTS alerts via voice pipeline
            if settings.alerts.tts_enabled:
                logger.info("TTS alerts will be delivered via voice pipeline")

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

            # Simple reminder callback - logs reminder (TTS delivery via voice loop)
            async def reminder_alert_callback(reminder):
                """Log reminder - TTS delivery handled by voice pipeline."""
                message = f"Reminder: {reminder.message}"
                logger.info("REMINDER: %s", message)
                # Note: For TTS delivery, use voice pipeline
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

    # Initialize presence service for room-aware device control
    presence_service = None
    espresense_subscriber = None
    if settings.presence_enabled:
        try:
            from .presence import (
                get_presence_service,
                start_espresense_subscriber,
                start_camera_presence_consumer,
                presence_config,
            )

            # Start core presence service
            presence_service = get_presence_service()
            await presence_service.start()
            logger.info("Presence service started")

            # Start ESPresense subscriber if enabled
            if presence_config.espresense_enabled and settings.mqtt.enabled:
                espresense_subscriber = await start_espresense_subscriber(
                    mqtt_host=settings.mqtt.host,
                    mqtt_port=settings.mqtt.port,
                    mqtt_username=settings.mqtt.username,
                    mqtt_password=settings.mqtt.password,
                )
                if espresense_subscriber:
                    logger.info("ESPresense BLE tracking enabled")

            # Start camera presence consumer if vision is running
            if presence_config.camera_enabled and vision_subscriber:
                camera_consumer = await start_camera_presence_consumer()
                if camera_consumer:
                    logger.info("Camera presence detection enabled")

        except Exception as e:
            logger.error("Failed to initialize presence service: %s", e)

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

    # Start webcam person detector if enabled
    webcam_detector = None
    if settings.webcam.enabled:
        try:
            from .vision import start_webcam_detector

            webcam_detector = await start_webcam_detector(
                device_index=settings.webcam.device_index,
                device_name=settings.webcam.device_name,
                camera_source_id=settings.webcam.source_id,
                fps=settings.webcam.fps,
            )
            if webcam_detector:
                logger.info("Webcam detector started: device=%s (name=%s) -> %s",
                           webcam_detector.device_index, settings.webcam.device_name,
                           settings.webcam.source_id)
            else:
                logger.warning("Webcam detector failed to start")
        except Exception as e:
            logger.error("Failed to start webcam detector: %s", e)

    # Start RTSP camera detectors if enabled
    rtsp_manager = None
    if settings.rtsp.enabled:
        try:
            import json
            from .vision import start_rtsp_cameras

            cameras = []
            if settings.rtsp.cameras_json:
                cameras = json.loads(settings.rtsp.cameras_json)

            if cameras:
                rtsp_manager = await start_rtsp_cameras(cameras)
                logger.info("RTSP cameras started: %d cameras", len(cameras))
            else:
                logger.warning("RTSP enabled but no cameras configured")
        except Exception as e:
            logger.error("Failed to start RTSP cameras: %s", e)

    logger.info("Atlas Brain startup complete")

    yield  # Application runs here

    # --- Shutdown ---
    logger.info("Atlas Brain shutting down...")

    # Stop presence service
    if presence_service:
        try:
            from .presence import stop_espresense_subscriber
            await stop_espresense_subscriber()
            await presence_service.stop()
            logger.info("Presence service stopped")
        except Exception as e:
            logger.error("Error stopping presence service: %s", e)

    # Stop vision subscriber
    if vision_subscriber and vision_subscriber.is_running:
        try:
            await vision_subscriber.stop()
            logger.info("Vision subscriber stopped")
        except Exception as e:
            logger.error("Error stopping vision subscriber: %s", e)

    # Stop webcam detector
    if webcam_detector:
        try:
            from .vision import stop_webcam_detector
            await stop_webcam_detector()
            logger.info("Webcam detector stopped")
        except Exception as e:
            logger.error("Error stopping webcam detector: %s", e)

    # Stop RTSP camera detectors
    if rtsp_manager:
        try:
            from .vision import stop_rtsp_cameras
            await stop_rtsp_cameras()
            logger.info("RTSP cameras stopped")
        except Exception as e:
            logger.error("Error stopping RTSP cameras: %s", e)

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
