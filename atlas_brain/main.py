"""
Atlas Brain - Central Intelligence Server

The main FastAPI application entry point.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from .api import router as api_router
from .config import settings
from .services import vlm_registry, stt_registry, llm_registry, tts_registry, vos_registry
from .storage import db_settings
from .storage.database import init_database, close_database

# Global voice client task
_voice_client_task: Optional[asyncio.Task] = None

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

    # Load speaker ID if enabled
    if settings.load_speaker_id_on_startup or settings.speaker_id.enabled:
        try:
            from .services import speaker_id_registry
            logger.info("Loading speaker ID: %s", settings.speaker_id.default_model)
            speaker_id_registry.activate(settings.speaker_id.default_model)
            logger.info("Speaker ID loaded successfully")
        except Exception as e:
            logger.error("Failed to load speaker ID: %s", e)

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

    # Initialize multi-model pool for fast routing
    try:
        from .services.model_pool import initialize_pool, ModelTier
        pool = await initialize_pool([ModelTier.FAST, ModelTier.BALANCED])
        logger.info("Model pool initialized: %s", pool.get_available_tiers())
    except Exception as e:
        logger.warning("Model pool initialization failed (will use single model): %s", e)

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

    # Start voice client if enabled
    global _voice_client_task
    if settings.voice.enabled:
        try:
            from atlas_voice.runner import VoiceRunner, RunnerConfig

            voice_config = RunnerConfig(
                atlas_url="ws://localhost:8000/api/v1/ws/orchestrated",
                input_device=settings.voice.input_device,
                output_device=settings.voice.output_device,
                sample_rate=settings.voice.sample_rate,
                require_wake_word=settings.voice.require_wake_word,
            )
            runner = VoiceRunner(voice_config)

            async def run_voice_client():
                try:
                    await runner.run()
                except asyncio.CancelledError:
                    logger.info("Voice client task cancelled")
                except Exception as e:
                    logger.error("Voice client error: %s", e)

            _voice_client_task = asyncio.create_task(run_voice_client())
            logger.info("Voice client started (wake_word=%s)", settings.voice.require_wake_word)
        except ImportError as e:
            logger.warning("Voice client not available (atlas_voice not installed): %s", e)
        except Exception as e:
            logger.error("Failed to start voice client: %s", e)

    logger.info("Atlas Brain startup complete")

    yield  # Application runs here

    # --- Shutdown ---
    logger.info("Atlas Brain shutting down...")

    # Stop voice client
    if _voice_client_task and not _voice_client_task.done():
        logger.info("Stopping voice client...")
        _voice_client_task.cancel()
        try:
            await _voice_client_task
        except asyncio.CancelledError:
            pass
        logger.info("Voice client stopped")

    # Shutdown discovery service
    if settings.discovery.enabled:
        try:
            from .discovery import shutdown_discovery
            await shutdown_discovery()
            logger.info("Discovery service shutdown complete")
        except Exception as e:
            logger.error("Error shutting down discovery service: %s", e)

    # Shutdown model pool
    try:
        from .services.model_pool import get_model_pool
        pool = get_model_pool()
        await pool.shutdown()
        logger.info("Model pool shutdown complete")
    except Exception as e:
        logger.error("Error shutting down model pool: %s", e)

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
