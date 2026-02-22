"""
Atlas Brain - Central Intelligence Server

The main FastAPI application entry point.
"""

# Load .env file FIRST, before any other imports
# .env.local overrides .env for machine-specific settings (API keys, ports, etc.)
from pathlib import Path
from dotenv import load_dotenv
_env_root = Path(__file__).parent.parent
load_dotenv(_env_root / ".env", override=True)
load_dotenv(_env_root / ".env.local", override=True)

import asyncio
import logging
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api import router as api_router
from .config import settings
from .services import llm_registry
from .storage import db_settings
from .storage.database import init_database, close_database

# Voice pipeline managed by voice.launcher module

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("atlas.main")


async def _start_asr_server() -> subprocess.Popen | None:
    """Start the ASR server as a subprocess if not already running."""
    import sys
    import httpx
    from urllib.parse import urlparse

    asr_url = settings.voice.asr_url
    parsed = urlparse(asr_url)
    base_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"

    # Check if ASR is already running
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url}/health")
            if resp.status_code == 200:
                logger.info("ASR server already running at %s", base_url)
                return None
    except Exception:
        pass

    # Find the asr_server.py script
    project_root = Path(__file__).parent.parent
    asr_script = project_root / "asr_server.py"
    if not asr_script.exists():
        logger.warning("asr_server.py not found at %s", asr_script)
        return None

    port = parsed.port or 8081
    model = settings.voice.asr_model
    device = settings.voice.asr_device

    logger.info("Starting ASR server: model=%s port=%d device=%s", model, port, device)

    proc = subprocess.Popen(
        [sys.executable, str(asr_script),
         "--model", model,
         "--port", str(port),
         "--device", device],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Wait for health check (up to 120s for model loading)
    logger.info("Waiting for ASR server to load model (pid=%d)...", proc.pid)
    for attempt in range(240):
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="replace")
            logger.error("ASR server exited during startup: %s", stderr[-500:])
            return None
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    logger.info("ASR server ready (took ~%ds)", attempt // 2)
                    return proc
        except Exception:
            pass
        await asyncio.sleep(0.5)

    logger.error("ASR server failed to become ready after 120s")
    proc.terminate()
    return None


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
            elif backend == "cloud":
                # Cloud backend (Groq primary + Together fallback)
                kwargs = {
                    "groq_model": settings.llm.groq_model,
                    "together_model": settings.llm.together_model,
                }
                if settings.llm.groq_api_key:
                    kwargs["groq_api_key"] = settings.llm.groq_api_key
                if settings.llm.together_api_key:
                    kwargs["together_api_key"] = settings.llm.together_api_key
                logger.info("Cloud LLM: Groq (%s) + Together (%s)",
                           settings.llm.groq_model, settings.llm.together_model)
            elif backend == "hybrid":
                # Hybrid: local (Ollama) for chat/streaming, cloud for tool calling
                kwargs = {
                    "local_kwargs": {
                        "model": settings.llm.ollama_model,
                        "base_url": settings.llm.ollama_url,
                    },
                    "cloud_kwargs": {
                        "groq_model": settings.llm.groq_model,
                        "together_model": settings.llm.together_model,
                    },
                }
                # Pass cloud API keys if configured
                if settings.llm.groq_api_key:
                    kwargs["cloud_kwargs"]["groq_api_key"] = settings.llm.groq_api_key
                if settings.llm.together_api_key:
                    kwargs["cloud_kwargs"]["together_api_key"] = settings.llm.together_api_key
                logger.info("Hybrid LLM: local=%s (%s), cloud=Groq+Together",
                           settings.llm.ollama_model, settings.llm.ollama_url)
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

    # Model swap startup check: if model_swap is enabled and it is currently daytime
    # (7 AM - midnight), ensure the night model is not occupying VRAM.
    # Runs as a background task so startup is not blocked.
    if settings.llm.model_swap_enabled and settings.llm.default_model == "ollama":
        try:
            from datetime import datetime as _dt
            from .services.llm.model_manager import unload_model as _unload

            _now_hour = _dt.now().hour
            if 7 <= _now_hour <= 23:  # day window: 7 AM to midnight
                async def _startup_unload_night():
                    ok = await _unload(settings.llm.night_model, settings.llm.ollama_url)
                    logger.info(
                        "Startup: requested unload of night model %s (ok=%s, hour=%d)",
                        settings.llm.night_model, ok, _now_hour,
                    )
                asyncio.create_task(_startup_unload_night())
        except Exception as e:
            logger.debug("Startup model swap check failed: %s", e)

    # Initialize cloud LLM for business workflows (booking, email)
    if settings.llm.cloud_enabled:
        from .services.llm_router import init_cloud_llm
        init_cloud_llm(
            model=settings.llm.cloud_ollama_model,
            base_url=settings.llm.ollama_url,
        )

    # Initialize draft LLM for email draft generation (Anthropic)
    if settings.email_draft.enabled:
        from .services.llm_router import init_draft_llm
        init_draft_llm(
            model=settings.email_draft.model_name,
            api_key=settings.llm.anthropic_api_key,
        )

    # Initialize triage LLM for email replyable classification (Anthropic Haiku).
    # Checked independently of email_draft.enabled so gmail_digest synthesis
    # (synthesis_llm: "email_triage") works even when auto-draft is disabled.
    if settings.email_draft.triage_enabled:
        from .services.llm_router import init_triage_llm
        init_triage_llm(
            model=settings.email_draft.triage_model,
            api_key=settings.llm.anthropic_api_key,
        )

    # Note: Speaker ID loaded lazily via get_speaker_id_service() when voice
    # pipeline starts. No registry needed - single Resemblyzer implementation.

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

    # Initialize autonomous scheduler
    autonomous_scheduler = None
    if settings.autonomous.enabled:
        try:
            from .autonomous import init_autonomous
            autonomous_scheduler = await init_autonomous()
            logger.info(
                "Autonomous scheduler initialized with %d tasks",
                autonomous_scheduler.scheduled_count,
            )
        except Exception as e:
            logger.error("Failed to initialize autonomous scheduler: %s", e)


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

    # Verify calendar credentials on startup
    if settings.tools.calendar_enabled:
        try:
            from .tools.calendar import calendar_tool
            if await calendar_tool.verify_credentials():
                await calendar_tool.prefetch()
            else:
                logger.warning("Calendar running in degraded mode -- credentials invalid")
        except Exception as e:
            logger.error("Calendar startup check failed: %s", e)

    # Start ASR server if voice is enabled and ASR isn't already running
    asr_process = None
    if settings.voice.enabled and settings.voice.asr_url:
        asr_process = await _start_asr_server()

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

    # Initialize Fine-Tune Labs tracing client
    try:
        from .services.tracing import tracer
        tracer.configure(
            base_url=settings.ftl_tracing.base_url,
            api_key=settings.ftl_tracing.api_key,
            user_id=settings.ftl_tracing.user_id,
            enabled=settings.ftl_tracing.enabled,
        )
    except Exception as e:
        logger.error("Failed to initialize FTL tracing: %s", e)

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

    # Stop ASR server subprocess
    if asr_process and asr_process.poll() is None:
        logger.info("Stopping ASR server (pid=%d)", asr_process.pid)
        asr_process.terminate()
        try:
            asr_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            asr_process.kill()
            asr_process.wait(timeout=2)
        logger.info("ASR server stopped")

    # Shutdown discovery service
    if settings.discovery.enabled:
        try:
            from .discovery import shutdown_discovery
            await shutdown_discovery()
            logger.info("Discovery service shutdown complete")
        except Exception as e:
            logger.error("Error shutting down discovery service: %s", e)

    # Shutdown autonomous scheduler
    if autonomous_scheduler:
        try:
            from .autonomous import shutdown_autonomous
            await shutdown_autonomous()
            logger.info("Autonomous scheduler shutdown complete")
        except Exception as e:
            logger.error("Error shutting down autonomous scheduler: %s", e)

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

    # Unload triage LLM singleton (Anthropic Haiku)
    from .services.llm_router import shutdown_triage_llm
    shutdown_triage_llm()

    # Unload draft LLM singleton (Anthropic)
    from .services.llm_router import shutdown_draft_llm
    shutdown_draft_llm()

    # Unload cloud LLM singleton
    from .services.llm_router import shutdown_cloud_llm
    shutdown_cloud_llm()

    # Unload models to free resources
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

# OpenAI-compatible endpoint at app root (HA expects /v1/chat/completions)
from .api.openai_compat import router as openai_compat_router
app.include_router(openai_compat_router)

# Ollama-compatible endpoints at app root (HA expects /api/tags, /api/chat)
from .api.ollama_compat import router as ollama_compat_router
app.include_router(ollama_compat_router)

# Serve the web UI production build (atlas-ui/dist) as static files.
# Root "/" uses content negotiation: browsers get the UI, API clients get
# the Ollama health-check response. Static assets mounted at /.
_ui_dist = Path(__file__).parent.parent / "atlas-ui" / "dist"
if _ui_dist.is_dir():
    from starlette.staticfiles import StaticFiles

    # Override the Ollama compat GET / with a content-negotiating handler.
    # Browsers (Accept: text/html) get the UI; everything else gets "Ollama is running".
    _ui_index = _ui_dist / "index.html"

    # Remove the Ollama root route and replace with content-negotiating version
    app.routes[:] = [r for r in app.routes if not (hasattr(r, 'path') and r.path == '/' and hasattr(r, 'methods') and 'GET' in (r.methods or set()))]

    from starlette.responses import FileResponse, PlainTextResponse
    from fastapi import Request

    @app.get("/", include_in_schema=False)
    async def root_with_ui(request: Request):
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            return FileResponse(str(_ui_index), media_type="text/html")
        return PlainTextResponse("Ollama is running")

    app.mount("/", StaticFiles(directory=str(_ui_dist)), name="ui")


