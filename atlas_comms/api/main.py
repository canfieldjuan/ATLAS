"""
FastAPI application for Atlas Communications.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..core.config import comms_settings

logger = logging.getLogger("atlas.comms.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(
        "Atlas Comms starting on %s:%d",
        comms_settings.server.host,
        comms_settings.server.port,
    )

    # Initialize communications service
    comms_service = None
    if comms_settings.enabled:
        from ..service import init_comms_service
        comms_service = await init_comms_service()
        if comms_service:
            logger.info("Communications service started")
        else:
            logger.warning("Failed to start communications service")

    yield

    # Shutdown
    logger.info("Atlas Comms shutting down")

    if comms_service:
        from ..service import shutdown_comms_service
        await shutdown_comms_service()
        logger.info("Communications service stopped")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    application = FastAPI(
        title="Atlas Communications",
        description="Telephony and messaging service for Atlas",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Include routers
    from .health import router as health_router
    from .calls import router as calls_router
    from .sms import router as sms_router
    from .contexts import router as contexts_router
    from .scheduling import router as scheduling_router

    application.include_router(health_router, tags=["health"])
    application.include_router(calls_router, prefix="/calls", tags=["calls"])
    application.include_router(sms_router, prefix="/sms", tags=["sms"])
    application.include_router(contexts_router, prefix="/contexts", tags=["contexts"])
    application.include_router(scheduling_router, prefix="/scheduling", tags=["scheduling"])

    return application


# Create app instance
app = create_app()
