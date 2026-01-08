"""
API routers for Atlas Brain.
"""

from fastapi import APIRouter

from .audio_events import router as audio_events_router
from .devices import router as devices_router
from .health import router as health_router
from .models import router as models_router
from .orchestration import router as orchestration_router
from .query import router as query_router

# Main router that aggregates all sub-routers
router = APIRouter()

router.include_router(health_router)
router.include_router(query_router)
router.include_router(models_router)
router.include_router(devices_router)
router.include_router(orchestration_router)
router.include_router(audio_events_router)
