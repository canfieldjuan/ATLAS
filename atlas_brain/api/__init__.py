"""
API routers for Atlas Brain.
"""

from fastapi import APIRouter

from .alerts import router as alerts_router
from .audio_events import router as audio_events_router
from .devices import router as devices_router
from .health import router as health_router
from .llm import router as llm_router
from .models import router as models_router
from .orchestration import router as orchestration_router
from .query import router as query_router
from .speaker_id import router as speaker_id_router
from .tts import router as tts_router
from .session import router as session_router
from .vision import router as vision_router

# Main router that aggregates all sub-routers
router = APIRouter()

router.include_router(health_router)
router.include_router(query_router)
router.include_router(models_router)
router.include_router(devices_router)
router.include_router(orchestration_router)
router.include_router(alerts_router)
router.include_router(audio_events_router)
router.include_router(speaker_id_router)
router.include_router(tts_router)
router.include_router(llm_router)
router.include_router(session_router)
router.include_router(vision_router)
