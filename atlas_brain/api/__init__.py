"""
API routers for Atlas Brain.
"""

from fastapi import APIRouter

from .alerts import router as alerts_router
from .comms import router as comms_router
from .devices import router as devices_router
from .health import router as health_router
from .llm import router as llm_router
from .models import router as models_router
from .query import router as query_router
from .session import router as session_router
from .vision import router as vision_router
from .video import router as video_router
from .recognition import router as recognition_router
from .speaker import router as speaker_router
from .edge import router as edge_router

# Main router that aggregates all sub-routers
router = APIRouter()

router.include_router(health_router)
router.include_router(query_router)
router.include_router(models_router)
router.include_router(devices_router)
router.include_router(alerts_router)
router.include_router(comms_router)
router.include_router(llm_router)
router.include_router(session_router)
router.include_router(vision_router)
router.include_router(video_router)
router.include_router(recognition_router)
router.include_router(speaker_router)
router.include_router(edge_router)
