"""
Query API routers for AI inference endpoints.
"""

from fastapi import APIRouter

from .audio import router as audio_router
from .text import router as text_router
from .vision import router as vision_router
from .vos import router as vos_router

router = APIRouter(prefix="/query", tags=["Query"])

router.include_router(text_router)
router.include_router(audio_router)
router.include_router(vision_router)
router.include_router(vos_router, prefix="/vos", tags=["VOS"])
