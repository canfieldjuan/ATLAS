"""
Query API routers for AI inference endpoints.
"""

from fastapi import APIRouter

from .text import router as text_router
from .vision import router as vision_router
from .vos import router as vos_router

router = APIRouter(prefix="/query", tags=["Query"])

router.include_router(text_router)
router.include_router(vision_router)
router.include_router(vos_router, prefix="/vos", tags=["VOS"])
