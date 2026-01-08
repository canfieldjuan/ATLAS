"""
Device management and control API routers.
"""

from fastapi import APIRouter

from .control import router as control_router

router = APIRouter(prefix="/devices", tags=["Devices"])

router.include_router(control_router)
