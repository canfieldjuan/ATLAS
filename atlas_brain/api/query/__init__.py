"""
Query API routers for AI inference endpoints.
"""

from fastapi import APIRouter

from .text import router as text_router

router = APIRouter(prefix="/query", tags=["Query"])

router.include_router(text_router)
