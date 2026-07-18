"""Invoicing API endpoints."""

from fastapi import APIRouter

from .actions import retired_router, router as actions_router
from .receivables import router as receivables_router

router = APIRouter()
router.include_router(retired_router)
router.include_router(actions_router)
router.include_router(receivables_router)

__all__ = ["router"]
