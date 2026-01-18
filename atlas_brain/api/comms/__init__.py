"""
Communications API endpoints.

Handles webhooks from telephony providers (SignalWire, Twilio)
and provides call/SMS management APIs.
"""

from fastapi import APIRouter

from .webhooks import router as webhooks_router
from .management import router as management_router

router = APIRouter(prefix="/comms", tags=["comms"])
router.include_router(webhooks_router)
router.include_router(management_router)

__all__ = ["router"]
