"""
Health check endpoints.
"""

from fastapi import APIRouter

from ..services import stt_registry, vlm_registry

router = APIRouter(tags=["Health"])


@router.get("/ping")
async def ping():
    """Simple endpoint to verify server is running."""
    return {"status": "ok", "message": "pong"}


@router.get("/health")
async def health():
    """Detailed health check with service status."""
    vlm_info = vlm_registry.get_active_info()
    stt_info = stt_registry.get_active_info()

    return {
        "status": "ok",
        "services": {
            "vlm": {
                "loaded": vlm_info is not None,
                "model": vlm_info.name if vlm_info else None,
                "device": vlm_info.device if vlm_info else None,
            },
            "stt": {
                "loaded": stt_info is not None,
                "model": stt_info.name if stt_info else None,
                "device": stt_info.device if stt_info else None,
            },
        },
    }
