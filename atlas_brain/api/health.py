"""
Health check endpoints.
"""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/ping")
async def ping():
    """Simple endpoint to verify server is running."""
    return {"status": "ok", "message": "pong"}


@router.get("/health")
async def health():
    """Detailed health check with service status."""
    # Get webcam detector status
    webcam_status = {"running": False, "person_detected": False}
    try:
        from ..vision.webcam_detector import get_webcam_detector
        detector = get_webcam_detector()
        if detector:
            webcam_status = {
                "running": detector._running,
                "person_detected": detector._person_detected,
                "source_id": detector.camera_source_id,
            }
    except Exception:
        pass

    # Get presence status
    presence_status = {"current_room": None}
    try:
        from ..presence import get_presence_service
        presence = get_presence_service()
        user = presence.get_user_presence()
        presence_status = {
            "current_room": user.current_room if user else None,
            "room_name": user.current_room_name if user else None,
            "confidence": user.confidence if user else 0,
            "pending_room": user.pending_room if user else None,
            "last_seen": str(user.last_seen) if user and user.last_seen else None,
        }
        room_states = {}
        for room_id, state in presence.get_all_room_states().items():
            if state.last_seen:
                room_states[room_id] = {
                    "occupied": state.occupied,
                    "confidence": state.confidence,
                }
        if room_states:
            presence_status["room_states"] = room_states
    except Exception as e:
        presence_status["error"] = str(e)

    # Get Google OAuth token status
    google_oauth_status = {}
    try:
        from ..services.google_oauth import get_google_token_store
        store = get_google_token_store()
        google_oauth_status = store.get_status()
    except Exception as e:
        google_oauth_status = {"error": str(e)}

    return {
        "status": "ok",
        "services": {
            "webcam": webcam_status,
            "presence": presence_status,
            "google_oauth": google_oauth_status,
        },
    }
