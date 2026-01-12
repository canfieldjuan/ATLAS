"""
Session API endpoints for conversation persistence.

Provides REST API for:
- Session creation and management
- Multi-terminal session continuity
- Conversation history retrieval
"""

import logging
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..storage import db_settings
from ..storage.database import get_db_pool
from ..storage.repositories.session import get_session_repo
from ..storage.repositories.conversation import get_conversation_repo

logger = logging.getLogger("atlas.api.session")

router = APIRouter(prefix="/session", tags=["session"])


class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None
    terminal_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class ContinueSessionRequest(BaseModel):
    user_id: str
    terminal_id: str


class SessionResponse(BaseModel):
    id: str
    user_id: Optional[str]
    terminal_id: Optional[str]
    is_active: bool
    started_at: str
    last_activity_at: str


def _check_db_enabled():
    """Check if database is enabled and initialized."""
    if not db_settings.enabled:
        raise HTTPException(
            status_code=503,
            detail="Database persistence is disabled"
        )
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Database not initialized"
        )


@router.post("/create")
async def create_session(request: CreateSessionRequest) -> SessionResponse:
    """Create a new conversation session."""
    _check_db_enabled()

    try:
        session_repo = get_session_repo()
        user_uuid = UUID(request.user_id) if request.user_id else None

        session = await session_repo.create_session(
            user_id=user_uuid,
            terminal_id=request.terminal_id,
            metadata=request.metadata,
        )

        return SessionResponse(
            id=str(session.id),
            user_id=str(session.user_id) if session.user_id else None,
            terminal_id=session.terminal_id,
            is_active=session.is_active,
            started_at=session.started_at.isoformat(),
            last_activity_at=session.last_activity_at.isoformat(),
        )
    except Exception as e:
        logger.error("Failed to create session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/continue")
async def continue_session(request: ContinueSessionRequest) -> SessionResponse:
    """
    Continue an existing session from a different terminal.

    If user has an active session, returns it with updated terminal.
    Otherwise creates a new session.
    """
    _check_db_enabled()

    try:
        session_repo = get_session_repo()
        user_uuid = UUID(request.user_id)

        session = await session_repo.get_or_create_session(
            user_id=user_uuid,
            terminal_id=request.terminal_id,
        )

        return SessionResponse(
            id=str(session.id),
            user_id=str(session.user_id) if session.user_id else None,
            terminal_id=session.terminal_id,
            is_active=session.is_active,
            started_at=session.started_at.isoformat(),
            last_activity_at=session.last_activity_at.isoformat(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to continue session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}")
async def get_session(session_id: str) -> SessionResponse:
    """Get session details by ID."""
    _check_db_enabled()

    try:
        session_repo = get_session_repo()
        session_uuid = UUID(session_id)

        session = await session_repo.get_session(session_uuid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionResponse(
            id=str(session.id),
            user_id=str(session.user_id) if session.user_id else None,
            terminal_id=session.terminal_id,
            is_active=session.is_active,
            started_at=session.started_at.isoformat(),
            last_activity_at=session.last_activity_at.isoformat(),
        )
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    except Exception as e:
        logger.error("Failed to get session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 20):
    """Get conversation history for a session."""
    _check_db_enabled()

    try:
        conv_repo = get_conversation_repo()
        session_uuid = UUID(session_id)

        history = await conv_repo.get_history(session_uuid, limit=limit)

        return {
            "session_id": session_id,
            "count": len(history),
            "turns": [
                {
                    "id": str(turn.id),
                    "role": turn.role,
                    "content": turn.content,
                    "speaker_id": turn.speaker_id,
                    "intent": turn.intent,
                    "created_at": turn.created_at.isoformat(),
                }
                for turn in history
            ],
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    except Exception as e:
        logger.error("Failed to get history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/close")
async def close_session(session_id: str):
    """Close a session (mark as inactive)."""
    _check_db_enabled()

    try:
        session_repo = get_session_repo()
        session_uuid = UUID(session_id)

        await session_repo.close_session(session_uuid)

        return {"success": True, "message": "Session closed"}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    except Exception as e:
        logger.error("Failed to close session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/db")
async def get_db_status():
    """Check database connection status."""
    if not db_settings.enabled:
        return {
            "enabled": False,
            "initialized": False,
            "message": "Database persistence disabled",
        }

    pool = get_db_pool()
    return {
        "enabled": True,
        "initialized": pool.is_initialized,
        "host": db_settings.host,
        "database": db_settings.database,
    }
