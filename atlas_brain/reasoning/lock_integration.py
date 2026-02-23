"""Voice session lock hooks for entity sovereignty.

Acquire entity locks when AtlasAgent starts a voice session with
a known contact. Release + drain when the session ends.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("atlas.reasoning.lock_integration")

_HOLDER = "atlas_agent"


async def on_voice_session_start(
    session_id: str,
    speaker_id: Optional[str],
    contact_id: Optional[str],
) -> bool:
    """Acquire an entity lock when a voice session starts with a known contact.

    Returns True if a lock was acquired.
    """
    if not contact_id:
        return False

    try:
        from ..config import settings
        if not settings.reasoning.enabled:
            return False
    except Exception:
        return False

    try:
        from .entity_locks import EntityLockManager

        mgr = EntityLockManager()
        acquired = await mgr.acquire(
            entity_type="contact",
            entity_id=contact_id,
            holder=_HOLDER,
            session_id=session_id,
        )
        return acquired
    except Exception:
        logger.debug("Entity lock acquisition failed (non-fatal)", exc_info=True)
        return False


async def on_voice_session_end(session_id: str) -> None:
    """Release all locks for a session and drain queued events.

    Drained events are fed to the ReasoningAgent with session context.
    """
    try:
        from ..config import settings
        if not settings.reasoning.enabled:
            return
    except Exception:
        return

    try:
        from .entity_locks import EntityLockManager
        from .agent import get_reasoning_agent

        mgr = EntityLockManager()
        released = await mgr.release_by_session(session_id)

        if not released:
            return

        agent = get_reasoning_agent()
        for entity_type, entity_id in released:
            events = await mgr.drain_queue(entity_type, entity_id)
            if events:
                logger.info(
                    "Draining %d queued events for %s/%s after session %s",
                    len(events), entity_type, entity_id, session_id,
                )
                try:
                    await agent.process_drained_events(events)
                except Exception:
                    logger.error(
                        "Failed to process drained events for %s/%s",
                        entity_type, entity_id, exc_info=True,
                    )
    except Exception:
        logger.debug("Voice session end lock release failed (non-fatal)", exc_info=True)


async def heartbeat_voice_session(session_id: str) -> None:
    """Update heartbeat for all locks held by the session."""
    try:
        from ..config import settings
        if not settings.reasoning.enabled:
            return
    except Exception:
        return

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return

        await pool.execute(
            """
            UPDATE entity_locks
            SET heartbeat_at = NOW()
            WHERE session_id = $1
              AND released_at IS NULL
            """,
            session_id,
        )
    except Exception:
        pass  # heartbeat failure is non-fatal
