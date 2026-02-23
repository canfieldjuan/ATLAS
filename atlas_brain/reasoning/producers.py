"""Safe event emission helpers for existing pipelines.

All producers call emit_if_enabled() which is a no-op when reasoning
is disabled and never breaks the caller on failure.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

logger = logging.getLogger("atlas.reasoning.producers")


async def emit_if_enabled(
    event_type: str,
    source: str,
    payload: dict[str, Any],
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
) -> Optional[UUID]:
    """Emit an event if reasoning is enabled. Never raises."""
    try:
        from ..config import settings
        if not settings.reasoning.enabled:
            return None
        from .events import emit_event
        return await emit_event(
            event_type, source, payload, entity_type, entity_id
        )
    except Exception:
        logger.debug("Event emission failed (non-fatal)", exc_info=True)
        return None
