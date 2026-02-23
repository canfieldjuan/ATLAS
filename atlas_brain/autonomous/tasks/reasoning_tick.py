"""Polling safety net for the reasoning agent.

Catches events that LISTEN/NOTIFY may have missed. Runs every 5 minutes
when reasoning is enabled.
"""

import logging

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.reasoning_tick")


async def run(task: ScheduledTask) -> dict:
    """Pick up unprocessed events and feed them to the reasoning agent."""
    if not settings.reasoning.enabled:
        return {"_skip_synthesis": "Reasoning disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "Database not available"}

    rows = await pool.fetch(
        """
        SELECT * FROM atlas_events
        WHERE processed_at IS NULL
          AND created_at > NOW() - INTERVAL '1 hour'
        ORDER BY created_at ASC
        LIMIT $1
        """,
        settings.reasoning.event_batch_size,
    )

    if not rows:
        return {"_skip_synthesis": True, "picked_up": 0}

    from ...reasoning.events import AtlasEvent
    from ...reasoning.consumer import EventConsumer

    # Process directly via the reasoning agent
    from ...reasoning.agent import get_reasoning_agent
    import json

    agent = get_reasoning_agent()
    processed = 0

    for row in rows:
        event = AtlasEvent.from_row(dict(row))
        try:
            result = await agent.process_event(event)
            await pool.execute(
                """
                UPDATE atlas_events
                SET processed_at = NOW(), processing_result = $1::jsonb
                WHERE id = $2
                """,
                json.dumps(result, default=str),
                event.id,
            )
            processed += 1
        except Exception:
            logger.warning("reasoning_tick: failed to process event %s", event.id, exc_info=True)

    logger.info("reasoning_tick: picked up %d missed events", processed)
    return {"_skip_synthesis": True, "picked_up": processed, "total_pending": len(rows)}
