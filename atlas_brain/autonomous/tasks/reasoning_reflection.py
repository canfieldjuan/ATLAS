"""Scheduled reflection task for proactive pattern detection.

Runs 4x daily (default: 9am, 1pm, 5pm, 9pm) to identify cross-domain
patterns that the reactive pipeline misses.
"""

import logging

from ...config import settings
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.reasoning_reflection")


async def run(task: ScheduledTask) -> dict:
    """Run the reflection cycle."""
    if not settings.reasoning.enabled:
        return {"_skip_synthesis": "Reasoning disabled"}

    from ...reasoning.reflection import run_reflection

    result = await run_reflection()

    if result.get("findings", 0) == 0:
        return {"_skip_synthesis": True, **result}

    logger.info(
        "Reflection complete: %d findings, %d actions, %d notifications",
        result.get("findings", 0),
        result.get("actions", 0),
        result.get("notifications", 0),
    )
    return {"_skip_synthesis": True, **result}
