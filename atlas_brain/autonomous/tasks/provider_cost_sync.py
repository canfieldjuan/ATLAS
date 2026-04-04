"""Autonomous task: sync provider billing totals for cost reconciliation."""

from __future__ import annotations

import logging
from typing import Any

from ...config import settings
from ...services.provider_cost_sync import sync_provider_costs
from ...storage.database import get_db_pool

logger = logging.getLogger("atlas.autonomous.tasks.provider_cost_sync")


async def run(task) -> dict[str, Any]:
    """Sync provider billing totals into local reconciliation tables."""
    if not bool(settings.provider_cost.enabled):
        return {"_skip_synthesis": "Provider cost sync disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "Database pool not initialized"}

    result = await sync_provider_costs(pool=pool)
    logger.info("provider_cost_sync result=%s", result)
    return result
