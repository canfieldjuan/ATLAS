"""
Campaign analytics refresh task.

Refreshes the campaign_funnel_stats materialized view every 6 hours
so analytics endpoints return up-to-date funnel metrics.
"""

import logging

from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.campaign_analytics_refresh")


async def run(task: ScheduledTask) -> dict:
    """Refresh the campaign_funnel_stats materialized view."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "Database unavailable"}

    try:
        await pool.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY campaign_funnel_stats")
        logger.info("Refreshed campaign_funnel_stats materialized view")
        return {"_skip_synthesis": True, "refreshed": True}
    except Exception as exc:
        logger.error("Failed to refresh campaign_funnel_stats: %s", exc)
        return {"_skip_synthesis": True, "refreshed": False, "error": str(exc)}
