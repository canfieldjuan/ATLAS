"""
Consumer analytics refresh task.

Refreshes the mv_brand_summary, mv_category_summary, and mv_asin_summary
materialized views every 6 hours so dashboard endpoints return up-to-date
aggregated metrics without live JSONB scans.
"""

import logging

from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.consumer_analytics_refresh")

_VIEWS = [
    "mv_brand_summary",
    "mv_category_summary",
    "mv_asin_summary",
]


async def run(task: ScheduledTask) -> dict:
    """Refresh consumer analytics materialized views."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "Database unavailable"}

    refreshed = []
    errors = []

    for view in _VIEWS:
        try:
            await pool.execute(
                f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}"
            )
            refreshed.append(view)
        except Exception as exc:
            logger.error("Failed to refresh %s: %s", view, exc)
            errors.append(f"{view}: {exc}")

    if refreshed:
        logger.info("Refreshed consumer analytics views: %s", ", ".join(refreshed))

    return {
        "_skip_synthesis": True,
        "refreshed": refreshed,
        "errors": errors or None,
    }
