"""Follow-up task: correlate news articles with vendor churn archetypes.

Lightweight task that reads vendor archetypes from b2b_churn_signals
and matches against classified news_articles. Persists correlations
to b2b_article_correlations.
"""

import logging
from datetime import date
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.b2b_article_correlation")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Correlate articles with vendor archetypes from persisted signals."""
    cfg = settings.b2b_churn
    if not cfg.enabled:
        return {"_skip_synthesis": "B2B churn disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    # Freshness gate: only run if core task completed today
    today = date.today()
    marker = await pool.fetchval(
        "SELECT 1 FROM b2b_intelligence "
        "WHERE report_type = 'core_run_complete' AND report_date = $1",
        today,
    )
    if not marker:
        return {"_skip_synthesis": "Core run not complete for today"}

    from .b2b_churn_intelligence import (
        _correlate_articles_with_archetypes,
        reconstruct_reasoning_lookup,
    )

    reasoning_lookup = await reconstruct_reasoning_lookup(pool, as_of=today)
    if not reasoning_lookup:
        return {"_skip_synthesis": "No reasoning data available"}

    vendor_names = list(reasoning_lookup.keys())
    correlated = await _correlate_articles_with_archetypes(
        pool, vendor_names, reasoning_lookup, today,
    )

    return {
        "_skip_synthesis": "Article correlation complete",
        "vendors_checked": len(vendor_names),
        "correlations_persisted": correlated,
    }
