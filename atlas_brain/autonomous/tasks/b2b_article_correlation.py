"""Follow-up task: correlate news articles with vendor churn wedges.

Reads vendor reasoning from synthesis and matches against classified
news_articles. Persists correlations to b2b_article_correlations.
"""

import logging
from datetime import date
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.b2b_article_correlation")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Correlate articles with vendor archetypes from persisted synthesis."""
    cfg = settings.b2b_churn
    if not cfg.enabled:
        return {"_skip_synthesis": "B2B churn disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    # Freshness gate: only run if the core task completed canonically today.
    today = date.today()
    from ._b2b_shared import has_complete_core_run_marker

    if not await has_complete_core_run_marker(pool, today):
        return {"_skip_synthesis": "Core run not complete for today"}

    from .b2b_churn_intelligence import (
        _correlate_articles_with_archetypes,
    )
    from ._b2b_synthesis_reader import (
        build_reasoning_lookup_from_views,
        discover_reasoning_vendor_names,
    )

    # Synthesis-first: load all synthesis views only.
    try:
        from ._b2b_synthesis_reader import load_best_reasoning_views
        all_vendors = await discover_reasoning_vendor_names(
            pool,
            as_of=today,
        )
        if all_vendors:
            views = await load_best_reasoning_views(
                pool,
                all_vendors,
                as_of=today,
            )
            synth_lookup = build_reasoning_lookup_from_views(views)
        else:
            synth_lookup = {}
    except Exception:
        logger.debug("Synthesis view loading failed", exc_info=True)
        synth_lookup = {}

    reasoning_lookup = synth_lookup
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
