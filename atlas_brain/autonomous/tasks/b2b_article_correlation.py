"""Follow-up task: correlate news articles with vendor churn wedges.

Reads vendor reasoning from synthesis (with legacy fallback) and matches
against classified news_articles. Persists correlations to
b2b_article_correlations.
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
    from ._b2b_synthesis_reader import build_reasoning_lookup_from_views

    # Synthesis-first: load all synthesis views, fill gaps with legacy
    try:
        from ._b2b_synthesis_reader import load_best_reasoning_views
        # Get all vendor names from churn_signals for scoping
        vendor_rows = await pool.fetch(
            "SELECT DISTINCT vendor_name FROM b2b_churn_signals "
            "WHERE archetype IS NOT NULL AND last_computed_at::date >= $1",
            today,
        )
        all_vendors = [r["vendor_name"] for r in vendor_rows if r["vendor_name"]]
        if all_vendors:
            views = await load_best_reasoning_views(pool, all_vendors, as_of=today)
            synth_lookup = build_reasoning_lookup_from_views(views)
        else:
            synth_lookup = {}
    except Exception:
        logger.debug("Synthesis view loading failed, falling back to legacy", exc_info=True)
        synth_lookup = {}

    legacy_lookup = await reconstruct_reasoning_lookup(pool, as_of=today)
    reasoning_lookup = {**legacy_lookup, **synth_lookup}
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
