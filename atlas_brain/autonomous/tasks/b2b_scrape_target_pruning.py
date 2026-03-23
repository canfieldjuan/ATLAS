"""Disable low-yield scrape targets according to configured policy."""

import logging

from ...config import settings
from ...services.scraping.source_yield import prune_low_yield_targets
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.b2b_scrape_target_pruning")


async def run(task: ScheduledTask) -> dict:
    """Run low-yield pruning policy for noisy scrape sources."""
    cfg = settings.b2b_scrape
    if not cfg.source_low_yield_pruning_enabled:
        return {"_skip_synthesis": "Low-yield source pruning disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "Database not ready"}

    result = await prune_low_yield_targets(
        pool,
        source=cfg.source_low_yield_pruning_source,
        lookback_runs=cfg.source_low_yield_pruning_lookback_runs,
        min_runs=cfg.source_low_yield_pruning_min_runs,
        max_inserted_total=cfg.source_low_yield_pruning_max_inserted_total,
        max_disable_per_run=cfg.source_low_yield_pruning_max_disable_per_run,
        dry_run=cfg.source_low_yield_pruning_dry_run,
        enabled_only=True,
    )
    logger.info(
        "Source pruning complete source=%s disabled=%d requested=%d dry_run=%s",
        result["source"],
        result["disabled"],
        result["requested"],
        result["dry_run"],
    )
    result["_skip_synthesis"] = True
    return result
