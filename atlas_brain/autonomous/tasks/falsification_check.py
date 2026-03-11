"""Autonomous task handler for nightly falsification checks.

Runs the FalsificationWatcher against all active semantic cache entries,
invalidating any whose falsification conditions have been triggered by
fresh vendor signals (snapshots, reviews, change events).
"""

from __future__ import annotations

import logging
from typing import Any

from atlas_brain.storage.database import get_db_pool
from atlas_brain.storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.falsification_check")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Nightly falsification check handler."""
    from atlas_brain.reasoning.falsification import FalsificationWatcher
    from atlas_brain.reasoning.semantic_cache import SemanticCache

    pool = get_db_pool()
    if not pool or not pool.is_initialized:
        logger.warning("DB pool not available, skipping falsification check")
        return {"_skip_synthesis": True, "skipped": True, "reason": "db_pool_unavailable"}

    cache = SemanticCache(pool)
    watcher = FalsificationWatcher(pool, cache)

    results = await watcher.run_nightly_check()

    entries_checked = len(results)
    entries_invalidated = sum(1 for r in results if r.invalidated)
    triggered_conditions = []
    for r in results:
        for cond in r.triggered_conditions:
            triggered_conditions.append(f"{r.vendor_name or r.pattern_sig}: {cond}")

    logger.info(
        "Falsification check complete: %d checked, %d invalidated",
        entries_checked,
        entries_invalidated,
    )

    return {
        "_skip_synthesis": True,
        "entries_checked": entries_checked,
        "entries_invalidated": entries_invalidated,
        "triggered_conditions": triggered_conditions[:20],
    }
