"""Autonomous task handler for nightly B2B knowledge graph sync.

Reads b2b_* Postgres tables and upserts entities/relationships into
Neo4j as B2b-prefixed nodes (B2bVendor, B2bProduct, B2bPainPoint, etc.).
Idempotent via MERGE -- safe to run repeatedly.
"""

from __future__ import annotations

import logging
from typing import Any

from atlas_brain.storage.database import get_db_pool
from atlas_brain.storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.knowledge_graph_sync")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Nightly knowledge graph sync handler."""
    from neo4j import AsyncGraphDatabase
    from atlas_brain.reasoning.knowledge_graph import KnowledgeGraphSync

    pool = get_db_pool()
    if not pool or not pool.is_initialized:
        logger.warning("DB pool not available, skipping knowledge graph sync")
        return {"_skip_synthesis": True, "skipped": True, "reason": "db_pool_unavailable"}

    from atlas_brain.reasoning.config import ReasoningConfig
    _rcfg = ReasoningConfig()
    driver = AsyncGraphDatabase.driver(
        _rcfg.neo4j_bolt_url, auth=(_rcfg.neo4j_user, _rcfg.neo4j_password),
    )
    try:
        sync = KnowledgeGraphSync(pool, driver)
        await sync.ensure_indexes()
        counts = await sync.full_sync()

        total = sum(counts.values())
        logger.info(
            "Knowledge graph sync complete: %d total entities/edges synced",
            total,
        )

        return {
            "_skip_synthesis": True,
            **counts,
            "total_synced": total,
        }
    finally:
        await driver.close()
