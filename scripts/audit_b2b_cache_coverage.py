#!/usr/bin/env python3
"""Print declared B2B cache strategies and current exact-cache row counts."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.services.b2b.cache_strategy import iter_core_b2b_cache_strategies
from atlas_brain.storage.database import close_database, get_db_pool, init_database


async def _main() -> None:
    await init_database()
    pool = get_db_pool()

    exact_namespaces = [
        strategy.namespace
        for strategy in iter_core_b2b_cache_strategies()
        if strategy.mode == "exact" and strategy.namespace
    ]
    counts: dict[str, int] = {}
    if exact_namespaces:
        rows = await pool.fetch(
            """
            SELECT namespace, count(*) AS cnt
            FROM b2b_llm_exact_cache
            WHERE namespace = ANY($1::text[])
            GROUP BY namespace
            ORDER BY namespace
            """,
            exact_namespaces,
        )
        counts = {str(row["namespace"]): int(row["cnt"]) for row in rows}

    report = []
    for strategy in iter_core_b2b_cache_strategies():
        entry = {
            "stage_id": strategy.stage_id,
            "mode": strategy.mode,
            "file_path": strategy.file_path,
            "rationale": strategy.rationale,
        }
        if strategy.namespace:
            entry["namespace"] = strategy.namespace
            entry["exact_cache_rows"] = counts.get(strategy.namespace, 0)
        report.append(entry)

    print(json.dumps(report, indent=2))
    await close_database()


if __name__ == "__main__":
    asyncio.run(_main())
