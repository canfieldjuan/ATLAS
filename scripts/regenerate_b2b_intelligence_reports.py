#!/usr/bin/env python3
"""Regenerate current global B2B intelligence reports.

Deletes the latest global report rows for the main report types, then reruns the
hardened `b2b_churn_intelligence` task so stored reports match current logic.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.autonomous.tasks.b2b_churn_intelligence import run
from atlas_brain.storage.database import close_database, get_db_pool, init_database
from atlas_brain.storage.models import ScheduledTask

GLOBAL_REPORT_TYPES = (
    "weekly_churn_feed",
    "vendor_scorecard",
    "displacement_report",
    "category_overview",
    "exploratory_overview",
)


async def _regenerate(prune_existing: bool) -> None:
    await init_database()
    pool = get_db_pool()

    if prune_existing:
        await pool.execute(
            """
            DELETE FROM b2b_intelligence
            WHERE vendor_filter IS NULL
              AND report_type = ANY($1::text[])
              AND report_date = CURRENT_DATE
            """,
            list(GLOBAL_REPORT_TYPES),
        )

    task = ScheduledTask(
        id=uuid4(),
        name="manual_b2b_churn_intelligence_regeneration",
        task_type="builtin",
        schedule_type="once",
        description="Manual regeneration of global B2B intelligence reports",
        timeout_seconds=600,
        metadata={
            "builtin_handler": "b2b_churn_intelligence",
            "trigger": "manual_regeneration_script",
        },
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    result = await run(task)
    print(result)
    await close_database()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate current B2B intelligence reports")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete today's existing global report rows before regeneration",
    )
    args = parser.parse_args()
    asyncio.run(_regenerate(prune_existing=not args.keep_existing))


if __name__ == "__main__":
    main()
