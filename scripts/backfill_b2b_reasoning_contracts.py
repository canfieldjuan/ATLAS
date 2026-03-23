#!/usr/bin/env python3
"""Backfill legacy B2B reasoning rows into contract-first storage.

Usage:
  python scripts/backfill_b2b_reasoning_contracts.py
  python scripts/backfill_b2b_reasoning_contracts.py --apply
  python scripts/backfill_b2b_reasoning_contracts.py --apply --limit 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.services.b2b_reasoning_backfill import normalize_reasoning_payload
from atlas_brain.storage.database import close_database, get_db_pool, init_database


_REPORT_TYPES = (
    "battle_card",
    "weekly_churn_feed",
    "vendor_scorecard",
    "challenger_brief",
    "accounts_in_motion",
)


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


async def _backfill_synthesis(pool, *, apply: bool, limit: int | None) -> dict[str, int]:
    rows = await pool.fetch(
        """
        SELECT ctid, vendor_name, synthesis
        FROM b2b_reasoning_synthesis
        ORDER BY created_at DESC
        LIMIT $1
        """,
        int(limit or 1000000),
    )
    scanned = 0
    changed = 0
    for row in rows:
        scanned += 1
        original = _loads(row["synthesis"])
        normalized = normalize_reasoning_payload(
            original,
            vendor_name=str(row["vendor_name"] or ""),
            synthesis_mode=True,
        )
        if _dumps(normalized) == _dumps(original):
            continue
        changed += 1
        if apply:
            await pool.execute(
                """
                UPDATE b2b_reasoning_synthesis
                SET synthesis = $1::jsonb
                WHERE ctid = $2
                """,
                _dumps(normalized),
                row["ctid"],
            )
    return {"scanned": scanned, "changed": changed}


async def _backfill_intelligence(pool, *, apply: bool, limit: int | None) -> dict[str, int]:
    rows = await pool.fetch(
        """
        SELECT id, report_type, intelligence_data
        FROM b2b_intelligence
        WHERE report_type = ANY($1::text[])
        ORDER BY created_at DESC
        LIMIT $2
        """,
        list(_REPORT_TYPES),
        int(limit or 1000000),
    )
    scanned = 0
    changed = 0
    for row in rows:
        scanned += 1
        original = _loads(row["intelligence_data"])
        normalized = normalize_reasoning_payload(original)
        if _dumps(normalized) == _dumps(original):
            continue
        changed += 1
        if apply:
            await pool.execute(
                """
                UPDATE b2b_intelligence
                SET intelligence_data = $1::jsonb
                WHERE id = $2
                """,
                _dumps(normalized),
                row["id"],
            )
    return {"scanned": scanned, "changed": changed}


async def _main(apply: bool, limit: int | None) -> None:
    await init_database()
    pool = get_db_pool()
    try:
        synthesis = await _backfill_synthesis(pool, apply=apply, limit=limit)
        intelligence = await _backfill_intelligence(pool, apply=apply, limit=limit)
    finally:
        await close_database()

    print(
        {
            "apply": apply,
            "limit": limit,
            "b2b_reasoning_synthesis": synthesis,
            "b2b_intelligence": intelligence,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write updates to the database.")
    parser.add_argument("--limit", type=int, default=None, help="Max rows per table to scan.")
    args = parser.parse_args()
    asyncio.run(_main(apply=args.apply, limit=args.limit))
