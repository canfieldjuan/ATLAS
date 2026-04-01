#!/usr/bin/env python3
"""Backfill detached campaign batch replay metadata to the versioned contract.

Usage:
  python scripts/backfill_campaign_batch_replay_contracts.py
  python scripts/backfill_campaign_batch_replay_contracts.py --apply
  python scripts/backfill_campaign_batch_replay_contracts.py --apply --limit 100
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

from atlas_brain.autonomous.tasks.b2b_campaign_generation import (
    _CAMPAIGN_BATCH_REPLAY_CONTRACT_VERSION,
    _build_campaign_batch_replay_entry,
    _normalize_campaign_batch_replay_entry,
)
from atlas_brain.storage.database import close_database, get_db_pool, init_database


def _normalized_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def _load_candidate_rows(pool, *, limit: int | None) -> list[dict[str, Any]]:
    query = """
    SELECT id, batch_id, custom_id, status, request_metadata
    FROM anthropic_message_batch_items
    WHERE COALESCE(request_metadata->>'replay_handler', '') = 'campaign_generation'
    ORDER BY created_at ASC
    """
    if limit and limit > 0:
        query += f"\nLIMIT {int(limit)}"
    rows = await pool.fetch(query)
    return [dict(row) for row in rows]


async def _apply_repair(pool, *, item_id: str, metadata: dict[str, Any]) -> None:
    await pool.execute(
        """
        UPDATE anthropic_message_batch_items
        SET request_metadata = $2::jsonb
        WHERE id = $1::uuid
        """,
        item_id,
        json.dumps(metadata, default=str),
    )


async def run_backfill(*, apply: bool, limit: int | None) -> dict[str, Any]:
    await init_database()
    try:
        pool = get_db_pool()
        rows = await _load_candidate_rows(pool, limit=limit)
        summary: dict[str, Any] = {
            "rows_scanned": 0,
            "already_versioned": 0,
            "repairable_legacy": 0,
            "repaired": 0,
            "invalid": 0,
            "invalid_items": [],
        }

        for row in rows:
            summary["rows_scanned"] += 1
            metadata = _normalized_metadata(row.get("request_metadata"))
            replay_entry = metadata.get("replay_entry")
            if (
                isinstance(replay_entry, dict)
                and int(replay_entry.get("contract_version") or 0) == _CAMPAIGN_BATCH_REPLAY_CONTRACT_VERSION
            ):
                summary["already_versioned"] += 1
                continue

            normalized = _normalize_campaign_batch_replay_entry(metadata)
            if normalized is None:
                summary["invalid"] += 1
                summary["invalid_items"].append(
                    {
                        "id": str(row["id"]),
                        "batch_id": str(row["batch_id"]),
                        "custom_id": str(row["custom_id"]),
                        "status": str(row["status"]),
                    }
                )
                continue

            persisted = _build_campaign_batch_replay_entry(normalized)
            if persisted is None:
                summary["invalid"] += 1
                summary["invalid_items"].append(
                    {
                        "id": str(row["id"]),
                        "batch_id": str(row["batch_id"]),
                        "custom_id": str(row["custom_id"]),
                        "status": str(row["status"]),
                    }
                )
                continue

            summary["repairable_legacy"] += 1
            repaired_metadata = dict(metadata)
            repaired_metadata["replay_entry"] = {
                "contract_version": _CAMPAIGN_BATCH_REPLAY_CONTRACT_VERSION,
                "entry": persisted,
            }
            if apply:
                await _apply_repair(pool, item_id=str(row["id"]), metadata=repaired_metadata)
                summary["repaired"] += 1

        return summary
    finally:
        await close_database()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Persist repairs instead of dry-run only")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = asyncio.run(run_backfill(apply=bool(args.apply), limit=args.limit))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
