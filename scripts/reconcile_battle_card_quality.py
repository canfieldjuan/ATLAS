#!/usr/bin/env python3
"""Reconcile persisted battle-card rows against the current quality pipeline."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from typing import Any
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atlas_brain.autonomous.tasks._b2b_shared import _sanitize_battle_card_sales_copy
from atlas_brain.autonomous.tasks.b2b_battle_cards import (
    _BATTLE_CARD_LLM_FIELDS,
    _apply_battle_card_quality,
    _battle_card_persist_summary,
    _battle_card_row_status,
)
from atlas_brain.storage import get_db_pool


def _load_card(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(raw, dict):
        return copy.deepcopy(raw)
    return dict(raw or {})


def _stale_only_failed_checks(card: dict[str, Any]) -> bool:
    quality = card.get("battle_card_quality")
    if not isinstance(quality, dict):
        return False
    failed_checks = quality.get("failed_checks")
    if not isinstance(failed_checks, list) or not failed_checks:
        return False
    normalized = [str(item or "").strip().lower() for item in failed_checks if str(item or "").strip()]
    return bool(normalized) and all(
        item.startswith("source data is stale for requested report date")
        for item in normalized
    )


def _normalized_signature(card: dict[str, Any]) -> dict[str, Any]:
    signature = copy.deepcopy(card)
    quality = signature.get("battle_card_quality")
    if isinstance(quality, dict):
        quality.pop("evaluated_at", None)
    return signature


def _reconcile_card(card: dict[str, Any]) -> tuple[dict[str, Any], str]:
    generated = {
        field: card[field]
        for field in _BATTLE_CARD_LLM_FIELDS
        if field in card
    }
    sanitized = _sanitize_battle_card_sales_copy(card, generated)
    replay = copy.deepcopy(card)
    for field in _BATTLE_CARD_LLM_FIELDS:
        if field in sanitized:
            replay[field] = sanitized[field]
    _apply_battle_card_quality(replay, phase="final")
    if _stale_only_failed_checks(replay):
        replay["llm_render_status"] = "skipped_preflight_quality_gate"
        failed = ((replay.get("battle_card_quality") or {}).get("failed_checks") or [])
        replay["llm_render_error"] = str(failed[0]) if failed else "source data is stale"
    return replay, _battle_card_row_status(replay)


async def _run(days: int, limit: int | None, write: bool) -> None:
    pool = get_db_pool()
    await pool.initialize()
    query = """
        SELECT id, report_date, vendor_filter, intelligence_data, status, llm_model, executive_summary
        FROM b2b_intelligence
        WHERE report_type = 'battle_card'
          AND created_at >= NOW() - ($1::int * INTERVAL '1 day')
          AND (
                status = 'deterministic_fallback'
             OR COALESCE(intelligence_data->>'llm_render_status', '') IN ('failed_quality_gate', 'failed')
          )
        ORDER BY created_at DESC
    """
    rows = await pool.fetch(query, days)
    if limit is not None:
        rows = rows[:limit]
    updated = 0
    inspected = 0
    for row in rows:
        inspected += 1
        original = _load_card(row["intelligence_data"])
        reconciled, status = _reconcile_card(original)
        executive_summary = _battle_card_persist_summary(reconciled)
        llm_model = "pipeline_deterministic" if _stale_only_failed_checks(reconciled) else str(row["llm_model"] or "pipeline_deterministic")
        changed = (
            _normalized_signature(reconciled) != _normalized_signature(original)
            or status != str(row["status"] or "")
            or executive_summary != str(row["executive_summary"] or "")
            or llm_model != str(row["llm_model"] or "")
        )
        result = {
            "vendor": row["vendor_filter"],
            "status_before": row["status"],
            "status_after": status,
            "llm_before": original.get("llm_render_status"),
            "llm_after": reconciled.get("llm_render_status"),
            "failed_after": ((reconciled.get("battle_card_quality") or {}).get("failed_checks") or [])[:3],
            "changed": changed,
        }
        print(json.dumps(result, default=str))
        if not write or not changed:
            continue
        await pool.execute(
            """
            UPDATE b2b_intelligence
            SET intelligence_data = $2,
                executive_summary = $3,
                status = $4,
                llm_model = $5
            WHERE id = $1
            """,
            row["id"],
            json.dumps(reconciled, default=str),
            executive_summary,
            status,
            llm_model,
        )
        updated += 1
    await pool.close()
    print(json.dumps({"inspected": inspected, "updated": updated, "write": write}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14, help="Look back this many days")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit")
    parser.add_argument("--write", action="store_true", help="Persist reconciled rows")
    args = parser.parse_args()
    asyncio.run(_run(args.days, args.limit, args.write))


if __name__ == "__main__":
    main()
