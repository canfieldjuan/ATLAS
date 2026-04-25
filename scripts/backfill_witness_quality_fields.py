#!/usr/bin/env python3
"""Backfill witness-quality fields into persisted reasoning/report JSON.

Read-only by default. The script decorates quote-bearing witness objects by
joining their witness_id/_sid/source_id to the latest b2b_vendor_witnesses row.
It does not rerun LLMs or regenerate reports.

Usage:
  python scripts/backfill_witness_quality_fields.py --days 30
  python scripts/backfill_witness_quality_fields.py --days 30 --apply
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

from atlas_brain.services.witness_quality_propagation import (
    collect_quote_witness_ids,
    compact_json,
    decorate_witness_quality_fields,
    decode_json_payload,
    normalize_witness_quality_row,
)
from atlas_brain.storage.database import close_database, get_db_pool, init_database

_REPORT_TYPES = (
    "battle_card",
    "challenger_brief",
    "accounts_in_motion",
    "weekly_churn_feed",
    "vendor_scorecard",
    "displacement_report",
    "category_overview",
)


def _limit_clause(limit: int | None, placeholder: str) -> str:
    return f"LIMIT {placeholder}" if limit is not None else ""


async def _quality_lookup(pool, witness_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not witness_ids:
        return {}
    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (witness_id)
            witness_id,
            grounding_status,
            phrase_polarity,
            phrase_subject,
            phrase_role,
            phrase_verbatim,
            pain_confidence
        FROM b2b_vendor_witnesses
        WHERE witness_id = ANY($1::text[])
        ORDER BY witness_id, as_of_date DESC, created_at DESC
        """,
        sorted(witness_ids),
    )
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        witness_id, quality = normalize_witness_quality_row(dict(row))
        if witness_id and quality:
            lookup[witness_id] = quality
    return lookup


def _merge_counts(total: dict[str, int], delta: dict[str, int]) -> None:
    for key, value in delta.items():
        total[key] = int(total.get(key, 0) or 0) + int(value or 0)


async def _backfill_table(
    pool,
    *,
    table: str,
    payload_column: str,
    rows: list[Any],
    apply: bool,
    overwrite: bool,
) -> dict[str, Any]:
    all_ids: set[str] = set()
    decoded_rows: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        payload = decode_json_payload(row[payload_column])
        row["_decoded_payload"] = payload
        row["_witness_ids"] = collect_quote_witness_ids(payload)
        all_ids.update(row["_witness_ids"])
        decoded_rows.append(row)

    lookup = await _quality_lookup(pool, all_ids)
    summary: dict[str, Any] = {
        "scanned_rows": len(decoded_rows),
        "quote_witness_ids": len(all_ids),
        "quality_matches": len(lookup),
        "changed_rows": 0,
        "witness_objects_seen": 0,
        "witness_objects_matched": 0,
        "witness_objects_updated": 0,
        "fields_written": 0,
    }

    for row in decoded_rows:
        original = row["_decoded_payload"]
        decorated, stats = decorate_witness_quality_fields(
            original,
            lookup,
            overwrite=overwrite,
        )
        _merge_counts(summary, stats)
        if compact_json(decorated) == compact_json(original):
            continue
        summary["changed_rows"] += 1
        if not apply:
            continue
        if table == "b2b_reasoning_synthesis":
            await pool.execute(
                """
                UPDATE b2b_reasoning_synthesis
                   SET synthesis = $1::jsonb
                 WHERE vendor_name = $2
                   AND as_of_date = $3
                   AND analysis_window_days = $4
                   AND schema_version = $5
                """,
                compact_json(decorated),
                row["vendor_name"],
                row["as_of_date"],
                row["analysis_window_days"],
                row["schema_version"],
            )
        elif table == "b2b_cross_vendor_reasoning_synthesis":
            await pool.execute(
                """
                UPDATE b2b_cross_vendor_reasoning_synthesis
                   SET synthesis = $1::jsonb
                 WHERE id = $2
                """,
                compact_json(decorated),
                row["id"],
            )
        elif table == "b2b_intelligence":
            await pool.execute(
                """
                UPDATE b2b_intelligence
                   SET intelligence_data = $1::jsonb
                 WHERE id = $2
                """,
                compact_json(decorated),
                row["id"],
            )
        else:
            raise ValueError(f"unsupported table: {table}")
    return summary


async def _main(args: argparse.Namespace) -> None:
    await init_database()
    pool = get_db_pool()
    try:
        limit = args.limit
        vendor_args = (args.days, limit) if limit is not None else (args.days,)
        intelligence_args = (
            (args.days, list(_REPORT_TYPES), limit)
            if limit is not None
            else (args.days, list(_REPORT_TYPES))
        )
        vendor_rows = await pool.fetch(
            f"""
            WITH latest_vendor AS (
                SELECT DISTINCT ON (vendor_name)
                    vendor_name,
                    as_of_date,
                    analysis_window_days,
                    schema_version,
                    synthesis,
                    created_at
                FROM b2b_reasoning_synthesis
                WHERE created_at >= NOW() - make_interval(days => $1)
                ORDER BY vendor_name, created_at DESC
            )
            SELECT vendor_name, as_of_date, analysis_window_days, schema_version, synthesis
            FROM latest_vendor
            ORDER BY created_at DESC
            {_limit_clause(limit, '$2')}
            """,
            *vendor_args,
        )
        cross_vendor_rows = await pool.fetch(
            f"""
            SELECT id, synthesis
            FROM b2b_cross_vendor_reasoning_synthesis
            WHERE created_at >= NOW() - make_interval(days => $1)
            ORDER BY created_at DESC
            {_limit_clause(limit, '$2')}
            """,
            *vendor_args,
        )
        intelligence_rows = await pool.fetch(
            f"""
            SELECT id, intelligence_data
            FROM b2b_intelligence
            WHERE created_at >= NOW() - make_interval(days => $1)
              AND report_type = ANY($2::text[])
            ORDER BY created_at DESC
            {_limit_clause(limit, '$3')}
            """,
            *intelligence_args,
        )
        result = {
            "apply": bool(args.apply),
            "days": args.days,
            "limit": args.limit,
            "overwrite": bool(args.overwrite),
            "tables": {
                "b2b_reasoning_synthesis": await _backfill_table(
                    pool,
                    table="b2b_reasoning_synthesis",
                    payload_column="synthesis",
                    rows=vendor_rows,
                    apply=args.apply,
                    overwrite=args.overwrite,
                ),
                "b2b_cross_vendor_reasoning_synthesis": await _backfill_table(
                    pool,
                    table="b2b_cross_vendor_reasoning_synthesis",
                    payload_column="synthesis",
                    rows=cross_vendor_rows,
                    apply=args.apply,
                    overwrite=args.overwrite,
                ),
                "b2b_intelligence": await _backfill_table(
                    pool,
                    table="b2b_intelligence",
                    payload_column="intelligence_data",
                    rows=intelligence_rows,
                    apply=args.apply,
                    overwrite=args.overwrite,
                ),
            },
        }
    finally:
        await close_database()

    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write updates.")
    parser.add_argument("--days", type=int, default=30, help="Lookback window.")
    parser.add_argument("--limit", type=int, default=None, help="Max rows per table.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing quality fields.")
    asyncio.run(_main(parser.parse_args()))
