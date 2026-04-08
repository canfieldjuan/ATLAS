#!/usr/bin/env python3
"""Check whether the reasoning rollout is safe to enable."""

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

from atlas_brain.config import settings
from atlas_brain.storage.database import close_database, get_db_pool, init_database

_REQUIRED_MIGRATIONS = (
    "245_cross_vendor_reasoning_synthesis",
    "247_b2b_vendor_witness_packets",
    "261_b2b_competitive_sets",
    "262_b2b_competitive_set_runs",
    "263_b2b_competitive_set_run_constraints",
    "265_b2b_report_subscriptions",
    "266_b2b_report_subscription_delivery_log",
    "268_b2b_report_subscription_delivery_dry_run_status",
    "269_b2b_report_subscription_delivery_mode",
)

_REQUIRED_TABLES = (
    "b2b_cross_vendor_reasoning_synthesis",
    "b2b_vendor_reasoning_packets",
    "b2b_vendor_witnesses",
    "b2b_competitive_sets",
    "b2b_competitive_set_runs",
    "b2b_report_subscriptions",
    "b2b_report_subscription_delivery_log",
)

_REQUIRED_TASKS = (
    "b2b_reasoning_synthesis",
    "b2b_report_subscription_delivery",
)


def _reasoning_v2_schema_predicate(column_name: str = "schema_version") -> str:
    """SQL predicate matching canonical reasoning v2 schema version formats."""
    return (
        f"(LOWER(COALESCE({column_name}, '')) IN ('v2', '2') "
        f"OR LOWER(COALESCE({column_name}, '')) LIKE 'v2.%' "
        f"OR COALESCE({column_name}, '') LIKE '2.%')"
    )


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _status(name: str, ok: bool, *, required: bool = True, detail: Any = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if ok else ("fail" if required else "warn"),
        "required": required,
        "detail": detail,
    }


def _exit_code(checks: list[dict[str, Any]]) -> int:
    return 1 if any(item["status"] == "fail" for item in checks) else 0


async def _run(limit: int) -> dict[str, Any]:
    await init_database()
    pool = get_db_pool()
    checks: list[dict[str, Any]] = []

    applied = await pool.fetch(
        "SELECT name FROM schema_migrations WHERE name = ANY($1::text[]) ORDER BY name",
        list(_REQUIRED_MIGRATIONS),
    )
    applied_names = {str(row["name"]) for row in applied}
    missing = [name for name in _REQUIRED_MIGRATIONS if name not in applied_names]
    checks.append(
        _status(
            "required_migrations",
            not missing,
            detail={"applied": sorted(applied_names), "missing": missing},
        ),
    )

    table_rows = await pool.fetch(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = ANY($1::text[])
        ORDER BY table_name
        """,
        list(_REQUIRED_TABLES),
    )
    present_tables = {str(row["table_name"]) for row in table_rows}
    expected_tables = set(_REQUIRED_TABLES)
    checks.append(
        _status(
            "required_tables",
            expected_tables.issubset(present_tables),
            detail={"present": sorted(present_tables), "missing": sorted(expected_tables - present_tables)},
        ),
    )

    constraint_rows = await pool.fetch(
        """
        SELECT conname
        FROM pg_constraint
        WHERE conname = ANY($1::text[])
        ORDER BY conname
        """,
        [
            "fk_b2b_competitive_set_runs_account",
            "chk_b2b_competitive_set_runs_trigger",
            "chk_b2b_competitive_set_runs_status",
        ],
    )
    present_constraints = {str(row["conname"]) for row in constraint_rows}
    expected_constraints = {
        "fk_b2b_competitive_set_runs_account",
        "chk_b2b_competitive_set_runs_trigger",
        "chk_b2b_competitive_set_runs_status",
    }
    checks.append(
        _status(
            "competitive_set_run_constraints",
            expected_constraints.issubset(present_constraints),
            detail={
                "present": sorted(present_constraints),
                "missing": sorted(expected_constraints - present_constraints),
            },
        ),
    )

    task_rows = await pool.fetch(
        """
        SELECT name, metadata
        FROM scheduled_tasks
        WHERE name = ANY($1::text[])
        ORDER BY name
        """,
        list(_REQUIRED_TASKS),
    )
    task_map = {str(row["name"]): _as_dict(row["metadata"]) for row in task_rows}
    checks.append(
        _status(
            "required_tasks_registered",
            set(_REQUIRED_TASKS).issubset(task_map.keys()),
            detail={
                "present": sorted(task_map.keys()),
                "missing": sorted(set(_REQUIRED_TASKS) - set(task_map.keys())),
                "metadata": task_map,
            },
        ),
    )

    strategy = str(
        getattr(settings.b2b_churn, "reasoning_synthesis_scheduled_scope_strategy", "") or "",
    ).strip().lower()
    checks.append(
        _status(
            "scheduled_scope_strategy",
            strategy in {"competitive_sets", "full_universe"},
            detail={"configured": strategy or None},
        ),
    )
    latest_rows = await pool.fetch(
        f"""
        SELECT vendor_name, schema_version, synthesis
        FROM (
            SELECT DISTINCT ON (vendor_name)
                   vendor_name, schema_version, synthesis, created_at
            FROM b2b_reasoning_synthesis
            WHERE {_reasoning_v2_schema_predicate()}
            ORDER BY vendor_name, created_at DESC
        ) latest
        ORDER BY vendor_name
        LIMIT $1
        """,
        limit,
    )
    checks.append(
        _status(
            "v2_synthesis_rows_present",
            bool(latest_rows),
            detail={"sampled_rows": len(latest_rows)},
        ),
    )

    rows_with_scope = 0
    rows_with_atoms = 0
    rows_with_delta = 0
    for row in latest_rows:
        synthesis = row["synthesis"]
        if isinstance(synthesis, str):
            synthesis = json.loads(synthesis)
        if not isinstance(synthesis, dict):
            continue
        if isinstance(synthesis.get("scope_manifest"), dict) and synthesis.get("scope_manifest"):
            rows_with_scope += 1
        if isinstance(synthesis.get("reasoning_atoms"), dict) and synthesis.get("reasoning_atoms"):
            rows_with_atoms += 1
        if isinstance(synthesis.get("reasoning_delta"), dict) and synthesis.get("reasoning_delta"):
            rows_with_delta += 1

    checks.append(
        _status(
            "scope_manifest_on_recent_rows",
            rows_with_scope > 0,
            required=False,
            detail={"rows_with_scope_manifest": rows_with_scope, "sampled_rows": len(latest_rows)},
        ),
    )
    checks.append(
        _status(
            "reasoning_atoms_on_recent_rows",
            rows_with_atoms > 0,
            required=False,
            detail={"rows_with_reasoning_atoms": rows_with_atoms, "sampled_rows": len(latest_rows)},
        ),
    )
    checks.append(
        _status(
            "reasoning_delta_on_recent_rows",
            rows_with_delta > 0,
            required=False,
            detail={"rows_with_reasoning_delta": rows_with_delta, "sampled_rows": len(latest_rows)},
        ),
    )

    competitive_set_count = await pool.fetchval(
        "SELECT COUNT(*)::int FROM b2b_competitive_sets",
    )
    checks.append(
        _status(
            "competitive_sets_seeded",
            bool(int(competitive_set_count or 0)),
            required=False,
            detail={"count": int(competitive_set_count or 0)},
        ),
    )

    return {
        "checks": checks,
        "summary": {
            "pass": sum(1 for item in checks if item["status"] == "pass"),
            "warn": sum(1 for item in checks if item["status"] == "warn"),
            "fail": sum(1 for item in checks if item["status"] == "fail"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=25, help="How many recent vendor rows to sample")
    args = parser.parse_args()

    async def _main() -> int:
        try:
            result = await _run(args.limit)
        finally:
            await close_database()
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return _exit_code(result["checks"])

    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
