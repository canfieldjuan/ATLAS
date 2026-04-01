#!/usr/bin/env python3
"""Rebuild stale B2B reasoning synthesis rows and optionally refresh reports.

Usage:
  python scripts/rebuild_b2b_reasoning_delivery.py
  python scripts/rebuild_b2b_reasoning_delivery.py --apply
  python scripts/rebuild_b2b_reasoning_delivery.py --apply --vendors "Zendesk,Zoho Desk"
  python scripts/rebuild_b2b_reasoning_delivery.py --apply --include-reports
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from types import SimpleNamespace
from uuid import uuid4

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.storage.database import close_database, get_db_pool, init_database

VALID_REPORT_TYPES = (
    "weekly_churn_feed",
    "vendor_scorecard",
    "displacement_report",
    "category_overview",
)


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


async def _discover_stale_vendors(
    pool,
    *,
    vendors: list[str],
    limit: int | None,
) -> list[dict]:
    params: list[object] = []
    vendor_filter = ""
    limit_clause = ""
    if vendors:
        params.append([vendor.lower() for vendor in vendors])
        vendor_filter = "WHERE LOWER(vendor_name) = ANY($1::text[])"
    if limit:
        params.append(int(limit))
        limit_clause = f"LIMIT ${len(params)}"

    query = f"""
        WITH latest AS (
            SELECT DISTINCT ON (LOWER(vendor_name))
                   vendor_name,
                   as_of_date,
                   created_at,
                   synthesis
            FROM b2b_reasoning_synthesis
            {vendor_filter}
            ORDER BY LOWER(vendor_name), as_of_date DESC, created_at DESC
        )
        SELECT
            vendor_name,
            as_of_date,
            created_at,
            jsonb_path_exists(synthesis, '$.packet_artifacts.witness_pack[*]') AS has_witness_pack,
            jsonb_path_exists(synthesis, '$.reference_ids.metric_ids[*]') AS has_metric_refs,
            jsonb_path_exists(synthesis, '$.reference_ids.witness_ids[*]') AS has_witness_refs
        FROM latest
        WHERE NOT jsonb_path_exists(synthesis, '$.packet_artifacts.witness_pack[*]')
           OR NOT jsonb_path_exists(synthesis, '$.reference_ids.metric_ids[*]')
           OR NOT jsonb_path_exists(synthesis, '$.reference_ids.witness_ids[*]')
        ORDER BY created_at DESC
        {limit_clause}
    """
    rows = await pool.fetch(query, *params)
    return [dict(row) for row in rows]


async def _summarize_report_coverage(
    pool,
    *,
    report_types: list[str],
) -> list[dict]:
    rows = await pool.fetch(
        """
        WITH latest AS (
            SELECT DISTINCT ON (
                report_type,
                COALESCE(vendor_filter, '')
            )
                   report_type,
                   vendor_filter,
                   report_date,
                   created_at,
                   intelligence_data
            FROM b2b_intelligence
            WHERE report_type = ANY($1::text[])
            ORDER BY
                report_type,
                COALESCE(vendor_filter, ''),
                report_date DESC,
                created_at DESC
        )
        SELECT
            report_type,
            COUNT(*) AS latest_rows,
            COUNT(*) FILTER (
                WHERE jsonb_path_exists(intelligence_data, '$.**.reference_ids')
                   OR jsonb_path_exists(intelligence_data, '$.**.reasoning_reference_ids')
            ) AS rows_with_reference_ids
        FROM latest
        GROUP BY report_type
        ORDER BY report_type
        """,
        report_types,
    )
    return [dict(row) for row in rows]


async def _run_reasoning_rebuild(
    *,
    vendors: list[str],
    rebuild_cross_vendor: bool,
) -> dict:
    from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run as run_reasoning_synthesis

    metadata = {
        "force_cross_vendor": rebuild_cross_vendor,
    }
    if vendors:
        metadata["force"] = True
        metadata["test_vendors"] = vendors
    task = SimpleNamespace(id=uuid4(), metadata=metadata)
    return await run_reasoning_synthesis(task)


async def _run_report_refresh(
    *,
    vendors: list[str],
    report_type: str,
) -> dict:
    from atlas_brain.autonomous.tasks.b2b_churn_reports import run as run_churn_reports

    metadata: dict[str, object] = {"test_report_type": report_type}
    if vendors:
        metadata["test_vendors"] = vendors
        metadata["persist_scoped_reports"] = True
    task = SimpleNamespace(id=uuid4(), metadata=metadata)
    return await run_churn_reports(task)


async def _main(args: argparse.Namespace) -> None:
    selected_vendors = _parse_csv(args.vendors)
    selected_report_types = _parse_csv(args.report_types) or list(VALID_REPORT_TYPES)
    invalid_report_types = sorted(set(selected_report_types).difference(VALID_REPORT_TYPES))
    if invalid_report_types:
        raise SystemExit(f"Invalid report types: {invalid_report_types}")

    await init_database()
    pool = get_db_pool()
    try:
        stale_vendors = await _discover_stale_vendors(
            pool,
            vendors=selected_vendors,
            limit=args.limit,
        )
        vendor_names = selected_vendors or [row["vendor_name"] for row in stale_vendors]
        report_summary = await _summarize_report_coverage(
            pool,
            report_types=selected_report_types,
        )

        result: dict[str, object] = {
            "apply": bool(args.apply),
            "requested_vendors": selected_vendors,
            "stale_vendor_rows": stale_vendors,
            "report_coverage": report_summary,
        }

        if not args.apply:
            print(json.dumps(result, indent=2, default=str))
            return

        should_run_reasoning_rebuild = bool(vendor_names) or bool(args.rebuild_cross_vendor)
        if should_run_reasoning_rebuild:
            rebuild_result = await _run_reasoning_rebuild(
                vendors=vendor_names,
                rebuild_cross_vendor=bool(args.rebuild_cross_vendor),
            )
        else:
            rebuild_result = {"_skip_synthesis": "No stale vendors selected"}
        result["reasoning_rebuild"] = rebuild_result

        if args.include_reports:
            report_results: dict[str, dict] = {}
            skipped: dict[str, str] = {}
            for report_type in selected_report_types:
                if vendor_names and report_type != "vendor_scorecard":
                    skipped[report_type] = "Scoped refresh only supports vendor_scorecard"
                    continue
                report_results[report_type] = await _run_report_refresh(
                    vendors=vendor_names,
                    report_type=report_type,
                )
            if report_results:
                result["report_refresh"] = report_results
            if skipped:
                result["report_refresh_skipped"] = skipped

        print(json.dumps(result, indent=2, default=str))
    finally:
        await close_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Execute rebuild tasks.")
    parser.add_argument("--vendors", default="", help="Comma-separated vendor names.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum stale vendor rows to inspect.")
    parser.add_argument(
        "--include-reports",
        action="store_true",
        help="Refresh report delivery after rebuilding reasoning synthesis.",
    )
    parser.add_argument(
        "--report-types",
        default=",".join(VALID_REPORT_TYPES),
        help="Comma-separated report types to refresh.",
    )
    parser.add_argument(
        "--rebuild-cross-vendor",
        action="store_true",
        help="Force a cross-vendor synthesis rebuild during apply.",
    )
    asyncio.run(_main(parser.parse_args()))
