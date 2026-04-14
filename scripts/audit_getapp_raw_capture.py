#!/usr/bin/env python3
"""Capture raw GetApp review HTML and audit reviewer fields."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for env_name in (".env", ".env.local"):
    env_path = ROOT / env_name
    if not env_path.exists():
        continue
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

from atlas_brain.services.scraping.getapp_audit import (
    capture_getapp_page,
    write_getapp_capture_artifacts,
)
from atlas_brain.services.scraping.parsers import ScrapeTarget
from atlas_brain.storage.database import close_database, get_db_pool, init_database


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-name", required=True, help="Exact vendor_name in b2b_scrape_targets")
    parser.add_argument("--page", type=int, default=1, help="Page number to capture")
    parser.add_argument(
        "--method",
        choices=("web_unlocker", "http", "browser"),
        default="web_unlocker",
        help="Capture method to use",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/getapp_audit",
        help="Where to write raw capture artifacts",
    )
    return parser


def _normalize_target_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _target_from_row(row: dict[str, Any]) -> ScrapeTarget:
    return ScrapeTarget(
        id=str(row["id"]),
        source="getapp",
        vendor_name=str(row["vendor_name"]),
        product_name=row.get("product_name"),
        product_slug=str(row["product_slug"]),
        product_category=row.get("product_category"),
        max_pages=int(row.get("max_pages") or 1),
        metadata=_normalize_target_metadata(row.get("metadata")),
        date_cutoff=str(row["last_scrape_date_cutoff"]) if row.get("last_scrape_date_cutoff") else None,
    )


async def _load_target(vendor_name: str) -> ScrapeTarget:
    await init_database()
    try:
        row = await get_db_pool().fetchrow(
            """
            SELECT id, vendor_name, product_name, product_slug, product_category,
                   max_pages, metadata, last_scrape_date_cutoff
            FROM b2b_scrape_targets
            WHERE source = 'getapp'
              AND LOWER(vendor_name) = LOWER($1)
            ORDER BY enabled DESC, COALESCE(last_scraped_at, created_at) DESC NULLS LAST
            LIMIT 1
            """,
            vendor_name,
        )
    finally:
        await close_database()
    if not row:
        raise SystemExit(f"No GetApp target found for vendor_name={vendor_name!r}")
    return _target_from_row(dict(row))


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    target = await _load_target(args.vendor_name)
    capture = await capture_getapp_page(target, page=max(1, args.page), method=args.method)
    artifacts = write_getapp_capture_artifacts(
        Path(args.output_dir),
        vendor_name=target.vendor_name,
        page=max(1, args.page),
        capture=capture,
    )
    report = dict(capture)
    report.pop("body", None)
    report["artifacts"] = artifacts
    report["vendor_name"] = target.vendor_name
    report["product_slug"] = target.product_slug
    return report


def _error_report(args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    return {
        "vendor_name": args.vendor_name,
        "method": args.method,
        "page": max(1, args.page),
        "classification": "fetch_error",
        "error": str(exc),
    }


def main() -> None:
    args = _build_parser().parse_args()
    try:
        report = asyncio.run(_run(args))
    except Exception as exc:
        report = _error_report(args, exc)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
