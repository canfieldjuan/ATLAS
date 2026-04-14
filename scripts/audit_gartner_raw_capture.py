#!/usr/bin/env python3
"""Capture raw Gartner pages and audit reviewer field presence."""

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

from atlas_brain.config import settings
from atlas_brain.services.scraping.gartner_audit import (
    capture_gartner_page,
    write_gartner_capture_artifacts,
)
from atlas_brain.services.scraping.parsers import ScrapeTarget
from atlas_brain.storage.database import close_database, get_db_pool, init_database


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-id", help="Specific b2b_scrape_targets.id to audit")
    parser.add_argument("--vendor-name", help="Resolve a Gartner target by vendor name")
    parser.add_argument("--product-slug", help="Resolve a Gartner target by product_slug")
    parser.add_argument("--pages", type=int, default=1, help="How many pages to capture")
    parser.add_argument(
        "--method",
        choices=("auto", "http", "web_unlocker", "both"),
        default="auto",
        help="Capture path to use",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "audits" / "gartner_raw"),
        help="Directory where raw HTML and metadata artifacts are written",
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
        source="gartner",
        vendor_name=str(row["vendor_name"]),
        product_name=row.get("product_name"),
        product_slug=str(row["product_slug"]),
        product_category=row.get("product_category"),
        max_pages=int(row.get("max_pages") or 1),
        metadata=_normalize_target_metadata(row.get("metadata")),
        date_cutoff=str(row["last_scrape_date_cutoff"]) if row.get("last_scrape_date_cutoff") else None,
    )


async def _resolve_target(args: argparse.Namespace) -> ScrapeTarget:
    pool = get_db_pool()
    filters = ["source = 'gartner'"]
    params: list[Any] = []
    idx = 1
    if args.target_id:
        filters.append(f"id = ${idx}::uuid")
        params.append(args.target_id)
        idx += 1
    if args.vendor_name:
        filters.append(f"LOWER(vendor_name) = LOWER(${idx})")
        params.append(args.vendor_name)
        idx += 1
    if args.product_slug:
        filters.append(f"product_slug = ${idx}")
        params.append(args.product_slug)
        idx += 1
    row = await pool.fetchrow(
        f"""
        SELECT id, vendor_name, product_name, product_slug, product_category,
               max_pages, metadata, last_scrape_date_cutoff
        FROM b2b_scrape_targets
        WHERE {' AND '.join(filters)}
        ORDER BY enabled DESC, COALESCE(last_scraped_at, created_at) DESC NULLS LAST
        LIMIT 1
        """,
        *params,
    )
    if not row:
        raise RuntimeError("No matching Gartner target found")
    return _target_from_row(dict(row))


def _selected_methods(method: str) -> list[str]:
    if method == "both":
        return ["web_unlocker", "http"]
    if method == "http":
        return ["http"]
    if method == "web_unlocker":
        return ["web_unlocker"]
    methods: list[str] = []
    unlocker_domains = {
        d.strip().lower()
        for d in str(settings.b2b_scrape.web_unlocker_domains or "").split(",")
        if d.strip()
    }
    if str(settings.b2b_scrape.web_unlocker_url or "").strip() and "gartner.com" in unlocker_domains:
        methods.append("web_unlocker")
    methods.append("http")
    return methods


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    await init_database()
    try:
        target = await _resolve_target(args)
    finally:
        await close_database()

    output_dir = Path(args.output_dir)
    methods = _selected_methods(args.method)
    captures: list[dict[str, Any]] = []
    for page in range(1, max(1, args.pages) + 1):
        for method in methods:
            try:
                capture = await capture_gartner_page(target, page=page, method=method)
            except Exception as exc:
                captures.append({"page": page, "method": method, "error": str(exc)})
                if args.method == "auto":
                    continue
                continue
            artifact_paths = write_gartner_capture_artifacts(
                output_dir,
                vendor_name=target.vendor_name,
                page=page,
                capture=capture,
            )
            capture["page"] = page
            capture["artifacts"] = artifact_paths
            captures.append(capture)
            if args.method == "auto" and int(capture.get("status_code") or 0) == 200:
                break

    employer_fields_present = any(
        bool(capture.get("analysis", {}).get("employer_fields_present"))
        for capture in captures
        if not capture.get("error")
    )
    title_fields_present = any(
        bool(capture.get("analysis", {}).get("title_fields_present"))
        for capture in captures
        if not capture.get("error")
    )
    company_size_fields_present = any(
        bool(capture.get("analysis", {}).get("company_size_fields_present"))
        for capture in captures
        if not capture.get("error")
    )
    industry_fields_present = any(
        bool(capture.get("analysis", {}).get("industry_fields_present"))
        for capture in captures
        if not capture.get("error")
    )
    return {
        "target": {
            "id": target.id,
            "vendor_name": target.vendor_name,
            "product_slug": target.product_slug,
            "product_category": target.product_category,
        },
        "captures": captures,
        "summary": {
            "methods_attempted": methods,
            "employer_fields_present": employer_fields_present,
            "title_fields_present": title_fields_present,
            "company_size_fields_present": company_size_fields_present,
            "industry_fields_present": industry_fields_present,
        },
    }


def main() -> None:
    args = _build_parser().parse_args()
    report = asyncio.run(_run(args))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
