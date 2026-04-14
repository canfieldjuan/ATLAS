#!/usr/bin/env python3
"""Audit GetApp targets for page availability and reviewer-identity field presence."""

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

from atlas_brain.services.scraping.getapp_audit import capture_getapp_page
from atlas_brain.services.scraping.parsers import ScrapeTarget
from atlas_brain.storage.database import close_database, get_db_pool, init_database


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="How many enabled GetApp targets to audit")
    parser.add_argument("--vendor-name", help="Optional exact vendor-name filter")
    parser.add_argument(
        "--method",
        choices=("web_unlocker", "http", "browser"),
        default="web_unlocker",
        help="Capture method to use for the audit",
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


def _classify_capture_result(capture: dict[str, Any]) -> str:
    status_code = int(capture.get("status_code") or 0)
    analysis = capture.get("analysis") or {}
    if status_code == 404:
        return "broken_target"
    if status_code != 200:
        return "fetch_error"
    if bool(analysis.get("employer_fields_present")):
        return "employer_identity_present"
    if bool(analysis.get("title_fields_present")) or bool(analysis.get("company_size_fields_present")):
        return "context_only_no_employer"
    return "field_empty_for_named_account"


async def _load_targets(args: argparse.Namespace) -> list[ScrapeTarget]:
    pool = get_db_pool()
    filters = ["source = 'getapp'", "enabled = true"]
    params: list[Any] = []
    idx = 1
    if args.vendor_name:
        filters.append(f"LOWER(vendor_name) = LOWER(${idx})")
        params.append(args.vendor_name)
        idx += 1
    limit_param = idx
    params.append(max(1, int(args.limit)))
    rows = await pool.fetch(
        f"""
        SELECT id, vendor_name, product_name, product_slug, product_category,
               max_pages, metadata, last_scrape_date_cutoff,
               last_scrape_status, last_scraped_at
        FROM b2b_scrape_targets
        WHERE {' AND '.join(filters)}
        ORDER BY COALESCE(last_scraped_at, created_at) DESC NULLS LAST
        LIMIT ${limit_param}
        """,
        *params,
    )
    return [_target_from_row(dict(row)) for row in rows]


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    await init_database()
    try:
        targets = await _load_targets(args)
    finally:
        await close_database()

    results: list[dict[str, Any]] = []
    for target in targets:
        try:
            capture = await capture_getapp_page(target, page=1, method=args.method)
            classification = _classify_capture_result(capture)
            analysis = capture.get("analysis") or {}
            review_cards = analysis.get("review_cards") or {}
            results.append(
                {
                    "id": target.id,
                    "vendor_name": target.vendor_name,
                    "product_slug": target.product_slug,
                    "product_category": target.product_category,
                    "status_code": capture.get("status_code"),
                    "classification": classification,
                    "employer_fields_present": bool(analysis.get("employer_fields_present")),
                    "title_fields_present": bool(analysis.get("title_fields_present")),
                    "industry_fields_present": bool(analysis.get("industry_fields_present")),
                    "company_size_fields_present": bool(analysis.get("company_size_fields_present")),
                    "review_card_count": review_cards.get("review_card_count"),
                    "reviewer_title_count": review_cards.get("reviewer_title_count"),
                    "reviewer_company_count": review_cards.get("reviewer_company_count"),
                    "reviewer_industry_count": review_cards.get("reviewer_industry_count"),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "id": target.id,
                    "vendor_name": target.vendor_name,
                    "product_slug": target.product_slug,
                    "classification": "fetch_error",
                    "error": str(exc),
                }
            )

    summary = {
        "targets_audited": len(results),
        "broken_target_count": sum(1 for item in results if item.get("classification") == "broken_target"),
        "fetch_error_count": sum(1 for item in results if item.get("classification") == "fetch_error"),
        "employer_identity_present_count": sum(
            1 for item in results if item.get("classification") == "employer_identity_present"
        ),
        "context_only_no_employer_count": sum(
            1 for item in results if item.get("classification") == "context_only_no_employer"
        ),
        "field_empty_count": sum(
            1 for item in results if item.get("classification") == "field_empty_for_named_account"
        ),
    }
    return {"summary": summary, "targets": results}


def main() -> None:
    args = _build_parser().parse_args()
    report = asyncio.run(_run(args))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
