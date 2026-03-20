#!/usr/bin/env python3
"""Backfill account ownership for safe legacy vendor_targets."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env.backup", override=False)
load_dotenv(ROOT / ".env", override=True)
load_dotenv(ROOT / ".env.local", override=True)

from atlas_brain.services.vendor_target_backfill import (  # noqa: E402
    apply_legacy_vendor_target_account_backfill,
    plan_legacy_vendor_target_account_backfill,
)
from atlas_brain.storage.database import close_database, init_database, get_db_pool  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_vendor_target_accounts")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Claim safe legacy vendor_targets into account scope.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the backfill. Default is dry-run.",
    )
    parser.add_argument(
        "--company-name",
        help="Exact company_name filter for a targeted run.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("vendor_retention", "challenger_intel"),
        help="Restrict to one target mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit how many claimable targets are selected.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of log lines.",
    )
    return parser


def _print_summary(result: dict[str, object], *, emit_json: bool) -> None:
    if emit_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    logger.info(
        "legacy=%s claimable=%s selected=%s skipped=%s",
        result.get("legacy_targets"),
        result.get("claimable_targets_total"),
        result.get("claimable_targets_selected"),
        result.get("skipped_targets"),
    )
    logger.info(
        "direct=%s own=%s competitor=%s overlap=%s applied=%s already_claimed=%s",
        result.get("direct_source_matches"),
        result.get("exact_own_matches"),
        result.get("exact_competitor_matches"),
        result.get("challenger_overlap_matches"),
        result.get("applied", 0),
        result.get("already_claimed", 0),
    )
    for candidate in result.get("candidates", []):
        logger.info(
            "candidate company=%s mode=%s account=%s reason=%s",
            candidate.get("company_name"),
            candidate.get("target_mode"),
            candidate.get("account_id"),
            candidate.get("claim_reason"),
        )


async def _run(args: argparse.Namespace) -> dict[str, object]:
    await init_database()
    pool = get_db_pool()
    try:
        kwargs = {
            "company_name": args.company_name,
            "target_mode": args.target_mode,
            "limit": args.limit,
        }
        if args.apply:
            return await apply_legacy_vendor_target_account_backfill(pool, **kwargs)
        return await plan_legacy_vendor_target_account_backfill(pool, **kwargs)
    finally:
        await close_database()


def main() -> None:
    args = _build_parser().parse_args()
    result = asyncio.run(_run(args))
    _print_summary(result, emit_json=args.json)


if __name__ == "__main__":
    main()
