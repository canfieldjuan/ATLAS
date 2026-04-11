#!/usr/bin/env python3
"""Rebuild grouped company-signal candidates from current review inventory.

Usage:
  python scripts/backfill_company_signal_candidate_groups.py
  python scripts/backfill_company_signal_candidate_groups.py --apply
  python scripts/backfill_company_signal_candidate_groups.py --apply --vendors "Zendesk,HubSpot"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env.backup", override=False)
load_dotenv(ROOT / ".env", override=True)
load_dotenv(ROOT / ".env.local", override=True)

from atlas_brain.autonomous.tasks._b2b_shared import (  # noqa: E402
    _fetch_company_signal_review_candidates,
)
from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (  # noqa: E402
    _build_company_signal_candidate_groups,
    _canonicalize_vendor,
    rebuild_company_signal_candidate_materializations,
)
from atlas_brain.storage.database import close_database, get_db_pool, init_database  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_company_signal_candidate_groups")


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _canonicalize_vendors(vendors: list[str]) -> list[str]:
    canonical: list[str] = []
    seen: set[str] = set()
    for vendor in vendors:
        normalized = _canonicalize_vendor(vendor)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        canonical.append(normalized)
    return canonical


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill grouped company-signal review clusters.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the grouped candidate materialization. Default is dry-run.",
    )
    parser.add_argument(
        "--vendors",
        default="",
        help="Comma-separated vendor filter for a scoped rebuild.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=90,
        help="Lookback window for eligible review candidates.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on fetched review candidates.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of log lines.",
    )
    return parser


def _top_groups(groups: list[dict[str, object]], *, topn: int = 10) -> list[dict[str, object]]:
    ranked = sorted(
        groups,
        key=lambda item: (
            -int(item.get("review_count") or 0),
            -float(item.get("corroborated_confidence_score") or 0),
            str(item.get("company_name") or ""),
            str(item.get("vendor_name") or ""),
        ),
    )
    return [
        {
            "company_name": item.get("company_name"),
            "display_company_name": item.get("display_company_name"),
            "vendor_name": item.get("vendor_name"),
            "review_count": item.get("review_count"),
            "distinct_source_count": item.get("distinct_source_count"),
            "corroborated_confidence_score": item.get("corroborated_confidence_score"),
            "candidate_bucket": item.get("candidate_bucket"),
            "canonical_gap_reason": item.get("canonical_gap_reason"),
        }
        for item in ranked[:topn]
    ]


def _print_summary(result: dict[str, object], *, emit_json: bool) -> None:
    if emit_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    logger.info(
        "candidates=%s canonical_ready_candidates=%s groups=%s canonical_ready_groups=%s persisted_candidates=%s persisted_groups=%s",
        result.get("company_signal_candidates"),
        result.get("canonical_ready_company_signal_candidates"),
        result.get("company_signal_candidate_groups"),
        result.get("canonical_ready_company_signal_candidate_groups"),
        result.get("company_signal_candidates_persisted", 0),
        result.get("company_signal_candidate_groups_persisted", 0),
    )
    for item in result.get("top_groups", []):
        logger.info(
            "group company=%s vendor=%s reviews=%s confidence=%s bucket=%s gap=%s",
            item.get("display_company_name") or item.get("company_name"),
            item.get("vendor_name"),
            item.get("review_count"),
            item.get("corroborated_confidence_score"),
            item.get("candidate_bucket"),
            item.get("canonical_gap_reason"),
        )


async def _plan(pool, *, vendors: list[str], window_days: int, limit: int | None) -> dict[str, object]:
    candidates = await _fetch_company_signal_review_candidates(
        pool,
        window_days=window_days,
        scoped_vendors=vendors or None,
        limit=limit,
    )
    groups = _build_company_signal_candidate_groups(candidates)
    return {
        "mode": "dry_run",
        "window_days": window_days,
        "vendors": vendors,
        "company_signal_candidates": len(candidates),
        "canonical_ready_company_signal_candidates": sum(
            1 for item in candidates if item.get("candidate_bucket") == "canonical_ready"
        ),
        "company_signal_candidate_groups": len(groups),
        "canonical_ready_company_signal_candidate_groups": sum(
            1 for item in groups if item.get("candidate_bucket") == "canonical_ready"
        ),
        "top_groups": _top_groups(groups),
    }


async def _apply(pool, *, vendors: list[str], window_days: int, limit: int | None) -> dict[str, object]:
    materialization_run_id = f"company-signal-groups-backfill:{uuid4()}"
    result = await rebuild_company_signal_candidate_materializations(
        pool,
        window_days=window_days,
        vendors=vendors or None,
        limit=limit,
        materialization_run_id=materialization_run_id,
    )
    result["mode"] = "apply"
    result["window_days"] = window_days
    result["vendors"] = vendors
    result["materialization_run_id"] = materialization_run_id
    return result


async def _run(args: argparse.Namespace) -> dict[str, object]:
    await init_database()
    pool = get_db_pool()
    try:
        vendors = _canonicalize_vendors(_parse_csv(args.vendors))
        if args.apply:
            return await _apply(
                pool,
                vendors=vendors,
                window_days=args.window_days,
                limit=args.limit,
            )
        return await _plan(
            pool,
            vendors=vendors,
            window_days=args.window_days,
            limit=args.limit,
        )
    finally:
        await close_database()


def main() -> None:
    args = _build_parser().parse_args()
    result = asyncio.run(_run(args))
    _print_summary(result, emit_json=args.json)


if __name__ == "__main__":
    main()
