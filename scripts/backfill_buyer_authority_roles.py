#!/usr/bin/env python3
"""Deterministically backfill buyer-authority role cues on enriched reviews.

Uses the same canonicalization and fallback inference helpers as production
enrichment. This updates the enrichment JSON in-place when role_type,
role_level, or decision_maker can be recovered safely.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_buyer_authority_roles")


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# APPROVED-ENRICHMENT-READ: buyer_authority
# Reason: backfill/migration script — direct enrichment access required
def _candidate_role_clause() -> str:
    return """
      AND enrichment->'buyer_authority' IS NOT NULL
      AND enrichment->'buyer_authority' != 'null'::jsonb
      AND (
        COALESCE(enrichment->'buyer_authority'->>'role_type', '') = ''
        OR COALESCE(enrichment->'buyer_authority'->>'role_type', '') = 'unknown'
        OR COALESCE(enrichment->'buyer_authority'->>'role_type', '') NOT IN (
          'economic_buyer', 'champion', 'evaluator', 'end_user', 'unknown'
        )
      )
    """


async def _fetch_rows(pool, args) -> list[dict[str, Any]]:
    clauses = [_candidate_role_clause()]
    params: list[Any] = []
    idx = 1
    if args.vendor:
        clauses.append(f"AND vendor_name = ${idx}")
        params.append(args.vendor)
        idx += 1
    if args.source:
        sources = [item.strip() for item in args.source.split(",") if item.strip()]
        if sources:
            clauses.append(f"AND source = ANY(${idx}::text[])")
            params.append(sources)
            idx += 1
    params.append(args.limit)
    return await pool.fetch(
        f"""
        SELECT id, vendor_name, source, content_type, summary, review_text,
               pros, cons, reviewer_title, reviewer_company, company_size_raw,
               reviewer_industry, rating, rating_max, raw_metadata,
               enrichment, enriched_at
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
        {' '.join(clauses)}
        ORDER BY enriched_at DESC NULLS LAST, id DESC
        LIMIT ${idx}
        """,
        *params,
    )


async def _apply_updates(pool, updates: list[tuple[str, Any]]) -> None:
    if not updates:
        return
    await pool.executemany(
        """
        UPDATE b2b_reviews
        SET enrichment = $1::jsonb
        WHERE id = $2
        """,
        updates,
    )


async def run(args) -> None:
    from atlas_brain.autonomous.tasks.b2b_enrichment import _finalize_enrichment_for_persist
    from atlas_brain.storage.database import close_database, get_db_pool, init_database

    await init_database()
    pool = get_db_pool()
    rows = await _fetch_rows(pool, args)
    logger.info("Scanned %d candidate reviews", len(rows))

    proposed: list[tuple[str, Any]] = []
    raw_to_new = Counter()
    inferred_counts = Counter()
    skipped_unknown = 0
    samples: list[dict[str, str]] = []

    for row in rows:
        enrichment = _as_dict(row.get("enrichment"))
        original_role = str(
            _as_dict(enrichment.get("buyer_authority")).get("role_type") or "unknown"
        ).strip() or "unknown"
        original_level = str(
            _as_dict(enrichment.get("reviewer_context")).get("role_level") or "unknown"
        ).strip() or "unknown"
        original_dm = _as_dict(enrichment.get("reviewer_context")).get("decision_maker")
        finalized, _ = _finalize_enrichment_for_persist(enrichment, dict(row))
        if not finalized:
            skipped_unknown += 1
            continue
        enrichment = finalized
        updated_role = str(
            _as_dict(enrichment.get("buyer_authority")).get("role_type") or "unknown"
        ).strip() or "unknown"
        updated_level = str(
            _as_dict(enrichment.get("reviewer_context")).get("role_level") or "unknown"
        ).strip() or "unknown"
        updated_dm = _as_dict(enrichment.get("reviewer_context")).get("decision_maker")
        changed = (
            updated_role != original_role
            or updated_level != original_level
            or updated_dm != original_dm
        )
        if updated_role == "unknown" and not changed:
            skipped_unknown += 1
            continue
        if not changed:
            continue
        proposed.append((json.dumps(enrichment), row["id"]))
        raw_to_new[f"{original_role}->{updated_role}"] += 1
        inferred_counts[updated_role] += 1
        if len(samples) < 10:
            samples.append(
                {
                    "vendor": str(row.get("vendor_name") or ""),
                    "source": str(row.get("source") or ""),
                    "title": str(row.get("reviewer_title") or ""),
                    "from": original_role,
                    "to": updated_role,
                }
            )

    logger.info("Proposed updates: %d", len(proposed))
    logger.info("Still unknown after inference: %d", skipped_unknown)
    if raw_to_new:
        logger.info("Transitions: %s", dict(raw_to_new))
    if inferred_counts:
        logger.info("Resolved roles: %s", dict(inferred_counts))
    for sample in samples:
        logger.info(
            "Sample [%s] %s | %s | %s -> %s",
            sample["source"],
            sample["vendor"],
            sample["title"] or "unknown_title",
            sample["from"],
            sample["to"],
        )

    if not args.dry_run:
        await _apply_updates(pool, proposed)
        logger.info("Applied %d updates", len(proposed))

    await close_database()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backfill enrichment.buyer_authority.role_type deterministically",
    )
    ap.add_argument("--vendor", default=None, help="Target one vendor_name")
    ap.add_argument(
        "--source",
        default=None,
        help="Comma-separated source filter (default: all)",
    )
    ap.add_argument("--limit", type=int, default=5000, help="Max reviews to scan")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show proposed changes without updating rows",
    )
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
