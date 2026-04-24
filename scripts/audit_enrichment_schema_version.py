#!/usr/bin/env python3
"""Audit b2b_reviews enrichment schema-version coverage (Phase 1b/2 rollout).

Phase 2's semantic gates only activate on enrichments that carry v2
phrase_metadata (enrichment_schema_version >= 4). This script reports the
v3/v4 split so we know how much of production is on the new path. Until
Phase 7 re-enrichment runs, the bulk of historical reviews stay on v3.

Output:
  - row counts per enrichment_schema_version
  - per-source breakdown (where new enrichments are landing first)
  - witness coverage: how many witnesses point at v4-enriched reviews

Read-only. Safe on production.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  python scripts/audit_enrichment_schema_version.py
  python scripts/audit_enrichment_schema_version.py --json /tmp/enrichment_audit.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)

from atlas_brain.storage.database import close_database, get_db_pool, init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audit_enrichment_schema_version")


async def _run_audit() -> dict:
    await init_database()
    pool = get_db_pool()
    try:
        version_breakdown = await pool.fetch(
            """
            SELECT
                COALESCE((enrichment->>'enrichment_schema_version')::int, 0) AS schema_version,
                count(*) AS reviews,
                count(*) FILTER (
                    WHERE jsonb_typeof(enrichment->'phrase_metadata') = 'array'
                ) AS reviews_with_phrase_metadata
            FROM b2b_reviews
            WHERE enrichment_status = 'enriched'
              AND enrichment IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """
        )

        per_source = await pool.fetch(
            """
            SELECT
                source,
                count(*) FILTER (
                    WHERE COALESCE((enrichment->>'enrichment_schema_version')::int, 0) >= 4
                ) AS v4_reviews,
                count(*) AS total_reviews
            FROM b2b_reviews
            WHERE enrichment_status = 'enriched'
              AND enrichment IS NOT NULL
            GROUP BY source
            ORDER BY total_reviews DESC
            """
        )

        witness_coverage = await pool.fetchrow(
            """
            SELECT
                count(*) AS total_witnesses,
                count(*) FILTER (
                    WHERE COALESCE((r.enrichment->>'enrichment_schema_version')::int, 0) >= 4
                ) AS witnesses_on_v4
            FROM b2b_vendor_witnesses w
            LEFT JOIN b2b_reviews r ON r.id = w.review_id::uuid
            """
        )

        recent_v4 = await pool.fetch(
            """
            SELECT id, vendor_name, source, enriched_at
            FROM b2b_reviews
            WHERE enrichment_status = 'enriched'
              AND COALESCE((enrichment->>'enrichment_schema_version')::int, 0) >= 4
            ORDER BY enriched_at DESC NULLS LAST
            LIMIT 10
            """
        )
    finally:
        await close_database()

    total_reviews = sum(int(r["reviews"]) for r in version_breakdown)
    return {
        "total_enriched_reviews": total_reviews,
        "by_schema_version": [
            {
                "schema_version": int(r["schema_version"]),
                "reviews": int(r["reviews"]),
                "reviews_with_phrase_metadata": int(r["reviews_with_phrase_metadata"]),
                "share": round(int(r["reviews"]) / total_reviews, 4) if total_reviews else None,
            }
            for r in version_breakdown
        ],
        "per_source": [
            {
                "source": r["source"],
                "v4_reviews": int(r["v4_reviews"]),
                "total_reviews": int(r["total_reviews"]),
                "v4_share": round(int(r["v4_reviews"]) / int(r["total_reviews"]), 4)
                if r["total_reviews"] else None,
            }
            for r in per_source
        ],
        "witness_coverage": {
            "total_witnesses": int(witness_coverage["total_witnesses"]),
            "witnesses_on_v4": int(witness_coverage["witnesses_on_v4"]),
            "v4_share": round(
                int(witness_coverage["witnesses_on_v4"]) /
                int(witness_coverage["total_witnesses"]),
                4,
            ) if witness_coverage["total_witnesses"] else None,
        },
        "recent_v4_samples": [
            {
                "id": str(r["id"]),
                "vendor_name": r["vendor_name"],
                "source": r["source"],
                "enriched_at": r["enriched_at"].isoformat() if r["enriched_at"] else None,
            }
            for r in recent_v4
        ],
    }


def _format_text(report: dict) -> str:
    lines = [
        "=" * 64,
        "Enrichment schema-version audit",
        "=" * 64,
        f"Total enriched reviews: {report['total_enriched_reviews']:,}",
        "",
        "By schema version:",
    ]
    for entry in report["by_schema_version"]:
        share_pct = (
            f"{100 * entry['share']:.2f}%" if entry["share"] is not None else "n/a"
        )
        lines.append(
            f"  v{entry['schema_version']}: "
            f"{entry['reviews']:,} reviews ({share_pct}), "
            f"{entry['reviews_with_phrase_metadata']:,} with phrase_metadata"
        )

    lines.append("")
    lines.append("Witness coverage (v4 = phrase-tagged):")
    wc = report["witness_coverage"]
    wc_share = f"{100 * wc['v4_share']:.2f}%" if wc["v4_share"] is not None else "n/a"
    lines.append(
        f"  {wc['witnesses_on_v4']:,} of {wc['total_witnesses']:,} witnesses "
        f"point at v4 reviews ({wc_share})"
    )

    lines.append("")
    lines.append("Top sources by total enriched reviews (v4 share):")
    for entry in report["per_source"][:10]:
        share_pct = (
            f"{100 * entry['v4_share']:.2f}%"
            if entry["v4_share"] is not None else "n/a"
        )
        lines.append(
            f"  {entry['source']:20s} "
            f"v4={entry['v4_reviews']:>5,} / total={entry['total_reviews']:>6,} "
            f"({share_pct})"
        )

    samples = report["recent_v4_samples"]
    if samples:
        lines.append("")
        lines.append(f"Most recent v4 enrichments (top {len(samples)}):")
        for s in samples:
            lines.append(
                f"  {s['enriched_at'] or '(none)':30s} "
                f"{s['source']:14s} {s['vendor_name']}  id={s['id']}"
            )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional: also write the report to this JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = asyncio.run(_run_audit())
    print(_format_text(report))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("wrote JSON report to %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
