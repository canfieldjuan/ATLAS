#!/usr/bin/env python3
"""Read-only smoke audit for blog publish and campaign send/approval gates.

Usage:
  python scripts/smoke_truthful_artifacts.py
  python scripts/smoke_truthful_artifacts.py --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.api.b2b_campaigns import _campaign_revalidation_audit
from atlas_brain.api.blog_admin import _publish_revalidation_report
from atlas_brain.storage.database import close_database, get_db_pool, init_database


async def _main(limit: int) -> None:
    await init_database()
    pool = get_db_pool()
    try:
        blog_rows = await pool.fetch(
            """
            SELECT *
            FROM blog_posts
            WHERE status IN ('draft', 'rejected')
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
        campaign_rows = await pool.fetch(
            """
            SELECT bc.*, cs.company_context
            FROM b2b_campaigns bc
            LEFT JOIN campaign_sequences cs ON cs.id = bc.sequence_id
            WHERE bc.status IN ('draft', 'approved', 'queued')
            ORDER BY bc.created_at DESC
            LIMIT $1
            """,
            limit,
        )
    finally:
        await close_database()

    blog_items: list[dict] = []
    for row in blog_rows:
        report = _publish_revalidation_report(row)
        blog_items.append(
            {
                "slug": row["slug"],
                "status": row["status"],
                "score": report.get("score"),
                "threshold": report.get("threshold"),
                "result": report.get("status"),
                "blocking_issues": list(report.get("blocking_issues") or [])[:3],
                "warnings": list(report.get("warnings") or [])[:3],
            }
        )

    campaign_items: list[dict] = []
    for row in campaign_rows:
        audit = _campaign_revalidation_audit(
            campaign=dict(row),
            company_context=row.get("company_context"),
        )
        campaign_items.append(
            {
                "id": str(row["id"]),
                "company_name": row["company_name"],
                "status": row["status"],
                "channel": row["channel"],
                "result": audit.get("status") if audit else "no_context",
                "blocking_issues": list(audit.get("blocking_issues") or [])[:3] if audit else [],
                "warnings": list(audit.get("warnings") or [])[:3] if audit else [],
            }
        )

    print(
        json.dumps(
            {
                "limit": limit,
                "blogs": {
                    "scanned": len(blog_items),
                    "passing": sum(1 for item in blog_items if item["result"] == "pass"),
                    "failing": sum(1 for item in blog_items if item["result"] == "fail"),
                    "preview": blog_items[:10],
                },
                "campaigns": {
                    "scanned": len(campaign_items),
                    "passing": sum(1 for item in campaign_items if item["result"] == "pass"),
                    "failing": sum(1 for item in campaign_items if item["result"] == "fail"),
                    "no_context": sum(1 for item in campaign_items if item["result"] == "no_context"),
                    "preview": campaign_items[:10],
                },
            },
            ensure_ascii=True,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Rows per artifact type to sample.")
    args = parser.parse_args()
    asyncio.run(_main(limit=args.limit))
