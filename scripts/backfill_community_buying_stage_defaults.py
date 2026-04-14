from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any
from pathlib import Path

import asyncpg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.autonomous.tasks import b2b_enrichment
from atlas_brain.storage.config import db_settings

COMMUNITY_SOURCES = (
    "reddit",
    "hackernews",
    "quora",
    "twitter",
    "github",
    "stackoverflow",
)


def _should_downgrade(row: dict[str, Any]) -> bool:
    enrichment = row.get("enrichment")
    if isinstance(enrichment, str):
        try:
            enrichment = json.loads(enrichment)
        except (TypeError, json.JSONDecodeError):
            return False
    if not isinstance(enrichment, dict):
        return False

    buyer_authority = enrichment.get("buyer_authority")
    if not isinstance(buyer_authority, dict):
        return False
    if str(buyer_authority.get("buying_stage") or "").strip().lower() != "post_purchase":
        return False

    review_blob = " ".join(
        str(row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    if b2b_enrichment._has_post_purchase_signal(row, review_blob):
        return False

    churn = enrichment.get("churn_signals")
    if isinstance(churn, dict):
        if churn.get("contract_renewal_mentioned") or churn.get("renewal_timing"):
            return False
        if churn.get("actively_evaluating") or churn.get("migration_in_progress"):
            return False

    return True


async def _run(limit: int, apply: bool) -> None:
    conn = await asyncpg.connect(
        host=db_settings.host,
        port=db_settings.port,
        database=db_settings.database,
        user=db_settings.user,
        password=db_settings.password,
        timeout=db_settings.connect_timeout,
        command_timeout=60,
    )
    try:
        rows = await conn.fetch(
            """
            SELECT id, source, summary, review_text, pros, cons, enrichment
            FROM b2b_reviews
            WHERE enrichment_status = 'enriched'
              AND duplicate_of_review_id IS NULL
              AND source = ANY($1::text[])
              AND lower(coalesce(enrichment->'buyer_authority'->>'buying_stage', '')) = 'post_purchase'
            ORDER BY enriched_at DESC NULLS LAST, id DESC
            LIMIT $2
            """,
            list(COMMUNITY_SOURCES),
            limit,
        )
        candidates = [dict(row) for row in rows]
        to_update = [row for row in candidates if _should_downgrade(row)]

        print(f"scanned={len(candidates)}")
        print(f"downgrade_candidates={len(to_update)}")
        if not to_update:
            return

        for row in to_update[:10]:
            print(f"candidate id={row['id']} source={row['source']}")

        if not apply:
            return

        await conn.executemany(
            """
            UPDATE b2b_reviews
            SET enrichment = jsonb_set(
                    enrichment,
                    '{buyer_authority,buying_stage}',
                    to_jsonb('unknown'::text),
                    true
                )
            WHERE id = $1::uuid
            """,
            [(row["id"],) for row in to_update],
        )
        print(f"updated={len(to_update)}")
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Downgrade weak community post_purchase buying-stage defaults.")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    asyncio.run(_run(limit=args.limit, apply=args.apply))


if __name__ == "__main__":
    main()
