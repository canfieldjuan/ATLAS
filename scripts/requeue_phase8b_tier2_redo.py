#!/usr/bin/env python3
"""One-shot: re-queue Phase 8b rows that drained under the wrong Tier 2 routing.

Background: between 11:48 UTC and the kill at ~11:52 UTC on 2026-04-25, the
90-day re-enrichment drain processed ~541 reviews under
enrichment_local_only=True with no per-tier override, sending Tier 2
(pain_categories / competitor / buyer_authority / sentiment_trajectory)
through local vLLM Qwen3-30B-A3B instead of the intended Claude. The
existing requeue_30d_v3_witness_candidates.py filters on schema<4 and
won't pick these up because vLLM correctly produced v4 phrase_metadata
(Tier 1 work was fine).

This script targets exactly the misrouted-Tier-2 rows by their
``requeue_reason='phase_7_v2_phrase_metadata'`` tag plus a recency
window. It flips them back to ``enrichment_status='pending'`` so the
drain re-runs them. The Tier 1 exact_match cache will hit on the same
(provider, model, payload) key and serve the existing extraction for
free; Tier 2 will miss cache (different model) and run fresh under
Sonnet 4.5 via OpenRouter.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  # dry-run first
  python scripts/requeue_phase8b_tier2_redo.py --dry-run
  # apply
  python scripts/requeue_phase8b_tier2_redo.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from uuid import UUID

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
logger = logging.getLogger("requeue_phase8b_tier2_redo")


_REQUEUE_REASON = "phase_8b_tier2_sonnet_redo"


# Scope: rows with the Phase-8b requeue tag that drained AFTER the bad
# routing went live (the manual 90-day requeue at 11:48 UTC) and BEFORE
# the kill. We use enriched_at as the fence -- it's the timestamp set by
# _persist_enrichment_result on terminal status. Anything earlier than
# the requeue itself is from prior runs that already completed under
# their own routing and shouldn't be retouched here.
_SCOPE_QUERY = """
SELECT id, vendor_name, enrichment_status, enriched_at,
       COALESCE((enrichment->>'enrichment_schema_version')::int, 0) AS sv
  FROM b2b_reviews
 WHERE requeue_reason = 'phase_7_v2_phrase_metadata'
   AND enrichment_status IN ('enriched', 'no_signal')
   AND enriched_at >= $1
"""

_REQUEUE_QUERY = """
UPDATE b2b_reviews
   SET enrichment_status = 'pending',
       enrichment_attempts = 0,
       requeue_reason = $1,
       low_fidelity = false,
       low_fidelity_reasons = '[]'::jsonb,
       low_fidelity_detected_at = NULL,
       enrichment_repair = NULL,
       enrichment_repair_status = NULL,
       enrichment_repair_attempts = 0,
       enrichment_repair_model = NULL,
       enrichment_repaired_at = NULL,
       enrichment_repair_applied_fields = '[]'::jsonb
 WHERE id = ANY($2::uuid[])
"""


async def _audit(pool, since: datetime) -> dict:
    rows = await pool.fetch(_SCOPE_QUERY, since)
    by_status: dict[str, int] = {}
    by_vendor: dict[str, int] = {}
    for row in rows:
        s = str(row["enrichment_status"])
        by_status[s] = by_status.get(s, 0) + 1
        v = str(row["vendor_name"] or "")
        by_vendor[v] = by_vendor.get(v, 0) + 1
    return {
        "total": len(rows),
        "by_status": by_status,
        "by_vendor_top10": sorted(by_vendor.items(), key=lambda x: -x[1])[:10],
        "row_ids": [row["id"] if isinstance(row["id"], UUID) else UUID(str(row["id"])) for row in rows],
    }


async def _main_async(args: argparse.Namespace) -> int:
    since_dt = datetime.fromisoformat(args.since)
    await init_database()
    pool = get_db_pool()
    try:
        audit = await _audit(pool, since_dt)
        logger.info("scope: enriched_at >= %s, requeue_reason='phase_7_v2_phrase_metadata'", since_dt.isoformat())
        logger.info("rows in scope: %d", audit["total"])
        logger.info("by status: %s", audit["by_status"])
        logger.info("top vendors:")
        for vendor, n in audit["by_vendor_top10"]:
            logger.info("  %s -> %d", vendor, n)
        if audit["total"] == 0:
            logger.info("nothing to redo")
            return 0
        if args.dry_run:
            logger.info("[DRY RUN] no rows updated")
            return 0
        await pool.execute(_REQUEUE_QUERY, _REQUEUE_REASON, audit["row_ids"])
        logger.info(
            "re-queued %d reviews to enrichment_status='pending' "
            "with requeue_reason='%s'",
            audit["total"],
            _REQUEUE_REASON,
        )
        logger.info("next: drain via run_b2b_enrichment_until_exhausted.py")
        return 0
    finally:
        await close_database()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        type=str,
        default="2026-04-25 11:48:00+00:00",
        help="ISO timestamp lower bound on enriched_at (default = 90d requeue start).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scope without UPDATEing.",
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_main_async(_parse_args()))


if __name__ == "__main__":
    sys.exit(main())
