"""
Auto-discover challenger vendors from displacement data.

Runs weekly after b2b_churn_intelligence. Scans b2b_displacement_edges for
top ``to_vendor`` companies (gaining customers) and inserts/updates
``vendor_targets`` rows with ``target_mode='challenger_intel'``.

Existing ``vendor_target_enrichment`` (daily 19:30) auto-enriches contacts
via Apollo using challenger_intel title scoring.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.b2b.challenger_target_discovery")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Discover challenger vendors from displacement edges."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": True, "error": "Database not ready"}

    # Top to_vendor companies gaining customers in the last 30 days
    rows = await pool.fetch(
        """
        SELECT to_vendor,
               COUNT(DISTINCT from_vendor) AS flow_count,
               SUM(mention_count)          AS total_mentions,
               ARRAY_AGG(DISTINCT from_vendor) AS losing_vendors
        FROM b2b_displacement_edges
        WHERE computed_date > NOW() - INTERVAL '30 days'
          AND signal_strength IN ('strong', 'moderate')
          AND mention_count >= 2
        GROUP BY to_vendor
        HAVING SUM(mention_count) >= 3
        ORDER BY total_mentions DESC
        LIMIT 30
        """
    )

    if not rows:
        logger.info("No qualifying challenger vendors found in displacement data")
        return {"_skip_synthesis": True, "created": 0, "updated": 0}

    created = 0
    updated = 0

    for row in rows:
        to_vendor = row["to_vendor"]
        losing_vendors = list(row["losing_vendors"] or [])
        total_mentions = int(row["total_mentions"] or 0)

        # Check if an active challenger_intel target already exists
        existing = await pool.fetchrow(
            """
            SELECT id, competitors_tracked
            FROM vendor_targets
            WHERE LOWER(company_name) = LOWER($1)
              AND target_mode = 'challenger_intel'
              AND status = 'active'
            """,
            to_vendor,
        )

        # Look up primary use cases from product profiles (if available)
        products_tracked: list[str] = []
        profile = await pool.fetchrow(
            """
            SELECT primary_use_cases
            FROM b2b_product_profiles
            WHERE LOWER(vendor_name) = LOWER($1)
            ORDER BY last_computed_at DESC
            LIMIT 1
            """,
            to_vendor,
        )
        if profile and profile["primary_use_cases"]:
            raw = profile["primary_use_cases"]
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    raw = []
            if isinstance(raw, list):
                products_tracked = [
                    u.get("use_case", str(u)) if isinstance(u, dict) else str(u)
                    for u in raw[:5]
                ]

        if existing:
            # Update competitors_tracked with latest losing vendors
            await pool.execute(
                """
                UPDATE vendor_targets
                SET competitors_tracked = $1,
                    notes = $2,
                    updated_at = NOW()
                WHERE id = $3
                """,
                losing_vendors,
                f"Auto-discovered from displacement data: {total_mentions} mentions across {len(losing_vendors)} flows",
                existing["id"],
            )
            updated += 1
        else:
            # Insert new challenger target
            await pool.execute(
                """
                INSERT INTO vendor_targets
                    (company_name, target_mode, competitors_tracked,
                     products_tracked, status, notes)
                VALUES ($1, 'challenger_intel', $2, $3, 'active', $4)
                """,
                to_vendor,
                losing_vendors,
                products_tracked or None,
                f"Auto-discovered from displacement data: {total_mentions} mentions across {len(losing_vendors)} flows",
            )
            created += 1

    logger.info(
        "Challenger target discovery: %d created, %d updated from %d candidates",
        created, updated, len(rows),
    )

    return {
        "_skip_synthesis": True,
        "created": created,
        "updated": updated,
        "candidates_evaluated": len(rows),
    }
