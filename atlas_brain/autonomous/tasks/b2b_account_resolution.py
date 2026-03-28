"""B2B account resolution: resolve anonymous reviewers to named companies.

Runs after enrichment. Reads enriched reviews that lack a resolution row,
runs the deterministic resolver, persists results to b2b_account_resolution,
and backfills reviewer_company on b2b_reviews for high/medium confidence.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...services.company_normalization import normalize_company_name
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.b2b_account_resolution")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task: resolve anonymous review authors to named companies."""
    cfg = settings.b2b_churn
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    batch_size = getattr(cfg, "account_resolution_batch_size", 100)
    backfill_min = getattr(cfg, "account_resolution_backfill_min_confidence", "medium")
    backfill_labels = {"high"}
    if backfill_min == "medium":
        backfill_labels.add("medium")
    elif backfill_min == "low":
        backfill_labels.update(("medium", "low"))

    t0 = time.monotonic()

    # Claim: enriched reviews without a resolution row
    rows = await pool.fetch(
        """
        SELECT r.id, r.source, r.source_url, r.reviewer_name,
               r.reviewer_title, r.reviewer_company, r.reviewer_company_norm,
               r.company_size_raw, r.reviewer_industry,
               r.review_text, r.enrichment, r.raw_metadata,
               r.vendor_name, r.product_category, r.rating, r.rating_max
        FROM b2b_reviews r
        LEFT JOIN b2b_account_resolution ar ON ar.review_id = r.id
        WHERE r.enrichment_status = 'enriched'
          AND ar.id IS NULL
        ORDER BY r.enriched_at DESC
        LIMIT $1
        """,
        batch_size,
    )

    if not rows:
        return {"_skip_synthesis": "No reviews pending resolution"}

    from ...services.b2b.account_resolver import resolve_review

    # Build vendor blocklist
    vendor_names = {r["vendor_name"] for r in rows if r["vendor_name"]}
    blocked_by_vendor: dict[str, set[str]] = {}
    for vn in vendor_names:
        norm = normalize_company_name(vn)
        blocked_by_vendor[vn] = {norm} if norm else set()

    resolved_count = 0
    unresolved_count = 0
    backfilled_count = 0
    high_count = 0
    medium_count = 0
    low_count = 0
    errors = 0

    for row in rows:
        review_dict = dict(row)
        # Parse JSONB fields
        for field in ("enrichment", "raw_metadata"):
            val = review_dict.get(field)
            if isinstance(val, str):
                try:
                    review_dict[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    review_dict[field] = {}

        vendor = review_dict.get("vendor_name") or ""
        blocked = blocked_by_vendor.get(vendor, set())

        try:
            result = resolve_review(
                review_dict,
                vendor_name=vendor,
                blocked_names=blocked,
            )
        except Exception:
            logger.warning(
                "Resolution failed for review %s", row["id"], exc_info=True,
            )
            errors += 1
            continue

        # Determine status
        if result.resolved_company_name:
            status = "resolved"
            resolved_count += 1
        else:
            status = "unresolved"
            unresolved_count += 1

        if result.confidence_label == "high":
            high_count += 1
        elif result.confidence_label == "medium":
            medium_count += 1
        elif result.confidence_label == "low":
            low_count += 1

        # Persist resolution
        try:
            await pool.execute(
                """
                INSERT INTO b2b_account_resolution (
                    review_id, source, source_item_url,
                    author_handle, author_profile_url,
                    reviewer_company_raw,
                    resolved_company_name, normalized_company_name,
                    confidence_score, confidence_label,
                    resolution_method, resolution_evidence,
                    resolution_status, resolved_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13, NOW()
                )
                ON CONFLICT (review_id) DO UPDATE SET
                    resolved_company_name = EXCLUDED.resolved_company_name,
                    normalized_company_name = EXCLUDED.normalized_company_name,
                    confidence_score = EXCLUDED.confidence_score,
                    confidence_label = EXCLUDED.confidence_label,
                    resolution_method = EXCLUDED.resolution_method,
                    resolution_evidence = EXCLUDED.resolution_evidence,
                    resolution_status = EXCLUDED.resolution_status,
                    resolved_at = EXCLUDED.resolved_at
                """,
                row["id"],
                row["source"],
                row["source_url"],
                row["reviewer_name"],
                None,  # author_profile_url -- future enhancement
                row["reviewer_company"],
                result.resolved_company_name,
                result.normalized_company_name,
                result.confidence_score,
                result.confidence_label,
                result.resolution_method,
                json.dumps(result.to_evidence_json()),
                status,
            )
        except Exception:
            logger.warning(
                "Failed to persist resolution for %s", row["id"], exc_info=True,
            )
            errors += 1
            continue

        # Backfill reviewer_company on b2b_reviews
        if (
            result.resolved_company_name
            and result.confidence_label in backfill_labels
            and not (row["reviewer_company"] or "").strip()
        ):
            try:
                await pool.execute(
                    """
                    UPDATE b2b_reviews
                    SET reviewer_company = $2,
                        reviewer_company_norm = $3
                    WHERE id = $1
                      AND (reviewer_company IS NULL OR reviewer_company = '')
                    """,
                    row["id"],
                    result.resolved_company_name,
                    result.normalized_company_name,
                )
                backfilled_count += 1
            except Exception:
                logger.warning(
                    "Failed to backfill reviewer_company for %s",
                    row["id"], exc_info=True,
                )

    elapsed = round(time.monotonic() - t0, 1)
    logger.info(
        "Account resolution: %d resolved, %d unresolved, %d backfilled, "
        "%d errors (high=%d med=%d low=%d) %.1fs",
        resolved_count, unresolved_count, backfilled_count,
        errors, high_count, medium_count, low_count, elapsed,
    )

    return {
        "_skip_synthesis": "Account resolution complete",
        "reviews_processed": len(rows),
        "resolved": resolved_count,
        "unresolved": unresolved_count,
        "backfilled": backfilled_count,
        "high_confidence": high_count,
        "medium_confidence": medium_count,
        "low_confidence": low_count,
        "errors": errors,
        "elapsed_seconds": elapsed,
    }
