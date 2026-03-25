"""Shadow repair pass for structurally weak enriched B2B reviews."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from . import b2b_enrichment as base_enrichment

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment_repair")


def _shadow_quarantine_reasons(row: dict[str, Any]) -> list[str]:
    source = str(row.get("source") or "").strip().lower()
    if source in {"stackoverflow", "github"}:
        return ["repair_shadowed_technical_source"]
    return []


async def _repair_single(pool, row: dict[str, Any], cfg, max_attempts: int) -> str:
    review_id = row["id"]
    baseline = base_enrichment._coerce_json_dict(row.get("enrichment"))
    if not baseline or not base_enrichment._has_structural_gap(baseline):
        await pool.execute(
            """
            UPDATE b2b_reviews
            SET enrichment_repair_status = 'shadowed',
                enrichment_repair_attempts = enrichment_repair_attempts + 1,
                enrichment_repaired_at = $2,
                enrichment_repair_applied_fields = '[]'::jsonb
            WHERE id = $1
            """,
            review_id,
            datetime.now(timezone.utc),
        )
        return "shadowed"

    repair_cfg = SimpleNamespace(
        enrichment_tier2_model=cfg.enrichment_repair_model,
        enrichment_max_tokens=cfg.enrichment_max_tokens,
    )
    try:
        repair_result, model_id = await asyncio.wait_for(
            base_enrichment._call_cloud_tier2(
                row,
                repair_cfg,
                cfg.review_truncate_length,
            ),
            timeout=120,
        )
    except Exception:
        logger.exception("Repair call failed for review %s", review_id)
        repair_result, model_id = None, None

    if repair_result is None:
        await pool.execute(
            """
            UPDATE b2b_reviews
            SET enrichment_repair_status = 'failed',
                enrichment_repair_attempts = enrichment_repair_attempts + 1,
                enrichment_repair_model = COALESCE($2, enrichment_repair_model),
                enrichment_repaired_at = $3
            WHERE id = $1
            """,
            review_id,
            model_id,
            datetime.now(timezone.utc),
        )
        return "failed"

    promoted, applied_fields = base_enrichment._apply_structural_repair(
        baseline,
        repair_result,
    )
    repaired_at = datetime.now(timezone.utc)
    if applied_fields and base_enrichment._validate_enrichment(promoted, row):
        await pool.execute(
            """
            UPDATE b2b_reviews
            SET enrichment_baseline = COALESCE(enrichment_baseline, enrichment),
                enrichment = $2::jsonb,
                enrichment_repair = $3::jsonb,
                enrichment_repair_status = 'promoted',
                enrichment_repair_attempts = enrichment_repair_attempts + 1,
                enrichment_repair_model = $4,
                enrichment_repaired_at = $5,
                enrichment_repair_applied_fields = $6::jsonb
            WHERE id = $1
            """,
            review_id,
            json.dumps(promoted),
            json.dumps(repair_result),
            model_id,
            repaired_at,
            json.dumps(applied_fields),
        )
        return "promoted"

    shadow_reasons = _shadow_quarantine_reasons(row)
    target_status = "quarantined" if shadow_reasons else row.get("enrichment_status") or "enriched"
    await pool.execute(
        """
        UPDATE b2b_reviews
        SET enrichment_repair = $2::jsonb,
            enrichment_repair_status = 'shadowed',
            enrichment_repair_attempts = enrichment_repair_attempts + 1,
            enrichment_repair_model = $3,
            enrichment_repaired_at = $4,
            enrichment_repair_applied_fields = $5::jsonb,
            enrichment_status = $6,
            low_fidelity = $7,
            low_fidelity_reasons = $8::jsonb,
            low_fidelity_detected_at = $9
        WHERE id = $1
        """,
        review_id,
        json.dumps(repair_result),
        model_id,
        repaired_at,
        json.dumps(applied_fields),
        target_status,
        bool(shadow_reasons),
        json.dumps(shadow_reasons),
        repaired_at if shadow_reasons else None,
    )
    return "shadowed"


async def _repair_rows(rows, cfg, pool, *, concurrency_override: int | None = None) -> dict[str, int]:
    max_attempts = cfg.enrichment_repair_max_attempts
    effective_concurrency = max(1, int(concurrency_override or cfg.enrichment_repair_concurrency))
    sem = asyncio.Semaphore(effective_concurrency)

    async def _bounded(row: dict[str, Any]) -> str:
        async with sem:
            return await _repair_single(pool, row, cfg, max_attempts)

    results = await asyncio.gather(*[_bounded(row) for row in rows], return_exceptions=True)
    counts = {"promoted": 0, "shadowed": 0, "failed": 0}
    for row, result in zip(rows, results):
        if isinstance(result, Exception):
            logger.error("Unexpected repair error for %s: %s", row["id"], result, exc_info=result)
            counts["failed"] += 1
            continue
        counts[result] = counts.get(result, 0) + 1
    return counts


async def _recover_orphaned_repairing(pool, max_attempts: int) -> int:
    result = await pool.execute(
        """
        UPDATE b2b_reviews
        SET enrichment_repair_attempts = enrichment_repair_attempts + 1,
            enrichment_repair_status = CASE
                WHEN enrichment_repair_attempts + 1 >= $1 THEN 'failed'
                ELSE NULL
            END
        WHERE enrichment_repair_status = 'repairing'
        """,
        max_attempts,
    )
    try:
        return int(str(result).split()[-1])
    except (TypeError, ValueError, IndexError):
        return 0


async def run(task: ScheduledTask) -> dict[str, Any]:
    cfg = settings.b2b_churn
    if not cfg.enabled:
        return {"_skip_synthesis": "B2B churn pipeline disabled"}
    if not cfg.enrichment_repair_enabled:
        return {"_skip_synthesis": "B2B enrichment repair disabled"}
    if not str(cfg.enrichment_repair_model or "").strip():
        return {"_skip_synthesis": "No B2B enrichment repair model configured"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    orphaned = await _recover_orphaned_repairing(pool, cfg.enrichment_repair_max_attempts)
    task_metadata = task.metadata if isinstance(task.metadata, dict) else {}
    max_batch = base_enrichment._coerce_int_override(
        task_metadata.get("enrichment_repair_max_per_batch"),
        cfg.enrichment_repair_max_per_batch,
        min_value=1,
        max_value=500,
    )
    max_rounds = base_enrichment._coerce_int_override(
        task_metadata.get("enrichment_repair_max_rounds_per_run"),
        cfg.enrichment_repair_max_rounds_per_run,
        min_value=1,
        max_value=100,
    )
    concurrency = base_enrichment._coerce_int_override(
        task_metadata.get("enrichment_repair_concurrency"),
        cfg.enrichment_repair_concurrency,
        min_value=1,
        max_value=100,
    )

    promoted = 0
    shadowed = 0
    failed = 0
    rounds = 0
    while rounds < max_rounds:
        rows = await pool.fetch(
            """
            WITH batch AS (
                SELECT id
                FROM b2b_reviews
                WHERE enrichment_status = 'enriched'
                  AND COALESCE(low_fidelity, false) = false
                  AND enrichment IS NOT NULL
                  AND enrichment_repair_attempts < $1
                  AND (enrichment_repair_status IS NULL OR enrichment_repair_status = 'failed')
                  AND (
                    COALESCE((enrichment->>'urgency_score')::numeric, 0) >= $3
                    OR COALESCE(enrichment->'churn_signals'->>'intent_to_leave', 'false') = 'true'
                    OR COALESCE(enrichment->'churn_signals'->>'actively_evaluating', 'false') = 'true'
                  )
                  AND (
                    COALESCE(enrichment->'buyer_authority'->>'role_type', 'unknown') = 'unknown'
                    OR COALESCE(enrichment->'timeline'->>'decision_timeline', 'unknown') = 'unknown'
                    OR COALESCE(enrichment->'contract_context'->>'contract_value_signal', 'unknown') = 'unknown'
                  )
                ORDER BY enriched_at DESC NULLS LAST, id
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            )
            UPDATE b2b_reviews r
            SET enrichment_repair_status = 'repairing'
            FROM batch
            WHERE r.id = batch.id
            RETURNING r.id, r.vendor_name, r.product_name, r.product_category,
                      r.source, r.raw_metadata,
                      r.rating, r.rating_max, r.summary, r.review_text, r.pros, r.cons,
                      r.reviewer_title, r.reviewer_company, r.company_size_raw,
                      r.reviewer_industry, r.content_type, r.enrichment,
                      r.enrichment_repair_attempts
            """,
            cfg.enrichment_repair_max_attempts,
            max_batch,
            cfg.enrichment_repair_min_urgency,
        )
        if not rows:
            break
        result = await _repair_rows(
            rows,
            cfg,
            pool,
            concurrency_override=concurrency,
        )
        promoted += result.get("promoted", 0)
        shadowed += result.get("shadowed", 0)
        failed += result.get("failed", 0)
        rounds += 1
        await asyncio.sleep(1)

    if rounds == 0:
        return {"_skip_synthesis": "No enriched reviews need repair"}
    return {
        "promoted": promoted,
        "shadowed": shadowed,
        "failed": failed,
        "rounds": rounds,
        "orphaned_recovered": orphaned,
        "_skip_synthesis": "B2B enrichment repair complete",
    }
