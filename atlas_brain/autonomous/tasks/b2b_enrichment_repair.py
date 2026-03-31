"""Shadow repair pass for structurally weak enriched B2B reviews."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...pipelines.llm import call_llm_with_skill, parse_json_response
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from . import b2b_enrichment as base_enrichment

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment_repair")

_REPAIR_JSON_SCHEMA: dict[str, Any] = {
    "title": "b2b_churn_repair_extraction",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "competitors_mentioned": {"type": "array"},
        "specific_complaints": {"type": "array"},
        "pricing_phrases": {"type": "array"},
        "recommendation_language": {"type": "array"},
        "feature_gaps": {"type": "array"},
        "event_mentions": {"type": "array"},
    },
}


def _shadow_quarantine_reasons(row: dict[str, Any]) -> list[str]:
    source = str(row.get("source") or "").strip().lower()
    if source in {"stackoverflow", "github"}:
        return ["repair_shadowed_technical_source"]
    return []


async def _repair_single(pool, row: dict[str, Any], cfg, max_attempts: int) -> str:
    review_id = row["id"]
    baseline = base_enrichment._coerce_json_dict(row.get("enrichment"))
    if not baseline or not base_enrichment._needs_field_repair(baseline, row):
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

    repair_model = str(cfg.enrichment_repair_model or "").strip()
    try:
        payload = _build_repair_payload(row, baseline, cfg.review_truncate_length)
        repair_result, model_id = await asyncio.wait_for(
            _call_repair_extractor(payload, repair_model, cfg),
            timeout=cfg.enrichment_full_extraction_timeout_seconds,
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

    promoted, applied_fields = base_enrichment._apply_field_repair(
        baseline,
        repair_result,
    )
    repaired_at = datetime.now(timezone.utc)
    if applied_fields:
        promoted = base_enrichment._compute_derived_fields(promoted, row)
    if applied_fields and base_enrichment._validate_enrichment(promoted, row):
        low_fidelity_reasons = (
            base_enrichment._detect_low_fidelity_reasons(row, promoted)
            if cfg.enrichment_low_fidelity_enabled
            else []
        )
        if not low_fidelity_reasons and base_enrichment._is_no_signal_result(promoted, row):
            target_status = "no_signal"
        else:
            target_status = "quarantined" if low_fidelity_reasons else "enriched"
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
                enrichment_repair_applied_fields = $6::jsonb,
                enrichment_status = $7,
                low_fidelity = $8,
                low_fidelity_reasons = $9::jsonb,
                low_fidelity_detected_at = $10
            WHERE id = $1
            """,
            review_id,
            json.dumps(promoted),
            json.dumps(repair_result),
            model_id,
            repaired_at,
            json.dumps(applied_fields),
            target_status,
            bool(low_fidelity_reasons),
            json.dumps(low_fidelity_reasons),
            repaired_at if low_fidelity_reasons else None,
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


def _build_repair_payload(row: dict[str, Any], baseline: dict[str, Any], review_truncate_length: int) -> dict[str, Any]:
    def _truncate(value: Any) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        return text[: int(review_truncate_length)] if review_truncate_length > 0 else text

    target_fields = base_enrichment._repair_target_fields(baseline, row)
    current_extraction = {
        field: baseline.get(field) or []
        for field in target_fields
    }
    return {
        "vendor_name": row.get("vendor_name"),
        "summary": _truncate(row.get("summary")),
        "review_text": _truncate(row.get("review_text")),
        "target_fields": target_fields,
        "current_extraction": current_extraction,
    }


async def _call_repair_extractor(payload: dict[str, Any], model_id: str, cfg) -> tuple[dict | None, str | None]:
    try:
        response = await asyncio.to_thread(
            call_llm_with_skill,
            "digest/b2b_churn_repair_extraction",
            json.dumps(payload, ensure_ascii=True),
            max_tokens=cfg.enrichment_repair_max_tokens,
            temperature=0.0,
            guided_json=_REPAIR_JSON_SCHEMA,
            response_format={"type": "json_object"},
            workload="openrouter",
            try_openrouter=True,
            openrouter_model=model_id,
        )
        if not response:
            return None, model_id
        parsed = parse_json_response(response, recover_truncated=True)
        if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
            return parsed, model_id
        return None, model_id
    except Exception:
        logger.exception("Repair extractor call failed")
        return None, None


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
        trusted_sources = list(base_enrichment._trusted_repair_sources())
        rows = await pool.fetch(
            """
            WITH batch AS (
                SELECT id
                FROM b2b_reviews
                WHERE enrichment_status IN ('enriched', 'no_signal')
                  AND COALESCE(low_fidelity, false) = false
                  AND enrichment IS NOT NULL
                  AND enrichment_repair_attempts < $1
                  AND (enrichment_repair_status IS NULL OR enrichment_repair_status = 'failed')
                  AND (
                    (
                      COALESCE(enrichment->>'pain_category', 'other') = 'other'
                      AND review_text ~* '(cancel|cancellation|billing dispute|refund denied|runaround|automatic renewal|auto renew|renewed without notice|charged|invoiced|price increase|overcharg)'
                    )
                    OR (
                      COALESCE(jsonb_array_length(enrichment->'competitors_mentioned'), 0) = 0
                      AND review_text ~* '(switched to|moved to|replaced with|evaluating|looking at|considering|alternative to|[[:<:]]vs[[:>:]])'
                    )
                    OR (
                      COALESCE(jsonb_array_length(enrichment->'pricing_phrases'), 0) = 0
                      AND review_text ~* '(billing|invoice|invoiced|charged|refund|renewal|price increase|cost increase|automatic renewal|auto renew|overcharg)'
                    )
                    OR (
                      COALESCE(enrichment->>'pain_category', 'other') NOT IN ('pricing', 'contract_lock_in')
                      AND (
                        review_text ~* '\\$\\s?\\d'
                        OR COALESCE(enrichment->'salience_flags', '[]'::jsonb) ? 'explicit_dollar'
                      )
                    )
                    OR (
                      COALESCE(jsonb_array_length(enrichment->'specific_complaints'), 0) = 0
                      AND review_text ~* '(cancel|cancellation|billing dispute|charged after cancellation|refund denied|runaround|automatic renewal|auto renew|renewed without notice)'
                    )
                    OR (
                      COALESCE(jsonb_array_length(enrichment->'event_mentions'), 0) = 0
                      AND COALESCE(enrichment->'timeline'->>'decision_timeline', 'unknown') = 'unknown'
                      AND review_text ~* '(renewal|contract end|contract expires|deadline|next quarter|q1|q2|q3|q4|30 days|60 days|90 days)'
                    )
                    OR (
                      enrichment_status = 'no_signal'
                      AND (
                        cardinality($3::text[]) = 0
                        OR lower(source) = ANY($3::text[])
                      )
                      AND review_text ~* '(cancel|cancellation|billing|invoice|charged|refund|automatic renewal|auto renew|switched to|moved to|considering|evaluating|replaced with)'
                    )
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
            trusted_sources,
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


# ---------------------------------------------------------------------------
# Quarantine retry: re-derive reviews that failed evidence engine compute
# ---------------------------------------------------------------------------

_RETRYABLE_QUARANTINE_REASONS = frozenset({
    "evidence_engine_compute_failure",
})


async def retry_quarantined_reviews(
    pool,
    *,
    limit: int = 100,
) -> dict[str, int]:
    """Re-attempt derivation on quarantined reviews where the root cause
    was an evidence engine failure (likely a bug that has since been fixed).

    Skips the LLM call entirely -- the tier-1 extraction is already in
    the JSONB.  Only re-runs ``_compute_derived_fields()`` and validation.

    Returns counts of recovered, still_failed, skipped reviews.
    """
    rows = await pool.fetch(
        """
        SELECT id, enrichment, source, rating, rating_max,
               review_text, summary, pros, cons,
               vendor_name, reviewer_company, raw_metadata,
               low_fidelity_reasons
        FROM b2b_reviews
        WHERE enrichment_status = 'quarantined'
          AND enrichment IS NOT NULL
          AND (enrichment->>'enrichment_schema_version')::int >= 1
        ORDER BY created_at DESC
        LIMIT $1
        """,
        limit,
    )

    recovered = 0
    still_failed = 0
    skipped = 0

    for row in rows:
        reasons = row.get("low_fidelity_reasons") or []
        if isinstance(reasons, str):
            try:
                reasons = json.loads(reasons)
            except Exception:
                reasons = []

        retryable = any(r in _RETRYABLE_QUARANTINE_REASONS for r in reasons)
        if not retryable:
            skipped += 1
            continue

        enrichment = base_enrichment._coerce_json_dict(row.get("enrichment"))
        if not enrichment:
            skipped += 1
            continue

        try:
            enrichment = base_enrichment._compute_derived_fields(enrichment, dict(row))
            valid = base_enrichment._validate_enrichment(enrichment)
            if not valid:
                still_failed += 1
                continue

            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment = $2::jsonb,
                    enrichment_status = 'enriched',
                    enriched_at = NOW(),
                    low_fidelity = false,
                    low_fidelity_reasons = '[]'::jsonb
                WHERE id = $1
                """,
                row["id"],
                json.dumps(enrichment, default=str),
            )
            recovered += 1
            logger.info("Quarantine retry: recovered %s", row["id"])

        except Exception:
            still_failed += 1
            logger.debug("Quarantine retry: still failing for %s", row["id"], exc_info=True)

    logger.info(
        "Quarantine retry: recovered=%d, still_failed=%d, skipped=%d (of %d)",
        recovered, still_failed, skipped, len(rows),
    )
    return {"recovered": recovered, "still_failed": still_failed, "skipped": skipped}
