from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment")


@dataclass(frozen=True)
class EnrichmentTaskRunnerDeps:
    recover_orphaned_enriching: Any
    mark_exhausted_pending_failed: Any
    queue_version_upgrades: Any
    queue_model_upgrades: Any
    task_run_id: Any
    enrich_rows: Any
    fetch_review_funnel_audit: Any
    empty_exact_cache_usage: Any
    accumulate_exact_cache_usage: Any
    coerce_int_value: Any
    coerce_int_override: Any
    coerce_float_value: Any


async def run_enrichment_task(
    *,
    task: Any,
    cfg: Any,
    pool: Any,
    deps: EnrichmentTaskRunnerDeps,
) -> dict[str, Any]:
    orphaned = await deps.recover_orphaned_enriching(pool, cfg.enrichment_max_attempts)
    exhausted = await deps.mark_exhausted_pending_failed(pool, cfg.enrichment_max_attempts)
    requeued_parser = await deps.queue_version_upgrades(pool)
    requeued_model = await deps.queue_model_upgrades(pool, cfg)
    requeued = requeued_parser + requeued_model

    task_metadata = task.metadata if isinstance(task.metadata, dict) else {}
    default_max_batch = min(
        deps.coerce_int_value(getattr(cfg, "enrichment_max_per_batch", 10), 10),
        500,
    )
    max_batch = deps.coerce_int_override(
        task_metadata.get("enrichment_max_per_batch"),
        default_max_batch,
        min_value=1,
        max_value=500,
    )
    max_attempts = deps.coerce_int_value(getattr(cfg, "enrichment_max_attempts", 3), 3)
    default_max_rounds = max(
        1,
        deps.coerce_int_value(getattr(cfg, "enrichment_max_rounds_per_run", 1), 1),
    )
    max_rounds = deps.coerce_int_override(
        task_metadata.get("enrichment_max_rounds_per_run"),
        default_max_rounds,
        min_value=1,
        max_value=100,
    )
    effective_concurrency = deps.coerce_int_override(
        task_metadata.get("enrichment_concurrency"),
        max(1, deps.coerce_int_value(getattr(cfg, "enrichment_concurrency", 10), 10)),
        min_value=1,
        max_value=100,
    )
    inter_batch_delay = max(
        0.0,
        deps.coerce_float_value(
            task_metadata.get(
                "enrichment_inter_batch_delay_seconds",
                getattr(cfg, "enrichment_inter_batch_delay_seconds", 2.0),
            ),
            deps.coerce_float_value(getattr(cfg, "enrichment_inter_batch_delay_seconds", 2.0), 2.0),
        ),
    )
    priority_sources = [
        source.strip().lower()
        for source in str(cfg.enrichment_priority_sources or "").split(",")
        if source.strip()
    ]
    run_id = deps.task_run_id(task)

    total_enriched = 0
    total_failed = 0
    total_no_signal = 0
    total_quarantined = 0
    cache_usage = deps.empty_exact_cache_usage()
    batch_metrics = {
        "anthropic_batch_jobs": 0,
        "anthropic_batch_items_submitted": 0,
        "anthropic_batch_cache_prefiltered_items": 0,
        "anthropic_batch_fallback_single_call_items": 0,
        "anthropic_batch_completed_items": 0,
        "anthropic_batch_failed_items": 0,
        "anthropic_batch_rows_deferred": 0,
        "anthropic_batch_tier2_single_fallback_rows": 0,
    }
    rounds = 0

    while rounds < max_rounds:
        rows = await pool.fetch(
            """
            WITH batch AS (
                SELECT id
                FROM b2b_reviews
                WHERE enrichment_status = 'pending'
                  AND enrichment_attempts < $1
                ORDER BY CASE
                    WHEN source = ANY($3::text[]) THEN 0
                    ELSE 1
                END,
                imported_at DESC
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            )
            UPDATE b2b_reviews r
            SET enrichment_status = 'enriching'
            FROM batch
            WHERE r.id = batch.id
            RETURNING r.id, r.vendor_name, r.product_name, r.product_category,
                      r.source, r.raw_metadata,
                      r.rating, r.rating_max, r.summary, r.review_text, r.pros, r.cons,
                      r.reviewer_title, r.reviewer_company, r.company_size_raw,
                      r.reviewer_industry, r.enrichment_attempts, r.content_type
            """,
            max_attempts,
            max_batch,
            priority_sources,
        )

        if not rows:
            break

        result = await deps.enrich_rows(
            rows,
            cfg,
            pool,
            concurrency_override=effective_concurrency,
            run_id=run_id,
            task=task,
        )
        total_enriched += result.get("enriched", 0)
        batch_failed = result.get("failed", 0)
        total_failed += batch_failed
        total_no_signal += result.get("no_signal", 0)
        total_quarantined += result.get("quarantined", 0)
        deps.accumulate_exact_cache_usage(cache_usage, result)
        for key in batch_metrics:
            batch_metrics[key] += int(result.get(key, 0) or 0)
        rounds += 1

        if batch_failed > len(rows) * 0.5:
            logger.warning(
                "B2B enrichment: >50%% failures in batch (%d/%d), stopping loop",
                batch_failed,
                len(rows),
            )
            break

        if inter_batch_delay > 0:
            await asyncio.sleep(inter_batch_delay)

    if rounds == 0:
        return {"_skip_synthesis": "No B2B reviews to enrich"}

    secondary_write_breakdown = {
        "company_backfills": int(cache_usage.get("secondary_write_hits", 0) or 0),
        "orphaned_requeued": int(orphaned or 0),
        "exhausted_marked_failed": int(exhausted or 0),
        "version_upgrade_requeued": int(requeued or 0),
    }
    secondary_write_hits = sum(secondary_write_breakdown.values())
    result = {
        "enriched": total_enriched,
        "quarantined": total_quarantined,
        "failed": total_failed,
        "no_signal": total_no_signal,
        **cache_usage,
        "rounds": rounds,
        "orphaned_requeued": orphaned,
        "exhausted_marked_failed": exhausted,
        "witness_rows": int(cache_usage.get("witness_rows", 0) or 0),
        "witness_count": int(cache_usage.get("witness_count", 0) or 0),
        "reviews_processed": total_enriched + total_quarantined + total_failed + total_no_signal,
        "secondary_write_hits": secondary_write_hits,
        "secondary_write_breakdown": secondary_write_breakdown,
        **batch_metrics,
        "_skip_synthesis": "B2B enrichment complete",
    }
    result["funnel_audit"] = await deps.fetch_review_funnel_audit(
        pool,
        int(getattr(cfg, "intelligence_window_days", 30) or 30),
    )
    if requeued:
        result["version_upgrade_requeued"] = requeued

    from ...autonomous.visibility import emit_event, record_attempt

    total_processed = total_enriched + total_quarantined + total_failed + total_no_signal
    await record_attempt(
        pool,
        artifact_type="enrichment",
        artifact_id="batch",
        run_id=run_id,
        stage="enrichment",
        status="succeeded" if total_failed == 0 else "failed",
        score=total_enriched,
        blocker_count=total_failed,
        warning_count=total_quarantined,
        error_message=f"{total_failed} failed, {total_quarantined} quarantined" if total_failed else None,
    )
    if total_failed > 0 or total_quarantined > 0 or secondary_write_hits > 0:
        if total_failed > 0:
            reason_code = "enrichment_failures"
        elif total_quarantined > 0:
            reason_code = "enrichment_quarantines"
        else:
            reason_code = "enrichment_secondary_writes"
        await emit_event(
            pool,
            stage="extraction",
            event_type="enrichment_run_summary",
            entity_type="pipeline",
            entity_id="enrichment",
            summary=f"Enrichment: {total_enriched} enriched, {total_failed} failed, {total_quarantined} quarantined",
            severity="warning" if total_failed > 0 else "info",
            actionable=total_failed > 5,
            run_id=run_id,
            reason_code=reason_code,
            detail={
                "enriched": total_enriched,
                "failed": total_failed,
                "quarantined": total_quarantined,
                "no_signal": total_no_signal,
                "processed": total_processed,
                "witness_rows": int(cache_usage.get("witness_rows", 0) or 0),
                "witness_count": int(cache_usage.get("witness_count", 0) or 0),
                "exact_cache_hits": int(cache_usage.get("exact_cache_hits", 0) or 0),
                "generated": int(cache_usage.get("generated", 0) or 0),
                "secondary_write_hits": secondary_write_hits,
                "secondary_write_breakdown": secondary_write_breakdown,
            },
        )

    return result
