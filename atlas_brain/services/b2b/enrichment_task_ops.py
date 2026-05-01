from __future__ import annotations

from typing import Any


def coerce_int_override(
    raw_value: Any,
    default_value: int,
    *,
    min_value: int,
    max_value: int,
    coerce_int_value: Any,
) -> int:
    default_coerced = coerce_int_value(default_value, min_value)
    coerced = coerce_int_value(raw_value, default_coerced)
    return max(min_value, min(max_value, coerced))


def _parse_update_count(result: Any) -> int:
    try:
        return int(str(result).split()[-1])
    except (TypeError, ValueError, IndexError):
        return 0


async def recover_orphaned_enriching(pool: Any, max_attempts: int, *, logger: Any) -> int:
    result = await pool.execute(
        """
        UPDATE b2b_reviews
        SET enrichment_attempts = enrichment_attempts + 1,
            enrichment_status = CASE
                WHEN enrichment_attempts + 1 >= $1 THEN 'failed'
                ELSE 'pending'
            END
        WHERE enrichment_status = 'enriching'
        """,
        max_attempts,
    )
    count = _parse_update_count(result)
    if count:
        logger.warning("Recovered %d orphaned B2B enrichment rows", count)
    return count


async def mark_exhausted_pending_failed(pool: Any, max_attempts: int, *, logger: Any) -> int:
    result = await pool.execute(
        """
        UPDATE b2b_reviews
        SET enrichment_status = 'failed'
        WHERE enrichment_status = 'pending'
          AND enrichment_attempts >= $1
        """,
        max_attempts,
    )
    count = _parse_update_count(result)
    if count:
        logger.warning("Marked %d exhausted pending rows as failed", count)
    return count


async def queue_version_upgrades(
    pool: Any,
    *,
    enabled: bool,
    get_all_parsers: Any,
    logger: Any,
) -> int:
    if not enabled:
        logger.info("Parser-version auto requeue disabled; skipping version-upgrade scan")
        return 0
    try:
        parsers = get_all_parsers()
        if not parsers:
            return 0

        total_requeued = 0
        for source_name, parser in parsers.items():
            current_version = getattr(parser, "version", None)
            if not current_version:
                continue

            count = await pool.fetchval(
                """
                WITH updated AS (
                    UPDATE b2b_reviews
                    SET enrichment_status = 'pending',
                        enrichment_attempts = 0,
                        requeue_reason = 'parser_upgrade',
                        low_fidelity = false,
                        low_fidelity_reasons = '[]'::jsonb,
                        low_fidelity_detected_at = NULL,
                        enrichment_repair = NULL,
                        enrichment_repair_status = NULL,
                        enrichment_repair_attempts = 0,
                        enrichment_repair_model = NULL,
                        enrichment_repaired_at = NULL,
                        enrichment_repair_applied_fields = '[]'::jsonb
                    WHERE source = $1
                      AND parser_version IS NOT NULL
                      AND parser_version != $2
                      AND enrichment_status IN ('enriched', 'no_signal', 'quarantined')
                    RETURNING 1
                )
                SELECT count(*) FROM updated
                """,
                source_name,
                current_version,
            )
            if count and count > 0:
                logger.info(
                    "Re-queued %d %s reviews for re-enrichment (parser %s -> %s)",
                    count, source_name, "old", current_version,
                )
                total_requeued += count

        return total_requeued
    except Exception:
        logger.debug("Version upgrade check skipped", exc_info=True)
        return 0


async def queue_model_upgrades(pool: Any, cfg: Any, *, logger: Any) -> int:
    if not cfg.enrichment_auto_requeue_model_upgrades:
        return 0

    current_sig = str(cfg.enrichment_tier1_model or "").strip()
    if not current_sig:
        return 0

    try:
        count = await pool.fetchval(
            """
            WITH updated AS (
                UPDATE b2b_reviews
                SET enrichment_status = 'pending',
                    enrichment_attempts = 0,
                    requeue_reason = 'enrichment_model_outdated',
                    low_fidelity = false,
                    low_fidelity_reasons = '[]'::jsonb,
                    low_fidelity_detected_at = NULL,
                    enrichment_repair = NULL,
                    enrichment_repair_status = NULL,
                    enrichment_repair_attempts = 0,
                    enrichment_repair_model = NULL,
                    enrichment_repaired_at = NULL,
                    enrichment_repair_applied_fields = '[]'::jsonb
                WHERE enrichment_status IN ('enriched', 'no_signal', 'quarantined')
                  AND (enrichment_model IS NULL OR enrichment_model != $1)
                RETURNING 1
            )
            SELECT count(*) FROM updated
            """,
            current_sig,
        )
        if count and count > 0:
            logger.info(
                "Re-queued %d reviews for re-enrichment (model drift -> %s)",
                count, current_sig,
            )
        return count
    except Exception:
        logger.debug("Model upgrade check skipped", exc_info=True)
        return 0
