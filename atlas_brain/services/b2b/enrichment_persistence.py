from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ...config import settings

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment")


@dataclass(frozen=True)
class EnrichmentFinalizationDeps:
    compute_derived_fields: Any
    validate_enrichment: Any


@dataclass(frozen=True)
class EnrichmentPersistenceDeps:
    finalize_enrichment_for_persist: Any
    witness_metrics: Any
    detect_low_fidelity_reasons: Any
    is_no_signal_result: Any
    notify_high_urgency: Any
    increment_attempts: Any
    normalize_company_name: Any


def finalize_enrichment_for_persist(
    result: dict[str, Any],
    source_row: dict[str, Any],
    *,
    deps: EnrichmentFinalizationDeps,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(result, dict):
        return None, "invalid_payload"

    payload = json.loads(json.dumps(result))
    try:
        payload = deps.compute_derived_fields(payload, source_row)
    except Exception:
        logger.warning(
            "Evidence engine compute failed while finalizing enrichment for %s",
            source_row.get("id"),
            exc_info=True,
        )
        return None, "compute_failed"

    if not deps.validate_enrichment(payload, source_row):
        return None, "validation_failed"

    return payload, None


async def persist_enrichment_result(
    pool,
    row: dict[str, Any],
    result: dict[str, Any] | None,
    *,
    model_id: str,
    max_attempts: int,
    run_id: str | None,
    cache_usage: dict[str, int],
    deps: EnrichmentPersistenceDeps,
) -> bool | str:
    review_id = row["id"]
    cfg = settings.b2b_churn

    if result:
        result, finalize_error = deps.finalize_enrichment_for_persist(result, row)
        if finalize_error == "compute_failed":
            logger.warning(
                "Evidence engine compute failed for %s -- quarantining to prevent model-dependent output",
                review_id,
                exc_info=True,
            )
            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment_status = 'quarantined',
                    enrichment_attempts = enrichment_attempts + 1,
                    low_fidelity = true,
                    low_fidelity_reasons = $2::jsonb
                WHERE id = $1
                """,
                review_id,
                json.dumps(["evidence_engine_compute_failure"]),
            )
            from ...autonomous.visibility import record_quarantine

            await record_quarantine(
                pool,
                review_id=str(review_id),
                vendor_name=row.get("vendor_name"),
                source=row.get("source"),
                reason_code="evidence_engine_compute_failure",
                severity="error",
                actionable=True,
                summary=f"Evidence engine failed for {row.get('vendor_name')} review",
                run_id=run_id,
            )
            return "quarantined"
        if finalize_error == "validation_failed":
            logger.warning(
                "Enrichment validation failed for %s -- quarantining",
                review_id,
            )
            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment_status = 'quarantined',
                    enrichment_attempts = enrichment_attempts + 1,
                    low_fidelity = true,
                    low_fidelity_reasons = $2::jsonb
                WHERE id = $1
                """,
                review_id,
                json.dumps(["enrichment_validation_failed"]),
            )
            from ...autonomous.visibility import record_quarantine

            await record_quarantine(
                pool,
                review_id=str(review_id),
                vendor_name=row.get("vendor_name"),
                source=row.get("source"),
                reason_code="enrichment_validation_failed",
                severity="warning",
                actionable=True,
                summary=f"Validation failed for {row.get('vendor_name')} review",
                run_id=run_id,
            )
            return "quarantined"

    if result:
        st = result.get("sentiment_trajectory") or {}
        st_direction = st.get("direction") if isinstance(st, dict) else None
        st_tenure = st.get("tenure") if isinstance(st, dict) else None
        st_turning = st.get("turning_point") if isinstance(st, dict) else None
        witness_rows, witness_count = deps.witness_metrics(result)
        cache_usage["witness_rows"] += witness_rows
        cache_usage["witness_count"] += witness_count
        low_fidelity_reasons = (
            deps.detect_low_fidelity_reasons(row, result)
            if cfg.enrichment_low_fidelity_enabled
            else []
        )
        detected_at = datetime.now(timezone.utc)
        if not low_fidelity_reasons and deps.is_no_signal_result(result, row):
            target_status = "no_signal"
        else:
            target_status = "quarantined" if low_fidelity_reasons else "enriched"

        await pool.execute(
            """
            UPDATE b2b_reviews
            SET enrichment = $1,
                enrichment_status = $8,
                enrichment_attempts = enrichment_attempts + 1,
                enriched_at = $2,
                enrichment_model = $4,
                sentiment_direction = $5,
                sentiment_tenure = $6,
                sentiment_turning_point = $7,
                low_fidelity = $9,
                low_fidelity_reasons = $10::jsonb,
                low_fidelity_detected_at = $11
            WHERE id = $3
            """,
            json.dumps(result),
            detected_at,
            review_id,
            model_id,
            st_direction,
            st_tenure,
            st_turning if st_turning and st_turning != "null" else None,
            target_status,
            bool(low_fidelity_reasons),
            json.dumps(low_fidelity_reasons),
            detected_at if low_fidelity_reasons else None,
        )

        if low_fidelity_reasons:
            from ...autonomous.visibility import record_quarantine

            await record_quarantine(
                pool,
                review_id=str(review_id),
                vendor_name=row.get("vendor_name"),
                source=row.get("source"),
                reason_code=low_fidelity_reasons[0],
                severity="warning",
                summary=f"Low-fidelity: {', '.join(low_fidelity_reasons[:3])}",
                evidence={"reasons": low_fidelity_reasons, "source": row.get("source")},
                run_id=run_id,
            )

        try:
            urgency = result.get("urgency_score", 0)
            threshold = settings.b2b_churn.high_churn_urgency_threshold
            if urgency >= threshold:
                signals = result.get("churn_signals", {})
                await deps.notify_high_urgency(
                    vendor_name=row["vendor_name"],
                    reviewer_company=row.get("reviewer_company") or "",
                    urgency=urgency,
                    pain_category=result.get("pain_category", ""),
                    intent_to_leave=bool(signals.get("intent_to_leave")),
                )
        except Exception:
            logger.warning("ntfy notification failed for review %s, enrichment preserved", review_id)

        try:
            reviewer_context = result.get("reviewer_context") or {}
            extracted_company = (reviewer_context.get("company_name") or "").strip()
            if extracted_company and not (row.get("reviewer_company") or "").strip():
                extracted_company_norm = deps.normalize_company_name(extracted_company) or None
                await pool.execute(
                    "UPDATE b2b_reviews SET reviewer_company = $1, reviewer_company_norm = $2 WHERE id = $3",
                    extracted_company,
                    extracted_company_norm,
                    review_id,
                )
                cache_usage["secondary_write_hits"] += 1
        except Exception:
            logger.debug("Company name backfill failed for %s (non-fatal)", review_id)

        return "quarantined" if target_status == "quarantined" else True

    await deps.increment_attempts(pool, review_id, row["enrichment_attempts"], max_attempts)
    return False
