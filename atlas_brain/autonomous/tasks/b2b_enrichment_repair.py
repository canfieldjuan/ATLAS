"""Shadow repair pass for structurally weak enriched B2B reviews."""

import asyncio
import inspect
import json
import logging
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...pipelines.llm import call_llm_with_skill, get_pipeline_llm, parse_json_response
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from . import b2b_enrichment as base_enrichment
from ._execution_progress import task_run_id as _task_run_id

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

_GENERIC_PAIN_BUCKETS = ("other", "general_dissatisfaction", "overall_dissatisfaction")
_ADJUDICATION_PREFIX = "adjudication:"
_COMPETITOR_PRESSURE_PATTERNS = (
    "cancel", "cancellation", "refund", "billing dispute", "renewal", "price increase",
    "overcharged", "not worth", "switch", "switched to", "moved to", "replaced with",
    "evaluating", "considering", "alternative", "frustrated", "pain", "issue", "problem",
)
_DISPLACEMENT_COMPETITOR_PATTERNS = (
    "switched to", "moved to", "replaced with", "migrating to", "migration to",
    "evaluating", "looking at", "considering", "shortlisting", "shortlisted",
    "poc with", "proof of concept with",
)
_HARD_GAP_SQL = """
(
  (enrichment->'evidence_spans' IS NULL OR jsonb_typeof(enrichment->'evidence_spans') != 'array'
    OR CASE WHEN jsonb_typeof(enrichment->'evidence_spans') = 'array' THEN jsonb_array_length(enrichment->'evidence_spans') ELSE 0 END = 0)
  OR COALESCE(enrichment->>'replacement_mode', '') = ''
  OR COALESCE(enrichment->>'operating_model_shift', '') = ''
  OR COALESCE(enrichment->>'productivity_delta_claim', '') = ''
  OR COALESCE(enrichment->>'org_pressure_type', '') = ''
  OR COALESCE(enrichment->>'evidence_map_hash', '') = ''
)
"""


def _normalize_test_vendors(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
        return [value for value in values if value]
    if isinstance(raw, (list, tuple, set)):
        normalized: list[str] = []
        for item in raw:
            value = str(item or "").strip()
            if value:
                normalized.append(value)
        return normalized
    value = str(raw).strip()
    return [value] if value else []


def _normalized_spans(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        span for span in (result.get("evidence_spans") or [])
        if isinstance(span, dict)
    ]


def _has_hard_gap_payload(result: dict[str, Any] | None) -> bool:
    payload = base_enrichment._coerce_json_dict(result)
    if not payload:
        return True
    spans = payload.get("evidence_spans")
    return (
        not isinstance(spans, list)
        or not spans
        or not str(payload.get("replacement_mode") or "").strip()
        or not str(payload.get("operating_model_shift") or "").strip()
        or not str(payload.get("productivity_delta_claim") or "").strip()
        or not str(payload.get("org_pressure_type") or "").strip()
        or not str(payload.get("evidence_map_hash") or "").strip()
    )


def _has_strong_competitor_signal(competitors: list[dict[str, Any]]) -> bool:
    return any(
        str(comp.get("evidence_type") or "").strip().lower() in {"explicit_switch", "active_evaluation"}
        or str(comp.get("displacement_confidence") or "").strip().lower() in {"high", "medium"}
        or str(comp.get("reason_category") or "").strip()
        or str(comp.get("reason_detail") or "").strip()
        or str(comp.get("reason") or "").strip()
        for comp in competitors
    )


def _strategic_adjudication_reasons(result: dict[str, Any], source_row: dict[str, Any]) -> list[str]:
    review_blob = base_enrichment._repair_text_blob(source_row)
    if not review_blob.strip():
        return []

    spans = _normalized_spans(result)
    salience_flags = {
        str(flag or "").strip().lower()
        for flag in (result.get("salience_flags") or [])
        if str(flag or "").strip()
    }
    churn = base_enrichment._coerce_json_dict(result.get("churn_signals"))
    reviewer = base_enrichment._coerce_json_dict(result.get("reviewer_context"))
    timeline = base_enrichment._coerce_json_dict(result.get("timeline"))
    content_type = str(source_row.get("content_type") or "").strip().lower()
    reviewer_title = str(source_row.get("reviewer_title") or "").strip()
    competitors = [
        comp for comp in (result.get("competitors_mentioned") or [])
        if isinstance(comp, dict) and str(comp.get("name") or "").strip()
    ]
    replacement_mode = str(result.get("replacement_mode") or "").strip().lower()
    structured_churn = (
        bool(churn.get("intent_to_leave"))
        or bool(churn.get("actively_evaluating"))
        or bool(churn.get("migration_in_progress"))
        or bool(churn.get("contract_renewal_mentioned"))
    )
    pressure_signal = (
        structured_churn
        or bool(result.get("specific_complaints"))
        or bool(result.get("pricing_phrases"))
        or bool(result.get("feature_gaps"))
        or base_enrichment._contains_any(review_blob, _COMPETITOR_PRESSURE_PATTERNS)
    )

    pricing_span = any(str(span.get("signal_type") or "").strip().lower() == "pricing_backlash" for span in spans)
    strong_competitor_signal = _has_strong_competitor_signal(competitors)
    displacement_framed = strong_competitor_signal or any(
        str(span.get("signal_type") or "").strip().lower() == "competitor_pressure"
        or str(span.get("competitor") or "").strip()
        for span in spans
    ) or replacement_mode == "competitor_switch"
    named_company = (
        str(source_row.get("reviewer_company") or "").strip()
        or str(reviewer.get("company_name") or "").strip()
    )
    discussion_noise = (
        content_type == "community_discussion"
        and not structured_churn
        and not reviewer_title
        and not named_company
        and not strong_competitor_signal
    )
    named_account_evidence = bool(named_company) and (
        "named_account" in salience_flags
        or any(
            str(span.get("company_name") or "").strip()
            or "named_org" in {
                str(flag or "").strip().lower()
                for flag in (span.get("flags") or [])
                if str(flag or "").strip()
            }
            for span in spans
        )
    )
    timing_anchor = any(
        str(span.get("time_anchor") or "").strip()
        or "deadline" in {
            str(flag or "").strip().lower()
            for flag in (span.get("flags") or [])
            if str(flag or "").strip()
        }
        for span in spans
    ) or any(
        str(value or "").strip()
        for value in (
            timeline.get("evaluation_deadline"),
            timeline.get("contract_end"),
            churn.get("renewal_timing"),
        )
    )

    reasons: list[str] = []
    if (
        (base_enrichment._REPAIR_CURRENCY_RE.search(review_blob) or "explicit_dollar" in salience_flags)
        and not pricing_span
    ):
        reasons.append("money_without_pricing_span")
    if (
        pressure_signal
        and (base_enrichment._contains_any(review_blob, _DISPLACEMENT_COMPETITOR_PATTERNS) or competitors)
        and not displacement_framed
        and not discussion_noise
    ):
        reasons.append("competitor_without_displacement_framing")
    if named_company and not named_account_evidence:
        reasons.append("named_company_without_named_account_evidence")
    if (
        (
            bool(churn.get("contract_renewal_mentioned"))
            or base_enrichment._contains_any(review_blob, base_enrichment._REPAIR_TIMELINE_PATTERNS)
        )
        and not timing_anchor
    ):
        reasons.append("timeline_language_without_timing_anchor")
    if (
        base_enrichment._contains_any(
            review_blob,
            base_enrichment._REPAIR_CATEGORY_SHIFT_PATTERNS,
        )
        and replacement_mode == "none"
    ):
        reasons.append("workflow_language_without_replacement_mode")
    return base_enrichment._dedupe_reason_codes(reasons)


def _strategic_target_fields(result: dict[str, Any], source_row: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    for reason in _strategic_adjudication_reasons(result, source_row):
        if reason == "money_without_pricing_span":
            for field in ("pricing_phrases", "specific_complaints"):
                if field not in targets:
                    targets.append(field)
        elif reason == "competitor_without_displacement_framing":
            for field in ("competitors_mentioned", "specific_complaints", "event_mentions"):
                if field not in targets:
                    targets.append(field)
        elif reason == "timeline_language_without_timing_anchor":
            if "event_mentions" not in targets:
                targets.append("event_mentions")
    return targets


def _repair_target_fields(baseline: dict[str, Any], source_row: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    for field in base_enrichment._repair_target_fields(baseline, source_row):
        if field not in targets:
            targets.append(field)
    for field in _strategic_target_fields(baseline, source_row):
        if field not in targets:
            targets.append(field)
    return targets


def _adjudication_markers(reasons: list[str]) -> list[str]:
    return [f"{_ADJUDICATION_PREFIX}{reason}" for reason in reasons if str(reason or "").strip()]


def _shadow_quarantine_reasons(row: dict[str, Any]) -> list[str]:
    source = str(row.get("source") or "").strip().lower()
    if source in {"stackoverflow", "github"}:
        return ["repair_shadowed_technical_source"]
    return []


async def _persist_shadow_result(
    pool,
    *,
    review_id: Any,
    row: dict[str, Any],
    repair_result: dict[str, Any] | None,
    model_id: str | None,
    applied_fields: list[str],
    repaired_at: datetime,
    persisted_enrichment: dict[str, Any] | None = None,
    shadow_reasons: list[str] | None = None,
) -> str:
    combined_shadow_reasons = base_enrichment._dedupe_reason_codes(
        list(shadow_reasons or []) + _shadow_quarantine_reasons(row)
    )
    hard_gap_payload = persisted_enrichment if persisted_enrichment is not None else row.get("enrichment")
    target_status = (
        "quarantined"
        if combined_shadow_reasons or _has_hard_gap_payload(hard_gap_payload)
        else row.get("enrichment_status") or "enriched"
    )
    if persisted_enrichment is not None:
        await pool.execute(
            """
            UPDATE b2b_reviews
            SET enrichment_baseline = COALESCE(enrichment_baseline, enrichment),
                enrichment = $2::jsonb,
                enrichment_repair = $3::jsonb,
                enrichment_repair_status = 'shadowed',
                enrichment_repair_attempts = enrichment_repair_attempts + 1,
                enrichment_repair_model = COALESCE($4, enrichment_repair_model),
                enrichment_repaired_at = $5,
                enrichment_repair_applied_fields = $6::jsonb,
                enrichment_status = $7,
                low_fidelity = $8,
                low_fidelity_reasons = $9::jsonb,
                low_fidelity_detected_at = $10
            WHERE id = $1
            """,
            review_id,
            json.dumps(persisted_enrichment),
            json.dumps(repair_result or {}),
            model_id,
            repaired_at,
            json.dumps(applied_fields),
            target_status,
            False,
            json.dumps(combined_shadow_reasons),
            repaired_at if combined_shadow_reasons else None,
        )
    else:
        await pool.execute(
            """
            UPDATE b2b_reviews
            SET enrichment_repair = $2::jsonb,
                enrichment_repair_status = 'shadowed',
                enrichment_repair_attempts = enrichment_repair_attempts + 1,
                enrichment_repair_model = COALESCE($3, enrichment_repair_model),
                enrichment_repaired_at = $4,
                enrichment_repair_applied_fields = $5::jsonb,
                enrichment_status = $6,
                low_fidelity = $7,
                low_fidelity_reasons = $8::jsonb,
                low_fidelity_detected_at = $9
            WHERE id = $1
            """,
            review_id,
            json.dumps(repair_result or {}),
            model_id,
            repaired_at,
            json.dumps(applied_fields),
            target_status,
            False,
            json.dumps(combined_shadow_reasons),
            repaired_at if combined_shadow_reasons else None,
        )
    return "shadowed"


async def _repair_single(
    pool,
    row: dict[str, Any],
    cfg,
    max_attempts: int,
    run_id: str | None = None,
    usage_out: dict[str, int] | None = None,
) -> str:
    review_id = row["id"]
    baseline = base_enrichment._coerce_json_dict(row.get("enrichment"))
    strategic_reasons = _strategic_adjudication_reasons(baseline, row) if baseline else []
    target_fields = _repair_target_fields(baseline, row) if baseline else []
    usage = {
        "exact_cache_hits": 0,
        "generated": 0,
        "witness_rows": 0,
        "witness_count": 0,
    }

    def _finish(status: str) -> str:
        if usage_out is not None:
            usage_out.clear()
            usage_out.update(usage)
        return status

    if not baseline or not target_fields:
        status = await _persist_shadow_result(
            pool,
            review_id=review_id,
            row=row,
            repair_result=None,
            model_id=None,
            applied_fields=_adjudication_markers(strategic_reasons),
            repaired_at=datetime.now(timezone.utc),
        )
        return _finish(status)

    repair_model = str(cfg.enrichment_repair_model or "").strip()
    repair_cache_hit = False
    try:
        payload = _build_repair_payload(row, baseline, cfg.review_truncate_length)
        repair_result, model_id, repair_cache_hit = base_enrichment._unpack_stage_result(await asyncio.wait_for(
            _call_repair_extractor(
                payload,
                repair_model,
                cfg,
                include_cache_hit=True,
                trace_metadata={
                    "run_id": run_id,
                    "vendor_name": str(row.get("vendor_name") or ""),
                    "review_id": str(review_id),
                    "source": str(row.get("source") or ""),
                    "stage": "repair_extraction",
                },
            ),
            timeout=cfg.enrichment_full_extraction_timeout_seconds,
        ))
    except Exception:
        logger.exception("Repair call failed for review %s", review_id)
        repair_result, model_id = None, None
        repair_cache_hit = False

    if repair_cache_hit:
        usage["exact_cache_hits"] += 1
    elif model_id is not None:
        usage["generated"] += 1

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
        return _finish("failed")

    promoted, applied_fields = base_enrichment._apply_field_repair(
        baseline,
        repair_result,
    )
    adjudication_markers = _adjudication_markers(strategic_reasons)
    persisted_applied_fields = list(applied_fields)
    for marker in adjudication_markers:
        if marker not in persisted_applied_fields:
            persisted_applied_fields.append(marker)
    repaired_at = datetime.now(timezone.utc)
    if applied_fields:
        promoted = base_enrichment._compute_derived_fields(promoted, row)
    if applied_fields and base_enrichment._validate_enrichment(promoted, row):
        _, baseline_witness_count = base_enrichment._witness_metrics(baseline)
        promoted_witness_rows, promoted_witness_count = base_enrichment._witness_metrics(promoted)
        witness_count_delta = max(promoted_witness_count - baseline_witness_count, 0)
        if witness_count_delta > 0:
            usage["witness_rows"] += 1 if promoted_witness_rows > 0 else 0
            usage["witness_count"] += witness_count_delta
        unresolved_strategic_reasons = _strategic_adjudication_reasons(promoted, row)
        low_fidelity_reasons = (
            base_enrichment._detect_low_fidelity_reasons(row, promoted)
            if getattr(cfg, "enrichment_low_fidelity_enabled", False)
            else []
        )
        if unresolved_strategic_reasons:
            status = await _persist_shadow_result(
                pool,
                review_id=review_id,
                row=row,
                repair_result=repair_result,
                model_id=model_id,
                applied_fields=persisted_applied_fields,
                repaired_at=repaired_at,
                persisted_enrichment=promoted,
                shadow_reasons=unresolved_strategic_reasons,
            )
            return _finish(status)
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
            json.dumps(persisted_applied_fields),
            target_status,
            bool(low_fidelity_reasons),
            json.dumps(low_fidelity_reasons),
            repaired_at if low_fidelity_reasons else None,
        )
        return _finish("promoted")

    status = await _persist_shadow_result(
        pool,
        review_id=review_id,
        row=row,
        repair_result=repair_result,
        model_id=model_id,
        applied_fields=persisted_applied_fields,
        repaired_at=repaired_at,
    )
    return _finish(status)


async def _repair_rows(
    rows,
    cfg,
    pool,
    *,
    concurrency_override: int | None = None,
    run_id: str | None = None,
) -> dict[str, int]:
    max_attempts = cfg.enrichment_repair_max_attempts
    effective_concurrency = max(1, int(concurrency_override or cfg.enrichment_repair_concurrency))
    sem = asyncio.Semaphore(effective_concurrency)
    repair_single_params = inspect.signature(_repair_single).parameters
    supports_run_id = "run_id" in repair_single_params

    async def _bounded(row: dict[str, Any]) -> dict[str, int | str]:
        async with sem:
            usage = {
                "exact_cache_hits": 0,
                "generated": 0,
                "witness_rows": 0,
                "witness_count": 0,
            }
            kwargs: dict[str, Any] = {"usage_out": usage}
            if supports_run_id:
                kwargs["run_id"] = run_id
            status = await _repair_single(pool, row, cfg, max_attempts, **kwargs)
            return {
                "status": status,
                **usage,
            }

    results = await asyncio.gather(*[_bounded(row) for row in rows], return_exceptions=True)
    counts = {
        "promoted": 0,
        "shadowed": 0,
        "failed": 0,
        "exact_cache_hits": 0,
        "generated": 0,
        "witness_rows": 0,
        "witness_count": 0,
    }
    for row, result in zip(rows, results):
        if isinstance(result, Exception):
            logger.error("Unexpected repair error for %s: %s", row["id"], result, exc_info=result)
            counts["failed"] += 1
            continue
        status = str(result.get("status") or "failed")
        counts[status] = counts.get(status, 0) + 1
        counts["exact_cache_hits"] += int(result.get("exact_cache_hits", 0) or 0)
        counts["generated"] += int(result.get("generated", 0) or 0)
        counts["witness_rows"] += int(result.get("witness_rows", 0) or 0)
        counts["witness_count"] += int(result.get("witness_count", 0) or 0)
    return counts


async def _demote_stale_no_signal_rows(pool, *, limit: int) -> int:
    rows = await pool.fetch(
        """
        SELECT id, vendor_name, product_name, product_category, source, raw_metadata,
               rating, rating_max, summary, review_text, pros, cons,
               reviewer_title, reviewer_company, company_size_raw,
               reviewer_industry, content_type, enrichment
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND COALESCE(low_fidelity, false) = false
          AND COALESCE(enrichment->>'pain_category', '') = ANY($1::text[])
          AND COALESCE(jsonb_array_length(enrichment->'competitors_mentioned'), 0) = 0
          AND COALESCE(jsonb_array_length(enrichment->'specific_complaints'), 0) = 0
          AND COALESCE(jsonb_array_length(enrichment->'quotable_phrases'), 0) = 0
          AND COALESCE(jsonb_array_length(enrichment->'pricing_phrases'), 0) = 0
          AND COALESCE(jsonb_array_length(enrichment->'recommendation_language'), 0) = 0
          AND COALESCE(jsonb_array_length(enrichment->'feature_gaps'), 0) = 0
          AND COALESCE(jsonb_array_length(enrichment->'event_mentions'), 0) = 0
        ORDER BY imported_at DESC NULLS LAST, id
        LIMIT $2
        """,
        list(_GENERIC_PAIN_BUCKETS),
        limit,
    )

    demoted = 0
    for row in rows:
        baseline = base_enrichment._coerce_json_dict(row.get("enrichment"))
        if not baseline or not base_enrichment._is_no_signal_result(baseline, dict(row)):
            continue
        await pool.execute(
            """
            UPDATE b2b_reviews
            SET enrichment_status = 'no_signal',
                enrichment_repair_status = 'promoted',
                enrichment_repair_attempts = enrichment_repair_attempts + 1,
                enrichment_repaired_at = $2,
                enrichment_repair_applied_fields = '["status:no_signal_cleanup"]'::jsonb
            WHERE id = $1
            """,
            row["id"],
            datetime.now(timezone.utc),
        )
        demoted += 1
    return demoted


async def _quarantine_shadowed_hard_gap_rows(pool, *, limit: int) -> int:
    result = await pool.execute(
        f"""
        WITH target AS (
            SELECT id
            FROM b2b_reviews
            WHERE enrichment_status = 'enriched'
              AND enrichment_repair_status = 'shadowed'
              AND {_HARD_GAP_SQL}
            ORDER BY enriched_at DESC NULLS LAST, imported_at DESC NULLS LAST, id
            LIMIT $1
        )
        UPDATE b2b_reviews AS r
        SET enrichment_status = 'quarantined',
            enrichment_repaired_at = $2,
            enrichment_repair_applied_fields =
              COALESCE(r.enrichment_repair_applied_fields, '[]'::jsonb)
              || '["status:quarantined_hard_gap_shadow"]'::jsonb
        FROM target
        WHERE r.id = target.id
        """,
        limit,
        datetime.now(timezone.utc),
    )
    try:
        return int(str(result).split()[-1])
    except (TypeError, ValueError, IndexError):
        return 0


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

    target_fields = _repair_target_fields(baseline, row)
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
        "strategic_adjudication_reasons": _strategic_adjudication_reasons(baseline, row),
    }


async def _call_repair_extractor(
    payload: dict[str, Any],
    model_id: str,
    cfg,
    *,
    include_cache_hit: bool = False,
    trace_metadata: dict[str, Any] | None = None,
) -> tuple[dict | None, str | None] | tuple[dict | None, str | None, bool]:
    from ...services.b2b.cache_runner import (
        lookup_b2b_exact_stage_text,
        prepare_b2b_exact_skill_stage_request,
        store_b2b_exact_stage_text,
    )
    from ...services.b2b.llm_exact_cache import CacheUnavailable, llm_identity

    cache_stage_id = "b2b_enrichment_repair.extraction"
    provider_name = ""
    resolved_model = ""
    request: Any | None = None
    max_tokens = int(
        getattr(
            cfg,
            "enrichment_repair_max_tokens",
            getattr(cfg, "enrichment_max_tokens", 2048),
        )
    )

    try:
        resolved_llm = get_pipeline_llm(
            workload="openrouter",
            try_openrouter=True,
            openrouter_model=model_id,
        )
        provider_name, resolved_model = llm_identity(resolved_llm)
        if provider_name and resolved_model:
            try:
                request, _ = prepare_b2b_exact_skill_stage_request(
                    cache_stage_id,
                    skill_name="digest/b2b_churn_repair_extraction",
                    payload=json.dumps(payload, ensure_ascii=True),
                    provider=provider_name,
                    model=resolved_model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    guided_json=_REPAIR_JSON_SCHEMA,
                    response_format={"type": "json_object"},
                    extra={"requested_model": model_id},
                )
            except CacheUnavailable:
                request = None

        if request is not None:
            cached = await lookup_b2b_exact_stage_text(request)
            if cached is not None:
                parsed = parse_json_response(
                    cached["response_text"],
                    recover_truncated=True,
                )
                if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
                    return base_enrichment._pack_stage_result(
                        parsed,
                        str(cached.get("model") or model_id),
                        True,
                        include_cache_hit=include_cache_hit,
                    )

        response = await asyncio.to_thread(
            call_llm_with_skill,
            "digest/b2b_churn_repair_extraction",
            json.dumps(payload, ensure_ascii=True),
            max_tokens=max_tokens,
            temperature=0.0,
            guided_json=_REPAIR_JSON_SCHEMA,
            response_format={"type": "json_object"},
            workload="openrouter",
            try_openrouter=True,
            openrouter_model=model_id,
            span_name="task.b2b_enrichment_repair.extraction",
            trace_metadata=trace_metadata,
        )
        if not response:
            return base_enrichment._pack_stage_result(
                None,
                model_id,
                False,
                include_cache_hit=include_cache_hit,
            )
        parsed = parse_json_response(response, recover_truncated=True)
        if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
            if request is not None:
                await store_b2b_exact_stage_text(
                    request,
                    response_text=response,
                    metadata={"requested_model": model_id},
                )
            return base_enrichment._pack_stage_result(
                parsed,
                resolved_model or model_id,
                False,
                include_cache_hit=include_cache_hit,
            )
        return base_enrichment._pack_stage_result(
            None,
            model_id,
            False,
            include_cache_hit=include_cache_hit,
        )
    except Exception:
        logger.exception("Repair extractor call failed")
        return base_enrichment._pack_stage_result(
            None,
            None,
            False,
            include_cache_hit=include_cache_hit,
        )


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
    run_id = _task_run_id(task)
    scoped_vendors = _normalize_test_vendors(task_metadata.get("test_vendors") or task_metadata.get("vendor_names"))
    stale_no_signal_demoted = await _demote_stale_no_signal_rows(
        pool,
        limit=max_batch,
    )
    shadowed_hard_gap_quarantined = await _quarantine_shadowed_hard_gap_rows(
        pool,
        limit=max_batch,
    )

    promoted = 0
    shadowed = 0
    failed = 0
    exact_cache_hits = 0
    generated = 0
    witness_rows = 0
    witness_count = 0
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
                  AND (
                    enrichment_repair_status IS NULL
                    OR enrichment_repair_status = 'failed'
                    OR enrichment_repair_status = 'promoted'
                    OR (
                      enrichment_repair_status = 'shadowed'
                      AND enrichment_status = 'quarantined'
                      AND jsonb_path_exists(
                        COALESCE(enrichment_repair_applied_fields, '[]'::jsonb),
                        '$[*] ? (@ like_regex "^adjudication:")'
                      )
                    )
                  )
                  AND (
                    cardinality($4::text[]) = 0
                    OR lower(vendor_name) = ANY($4::text[])
                  )
                  AND (
                    (
                      COALESCE(enrichment->>'pain_category', 'overall_dissatisfaction') IN ('other', 'general_dissatisfaction', 'overall_dissatisfaction')
                      AND review_text ~* '(cancel|cancellation|billing dispute|refund denied|runaround|automatic renewal|auto renew|renewed without notice|charged|invoiced|price increase|overcharg)'
                    )
                    OR (
                      COALESCE(jsonb_array_length(enrichment->'competitors_mentioned'), 0) = 0
                      AND review_text ~* '(switched to|moved to|replaced with|migrating to|migration to|evaluating|looking at|considering|shortlisting|shortlisted|poc with|proof of concept with)'
                    )
                    OR (
                      COALESCE(jsonb_array_length(enrichment->'pricing_phrases'), 0) = 0
                      AND review_text ~* '(billing|invoice|invoiced|charged|refund|renewal|price increase|cost increase|automatic renewal|auto renew|overcharg)'
                    )
                    OR (
                      COALESCE(enrichment->>'pain_category', 'overall_dissatisfaction') NOT IN ('pricing', 'contract_lock_in')
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
                    OR (
                      (review_text ~* '\\$\\s?\\d' OR COALESCE(enrichment->'salience_flags', '[]'::jsonb) ? 'explicit_dollar')
                      AND NOT jsonb_path_exists(COALESCE(enrichment->'evidence_spans', '[]'::jsonb), '$[*] ? (@.signal_type == "pricing_backlash")')
                    )
                    OR (
                      (
                        review_text ~* '(switched to|moved to|replaced with|migrating to|migration to|evaluating|looking at|considering|shortlisting|shortlisted|poc with|proof of concept with)'
                        OR COALESCE(jsonb_array_length(enrichment->'competitors_mentioned'), 0) > 0
                      )
                      AND (
                        COALESCE((enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
                        OR COALESCE((enrichment->'churn_signals'->>'actively_evaluating')::boolean, false)
                        OR COALESCE((enrichment->'churn_signals'->>'migration_in_progress')::boolean, false)
                        OR COALESCE((enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean, false)
                        OR COALESCE(jsonb_array_length(enrichment->'specific_complaints'), 0) > 0
                        OR COALESCE(jsonb_array_length(enrichment->'pricing_phrases'), 0) > 0
                        OR COALESCE(jsonb_array_length(enrichment->'feature_gaps'), 0) > 0
                        OR review_text ~* '(cancel|cancellation|refund|billing dispute|renewal|price increase|overcharg|not worth|switch|switched to|moved to|replaced with|evaluating|considering|alternative|frustrated|pain|issue|problem)'
                      )
                      AND NOT (
                        content_type = 'community_discussion'
                        AND NOT (
                          COALESCE((enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
                          OR COALESCE((enrichment->'churn_signals'->>'actively_evaluating')::boolean, false)
                          OR COALESCE((enrichment->'churn_signals'->>'migration_in_progress')::boolean, false)
                          OR COALESCE((enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean, false)
                        )
                        AND COALESCE(NULLIF(reviewer_title, ''), NULLIF(reviewer_company, ''), NULLIF(enrichment->'reviewer_context'->>'company_name', '')) IS NULL
                        AND NOT jsonb_path_exists(
                          COALESCE(enrichment->'competitors_mentioned', '[]'::jsonb),
                          '$[*] ? (@.evidence_type == "explicit_switch" || @.evidence_type == "active_evaluation" || @.displacement_confidence == "high" || @.displacement_confidence == "medium" || @.reason != null || @.reason_category != null || @.reason_detail != null)'
                        )
                      )
                      AND NOT (
                        jsonb_path_exists(COALESCE(enrichment->'competitors_mentioned', '[]'::jsonb), '$[*] ? (@.evidence_type == "explicit_switch" || @.evidence_type == "active_evaluation" || @.displacement_confidence == "high" || @.displacement_confidence == "medium")')
                        OR jsonb_path_exists(COALESCE(enrichment->'evidence_spans', '[]'::jsonb), '$[*] ? (@.signal_type == "competitor_pressure")')
                        OR lower(COALESCE(enrichment->>'replacement_mode', 'none')) = 'competitor_switch'
                      )
                    )
                    OR (
                      COALESCE(NULLIF(reviewer_company, ''), NULLIF(enrichment->'reviewer_context'->>'company_name', '')) IS NOT NULL
                      AND NOT (
                        COALESCE(enrichment->'salience_flags', '[]'::jsonb) ? 'named_account'
                        OR jsonb_path_exists(COALESCE(enrichment->'evidence_spans', '[]'::jsonb), '$[*] ? (@.company_name != null && @.company_name != "")')
                      )
                    )
                    OR (
                      (
                        COALESCE((enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean, false)
                        OR review_text ~* '(renewal|contract end|contract expires|deadline|next quarter|q1|q2|q3|q4|30 days|60 days|90 days)'
                      )
                      AND NOT jsonb_path_exists(COALESCE(enrichment->'evidence_spans', '[]'::jsonb), '$[*] ? ((@.time_anchor != null && @.time_anchor != "") || @.flags[*] == "deadline")')
                    )
                    OR (
                      review_text ~* '(async|docs|documentation|notion|confluence|bundle|workspace|microsoft 365|google workspace|internal tool|homegrown|home-grown|custom tool)'
                      AND lower(COALESCE(enrichment->>'replacement_mode', 'none')) = 'none'
                    )
                  )
                ORDER BY
                  CASE
                    WHEN enrichment_repair_status IS NULL THEN 0
                    WHEN enrichment_repair_status = 'failed' THEN 1
                    WHEN enrichment_repair_status = 'promoted' THEN 2
                    WHEN enrichment_repair_status = 'shadowed' THEN 3
                    ELSE 4
                  END,
                  CASE
                    WHEN enrichment_status = 'enriched' THEN 0
                    WHEN enrichment_status = 'no_signal' THEN 1
                    ELSE 2
                  END,
                  enriched_at DESC NULLS LAST,
                  imported_at DESC NULLS LAST,
                  id
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
            [vendor.lower() for vendor in scoped_vendors],
        )
        if not rows:
            break
        result = await _repair_rows(
            rows,
            cfg,
            pool,
            concurrency_override=concurrency,
            run_id=run_id,
        )
        promoted += result.get("promoted", 0)
        shadowed += result.get("shadowed", 0)
        failed += result.get("failed", 0)
        exact_cache_hits += result.get("exact_cache_hits", 0)
        generated += result.get("generated", 0)
        witness_rows += result.get("witness_rows", 0)
        witness_count += result.get("witness_count", 0)
        rounds += 1
        await asyncio.sleep(1)

    if rounds == 0 and stale_no_signal_demoted == 0 and shadowed_hard_gap_quarantined == 0:
        return {"_skip_synthesis": "No enriched reviews need repair"}
    secondary_write_breakdown = {
        "stale_no_signal_demoted": int(stale_no_signal_demoted or 0),
        "shadowed_hard_gap_quarantined": int(shadowed_hard_gap_quarantined or 0),
        "orphaned_recovered": int(orphaned or 0),
    }
    secondary_write_hits = sum(secondary_write_breakdown.values())
    result = {
        "promoted": promoted,
        "shadowed": shadowed,
        "failed": failed,
        "exact_cache_hits": exact_cache_hits,
        "generated": generated,
        "witness_rows": witness_rows,
        "witness_count": witness_count,
        "reviews_processed": promoted + shadowed + failed,
        "rounds": rounds,
        "stale_no_signal_demoted": stale_no_signal_demoted,
        "shadowed_hard_gap_quarantined": shadowed_hard_gap_quarantined,
        "orphaned_recovered": orphaned,
        "secondary_write_hits": secondary_write_hits,
        "secondary_write_breakdown": secondary_write_breakdown,
        "scoped_vendors": scoped_vendors,
        "_skip_synthesis": "B2B enrichment repair complete",
    }
    from ..visibility import emit_event, record_attempt

    warning_count = int(shadowed) + int(shadowed_hard_gap_quarantined)
    error_message = None
    if failed or shadowed or shadowed_hard_gap_quarantined:
        error_message = (
            f"{failed} failed, {shadowed} shadowed, "
            f"{shadowed_hard_gap_quarantined} hard-gap quarantined"
        )
    await record_attempt(
        pool,
        artifact_type="enrichment_repair",
        artifact_id="batch",
        run_id=run_id,
        stage="repair",
        status="succeeded" if failed == 0 else "failed",
        score=promoted,
        blocker_count=failed,
        warning_count=warning_count,
        error_message=error_message,
    )
    await emit_event(
        pool,
        stage="enrichment_repair",
        event_type="repair_run_summary",
        entity_type="pipeline",
        entity_id="enrichment_repair",
        artifact_type="enrichment_repair",
        summary=(
            f"Repair: {promoted} promoted, {shadowed} shadowed, "
            f"{failed} failed, {shadowed_hard_gap_quarantined} hard-gap quarantined"
        ),
        severity="warning" if failed > 0 or shadowed_hard_gap_quarantined > 0 else "info",
        actionable=failed > 0 or shadowed_hard_gap_quarantined > 0 or shadowed > 5,
        run_id=run_id,
        reason_code=(
            "enrichment_repair_failures"
            if failed > 0
            else "enrichment_repair_quarantines"
            if shadowed_hard_gap_quarantined > 0
            else "enrichment_repair_shadowed"
            if shadowed > 0
            else "enrichment_repair_secondary_writes"
            if secondary_write_hits > 0
            else "enrichment_repair_completed"
        ),
        detail={
            "promoted": promoted,
            "shadowed": shadowed,
            "failed": failed,
            "exact_cache_hits": exact_cache_hits,
            "generated": generated,
            "witness_rows": witness_rows,
            "witness_count": witness_count,
            "reviews_processed": promoted + shadowed + failed,
            "rounds": rounds,
            "stale_no_signal_demoted": stale_no_signal_demoted,
            "shadowed_hard_gap_quarantined": shadowed_hard_gap_quarantined,
            "orphaned_recovered": orphaned,
            "secondary_write_hits": secondary_write_hits,
            "secondary_write_breakdown": secondary_write_breakdown,
            "scoped_vendors": scoped_vendors,
        },
        update_review_state=False,
    )
    return result


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
            enrichment, finalize_error = base_enrichment._finalize_enrichment_for_persist(
                enrichment,
                dict(row),
            )
            if not enrichment or finalize_error:
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
