"""B2B reasoning synthesis v2: scored compression, source traceability, validation.

Runs after b2b_churn_intelligence (which populates pools) and before
b2b_battle_cards / b2b_churn_reports (which consume the synthesis).

V2 changes over v1:
- Scored pool compression replaces naive top-10 truncation
- Source IDs (_sid) on every item and aggregate for traceability
- Post-LLM validation blocks bad output from persisting
- Wedge types from the wedge registry (not the old archetype enum)

V1 rows remain in DB as automatic fallback -- consumers pick latest by
created_at DESC, so v2 wins when present.

This task still prompts in a battle-card-oriented shape, but it now also
decomposes the response into reusable reasoning contracts:
- vendor_core_reasoning
- displacement_reasoning
- category_reasoning
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import date, datetime, timezone
from typing import Any
from uuid import UUID

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ._b2b_witnesses import compute_witness_hash
from ._execution_progress import task_run_id as _task_execution_run_id

logger = logging.getLogger("atlas.autonomous.tasks.b2b_reasoning_synthesis")

_SCHEMA_VERSION = "v2"
_PACKET_SCHEMA_VERSION = "witness_packet_v1"

_QUALITY_STATUS_PASS = "pass"
_QUALITY_STATUS_WEAK = "weak"
_QUALITY_STATUS_REJECTED = "rejected"

_CORE_SECTIONS = (
    "causal_narrative", "segment_playbook", "timing_intelligence",
    "migration_proof", "competitive_reframes",
    "category_reasoning", "account_reasoning",
)


def _evaluate_synthesis_quality(synthesis: dict) -> tuple[str, list[str]]:
    """Evaluate synthesis quality beyond schema validation.

    Returns (quality_status, quality_reasons).
    - "pass": synthesis is strong enough for downstream consumers
    - "weak": synthesis persists but reuse decisions should re-run
    - "rejected": synthesis should NOT be persisted
    """
    reasons: list[str] = []
    contracts = synthesis.get("reasoning_contracts") or {}

    # Collect confidence levels across all sections
    section_confidences: dict[str, str] = {}
    for section_name in _CORE_SECTIONS:
        sec = contracts.get(section_name)
        if sec is None:
            # Check nested under vendor_core_reasoning / displacement_reasoning
            for parent in ("vendor_core_reasoning", "displacement_reasoning"):
                parent_dict = contracts.get(parent)
                if isinstance(parent_dict, dict):
                    sec = parent_dict.get(section_name)
                    if sec is not None:
                        break
        if not isinstance(sec, dict):
            reasons.append(f"missing_section:{section_name}")
            section_confidences[section_name] = "missing"
            continue
        conf = str(sec.get("confidence") or "").strip().lower()
        section_confidences[section_name] = conf or "missing"
        if not conf:
            reasons.append(f"no_confidence:{section_name}")

    # Check if ALL sections are insufficient or missing
    meaningful = [
        s for s, c in section_confidences.items()
        if c in ("high", "medium", "low")
    ]
    insufficient_or_missing = [
        s for s, c in section_confidences.items()
        if c in ("insufficient", "missing", "")
    ]

    if not meaningful:
        reasons.append("all_sections_insufficient_or_missing")
        return _QUALITY_STATUS_REJECTED, reasons

    # Check if causal_narrative (the core wedge) is insufficient
    causal_conf = section_confidences.get("causal_narrative", "missing")
    if causal_conf in ("insufficient", "missing"):
        reasons.append("causal_narrative_insufficient")
        return _QUALITY_STATUS_REJECTED, reasons

    # Check witness pack
    packet = synthesis.get("packet_artifacts") or {}
    witness_pack = packet.get("witness_pack") or []
    if not witness_pack:
        reasons.append("empty_witness_pack")

    # Weak if more than half the sections are insufficient
    if len(insufficient_or_missing) > len(_CORE_SECTIONS) // 2:
        reasons.append(f"majority_sections_weak:{len(insufficient_or_missing)}/{len(_CORE_SECTIONS)}")
        return _QUALITY_STATUS_WEAK, reasons

    # Lean mode: evidence was stripped before LLM call
    lean = synthesis.get("_lean_mode")
    if isinstance(lean, dict) and lean.get("omitted"):
        reasons.append(f"lean_mode_omissions:{','.join(lean['omitted'])}")

    if reasons:
        return _QUALITY_STATUS_WEAK, reasons

    return _QUALITY_STATUS_PASS, []


def _approx_token_count(text: str) -> int:
    """Approximate token count without relying on a model-specific tokenizer."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _approx_prompt_input_tokens(prompt: str, payload: str) -> int:
    """Approximate total prompt tokens for a system + user JSON call."""
    return _approx_token_count(prompt) + _approx_token_count(payload) + 16


def _trace_reasoning_result(
    span_name: str,
    *,
    llm: Any,
    messages: list[Any],
    result: dict[str, Any],
    metadata: dict[str, Any],
    duration_ms: float,
) -> None:
    from ...pipelines.llm import _trace_cache_metrics, trace_llm_call

    usage = result.get("usage", {}) if isinstance(result, dict) else {}
    trace_meta = result.get("_trace_meta", {}) if isinstance(result, dict) else {}
    cached_tokens, cache_write_tokens, billable_input_tokens = _trace_cache_metrics(usage, trace_meta)
    response_text = (result.get("response", "") if isinstance(result, dict) else "") or ""
    trace_llm_call(
        span_name,
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        billable_input_tokens=billable_input_tokens,
        model=getattr(llm, "model", ""),
        provider=getattr(llm, "name", ""),
        duration_ms=duration_ms,
        metadata=metadata,
        input_data={"messages": [{"role": m.role, "content": (m.content or "")[:500]} for m in messages]},
        output_data={"response": response_text[:2000]} if response_text else None,
        api_endpoint=trace_meta.get("api_endpoint"),
        provider_request_id=trace_meta.get("provider_request_id"),
        ttft_ms=trace_meta.get("ttft_ms"),
        inference_time_ms=trace_meta.get("inference_time_ms"),
        queue_time_ms=trace_meta.get("queue_time_ms"),
    )


_POOL_HASH_IGNORED_TOP_LEVEL_FIELDS = frozenset({"as_of_date"})
_EVIDENCE_VAULT_HASH_IGNORED_PATHS = {
    ("metric_snapshot", "snapshot_date"),
    ("provenance", "enrichment_window_start"),
    ("provenance", "enrichment_window_end"),
}


def _normalize_pool_hash_value(
    layer_name: str,
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, dict):
        normalized: dict[Any, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            child_path = path + (key_text,)
            if not path and key_text in _POOL_HASH_IGNORED_TOP_LEVEL_FIELDS:
                continue
            if (
                layer_name == "evidence_vault"
                and child_path in _EVIDENCE_VAULT_HASH_IGNORED_PATHS
            ):
                continue
            normalized[key] = _normalize_pool_hash_value(
                layer_name,
                child,
                path=child_path,
            )
        return normalized
    if isinstance(value, list):
        return [
            _normalize_pool_hash_value(layer_name, child, path=path)
            for child in value
        ]
    return value


def _normalize_pool_hash_layers(layers: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for layer_name, layer_value in layers.items():
        normalized[layer_name] = _normalize_pool_hash_value(
            str(layer_name),
            layer_value,
        )
    return normalized


def _compute_pool_hash_legacy(layers: dict[str, Any]) -> str:
    raw = json.dumps(layers, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _compute_pool_hash(layers: dict[str, Any]) -> str:
    """Deterministic hash of all pool layer data for a vendor."""
    normalized_layers = _normalize_pool_hash_layers(layers)
    return _compute_pool_hash_legacy(normalized_layers)


def _coerce_as_of_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _row_has_packet_artifacts(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    if "has_witness_pack" not in row:
        return True
    return bool(row.get("has_witness_pack"))


def _row_has_reference_ids(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    if "has_metric_refs" not in row and "has_witness_refs" not in row:
        return True
    return bool(row.get("has_metric_refs")) and bool(row.get("has_witness_refs"))


def _prior_synthesis_is_weak(latest_row: dict[str, Any] | None) -> bool:
    """Check if prior synthesis was flagged as weak or rejected."""
    if latest_row is None:
        return False
    synthesis = latest_row.get("synthesis")
    if isinstance(synthesis, str):
        try:
            synthesis = json.loads(synthesis)
        except (json.JSONDecodeError, TypeError):
            return True  # unparseable = weak
    if not isinstance(synthesis, dict):
        return True
    quality = synthesis.get("_quality_status", "")
    if quality in (_QUALITY_STATUS_WEAK, _QUALITY_STATUS_REJECTED):
        return True
    # Also check for legacy rows with no quality_status but
    # structurally weak content (all insufficient, empty witness pack)
    qs, _ = _evaluate_synthesis_quality(synthesis)
    return qs in (_QUALITY_STATUS_WEAK, _QUALITY_STATUS_REJECTED)


def _classify_vendor_reasoning_decision(
    *,
    vendor_name: str,
    today: date,
    evidence_hash: str,
    latest_row: dict[str, Any] | None,
    force: bool,
    max_stale_days: int,
    rerun_if_missing_packet_artifacts: bool,
    rerun_if_missing_reference_ids: bool,
    hash_matches_prior: bool | None = None,
) -> dict[str, Any]:
    prior_hash = str((latest_row or {}).get("evidence_hash") or "")
    prior_as_of_date = _coerce_as_of_date((latest_row or {}).get("as_of_date"))
    prior_row_age_days = (
        max(0, (today - prior_as_of_date).days)
        if prior_as_of_date is not None
        else None
    )
    if latest_row is not None and hash_matches_prior is None:
        hash_matches_prior = prior_hash == evidence_hash
    hash_changed = bool(latest_row) and not bool(hash_matches_prior)
    missing_packet_artifacts = bool(latest_row) and not _row_has_packet_artifacts(latest_row)
    missing_reference_ids = bool(latest_row) and not _row_has_reference_ids(latest_row)

    if force:
        reason = "forced"
        should_reason = True
    elif latest_row is None:
        reason = "missing_prior_row"
        should_reason = True
    elif hash_changed:
        reason = "hash_changed"
        should_reason = True
    elif rerun_if_missing_packet_artifacts and missing_packet_artifacts:
        reason = "missing_packet_artifacts"
        should_reason = True
    elif rerun_if_missing_reference_ids and missing_reference_ids:
        reason = "missing_reference_ids"
        should_reason = True
    elif _prior_synthesis_is_weak(latest_row):
        reason = "prior_quality_weak"
        should_reason = True
    elif (
        prior_row_age_days is not None
        and prior_row_age_days > max_stale_days
    ):
        reason = "stale_reused"
        should_reason = False
    else:
        reason = "hash_reuse"
        should_reason = False

    return {
        "vendor_name": vendor_name,
        "reason": reason,
        "should_reason": should_reason,
        "hash_changed": hash_changed,
        "prior_as_of_date": prior_as_of_date,
        "prior_row_age_days": prior_row_age_days,
        "missing_packet_artifacts": missing_packet_artifacts,
        "missing_reference_ids": missing_reference_ids,
    }


def _validation_feedback(vresult: Any, limit: int) -> list[str]:
    """Convert validator errors into compact retry feedback lines."""
    feedback: list[str] = []
    for err in list(getattr(vresult, "errors", []))[:limit]:
        path = getattr(err, "path", "") or "$"
        message = getattr(err, "message", "") or "validation failed"
        feedback.append(f"{path}: {message}")
    return feedback


def _retry_guidance(vresult: Any) -> list[str]:
    """Add targeted correction hints for recurring synthesis failures."""
    findings = list(getattr(vresult, "errors", []) or []) + list(
        getattr(vresult, "warnings", []) or [],
    )
    seen_codes = {
        str(getattr(finding, "code", "") or "").strip()
        for finding in findings
        if str(getattr(finding, "code", "") or "").strip()
    }
    guidance: list[str] = []
    if "missing_section" in seen_codes:
        guidance.append(
            "Include reasoning_contracts.displacement_reasoning.migration_proof "
            "as a complete object. Do not omit migration_proof even when evidence "
            "is thin; keep the section present and use cautious confidence/data_gaps.",
        )
    if (
        "missing_category_reasoning" in seen_codes
        or "empty_category_reasoning" in seen_codes
    ):
        guidance.append(
            "category_reasoning must not leave both market_regime and narrative "
            "empty. If evidence is mixed, provide a cautious narrative explaining "
            "the uncertainty instead of returning blanks.",
        )
    if (
        "missing_citations" in seen_codes
        or "unknown_packet_citation" in seen_codes
    ):
        guidance.append(
            "Every competitive_reframes.reframes[*].citations and other cited "
            "sections must use only _sid values already present in the input "
            "packet. Do not invent witness ids or leave reframe citations empty.",
        )
    if "proof_point_requires_numeric_support_source" in seen_codes:
        guidance.append(
            "competitive_reframes.reframes[*].proof_point.source_id must use "
            "numeric support or shortlist source_ids from section_packets. "
            "Do not use witness ids in proof_point.source_id; keep witness ids "
            "in citations only.",
        )
    return guidance


def _validation_rows(vresult: Any) -> list[dict[str, Any]]:
    """Normalize validator findings into row payloads."""
    rows: list[dict[str, Any]] = []
    for passed_group, findings in (
        (False, list(getattr(vresult, "errors", []) or [])),
        (False, list(getattr(vresult, "warnings", []) or [])),
    ):
        for finding in findings:
            rows.append(
                {
                    "rule_code": getattr(finding, "code", "") or "unknown",
                    "severity": getattr(finding, "severity", "") or "warning",
                    "passed": passed_group,
                    "summary": getattr(finding, "message", "") or "validation finding",
                    "field_path": getattr(finding, "path", None),
                    "detail": {
                        "path": getattr(finding, "path", None),
                        "message": getattr(finding, "message", ""),
                    },
                }
            )
    return rows


async def _record_validation_attempt(
    pool,
    *,
    vendor_name: str,
    run_id: str | None,
    as_of_date: date,
    analysis_window_days: int,
    attempt_no: int,
    vresult: Any,
    feedback_limit: int,
    attempt_tokens: int = 0,
    escalation_window_hours: int = 24,
    repeat_rule_threshold: int = 3,
    cost_min_retries: int = 2,
    cost_tokens_threshold: int = 80000,
    emit_retry_event: bool = False,
) -> None:
    from ..visibility import (
        emit_event,
        record_attempt,
        record_synthesis_validation_results,
    )

    blocking_issues = _validation_feedback(vresult, feedback_limit)
    warning_messages = [
        getattr(warning, "message", "") or "validation warning"
        for warning in list(getattr(vresult, "warnings", []) or [])
    ]
    await record_attempt(
        pool,
        artifact_type="reasoning_synthesis",
        artifact_id=vendor_name,
        run_id=run_id,
        attempt_no=attempt_no,
        stage="validation",
        status="rejected",
        blocker_count=len(list(getattr(vresult, "errors", []) or [])),
        warning_count=len(list(getattr(vresult, "warnings", []) or [])),
        blocking_issues=blocking_issues[:5],
        warnings=warning_messages[:5],
        feedback_summary=vresult.summary(),
        failure_step="validation",
        error_message=vresult.summary()[:200],
    )
    await record_synthesis_validation_results(
        pool,
        vendor_name=vendor_name,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
        schema_version=_SCHEMA_VERSION,
        run_id=run_id,
        attempt_no=attempt_no,
        results=_validation_rows(vresult),
    )
    if emit_retry_event:
        rule_codes = sorted({
            str(getattr(err, "code", "") or "").strip()
            for err in list(getattr(vresult, "errors", []) or [])
            if str(getattr(err, "code", "") or "").strip()
        })
        await emit_event(
            pool,
            stage="synthesis",
            event_type="validation_retry_rejected",
            entity_type="vendor",
            entity_id=vendor_name,
            summary=f"Synthesis retry rejected for {vendor_name}: {vresult.summary()[:100]}",
            severity="warning",
            actionable=False,
            artifact_type="reasoning_synthesis",
            run_id=run_id,
            reason_code="validation_retry_rejected",
            detail={
                "attempt_no": attempt_no,
                "tokens_used": int(attempt_tokens or 0),
                "rule_codes": rule_codes,
                "errors": [
                    {
                        "path": getattr(err, "path", None),
                        "code": getattr(err, "code", None),
                        "message": getattr(err, "message", None),
                    }
                    for err in list(getattr(vresult, "errors", []) or [])[:5]
                ],
            },
            update_review_state=False,
        )
        fetchval = getattr(pool, "fetchval", None)
        fetchrow = getattr(pool, "fetchrow", None)
        if callable(fetchval):
            for rule_code in rule_codes:
                try:
                    retry_count = await fetchval(
                        """
                        SELECT COUNT(*)
                        FROM pipeline_visibility_events
                        WHERE stage = 'synthesis'
                          AND event_type = 'validation_retry_rejected'
                          AND entity_type = 'vendor'
                          AND entity_id = $1
                          AND occurred_at >= NOW() - make_interval(hours => $2)
                          AND detail->'rule_codes' ? $3
                        """,
                        vendor_name,
                        escalation_window_hours,
                        rule_code,
                    )
                except Exception:
                    logger.debug(
                        "Failed to count repeated retry events for %s/%s",
                        vendor_name,
                        rule_code,
                        exc_info=True,
                    )
                    continue
                if int(retry_count or 0) >= repeat_rule_threshold:
                    await emit_event(
                        pool,
                        stage="synthesis",
                        event_type="validation_retry_escalated",
                        entity_type="vendor",
                        entity_id=vendor_name,
                        summary=(
                            f"Recovered validation retries repeated for {vendor_name}: "
                            f"{rule_code} hit {int(retry_count)} times in "
                            f"{escalation_window_hours}h"
                        ),
                        severity="warning",
                        actionable=True,
                        artifact_type="reasoning_synthesis",
                        run_id=run_id,
                        reason_code="repeated_validation_retry",
                        rule_code=rule_code,
                        detail={
                            "attempt_no": attempt_no,
                            "rule_code": rule_code,
                            "retry_count": int(retry_count or 0),
                            "window_hours": escalation_window_hours,
                            "threshold": repeat_rule_threshold,
                        },
                    )
        if callable(fetchrow):
            try:
                cost_row = await fetchrow(
                    """
                    SELECT
                        COUNT(*) AS retry_count,
                        COALESCE(
                            SUM(COALESCE(NULLIF(detail->>'tokens_used', ''), '0')::int),
                            0
                        ) AS retry_tokens
                    FROM pipeline_visibility_events
                    WHERE stage = 'synthesis'
                      AND event_type = 'validation_retry_rejected'
                      AND entity_type = 'vendor'
                      AND entity_id = $1
                      AND occurred_at >= NOW() - make_interval(hours => $2)
                    """,
                    vendor_name,
                    escalation_window_hours,
                )
            except Exception:
                logger.debug(
                    "Failed to compute retry token escalation for %s",
                    vendor_name,
                    exc_info=True,
                )
            else:
                cost_data = dict(cost_row) if cost_row else {}
                retry_count = int(cost_data.get("retry_count", 0) or 0)
                retry_tokens = int(cost_data.get("retry_tokens", 0) or 0)
                if (
                    retry_count >= cost_min_retries
                    and retry_tokens >= cost_tokens_threshold
                ):
                    await emit_event(
                        pool,
                        stage="synthesis",
                        event_type="validation_retry_escalated",
                        entity_type="vendor",
                        entity_id=vendor_name,
                        summary=(
                            f"Recovered validation retries costly for {vendor_name}: "
                            f"{retry_tokens} tokens across {retry_count} retries in "
                            f"{escalation_window_hours}h"
                        ),
                        severity="warning",
                        actionable=True,
                        artifact_type="reasoning_synthesis",
                        run_id=run_id,
                        reason_code="costly_validation_retry",
                        detail={
                            "attempt_no": attempt_no,
                            "retry_count": retry_count,
                            "retry_tokens": retry_tokens,
                            "window_hours": escalation_window_hours,
                            "min_retries": cost_min_retries,
                            "tokens_threshold": cost_tokens_threshold,
                        },
                    )


def _task_run_id(task: ScheduledTask | Any) -> str | None:
    """Return a stable run identifier for scheduled, manual, and test invocations."""
    return _task_execution_run_id(task)


def _metadata_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",")]
    result: list[str] = []
    seen: set[str] = set()
    for item in value or []:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _metadata_vendor_pairs(value: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in value or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        vendor_a = str(item[0] or "").strip()
        vendor_b = str(item[1] or "").strip()
        if not vendor_a or not vendor_b or vendor_a.lower() == vendor_b.lower():
            continue
        normalized = tuple(sorted((vendor_a.lower(), vendor_b.lower())))
        if normalized in seen:
            continue
        seen.add(normalized)
        pairs.append((vendor_a, vendor_b))
    return pairs


def _competitive_scope_metadata(task: ScheduledTask | Any) -> dict[str, Any]:
    metadata = getattr(task, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    if str(metadata.get("scope_type") or "").strip() != "competitive_set":
        return {}
    return {
        "scope_id": str(metadata.get("scope_id") or "").strip(),
        "scope_vendor_names": _metadata_text_list(metadata.get("scope_vendor_names")),
        "scope_pairwise_pairs": _metadata_vendor_pairs(metadata.get("scope_pairwise_pairs")),
        "scope_category_names": _metadata_text_list(metadata.get("scope_category_names")),
        "scope_asymmetry_pairs": _metadata_vendor_pairs(metadata.get("scope_asymmetry_pairs")),
        "scope_trigger": str(metadata.get("scope_trigger") or "manual").strip() or "manual",
        "scope_name": str(metadata.get("scope_name") or "").strip(),
    }


def _competitive_scope_run_id(
    task: ScheduledTask | Any,
    scope_meta: dict[str, Any],
) -> str | None:
    metadata = getattr(task, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    if scope_meta.get("scope_id"):
        explicit = str(metadata.get("run_id") or "").strip()
        if explicit:
            return explicit
    return _task_run_id(task)


def _coerce_timestamptz(value: Any) -> Any:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _witness_row_payload(witness: dict[str, Any]) -> dict[str, Any]:
    return {
        "witness_id": str(witness.get("witness_id") or witness.get("_sid") or ""),
        "review_id": str(witness.get("review_id") or ""),
        "witness_type": str(witness.get("witness_type") or ""),
        "excerpt_text": str(witness.get("excerpt_text") or ""),
        "source": str(witness.get("source") or ""),
        "reviewed_at": _coerce_timestamptz(witness.get("reviewed_at")),
        "reviewer_company": witness.get("reviewer_company"),
        "reviewer_title": witness.get("reviewer_title"),
        "pain_category": witness.get("pain_category"),
        "competitor": witness.get("competitor"),
        "salience_score": float(witness.get("salience_score") or 0.0),
        "selection_reason": str(witness.get("selection_reason") or ""),
        "signal_tags": json.dumps(witness.get("signal_tags") or []),
        "source_id": str(witness.get("_sid") or witness.get("witness_id") or ""),
        "specificity_score": float(witness.get("specificity_score") or 0.0),
        "generic_reason": str(witness.get("generic_reason") or "").strip() or None,
        "witness_hash": str(witness.get("witness_hash") or "").strip()
        or compute_witness_hash(witness),
    }


async def _persist_packet_artifacts(
    pool,
    *,
    vendor_name: str,
    as_of_date: date,
    analysis_window_days: int,
    evidence_hash: str,
    packet: Any,
) -> None:
    """Persist witness-backed packet artifacts for inspection and caching."""
    packet_payload = {
        "schema_version": _PACKET_SCHEMA_VERSION,
        "vendor_name": vendor_name,
        "payload": packet.to_llm_payload(),
        "source_ids": sorted(packet.source_ids()),
    }
    await pool.execute(
        """
        INSERT INTO b2b_vendor_reasoning_packets
            (vendor_name, as_of_date, analysis_window_days,
             schema_version, evidence_hash, packet)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        ON CONFLICT (vendor_name, as_of_date, analysis_window_days, schema_version)
        DO UPDATE SET
            evidence_hash = EXCLUDED.evidence_hash,
            packet = EXCLUDED.packet,
            created_at = NOW()
        """,
        vendor_name,
        as_of_date,
        analysis_window_days,
        _PACKET_SCHEMA_VERSION,
        evidence_hash,
        json.dumps(packet_payload, default=str),
    )
    existing_rows = await pool.fetch(
        """
        SELECT witness_id, witness_hash
        FROM b2b_vendor_witnesses
        WHERE vendor_name = $1
          AND as_of_date = $2
          AND analysis_window_days = $3
          AND schema_version = $4
        """,
        vendor_name,
        as_of_date,
        analysis_window_days,
        _PACKET_SCHEMA_VERSION,
    )
    existing_hashes = {
        str(row["witness_id"]): str(row["witness_hash"] or "")
        for row in existing_rows
    }
    incoming_rows = [
        _witness_row_payload(witness)
        for witness in (getattr(packet, "witness_pack", []) or [])
        if isinstance(witness, dict)
    ]
    incoming_ids = {
        row["witness_id"] for row in incoming_rows
        if row["witness_id"]
    }
    stale_ids = sorted(set(existing_hashes) - incoming_ids)
    if stale_ids:
        await pool.execute(
            """
            DELETE FROM b2b_vendor_witnesses
            WHERE vendor_name = $1
              AND as_of_date = $2
              AND analysis_window_days = $3
              AND schema_version = $4
              AND witness_id = ANY($5::text[])
            """,
            vendor_name,
            as_of_date,
            analysis_window_days,
            _PACKET_SCHEMA_VERSION,
            stale_ids,
        )
    for witness in incoming_rows:
        if existing_hashes.get(witness["witness_id"]) == witness["witness_hash"]:
            continue
        await pool.execute(
            """
            INSERT INTO b2b_vendor_witnesses
                (vendor_name, as_of_date, analysis_window_days, schema_version,
                 evidence_hash, witness_id, review_id, witness_type, excerpt_text,
                 source, reviewed_at, reviewer_company, reviewer_title,
                 pain_category, competitor, salience_score, selection_reason,
                 signal_tags, source_id, specificity_score, generic_reason,
                 witness_hash)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    $14, $15, $16, $17, $18::jsonb, $19, $20, $21, $22)
            ON CONFLICT (
                vendor_name, as_of_date, analysis_window_days, schema_version, witness_id
            )
            DO UPDATE SET
                evidence_hash = EXCLUDED.evidence_hash,
                review_id = EXCLUDED.review_id,
                witness_type = EXCLUDED.witness_type,
                excerpt_text = EXCLUDED.excerpt_text,
                source = EXCLUDED.source,
                reviewed_at = EXCLUDED.reviewed_at,
                reviewer_company = EXCLUDED.reviewer_company,
                reviewer_title = EXCLUDED.reviewer_title,
                pain_category = EXCLUDED.pain_category,
                competitor = EXCLUDED.competitor,
                salience_score = EXCLUDED.salience_score,
                selection_reason = EXCLUDED.selection_reason,
                signal_tags = EXCLUDED.signal_tags,
                source_id = EXCLUDED.source_id,
                specificity_score = EXCLUDED.specificity_score,
                generic_reason = EXCLUDED.generic_reason,
                witness_hash = EXCLUDED.witness_hash
            """,
            vendor_name,
            as_of_date,
            analysis_window_days,
            _PACKET_SCHEMA_VERSION,
            evidence_hash,
            witness["witness_id"],
            witness["review_id"],
            witness["witness_type"],
            witness["excerpt_text"],
            witness["source"],
            witness["reviewed_at"],
            witness["reviewer_company"],
            witness["reviewer_title"],
            witness["pain_category"],
            witness["competitor"],
            witness["salience_score"],
            witness["selection_reason"],
            witness["signal_tags"],
            witness["source_id"],
            witness["specificity_score"],
            witness["generic_reason"],
            witness["witness_hash"],
        )


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: generate reasoning synthesis per vendor."""
    cfg = settings.b2b_churn
    scope_meta = _competitive_scope_metadata(task)
    run_id = _competitive_scope_run_id(task, scope_meta)
    scope_id = scope_meta.get("scope_id") or ""
    scope_vendor_names = scope_meta.get("scope_vendor_names") or []
    scope_pairwise_pairs = scope_meta.get("scope_pairwise_pairs") or []
    scope_category_names = scope_meta.get("scope_category_names") or []
    scope_asymmetry_pairs = scope_meta.get("scope_asymmetry_pairs") or []
    scope_trigger = scope_meta.get("scope_trigger") or "manual"

    async def _finalize_scope_result(result: dict[str, Any]) -> dict[str, Any]:
        if not scope_id:
            return result
        try:
            from ...storage.repositories.competitive_set import get_competitive_set_repo

            status = "succeeded"
            if result.get("vendors_failed") and result.get("vendors_reasoned"):
                status = "partial"
            elif result.get("vendors_failed") and not result.get("vendors_reasoned"):
                status = "failed"
            await get_competitive_set_repo().mark_run_completed(
                UUID(scope_id),
                status=status,
                summary={
                    "run_id": run_id,
                    "trigger": scope_trigger,
                    **result,
                },
            )
        except Exception:
            logger.debug(
                "Failed to finalize competitive set run %s",
                scope_id,
                exc_info=True,
            )
        return result

    pool = get_db_pool()
    if not pool.is_initialized:
        return await _finalize_scope_result({"_skip_synthesis": "DB not ready"})

    if scope_id:
        try:
            from ...storage.repositories.competitive_set import get_competitive_set_repo

            await get_competitive_set_repo().mark_run_started(
                UUID(scope_id),
                run_id=run_id,
                trigger=scope_trigger,
                execution_id=str((task.metadata or {}).get("_execution_id") or "") or None,
            )
        except Exception:
            logger.debug(
                "Failed to mark competitive set run started %s",
                scope_id,
                exc_info=True,
            )

    scheduled_scope_strategy = str(
        ((task.metadata or {}).get("scheduled_scope_strategy") or "")
    ).strip()
    if not scope_id and scheduled_scope_strategy == "competitive_sets":
        from ...services.b2b_competitive_sets import (
            build_competitive_set_plan,
            load_vendor_category_map,
            plan_to_synthesis_metadata,
        )
        from ...storage.repositories.competitive_set import get_competitive_set_repo

        repo = get_competitive_set_repo()
        due_sets = await repo.list_due_scheduled(
            limit=max(1, int(getattr(cfg, "competitive_set_refresh_batch_size", 10))),
        )
        if not due_sets:
            return {"_skip_synthesis": "No due competitive sets"}

        aggregate = {
            "competitive_sets_processed": 0,
            "competitive_sets_failed": 0,
            "vendors_total": 0,
            "vendors_reasoned": 0,
            "vendors_failed": 0,
            "vendors_skipped": 0,
            "cross_vendor_succeeded": 0,
            "cross_vendor_failed": 0,
            "total_tokens": 0,
            "competitive_set_ids": [str(item.id) for item in due_sets],
        }
        for competitive_set in due_sets:
            try:
                category_by_vendor = await load_vendor_category_map(
                    pool,
                    [competitive_set.focal_vendor_name, *competitive_set.competitor_vendor_names],
                )
                plan = build_competitive_set_plan(
                    competitive_set,
                    category_by_vendor=category_by_vendor,
                )
                scoped_task = ScheduledTask(
                    id=task.id,
                    name=task.name,
                    task_type=task.task_type,
                    schedule_type=task.schedule_type,
                    description=task.description,
                    prompt=task.prompt,
                    agent_type=task.agent_type,
                    cron_expression=task.cron_expression,
                    interval_seconds=task.interval_seconds,
                    run_at=task.run_at,
                    timezone=task.timezone,
                    enabled=task.enabled,
                    max_retries=task.max_retries,
                    retry_delay_seconds=task.retry_delay_seconds,
                    timeout_seconds=task.timeout_seconds,
                    metadata={
                        **(task.metadata or {}),
                        **plan_to_synthesis_metadata(plan),
                        "scope_name": competitive_set.name,
                        "scope_trigger": "scheduled",
                        "run_id": f"{run_id}:{competitive_set.id}",
                    },
                    created_at=task.created_at,
                    updated_at=task.updated_at,
                    last_run_at=task.last_run_at,
                    next_run_at=task.next_run_at,
                )
                child_result = await run(scoped_task)
            except Exception:
                logger.exception(
                    "Competitive set synthesis scan failed for %s",
                    competitive_set.id,
                )
                aggregate["competitive_sets_failed"] += 1
                try:
                    await repo.mark_run_completed(
                        competitive_set.id,
                        status="failed",
                        summary={
                            "run_id": f"{run_id}:{competitive_set.id}",
                            "trigger": "scheduled",
                            "error": "Competitive-set scan raised an exception before synthesis",
                        },
                    )
                except Exception:
                    logger.debug(
                        "Failed to mark competitive set failure %s",
                        competitive_set.id,
                        exc_info=True,
                    )
                continue
            aggregate["competitive_sets_processed"] += 1
            if child_result.get("vendors_failed") and not child_result.get("vendors_reasoned"):
                aggregate["competitive_sets_failed"] += 1
            for key in (
                "vendors_total",
                "vendors_reasoned",
                "vendors_failed",
                "vendors_skipped",
                "cross_vendor_succeeded",
                "cross_vendor_failed",
                "total_tokens",
            ):
                aggregate[key] += int(child_result.get(key, 0) or 0)
        return aggregate

    from ._b2b_shared import fetch_all_pool_layers

    window_days = cfg.intelligence_window_days
    today = date.today()

    # Load all pool layers
    t0 = time.monotonic()
    vendor_pools = await fetch_all_pool_layers(
        pool, as_of=today, analysis_window_days=window_days,
    )
    load_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "Loaded pool layers for %d vendors in %.0fms",
        len(vendor_pools), load_ms,
    )

    if not vendor_pools:
        return await _finalize_scope_result({"_skip_synthesis": "No pool data available"})

    # Optional vendor filter: scoped competitive set or legacy test_vendors.
    test_vendors = (task.metadata or {}).get("test_vendors")
    filter_vendors = scope_vendor_names[:] or _metadata_text_list(test_vendors)
    if filter_vendors:
        vendor_set = {v.lower() for v in filter_vendors}
        vendor_pools = {
            k: v for k, v in vendor_pools.items()
            if k.lower() in vendor_set
        }
        logger.info(
            "Filtered reasoning synthesis to %d vendors: %s",
            len(vendor_pools), sorted(vendor_pools.keys()),
        )
    if scope_vendor_names and not vendor_pools:
        return await _finalize_scope_result({
            "vendors_total": 0,
            "vendors_reasoned": 0,
            "vendors_failed": 0,
            "vendors_skipped": 0,
            "_skip_synthesis": "Competitive set scope has no matching vendor pools",
        })

    # Check for existing synthesis to skip unchanged vendors
    existing = await pool.fetch(
        """
        WITH latest AS (
            SELECT DISTINCT ON (vendor_name)
                   vendor_name,
                   as_of_date,
                   evidence_hash,
                   jsonb_path_exists(synthesis, '$.packet_artifacts.witness_pack[*]') AS has_witness_pack,
                   jsonb_path_exists(synthesis, '$.reference_ids.metric_ids[*]') AS has_metric_refs,
                   jsonb_path_exists(synthesis, '$.reference_ids.witness_ids[*]') AS has_witness_refs
            FROM b2b_reasoning_synthesis
            WHERE analysis_window_days = $1
              AND schema_version = $2
            ORDER BY vendor_name, as_of_date DESC, created_at DESC
        )
        SELECT vendor_name, as_of_date, evidence_hash, has_witness_pack, has_metric_refs, has_witness_refs
        FROM latest
        """,
        window_days, _SCHEMA_VERSION,
    )
    latest_rows: dict[str, dict[str, Any]] = {
        r["vendor_name"]: dict(r) for r in existing
    }

    # Filter to vendors needing reasoning
    max_stale_days = max(
        0,
        int(getattr(cfg, "reasoning_synthesis_max_stale_days", 3)),
    )
    rerun_if_missing_packet_artifacts = bool(
        getattr(cfg, "reasoning_synthesis_rerun_if_missing_packet_artifacts", True),
    )
    rerun_if_missing_reference_ids = bool(
        getattr(cfg, "reasoning_synthesis_rerun_if_missing_reference_ids", True),
    )
    force = bool((task.metadata or {}).get("force"))
    force_cross_vendor = bool((task.metadata or {}).get("force_cross_vendor"))
    rerun_reason_counts = {
        "forced": 0,
        "missing_prior_row": 0,
        "hash_changed": 0,
        "prior_quality_weak": 0,
        "missing_packet_artifacts": 0,
        "missing_reference_ids": 0,
        "hash_reuse": 0,
        "stale_reused": 0,
    }
    normalized_hashes: dict[str, str] = {}
    legacy_current_hashes: dict[str, str] = {}
    transition_candidates_by_date: dict[date, list[str]] = {}
    for vendor_name, layers in vendor_pools.items():
        ev_hash = _compute_pool_hash(layers)
        legacy_ev_hash = _compute_pool_hash_legacy(layers)
        normalized_hashes[vendor_name] = ev_hash
        legacy_current_hashes[vendor_name] = legacy_ev_hash
        latest_row = latest_rows.get(vendor_name)
        if latest_row is None:
            continue
        prior_hash = str(latest_row.get("evidence_hash") or "")
        if prior_hash in {ev_hash, legacy_ev_hash}:
            continue
        prior_as_of_date = _coerce_as_of_date(latest_row.get("as_of_date"))
        if prior_as_of_date is None or prior_as_of_date == today:
            continue
        transition_candidates_by_date.setdefault(
            prior_as_of_date, [],
        ).append(vendor_name)

    legacy_hash_compatible_vendors: set[str] = set()
    for prior_as_of_date, candidate_vendors in transition_candidates_by_date.items():
        prior_vendor_pools = await fetch_all_pool_layers(
            pool,
            as_of=prior_as_of_date,
            analysis_window_days=window_days,
        )
        for vendor_name in candidate_vendors:
            prior_layers = prior_vendor_pools.get(vendor_name)
            latest_row = latest_rows.get(vendor_name)
            if prior_layers is None or latest_row is None:
                continue
            prior_hash = str(latest_row.get("evidence_hash") or "")
            prior_normalized_hash = _compute_pool_hash(prior_layers)
            prior_legacy_hash = _compute_pool_hash_legacy(prior_layers)
            if prior_hash not in {prior_normalized_hash, prior_legacy_hash}:
                continue
            if prior_normalized_hash == normalized_hashes.get(vendor_name):
                legacy_hash_compatible_vendors.add(vendor_name)

    vendors_to_reason: list[tuple[str, dict, str, dict[str, Any]]] = []
    for vendor_name, layers in vendor_pools.items():
        ev_hash = normalized_hashes[vendor_name]
        latest_row = latest_rows.get(vendor_name)
        prior_hash = str((latest_row or {}).get("evidence_hash") or "")
        hash_matches_prior = bool(latest_row) and (
            prior_hash in {ev_hash, legacy_current_hashes[vendor_name]}
            or vendor_name in legacy_hash_compatible_vendors
        )
        decision = _classify_vendor_reasoning_decision(
            vendor_name=vendor_name,
            today=today,
            evidence_hash=ev_hash,
            latest_row=latest_row,
            force=force,
            max_stale_days=max_stale_days,
            rerun_if_missing_packet_artifacts=rerun_if_missing_packet_artifacts,
            rerun_if_missing_reference_ids=rerun_if_missing_reference_ids,
            hash_matches_prior=hash_matches_prior,
        )
        rerun_reason_counts[decision["reason"]] += 1
        if not decision["should_reason"]:
            continue
        vendors_to_reason.append((vendor_name, layers, ev_hash, decision))

    skipped = len(vendor_pools) - len(vendors_to_reason)
    logger.info(
        "Reasoning synthesis v2: %d vendors to process, %d skipped (unchanged)",
        len(vendors_to_reason), skipped,
    )

    if not vendors_to_reason and not (
        cfg.cross_vendor_synthesis_enabled and force_cross_vendor
    ):
        return await _finalize_scope_result({
            "vendors_total": len(vendor_pools),
            "vendors_reasoned": 0,
            "vendors_failed": 0,
            "vendors_skipped": skipped,
            "vendors_skipped_hash_reuse": rerun_reason_counts["hash_reuse"],
            "vendors_skipped_stale_reuse": rerun_reason_counts["stale_reused"],
            "vendors_rerun_hash_changed": rerun_reason_counts["hash_changed"],
            "vendors_rerun_prior_quality_weak": rerun_reason_counts["prior_quality_weak"],
            "vendors_rerun_missing_prior": rerun_reason_counts["missing_prior_row"],
            "vendors_rerun_missing_packet_artifacts": rerun_reason_counts["missing_packet_artifacts"],
            "vendors_rerun_missing_reference_ids": rerun_reason_counts["missing_reference_ids"],
            "vendors_forced": rerun_reason_counts["forced"],
            "_skip_synthesis": "All vendors unchanged",
        })
    if not vendors_to_reason and cfg.cross_vendor_synthesis_enabled and force_cross_vendor:
        logger.info(
            "Reasoning synthesis v2: vendor phase skipped; force_cross_vendor enabled",
        )

    # Resolve LLM via standard pipeline routing
    from ...pipelines.llm import get_pipeline_llm

    synthesis_model = str(
        getattr(cfg, "reasoning_synthesis_model", "") or ""
    ).strip()
    if not synthesis_model:
        synthesis_model = str(
            getattr(settings.llm, "openrouter_reasoning_model", "") or ""
        ).strip()
    llm = get_pipeline_llm(
        workload="synthesis",
        openrouter_model=synthesis_model or None,
        auto_activate_ollama=False,
    )
    if llm is None:
        return await _finalize_scope_result(
            {"_skip_synthesis": "No LLM available for reasoning synthesis"}
        )

    from ...reasoning.single_pass_prompts.reasoning_synthesis import (
        REASONING_SYNTHESIS_PROMPT,
    )
    from ...services.protocols import Message

    from ._b2b_pool_compression import compress_vendor_pools
    from ._b2b_reasoning_contracts import build_persistable_synthesis
    from ._b2b_synthesis_validation import (
        normalize_synthesis_source_ids,
        validate_synthesis,
    )

    # Process vendors with concurrency limit
    max_concurrent = getattr(cfg, "reasoning_synthesis_concurrency", 4)
    max_attempts = max(1, int(getattr(cfg, "reasoning_synthesis_attempts", 2)))
    retry_delay = float(
        getattr(cfg, "reasoning_synthesis_retry_delay_seconds", 0.5),
    )
    llm_timeout_seconds = max(
        1.0,
        float(getattr(cfg, "reasoning_synthesis_timeout_seconds", 180.0)),
    )
    llm_max_input_tokens = max(
        512,
        int(getattr(cfg, "reasoning_synthesis_max_input_tokens", 20000)),
    )
    _default_max_tokens = 16384
    llm_max_tokens = max(
        256,
        min(
            int(getattr(cfg, "reasoning_synthesis_max_tokens", _default_max_tokens)),
            _default_max_tokens,
        ),
    )
    llm_temperature = float(
        getattr(cfg, "reasoning_synthesis_temperature", 0.3),
    )
    full_max_items_per_pool = max(
        1,
        int(getattr(cfg, "reasoning_synthesis_max_items_per_pool", 8)),
    )
    lean_max_items_per_pool = max(
        1,
        int(getattr(cfg, "reasoning_synthesis_lean_max_items_per_pool", 4)),
    )
    lean_max_witnesses = max(
        1,
        int(getattr(cfg, "reasoning_synthesis_lean_max_witnesses", 6)),
    )
    feedback_limit = max(
        1, int(getattr(cfg, "reasoning_synthesis_feedback_limit", 5)),
    )
    sem = asyncio.Semaphore(max_concurrent)
    total_tokens = 0
    succeeded = 0
    failed = 0
    validation_failures = 0
    witness_count = 0
    witness_vendor_rows = 0
    input_budget_rejections = 0
    payload_mode_full = 0
    payload_mode_lean = 0
    failed_vendors: list[dict[str, Any]] = []

    async def _reason_one(
        vendor_name: str, layers: dict, ev_hash: str, decision: dict[str, Any],
    ) -> None:
        nonlocal total_tokens, succeeded, failed, validation_failures, failed_vendors
        nonlocal witness_count, witness_vendor_rows
        nonlocal input_budget_rejections, payload_mode_full, payload_mode_lean
        async with sem:
            packet = compress_vendor_pools(
                vendor_name,
                layers,
                max_items_per_pool=full_max_items_per_pool,
            )
            packet_witness_count = len(getattr(packet, "witness_pack", []) or [])
            if packet_witness_count > 0:
                witness_count += packet_witness_count
                witness_vendor_rows += 1
            try:
                await _persist_packet_artifacts(
                    pool,
                    vendor_name=vendor_name,
                    as_of_date=today,
                    analysis_window_days=window_days,
                    evidence_hash=ev_hash,
                    packet=packet,
                )
            except Exception:
                logger.warning(
                    "Witness packet persistence failed for %s",
                    vendor_name,
                    exc_info=True,
                )
            payload_mode = "full"
            payload_items_per_pool = full_max_items_per_pool
            section_packets_included = True
            include_contradiction_rows = True
            include_minority_signals = True
            payload_packet = packet
            payload = json.dumps(
                payload_packet.to_reasoning_payload(
                    compact_metric_ledger=True,
                    compact_aggregates=True,
                ),
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            )
            estimated_input_tokens = _approx_prompt_input_tokens(
                REASONING_SYNTHESIS_PROMPT,
                payload,
            )
            payload_witness_count = packet_witness_count
            prompt_packet = packet.prompt_validation_view(
                compact_aggregates=True,
                include_contradiction_rows=include_contradiction_rows,
                include_minority_signals=include_minority_signals,
                include_section_packets=section_packets_included,
            )
            if estimated_input_tokens > llm_max_input_tokens:
                lean_packet = compress_vendor_pools(
                    vendor_name,
                    layers,
                    max_items_per_pool=lean_max_items_per_pool,
                    max_witnesses=lean_max_witnesses,
                )
                include_contradiction_rows = False
                include_minority_signals = False
                payload = json.dumps(
                    lean_packet.to_reasoning_payload(
                        compact_metric_ledger=True,
                        compact_aggregates=True,
                        include_contradiction_rows=False,
                        include_minority_signals=False,
                        include_section_packets=False,
                    ),
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                )
                estimated_input_tokens = _approx_prompt_input_tokens(
                    REASONING_SYNTHESIS_PROMPT,
                    payload,
                )
                payload_mode = "lean"
                payload_items_per_pool = lean_max_items_per_pool
                payload_witness_count = len(getattr(lean_packet, "witness_pack", []) or [])
                section_packets_included = False
                payload_packet = lean_packet
                prompt_packet = lean_packet.prompt_validation_view(
                    compact_aggregates=True,
                    include_contradiction_rows=include_contradiction_rows,
                    include_minority_signals=include_minority_signals,
                    include_section_packets=section_packets_included,
                )
            if estimated_input_tokens > llm_max_input_tokens:
                logger.warning(
                    "Reasoning synthesis rejected for %s: estimated input %d exceeds cap %d",
                    vendor_name,
                    estimated_input_tokens,
                    llm_max_input_tokens,
                )
                failed_vendors.append({
                    "vendor_name": vendor_name,
                    "stage": "input_budget",
                    "reasons": [
                        "input token budget exceeded",
                    ],
                    "tokens_used": 0,
                    "attempts_used": 0,
                })
                failed += 1
                input_budget_rejections += 1
                from ..visibility import emit_event, record_attempt

                await record_attempt(
                    pool,
                    artifact_type="reasoning_synthesis",
                    artifact_id=vendor_name,
                    run_id=run_id,
                    attempt_no=1,
                    stage="generation",
                    status="rejected",
                    blocker_count=1,
                    blocking_issues=[
                        (
                            "input token budget exceeded: "
                            f"estimated_input_tokens={estimated_input_tokens}, "
                            f"cap={llm_max_input_tokens}"
                        ),
                    ],
                    failure_step="input_budget",
                    error_message="Vendor reasoning prompt exceeded the configured input token cap",
                )
                await emit_event(
                    pool,
                    stage="synthesis",
                    event_type="input_budget_rejected",
                    entity_type="vendor",
                    entity_id=vendor_name,
                    summary=(
                        "Vendor reasoning prompt exceeded the configured input token cap"
                    ),
                    severity="warning",
                    actionable=True,
                    artifact_type="reasoning_synthesis",
                    run_id=run_id,
                    reason_code="input_budget",
                    detail={
                        "estimated_input_tokens": estimated_input_tokens,
                        "cap": llm_max_input_tokens,
                        "payload_mode": payload_mode,
                    },
                )
                return
            if payload_mode == "lean":
                payload_mode_lean += 1
            else:
                payload_mode_full += 1
            failure_reasons: list[str] = []
            last_text = ""
            last_validation = None
            synthesis: dict[str, Any] | None = None
            vendor_tokens = 0

            for attempt in range(max_attempts):
                messages = [
                    Message(role="system", content=REASONING_SYNTHESIS_PROMPT),
                    Message(role="user", content=payload),
                ]
                if attempt > 0 and last_text:
                    messages.append(Message(role="assistant", content=last_text))
                if attempt > 0 and failure_reasons:
                    feedback = "\n".join(
                        f"- {reason}" for reason in failure_reasons[:feedback_limit]
                    )
                    retry_guidance = (
                        _retry_guidance(last_validation)
                        if last_validation is not None
                        else []
                    )
                    guidance_block = ""
                    if retry_guidance:
                        guidance_block = (
                            "\nAdditional requirements:\n"
                            + "\n".join(f"- {item}" for item in retry_guidance)
                        )
                    messages.append(Message(
                        role="user",
                        content=(
                            "Your previous response was rejected. "
                            "Return a complete corrected JSON object only.\n"
                            f"Fix these issues:\n{feedback}"
                            f"{guidance_block}"
                        ),
                    ))
                try:
                    call_started = time.monotonic()
                    import re
                    from ...pipelines.llm import parse_json_response

                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            llm.chat,
                            messages=messages,
                            max_tokens=llm_max_tokens,
                            temperature=llm_temperature,
                            response_format={"type": "json_object"},
                        ),
                        timeout=llm_timeout_seconds,
                    )
                    _trace_reasoning_result(
                        "task.b2b_reasoning_synthesis",
                        llm=llm,
                        messages=messages,
                        result=result,
                        metadata={
                            "run_id": run_id,
                            "vendor_name": vendor_name,
                            "reasoning_mode": "vendor_synthesis",
                            "attempt_no": attempt + 1,
                            "schema_version": _SCHEMA_VERSION,
                            "rerun_reason": decision["reason"],
                            "payload_mode": payload_mode,
                            "estimated_input_tokens": estimated_input_tokens,
                            "prior_row_age_days": decision.get("prior_row_age_days"),
                            "prior_row_as_of_date": (
                                decision["prior_as_of_date"].isoformat()
                                if decision.get("prior_as_of_date") is not None
                                else None
                            ),
                            "hash_changed": bool(decision.get("hash_changed")),
                            "forced": decision["reason"] == "forced",
                        },
                        duration_ms=(time.monotonic() - call_started) * 1000,
                    )
                    text = result.get("response", "").strip()
                    usage = result.get("usage", {})
                    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    vendor_tokens += tokens
                    total_tokens += tokens

                    text = re.sub(
                        r"<think>.*?</think>", "", text, flags=re.DOTALL,
                    ).strip()
                    if "<scratchpad>" in text:
                        text = text.split("</scratchpad>")[-1].strip()
                    last_text = text

                    parsed = parse_json_response(text, recover_truncated=True)
                    if not isinstance(parsed, dict):
                        failure_reasons = ["LLM did not return a JSON object"]
                        synthesis = None
                    elif parsed.get("_parse_fallback"):
                        failure_reasons = ["LLM did not return valid JSON"]
                        synthesis = None
                    else:
                        parsed = normalize_synthesis_source_ids(parsed, prompt_packet)
                        # Validate the canonical persisted shape, not the raw model
                        # payload. Deterministic contract repairs already know how to
                        # normalize migration semantics, named examples, and related
                        # evidence wiring; paying for a second LLM call before those
                        # repairs run is wasted token burn.
                        candidate = build_persistable_synthesis(parsed, prompt_packet)
                        candidate = normalize_synthesis_source_ids(candidate, prompt_packet)
                        vresult = validate_synthesis(
                            candidate, prompt_packet, governance_blocking=True,
                        )
                        if vresult.is_valid:
                            synthesis = candidate
                            last_validation = vresult
                            break
                        synthesis = None
                        last_validation = vresult
                        await _record_validation_attempt(
                            pool,
                            vendor_name=vendor_name,
                            run_id=run_id,
                            as_of_date=today,
                            analysis_window_days=window_days,
                            attempt_no=attempt + 1,
                            vresult=vresult,
                            feedback_limit=feedback_limit,
                            attempt_tokens=tokens,
                            escalation_window_hours=max(
                                1,
                                int(getattr(cfg, "reasoning_retry_escalation_window_hours", 24)),
                            ),
                            repeat_rule_threshold=max(
                                2,
                                int(getattr(cfg, "reasoning_retry_repeat_rule_threshold", 3)),
                            ),
                            cost_min_retries=max(
                                1,
                                int(getattr(cfg, "reasoning_retry_cost_min_retries", 2)),
                            ),
                            cost_tokens_threshold=max(
                                1000,
                                int(getattr(cfg, "reasoning_retry_cost_tokens_threshold", 80000)),
                            ),
                            emit_retry_event=(attempt + 1) < max_attempts,
                        )
                        failure_reasons = _validation_feedback(
                            vresult, feedback_limit,
                        ) or [vresult.summary()]
                        for err in vresult.errors:
                            logger.debug(
                                "  [%s] %s: %s", err.code, err.path, err.message,
                            )
                except asyncio.TimeoutError:
                    synthesis = None
                    failure_reasons = [
                        "TimeoutError: reasoning LLM call exceeded "
                        f"{llm_timeout_seconds:.1f}s",
                    ]
                    if attempt + 1 >= max_attempts:
                        logger.warning(
                            "Reasoning synthesis timed out for %s after %.1fs",
                            vendor_name,
                            llm_timeout_seconds,
                        )
                        failed_vendors.append({
                            "vendor_name": vendor_name,
                            "stage": "llm_exception",
                            "reasons": failure_reasons[:feedback_limit],
                            "tokens_used": vendor_tokens,
                            "attempts_used": attempt + 1,
                        })
                        failed += 1
                        from ..visibility import record_attempt, emit_event
                        await record_attempt(
                            pool, artifact_type="reasoning_synthesis",
                            artifact_id=vendor_name,
                            run_id=run_id, attempt_no=attempt + 1,
                            stage="llm_call", status="failed",
                            failure_step="timeout",
                            error_message=f"LLM call exceeded {llm_timeout_seconds:.1f}s",
                        )
                        await emit_event(
                            pool, stage="synthesis", event_type="llm_timeout",
                            entity_type="vendor", entity_id=vendor_name,
                            summary=f"Reasoning synthesis timed out for {vendor_name}",
                            severity="warning", actionable=True,
                            artifact_type="reasoning_synthesis",
                            run_id=run_id, reason_code="llm_timeout",
                        )
                        return
                except Exception as exc:
                    synthesis = None
                    failure_reasons = [f"{type(exc).__name__}: {exc}"]
                    if attempt + 1 >= max_attempts:
                        logger.warning(
                            "Reasoning synthesis failed for %s",
                            vendor_name, exc_info=True,
                        )
                        failed_vendors.append({
                            "vendor_name": vendor_name,
                            "stage": "llm_exception",
                            "reasons": failure_reasons[:feedback_limit],
                            "tokens_used": vendor_tokens,
                            "attempts_used": attempt + 1,
                        })
                        failed += 1
                        from ..visibility import record_attempt, emit_event
                        await record_attempt(
                            pool, artifact_type="reasoning_synthesis",
                            artifact_id=vendor_name,
                            run_id=run_id, attempt_no=attempt + 1,
                            stage="llm_call", status="failed",
                            failure_step="llm_exception",
                            error_message=str(exc)[:200],
                        )
                        await emit_event(
                            pool, stage="synthesis", event_type="llm_exception",
                            entity_type="vendor", entity_id=vendor_name,
                            summary=f"Reasoning synthesis exception for {vendor_name}: {type(exc).__name__}",
                            severity="error", actionable=True,
                            artifact_type="reasoning_synthesis",
                            run_id=run_id, reason_code="llm_exception",
                        )
                        return

                if synthesis is not None:
                    break

                if attempt + 1 >= max_attempts:
                    if last_validation is not None:
                        logger.warning(
                            "Reasoning synthesis for %s failed validation: %s",
                            vendor_name, last_validation.summary(),
                        )
                        failed_vendors.append({
                            "vendor_name": vendor_name,
                            "stage": "validation",
                            "summary": last_validation.summary(),
                            "reasons": failure_reasons[:feedback_limit],
                            "tokens_used": vendor_tokens,
                            "attempts_used": attempt + 1,
                        })
                        validation_failures += 1
                    else:
                        logger.warning(
                            "Reasoning synthesis for %s failed: %s",
                            vendor_name, "; ".join(failure_reasons[:3]),
                        )
                        failed_vendors.append({
                            "vendor_name": vendor_name,
                            "stage": "llm_response",
                            "reasons": failure_reasons[:feedback_limit],
                            "tokens_used": vendor_tokens,
                            "attempts_used": attempt + 1,
                        })
                    failed += 1
                    if last_validation is None:
                        from ..visibility import record_attempt
                        await record_attempt(
                            pool, artifact_type="reasoning_synthesis",
                            artifact_id=vendor_name,
                            run_id=run_id, attempt_no=attempt + 1,
                            stage="validation", status="rejected",
                            blocker_count=len(failure_reasons),
                            blocking_issues=failure_reasons[:5],
                            failure_step="llm_response",
                            error_message="; ".join(failure_reasons[:2])[:200],
                        )
                    return

                logger.info(
                    "Reasoning synthesis retrying %s (%d/%d): %s",
                    vendor_name,
                    attempt + 2,
                    max_attempts,
                    "; ".join(failure_reasons[:3]),
                )
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)

            assert synthesis is not None
            assert last_validation is not None
            vresult = last_validation

            if "meta" not in synthesis:
                synthesis["meta"] = {}
            meta = synthesis["meta"]
            meta.setdefault(
                "synthesized_at",
                datetime.now(timezone.utc).isoformat(),
            )
            total_items = sum(
                len(items) for items in packet.pools.values()
            )
            meta.setdefault("total_evidence_items", total_items)
            meta["rerun_reason"] = decision["reason"]
            meta["payload_mode"] = payload_mode
            meta["estimated_input_tokens"] = estimated_input_tokens
            meta["packet_witness_count"] = payload_witness_count
            meta["packet_items_per_pool"] = payload_items_per_pool
            meta["section_packets_included"] = section_packets_included
            meta["payload_component_tokens"] = payload_packet.reasoning_payload_component_tokens(
                compact_metric_ledger=True,
                compact_aggregates=True,
                include_contradiction_rows=include_contradiction_rows,
                include_minority_signals=include_minority_signals,
                include_section_packets=section_packets_included,
            )
            if decision.get("prior_as_of_date") is not None:
                meta["prior_row_as_of_date"] = decision["prior_as_of_date"].isoformat()
            if decision.get("prior_row_age_days") is not None:
                meta["prior_row_age_days"] = int(decision["prior_row_age_days"])
            synthesis = build_persistable_synthesis(synthesis, packet)
            if payload_mode == "lean":
                lean_mode_info = {
                    "omitted": [
                        k for k, v in [
                            ("contradiction_rows", not include_contradiction_rows),
                            ("minority_signals", not include_minority_signals),
                            ("section_packets", not section_packets_included),
                        ] if v
                    ],
                    "items_per_pool": payload_items_per_pool,
                    "witness_count": payload_witness_count,
                }
                # Store in both meta (canonical) and top-level (quality gate reads it)
                synthesis.setdefault("meta", {})["lean_mode"] = lean_mode_info
                synthesis["_lean_mode"] = lean_mode_info
            persisted_vresult = validate_synthesis(synthesis, packet, governance_blocking=True)
            if not persisted_vresult.is_valid:
                logger.warning(
                    "Persisted reasoning synthesis for %s failed validation: %s",
                    vendor_name, persisted_vresult.summary(),
                )
                from ..visibility import emit_event, record_synthesis_validation_results
                await record_synthesis_validation_results(
                    pool,
                    vendor_name=vendor_name,
                    as_of_date=today,
                    analysis_window_days=window_days,
                    schema_version=_SCHEMA_VERSION,
                    run_id=run_id,
                    attempt_no=attempt + 1,
                    results=_validation_rows(persisted_vresult),
                )
                await emit_event(
                    pool, stage="synthesis", event_type="validation_failure",
                    entity_type="vendor", entity_id=vendor_name,
                    summary=f"Synthesis validation failed: {persisted_vresult.summary()[:120]}",
                    severity="error", actionable=True,
                    artifact_type="reasoning_synthesis",
                    run_id=run_id,
                    reason_code="validation_blocked",
                    detail={"errors": [str(e) for e in persisted_vresult.errors[:5]]},
                )
                failed_vendors.append({
                    "vendor_name": vendor_name,
                    "stage": "persisted_validation",
                    "summary": persisted_vresult.summary(),
                    "reasons": _validation_feedback(
                        persisted_vresult, feedback_limit,
                    ) or [persisted_vresult.summary()],
                    "tokens_used": vendor_tokens,
                    "attempts_used": max_attempts,
                })
                validation_failures += 1
                failed += 1
                return
            if persisted_vresult.warnings:
                from ..visibility import record_synthesis_validation_results
                await record_synthesis_validation_results(
                    pool,
                    vendor_name=vendor_name,
                    as_of_date=today,
                    analysis_window_days=window_days,
                    schema_version=_SCHEMA_VERSION,
                    run_id=run_id,
                    attempt_no=attempt + 1,
                    results=_validation_rows(persisted_vresult),
                )
                synthesis["_validation_warnings"] = [
                    {
                        "path": w.path,
                        "code": w.code,
                        "message": w.message,
                    }
                    for w in persisted_vresult.warnings
                ]
                # Emit per-rule visibility events for queryable warning surface
                from ..visibility import emit_event
                for w in persisted_vresult.warnings[:10]:
                    await emit_event(
                        pool, stage="synthesis", event_type="validation_warning",
                        entity_type="vendor", entity_id=vendor_name,
                        summary=f"{w.code}: {w.message[:100]}",
                        severity="warning",
                        artifact_type="reasoning_synthesis",
                        run_id=run_id,
                        reason_code=w.code,
                        rule_code=w.code,
                        detail={"path": w.path, "message": w.message},
                    )
            else:
                from ..visibility import record_synthesis_validation_results
                await record_synthesis_validation_results(
                    pool,
                    vendor_name=vendor_name,
                    as_of_date=today,
                    analysis_window_days=window_days,
                    schema_version=_SCHEMA_VERSION,
                    run_id=run_id,
                    attempt_no=attempt + 1,
                    results=_validation_rows(persisted_vresult),
                )
                synthesis.pop("_validation_warnings", None)
            vresult = persisted_vresult

            # Quality gate: reject or flag weak synthesis before persistence
            quality_status, quality_reasons = _evaluate_synthesis_quality(synthesis)
            synthesis["_quality_status"] = quality_status
            synthesis["_quality_reasons"] = quality_reasons

            if quality_status == _QUALITY_STATUS_REJECTED:
                logger.warning(
                    "Synthesis quality gate REJECTED for %s: %s",
                    vendor_name, ", ".join(quality_reasons),
                )
                failed_vendors.append({
                    "vendor_name": vendor_name,
                    "reason": "quality_gate_rejected",
                    "quality_reasons": quality_reasons,
                })
                failed += 1
                return

            if quality_status == _QUALITY_STATUS_WEAK:
                logger.info(
                    "Synthesis quality gate WEAK for %s: %s",
                    vendor_name, ", ".join(quality_reasons),
                )

            try:
                await pool.execute(
                    """
                    INSERT INTO b2b_reasoning_synthesis
                        (vendor_name, as_of_date, analysis_window_days,
                         schema_version, evidence_hash, synthesis,
                         tokens_used, llm_model)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8)
                    ON CONFLICT (vendor_name, as_of_date,
                                 analysis_window_days, schema_version)
                    DO UPDATE SET
                        evidence_hash = EXCLUDED.evidence_hash,
                        synthesis = EXCLUDED.synthesis,
                        tokens_used = EXCLUDED.tokens_used,
                        llm_model = EXCLUDED.llm_model,
                        created_at = NOW()
                    """,
                    vendor_name,
                    today,
                    window_days,
                    _SCHEMA_VERSION,
                    ev_hash,
                    json.dumps(synthesis, default=str),
                    vendor_tokens,
                    getattr(llm, "model", getattr(llm, "model_id", "")),
                )
            except Exception as persist_exc:
                logger.warning(
                    "Reasoning synthesis failed for %s",
                    vendor_name, exc_info=True,
                )
                failed_vendors.append({
                    "vendor_name": vendor_name,
                    "stage": "persist",
                    "reasons": ["Failed to persist reasoning synthesis row"],
                    "tokens_used": vendor_tokens,
                    "attempts_used": max_attempts,
                })
                failed += 1
                from ..visibility import emit_event, record_attempt
                await record_attempt(
                    pool, artifact_type="reasoning_synthesis",
                    artifact_id=vendor_name,
                    run_id=run_id, attempt_no=max_attempts,
                    stage="persistence", status="failed",
                    failure_step="persist",
                    error_message=str(persist_exc)[:200],
                )
                await emit_event(
                    pool, stage="synthesis", event_type="persistence_failure",
                    entity_type="vendor", entity_id=vendor_name,
                    summary=f"Failed to persist synthesis for {vendor_name}: {persist_exc}",
                    severity="error", actionable=True,
                    artifact_type="reasoning_synthesis",
                    run_id=run_id,
                    reason_code="persistence_exception",
                )
                return

            succeeded += 1
            from ..visibility import record_attempt as _rec_ok
            await _rec_ok(
                pool, artifact_type="reasoning_synthesis",
                artifact_id=vendor_name,
                run_id=run_id, attempt_no=attempt + 1,
                stage="complete", status="succeeded",
                warning_count=len(vresult.warnings),
            )
            logger.info(
                "Reasoning synthesis v2: %s (%d tokens, %d warnings)",
                vendor_name, vendor_tokens, len(vresult.warnings),
            )

    await asyncio.gather(*[
        _reason_one(vn, layers, eh, decision)
        for vn, layers, eh, decision in vendors_to_reason
    ])

    vendor_elapsed = round(time.monotonic() - t0, 1)
    logger.info(
        "Reasoning synthesis v2 vendor phase: %d succeeded, %d failed "
        "(%d validation), %d skipped, %d tokens, %.1fs",
        succeeded, failed, validation_failures, skipped, total_tokens, vendor_elapsed,
    )

    # ------------------------------------------------------------------
    # Phase 2: Cross-vendor synthesis (battles, councils, asymmetry)
    # ------------------------------------------------------------------
    xv_succeeded = 0
    xv_failed = 0
    xv_tokens = 0
    xv_mirrored = 0
    xv_rejected_input_budget = 0

    run_cross_vendor = bool(cfg.cross_vendor_synthesis_enabled)
    scope_has_cross_vendor = bool(
        scope_pairwise_pairs or scope_category_names or scope_asymmetry_pairs
    )
    if filter_vendors and not force_cross_vendor and not scope_has_cross_vendor:
        run_cross_vendor = False

    if run_cross_vendor:
        try:
            xv_succeeded, xv_failed, xv_tokens, xv_mirrored, xv_rejected_input_budget = await _run_cross_vendor_synthesis(
                pool=pool,
                vendor_pools=vendor_pools,
                llm=llm,
                cfg=cfg,
                today=today,
                window_days=window_days,
                run_id=run_id,
                force=force_cross_vendor,
                scope_pairwise_pairs=scope_pairwise_pairs,
                scope_category_names=scope_category_names,
                scope_asymmetry_pairs=scope_asymmetry_pairs,
            )
            total_tokens += xv_tokens
        except Exception:
            logger.exception("Cross-vendor synthesis phase failed")

    elapsed = round(time.monotonic() - t0, 1)
    logger.info(
        "Reasoning synthesis v2 complete: vendors=%d/%d, xv=%d/%d, "
        "tokens=%d, %.1fs",
        succeeded, succeeded + failed,
        xv_succeeded, xv_succeeded + xv_failed,
        total_tokens, elapsed,
    )

    return await _finalize_scope_result({
        "vendors_total": len(vendor_pools),
        "vendors_reasoned": succeeded,
        "vendors_failed": failed,
        "vendors_validation_failures": validation_failures,
        "vendors_rejected_input_budget": input_budget_rejections,
        "cross_vendor_rejected_input_budget": xv_rejected_input_budget,
        "failed_vendors": failed_vendors,
        "vendors_skipped": skipped,
        "vendors_skipped_hash_reuse": rerun_reason_counts["hash_reuse"],
        "vendors_skipped_stale_reuse": rerun_reason_counts["stale_reused"],
        "vendors_rerun_hash_changed": rerun_reason_counts["hash_changed"],
        "vendors_rerun_prior_quality_weak": rerun_reason_counts["prior_quality_weak"],
        "vendors_rerun_missing_prior": rerun_reason_counts["missing_prior_row"],
        "vendors_rerun_missing_packet_artifacts": rerun_reason_counts["missing_packet_artifacts"],
        "vendors_rerun_missing_reference_ids": rerun_reason_counts["missing_reference_ids"],
        "vendors_forced": rerun_reason_counts["forced"],
        "vendors_payload_mode_full": payload_mode_full,
        "vendors_payload_mode_lean": payload_mode_lean,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "schema_version": _SCHEMA_VERSION,
        "witness_count": witness_count,
        "witness_vendor_rows": witness_vendor_rows,
        "cross_vendor_succeeded": xv_succeeded,
        "cross_vendor_failed": xv_failed,
        "cross_vendor_tokens": xv_tokens,
        "cross_vendor_mirrored": xv_mirrored,
    })


# ---------------------------------------------------------------------------
# Cross-vendor synthesis phase
# ---------------------------------------------------------------------------

_XV_SCHEMA_VERSION = "synthesis_v1"


async def _run_cross_vendor_synthesis(
    *,
    pool,
    vendor_pools: dict[str, dict],
    llm,
    cfg,
    today: date,
    window_days: int,
    run_id: str | None = None,
    force: bool = False,
    scope_pairwise_pairs: list[tuple[str, str]] | None = None,
    scope_category_names: list[str] | None = None,
    scope_asymmetry_pairs: list[tuple[str, str]] | None = None,
) -> tuple[int, int, int, int]:
    """Run cross-vendor synthesis: battles, councils, asymmetry.

    Returns (succeeded, failed, tokens_used, mirrored_to_legacy, input_budget_rejections).
    """
    import re

    from ...reasoning.cross_vendor_selection import (
        select_asymmetry_pairs,
        select_battles,
        select_categories,
    )
    from ...reasoning.single_pass_prompts.category_council_synthesis import (
        CATEGORY_COUNCIL_SYNTHESIS_PROMPT,
    )
    from ...reasoning.single_pass_prompts.cross_vendor_battle_synthesis import (
        CROSS_VENDOR_BATTLE_SYNTHESIS_PROMPT,
    )
    from ...reasoning.single_pass_prompts.resource_asymmetry_synthesis import (
        RESOURCE_ASYMMETRY_SYNTHESIS_PROMPT,
    )
    from ...services.protocols import Message
    from ._b2b_cross_vendor_synthesis import (
        _sorted_vendors,
        attach_cross_vendor_citation_registry,
        build_category_council_packet,
        build_pairwise_battle_packet,
        build_resource_asymmetry_packet,
        compute_cross_vendor_evidence_hash,
        materialize_cross_vendor_reference_ids,
        normalize_cross_vendor_contract,
        prompt_compact_cross_vendor_packet,
        to_legacy_cross_vendor_conclusion,
    )
    from ..visibility import emit_event, record_attempt

    def _cross_vendor_artifact_id(
        analysis_type: str,
        vendors: list[str],
        category: str | None,
    ) -> str:
        if analysis_type == "category_council":
            category_key = str(category or "unknown").strip().lower()
            return f"{analysis_type}:{category_key}"
        vendor_key = "|".join(
            sorted(str(vendor).strip().lower() for vendor in vendors if str(vendor).strip()),
        )
        return f"{analysis_type}:{vendor_key or 'unknown'}"

    def _selector_evidence_from_layers(layers: dict[str, Any]) -> dict[str, Any]:
        evidence_vault = layers.get("evidence_vault") or {}
        segment = layers.get("segment") or {}
        accounts = layers.get("accounts") or {}
        displacement = layers.get("displacement") or []
        category = layers.get("category") or {}

        product_category = str(
            category.get("category")
            or evidence_vault.get("product_category")
            or ""
        ).strip()

        pain_categories: list[dict[str, Any]] = []
        seen_pains: set[str] = set()
        for item in segment.get("affected_roles") or []:
            if not isinstance(item, dict):
                continue
            category_name = str(item.get("top_pain") or "").strip()
            if not category_name or category_name in seen_pains:
                continue
            seen_pains.add(category_name)
            pain_categories.append({
                "category": category_name,
                "count": int(item.get("review_count") or 0),
            })

        competitor_rows: list[dict[str, Any]] = []
        for item in displacement:
            if not isinstance(item, dict):
                continue
            competitor = str(item.get("to_vendor") or "").strip()
            if not competitor:
                continue
            mentions = 0
            flow_summary = item.get("flow_summary") or {}
            edge_metrics = item.get("edge_metrics") or {}
            try:
                mentions = int(
                    flow_summary.get("total_flow_mentions")
                    or edge_metrics.get("mention_count")
                    or 0,
                )
            except (TypeError, ValueError):
                mentions = 0
            competitor_rows.append({
                "name": competitor,
                "mentions": mentions,
            })

        buyer_role_counts: dict[str, int] = {}
        for item in accounts.get("accounts") or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("buyer_role") or "").strip().lower()
            if not role or role == "unknown":
                continue
            buyer_role_counts[role] = buyer_role_counts.get(role, 0) + 1
        for item in segment.get("affected_roles") or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role_type") or "").strip().lower()
            if not role or role == "unknown":
                continue
            buyer_role_counts[role] = buyer_role_counts.get(role, 0) + int(
                item.get("review_count") or 0,
            )

        top_use_cases: list[str] = []
        for item in segment.get("top_use_cases_under_pressure") or []:
            if not isinstance(item, dict):
                continue
            label = str(
                item.get("use_case")
                or item.get("name")
                or item.get("label")
                or ""
            ).strip()
            if label and label not in top_use_cases:
                top_use_cases.append(label)

        evidence: dict[str, Any] = {
            "product_category": product_category,
            "total_reviews": evidence_vault.get("metric_snapshot", {}).get("total_reviews")
            or 0,
            "pain_categories": pain_categories[:5],
            "competitors": competitor_rows[:10],
            "top_use_cases": top_use_cases[:5],
        }
        if buyer_role_counts:
            evidence["buyer_authority"] = {"role_types": buyer_role_counts}
        return evidence

    # Reconstruct inputs for selectors from pool layers
    # Build selector evidence from the persisted pool layers.
    vendor_scores: list[dict[str, Any]] = []
    evidence_lookup: dict[str, dict[str, Any]] = {}
    product_profiles: dict[str, dict[str, Any]] = {}

    for vname, layers in vendor_pools.items():
        selector_evidence = _selector_evidence_from_layers(layers)
        evidence_lookup[vname] = selector_evidence
        accounts_summary = (layers.get("accounts") or {}).get("summary") or {}
        metric_snapshot = (layers.get("evidence_vault") or {}).get("metric_snapshot") or {}
        vendor_scores.append({
            "vendor_name": vname,
            "avg_urgency": metric_snapshot.get("avg_urgency_score") or 0,
            "total_reviews": metric_snapshot.get("total_reviews") or 0,
            "product_category": selector_evidence.get("product_category") or "",
            "high_intent_count": accounts_summary.get("high_intent_count") or 0,
        })

    # Load product profiles
    try:
        profile_rows = await pool.fetch(
            "SELECT vendor_name, product_category, strengths, weaknesses, "
            "top_integrations, primary_use_cases, typical_company_size, typical_industries "
            "FROM b2b_product_profiles"
        )
        for r in profile_rows:
            product_profiles[r["vendor_name"]] = dict(r)
    except Exception:
        logger.debug("Product profile fetch failed for xv synthesis")

    vendor_reference_lookup: dict[str, dict[str, Any]] = {}
    try:
        reference_rows = await pool.fetch(
            """
            SELECT DISTINCT ON (vendor_name)
                   vendor_name, synthesis
            FROM b2b_reasoning_synthesis
            WHERE as_of_date <= $1
              AND analysis_window_days = $2
            ORDER BY vendor_name, as_of_date DESC, created_at DESC
            """,
            today,
            window_days,
        )
        for row in reference_rows:
            synthesis = row["synthesis"]
            if isinstance(synthesis, str):
                try:
                    synthesis = json.loads(synthesis)
                except Exception:
                    synthesis = {}
            if not isinstance(synthesis, dict):
                continue
            vendor_reference_lookup[row["vendor_name"]] = (
                synthesis.get("reference_ids") or {}
            )
    except Exception:
        logger.debug("Vendor synthesis reference lookup failed for xv synthesis")

    # Build displacement map
    try:
        disp_rows = await pool.fetch(
            "SELECT from_vendor, to_vendor, mention_count, primary_driver, "
            "signal_strength, velocity_7d "
            "FROM b2b_displacement_edges "
            "WHERE computed_date = (SELECT MAX(computed_date) FROM b2b_displacement_edges)"
        )
        displacement_edges = [dict(r) for r in disp_rows]
    except Exception:
        logger.debug("Displacement edge fetch failed for xv synthesis")
        displacement_edges = []

    # Build ecosystem evidence for category selection
    ecosystem_evidence: dict[str, dict[str, Any]] = {}
    try:
        eco_rows = await pool.fetch(
            "SELECT category, hhi, market_structure, displacement_intensity, "
            "dominant_archetype, archetype_distribution "
            "FROM b2b_category_dynamics "
            "WHERE as_of_date = (SELECT MAX(as_of_date) FROM b2b_category_dynamics)"
        )
        for r in eco_rows:
            ecosystem_evidence[r["category"]] = dict(r)
    except Exception:
        logger.debug("Category dynamics flat fetch failed for xv synthesis")
        try:
            eco_rows = await pool.fetch(
                "SELECT category, dynamics "
                "FROM b2b_category_dynamics "
                "WHERE as_of_date = (SELECT MAX(as_of_date) FROM b2b_category_dynamics)"
            )
            for r in eco_rows:
                raw = r["dynamics"]
                dynamics = raw if isinstance(raw, dict) else {}
                if isinstance(raw, str):
                    try:
                        dynamics = json.loads(raw)
                    except Exception:
                        dynamics = {}
                market_regime = dynamics.get("market_regime")
                market_regime = market_regime if isinstance(market_regime, dict) else {}
                displacement_intensity = (
                    dynamics.get("displacement_intensity")
                    or dynamics.get("displacement_flow_count")
                    or 0
                )
                try:
                    displacement_intensity = float(displacement_intensity)
                except (TypeError, ValueError):
                    displacement_intensity = 0.0
                ecosystem_evidence[r["category"]] = {
                    "category": r["category"],
                    "hhi": dynamics.get("hhi"),
                    "market_structure": (
                        dynamics.get("market_structure")
                        or market_regime.get("regime_type")
                    ),
                    "displacement_intensity": displacement_intensity,
                    "dominant_archetype": dynamics.get("dominant_archetype"),
                    "archetype_distribution": (
                        dynamics.get("archetype_distribution") or {}
                    ),
                }
        except Exception:
            logger.debug("Category dynamics JSON fetch failed for xv synthesis")

    # Build vendor membership map for category selection.
    # select_categories() only uses the mapping keys to count vendors with
    # evidence in each category; it does not consume legacy archetype values.
    category_vendor_lookup: dict[str, dict[str, Any]] = {}
    for vname, ev in evidence_lookup.items():
        if (ev or {}).get("product_category"):
            category_vendor_lookup[vname] = {}

    scoped_pairwise = scope_pairwise_pairs or []
    scoped_categories = _metadata_text_list(scope_category_names or [])
    scoped_asymmetry = scope_asymmetry_pairs or []

    if scoped_pairwise or scoped_categories or scoped_asymmetry:
        edge_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for edge in displacement_edges:
            pair = tuple(
                sorted(
                    [
                        str(edge.get("from_vendor") or "").strip().lower(),
                        str(edge.get("to_vendor") or "").strip().lower(),
                    ]
                )
            )
            if len(pair) == 2 and pair not in edge_lookup:
                edge_lookup[pair] = edge
        battles = []
        for vendor_a, vendor_b in scoped_pairwise:
            if vendor_a not in vendor_pools or vendor_b not in vendor_pools:
                continue
            edge = edge_lookup.get(tuple(sorted((vendor_a.lower(), vendor_b.lower()))), {})
            battles.append((vendor_a, vendor_b, edge))
        categories = []
        for category_name in scoped_categories:
            ecosystem = ecosystem_evidence.get(category_name) or ecosystem_evidence.get(str(category_name or "").strip())
            if ecosystem is None:
                for key, value in ecosystem_evidence.items():
                    if str(key or "").strip().lower() == str(category_name or "").strip().lower():
                        ecosystem = value
                        break
            if ecosystem is not None:
                categories.append((category_name, ecosystem))
        asymmetry_pairs = [
            (vendor_a, vendor_b)
            for vendor_a, vendor_b in scoped_asymmetry
            if vendor_a in vendor_pools and vendor_b in vendor_pools
        ]
    else:
        battles = await select_battles(
            pool,
            displacement_edges,
            evidence_lookup,
            product_profiles=product_profiles,
            max_battles=cfg.cross_vendor_max_battles,
            min_context_score=cfg.cross_vendor_battle_min_context_score,
        )
        categories = select_categories(
            ecosystem_evidence,
            category_vendor_lookup,
            evidence_lookup,
            min_vendors=cfg.cross_vendor_category_min_vendors,
            min_context_vendors=cfg.cross_vendor_category_min_context_vendors,
            min_displacement_intensity=cfg.cross_vendor_category_min_displacement_intensity,
            max_categories=cfg.cross_vendor_max_categories,
        )
        asymmetry_pairs = await select_asymmetry_pairs(
            vendor_scores,
            evidence_lookup,
            product_profiles,
            max_pairs=cfg.cross_vendor_max_asymmetry,
            pressure_delta_max=cfg.cross_vendor_asymmetry_pressure_delta_max,
            review_ratio_min=cfg.cross_vendor_asymmetry_review_ratio_min,
            segment_divergence_bonus=cfg.cross_vendor_asymmetry_segment_divergence_bonus,
            min_divergence_score=cfg.cross_vendor_asymmetry_min_divergence_score,
            min_context_score=cfg.cross_vendor_asymmetry_min_context_score,
        )

    logger.info(
        "Cross-vendor synthesis targets: %d battles, %d councils, %d asymmetries",
        len(battles), len(categories), len(asymmetry_pairs),
    )

    # --- Build packets ---
    work_items: list[tuple[str, str, list[str], str | None, dict[str, Any]]] = []

    for vendor_a, vendor_b, edge in battles:
        packet = build_pairwise_battle_packet(
            vendor_a, vendor_b, edge, vendor_pools, product_profiles,
        )
        work_items.append((
            "pairwise_battle",
            CROSS_VENDOR_BATTLE_SYNTHESIS_PROMPT,
            _sorted_vendors(vendor_a, vendor_b),
            None,
            attach_cross_vendor_citation_registry(
                packet,
                analysis_type="pairwise_battle",
                vendors=_sorted_vendors(vendor_a, vendor_b),
                category=None,
                vendor_reference_lookup=vendor_reference_lookup,
            ),
        ))

    for cat, eco_ev in categories:
        packet = build_category_council_packet(
            cat,
            eco_ev,
            vendor_pools,
            product_profiles,
            displacement_edges,
            vendor_summary_limit=max(
                1,
                int(getattr(cfg, "cross_vendor_category_vendor_summary_limit", 8)),
            ),
            flow_limit=max(
                0,
                int(getattr(cfg, "cross_vendor_category_flow_limit", 10)),
            ),
        )
        cat_lower = (cat or "").strip().lower()
        cat_vendors = [
            vn for vn, pp in product_profiles.items()
            if (pp.get("product_category") or "").strip().lower() == cat_lower
        ]
        work_items.append((
            "category_council",
            CATEGORY_COUNCIL_SYNTHESIS_PROMPT,
            sorted(cat_vendors),
            cat,
            attach_cross_vendor_citation_registry(
                packet,
                analysis_type="category_council",
                vendors=sorted(cat_vendors),
                category=cat,
                vendor_reference_lookup=vendor_reference_lookup,
            ),
        ))

    for vendor_a, vendor_b in asymmetry_pairs:
        packet = build_resource_asymmetry_packet(
            vendor_a, vendor_b, vendor_pools, product_profiles,
        )
        work_items.append((
            "resource_asymmetry",
            RESOURCE_ASYMMETRY_SYNTHESIS_PROMPT,
            _sorted_vendors(vendor_a, vendor_b),
            None,
            attach_cross_vendor_citation_registry(
                packet,
                analysis_type="resource_asymmetry",
                vendors=_sorted_vendors(vendor_a, vendor_b),
                category=None,
                vendor_reference_lookup=vendor_reference_lookup,
            ),
        ))

    if not work_items:
        logger.info("No cross-vendor synthesis targets selected")
        return 0, 0, 0, 0

    # --- Check existing hashes to skip unchanged ---
    existing_xv = await pool.fetch(
        """
        SELECT analysis_type, vendors, category, evidence_hash
        FROM b2b_cross_vendor_reasoning_synthesis
        WHERE as_of_date = $1 AND analysis_window_days = $2
          AND schema_version = $3
        """,
        today, window_days, _XV_SCHEMA_VERSION,
    )
    existing_xv_hashes: dict[str, str] = {}
    for r in existing_xv:
        v_key = "|".join(sorted(r["vendors"])) if r["vendors"] else ""
        key = f"{r['analysis_type']}:{v_key}:{r['category'] or ''}"
        existing_xv_hashes[key] = r["evidence_hash"]

    # --- Process each work item ---
    from ...pipelines.llm import parse_json_response

    xv_sem = asyncio.Semaphore(cfg.cross_vendor_synthesis_concurrency)
    xv_max_attempts = max(1, cfg.cross_vendor_synthesis_attempts)
    xv_llm_timeout_seconds = max(
        1.0,
        float(getattr(cfg, "reasoning_synthesis_timeout_seconds", 180.0)),
    )
    _default_max_tokens = 16384
    xv_llm_max_tokens = max(
        256,
        min(
            int(getattr(cfg, "reasoning_synthesis_max_tokens", _default_max_tokens)),
            _default_max_tokens,
        ),
    )
    xv_llm_max_input_tokens = max(
        512,
        int(getattr(cfg, "cross_vendor_llm_max_input_tokens", 12000)),
    )
    xv_llm_temperature = float(
        getattr(cfg, "reasoning_synthesis_temperature", 0.3),
    )
    _succeeded = 0
    _failed = 0
    _tokens = 0
    _mirrored = 0
    _input_budget_rejections = 0
    llm_model_name = getattr(llm, "model", getattr(llm, "model_id", ""))

    async def _xv_one(
        analysis_type: str,
        prompt: str,
        vendors: list[str],
        category: str | None,
        packet: dict[str, Any],
    ) -> None:
        nonlocal _succeeded, _failed, _tokens, _mirrored, _input_budget_rejections

        ev_hash = compute_cross_vendor_evidence_hash(packet)
        v_key = "|".join(sorted(vendors))
        cache_key = f"{analysis_type}:{v_key}:{category or ''}"
        artifact_id = _cross_vendor_artifact_id(analysis_type, vendors, category)
        if not force and existing_xv_hashes.get(cache_key) == ev_hash:
            return  # unchanged

        async with xv_sem:
            payload = json.dumps(
                prompt_compact_cross_vendor_packet(packet),
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            )
            estimated_input_tokens = _approx_prompt_input_tokens(prompt, payload)
            if estimated_input_tokens > xv_llm_max_input_tokens:
                _failed += 1
                _input_budget_rejections += 1
                logger.warning(
                    "Cross-vendor synthesis rejected for %s %s: estimated input %d exceeds cap %d",
                    analysis_type,
                    vendors,
                    estimated_input_tokens,
                    xv_llm_max_input_tokens,
                )
                await record_attempt(
                    pool,
                    artifact_type="cross_vendor_reasoning",
                    artifact_id=artifact_id,
                    run_id=run_id,
                    attempt_no=1,
                    stage="generation",
                    status="rejected",
                    blocker_count=1,
                    blocking_issues=[
                        (
                            "input token budget exceeded: "
                            f"estimated_input_tokens={estimated_input_tokens}, "
                            f"cap={xv_llm_max_input_tokens}"
                        ),
                    ],
                    failure_step="input_budget",
                    error_message=(
                        "Cross-vendor prompt exceeded the configured input token cap"
                    ),
                )
                await emit_event(
                    pool,
                    stage="synthesis",
                    event_type="input_budget_rejected",
                    entity_type="cross_vendor",
                    entity_id=artifact_id,
                    summary=(
                        "Cross-vendor reasoning prompt exceeded the configured input token cap"
                    ),
                    severity="warning",
                    actionable=True,
                    artifact_type="cross_vendor_reasoning",
                    run_id=run_id,
                    reason_code="input_budget",
                    detail={
                        "estimated_input_tokens": estimated_input_tokens,
                        "cap": xv_llm_max_input_tokens,
                        "analysis_type": analysis_type,
                        "vendors": vendors,
                        "category": category,
                    },
                )
                return
            synthesis: dict[str, Any] | None = None
            item_tokens = 0
            last_failure_reasons: list[str] = []
            last_failure_step = "llm_response"
            terminal_attempt_recorded = False
            attempt_no = 1

            for attempt in range(xv_max_attempts):
                attempt_no = attempt + 1
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=payload),
                ]
                try:
                    call_started = time.monotonic()
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            llm.chat,
                            messages=messages,
                            max_tokens=xv_llm_max_tokens,
                            temperature=xv_llm_temperature,
                            response_format={"type": "json_object"},
                        ),
                        timeout=xv_llm_timeout_seconds,
                    )
                    _trace_reasoning_result(
                        "task.b2b_reasoning_synthesis.cross_vendor",
                        llm=llm,
                        messages=messages,
                        result=result,
                        metadata={
                            "run_id": run_id,
                            "reasoning_mode": "cross_vendor",
                            "analysis_type": analysis_type,
                            "vendor_name": vendors[0] if len(vendors) == 1 else None,
                            "vendors": vendors,
                            "artifact_id": artifact_id,
                            "attempt_no": attempt + 1,
                        },
                        duration_ms=(time.monotonic() - call_started) * 1000,
                    )
                    text = result.get("response", "").strip()
                    usage = result.get("usage", {})
                    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    item_tokens += tokens
                    _tokens += tokens

                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                    parsed = parse_json_response(text, recover_truncated=True)
                    if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
                        synthesis = normalize_cross_vendor_contract(parsed, analysis_type)
                        synthesis = materialize_cross_vendor_reference_ids(
                            synthesis, packet,
                        )
                        break
                    last_failure_reasons = ["LLM did not return valid JSON object"]
                    last_failure_step = "llm_response"
                except asyncio.TimeoutError:
                    last_failure_reasons = [
                        "TimeoutError: cross-vendor reasoning LLM call exceeded "
                        f"{xv_llm_timeout_seconds:.1f}s",
                    ]
                    last_failure_step = "timeout"
                    if attempt + 1 >= xv_max_attempts:
                        logger.warning(
                            "Cross-vendor synthesis timed out: %s %s after %.1fs",
                            analysis_type,
                            vendors,
                            xv_llm_timeout_seconds,
                        )
                        terminal_attempt_recorded = True
                        await record_attempt(
                            pool,
                            artifact_type="cross_vendor_reasoning",
                            artifact_id=artifact_id,
                            run_id=run_id,
                            attempt_no=attempt_no,
                            stage="llm_call",
                            status="failed",
                            failure_step="timeout",
                            error_message=(
                                f"LLM call exceeded {xv_llm_timeout_seconds:.1f}s"
                            ),
                        )
                except Exception:
                    last_failure_reasons = [
                        "Cross-vendor reasoning LLM call raised an exception",
                    ]
                    last_failure_step = "llm_exception"
                    if attempt + 1 >= xv_max_attempts:
                        logger.warning(
                            "Cross-vendor synthesis failed: %s %s",
                            analysis_type, vendors, exc_info=True,
                        )
                        terminal_attempt_recorded = True
                        await record_attempt(
                            pool,
                            artifact_type="cross_vendor_reasoning",
                            artifact_id=artifact_id,
                            run_id=run_id,
                            attempt_no=attempt_no,
                            stage="llm_call",
                            status="failed",
                            failure_step="llm_exception",
                            error_message=last_failure_reasons[0],
                        )

            if synthesis is None:
                _failed += 1
                if not terminal_attempt_recorded:
                    await record_attempt(
                        pool,
                        artifact_type="cross_vendor_reasoning",
                        artifact_id=artifact_id,
                        run_id=run_id,
                        attempt_no=attempt_no,
                        stage="generation",
                        status="rejected",
                        blocker_count=len(last_failure_reasons),
                        blocking_issues=last_failure_reasons[:5],
                        failure_step=last_failure_step,
                        error_message="; ".join(last_failure_reasons[:2])[:200],
                    )
                return

            # Persist to canonical table (upsert -- reruns update with fresh synthesis).
            # Uses ON CONFLICT on the named partial unique index so PostgreSQL
            # matches the correct predicate automatically.
            try:
                if analysis_type == "category_council":
                    await pool.execute(
                        """
                        INSERT INTO b2b_cross_vendor_reasoning_synthesis
                            (analysis_type, vendors, category, as_of_date,
                             analysis_window_days, schema_version, evidence_hash,
                             synthesis, tokens_used, llm_model)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10)
                        ON CONFLICT (analysis_type, category, as_of_date,
                                     analysis_window_days, schema_version)
                            WHERE analysis_type = 'category_council'
                        DO UPDATE SET
                            evidence_hash = EXCLUDED.evidence_hash,
                            synthesis = EXCLUDED.synthesis,
                            tokens_used = EXCLUDED.tokens_used,
                            llm_model = EXCLUDED.llm_model,
                            vendors = EXCLUDED.vendors,
                            created_at = NOW()
                        """,
                        analysis_type, vendors, category, today,
                        window_days, _XV_SCHEMA_VERSION, ev_hash,
                        json.dumps(synthesis, default=str),
                        item_tokens, llm_model_name,
                    )
                else:
                    await pool.execute(
                        """
                        INSERT INTO b2b_cross_vendor_reasoning_synthesis
                            (analysis_type, vendors, category, as_of_date,
                             analysis_window_days, schema_version, evidence_hash,
                             synthesis, tokens_used, llm_model)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10)
                        ON CONFLICT (analysis_type, vendors, as_of_date,
                                     analysis_window_days, schema_version)
                            WHERE analysis_type IN ('pairwise_battle', 'resource_asymmetry')
                        DO UPDATE SET
                            evidence_hash = EXCLUDED.evidence_hash,
                            synthesis = EXCLUDED.synthesis,
                            tokens_used = EXCLUDED.tokens_used,
                            llm_model = EXCLUDED.llm_model,
                            category = EXCLUDED.category,
                            created_at = NOW()
                        """,
                        analysis_type, vendors, category, today,
                        window_days, _XV_SCHEMA_VERSION, ev_hash,
                        json.dumps(synthesis, default=str),
                        item_tokens, llm_model_name,
                    )
            except Exception:
                logger.warning(
                    "Failed to persist xv synthesis: %s %s",
                    analysis_type, vendors, exc_info=True,
                )
                _failed += 1
                await record_attempt(
                    pool,
                    artifact_type="cross_vendor_reasoning",
                    artifact_id=artifact_id,
                    run_id=run_id,
                    attempt_no=attempt_no,
                    stage="persistence",
                    status="failed",
                    failure_step="persist",
                    error_message="Failed to persist cross-vendor synthesis row",
                )
                return

            # Mirror into legacy b2b_cross_vendor_conclusions (idempotent: delete-then-insert)
            try:
                legacy = to_legacy_cross_vendor_conclusion(
                    synthesis, analysis_type, vendors,
                    category=category,
                    evidence_hash=ev_hash,
                    tokens_used=item_tokens,
                )
                # Delete any existing synthesis-produced mirror row for this
                # (type, vendors, date) combo, then insert fresh.  The legacy
                # table has no unique constraint, so upsert is not possible.
                await pool.execute(
                    """
                    DELETE FROM b2b_cross_vendor_conclusions
                    WHERE analysis_type = $1
                      AND vendors = $2::text[]
                      AND computed_date = $3
                      AND evidence_hash LIKE 'synth_%'
                    """,
                    analysis_type,
                    vendors,
                    today,
                )
                await pool.execute(
                    """
                    INSERT INTO b2b_cross_vendor_conclusions
                        (analysis_type, vendors, category, conclusion,
                         confidence, evidence_hash, tokens_used, cached,
                         computed_date)
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9)
                    """,
                    legacy["analysis_type"],
                    legacy["vendors"],
                    legacy["category"],
                    json.dumps(legacy["conclusion"], default=str),
                    legacy["confidence"],
                    f"synth_{ev_hash}",
                    legacy["tokens_used"],
                    legacy["cached"],
                    today,
                )
                _mirrored += 1
            except Exception:
                logger.debug(
                    "Legacy mirror failed for %s %s", analysis_type, vendors,
                    exc_info=True,
                )

            _succeeded += 1
            await record_attempt(
                pool,
                artifact_type="cross_vendor_reasoning",
                artifact_id=artifact_id,
                run_id=run_id,
                attempt_no=attempt_no,
                stage="complete",
                status="succeeded",
            )
            logger.info(
                "Cross-vendor synthesis: %s %s (%d tokens)",
                analysis_type, vendors, item_tokens,
            )

    await asyncio.gather(*[
        _xv_one(at, prompt, vendors, cat, packet)
        for at, prompt, vendors, cat, packet in work_items
    ])

    logger.info(
        "Cross-vendor synthesis complete: %d succeeded, %d failed, "
        "%d mirrored, %d tokens",
        _succeeded, _failed, _mirrored, _tokens,
    )
    return _succeeded, _failed, _tokens, _mirrored, _input_budget_rejections
