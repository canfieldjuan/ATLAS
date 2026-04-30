from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...autonomous.tasks._b2b_batch_utils import reconcile_existing_batch_artifacts
from .enrichment_stage_runs import get_stage_run, mark_stage_run


@dataclass(frozen=True)
class StageExecutionDecision:
    action: str
    stage_row: dict[str, Any] | None
    parsed_result: dict[str, Any] | None
    batch_result: dict[str, Any] | None


@dataclass(frozen=True)
class StageBatchExecution:
    execution: Any
    metrics: dict[str, int]


@dataclass(frozen=True)
class StageBatchOutcome:
    row: dict[str, Any]
    parsed_result: dict[str, Any] | None
    usage: dict[str, int] | None
    custom_id: str
    success: bool
    error_text: str | None


@dataclass(frozen=True)
class StageDecisionApplication:
    action: str
    parsed_result: dict[str, Any] | None
    usage: dict[str, int] | None
    cache_hit: bool
    model: str | None
    batch_id: str | None
    custom_id: str | None
    response_text: str | None


@dataclass(frozen=True)
class ReviewTransitionResult:
    action: str
    row_result: dict[str, Any]


async def prepare_stage_execution(
    *,
    pool: Any,
    llm: Any | None,
    task_name: str | None,
    artifact_type: str | None,
    artifact_id: str | None,
    review_id: Any,
    stage_id: str,
    work_fingerprint: str,
    request_fingerprint: str,
    parse_response_text: Any,
    defer_on_submitted: bool = False,
    reconcile_batch: bool = False,
) -> StageExecutionDecision:
    stage_row = await get_stage_run(
        pool,
        review_id=review_id,
        stage_id=stage_id,
        work_fingerprint=work_fingerprint,
    )
    parsed_result = parse_response_text(stage_row)
    stage_state = str((stage_row or {}).get("state") or "").strip().lower()
    if parsed_result is not None and stage_state == "succeeded":
        return StageExecutionDecision("reuse_stage", stage_row, parsed_result, None)

    if reconcile_batch and task_name and artifact_type and artifact_id:
        batch_results = await reconcile_existing_batch_artifacts(
            pool=pool,
            llm=llm,
            task_name=task_name,
            artifact_type=artifact_type,
            artifact_ids=[str(artifact_id)],
            expected_request_fingerprints={str(artifact_id): str(request_fingerprint)},
        )
        batch_result = batch_results.get(str(artifact_id))
        batch_state = str((batch_result or {}).get("state") or "").strip().lower()
        if batch_state == "succeeded":
            parsed_batch_result = parse_response_text(
                {"response_text": str(batch_result.get("response_text") or "")}
            )
            if parsed_batch_result is not None:
                return StageExecutionDecision(
                    "reuse_batch_result",
                    stage_row,
                    parsed_batch_result,
                    batch_result,
                )
        if batch_state == "pending":
            return StageExecutionDecision("defer_pending_batch", stage_row, None, batch_result)

    if defer_on_submitted and stage_state == "submitted":
        return StageExecutionDecision("defer_submitted_stage", stage_row, None, None)
    return StageExecutionDecision("execute", stage_row, None, None)


async def apply_stage_decision(
    *,
    pool: Any,
    decision: StageExecutionDecision,
    review_id: Any,
    stage_id: str,
    work_fingerprint: str,
    tier: int,
    usage_from_stage_row: Any,
    pending_metadata: dict[str, Any],
    success_metadata: dict[str, Any],
    stage_usage_snapshot: Any,
) -> StageDecisionApplication | None:
    if decision.action == "reuse_stage":
        stage_row = decision.stage_row or {}
        return StageDecisionApplication(
            action=decision.action,
            parsed_result=decision.parsed_result,
            usage=usage_from_stage_row(stage_row, tier=tier),
            cache_hit=str(stage_row.get("result_source") or "").strip().lower() == "exact_cache",
            model=str(stage_row.get("model") or "") or None,
            batch_id=str(stage_row.get("batch_id") or "") or None,
            custom_id=str(stage_row.get("batch_custom_id") or "") or None,
            response_text=str(stage_row.get("response_text") or "") or None,
        )

    if decision.action == "defer_submitted_stage":
        stage_row = decision.stage_row or {}
        return StageDecisionApplication(
            action=decision.action,
            parsed_result=None,
            usage=None,
            cache_hit=False,
            model=str(stage_row.get("model") or "") or None,
            batch_id=str(stage_row.get("batch_id") or "") or None,
            custom_id=str(stage_row.get("batch_custom_id") or "") or None,
            response_text=str(stage_row.get("response_text") or "") or None,
        )

    if decision.action not in {"reuse_batch_result", "defer_pending_batch"}:
        return None

    batch_result = decision.batch_result or {}
    batch_id = batch_result.get("batch_id")
    custom_id = str(batch_result.get("custom_id") or "") or None
    cache_hit = bool(batch_result.get("cached"))
    response_text = str(batch_result.get("response_text") or "") or None

    if decision.action == "reuse_batch_result":
        usage = stage_usage_snapshot(
            tier=tier,
            cache_hit=cache_hit,
            generated=not cache_hit,
        )
        await mark_stage_run(
            pool,
            review_id=review_id,
            stage_id=stage_id,
            work_fingerprint=str(work_fingerprint),
            state="succeeded",
            result_source="exact_cache" if cache_hit else "batch_reuse",
            backend="anthropic_batch",
            batch_id=batch_id,
            batch_custom_id=custom_id,
            response_text=response_text,
            metadata=success_metadata,
            completed=True,
        )
        return StageDecisionApplication(
            action=decision.action,
            parsed_result=decision.parsed_result,
            usage=usage,
            cache_hit=cache_hit,
            model=None,
            batch_id=str(batch_id) if batch_id is not None else None,
            custom_id=custom_id,
            response_text=response_text,
        )

    await mark_stage_run(
        pool,
        review_id=review_id,
        stage_id=stage_id,
        work_fingerprint=str(work_fingerprint),
        state="submitted",
        backend="anthropic_batch",
        batch_id=batch_id,
        batch_custom_id=custom_id,
        metadata=pending_metadata,
    )
    return StageDecisionApplication(
        action=decision.action,
        parsed_result=None,
        usage=None,
        cache_hit=cache_hit,
        model=None,
        batch_id=str(batch_id) if batch_id is not None else None,
        custom_id=custom_id,
        response_text=response_text,
    )


async def defer_review_transition(
    *,
    row: dict[str, Any],
    tier: str,
    custom_id: str | None,
    usage: dict[str, int] | None,
    defer_review: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "tier": str(tier),
        "custom_id": str(custom_id or ""),
    }
    if usage is not None:
        kwargs["usage"] = usage
    return await defer_review(row, **kwargs)


async def persist_review_transition(
    *,
    row: dict[str, Any],
    tier1_result: dict[str, Any],
    tier2_result: dict[str, Any] | None,
    model_id: str,
    usage: dict[str, int],
    merge_results: Any,
    persist_review: Any,
) -> dict[str, Any]:
    return await persist_review(
        row,
        merge_results(tier1_result, tier2_result),
        model_id=model_id,
        usage=usage,
    )


async def apply_review_stage_transition(
    *,
    applied: StageDecisionApplication | None,
    row: dict[str, Any],
    tier: str,
    usage: dict[str, int],
    tier1_result: dict[str, Any],
    model_id: str,
    accumulate_usage: Any,
    merge_results: Any,
    persist_review: Any,
    defer_review: Any,
) -> ReviewTransitionResult | None:
    if applied is None:
        return None
    if applied.action in {"defer_submitted_stage", "defer_pending_batch"}:
        row_result = await defer_review_transition(
            row=row,
            tier=tier,
            custom_id=applied.custom_id,
            usage=usage,
            defer_review=defer_review,
        )
        return ReviewTransitionResult(action=applied.action, row_result=row_result)
    if applied.action in {"reuse_stage", "reuse_batch_result"}:
        if applied.usage is not None:
            accumulate_usage(usage, applied.usage)
        row_result = await persist_review_transition(
            row=row,
            tier1_result=tier1_result,
            tier2_result=applied.parsed_result,
            model_id=model_id,
            usage=usage,
            merge_results=merge_results,
            persist_review=persist_review,
        )
        return ReviewTransitionResult(action=applied.action, row_result=row_result)
    return None


async def submit_stage_batch(
    *,
    run_batch: Any,
    llm: Any,
    stage_id: str,
    task_name: str,
    items: list[Any],
    run_id: str | None,
    min_batch_size: int,
    batch_metadata: dict[str, Any],
    pool: Any,
    entries: list[dict[str, Any]],
    custom_id_for_entry: Any,
    pending_metadata: dict[str, Any],
) -> StageBatchExecution:
    execution = await run_batch(
        llm=llm,
        stage_id=stage_id,
        task_name=task_name,
        items=items,
        run_id=run_id,
        min_batch_size=min_batch_size,
        batch_metadata=batch_metadata,
        pool=pool,
    )
    metrics = {
        "jobs": 1 if getattr(execution, "provider_batch_id", None) else 0,
        "submitted_items": int(getattr(execution, "submitted_items", 0) or 0),
        "cache_prefiltered_items": int(getattr(execution, "cache_prefiltered_items", 0) or 0),
        "fallback_single_call_items": int(getattr(execution, "fallback_single_call_items", 0) or 0),
        "completed_items": int(getattr(execution, "completed_items", 0) or 0),
        "failed_items": int(getattr(execution, "failed_items", 0) or 0),
    }
    if str(getattr(execution, "status", "") or "").lower() != "ended":
        for entry in entries:
            row = entry["row"]
            await mark_stage_run(
                pool,
                review_id=row["id"],
                stage_id=stage_id,
                work_fingerprint=str(entry["work_fingerprint"]),
                state="submitted",
                backend="anthropic_batch",
                batch_id=getattr(execution, "local_batch_id", None),
                batch_custom_id=str(custom_id_for_entry(entry)),
                metadata=pending_metadata,
            )
    return StageBatchExecution(execution=execution, metrics=metrics)


async def finalize_stage_batch(
    *,
    pool: Any,
    execution: Any,
    entries: list[dict[str, Any]],
    stage_id: str,
    custom_id_for_entry: Any,
    parse_response_text: Any,
    normalize_response_text: Any,
    store_cached_response: Any,
    stage_usage_snapshot: Any,
    record_batch_fallback: Any,
    success_metadata: dict[str, Any],
    failure_metadata: dict[str, Any],
    failure_error_code: str,
) -> list[StageBatchOutcome]:
    outcomes: list[StageBatchOutcome] = []
    tier = int(success_metadata.get("tier") or 0)
    for entry in entries:
        row = entry["row"]
        custom_id = str(custom_id_for_entry(entry))
        raw_outcome = getattr(execution, "results_by_custom_id", {}).get(custom_id)
        parsed_result = parse_response_text(raw_outcome.response_text if raw_outcome is not None else None)
        if parsed_result is None:
            error_text = (raw_outcome.error_text if raw_outcome is not None else None) or failure_error_code
            await mark_stage_run(
                pool,
                review_id=row["id"],
                stage_id=stage_id,
                work_fingerprint=str(entry["work_fingerprint"]),
                state="failed",
                backend="anthropic_batch",
                batch_id=getattr(execution, "local_batch_id", None),
                batch_custom_id=custom_id,
                error_code=error_text,
                metadata=failure_metadata,
                completed=True,
            )
            if raw_outcome is not None:
                await record_batch_fallback(
                    batch_id=execution.local_batch_id,
                    custom_id=custom_id,
                    succeeded=False,
                    error_text=raw_outcome.error_text or failure_error_code,
                    pool=pool,
                )
            outcomes.append(
                StageBatchOutcome(
                    row=row,
                    parsed_result=None,
                    usage=None,
                    custom_id=custom_id,
                    success=False,
                    error_text=error_text,
                )
            )
            continue

        cache_hit = bool(raw_outcome is not None and getattr(raw_outcome, "cached", False))
        usage = stage_usage_snapshot(
            tier=tier,
            cache_hit=cache_hit,
            generated=not cache_hit,
        )
        normalized_text = normalize_response_text(raw_outcome.response_text or "") if raw_outcome is not None else None
        if not cache_hit and normalized_text is not None:
            await store_cached_response(
                entry["request"],
                response_text=normalized_text,
                metadata={"tier": tier, "backend": "anthropic_batch"},
            )
        await mark_stage_run(
            pool,
            review_id=row["id"],
            stage_id=stage_id,
            work_fingerprint=str(entry["work_fingerprint"]),
            state="succeeded",
            result_source="exact_cache" if cache_hit else "generated",
            backend="anthropic_batch",
            batch_id=getattr(execution, "local_batch_id", None),
            batch_custom_id=custom_id,
            usage=usage,
            response_text=normalized_text,
            metadata=success_metadata,
            completed=True,
        )
        outcomes.append(
            StageBatchOutcome(
                row=row,
                parsed_result=parsed_result,
                usage=usage,
                custom_id=custom_id,
                success=True,
                error_text=None,
            )
        )
    return outcomes
