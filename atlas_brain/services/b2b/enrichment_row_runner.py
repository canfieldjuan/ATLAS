from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from ...config import settings
from .enrichment_stage_runs import ensure_stage_run, mark_stage_run

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment")


@dataclass(frozen=True)
class EnrichmentRunnerDeps:
    apply_review_stage_transition: Any
    apply_stage_decision: Any
    build_tier1_stage_plan: Any
    build_tier2_stage_plan: Any
    defer_review_transition: Any
    enrich_single: Any
    finalize_stage_batch: Any
    parse_stage_row_result: Any
    empty_exact_cache_usage: Any
    persist_review_transition: Any
    prepare_stage_execution: Any
    row_usage_result: Any
    resolve_tier_routing: Any
    combined_review_text_length: Any
    effective_min_review_text_length: Any
    effective_enrichment_skip_sources: Any
    build_classify_payload: Any
    prepare_stage_request: Any
    tier2_system_prompt_for_content_type: Any
    stage_usage_from_row: Any
    stage_usage_snapshot: Any
    accumulate_exact_cache_usage: Any
    tier1_has_extraction_gaps: Any
    merge_tier1_tier2: Any
    persist_enrichment_result: Any
    defer_batch_row: Any
    submit_stage_batch: Any
    unpack_stage_result: Any
    call_openrouter_tier2: Any
    call_vllm_tier2: Any
    get_tier2_client: Any


def build_batch_metrics() -> dict[str, int]:
    return {
        "jobs": 0,
        "submitted_items": 0,
        "cache_prefiltered_items": 0,
        "fallback_single_call_items": 0,
        "completed_items": 0,
        "failed_items": 0,
        "reused_completed_items": 0,
        "reused_pending_items": 0,
        "rows_deferred": 0,
        "tier2_single_fallback_rows": 0,
    }


async def run_rows_concurrently(
    target_rows: list[dict[str, Any]],
    *,
    enrich_single: Any,
    pool: Any,
    max_attempts: int,
    local_only: bool,
    max_tokens: int,
    truncate_length: int,
    effective_concurrency: int,
    empty_exact_cache_usage: Any,
    row_usage_result: Any,
    run_id: str | None = None,
) -> list[dict[str, Any] | Exception]:
    if not target_rows:
        return []

    sem = asyncio.Semaphore(max(1, int(effective_concurrency or 1)))
    enrich_single_params = inspect.signature(enrich_single).parameters
    supports_usage_out = "usage_out" in enrich_single_params
    supports_run_id = "run_id" in enrich_single_params

    async def _bounded_enrich(row):
        async with sem:
            usage = empty_exact_cache_usage()
            kwargs: dict[str, Any] = {}
            if supports_run_id:
                kwargs["run_id"] = run_id
            if supports_usage_out:
                status = await enrich_single(
                    pool,
                    row,
                    max_attempts,
                    local_only,
                    max_tokens,
                    truncate_length,
                    usage_out=usage,
                    **kwargs,
                )
            else:
                status = await enrich_single(
                    pool,
                    row,
                    max_attempts,
                    local_only,
                    max_tokens,
                    truncate_length,
                    **kwargs,
                )
            return row_usage_result(status, usage)

    return await asyncio.gather(
        *[_bounded_enrich(row) for row in target_rows],
        return_exceptions=True,
    )


async def run_enrichment_rows(
    rows,
    cfg,
    pool,
    *,
    max_attempts: int,
    effective_concurrency: int,
    run_id: str | None,
    task: Any | None,
    deps: EnrichmentRunnerDeps,
) -> dict[str, Any]:
    from ...pipelines.llm import clean_llm_output, parse_json_response
    from ...services.b2b.anthropic_batch import (
        AnthropicBatchItem,
        mark_batch_fallback_result,
        run_anthropic_message_batch,
    )
    from ...services.b2b.cache_runner import (
        lookup_b2b_exact_stage_text,
        store_b2b_exact_stage_text,
    )
    from ...services.llm.anthropic import AnthropicBatchableLLM
    from ...services.protocols import Message
    from ...skills import get_skill_registry
    from ...autonomous.tasks._b2b_batch_utils import (
        anthropic_batch_min_items,
        anthropic_batch_requested,
        resolve_anthropic_batch_llm,
    )

    async def _persist_wrapped(
        row: dict[str, Any],
        result: dict[str, Any] | None,
        *,
        model_id: str,
        usage: dict[str, int],
    ) -> dict[str, Any]:
        status = await deps.persist_enrichment_result(
            pool,
            row,
            result,
            model_id=model_id,
            max_attempts=max_attempts,
            run_id=run_id,
            cache_usage=usage,
        )
        return deps.row_usage_result(status, usage)

    def _eligible_for_batch(row: dict[str, Any]) -> bool:
        if deps.combined_review_text_length(row) < deps.effective_min_review_text_length(row):
            return False
        source = str(row.get("source") or "").strip().lower()
        return source not in deps.effective_enrichment_skip_sources()

    def _parse_batch_text(text: str | None) -> dict[str, Any] | None:
        if not text:
            return None
        cleaned = clean_llm_output(text)
        parsed = parse_json_response(cleaned, recover_truncated=True)
        if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
            return parsed
        return None

    async def _run_single_rows(target_rows: list[dict[str, Any]]) -> list[dict[str, Any] | Exception]:
        return await run_rows_concurrently(
            target_rows,
            enrich_single=deps.enrich_single,
            pool=pool,
            max_attempts=max_attempts,
            local_only=cfg.enrichment_local_only,
            max_tokens=cfg.enrichment_max_tokens,
            truncate_length=cfg.review_truncate_length,
            effective_concurrency=effective_concurrency,
            empty_exact_cache_usage=deps.empty_exact_cache_usage,
            row_usage_result=deps.row_usage_result,
            run_id=run_id,
        )

    use_openrouter_tier1, use_openrouter_tier2 = deps.resolve_tier_routing(cfg)
    batch_requested = anthropic_batch_requested(
        task,
        global_default=bool(getattr(settings.b2b_churn, "anthropic_batch_enabled", False)),
        task_default=True,
        task_keys=("enrichment_anthropic_batch_enabled",),
    )
    tier1_batch_llm = (
        resolve_anthropic_batch_llm(
            current_llm=SimpleNamespace(
                name="openrouter",
                model=str(getattr(cfg, "enrichment_openrouter_model", "") or "anthropic/claude-haiku-4-5"),
            ),
            target_model_candidates=(getattr(cfg, "enrichment_openrouter_model", ""),),
        )
        if use_openrouter_tier1 and batch_requested
        else None
    )
    tier2_model_id = (
        getattr(cfg, "enrichment_tier2_openrouter_model", "")
        or getattr(cfg, "enrichment_openrouter_model", "")
        or "anthropic/claude-haiku-4-5"
    )
    tier2_batch_llm = (
        resolve_anthropic_batch_llm(
            current_llm=SimpleNamespace(name="openrouter", model=str(tier2_model_id)),
            target_model_candidates=(tier2_model_id,),
        )
        if use_openrouter_tier2 and batch_requested
        else None
    )

    if not isinstance(tier1_batch_llm, AnthropicBatchableLLM):
        tier1_batch_llm = None
    if not isinstance(tier2_batch_llm, AnthropicBatchableLLM):
        tier2_batch_llm = None

    tier1_batch_model_id = str(
        getattr(cfg, "enrichment_openrouter_model", "") or "anthropic/claude-haiku-4-5"
    )
    full_extraction_timeout = max(
        0.0,
        float(getattr(cfg, "enrichment_full_extraction_timeout_seconds", 120.0) or 120.0),
    )
    tier2_client = None
    batch_metrics = build_batch_metrics()

    async def _run_single_tier2_fallback(
        row: dict[str, Any],
        tier1: dict[str, Any],
        usage: dict[str, int],
    ) -> dict[str, Any]:
        nonlocal tier2_client
        logger.info(
            "B2B enrichment: Tier 2 batch unavailable for %s; falling back to single-call Tier 2",
            row["id"],
        )
        batch_metrics["tier2_single_fallback_rows"] += 1

        trace_metadata = {
            "run_id": run_id,
            "vendor_name": str(row.get("vendor_name") or ""),
            "review_id": str(row["id"]),
            "source": str(row.get("source") or ""),
            "tier": "tier2",
            "workload": "single_call_fallback",
            "batch_fallback_reason": "tier2_batch_unavailable",
        }
        request_fingerprint = None
        work_fingerprint = None
        existing_tier2_stage = None
        tier2_reused_from_stage = False
        tier2_skill_local = get_skill_registry().get("digest/b2b_churn_extraction_tier2")
        if tier2_skill_local is not None:
            tier2_provider = "openrouter" if use_openrouter_tier2 else "vllm"
            tier2_plan = deps.build_tier2_stage_plan(
                row=row,
                base_payload=deps.build_classify_payload(row, cfg.review_truncate_length),
                tier1_result=tier1,
                system_prompt=str(tier2_skill_local.content or ""),
                model=str(
                    tier2_model_id if use_openrouter_tier2 else (cfg.enrichment_tier2_model or cfg.enrichment_tier1_model)
                ),
                provider=tier2_provider,
                batch_enabled=False,
                run_id=run_id,
                prepare_stage_request=deps.prepare_stage_request,
                prompt_for_content_type=deps.tier2_system_prompt_for_content_type,
                max_tokens=cfg.enrichment_tier2_max_tokens,
                workload="single_call_fallback",
            )
            request_fingerprint = tier2_plan.request_fingerprint
            work_fingerprint = tier2_plan.work_fingerprint
            await ensure_stage_run(
                pool,
                review_id=row["id"],
                stage_id="b2b_enrichment.tier2",
                work_fingerprint=str(tier2_plan.work_fingerprint),
                request_fingerprint=str(tier2_plan.request_fingerprint),
                provider=tier2_plan.provider,
                model=tier2_plan.model,
                backend=tier2_plan.backend,
                run_id=run_id,
                metadata=tier2_plan.metadata,
            )
            stage_decision = await deps.prepare_stage_execution(
                pool=pool,
                llm=None,
                task_name=None,
                artifact_type=None,
                artifact_id=None,
                review_id=row["id"],
                stage_id="b2b_enrichment.tier2",
                work_fingerprint=str(work_fingerprint),
                request_fingerprint=str(request_fingerprint),
                parse_response_text=deps.parse_stage_row_result,
                defer_on_submitted=True,
                reconcile_batch=False,
            )
            existing_tier2_stage = stage_decision.stage_row
            if stage_decision.action == "reuse_stage":
                applied = await deps.apply_stage_decision(
                    pool=pool,
                    decision=stage_decision,
                    review_id=row["id"],
                    stage_id="b2b_enrichment.tier2",
                    work_fingerprint=str(work_fingerprint),
                    tier=2,
                    usage_from_stage_row=deps.stage_usage_from_row,
                    pending_metadata={"tier": 2, "workload": "single_call_fallback"},
                    success_metadata={"tier": 2, "workload": "single_call_fallback"},
                    stage_usage_snapshot=deps.stage_usage_snapshot,
                )
                tier2 = applied.parsed_result if applied is not None else stage_decision.parsed_result
                tier2_model = applied.model if applied is not None else None
                tier2_cache_hit = bool(applied.cache_hit) if applied is not None else False
                tier2_reused_from_stage = True
            elif stage_decision.action == "defer_submitted_stage":
                applied = await deps.apply_stage_decision(
                    pool=pool,
                    decision=stage_decision,
                    review_id=row["id"],
                    stage_id="b2b_enrichment.tier2",
                    work_fingerprint=str(work_fingerprint),
                    tier=2,
                    usage_from_stage_row=deps.stage_usage_from_row,
                    pending_metadata={"tier": 2, "workload": "single_call_fallback"},
                    success_metadata={"tier": 2, "workload": "single_call_fallback"},
                    stage_usage_snapshot=deps.stage_usage_snapshot,
                )
                return await deps.defer_review_transition(
                    row=row,
                    tier="tier2",
                    custom_id=str((applied.custom_id if applied is not None else "") or ""),
                    usage=usage,
                    defer_review=lambda target_row, **kwargs: deps.defer_batch_row(pool, target_row, **kwargs),
                )
            else:
                tier2 = None
                tier2_model = None
                tier2_cache_hit = False
        else:
            tier2 = None
            tier2_model = None
            tier2_cache_hit = False

        if tier2_reused_from_stage:
            deps.accumulate_exact_cache_usage(usage, deps.stage_usage_from_row(existing_tier2_stage, tier=2))
        elif use_openrouter_tier2:
            tier2, tier2_model, tier2_cache_hit = deps.unpack_stage_result(
                await asyncio.wait_for(
                    deps.call_openrouter_tier2(
                        tier1,
                        row,
                        cfg,
                        cfg.review_truncate_length,
                        include_cache_hit=True,
                        trace_metadata=trace_metadata,
                    ),
                    timeout=full_extraction_timeout,
                )
            )
        else:
            if tier2_client is None:
                tier2_client = deps.get_tier2_client(cfg)
            tier2, tier2_model, tier2_cache_hit = deps.unpack_stage_result(
                await asyncio.wait_for(
                    deps.call_vllm_tier2(
                        tier1,
                        row,
                        cfg,
                        tier2_client,
                        cfg.review_truncate_length,
                        include_cache_hit=True,
                        trace_metadata=trace_metadata,
                    ),
                    timeout=full_extraction_timeout,
                )
            )

        if not tier2_reused_from_stage:
            if tier2_cache_hit:
                usage["tier2_exact_cache_hits"] += 1
                usage["exact_cache_hits"] += 1
            elif tier2_model is not None:
                usage["tier2_generated_calls"] += 1
                usage["generated"] += 1
            if request_fingerprint is not None:
                stage_usage = deps.stage_usage_snapshot(
                    tier=2,
                    cache_hit=bool(tier2_cache_hit),
                    generated=tier2 is not None and not bool(tier2_cache_hit),
                )
                await mark_stage_run(
                    pool,
                    review_id=row["id"],
                    stage_id="b2b_enrichment.tier2",
                    work_fingerprint=str(work_fingerprint),
                    state="succeeded" if tier2 is not None else "failed",
                    result_source="exact_cache" if tier2_cache_hit else ("generated" if tier2 is not None else None),
                    backend="direct_openrouter" if use_openrouter_tier2 else "direct_vllm",
                    usage=stage_usage if tier2 is not None else None,
                    response_text=json.dumps(tier2) if tier2 is not None else None,
                    error_code=None if tier2 is not None else "tier2_empty_result",
                    metadata={"tier": 2, "workload": "single_call_fallback"},
                    completed=True,
                )

        model_id = f"hybrid:{tier1_batch_model_id}+{tier2_model}" if tier2 is not None else tier1_batch_model_id
        return await deps.persist_review_transition(
            row=row,
            tier1_result=tier1,
            tier2_result=tier2,
            model_id=model_id,
            usage=usage,
            merge_results=deps.merge_tier1_tier2,
            persist_review=_persist_wrapped,
        )

    if tier1_batch_llm is None:
        results = await _run_single_rows(rows)
    else:
        tier1_skill = get_skill_registry().get("digest/b2b_churn_extraction_tier1")
        tier2_skill = get_skill_registry().get("digest/b2b_churn_extraction_tier2")
        if not tier1_skill or not tier2_skill:
            results = await _run_single_rows(rows)
        else:
            direct_rows = [row for row in rows if not _eligible_for_batch(row)]
            batched_rows = [row for row in rows if _eligible_for_batch(row)]
            row_results: dict[Any, dict[str, Any] | Exception] = {}

            if direct_rows:
                direct_results = await _run_single_rows(direct_rows)
                for row, result in zip(direct_rows, direct_results):
                    row_results[row["id"]] = result

            tier1_ready_entries: list[dict[str, Any]] = []
            tier1_entries: list[dict[str, Any]] = []
            for row in batched_rows:
                payload_json = json.dumps(deps.build_classify_payload(row, cfg.review_truncate_length))
                tier1_plan = deps.build_tier1_stage_plan(
                    row=row,
                    payload_json=payload_json,
                    system_prompt=str(tier1_skill.content or ""),
                    model=str(cfg.enrichment_openrouter_model or "anthropic/claude-haiku-4-5"),
                    provider="openrouter",
                    batch_enabled=True,
                    run_id=run_id,
                    prepare_stage_request=deps.prepare_stage_request,
                    max_tokens=max(cfg.enrichment_tier1_max_tokens, 4096),
                    guided_json=None,
                )
                await ensure_stage_run(
                    pool,
                    review_id=row["id"],
                    stage_id="b2b_enrichment.tier1",
                    work_fingerprint=tier1_plan.work_fingerprint,
                    request_fingerprint=tier1_plan.request_fingerprint,
                    provider=tier1_plan.provider,
                    model=tier1_plan.model,
                    backend=tier1_plan.backend,
                    run_id=run_id,
                    metadata=tier1_plan.metadata,
                )
                stage_decision = await deps.prepare_stage_execution(
                    pool=pool,
                    llm=tier1_batch_llm,
                    task_name="b2b_enrichment",
                    artifact_type="review_enrichment_tier1",
                    artifact_id=str(row["id"]),
                    review_id=row["id"],
                    stage_id="b2b_enrichment.tier1",
                    work_fingerprint=tier1_plan.work_fingerprint,
                    request_fingerprint=tier1_plan.request_fingerprint,
                    parse_response_text=deps.parse_stage_row_result,
                    defer_on_submitted=False,
                    reconcile_batch=True,
                )
                if stage_decision.action == "reuse_stage":
                    applied = await deps.apply_stage_decision(
                        pool=pool,
                        decision=stage_decision,
                        review_id=row["id"],
                        stage_id="b2b_enrichment.tier1",
                        work_fingerprint=str(tier1_plan.work_fingerprint),
                        tier=1,
                        usage_from_stage_row=deps.stage_usage_from_row,
                        pending_metadata={"tier": 1, "workload": "anthropic_batch"},
                        success_metadata={"tier": 1, "workload": "anthropic_batch"},
                        stage_usage_snapshot=deps.stage_usage_snapshot,
                    )
                    tier1_ready_entries.append(
                        {
                            "row": row,
                            "tier1": applied.parsed_result if applied is not None else stage_decision.parsed_result,
                            "cached": bool(applied.cache_hit) if applied is not None else False,
                            "request": tier1_plan.request,
                            "work_fingerprint": tier1_plan.work_fingerprint,
                            "request_fingerprint": tier1_plan.request_fingerprint,
                            "stage_reused": True,
                        }
                    )
                    continue
                if stage_decision.action == "reuse_batch_result":
                    applied = await deps.apply_stage_decision(
                        pool=pool,
                        decision=stage_decision,
                        review_id=row["id"],
                        stage_id="b2b_enrichment.tier1",
                        work_fingerprint=str(tier1_plan.work_fingerprint),
                        tier=1,
                        usage_from_stage_row=deps.stage_usage_from_row,
                        pending_metadata={"tier": 1, "workload": "anthropic_batch_pending"},
                        success_metadata={"tier": 1, "workload": "anthropic_batch_reuse"},
                        stage_usage_snapshot=deps.stage_usage_snapshot,
                    )
                    tier1_ready_entries.append(
                        {
                            "row": row,
                            "tier1": applied.parsed_result if applied is not None else stage_decision.parsed_result,
                            "cached": bool(applied.cache_hit) if applied is not None else False,
                            "request": tier1_plan.request,
                            "work_fingerprint": tier1_plan.work_fingerprint,
                            "request_fingerprint": tier1_plan.request_fingerprint,
                        }
                    )
                    batch_metrics["reused_completed_items"] += 1
                    continue
                if stage_decision.action == "defer_pending_batch":
                    applied = await deps.apply_stage_decision(
                        pool=pool,
                        decision=stage_decision,
                        review_id=row["id"],
                        stage_id="b2b_enrichment.tier1",
                        work_fingerprint=str(tier1_plan.work_fingerprint),
                        tier=1,
                        usage_from_stage_row=deps.stage_usage_from_row,
                        pending_metadata={"tier": 1, "workload": "anthropic_batch_pending"},
                        success_metadata={"tier": 1, "workload": "anthropic_batch_reuse"},
                        stage_usage_snapshot=deps.stage_usage_snapshot,
                    )
                    row_results[row["id"]] = await deps.defer_batch_row(
                        pool,
                        row,
                        tier="tier1",
                        custom_id=str((applied.custom_id if applied is not None else "") or ""),
                    )
                    batch_metrics["reused_pending_items"] += 1
                    batch_metrics["rows_deferred"] += 1
                    continue
                cached = await lookup_b2b_exact_stage_text(tier1_plan.request)
                tier1_entries.append(
                    {
                        "row": row,
                        "payload_json": payload_json,
                        "messages": tier1_plan.messages,
                        "request": tier1_plan.request,
                        "work_fingerprint": tier1_plan.work_fingerprint,
                        "request_fingerprint": tier1_plan.request_fingerprint,
                        "cached_response_text": str(cached["response_text"] or "") if cached is not None else None,
                        "cached_usage": dict(cached.get("usage") or {}) if cached is not None else {},
                    }
                )

            if tier1_entries:
                tier1_batch = await deps.submit_stage_batch(
                    run_batch=run_anthropic_message_batch,
                    llm=tier1_batch_llm,
                    stage_id="b2b_enrichment.tier1",
                    task_name="b2b_enrichment",
                    items=[
                        AnthropicBatchItem(
                            custom_id=f"tier1_{entry['row']['id']}"[:64],
                            artifact_type="review_enrichment_tier1",
                            artifact_id=str(entry["row"]["id"]),
                            vendor_name=str(entry["row"].get("vendor_name") or "") or None,
                            messages=[
                                Message(role=str(message["role"]), content=str(message["content"]))
                                for message in entry["messages"]
                            ],
                            max_tokens=max(cfg.enrichment_tier1_max_tokens, 4096),
                            temperature=0.0,
                            trace_span_name="task.b2b_enrichment.tier1",
                            trace_metadata={
                                "run_id": run_id,
                                "vendor_name": str(entry["row"].get("vendor_name") or ""),
                                "review_id": str(entry["row"]["id"]),
                                "source": str(entry["row"].get("source") or ""),
                                "tier": "tier1",
                                "workload": "anthropic_batch",
                            },
                            request_metadata={
                                "review_id": str(entry["row"]["id"]),
                                "tier": 1,
                                "request_fingerprint": str(entry["request_fingerprint"]),
                            },
                            cached_response_text=entry["cached_response_text"],
                            cached_usage=entry["cached_usage"],
                        )
                        for entry in tier1_entries
                    ],
                    run_id=run_id,
                    min_batch_size=anthropic_batch_min_items(
                        task,
                        default=2,
                        keys=("enrichment_anthropic_batch_min_items",),
                    ),
                    batch_metadata={"stage": "tier1"},
                    pool=pool,
                    entries=tier1_entries,
                    custom_id_for_entry=lambda entry: f"tier1_{entry['row']['id']}"[:64],
                    pending_metadata={"tier": 1, "workload": "anthropic_batch_pending"},
                )
                tier1_execution = tier1_batch.execution
                for key, value in tier1_batch.metrics.items():
                    batch_metrics[key] += int(value or 0)
            else:
                tier1_execution = SimpleNamespace(results_by_custom_id={})

            tier2_entries: list[dict[str, Any]] = []
            fallback_rows: list[dict[str, Any]] = []
            per_row_batch_usage: dict[Any, dict[str, int]] = {}

            for ready_entry in tier1_ready_entries:
                row = ready_entry["row"]
                usage = deps.empty_exact_cache_usage()
                if ready_entry["cached"]:
                    usage["tier1_exact_cache_hits"] += 1
                    usage["exact_cache_hits"] += 1
                else:
                    usage["tier1_generated_calls"] += 1
                    usage["generated"] += 1
                per_row_batch_usage[row["id"]] = usage
                needs_tier2 = deps.tier1_has_extraction_gaps(ready_entry["tier1"], source=row.get("source"))
                if needs_tier2 and tier2_batch_llm is not None:
                    tier2_plan = deps.build_tier2_stage_plan(
                        row=row,
                        base_payload=deps.build_classify_payload(row, cfg.review_truncate_length),
                        tier1_result=ready_entry["tier1"],
                        system_prompt=str(tier2_skill.content or ""),
                        model=str(tier2_model_id),
                        provider="openrouter",
                        batch_enabled=True,
                        run_id=run_id,
                        prepare_stage_request=deps.prepare_stage_request,
                        prompt_for_content_type=deps.tier2_system_prompt_for_content_type,
                        max_tokens=cfg.enrichment_tier2_max_tokens,
                        workload="anthropic_batch",
                    )
                    await ensure_stage_run(
                        pool,
                        review_id=row["id"],
                        stage_id="b2b_enrichment.tier2",
                        work_fingerprint=tier2_plan.work_fingerprint,
                        request_fingerprint=tier2_plan.request_fingerprint,
                        provider=tier2_plan.provider,
                        model=tier2_plan.model,
                        backend=tier2_plan.backend,
                        run_id=run_id,
                        metadata=tier2_plan.metadata,
                    )
                    stage_decision = await deps.prepare_stage_execution(
                        pool=pool,
                        llm=tier2_batch_llm,
                        task_name="b2b_enrichment",
                        artifact_type="review_enrichment_tier2",
                        artifact_id=str(row["id"]),
                        review_id=row["id"],
                        stage_id="b2b_enrichment.tier2",
                        work_fingerprint=tier2_plan.work_fingerprint,
                        request_fingerprint=tier2_plan.request_fingerprint,
                        parse_response_text=deps.parse_stage_row_result,
                        defer_on_submitted=False,
                        reconcile_batch=True,
                    )
                    applied = await deps.apply_stage_decision(
                        pool=pool,
                        decision=stage_decision,
                        review_id=row["id"],
                        stage_id="b2b_enrichment.tier2",
                        work_fingerprint=str(tier2_plan.work_fingerprint),
                        tier=2,
                        usage_from_stage_row=deps.stage_usage_from_row,
                        pending_metadata={"tier": 2, "workload": "anthropic_batch_pending"},
                        success_metadata={
                            "tier": 2,
                            "workload": "anthropic_batch_reuse" if stage_decision.action == "reuse_batch_result" else "anthropic_batch",
                        },
                        stage_usage_snapshot=deps.stage_usage_snapshot,
                    )
                    if stage_decision.action in {"reuse_stage", "reuse_batch_result", "defer_pending_batch"}:
                        transition = await deps.apply_review_stage_transition(
                            applied=applied,
                            row=row,
                            tier="tier2",
                            usage=usage,
                            tier1_result=ready_entry["tier1"],
                            model_id=f"hybrid:{tier1_batch_model_id}+{tier2_model_id}",
                            accumulate_usage=deps.accumulate_exact_cache_usage,
                            merge_results=deps.merge_tier1_tier2,
                            persist_review=_persist_wrapped,
                            defer_review=lambda target_row, **kwargs: deps.defer_batch_row(pool, target_row, **kwargs),
                        )
                        row_results[row["id"]] = transition.row_result
                        if stage_decision.action == "reuse_batch_result":
                            batch_metrics["reused_completed_items"] += 1
                        elif stage_decision.action == "defer_pending_batch":
                            batch_metrics["reused_pending_items"] += 1
                            batch_metrics["rows_deferred"] += 1
                        continue
                    cached = await lookup_b2b_exact_stage_text(tier2_plan.request)
                    tier2_entries.append(
                        {
                            "row": row,
                            "tier1": ready_entry["tier1"],
                            "messages": tier2_plan.messages,
                            "request": tier2_plan.request,
                            "work_fingerprint": tier2_plan.work_fingerprint,
                            "request_fingerprint": tier2_plan.request_fingerprint,
                            "cached_response_text": str(cached["response_text"] or "") if cached is not None else None,
                            "cached_usage": dict(cached.get("usage") or {}) if cached is not None else {},
                        }
                    )
                elif needs_tier2:
                    row_results[row["id"]] = await _run_single_tier2_fallback(row, ready_entry["tier1"], usage)
                else:
                    row_results[row["id"]] = await deps.persist_review_transition(
                        row=row,
                        tier1_result=ready_entry["tier1"],
                        tier2_result=None,
                        model_id=tier1_batch_model_id,
                        usage=usage,
                        merge_results=deps.merge_tier1_tier2,
                        persist_review=_persist_wrapped,
                    )

            tier1_outcomes = await deps.finalize_stage_batch(
                pool=pool,
                execution=tier1_execution,
                entries=tier1_entries,
                stage_id="b2b_enrichment.tier1",
                custom_id_for_entry=lambda entry: f"tier1_{entry['row']['id']}"[:64],
                parse_response_text=_parse_batch_text,
                normalize_response_text=clean_llm_output,
                store_cached_response=store_b2b_exact_stage_text,
                stage_usage_snapshot=deps.stage_usage_snapshot,
                record_batch_fallback=mark_batch_fallback_result,
                success_metadata={"tier": 1, "workload": "anthropic_batch"},
                failure_metadata={"tier": 1, "workload": "anthropic_batch"},
                failure_error_code="tier1_batch_parse_failed",
            )

            for entry, outcome in zip(tier1_entries, tier1_outcomes):
                row = entry["row"]
                tier1 = outcome.parsed_result
                if not outcome.success or tier1 is None:
                    fallback_rows.append(row)
                    continue
                usage = outcome.usage or deps.empty_exact_cache_usage()
                per_row_batch_usage[row["id"]] = usage
                needs_tier2 = deps.tier1_has_extraction_gaps(tier1, source=row.get("source"))
                if needs_tier2 and tier2_batch_llm is not None:
                    tier2_plan = deps.build_tier2_stage_plan(
                        row=row,
                        base_payload=deps.build_classify_payload(row, cfg.review_truncate_length),
                        tier1_result=tier1,
                        system_prompt=str(tier2_skill.content or ""),
                        model=str(tier2_model_id),
                        provider="openrouter",
                        batch_enabled=True,
                        run_id=run_id,
                        prepare_stage_request=deps.prepare_stage_request,
                        prompt_for_content_type=deps.tier2_system_prompt_for_content_type,
                        max_tokens=cfg.enrichment_tier2_max_tokens,
                        workload="anthropic_batch",
                    )
                    await ensure_stage_run(
                        pool,
                        review_id=row["id"],
                        stage_id="b2b_enrichment.tier2",
                        work_fingerprint=tier2_plan.work_fingerprint,
                        request_fingerprint=tier2_plan.request_fingerprint,
                        provider=tier2_plan.provider,
                        model=tier2_plan.model,
                        backend=tier2_plan.backend,
                        run_id=run_id,
                        metadata=tier2_plan.metadata,
                    )
                    stage_decision = await deps.prepare_stage_execution(
                        pool=pool,
                        llm=tier2_batch_llm,
                        task_name="b2b_enrichment",
                        artifact_type="review_enrichment_tier2",
                        artifact_id=str(row["id"]),
                        review_id=row["id"],
                        stage_id="b2b_enrichment.tier2",
                        work_fingerprint=tier2_plan.work_fingerprint,
                        request_fingerprint=tier2_plan.request_fingerprint,
                        parse_response_text=deps.parse_stage_row_result,
                        defer_on_submitted=False,
                        reconcile_batch=True,
                    )
                    applied = await deps.apply_stage_decision(
                        pool=pool,
                        decision=stage_decision,
                        review_id=row["id"],
                        stage_id="b2b_enrichment.tier2",
                        work_fingerprint=str(tier2_plan.work_fingerprint),
                        tier=2,
                        usage_from_stage_row=deps.stage_usage_from_row,
                        pending_metadata={"tier": 2, "workload": "anthropic_batch_pending"},
                        success_metadata={
                            "tier": 2,
                            "workload": "anthropic_batch_reuse" if stage_decision.action == "reuse_batch_result" else "anthropic_batch",
                        },
                        stage_usage_snapshot=deps.stage_usage_snapshot,
                    )
                    if stage_decision.action in {"reuse_stage", "reuse_batch_result", "defer_pending_batch"}:
                        transition = await deps.apply_review_stage_transition(
                            applied=applied,
                            row=row,
                            tier="tier2",
                            usage=usage,
                            tier1_result=tier1,
                            model_id=f"hybrid:{tier1_batch_model_id}+{tier2_model_id}",
                            accumulate_usage=deps.accumulate_exact_cache_usage,
                            merge_results=deps.merge_tier1_tier2,
                            persist_review=_persist_wrapped,
                            defer_review=lambda target_row, **kwargs: deps.defer_batch_row(pool, target_row, **kwargs),
                        )
                        row_results[row["id"]] = transition.row_result
                        if stage_decision.action == "reuse_batch_result":
                            batch_metrics["reused_completed_items"] += 1
                        elif stage_decision.action == "defer_pending_batch":
                            batch_metrics["reused_pending_items"] += 1
                            batch_metrics["rows_deferred"] += 1
                        continue
                    cached = await lookup_b2b_exact_stage_text(tier2_plan.request)
                    tier2_entries.append(
                        {
                            "row": row,
                            "tier1": tier1,
                            "messages": tier2_plan.messages,
                            "request": tier2_plan.request,
                            "work_fingerprint": tier2_plan.work_fingerprint,
                            "request_fingerprint": tier2_plan.request_fingerprint,
                            "cached_response_text": str(cached["response_text"] or "") if cached is not None else None,
                            "cached_usage": dict(cached.get("usage") or {}) if cached is not None else {},
                        }
                    )
                elif needs_tier2:
                    row_results[row["id"]] = await _run_single_tier2_fallback(row, tier1, usage)
                else:
                    row_results[row["id"]] = await deps.persist_review_transition(
                        row=row,
                        tier1_result=tier1,
                        tier2_result=None,
                        model_id=tier1_batch_model_id,
                        usage=usage,
                        merge_results=deps.merge_tier1_tier2,
                        persist_review=_persist_wrapped,
                    )

            if tier2_entries:
                tier2_batch = await deps.submit_stage_batch(
                    run_batch=run_anthropic_message_batch,
                    llm=tier2_batch_llm,
                    stage_id="b2b_enrichment.tier2",
                    task_name="b2b_enrichment",
                    items=[
                        AnthropicBatchItem(
                            custom_id=f"tier2_{entry['row']['id']}"[:64],
                            artifact_type="review_enrichment_tier2",
                            artifact_id=str(entry["row"]["id"]),
                            vendor_name=str(entry["row"].get("vendor_name") or "") or None,
                            messages=[
                                Message(role=str(message["role"]), content=str(message["content"]))
                                for message in entry["messages"]
                            ],
                            max_tokens=cfg.enrichment_tier2_max_tokens,
                            temperature=0.0,
                            trace_span_name="task.b2b_enrichment.tier2",
                            trace_metadata={
                                "run_id": run_id,
                                "vendor_name": str(entry["row"].get("vendor_name") or ""),
                                "review_id": str(entry["row"]["id"]),
                                "source": str(entry["row"].get("source") or ""),
                                "tier": "tier2",
                                "workload": "anthropic_batch",
                            },
                            request_metadata={
                                "review_id": str(entry["row"]["id"]),
                                "tier": 2,
                                "request_fingerprint": str(entry["request_fingerprint"]),
                            },
                            cached_response_text=entry["cached_response_text"],
                            cached_usage=entry["cached_usage"],
                        )
                        for entry in tier2_entries
                    ],
                    run_id=run_id,
                    min_batch_size=anthropic_batch_min_items(
                        task,
                        default=2,
                        keys=("enrichment_anthropic_batch_min_items",),
                    ),
                    batch_metadata={"stage": "tier2"},
                    pool=pool,
                    entries=tier2_entries,
                    custom_id_for_entry=lambda entry: f"tier2_{entry['row']['id']}"[:64],
                    pending_metadata={"tier": 2, "workload": "anthropic_batch_pending"},
                )
                tier2_execution = tier2_batch.execution
                for key, value in tier2_batch.metrics.items():
                    batch_metrics[key] += int(value or 0)
            else:
                tier2_execution = SimpleNamespace(results_by_custom_id={})

            tier2_outcomes = await deps.finalize_stage_batch(
                pool=pool,
                execution=tier2_execution,
                entries=tier2_entries,
                stage_id="b2b_enrichment.tier2",
                custom_id_for_entry=lambda entry: f"tier2_{entry['row']['id']}"[:64],
                parse_response_text=_parse_batch_text,
                normalize_response_text=clean_llm_output,
                store_cached_response=store_b2b_exact_stage_text,
                stage_usage_snapshot=deps.stage_usage_snapshot,
                record_batch_fallback=mark_batch_fallback_result,
                success_metadata={"tier": 2, "workload": "anthropic_batch"},
                failure_metadata={"tier": 2, "workload": "anthropic_batch"},
                failure_error_code="tier2_batch_parse_failed",
            )

            for entry, outcome in zip(tier2_entries, tier2_outcomes):
                row = entry["row"]
                if not outcome.success or outcome.parsed_result is None:
                    fallback_rows.append(row)
                    continue
                usage = per_row_batch_usage[row["id"]]
                deps.accumulate_exact_cache_usage(usage, outcome.usage)
                row_results[row["id"]] = await deps.persist_review_transition(
                    row=row,
                    tier1_result=entry["tier1"],
                    tier2_result=outcome.parsed_result,
                    model_id=f"hybrid:{tier1_batch_model_id}+{tier2_model_id}",
                    usage=usage,
                    merge_results=deps.merge_tier1_tier2,
                    persist_review=_persist_wrapped,
                )

            fallback_results = await _run_single_rows(fallback_rows)
            for row, result in zip(fallback_rows, fallback_results):
                row_results[row["id"]] = result

            results = [row_results[row["id"]] for row in rows]

    for row, result in zip(rows, results):
        if isinstance(result, Exception):
            logger.error("Unexpected enrichment error for %s: %s", row["id"], result, exc_info=result)

    cache_usage = deps.empty_exact_cache_usage()
    for result in results:
        if isinstance(result, Exception):
            continue
        if isinstance(result, dict):
            deps.accumulate_exact_cache_usage(cache_usage, result)

    batch_ids = [row["id"] for row in rows]
    status_rows = await pool.fetch(
        """
        SELECT enrichment_status, count(*) AS ct
        FROM b2b_reviews
        WHERE id = ANY($1::uuid[])
        GROUP BY enrichment_status
        """,
        batch_ids,
    )
    status_counts = {r["enrichment_status"]: int(r["ct"]) for r in status_rows}
    enriched = status_counts.get("enriched", 0)
    quarantined = status_counts.get("quarantined", 0)
    no_signal = status_counts.get("no_signal", 0)
    failed = status_counts.get("failed", 0)

    logger.info(
        "B2B enrichment: %d enriched, %d quarantined, %d no_signal, %d failed (of %d)",
        enriched, quarantined, no_signal or 0, failed, len(rows),
    )

    return {
        "total": len(rows),
        "enriched": enriched,
        "quarantined": quarantined,
        "no_signal": no_signal or 0,
        "failed": failed,
        "anthropic_batch_jobs": batch_metrics["jobs"],
        "anthropic_batch_items_submitted": batch_metrics["submitted_items"],
        "anthropic_batch_cache_prefiltered_items": batch_metrics["cache_prefiltered_items"],
        "anthropic_batch_fallback_single_call_items": batch_metrics["fallback_single_call_items"],
        "anthropic_batch_completed_items": batch_metrics["completed_items"],
        "anthropic_batch_failed_items": batch_metrics["failed_items"],
        "anthropic_batch_reused_completed_items": batch_metrics["reused_completed_items"],
        "anthropic_batch_reused_pending_items": batch_metrics["reused_pending_items"],
        "anthropic_batch_rows_deferred": batch_metrics["rows_deferred"],
        "anthropic_batch_tier2_single_fallback_rows": batch_metrics["tier2_single_fallback_rows"],
        **cache_usage,
    }
