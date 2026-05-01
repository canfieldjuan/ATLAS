from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment")


@dataclass(frozen=True)
class EnrichmentSingleRunnerDeps:
    empty_exact_cache_usage: Any
    combined_review_text_length: Any
    effective_min_review_text_length: Any
    effective_enrichment_skip_sources: Any
    coerce_float_value: Any
    build_classify_payload: Any
    get_tier1_client: Any
    resolve_tier_routing: Any
    prepare_stage_request: Any
    parse_stage_row_result: Any
    stage_usage_from_row: Any
    accumulate_exact_cache_usage: Any
    unpack_stage_result: Any
    stage_usage_snapshot: Any
    stage_result_text: Any
    stage_backend_name: Any
    tier1_has_extraction_gaps: Any
    get_tier2_client: Any
    merge_tier1_tier2: Any
    persist_enrichment_result: Any
    defer_batch_row: Any
    increment_attempts: Any
    ensure_stage_run: Any
    mark_stage_run: Any
    prepare_stage_execution: Any
    apply_stage_decision: Any
    defer_review_transition: Any
    persist_review_transition: Any
    build_tier1_stage_plan: Any
    build_tier2_stage_plan: Any
    tier2_system_prompt_for_content_type: Any
    call_openrouter_tier1: Any
    call_vllm_tier1: Any
    call_openrouter_tier2: Any
    call_vllm_tier2: Any
    tier1_json_schema: dict[str, Any]


async def run_single_enrichment_review(
    pool: Any,
    row: dict[str, Any],
    *,
    max_attempts: int,
    local_only: bool,
    max_tokens: int,
    truncate_length: int,
    run_id: str | None,
    usage_out: dict[str, int] | None,
    cfg: Any,
    deps: EnrichmentSingleRunnerDeps,
) -> bool | str:
    review_id = row["id"]
    cache_usage = deps.empty_exact_cache_usage()

    def _finish(status: bool | str) -> bool | str:
        if usage_out is not None:
            usage_out.clear()
            usage_out.update(cache_usage)
        return status

    combined_text_len = deps.combined_review_text_length(row)
    if combined_text_len < deps.effective_min_review_text_length(row):
        await pool.execute(
            "UPDATE b2b_reviews SET enrichment_status = 'not_applicable' WHERE id = $1",
            review_id,
        )
        return _finish(False)

    source = str(row.get("source") or "").strip().lower()
    skip_sources = deps.effective_enrichment_skip_sources()
    if source in skip_sources:
        await pool.execute(
            """
            UPDATE b2b_reviews
            SET enrichment_status = 'not_applicable',
                low_fidelity = false,
                low_fidelity_reasons = '[]'::jsonb,
                low_fidelity_detected_at = NULL
            WHERE id = $1
            """,
            review_id,
        )
        logger.debug(
            "Skipping unsupported churn-enrichment source %s for review %s",
            source,
            review_id,
        )
        return _finish(False)

    try:
        full_extraction_timeout = max(
            0.0,
            deps.coerce_float_value(
                getattr(cfg, "enrichment_full_extraction_timeout_seconds", 120.0),
                120.0,
            ),
        )
        payload = deps.build_classify_payload(row, truncate_length)
        payload_json = json.dumps(payload)
        trace_metadata = {
            "run_id": run_id,
            "vendor_name": str(row.get("vendor_name") or ""),
            "review_id": str(review_id),
            "source": str(row.get("source") or ""),
        }
        client = deps.get_tier1_client(cfg)
        from ...skills import get_skill_registry

        registry = get_skill_registry()
        tier1_skill = registry.get("digest/b2b_churn_extraction_tier1")
        tier1_request_fingerprint = None
        tier1_work_fingerprint = None
        existing_tier1_stage = None
        tier1 = None
        tier1_model = None
        tier1_cache_hit = False
        tier1_reused_from_stage = False

        use_openrouter_tier1, use_openrouter_tier2 = deps.resolve_tier_routing(
            cfg, local_only_override=local_only,
        )
        if tier1_skill is not None:
            tier1_provider = "openrouter" if use_openrouter_tier1 else "vllm"
            tier1_model_planned = (
                cfg.enrichment_openrouter_model or "anthropic/claude-haiku-4-5"
                if use_openrouter_tier1
                else cfg.enrichment_tier1_model
            )
            tier1_plan = deps.build_tier1_stage_plan(
                row=row,
                payload_json=payload_json,
                system_prompt=str(tier1_skill.content or ""),
                model=str(tier1_model_planned or ""),
                provider=tier1_provider,
                batch_enabled=False,
                run_id=run_id,
                prepare_stage_request=deps.prepare_stage_request,
                max_tokens=max(cfg.enrichment_tier1_max_tokens, 4096)
                if use_openrouter_tier1 else cfg.enrichment_tier1_max_tokens,
                guided_json=None if use_openrouter_tier1 else deps.tier1_json_schema,
            )
            tier1_request_fingerprint = tier1_plan.request_fingerprint
            tier1_work_fingerprint = tier1_plan.work_fingerprint
            await deps.ensure_stage_run(
                pool,
                review_id=review_id,
                stage_id="b2b_enrichment.tier1",
                work_fingerprint=str(tier1_plan.work_fingerprint),
                request_fingerprint=str(tier1_plan.request_fingerprint),
                provider=tier1_plan.provider,
                model=tier1_plan.model,
                backend=tier1_plan.backend,
                run_id=run_id,
                metadata=tier1_plan.metadata,
            )
            stage_decision = await deps.prepare_stage_execution(
                pool=pool,
                llm=None,
                task_name=None,
                artifact_type=None,
                artifact_id=None,
                review_id=review_id,
                stage_id="b2b_enrichment.tier1",
                work_fingerprint=str(tier1_work_fingerprint),
                request_fingerprint=str(tier1_request_fingerprint),
                parse_response_text=deps.parse_stage_row_result,
                defer_on_submitted=True,
                reconcile_batch=False,
            )
            existing_tier1_stage = stage_decision.stage_row
            if stage_decision.action == "reuse_stage":
                applied = await deps.apply_stage_decision(
                    pool=pool,
                    decision=stage_decision,
                    review_id=review_id,
                    stage_id="b2b_enrichment.tier1",
                    work_fingerprint=str(tier1_work_fingerprint),
                    tier=1,
                    usage_from_stage_row=deps.stage_usage_from_row,
                    pending_metadata={"tier": 1, "workload": "direct"},
                    success_metadata={"tier": 1, "workload": "direct"},
                    stage_usage_snapshot=deps.stage_usage_snapshot,
                )
                tier1 = applied.parsed_result if applied is not None else stage_decision.parsed_result
                tier1_model = applied.model if applied is not None else None
                tier1_cache_hit = bool(applied.cache_hit) if applied is not None else False
                tier1_reused_from_stage = True
            elif stage_decision.action == "defer_submitted_stage":
                applied = await deps.apply_stage_decision(
                    pool=pool,
                    decision=stage_decision,
                    review_id=review_id,
                    stage_id="b2b_enrichment.tier1",
                    work_fingerprint=str(tier1_work_fingerprint),
                    tier=1,
                    usage_from_stage_row=deps.stage_usage_from_row,
                    pending_metadata={"tier": 1, "workload": "direct"},
                    success_metadata={"tier": 1, "workload": "direct"},
                    stage_usage_snapshot=deps.stage_usage_snapshot,
                )
                await deps.defer_review_transition(
                    row=row,
                    tier="tier1",
                    custom_id=str((applied.custom_id if applied is not None else "") or ""),
                    usage=None,
                    defer_review=lambda target_row, **kwargs: deps.defer_batch_row(pool, target_row, **kwargs),
                )
                return _finish("deferred")
        if tier1_skill is not None and tier1 is not None:
            stage_reuse_usage = deps.stage_usage_from_row(existing_tier1_stage, tier=1)
            deps.accumulate_exact_cache_usage(cache_usage, stage_reuse_usage)
        elif use_openrouter_tier1:
            tier1, tier1_model, tier1_cache_hit = deps.unpack_stage_result(await asyncio.wait_for(
                deps.call_openrouter_tier1(
                    payload_json,
                    cfg,
                    include_cache_hit=True,
                    trace_metadata=trace_metadata | {"tier": "tier1"},
                ),
                timeout=full_extraction_timeout,
            ))
        elif tier1_skill is not None:
            tier1, tier1_model, tier1_cache_hit = deps.unpack_stage_result(await asyncio.wait_for(
                deps.call_vllm_tier1(
                    payload_json,
                    cfg,
                    client,
                    include_cache_hit=True,
                    trace_metadata=trace_metadata | {"tier": "tier1"},
                ),
                timeout=full_extraction_timeout,
            ))
        if tier1_skill is not None and tier1 is not None and existing_tier1_stage is not None and str(existing_tier1_stage.get("state") or "") == "succeeded":
            pass
        elif tier1_cache_hit:
            cache_usage["tier1_exact_cache_hits"] += 1
            cache_usage["exact_cache_hits"] += 1
        elif tier1_model is not None:
            cache_usage["tier1_generated_calls"] += 1
            cache_usage["generated"] += 1
        tier1_stage_usage = deps.stage_usage_snapshot(
            tier=1,
            cache_hit=bool(tier1_cache_hit),
            generated=tier1 is not None and not bool(tier1_cache_hit),
        )
        if tier1 is None:
            if tier1_request_fingerprint is not None:
                await deps.mark_stage_run(
                    pool,
                    review_id=review_id,
                    stage_id="b2b_enrichment.tier1",
                    work_fingerprint=str(tier1_work_fingerprint),
                    state="failed",
                    backend=deps.stage_backend_name(
                        batch_enabled=False,
                        provider="openrouter" if use_openrouter_tier1 else "vllm",
                    ),
                    error_code="tier1_empty_result",
                    metadata={"tier": 1, "workload": "direct"},
                    completed=True,
                )
            logger.debug("Tier 1 returned None for %s, deferring to next cycle", review_id)
            await deps.increment_attempts(pool, review_id, row["enrichment_attempts"], max_attempts)
            return _finish(False)
        if tier1_request_fingerprint is not None and not tier1_reused_from_stage:
            await deps.mark_stage_run(
                pool,
                review_id=review_id,
                stage_id="b2b_enrichment.tier1",
                work_fingerprint=str(tier1_work_fingerprint),
                state="succeeded",
                result_source="exact_cache" if tier1_cache_hit else "generated",
                backend=deps.stage_backend_name(
                    batch_enabled=False,
                    provider="openrouter" if use_openrouter_tier1 else "vllm",
                ),
                usage=tier1_stage_usage,
                response_text=deps.stage_result_text(tier1),
                metadata={"tier": 1, "workload": "direct"},
                completed=True,
            )

        tier2 = None
        tier2_model = None
        tier2_cache_hit = False
        needs_tier2 = deps.tier1_has_extraction_gaps(tier1, source=row.get("source"))
        tier2_request_fingerprint = None
        tier2_work_fingerprint = None
        existing_tier2_stage = None
        tier2_reused_from_stage = False
        if needs_tier2:
            tier2_skill = registry.get("digest/b2b_churn_extraction_tier2")
            if tier2_skill is not None:
                tier2_provider = "openrouter" if use_openrouter_tier2 else "vllm"
                tier2_model_planned = (
                    cfg.enrichment_tier2_openrouter_model
                    or cfg.enrichment_openrouter_model
                    or "anthropic/claude-haiku-4-5"
                    if use_openrouter_tier2
                    else (cfg.enrichment_tier2_model or cfg.enrichment_tier1_model)
                )
                tier2_payload = dict(payload)
                tier2_plan = deps.build_tier2_stage_plan(
                    row=row,
                    base_payload=tier2_payload,
                    tier1_result=tier1,
                    system_prompt=str(tier2_skill.content or ""),
                    model=str(tier2_model_planned or ""),
                    provider=tier2_provider,
                    batch_enabled=False,
                    run_id=run_id,
                    prepare_stage_request=deps.prepare_stage_request,
                    prompt_for_content_type=deps.tier2_system_prompt_for_content_type,
                    max_tokens=cfg.enrichment_tier2_max_tokens,
                    workload="direct",
                )
                tier2_request_fingerprint = tier2_plan.request_fingerprint
                tier2_work_fingerprint = tier2_plan.work_fingerprint
                await deps.ensure_stage_run(
                    pool,
                    review_id=review_id,
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
                    review_id=review_id,
                    stage_id="b2b_enrichment.tier2",
                    work_fingerprint=str(tier2_work_fingerprint),
                    request_fingerprint=str(tier2_request_fingerprint),
                    parse_response_text=deps.parse_stage_row_result,
                    defer_on_submitted=True,
                    reconcile_batch=False,
                )
                existing_tier2_stage = stage_decision.stage_row
                if stage_decision.action == "reuse_stage":
                    applied = await deps.apply_stage_decision(
                        pool=pool,
                        decision=stage_decision,
                        review_id=review_id,
                        stage_id="b2b_enrichment.tier2",
                        work_fingerprint=str(tier2_work_fingerprint),
                        tier=2,
                        usage_from_stage_row=deps.stage_usage_from_row,
                        pending_metadata={"tier": 2, "workload": "direct"},
                        success_metadata={"tier": 2, "workload": "direct"},
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
                        review_id=review_id,
                        stage_id="b2b_enrichment.tier2",
                        work_fingerprint=str(tier2_work_fingerprint),
                        tier=2,
                        usage_from_stage_row=deps.stage_usage_from_row,
                        pending_metadata={"tier": 2, "workload": "direct"},
                        success_metadata={"tier": 2, "workload": "direct"},
                        stage_usage_snapshot=deps.stage_usage_snapshot,
                    )
                    await deps.defer_review_transition(
                        row=row,
                        tier="tier2",
                        custom_id=str((applied.custom_id if applied is not None else "") or ""),
                        usage=cache_usage,
                        defer_review=lambda target_row, **kwargs: deps.defer_batch_row(pool, target_row, **kwargs),
                    )
                    return _finish("deferred")
        if needs_tier2:
            try:
                if tier2 is not None:
                    stage_reuse_usage = deps.stage_usage_from_row(existing_tier2_stage, tier=2)
                    deps.accumulate_exact_cache_usage(cache_usage, stage_reuse_usage)
                elif use_openrouter_tier2:
                    tier2, tier2_model, tier2_cache_hit = deps.unpack_stage_result(await asyncio.wait_for(
                        deps.call_openrouter_tier2(
                            tier1,
                            row,
                            cfg,
                            truncate_length,
                            include_cache_hit=True,
                            trace_metadata=trace_metadata | {"tier": "tier2"},
                        ),
                        timeout=full_extraction_timeout,
                    ))
                else:
                    tier2_client = deps.get_tier2_client(cfg)
                    tier2, tier2_model, tier2_cache_hit = deps.unpack_stage_result(await asyncio.wait_for(
                        deps.call_vllm_tier2(
                            tier1,
                            row,
                            cfg,
                            tier2_client,
                            truncate_length,
                            include_cache_hit=True,
                            trace_metadata=trace_metadata | {"tier": "tier2"},
                        ),
                        timeout=full_extraction_timeout,
                    ))
            except Exception:
                logger.warning(
                    "Tier 2 enrichment failed for review %s; persisting tier 1 result only",
                    review_id,
                    exc_info=True,
                )
        if needs_tier2 and tier2 is not None and existing_tier2_stage is not None and str(existing_tier2_stage.get("state") or "") == "succeeded":
            pass
        elif tier2_cache_hit:
            cache_usage["tier2_exact_cache_hits"] += 1
            cache_usage["exact_cache_hits"] += 1
        elif tier2_model is not None:
            cache_usage["tier2_generated_calls"] += 1
            cache_usage["generated"] += 1
        if needs_tier2 and tier2_request_fingerprint is not None and not tier2_reused_from_stage:
            tier2_stage_usage = deps.stage_usage_snapshot(
                tier=2,
                cache_hit=bool(tier2_cache_hit),
                generated=tier2 is not None and not bool(tier2_cache_hit),
            )
            await deps.mark_stage_run(
                pool,
                review_id=review_id,
                stage_id="b2b_enrichment.tier2",
                work_fingerprint=str(tier2_work_fingerprint),
                state="succeeded" if tier2 is not None else "failed",
                result_source="exact_cache" if tier2_cache_hit else ("generated" if tier2 is not None else None),
                backend=deps.stage_backend_name(
                    batch_enabled=False,
                    provider="openrouter" if use_openrouter_tier2 else "vllm",
                ),
                usage=tier2_stage_usage if tier2 is not None else None,
                response_text=deps.stage_result_text(tier2),
                error_code=None if tier2 is not None else "tier2_empty_result",
                metadata={"tier": 2, "workload": "direct"},
                completed=True,
            )
        model_id = f"hybrid:{tier1_model}+{tier2_model}" if tier2 is not None else (tier1_model or "")

        return _finish(
            await deps.persist_review_transition(
                row=row,
                tier1_result=tier1,
                tier2_result=tier2,
                model_id=model_id,
                usage=cache_usage,
                merge_results=deps.merge_tier1_tier2,
                persist_review=lambda target_row, result, *, model_id, usage: deps.persist_enrichment_result(
                    pool,
                    target_row,
                    result,
                    model_id=model_id,
                    max_attempts=max_attempts,
                    run_id=run_id,
                    cache_usage=usage,
                ),
            )
        )

    except Exception:
        logger.exception("Failed to enrich B2B review %s", review_id)
        try:
            new_status = "failed" if (row["enrichment_attempts"] + 1) >= max_attempts else "pending"
            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment_attempts = enrichment_attempts + 1,
                    enrichment_status = $1
                WHERE id = $2
                """,
                new_status, review_id,
            )
        except Exception:
            pass
        return _finish(False)
