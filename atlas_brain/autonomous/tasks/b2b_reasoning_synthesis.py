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

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.b2b_reasoning_synthesis")

_SCHEMA_VERSION = "v2"
_PACKET_SCHEMA_VERSION = "witness_packet_v1"


def _compute_pool_hash(layers: dict[str, Any]) -> str:
    """Deterministic hash of all pool layer data for a vendor."""
    raw = json.dumps(layers, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _validation_feedback(vresult: Any, limit: int) -> list[str]:
    """Convert validator errors into compact retry feedback lines."""
    feedback: list[str] = []
    for err in list(getattr(vresult, "errors", []))[:limit]:
        path = getattr(err, "path", "") or "$"
        message = getattr(err, "message", "") or "validation failed"
        feedback.append(f"{path}: {message}")
    return feedback


def _task_run_id(task: ScheduledTask | Any) -> str | None:
    """Return a stable run identifier for scheduled, manual, and test invocations."""
    task_id = getattr(task, "id", None)
    if task_id:
        return str(task_id)
    metadata = getattr(task, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("run_id", "task_id", "invocation_id"):
            value = metadata.get(key)
            if value:
                return str(value)
    return None


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
    await pool.execute(
        """
        DELETE FROM b2b_vendor_witnesses
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
    for witness in getattr(packet, "witness_pack", []) or []:
        await pool.execute(
            """
            INSERT INTO b2b_vendor_witnesses
                (vendor_name, as_of_date, analysis_window_days, schema_version,
                 evidence_hash, witness_id, review_id, witness_type, excerpt_text,
                 source, reviewed_at, reviewer_company, reviewer_title,
                 pain_category, competitor, salience_score, selection_reason,
                 signal_tags, source_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    $14, $15, $16, $17, $18::jsonb, $19)
            """,
            vendor_name,
            as_of_date,
            analysis_window_days,
            _PACKET_SCHEMA_VERSION,
            evidence_hash,
            str(witness.get("witness_id") or witness.get("_sid") or ""),
            str(witness.get("review_id") or ""),
            str(witness.get("witness_type") or ""),
            str(witness.get("excerpt_text") or ""),
            str(witness.get("source") or ""),
            witness.get("reviewed_at"),
            witness.get("reviewer_company"),
            witness.get("reviewer_title"),
            witness.get("pain_category"),
            witness.get("competitor"),
            float(witness.get("salience_score") or 0.0),
            str(witness.get("selection_reason") or ""),
            json.dumps(witness.get("signal_tags") or []),
            str(witness.get("_sid") or witness.get("witness_id") or ""),
        )


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: generate reasoning synthesis per vendor."""
    cfg = settings.b2b_churn
    run_id = _task_run_id(task)

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

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
        return {"_skip_synthesis": "No pool data available"}

    # Optional vendor filter: task.metadata.test_vendors or config
    test_vendors = (task.metadata or {}).get("test_vendors")
    if test_vendors:
        if isinstance(test_vendors, str):
            test_vendors = [v.strip() for v in test_vendors.split(",")]
        vendor_set = set(v.lower() for v in test_vendors)
        vendor_pools = {
            k: v for k, v in vendor_pools.items()
            if k.lower() in vendor_set
        }
        logger.info(
            "Filtered to %d test vendors: %s",
            len(vendor_pools), sorted(vendor_pools.keys()),
        )

    # Check for existing synthesis to skip unchanged vendors
    existing = await pool.fetch(
        """
        SELECT vendor_name, evidence_hash
        FROM b2b_reasoning_synthesis
        WHERE as_of_date = $1
          AND analysis_window_days = $2
          AND schema_version = $3
        """,
        today, window_days, _SCHEMA_VERSION,
    )
    existing_hashes: dict[str, str] = {
        r["vendor_name"]: r["evidence_hash"] for r in existing
    }

    # Filter to vendors needing reasoning
    force = bool((task.metadata or {}).get("force"))
    vendors_to_reason: list[tuple[str, dict, str]] = []
    for vendor_name, layers in vendor_pools.items():
        ev_hash = _compute_pool_hash(layers)
        if not force and existing_hashes.get(vendor_name) == ev_hash:
            continue
        vendors_to_reason.append((vendor_name, layers, ev_hash))

    skipped = len(vendor_pools) - len(vendors_to_reason)
    logger.info(
        "Reasoning synthesis v2: %d vendors to process, %d skipped (unchanged)",
        len(vendors_to_reason), skipped,
    )

    if not vendors_to_reason:
        return {
            "vendors_total": len(vendor_pools),
            "vendors_reasoned": 0,
            "vendors_skipped": skipped,
            "_skip_synthesis": "All vendors unchanged",
        }

    # Resolve LLM
    from ...reasoning.config import ReasoningConfig
    from ...reasoning.llm_utils import resolve_stratified_llm

    rcfg = ReasoningConfig()
    llm_cfg = rcfg.model_copy(deep=True)
    synthesis_model = str(
        getattr(cfg, "reasoning_synthesis_model", "") or ""
    ).strip()
    if not synthesis_model:
        synthesis_model = str(
            getattr(settings.llm, "openrouter_reasoning_model", "") or ""
        ).strip()
    if synthesis_model:
        llm_cfg.stratified_llm_workload = "openrouter"
        llm_cfg.stratified_openrouter_model = synthesis_model
        llm_cfg.stratified_openrouter_model_light = synthesis_model
    llm = resolve_stratified_llm(llm_cfg)
    if llm is None:
        return {"_skip_synthesis": "No LLM available for reasoning synthesis"}

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
    llm_max_tokens = max(
        256,
        min(
            int(getattr(cfg, "reasoning_synthesis_max_tokens", rcfg.max_tokens)),
            int(rcfg.max_tokens),
        ),
    )
    llm_temperature = float(
        getattr(cfg, "reasoning_synthesis_temperature", llm_cfg.temperature),
    )
    feedback_limit = max(
        1, int(getattr(cfg, "reasoning_synthesis_feedback_limit", 5)),
    )
    sem = asyncio.Semaphore(max_concurrent)
    total_tokens = 0
    succeeded = 0
    failed = 0
    validation_failures = 0
    failed_vendors: list[dict[str, Any]] = []

    async def _reason_one(
        vendor_name: str, layers: dict, ev_hash: str,
    ) -> None:
        nonlocal total_tokens, succeeded, failed, validation_failures, failed_vendors
        async with sem:
            packet = compress_vendor_pools(vendor_name, layers)
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
            payload = json.dumps(
                packet.to_llm_payload(
                    compact_metric_ledger=True,
                    compact_aggregates=True,
                ),
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            )
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
                    messages.append(Message(
                        role="user",
                        content=(
                            "Your previous response was rejected. "
                            "Return a complete corrected JSON object only.\n"
                            f"Fix these issues:\n{feedback}"
                        ),
                    ))
                try:
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
                        parsed = normalize_synthesis_source_ids(parsed, packet)
                        vresult = validate_synthesis(parsed, packet)
                        if vresult.is_valid:
                            synthesis = parsed
                            last_validation = vresult
                            break
                        synthesis = None
                        last_validation = vresult
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
                    from ..visibility import record_attempt
                    await record_attempt(
                        pool, artifact_type="reasoning_synthesis",
                        artifact_id=vendor_name,
                        run_id=run_id, attempt_no=attempt + 1,
                        stage="validation", status="rejected",
                        blocker_count=len(failure_reasons),
                        blocking_issues=failure_reasons[:5],
                        failure_step="validation" if last_validation else "llm_response",
                        error_message=(
                            last_validation.summary()[:200] if last_validation
                            else "; ".join(failure_reasons[:2])[:200]
                        ),
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
            synthesis = build_persistable_synthesis(synthesis, packet)
            persisted_vresult = validate_synthesis(synthesis, packet)
            if not persisted_vresult.is_valid:
                logger.warning(
                    "Persisted reasoning synthesis for %s failed validation: %s",
                    vendor_name, persisted_vresult.summary(),
                )
                from ..visibility import emit_event
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
                synthesis.pop("_validation_warnings", None)
            vresult = persisted_vresult

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
                from ..visibility import emit_event
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
            logger.info(
                "Reasoning synthesis v2: %s (%d tokens, %d warnings)",
                vendor_name, vendor_tokens, len(vresult.warnings),
            )

    await asyncio.gather(*[
        _reason_one(vn, layers, eh)
        for vn, layers, eh in vendors_to_reason
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

    if cfg.cross_vendor_synthesis_enabled:
        try:
            xv_succeeded, xv_failed, xv_tokens, xv_mirrored = await _run_cross_vendor_synthesis(
                pool=pool,
                vendor_pools=vendor_pools,
                llm=llm,
                rcfg=llm_cfg,
                cfg=cfg,
                today=today,
                window_days=window_days,
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

    return {
        "vendors_total": len(vendor_pools),
        "vendors_reasoned": succeeded,
        "vendors_failed": failed,
        "vendors_validation_failures": validation_failures,
        "failed_vendors": failed_vendors,
        "vendors_skipped": skipped,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "schema_version": _SCHEMA_VERSION,
        "cross_vendor_succeeded": xv_succeeded,
        "cross_vendor_failed": xv_failed,
        "cross_vendor_tokens": xv_tokens,
        "cross_vendor_mirrored": xv_mirrored,
    }


# ---------------------------------------------------------------------------
# Cross-vendor synthesis phase
# ---------------------------------------------------------------------------

_XV_SCHEMA_VERSION = "synthesis_v1"


async def _run_cross_vendor_synthesis(
    *,
    pool,
    vendor_pools: dict[str, dict],
    llm,
    rcfg,
    cfg,
    today: date,
    window_days: int,
) -> tuple[int, int, int, int]:
    """Run cross-vendor synthesis: battles, councils, asymmetry.

    Returns (succeeded, failed, tokens_used, mirrored_to_legacy).
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
        build_category_council_packet,
        build_pairwise_battle_packet,
        build_resource_asymmetry_packet,
        compute_cross_vendor_evidence_hash,
        normalize_cross_vendor_contract,
        to_legacy_cross_vendor_conclusion,
    )

    # Reconstruct inputs for selectors from pool layers
    # Build churn signal rows from pool cores
    vendor_scores: list[dict[str, Any]] = []
    evidence_lookup: dict[str, dict[str, Any]] = {}
    product_profiles: dict[str, dict[str, Any]] = {}

    for vname, layers in vendor_pools.items():
        core = layers.get("core") or layers.get("churn_signal") or {}
        evidence_lookup[vname] = core
        vendor_scores.append({
            "vendor_name": vname,
            "avg_urgency_score": core.get("avg_urgency_score") or core.get("avg_urgency") or 0,
            "total_reviews": core.get("total_reviews") or core.get("review_count") or 0,
            "product_category": core.get("product_category") or "",
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

    # Build displacement map
    try:
        disp_rows = await pool.fetch(
            "SELECT from_vendor, to_vendor, mention_count, primary_driver, "
            "signal_strength, velocity_7d, evidence_breakdown "
            "FROM b2b_displacement_edges "
            "WHERE as_of_date = (SELECT MAX(as_of_date) FROM b2b_displacement_edges)"
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
        logger.debug("Category dynamics fetch failed for xv synthesis")

    # Build vendor membership map for category selection.
    # select_categories() only uses the mapping keys to count vendors with
    # evidence in each category; it does not consume legacy archetype values.
    category_vendor_lookup: dict[str, dict[str, Any]] = {}
    for vname, ev in evidence_lookup.items():
        if (ev or {}).get("product_category"):
            category_vendor_lookup[vname] = {}

    # --- Select targets ---
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
            packet,
        ))

    for cat, eco_ev in categories:
        packet = build_category_council_packet(
            cat, eco_ev, vendor_pools, product_profiles, displacement_edges,
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
            packet,
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
            packet,
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
    xv_llm_max_tokens = max(
        256,
        min(
            int(getattr(cfg, "reasoning_synthesis_max_tokens", rcfg.max_tokens)),
            int(rcfg.max_tokens),
        ),
    )
    xv_llm_temperature = float(
        getattr(cfg, "reasoning_synthesis_temperature", rcfg.temperature),
    )
    _succeeded = 0
    _failed = 0
    _tokens = 0
    _mirrored = 0
    llm_model_name = getattr(llm, "model", getattr(llm, "model_id", ""))

    async def _xv_one(
        analysis_type: str,
        prompt: str,
        vendors: list[str],
        category: str | None,
        packet: dict[str, Any],
    ) -> None:
        nonlocal _succeeded, _failed, _tokens, _mirrored

        ev_hash = compute_cross_vendor_evidence_hash(packet)
        v_key = "|".join(sorted(vendors))
        cache_key = f"{analysis_type}:{v_key}:{category or ''}"
        if existing_xv_hashes.get(cache_key) == ev_hash:
            return  # unchanged

        async with xv_sem:
            payload = json.dumps(packet, separators=(",", ":"), sort_keys=True, default=str)
            synthesis: dict[str, Any] | None = None
            item_tokens = 0

            for attempt in range(xv_max_attempts):
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=payload),
                ]
                try:
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
                    text = result.get("response", "").strip()
                    usage = result.get("usage", {})
                    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    item_tokens += tokens
                    _tokens += tokens

                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                    parsed = parse_json_response(text, recover_truncated=True)
                    if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
                        synthesis = normalize_cross_vendor_contract(parsed, analysis_type)
                        break
                except asyncio.TimeoutError:
                    if attempt + 1 >= xv_max_attempts:
                        logger.warning(
                            "Cross-vendor synthesis timed out: %s %s after %.1fs",
                            analysis_type,
                            vendors,
                            xv_llm_timeout_seconds,
                        )
                except Exception:
                    if attempt + 1 >= xv_max_attempts:
                        logger.warning(
                            "Cross-vendor synthesis failed: %s %s",
                            analysis_type, vendors, exc_info=True,
                        )

            if synthesis is None:
                _failed += 1
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
    return _succeeded, _failed, _tokens, _mirrored
