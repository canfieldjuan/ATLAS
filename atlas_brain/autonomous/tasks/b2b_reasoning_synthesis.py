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


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: generate reasoning synthesis per vendor."""
    cfg = settings.b2b_churn

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
    llm = resolve_stratified_llm(rcfg)
    if llm is None:
        return {"_skip_synthesis": "No LLM available for reasoning synthesis"}

    from ...reasoning.single_pass_prompts.battle_card_reasoning import (
        BATTLE_CARD_REASONING_PROMPT,
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
            payload = json.dumps(
                packet.to_llm_payload(),
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
                    Message(role="system", content=BATTLE_CARD_REASONING_PROMPT),
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

                    result = await asyncio.to_thread(
                        llm.chat,
                        messages=messages,
                        max_tokens=rcfg.max_tokens,
                        temperature=rcfg.temperature,
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
            except Exception:
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

    elapsed = round(time.monotonic() - t0, 1)
    logger.info(
        "Reasoning synthesis v2 complete: %d succeeded, %d failed "
        "(%d validation), %d skipped, %d tokens, %.1fs",
        succeeded, failed, validation_failures, skipped, total_tokens, elapsed,
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
    }
