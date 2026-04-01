"""Shared Anthropic Message Batch helpers for B2B content workloads."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from ...config import settings
from ...pipelines.llm import trace_llm_call
from ..llm.anthropic import AnthropicLLM
from ..protocols import Message

logger = logging.getLogger("atlas.services.b2b.anthropic_batch")

_BATCH_ENDPOINT = "https://api.anthropic.com/v1/messages/batches"
_BATCH_DISCOUNT_FACTOR = 0.5


@dataclass(frozen=True)
class AnthropicBatchItem:
    custom_id: str
    artifact_type: str
    artifact_id: str
    messages: Sequence[Message | dict[str, Any]]
    max_tokens: int
    temperature: float
    vendor_name: str | None = None
    trace_span_name: str = ""
    trace_metadata: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)
    cached_response_text: str | None = None
    cached_usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnthropicBatchItemResult:
    custom_id: str
    artifact_type: str
    artifact_id: str
    vendor_name: str | None
    response_text: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    success: bool = False
    fallback_required: bool = False
    error_text: str | None = None
    provider_request_id: str | None = None
    item_status: str = ""


@dataclass
class AnthropicBatchExecution:
    local_batch_id: str
    provider_batch_id: str | None
    stage_id: str
    task_name: str
    status: str
    results_by_custom_id: dict[str, AnthropicBatchItemResult]
    submitted_items: int
    cache_prefiltered_items: int
    fallback_single_call_items: int
    completed_items: int
    failed_items: int


def _persisted_request_metadata(item: AnthropicBatchItem) -> dict[str, Any]:
    metadata = dict(item.request_metadata or {})
    metadata.setdefault("_trace_span_name", item.trace_span_name)
    metadata.setdefault("_trace_metadata", dict(item.trace_metadata or {}))
    metadata.setdefault(
        "_messages",
        [
            {
                "role": _message_from_value(message).role,
                "content": _message_from_value(message).content,
            }
            for message in item.messages
        ],
    )
    metadata.setdefault("_max_tokens", int(item.max_tokens))
    metadata.setdefault("_temperature", float(item.temperature))
    return metadata


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _message_from_value(value: Message | dict[str, Any]) -> Message:
    if isinstance(value, Message):
        return value
    if isinstance(value, dict):
        return Message(
            role=str(value.get("role") or ""),
            content=str(value.get("content") or ""),
            tool_calls=value.get("tool_calls"),
            tool_call_id=value.get("tool_call_id"),
        )
    raise TypeError(f"Unsupported Anthropic batch message type: {type(value)!r}")


def _resolve_pool(pool: Any | None) -> Any | None:
    if pool is not None:
        return pool
    from ...storage.database import get_db_pool

    db_pool = get_db_pool()
    if not db_pool.is_initialized:
        return None
    return db_pool


async def _safe_execute(pool: Any | None, query: str, *args: Any) -> None:
    if pool is None:
        return
    try:
        await pool.execute(query, *args)
    except Exception:
        logger.exception("anthropic_batch.db_execute_failed")


def _standard_cost_usd(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
    billable_input_tokens: int | None = None,
) -> float:
    return float(
        settings.ftl_tracing.pricing.cost_usd(
            provider,
            model,
            input_tokens,
            output_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            billable_input_tokens=billable_input_tokens,
        )
    )


def _batch_cost_usd(**kwargs: Any) -> float:
    return round(_standard_cost_usd(**kwargs) * _BATCH_DISCOUNT_FACTOR, 6)


def _extract_message_text_and_usage(message: Any) -> tuple[str, dict[str, Any], str | None]:
    text_parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if text:
                text_parts.append(str(text))
    usage = getattr(message, "usage", None)
    cache_read = getattr(usage, "cache_read_input_tokens", None) if usage is not None else None
    cache_write = (
        getattr(usage, "cache_creation_input_tokens", None)
        if usage is not None
        else None
    )
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0
    usage_payload = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "billable_input_tokens": input_tokens,
        "cached_tokens": int(cache_read or 0),
        "cache_write_tokens": int(cache_write or 0),
    }
    provider_request_id = getattr(message, "id", None)
    return "\n".join(text_parts).strip() or None, usage_payload, str(provider_request_id or "") or None


def _trace_batched_item(
    *,
    item: AnthropicBatchItem,
    llm: AnthropicLLM,
    usage: dict[str, Any],
    response_text: str,
    provider_request_id: str | None,
    duration_ms: float,
    provider_batch_id: str,
) -> None:
    metadata = dict(item.trace_metadata)
    metadata.setdefault("execution_strategy", "anthropic_batch")
    metadata.setdefault("stage_id", "")
    metadata["provider_batch_id"] = provider_batch_id
    if item.vendor_name and not metadata.get("vendor_name"):
        metadata["vendor_name"] = item.vendor_name

    batch_cost = _batch_cost_usd(
        provider=str(getattr(llm, "name", "") or ""),
        model=str(getattr(llm, "model", "") or ""),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cached_tokens=int(usage.get("cached_tokens") or 0),
        cache_write_tokens=int(usage.get("cache_write_tokens") or 0),
        billable_input_tokens=(
            int(usage["billable_input_tokens"])
            if usage.get("billable_input_tokens") is not None
            else None
        ),
    )

    trace_llm_call(
        span_name=item.trace_span_name,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cached_tokens=int(usage.get("cached_tokens") or 0),
        cache_write_tokens=int(usage.get("cache_write_tokens") or 0),
        billable_input_tokens=(
            int(usage["billable_input_tokens"])
            if usage.get("billable_input_tokens") is not None
            else None
        ),
        model=str(getattr(llm, "model", "") or ""),
        provider=str(getattr(llm, "name", "") or ""),
        duration_ms=duration_ms,
        metadata=metadata,
        input_data={
            "messages": [
                {
                    "role": _message_from_value(msg).role,
                    "content": _message_from_value(msg).content[:500],
                }
                for msg in item.messages
            ]
        },
        output_data={"response": response_text[:2000]} if response_text else None,
        api_endpoint=_BATCH_ENDPOINT,
        provider_request_id=provider_request_id,
        cost_usd_override=batch_cost,
    )


async def _insert_batch_job(
    *,
    pool: Any | None,
    local_batch_id: str,
    stage_id: str,
    task_name: str,
    run_id: str | None,
    total_items: int,
    cache_prefiltered_items: int,
    metadata: dict[str, Any] | None,
) -> None:
    await _safe_execute(
        pool,
        """
        INSERT INTO anthropic_message_batches (
            id, stage_id, task_name, run_id, status, total_items,
            cache_prefiltered_items, metadata
        ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8::jsonb)
        ON CONFLICT (id) DO NOTHING
        """,
        local_batch_id,
        stage_id,
        task_name,
        run_id,
        "preparing",
        total_items,
        cache_prefiltered_items,
        json.dumps(metadata or {}, default=str),
    )


async def _insert_batch_item(
    *,
    pool: Any | None,
    local_batch_id: str,
    stage_id: str,
    item: AnthropicBatchItem,
    status: str,
    cache_prefiltered: bool,
    fallback_single_call: bool,
    response_text: str | None = None,
    usage: dict[str, Any] | None = None,
    provider_request_id: str | None = None,
    error_text: str | None = None,
) -> None:
    usage_payload = dict(usage or {})
    cost_usd = 0.0
    if response_text and usage_payload:
        cost_usd = _batch_cost_usd(
            provider="anthropic",
            model=str(item.trace_metadata.get("model") or ""),
            input_tokens=int(usage_payload.get("input_tokens") or 0),
            output_tokens=int(usage_payload.get("output_tokens") or 0),
            cached_tokens=int(usage_payload.get("cached_tokens") or 0),
            cache_write_tokens=int(usage_payload.get("cache_write_tokens") or 0),
            billable_input_tokens=(
                int(usage_payload["billable_input_tokens"])
                if usage_payload.get("billable_input_tokens") is not None
                else None
            ),
        )
    await _safe_execute(
        pool,
        """
        INSERT INTO anthropic_message_batch_items (
            id, batch_id, custom_id, stage_id, artifact_type, artifact_id,
            vendor_name, status, cache_prefiltered, fallback_single_call,
            response_text, input_tokens, billable_input_tokens, cached_tokens,
            cache_write_tokens, output_tokens, cost_usd, provider_request_id,
            error_text, request_metadata, completed_at
        ) VALUES (
            gen_random_uuid(), $1::uuid, $2, $3, $4, $5,
            $6, $7, $8, $9,
            $10, $11, $12, $13,
            $14, $15, $16, $17,
            $18, $19::jsonb,
            CASE WHEN $7 IN ('cache_hit', 'batch_succeeded', 'fallback_succeeded', 'fallback_failed', 'batch_errored', 'batch_expired', 'batch_canceled') THEN NOW() ELSE NULL END
        )
        ON CONFLICT (batch_id, custom_id) DO UPDATE
        SET status = EXCLUDED.status,
            cache_prefiltered = EXCLUDED.cache_prefiltered,
            fallback_single_call = EXCLUDED.fallback_single_call,
            response_text = EXCLUDED.response_text,
            input_tokens = EXCLUDED.input_tokens,
            billable_input_tokens = EXCLUDED.billable_input_tokens,
            cached_tokens = EXCLUDED.cached_tokens,
            cache_write_tokens = EXCLUDED.cache_write_tokens,
            output_tokens = EXCLUDED.output_tokens,
            cost_usd = EXCLUDED.cost_usd,
            provider_request_id = EXCLUDED.provider_request_id,
            error_text = EXCLUDED.error_text,
            request_metadata = EXCLUDED.request_metadata,
            completed_at = EXCLUDED.completed_at
        """,
        local_batch_id,
        item.custom_id,
        stage_id,
        item.artifact_type,
        item.artifact_id,
        item.vendor_name,
        status,
        cache_prefiltered,
        fallback_single_call,
        response_text,
        int(usage_payload.get("input_tokens") or 0),
        int(usage_payload.get("billable_input_tokens") or 0),
        int(usage_payload.get("cached_tokens") or 0),
        int(usage_payload.get("cache_write_tokens") or 0),
        int(usage_payload.get("output_tokens") or 0),
        cost_usd,
        provider_request_id,
        error_text,
        json.dumps(_persisted_request_metadata(item), default=str),
    )


async def _update_batch_job(
    *,
    pool: Any | None,
    local_batch_id: str,
    status: str,
    provider_batch_id: str | None = None,
    submitted_items: int | None = None,
    fallback_single_call_items: int | None = None,
    completed_items: int | None = None,
    failed_items: int | None = None,
    estimated_sequential_cost_usd: float | None = None,
    estimated_batch_cost_usd: float | None = None,
    provider_error: str | None = None,
    completed_at: datetime | None = None,
) -> None:
    await _safe_execute(
        pool,
        """
        UPDATE anthropic_message_batches
        SET status = $2,
            provider_batch_id = COALESCE($3, provider_batch_id),
            submitted_items = COALESCE($4, submitted_items),
            fallback_single_call_items = COALESCE($5, fallback_single_call_items),
            completed_items = COALESCE($6, completed_items),
            failed_items = COALESCE($7, failed_items),
            estimated_sequential_cost_usd = COALESCE($8, estimated_sequential_cost_usd),
            estimated_batch_cost_usd = COALESCE($9, estimated_batch_cost_usd),
            provider_error = COALESCE($10, provider_error),
            submitted_at = CASE WHEN $3 IS NOT NULL THEN COALESCE(submitted_at, NOW()) ELSE submitted_at END,
            completed_at = COALESCE($11, completed_at),
            updated_at = NOW()
        WHERE id = $1::uuid
        """,
        local_batch_id,
        status,
        provider_batch_id,
        submitted_items,
        fallback_single_call_items,
        completed_items,
        failed_items,
        estimated_sequential_cost_usd,
        estimated_batch_cost_usd,
        provider_error,
        completed_at,
    )


def _build_request_params(llm: AnthropicLLM, item: AnthropicBatchItem) -> dict[str, Any]:
    messages = [_message_from_value(message) for message in item.messages]
    system_prompt, api_messages = llm._convert_messages(messages)  # type: ignore[attr-defined]
    params: dict[str, Any] = {
        "model": llm.model,
        "messages": api_messages,
        "max_tokens": int(item.max_tokens),
        "temperature": float(item.temperature),
    }
    if system_prompt:
        params["system"] = system_prompt
    return params


def _rebuild_item_from_row(row: Any) -> AnthropicBatchItem:
    metadata = row.get("request_metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    raw_messages = metadata.get("_messages") or []
    messages = [
        Message(
            role=str((message or {}).get("role") or ""),
            content=str((message or {}).get("content") or ""),
        )
        for message in raw_messages
        if isinstance(message, dict)
    ]
    trace_metadata = metadata.get("_trace_metadata")
    if not isinstance(trace_metadata, dict):
        trace_metadata = {}
    public_request_metadata = {
        key: value for key, value in metadata.items() if not str(key).startswith("_")
    }
    return AnthropicBatchItem(
        custom_id=str(row.get("custom_id") or ""),
        artifact_type=str(row.get("artifact_type") or ""),
        artifact_id=str(row.get("artifact_id") or ""),
        vendor_name=str(row.get("vendor_name") or "") or None,
        messages=messages,
        max_tokens=int(metadata.get("_max_tokens") or 0 or 256),
        temperature=float(metadata.get("_temperature") or 0.0),
        trace_span_name=str(metadata.get("_trace_span_name") or ""),
        trace_metadata=trace_metadata,
        request_metadata=public_request_metadata,
    )


async def submit_anthropic_message_batch(
    *,
    llm: Any,
    stage_id: str,
    task_name: str,
    items: Sequence[AnthropicBatchItem],
    run_id: str | None = None,
    min_batch_size: int = 2,
    batch_metadata: dict[str, Any] | None = None,
    pool: Any | None = None,
) -> AnthropicBatchExecution:
    """Submit a batch and persist durable item state without waiting for results."""
    local_batch_id = str(uuid.uuid4())
    db_pool = _resolve_pool(pool)
    results: dict[str, AnthropicBatchItemResult] = {}
    cache_prefiltered_items = 0
    pending_items: list[AnthropicBatchItem] = []

    await _insert_batch_job(
        pool=db_pool,
        local_batch_id=local_batch_id,
        stage_id=stage_id,
        task_name=task_name,
        run_id=run_id,
        total_items=len(items),
        cache_prefiltered_items=sum(1 for item in items if item.cached_response_text is not None),
        metadata=batch_metadata,
    )

    for item in items:
        if item.cached_response_text is not None:
            cache_prefiltered_items += 1
            results[item.custom_id] = AnthropicBatchItemResult(
                custom_id=item.custom_id,
                artifact_type=item.artifact_type,
                artifact_id=item.artifact_id,
                vendor_name=item.vendor_name,
                response_text=item.cached_response_text,
                usage=dict(item.cached_usage),
                cached=True,
                success=True,
                item_status="cache_hit",
            )
            await _insert_batch_item(
                pool=db_pool,
                local_batch_id=local_batch_id,
                stage_id=stage_id,
                item=item,
                status="cache_hit",
                cache_prefiltered=True,
                fallback_single_call=False,
                response_text=item.cached_response_text,
                usage=item.cached_usage,
            )
        else:
            pending_items.append(item)
            await _insert_batch_item(
                pool=db_pool,
                local_batch_id=local_batch_id,
                stage_id=stage_id,
                item=item,
                status="pending",
                cache_prefiltered=False,
                fallback_single_call=False,
            )

    if not pending_items:
        await _update_batch_job(
            pool=db_pool,
            local_batch_id=local_batch_id,
            status="prefiltered_only",
            submitted_items=0,
            completed_items=0,
            failed_items=0,
            fallback_single_call_items=0,
            completed_at=_utcnow(),
        )
        return AnthropicBatchExecution(
            local_batch_id=local_batch_id,
            provider_batch_id=None,
            stage_id=stage_id,
            task_name=task_name,
            status="prefiltered_only",
            results_by_custom_id=results,
            submitted_items=0,
            cache_prefiltered_items=cache_prefiltered_items,
            fallback_single_call_items=0,
            completed_items=0,
            failed_items=0,
        )

    if not isinstance(llm, AnthropicLLM) or getattr(llm, "_async_client", None) is None:
        for item in pending_items:
            results[item.custom_id] = AnthropicBatchItemResult(
                custom_id=item.custom_id,
                artifact_type=item.artifact_type,
                artifact_id=item.artifact_id,
                vendor_name=item.vendor_name,
                fallback_required=True,
                error_text="anthropic_batch_unavailable",
                item_status="fallback_pending",
            )
            await _insert_batch_item(
                pool=db_pool,
                local_batch_id=local_batch_id,
                stage_id=stage_id,
                item=item,
                status="fallback_pending",
                cache_prefiltered=False,
                fallback_single_call=True,
                error_text="anthropic_batch_unavailable",
            )
        await _update_batch_job(
            pool=db_pool,
            local_batch_id=local_batch_id,
            status="fallback_only",
            submitted_items=0,
            fallback_single_call_items=len(pending_items),
            completed_items=0,
            failed_items=len(pending_items),
            provider_error="anthropic_batch_unavailable",
            completed_at=_utcnow(),
        )
        return AnthropicBatchExecution(
            local_batch_id=local_batch_id,
            provider_batch_id=None,
            stage_id=stage_id,
            task_name=task_name,
            status="fallback_only",
            results_by_custom_id=results,
            submitted_items=0,
            cache_prefiltered_items=cache_prefiltered_items,
            fallback_single_call_items=len(pending_items),
            completed_items=0,
            failed_items=len(pending_items),
        )

    if len(pending_items) < int(min_batch_size):
        for item in pending_items:
            results[item.custom_id] = AnthropicBatchItemResult(
                custom_id=item.custom_id,
                artifact_type=item.artifact_type,
                artifact_id=item.artifact_id,
                vendor_name=item.vendor_name,
                fallback_required=True,
                error_text="batch_min_items_not_met",
                item_status="fallback_pending",
            )
            await _insert_batch_item(
                pool=db_pool,
                local_batch_id=local_batch_id,
                stage_id=stage_id,
                item=item,
                status="fallback_pending",
                cache_prefiltered=False,
                fallback_single_call=True,
                error_text="batch_min_items_not_met",
            )
        await _update_batch_job(
            pool=db_pool,
            local_batch_id=local_batch_id,
            status="fallback_only",
            submitted_items=0,
            fallback_single_call_items=len(pending_items),
            completed_items=0,
            failed_items=len(pending_items),
            provider_error="batch_min_items_not_met",
            completed_at=_utcnow(),
        )
        return AnthropicBatchExecution(
            local_batch_id=local_batch_id,
            provider_batch_id=None,
            stage_id=stage_id,
            task_name=task_name,
            status="fallback_only",
            results_by_custom_id=results,
            submitted_items=0,
            cache_prefiltered_items=cache_prefiltered_items,
            fallback_single_call_items=len(pending_items),
            completed_items=0,
            failed_items=len(pending_items),
        )

    request_items = [
        {"custom_id": item.custom_id, "params": _build_request_params(llm, item)}
        for item in pending_items
    ]
    client = llm._async_client
    batch = await client.messages.batches.create(requests=request_items)
    provider_batch_id = str(batch.id)
    await _update_batch_job(
        pool=db_pool,
        local_batch_id=local_batch_id,
        status=str(batch.processing_status),
        provider_batch_id=provider_batch_id,
        submitted_items=len(pending_items),
    )
    return AnthropicBatchExecution(
        local_batch_id=local_batch_id,
        provider_batch_id=provider_batch_id,
        stage_id=stage_id,
        task_name=task_name,
        status=str(batch.processing_status),
        results_by_custom_id=results,
        submitted_items=len(pending_items),
        cache_prefiltered_items=cache_prefiltered_items,
        fallback_single_call_items=0,
        completed_items=0,
        failed_items=0,
    )


async def reconcile_anthropic_message_batch(
    *,
    llm: Any,
    local_batch_id: str,
    pool: Any | None = None,
    timeout_seconds: float | None = None,
) -> AnthropicBatchExecution:
    """Reconcile one submitted batch by polling provider status and ingesting results."""
    db_pool = _resolve_pool(pool)
    row = await db_pool.fetchrow(
        """
        SELECT id, stage_id, task_name, run_id, status, provider_batch_id
        FROM anthropic_message_batches
        WHERE id = $1::uuid
        """,
        local_batch_id,
    ) if db_pool is not None else None
    if not row:
        raise ValueError(f"Unknown Anthropic batch id: {local_batch_id}")

    item_rows = await db_pool.fetch(
        """
        SELECT *
        FROM anthropic_message_batch_items
        WHERE batch_id = $1::uuid
        ORDER BY created_at ASC
        """,
        local_batch_id,
    ) if db_pool is not None else []

    results: dict[str, AnthropicBatchItemResult] = {}
    for item_row in item_rows:
        status = str(item_row.get("status") or "")
        response_text = item_row.get("response_text")
        usage = {
            "input_tokens": int(item_row.get("input_tokens") or 0),
            "billable_input_tokens": int(item_row.get("billable_input_tokens") or 0),
            "cached_tokens": int(item_row.get("cached_tokens") or 0),
            "cache_write_tokens": int(item_row.get("cache_write_tokens") or 0),
            "output_tokens": int(item_row.get("output_tokens") or 0),
        }
        if status in {
            "cache_hit",
            "batch_succeeded",
            "fallback_succeeded",
            "fallback_failed",
            "batch_errored",
            "batch_expired",
            "batch_canceled",
        }:
            results[str(item_row["custom_id"])] = AnthropicBatchItemResult(
                custom_id=str(item_row["custom_id"]),
                artifact_type=str(item_row["artifact_type"]),
                artifact_id=str(item_row["artifact_id"]),
                vendor_name=str(item_row.get("vendor_name") or "") or None,
                response_text=str(response_text) if response_text else None,
                usage=usage,
                cached=status == "cache_hit",
                success=status in {"cache_hit", "batch_succeeded", "fallback_succeeded"},
                fallback_required=status == "fallback_pending",
                error_text=str(item_row.get("error_text") or "") or None,
                provider_request_id=str(item_row.get("provider_request_id") or "") or None,
                item_status=status,
            )

    pending_rows = [row for row in item_rows if str(row.get("status") or "") == "pending"]
    cache_prefiltered_items = sum(1 for item_row in item_rows if bool(item_row.get("cache_prefiltered")))
    fallback_items = sum(1 for item_row in item_rows if bool(item_row.get("fallback_single_call")))

    if not pending_rows or not row.get("provider_batch_id") or not isinstance(llm, AnthropicLLM) or getattr(llm, "_async_client", None) is None:
        completed_items = sum(1 for item_row in item_rows if str(item_row.get("status") or "") in {"cache_hit", "batch_succeeded", "fallback_succeeded"})
        failed_items = sum(1 for item_row in item_rows if str(item_row.get("status") or "") in {"fallback_failed", "batch_errored", "batch_expired", "batch_canceled"})
        return AnthropicBatchExecution(
            local_batch_id=str(row["id"]),
            provider_batch_id=str(row.get("provider_batch_id") or "") or None,
            stage_id=str(row.get("stage_id") or ""),
            task_name=str(row.get("task_name") or ""),
            status=str(row.get("status") or ""),
            results_by_custom_id=results,
            submitted_items=len(pending_rows),
            cache_prefiltered_items=cache_prefiltered_items,
            fallback_single_call_items=fallback_items,
            completed_items=completed_items,
            failed_items=failed_items,
        )

    timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else settings.b2b_churn.anthropic_batch_timeout_seconds
    )
    client = llm._async_client
    provider_batch_id = str(row["provider_batch_id"])
    batch = await client.messages.batches.retrieve(provider_batch_id, timeout=timeout)
    await _update_batch_job(
        pool=db_pool,
        local_batch_id=str(row["id"]),
        status=str(batch.processing_status),
        provider_batch_id=provider_batch_id,
        submitted_items=len(pending_rows),
    )
    if str(batch.processing_status) != "ended":
        completed_items = sum(1 for item_row in item_rows if str(item_row.get("status") or "") in {"cache_hit", "batch_succeeded", "fallback_succeeded"})
        failed_items = sum(1 for item_row in item_rows if str(item_row.get("status") or "") in {"fallback_failed", "batch_errored", "batch_expired", "batch_canceled"})
        return AnthropicBatchExecution(
            local_batch_id=str(row["id"]),
            provider_batch_id=provider_batch_id,
            stage_id=str(row.get("stage_id") or ""),
            task_name=str(row.get("task_name") or ""),
            status=str(batch.processing_status),
            results_by_custom_id=results,
            submitted_items=len(pending_rows),
            cache_prefiltered_items=cache_prefiltered_items,
            fallback_single_call_items=fallback_items,
            completed_items=completed_items,
            failed_items=failed_items,
        )

    result_stream = await client.messages.batches.results(provider_batch_id, timeout=timeout)
    result_map: dict[str, Any] = {}
    async for result_row in result_stream:
        result_map[str(result_row.custom_id)] = result_row.result

    completed_items = 0
    failed_items = 0
    fallback_items = 0
    sequential_cost_usd = 0.0
    batch_cost_usd = 0.0
    for item_row in pending_rows:
        item = _rebuild_item_from_row(item_row)
        raw_result = result_map.get(item.custom_id)
        if raw_result is None:
            failed_items += 1
            fallback_items += 1
            results[item.custom_id] = AnthropicBatchItemResult(
                custom_id=item.custom_id,
                artifact_type=item.artifact_type,
                artifact_id=item.artifact_id,
                vendor_name=item.vendor_name,
                fallback_required=True,
                error_text="missing_batch_result",
                item_status="fallback_pending",
            )
            await _insert_batch_item(
                pool=db_pool,
                local_batch_id=str(row["id"]),
                stage_id=str(row["stage_id"]),
                item=item,
                status="fallback_pending",
                cache_prefiltered=False,
                fallback_single_call=True,
                error_text="missing_batch_result",
            )
            continue

        result_type = str(getattr(raw_result, "type", "") or "")
        if result_type == "succeeded":
            response_text, usage, provider_request_id = _extract_message_text_and_usage(raw_result.message)
            if response_text:
                completed_items += 1
                standard_cost = _standard_cost_usd(
                    provider=str(getattr(llm, "name", "") or ""),
                    model=str(getattr(llm, "model", "") or ""),
                    input_tokens=int(usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    cached_tokens=int(usage.get("cached_tokens") or 0),
                    cache_write_tokens=int(usage.get("cache_write_tokens") or 0),
                    billable_input_tokens=(
                        int(usage["billable_input_tokens"])
                        if usage.get("billable_input_tokens") is not None
                        else None
                    ),
                )
                sequential_cost_usd += standard_cost
                batch_cost_usd += round(standard_cost * _BATCH_DISCOUNT_FACTOR, 6)
                _trace_batched_item(
                    item=item,
                    llm=llm,
                    usage=usage,
                    response_text=response_text,
                    provider_request_id=provider_request_id,
                    duration_ms=0.0,
                    provider_batch_id=provider_batch_id,
                )
                results[item.custom_id] = AnthropicBatchItemResult(
                    custom_id=item.custom_id,
                    artifact_type=item.artifact_type,
                    artifact_id=item.artifact_id,
                    vendor_name=item.vendor_name,
                    response_text=response_text,
                    usage=usage,
                    success=True,
                    provider_request_id=provider_request_id,
                    item_status="batch_succeeded",
                )
                await _insert_batch_item(
                    pool=db_pool,
                    local_batch_id=str(row["id"]),
                    stage_id=str(row["stage_id"]),
                    item=item,
                    status="batch_succeeded",
                    cache_prefiltered=False,
                    fallback_single_call=False,
                    response_text=response_text,
                    usage=usage,
                    provider_request_id=provider_request_id,
                )
                continue

        failed_items += 1
        fallback_items += 1
        if result_type == "errored":
            error = getattr(raw_result, "error", None)
            error_text = str(getattr(error, "message", "") or getattr(error, "type", "") or "batch_errored")
            item_status = "batch_errored"
        elif result_type == "expired":
            error_text = "batch_expired"
            item_status = "batch_expired"
        else:
            error_text = "batch_canceled"
            item_status = "batch_canceled"
        results[item.custom_id] = AnthropicBatchItemResult(
            custom_id=item.custom_id,
            artifact_type=item.artifact_type,
            artifact_id=item.artifact_id,
            vendor_name=item.vendor_name,
            fallback_required=True,
            error_text=error_text,
            item_status="fallback_pending",
        )
        await _insert_batch_item(
            pool=db_pool,
            local_batch_id=str(row["id"]),
            stage_id=str(row["stage_id"]),
            item=item,
            status=item_status,
            cache_prefiltered=False,
            fallback_single_call=True,
            error_text=error_text,
        )

    await _update_batch_job(
        pool=db_pool,
        local_batch_id=str(row["id"]),
        status="ended",
        provider_batch_id=provider_batch_id,
        submitted_items=len(pending_rows),
        fallback_single_call_items=fallback_items,
        completed_items=completed_items,
        failed_items=failed_items,
        estimated_sequential_cost_usd=round(sequential_cost_usd, 6),
        estimated_batch_cost_usd=round(batch_cost_usd, 6),
        completed_at=_utcnow(),
    )
    return AnthropicBatchExecution(
        local_batch_id=str(row["id"]),
        provider_batch_id=provider_batch_id,
        stage_id=str(row.get("stage_id") or ""),
        task_name=str(row.get("task_name") or ""),
        status="ended",
        results_by_custom_id=results,
        submitted_items=len(pending_rows),
        cache_prefiltered_items=cache_prefiltered_items,
        fallback_single_call_items=fallback_items,
        completed_items=completed_items,
        failed_items=failed_items,
    )


async def run_anthropic_message_batch(
    *,
    llm: Any,
    stage_id: str,
    task_name: str,
    items: Sequence[AnthropicBatchItem],
    run_id: str | None = None,
    min_batch_size: int = 2,
    poll_interval_seconds: float | None = None,
    timeout_seconds: float | None = None,
    batch_metadata: dict[str, Any] | None = None,
    pool: Any | None = None,
) -> AnthropicBatchExecution:
    """Submit and reconcile one Anthropic Message Batch with safe fallback flags."""
    local_batch_id = str(uuid.uuid4())
    db_pool = _resolve_pool(pool)
    results: dict[str, AnthropicBatchItemResult] = {}
    cache_prefiltered_items = 0
    pending_items: list[AnthropicBatchItem] = []

    await _insert_batch_job(
        pool=db_pool,
        local_batch_id=local_batch_id,
        stage_id=stage_id,
        task_name=task_name,
        run_id=run_id,
        total_items=len(items),
        cache_prefiltered_items=sum(1 for item in items if item.cached_response_text is not None),
        metadata=batch_metadata,
    )

    for item in items:
        if item.cached_response_text is not None:
            cache_prefiltered_items += 1
            result = AnthropicBatchItemResult(
                custom_id=item.custom_id,
                artifact_type=item.artifact_type,
                artifact_id=item.artifact_id,
                vendor_name=item.vendor_name,
                response_text=item.cached_response_text,
                usage=dict(item.cached_usage),
                cached=True,
                success=True,
                item_status="cache_hit",
            )
            results[item.custom_id] = result
            await _insert_batch_item(
                pool=db_pool,
                local_batch_id=local_batch_id,
                stage_id=stage_id,
                item=item,
                status="cache_hit",
                cache_prefiltered=True,
                fallback_single_call=False,
                response_text=item.cached_response_text,
                usage=item.cached_usage,
            )
        else:
            pending_items.append(item)
            await _insert_batch_item(
                pool=db_pool,
                local_batch_id=local_batch_id,
                stage_id=stage_id,
                item=item,
                status="pending",
                cache_prefiltered=False,
                fallback_single_call=False,
            )

    if not pending_items:
        await _update_batch_job(
            pool=db_pool,
            local_batch_id=local_batch_id,
            status="prefiltered_only",
            submitted_items=0,
            completed_items=0,
            failed_items=0,
            fallback_single_call_items=0,
            completed_at=_utcnow(),
        )
        return AnthropicBatchExecution(
            local_batch_id=local_batch_id,
            provider_batch_id=None,
            stage_id=stage_id,
            task_name=task_name,
            status="prefiltered_only",
            results_by_custom_id=results,
            submitted_items=0,
            cache_prefiltered_items=cache_prefiltered_items,
            fallback_single_call_items=0,
            completed_items=0,
            failed_items=0,
        )

    if not isinstance(llm, AnthropicLLM) or getattr(llm, "_async_client", None) is None:
        for item in pending_items:
            results[item.custom_id] = AnthropicBatchItemResult(
                custom_id=item.custom_id,
                artifact_type=item.artifact_type,
                artifact_id=item.artifact_id,
                vendor_name=item.vendor_name,
                fallback_required=True,
                error_text="anthropic_batch_unavailable",
                item_status="fallback_pending",
            )
            await _insert_batch_item(
                pool=db_pool,
                local_batch_id=local_batch_id,
                stage_id=stage_id,
                item=item,
                status="fallback_pending",
                cache_prefiltered=False,
                fallback_single_call=True,
                error_text="anthropic_batch_unavailable",
            )
        await _update_batch_job(
            pool=db_pool,
            local_batch_id=local_batch_id,
            status="fallback_only",
            submitted_items=0,
            fallback_single_call_items=len(pending_items),
            completed_items=0,
            failed_items=len(pending_items),
            provider_error="anthropic_batch_unavailable",
            completed_at=_utcnow(),
        )
        return AnthropicBatchExecution(
            local_batch_id=local_batch_id,
            provider_batch_id=None,
            stage_id=stage_id,
            task_name=task_name,
            status="fallback_only",
            results_by_custom_id=results,
            submitted_items=0,
            cache_prefiltered_items=cache_prefiltered_items,
            fallback_single_call_items=len(pending_items),
            completed_items=0,
            failed_items=len(pending_items),
        )

    if len(pending_items) < int(min_batch_size):
        for item in pending_items:
            results[item.custom_id] = AnthropicBatchItemResult(
                custom_id=item.custom_id,
                artifact_type=item.artifact_type,
                artifact_id=item.artifact_id,
                vendor_name=item.vendor_name,
                fallback_required=True,
                error_text="batch_min_items_not_met",
                item_status="fallback_pending",
            )
            await _insert_batch_item(
                pool=db_pool,
                local_batch_id=local_batch_id,
                stage_id=stage_id,
                item=item,
                status="fallback_pending",
                cache_prefiltered=False,
                fallback_single_call=True,
                error_text="batch_min_items_not_met",
            )
        await _update_batch_job(
            pool=db_pool,
            local_batch_id=local_batch_id,
            status="fallback_only",
            submitted_items=0,
            fallback_single_call_items=len(pending_items),
            completed_items=0,
            failed_items=len(pending_items),
            provider_error="batch_min_items_not_met",
            completed_at=_utcnow(),
        )
        return AnthropicBatchExecution(
            local_batch_id=local_batch_id,
            provider_batch_id=None,
            stage_id=stage_id,
            task_name=task_name,
            status="fallback_only",
            results_by_custom_id=results,
            submitted_items=0,
            cache_prefiltered_items=cache_prefiltered_items,
            fallback_single_call_items=len(pending_items),
            completed_items=0,
            failed_items=len(pending_items),
        )

    request_items = [
        {
            "custom_id": item.custom_id,
            "params": _build_request_params(llm, item),
        }
        for item in pending_items
    ]

    poll_seconds = float(
        poll_interval_seconds
        if poll_interval_seconds is not None
        else settings.b2b_churn.anthropic_batch_poll_interval_seconds
    )
    timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else settings.b2b_churn.anthropic_batch_timeout_seconds
    )
    client = llm._async_client
    provider_batch_id: str | None = None
    started = time.monotonic()

    try:
        batch = await client.messages.batches.create(requests=request_items, timeout=timeout)
        provider_batch_id = str(batch.id)
        await _update_batch_job(
            pool=db_pool,
            local_batch_id=local_batch_id,
            status=str(batch.processing_status),
            provider_batch_id=provider_batch_id,
            submitted_items=len(pending_items),
        )

        while str(batch.processing_status) != "ended":
            if time.monotonic() - started >= timeout:
                raise TimeoutError("anthropic_batch_timeout")
            await asyncio.sleep(poll_seconds)
            batch = await client.messages.batches.retrieve(provider_batch_id, timeout=timeout)
            await _update_batch_job(
                pool=db_pool,
                local_batch_id=local_batch_id,
                status=str(batch.processing_status),
                provider_batch_id=provider_batch_id,
                submitted_items=len(pending_items),
            )

        result_stream = await client.messages.batches.results(provider_batch_id, timeout=timeout)
        result_map: dict[str, Any] = {}
        async for row in result_stream:
            result_map[str(row.custom_id)] = row.result
    except Exception as exc:
        error_text = str(exc) or type(exc).__name__
        for item in pending_items:
            results[item.custom_id] = AnthropicBatchItemResult(
                custom_id=item.custom_id,
                artifact_type=item.artifact_type,
                artifact_id=item.artifact_id,
                vendor_name=item.vendor_name,
                fallback_required=True,
                error_text=error_text,
                item_status="fallback_pending",
            )
            await _insert_batch_item(
                pool=db_pool,
                local_batch_id=local_batch_id,
                stage_id=stage_id,
                item=item,
                status="fallback_pending",
                cache_prefiltered=False,
                fallback_single_call=True,
                error_text=error_text,
            )
        await _update_batch_job(
            pool=db_pool,
            local_batch_id=local_batch_id,
            status="timed_out" if "timeout" in error_text.lower() else "failed",
            provider_batch_id=provider_batch_id,
            submitted_items=len(pending_items),
            fallback_single_call_items=len(pending_items),
            completed_items=0,
            failed_items=len(pending_items),
            provider_error=error_text[:500],
            completed_at=_utcnow(),
        )
        return AnthropicBatchExecution(
            local_batch_id=local_batch_id,
            provider_batch_id=provider_batch_id,
            stage_id=stage_id,
            task_name=task_name,
            status="timed_out" if "timeout" in error_text.lower() else "failed",
            results_by_custom_id=results,
            submitted_items=len(pending_items),
            cache_prefiltered_items=cache_prefiltered_items,
            fallback_single_call_items=len(pending_items),
            completed_items=0,
            failed_items=len(pending_items),
        )

    completed_items = 0
    failed_items = 0
    fallback_items = 0
    sequential_cost_usd = 0.0
    batch_cost_usd = 0.0

    for item in pending_items:
        raw_result = result_map.get(item.custom_id)
        if raw_result is None:
            failed_items += 1
            fallback_items += 1
            results[item.custom_id] = AnthropicBatchItemResult(
                custom_id=item.custom_id,
                artifact_type=item.artifact_type,
                artifact_id=item.artifact_id,
                vendor_name=item.vendor_name,
                fallback_required=True,
                error_text="missing_batch_result",
                item_status="fallback_pending",
            )
            await _insert_batch_item(
                pool=db_pool,
                local_batch_id=local_batch_id,
                stage_id=stage_id,
                item=item,
                status="fallback_pending",
                cache_prefiltered=False,
                fallback_single_call=True,
                error_text="missing_batch_result",
            )
            continue

        result_type = str(getattr(raw_result, "type", "") or "")
        if result_type == "succeeded":
            response_text, usage, provider_request_id = _extract_message_text_and_usage(raw_result.message)
            if response_text:
                completed_items += 1
                standard_cost = _standard_cost_usd(
                    provider=str(getattr(llm, "name", "") or ""),
                    model=str(getattr(llm, "model", "") or ""),
                    input_tokens=int(usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    cached_tokens=int(usage.get("cached_tokens") or 0),
                    cache_write_tokens=int(usage.get("cache_write_tokens") or 0),
                    billable_input_tokens=(
                        int(usage["billable_input_tokens"])
                        if usage.get("billable_input_tokens") is not None
                        else None
                    ),
                )
                sequential_cost_usd += standard_cost
                batch_cost_usd += round(standard_cost * _BATCH_DISCOUNT_FACTOR, 6)
                _trace_batched_item(
                    item=item,
                    llm=llm,
                    usage=usage,
                    response_text=response_text,
                    provider_request_id=provider_request_id,
                    duration_ms=(time.monotonic() - started) * 1000,
                    provider_batch_id=provider_batch_id or "",
                )
                results[item.custom_id] = AnthropicBatchItemResult(
                    custom_id=item.custom_id,
                    artifact_type=item.artifact_type,
                    artifact_id=item.artifact_id,
                    vendor_name=item.vendor_name,
                    response_text=response_text,
                    usage=usage,
                    success=True,
                    provider_request_id=provider_request_id,
                    item_status="batch_succeeded",
                )
                await _insert_batch_item(
                    pool=db_pool,
                    local_batch_id=local_batch_id,
                    stage_id=stage_id,
                    item=item,
                    status="batch_succeeded",
                    cache_prefiltered=False,
                    fallback_single_call=False,
                    response_text=response_text,
                    usage=usage,
                    provider_request_id=provider_request_id,
                )
                continue

        failed_items += 1
        fallback_items += 1
        error_text = ""
        if result_type == "errored":
            error = getattr(raw_result, "error", None)
            error_text = str(getattr(error, "message", "") or getattr(error, "type", "") or "batch_errored")
            item_status = "batch_errored"
        elif result_type == "expired":
            error_text = "batch_expired"
            item_status = "batch_expired"
        else:
            error_text = "batch_canceled"
            item_status = "batch_canceled"
        results[item.custom_id] = AnthropicBatchItemResult(
            custom_id=item.custom_id,
            artifact_type=item.artifact_type,
            artifact_id=item.artifact_id,
            vendor_name=item.vendor_name,
            fallback_required=True,
            error_text=error_text,
            item_status="fallback_pending",
        )
        await _insert_batch_item(
            pool=db_pool,
            local_batch_id=local_batch_id,
            stage_id=stage_id,
            item=item,
            status=item_status,
            cache_prefiltered=False,
            fallback_single_call=True,
            error_text=error_text,
        )

    await _update_batch_job(
        pool=db_pool,
        local_batch_id=local_batch_id,
        status="ended",
        provider_batch_id=provider_batch_id,
        submitted_items=len(pending_items),
        fallback_single_call_items=fallback_items,
        completed_items=completed_items,
        failed_items=failed_items,
        estimated_sequential_cost_usd=round(sequential_cost_usd, 6),
        estimated_batch_cost_usd=round(batch_cost_usd, 6),
        completed_at=_utcnow(),
    )
    return AnthropicBatchExecution(
        local_batch_id=local_batch_id,
        provider_batch_id=provider_batch_id,
        stage_id=stage_id,
        task_name=task_name,
        status="ended",
        results_by_custom_id=results,
        submitted_items=len(pending_items),
        cache_prefiltered_items=cache_prefiltered_items,
        fallback_single_call_items=fallback_items,
        completed_items=completed_items,
        failed_items=failed_items,
    )


async def mark_batch_fallback_result(
    *,
    batch_id: str,
    custom_id: str,
    succeeded: bool,
    pool: Any | None = None,
    error_text: str | None = None,
    response_text: str | None = None,
) -> None:
    """Finalize a fallback item after the caller runs the single-call path."""
    db_pool = _resolve_pool(pool)
    await _safe_execute(
        db_pool,
        """
        UPDATE anthropic_message_batch_items
        SET status = $3,
            error_text = COALESCE($4, error_text),
            response_text = COALESCE($5, response_text),
            fallback_single_call = TRUE,
            completed_at = NOW()
        WHERE batch_id = $1::uuid
          AND custom_id = $2
        """,
        batch_id,
        custom_id,
        "fallback_succeeded" if succeeded else "fallback_failed",
        error_text,
        response_text,
    )
