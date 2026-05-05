"""LLM Gateway customer-facing API endpoints (PR-D4).

Wraps the ``extracted_llm_infrastructure`` engine in a per-account
HTTP surface mounted at ``/api/v1/llm/*``. Customers authenticate
via API keys (``atls_live_*``, PR-D1), get plan-tier gating
(PR-D2), and per-account scoping on usage / caches (PR-D3) for
free.

Endpoints in this PR:

  POST /api/v1/llm/chat   -- sync chat completion (Anthropic only for v1)
  GET  /api/v1/llm/usage  -- per-account spend by provider

Deferred to PR-D4b:
  - POST /api/v1/llm/chat/stream  (SSE streaming)
  - POST /api/v1/llm/batch        (Anthropic Message Batches)
  - GET  /api/v1/llm/batch/{id}   (batch status / results)
  - POST /api/v1/llm/embed        (sync embeddings)

Provider keys come from BYOK -- the customer configures their own
Anthropic / OpenRouter / etc. keys in the dashboard (PR-D5);
``lookup_provider_key`` resolves them per request. PR-D4 ships the
BYOK service with an env-var fallback so the endpoint is testable
before PR-D5 lands the DB-backed storage.
"""

from __future__ import annotations

import json
import logging
import time
import uuid as _uuid
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..api.billing import LLM_PLAN_LIMITS
from ..auth.dependencies import AuthUser, require_llm_plan
from ..pipelines.llm import trace_llm_call
from ..services.byok_keys import lookup_provider_key_async
from ..services.llm.anthropic import convert_messages
from ..services.llm_gateway_batch import (
    CustomerBatchItem,
    get_customer_batch,
    refresh_customer_batch_status,
    submit_customer_batch,
)
from ..services.protocols import Message
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.llm_gateway")

router = APIRouter(prefix="/llm", tags=["llm-gateway"])


# Supported providers in PR-D4. Other providers (OpenRouter, Together,
# Groq) layer in subsequent PRs once their LLMService implementations
# are exercised through the gateway path.
_PROVIDERS_THIS_PR = ("anthropic",)


# ---- Request / response schemas -----------------------------------------


class ChatMessageBody(BaseModel):
    role: str = Field(..., description="system | user | assistant | tool")
    content: str = Field(..., max_length=64_000)
    # Tool-use round-trip support (Codex review fix on PR-D4b). The
    # internal Message dataclass carries these; without threading
    # them through the schema, customer requests with tool transcripts
    # were serialized into malformed Anthropic payloads (assistant
    # tool_calls dropped; tool result tool_call_id missing).
    tool_calls: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="For assistant messages that invoke tools.",
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="For tool messages that return a result.",
    )


def _to_internal_message(m: "ChatMessageBody") -> Message:
    """Convert the API schema to the internal Message dataclass with
    full tool-use fields preserved."""
    return Message(
        role=m.role,
        content=m.content,
        tool_calls=m.tool_calls,
        tool_call_id=m.tool_call_id,
    )


class ChatRequest(BaseModel):
    provider: str = Field(..., description="anthropic (only supported provider in PR-D4)")
    model: str = Field(..., min_length=1, max_length=128)
    messages: list[ChatMessageBody] = Field(..., min_length=1, max_length=200)
    max_tokens: int = Field(default=1024, ge=1, le=128_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    provider: str
    model: str
    response: str
    usage: ChatUsage


class UsageBreakdownRow(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0


class UsageResponse(BaseModel):
    account_id: str
    period_start: str
    period_end: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    by_provider: list[UsageBreakdownRow]


# ---- Helpers ------------------------------------------------------------


def _validate_chat_provider(provider: str) -> None:
    if provider not in _PROVIDERS_THIS_PR:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Provider '{provider}' is not supported by the chat endpoint "
                f"in this release. Supported: {sorted(_PROVIDERS_THIS_PR)}."
            ),
        )


async def _resolve_byok_or_503(pool, provider: str, account_id: str) -> str:
    """Look up the customer's stored BYOK key for ``provider``. Raises
    HTTPException 503 when no key is configured.

    Calls the async DB-backed resolver so keys added via
    ``POST /api/v1/byok-keys`` are honored for live gateway calls.
    The resolver also has an env-var fallback for local dev where
    the DB has no row yet.
    """
    raw = await lookup_provider_key_async(pool, provider, account_id)
    if not raw:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No BYOK key configured for provider '{provider}'. "
                "Configure your provider key in the dashboard before "
                "using this endpoint."
            ),
        )
    return raw


# ---- /chat (sync) -------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    user: AuthUser = Depends(require_llm_plan("llm_trial")),
) -> ChatResponse:
    """Sync chat completion. Account-scoped via the API key auth path
    (PR-D1) and plan-gated to llm_trial+ (PR-D2)."""
    _validate_chat_provider(body.provider)

    pool = get_db_pool()
    api_key = await _resolve_byok_or_503(pool, body.provider, user.account_id)

    # Normalize the model id via AnthropicLLM (alias map) but do NOT
    # call load() -- atlas's AnthropicLLM is shaped for long-running
    # pipeline use (one instance kept alive across calls). Per-request
    # gateway calls leak httpx connections through ``_async_client``,
    # so we use ``AsyncAnthropic`` directly in an ``async with`` block
    # below for proper resource cleanup. Codex P2 fix on PR-D4b.
    from anthropic import AsyncAnthropic

    from ..services.llm.anthropic import AnthropicLLM

    llm = AnthropicLLM(model=body.model, api_key=api_key)
    messages = [_to_internal_message(m) for m in body.messages]
    system_prompt, api_messages = convert_messages(messages)

    create_kwargs: dict[str, Any] = {
        "model": llm.model,
        "messages": api_messages,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
    }
    if system_prompt:
        create_kwargs["system"] = system_prompt

    # Capture full response (text + usage) via direct SDK call.
    # PR-D4 review fix already addressed the "chat_async returned
    # only text -> zero token usage" bug; PR-D4b also closes the
    # httpx pool after each request via async-with.
    start = time.monotonic()
    try:
        async with AsyncAnthropic(api_key=api_key) as client:
            response = await client.messages.create(**create_kwargs)
    except Exception as exc:
        logger.warning(
            "llm_gateway.chat call failed account=%s provider=%s model=%s: %s",
            user.account_id,
            body.provider,
            body.model,
            exc,
        )
        raise HTTPException(status_code=502, detail="Provider call failed")
    duration_ms = (time.monotonic() - start) * 1000.0

    text_parts: list[str] = []
    for block in response.content or []:
        if getattr(block, "type", None) == "text":
            text_parts.append(getattr(block, "text", "") or "")
    text = "\n".join(text_parts).strip()

    usage_obj = getattr(response, "usage", None)
    input_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
    cached_tokens = int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0)
    cache_write_tokens = int(getattr(usage_obj, "cache_creation_input_tokens", 0) or 0)
    provider_request_id = getattr(response, "id", None)

    response_id = f"llm_{_uuid.uuid4().hex[:24]}"

    # Usage tracking now carries real token counts so /usage rollups
    # populate. trace_llm_call writes to llm_usage via the FTL tracer;
    # PR-D3 added account_id to that INSERT (read from metadata).
    try:
        trace_llm_call(
            span_name="llm_gateway.chat",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            billable_input_tokens=input_tokens,
            model=body.model,
            provider=body.provider,
            duration_ms=duration_ms,
            provider_request_id=str(provider_request_id) if provider_request_id else None,
            metadata={
                "account_id": user.account_id,
                "request_id": response_id,
                "endpoint": "llm_gateway.chat",
            },
        )
    except Exception:
        logger.exception("llm_gateway.chat usage tracking failed")

    return ChatResponse(
        id=response_id,
        provider=body.provider,
        model=body.model,
        response=text,
        usage=ChatUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


# ---- /usage (read-only) -------------------------------------------------


@router.get("/usage", response_model=UsageResponse)
async def usage(
    days: int = Query(default=30, ge=1, le=365),
    user: AuthUser = Depends(require_llm_plan("llm_trial")),
) -> UsageResponse:
    """Per-account spend over the last ``days`` days, by provider+model.

    Reads ``llm_usage`` filtered on ``account_id`` (PR-D3 added this
    column). Atlas's internal pipeline writes use the SENTINEL UUID
    so they never appear in customer-facing rollups.
    """
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    rows = await pool.fetch(
        """
        SELECT model_provider, model_name,
               COALESCE(SUM(input_tokens), 0)::bigint  AS input_tokens,
               COALESCE(SUM(output_tokens), 0)::bigint AS output_tokens,
               COALESCE(SUM(total_tokens), 0)::bigint  AS total_tokens,
               COALESCE(SUM(cost_usd), 0)::float       AS cost_usd,
               COUNT(*)::bigint                         AS call_count,
               MIN(created_at)                          AS period_start,
               MAX(created_at)                          AS period_end
        FROM llm_usage
        WHERE account_id = $1
          AND created_at >= NOW() - ($2 || ' days')::interval
        GROUP BY model_provider, model_name
        ORDER BY total_tokens DESC
        """,
        _uuid.UUID(user.account_id),
        str(days),
    )

    breakdown: list[UsageBreakdownRow] = []
    total_input = 0
    total_output = 0
    total_cost = 0.0
    period_start = None
    period_end = None
    for row in rows:
        breakdown.append(
            UsageBreakdownRow(
                provider=row["model_provider"],
                model=row["model_name"],
                input_tokens=int(row["input_tokens"] or 0),
                output_tokens=int(row["output_tokens"] or 0),
                total_tokens=int(row["total_tokens"] or 0),
                cost_usd=float(row["cost_usd"] or 0.0),
                call_count=int(row["call_count"] or 0),
            )
        )
        total_input += int(row["input_tokens"] or 0)
        total_output += int(row["output_tokens"] or 0)
        total_cost += float(row["cost_usd"] or 0.0)
        if period_start is None or (row["period_start"] and row["period_start"] < period_start):
            period_start = row["period_start"]
        if period_end is None or (row["period_end"] and row["period_end"] > period_end):
            period_end = row["period_end"]

    def _fmt(value):
        if value is None:
            return ""
        try:
            return value.isoformat()
        except AttributeError:
            return str(value)

    return UsageResponse(
        account_id=user.account_id,
        period_start=_fmt(period_start),
        period_end=_fmt(period_end),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cost_usd=total_cost,
        by_provider=breakdown,
    )


# ---- /chat/stream (SSE streaming) -----------------------------------------


def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    """Format one SSE event line per the EventSource spec.

    SSE messages are ``event: <name>\\ndata: <json>\\n\\n``.
    Customers consume these via standard SSE clients (browser
    EventSource, curl --no-buffer, anthropic.events()).
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


async def _stream_chat_chunks(
    *,
    user: AuthUser,
    body: ChatRequest,
    api_key: str,
) -> AsyncIterator[bytes]:
    """Async generator yielding SSE bytes for /chat/stream. Emits a
    ``content`` event per text delta, then a ``usage`` event with
    final token counts, then a ``done`` event. On error: ``error``
    event with the detail.
    """
    # Use AnthropicLLM only for model-id normalization (alias map).
    # We construct AsyncAnthropic in async-with below for proper
    # connection-pool cleanup -- AnthropicLLM's _async_client would
    # leak per request. Codex P2 fix on PR-D4b.
    from anthropic import AsyncAnthropic

    from ..services.llm.anthropic import AnthropicLLM

    llm = AnthropicLLM(model=body.model, api_key=api_key)

    messages = [_to_internal_message(m) for m in body.messages]
    system_prompt, api_messages = convert_messages(messages)

    create_kwargs: dict[str, Any] = {
        "model": llm.model,
        "messages": api_messages,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
    }
    if system_prompt:
        create_kwargs["system"] = system_prompt

    response_id = f"llm_{_uuid.uuid4().hex[:24]}"
    start = time.monotonic()
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    cache_write_tokens = 0
    provider_request_id: Optional[str] = None

    try:
        async with AsyncAnthropic(api_key=api_key) as client:
            async with client.messages.stream(**create_kwargs) as stream:
                async for event in stream:
                    etype = getattr(event, "type", None)
                    if etype == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        text = getattr(delta, "text", "") if delta else ""
                        if text:
                            yield _sse_event("content", {"id": response_id, "text": text})
                    elif etype == "message_start":
                        message = getattr(event, "message", None)
                        provider_request_id = getattr(message, "id", None)
                        msg_usage = getattr(message, "usage", None)
                        if msg_usage is not None:
                            input_tokens = int(getattr(msg_usage, "input_tokens", 0) or 0)
                            cached_tokens = int(getattr(msg_usage, "cache_read_input_tokens", 0) or 0)
                            cache_write_tokens = int(
                                getattr(msg_usage, "cache_creation_input_tokens", 0) or 0
                            )
                    elif etype == "message_delta":
                        msg_usage = getattr(event, "usage", None)
                        if msg_usage is not None:
                            output_tokens = int(getattr(msg_usage, "output_tokens", output_tokens) or output_tokens)
                final = await stream.get_final_message()
                final_usage = getattr(final, "usage", None)
                if final_usage is not None:
                    # Final values overwrite any partials we captured during
                    # message_delta events.
                    input_tokens = int(getattr(final_usage, "input_tokens", input_tokens) or input_tokens)
                    output_tokens = int(getattr(final_usage, "output_tokens", output_tokens) or output_tokens)
                    cached_tokens = int(getattr(final_usage, "cache_read_input_tokens", cached_tokens) or cached_tokens)
                    cache_write_tokens = int(
                        getattr(final_usage, "cache_creation_input_tokens", cache_write_tokens) or cache_write_tokens
                    )
                if not provider_request_id:
                    provider_request_id = getattr(final, "id", None)
    except Exception as exc:
        logger.warning(
            "llm_gateway.chat_stream failed account=%s model=%s: %s",
            user.account_id,
            body.model,
            exc,
        )
        yield _sse_event("error", {"detail": "Provider stream failed"})
        return

    duration_ms = (time.monotonic() - start) * 1000.0

    # Best-effort usage tracking (account-scoped via metadata, PR-D3).
    try:
        trace_llm_call(
            span_name="llm_gateway.chat_stream",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            billable_input_tokens=input_tokens,
            model=body.model,
            provider=body.provider,
            duration_ms=duration_ms,
            provider_request_id=str(provider_request_id) if provider_request_id else None,
            metadata={
                "account_id": user.account_id,
                "request_id": response_id,
                "endpoint": "llm_gateway.chat_stream",
            },
        )
    except Exception:
        logger.exception("llm_gateway.chat_stream usage tracking failed")

    yield _sse_event(
        "usage",
        {
            "id": response_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
        },
    )
    yield _sse_event("done", {"id": response_id})


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    user: AuthUser = Depends(require_llm_plan("llm_trial")),
) -> StreamingResponse:
    """SSE-streaming chat completion. Same auth + plan gating as
    /chat. Customer reads with EventSource or any SSE client.

    Event types: ``content`` (text delta), ``usage`` (final token
    counts), ``done`` (final marker), ``error`` (provider failure).
    """
    _validate_chat_provider(body.provider)

    pool = get_db_pool()
    api_key = await _resolve_byok_or_503(pool, body.provider, user.account_id)

    return StreamingResponse(
        _stream_chat_chunks(user=user, body=body, api_key=api_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )


# ---- /batch (Anthropic Message Batches API) -----------------------------


class BatchItemBody(BaseModel):
    # Custom-id capped at 64 chars to match the limit Anthropic accepts
    # (atlas's existing batch path also sanitizes to 64) -- avoids a
    # local validation pass + remote rejection on submit. Codex review
    # fix on PR-D4b.
    custom_id: str = Field(..., min_length=1, max_length=64)
    # Messages capped at 50 per item (down from 200) -- a single item
    # rarely needs more, and the per-message char cap dropped to 64k
    # in ChatMessageBody so bulk DoS surface is bounded.
    messages: list[ChatMessageBody] = Field(..., min_length=1, max_length=50)
    max_tokens: int = Field(default=1024, ge=1, le=128_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class BatchSubmitRequest(BaseModel):
    provider: str = Field(..., description="anthropic")
    model: str = Field(..., min_length=1, max_length=128)
    # Items cap dropped from 10k to 1k. Anthropic's hard limit is
    # higher, but FastAPI/Pydantic deserializes the entire JSON
    # payload before validators run; combined with the per-item +
    # per-message caps above, 1k items is still huge (1k items x
    # 50 messages x 64k chars = ~3 GB worst-case JSON, gated by
    # downstream FastAPI body limits). Customers needing more
    # submit multiple batches.
    items: list[BatchItemBody] = Field(..., min_length=1, max_length=1_000)


class BatchView(BaseModel):
    """Display-safe view of a customer batch row."""

    id: str
    provider: str
    provider_batch_id: Optional[str] = None
    model: str
    status: str
    total_items: int
    completed_items: int
    failed_items: int
    error_text: Optional[str] = None
    created_at: str
    updated_at: str
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None


def _batch_record_to_view(record) -> BatchView:
    def _fmt(value):
        if value is None:
            return None
        try:
            return value.isoformat()
        except AttributeError:
            return str(value)

    return BatchView(
        id=str(record.id),
        provider=record.provider,
        provider_batch_id=record.provider_batch_id,
        model=record.model,
        status=record.status,
        total_items=record.total_items,
        completed_items=record.completed_items,
        failed_items=record.failed_items,
        error_text=record.error_text,
        created_at=_fmt(record.created_at) or "",
        updated_at=_fmt(record.updated_at) or "",
        submitted_at=_fmt(record.submitted_at),
        completed_at=_fmt(record.completed_at),
    )


def _require_batch_enabled(user: AuthUser) -> None:
    """Plan gate: only paying tiers (llm_starter+) get batch -- the
    50% Anthropic discount is a wedge feature reserved for paid
    plans so trial cannot abuse it for free volume."""
    plan_limits = LLM_PLAN_LIMITS.get(user.plan, {})
    if not plan_limits.get("batch_enabled", False):
        raise HTTPException(
            status_code=403,
            detail=(
                f"Plan '{user.plan}' does not include batch access. "
                "Upgrade to llm_starter or higher to use the Anthropic "
                "Message Batches API."
            ),
        )


def _validate_batch_provider(provider: str) -> None:
    if provider not in _PROVIDERS_THIS_PR:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Provider '{provider}' is not supported by /api/v1/llm/batch "
                f"in this release. Supported: {sorted(_PROVIDERS_THIS_PR)}."
            ),
        )


@router.post("/batch", response_model=BatchView, status_code=202)
async def submit_batch(
    body: BatchSubmitRequest,
    user: AuthUser = Depends(require_llm_plan("llm_starter")),
) -> BatchView:
    """Submit an Anthropic Message Batch.

    Anthropic's batch API gives a **50% discount** on input + output
    tokens vs the synchronous /chat endpoint -- the headline feature
    of the LLM Gateway product. Submission is async-accept (202): the
    batch is queued and processing happens upstream. Customer polls
    ``GET /batch/{id}`` for status; the response includes
    ``completed_items`` / ``failed_items`` counters and a terminal
    ``status`` of ``ended`` / ``canceled`` / ``expired`` when done.

    Result content (per-item response text) is not returned by this
    PR -- a follow-up adds /batch/{id}/results to fetch the JSONL
    file Anthropic produces. For PR-D4b, customers can pull results
    via Anthropic's API directly using the ``provider_batch_id``
    returned here.

    On submit failure (Anthropic rejects, validation error, etc.)
    the response body still carries the customer's ``id`` /
    ``provider_batch_id`` (if any) and ``status='submit_failed'``
    plus ``error_text`` so the caller can debug without losing the
    handle. Distinct from terminal ``status='ended'`` which means
    Anthropic finished processing successfully.
    """
    _validate_batch_provider(body.provider)
    _require_batch_enabled(user)

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    api_key = await _resolve_byok_or_503(pool, body.provider, user.account_id)

    # Codex review fix on PR-D4b: normalize the model id via
    # AnthropicLLM's alias map so deprecated names like
    # ``claude-3-5-haiku-latest`` work on /batch the same way they
    # work on /chat. Without this, callers see /chat accept the
    # alias but /batch reject it -- inconsistent gateway surface.
    from ..services.llm.anthropic import AnthropicLLM as _AnthropicLLM

    normalized_model = _AnthropicLLM(model=body.model).model

    customer_items = [
        CustomerBatchItem(
            custom_id=item.custom_id,
            messages=[_to_internal_message(m) for m in item.messages],
            max_tokens=item.max_tokens,
            temperature=item.temperature,
        )
        for item in body.items
    ]

    try:
        record = await submit_customer_batch(
            pool,
            account_id=_uuid.UUID(user.account_id),
            api_key=api_key,
            model=normalized_model,
            items=customer_items,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # ``submit_customer_batch`` no longer raises on Anthropic-side
    # rejection -- it persists the row with ``status='submit_failed'``
    # and ``error_text=<reason>`` and returns the record. The customer
    # receives the batch_id and can read ``error_text`` to debug.
    return _batch_record_to_view(record)


@router.get("/batch/{batch_id}", response_model=BatchView)
async def get_batch(
    batch_id: str,
    user: AuthUser = Depends(require_llm_plan("llm_starter")),
) -> BatchView:
    """Get the latest status of a customer batch. Polls Anthropic
    when the row is not yet terminal; returns cached row when
    already ended/canceled/expired. Account-scoped: 404 when the
    batch_id belongs to another account (avoids leaking existence).
    """
    _require_batch_enabled(user)

    try:
        batch_uuid = _uuid.UUID(batch_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Batch not found")

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    record = await get_customer_batch(
        pool,
        account_id=_uuid.UUID(user.account_id),
        batch_id=batch_uuid,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Refresh status from Anthropic if not yet terminal. Codex review
    # fix on PR-D4b: distinguish "key was revoked" (legitimate; show
    # stale row) from "DB outage or decrypt failure" (surface 503 so
    # the customer doesn't see a frozen batch silently). Probe for an
    # active byok_keys row directly first; if one exists but the
    # resolver returned None, the failure is on our side.
    from ..services.llm_gateway_batch import TERMINAL_STATUSES

    if record.status not in TERMINAL_STATUSES:
        api_key = await lookup_provider_key_async(
            pool, record.provider, user.account_id
        )
        if not api_key:
            has_active_key_row = await pool.fetchval(
                """
                SELECT 1 FROM byok_keys
                WHERE account_id = $1 AND provider = $2
                  AND revoked_at IS NULL
                LIMIT 1
                """,
                _uuid.UUID(user.account_id),
                record.provider,
            )
            if has_active_key_row:
                # Row exists but resolver couldn't decrypt it -- KEK
                # rotation drift, DB transient error, etc. Customer
                # should retry; we don't silently freeze the batch.
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Could not refresh batch status: BYOK key "
                        "lookup failed (provider key may need re-add)."
                    ),
                )
            # No active key row -- customer revoked. Return last-
            # known status; this is legitimate.
        else:
            refreshed = await refresh_customer_batch_status(
                pool,
                account_id=_uuid.UUID(user.account_id),
                batch_id=batch_uuid,
                api_key=api_key,
            )
            if refreshed is not None:
                record = refreshed

    return _batch_record_to_view(record)
