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

import logging
import time
import uuid as _uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth.dependencies import AuthUser, require_llm_plan
from ..pipelines.llm import trace_llm_call
from ..services.byok_keys import lookup_provider_key_async
from ..services.llm.anthropic import convert_messages
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
    content: str = Field(..., max_length=200_000)


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

    from ..services.llm.anthropic import AnthropicLLM

    llm = AnthropicLLM(model=body.model, api_key=api_key)
    try:
        # AnthropicLLM.load() is synchronous (returns None). Do NOT await.
        llm.load()
    except Exception as exc:
        logger.warning(
            "llm_gateway.chat load failed account=%s provider=%s model=%s: %s",
            user.account_id,
            body.provider,
            body.model,
            exc,
        )
        raise HTTPException(status_code=502, detail="Provider key invalid or load failed")

    messages = [Message(role=m.role, content=m.content) for m in body.messages]
    system_prompt, api_messages = convert_messages(messages)

    create_kwargs: dict[str, Any] = {
        "model": llm.model,
        "messages": api_messages,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
    }
    if system_prompt:
        create_kwargs["system"] = system_prompt

    # Bypass llm.chat_async() so we can capture full response (text +
    # usage). Codex P1 fix on PR-D4: chat_async returned only text,
    # which left input_tokens=output_tokens=0 for trace_llm_call.
    # _store_local then short-circuited (all token fields falsy) and
    # /usage returned empty for every customer call.
    start = time.monotonic()
    try:
        response = await llm._async_client.messages.create(**create_kwargs)
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
