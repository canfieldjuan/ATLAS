"""Fine-Tune Labs tracing client.

Sends hierarchical trace spans to the Fine-Tune Labs observability API.
Non-blocking -- tracing failures never affect Atlas operation.

Usage:
    from atlas_brain.services.tracing import tracer

    ctx = tracer.start_span("agent.process", "llm_call", model_name="qwen3:14b")
    ...
    tracer.end_span(ctx, status="completed", output_tokens=150)
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from ..config import settings

logger = logging.getLogger("atlas.tracing")

_TRACE_BUSINESS_KEYS = (
    "account_id",
    "product",
    "workflow",
    "report_type",
    "event_type",
    "crm_provider",
    "vendor_name",
    "company_name",
    "signal_type",
    "entity_type",
    "entity_id",
    "correction_type",
    "source_name",
    "subscription_id",
)


def _metadata_text_value(metadata: object, key: str) -> str | None:
    if not isinstance(metadata, dict):
        return None

    def _normalize(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, (str, int, float)):
            text = str(value).strip()
            return text or None
        return None

    direct = _normalize(metadata.get(key))
    if direct:
        return direct

    business = metadata.get("business")
    if isinstance(business, dict):
        return _normalize(business.get(key))
    return None


@dataclass
class SpanContext:
    """Active trace span context."""

    trace_id: str
    span_id: str
    span_name: str
    operation_type: str
    start_time: float  # monotonic ns
    start_iso: str
    parent_span_id: Optional[str] = None
    model_name: Optional[str] = None
    model_provider: Optional[str] = None
    session_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FTLTracingClient:
    """Async client for Fine-Tune Labs trace ingestion API."""

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._enabled: bool = False
        self._base_url: str = ""
        self._api_key: str = ""
        self._user_id: str = ""

    def configure(
        self,
        base_url: str,
        api_key: str,
        user_id: str = "",
        enabled: bool = True,
    ) -> None:
        """Configure the tracing client. Called once at startup."""
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._user_id = user_id
        self._enabled = enabled and bool(api_key)
        if self._enabled:
            logger.info("FTL tracing enabled -> %s", self._base_url)
        else:
            logger.info("FTL tracing disabled (no API key or explicitly off)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # --- Span lifecycle ---

    def start_span(
        self,
        span_name: str,
        operation_type: str,
        parent: Optional[SpanContext] = None,
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SpanContext:
        """Start a new trace span (sync -- no I/O)."""
        trace_id = parent.trace_id if parent else f"atlas_{uuid.uuid4().hex[:16]}"
        return SpanContext(
            trace_id=trace_id,
            span_id=f"span_{uuid.uuid4().hex[:16]}",
            span_name=span_name,
            operation_type=operation_type,
            start_time=time.monotonic_ns(),
            start_iso=datetime.now(timezone.utc).isoformat(),
            parent_span_id=parent.span_id if parent else None,
            model_name=model_name,
            model_provider=model_provider,
            session_id=session_id,
            metadata=metadata or {},
        )

    def end_span(
        self,
        ctx: SpanContext,
        status: str = "completed",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        ttft_ms: Optional[float] = None,
        inference_time_ms: Optional[float] = None,
        queue_time_ms: Optional[float] = None,
        cached_tokens: Optional[int] = None,
        cache_write_tokens: Optional[int] = None,
        billable_input_tokens: Optional[int] = None,
        context_tokens: Optional[int] = None,
        retrieval_latency_ms: Optional[float] = None,
        rag_graph_used: Optional[bool] = None,
        rag_nodes_retrieved: Optional[int] = None,
        rag_chunks_used: Optional[int] = None,
        api_endpoint: Optional[str] = None,
        request_headers_sanitized: Optional[dict[str, Any]] = None,
        provider_request_id: Optional[str] = None,
        reasoning: Optional[str] = None,
        duration_ms_override: Optional[float] = None,
        cost_usd_override: Optional[float] = None,
    ) -> None:
        """End a span and fire-and-forget the trace payload.

        If *duration_ms_override* is set, it replaces the monotonic-clock
        duration.  Useful for ``trace_llm_call()`` where the LLM call
        happened before the span was created.
        """
        if duration_ms_override is not None and duration_ms_override > 0:
            duration_ms = duration_ms_override
        else:
            duration_ns = time.monotonic_ns() - ctx.start_time
            duration_ms = duration_ns / 1_000_000
        end_iso = datetime.now(timezone.utc).isoformat()

        payload: dict[str, Any] = {
            "trace_id": ctx.trace_id,
            "span_id": ctx.span_id,
            "span_name": ctx.span_name,
            "operation_type": ctx.operation_type,
            "start_time": ctx.start_iso,
            "end_time": end_iso,
            "duration_ms": int(round(duration_ms)),
            "status": "failed" if error_message else status,
            "model_name": ctx.model_name,
            "model_provider": ctx.model_provider,
            "session_tag": ctx.session_id,
            "metadata": {**ctx.metadata, **(metadata or {})},
        }
        reasoning_text = _derive_reasoning_text(payload["metadata"], reasoning)
        if reasoning_text:
            payload["reasoning"] = reasoning_text

        if ctx.parent_span_id:
            payload["parent_trace_id"] = ctx.parent_span_id

        if self._user_id:
            payload["user_id"] = self._user_id

        if input_data:
            payload["input_data"] = _truncate(input_data, 50_000)
        if output_data:
            payload["output_data"] = _truncate(output_data, 10_000)

        if input_tokens is not None:
            payload["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            payload["output_tokens"] = int(output_tokens)
        if input_tokens is not None or output_tokens is not None:
            payload["total_tokens"] = int(input_tokens or 0) + int(output_tokens or 0)
        if cached_tokens is not None:
            payload["cached_tokens"] = int(cached_tokens)
        if cache_write_tokens is not None:
            payload["cache_write_tokens"] = int(cache_write_tokens)
        if billable_input_tokens is not None:
            payload["billable_input_tokens"] = int(billable_input_tokens)

        # Calculate cost from model pricing
        if cost_usd_override is not None:
            payload["cost_usd"] = round(float(cost_usd_override), 6)
        elif (
            input_tokens is not None
            or output_tokens is not None
            or cached_tokens is not None
            or cache_write_tokens is not None
        ) and ctx.model_provider:
            cost = settings.ftl_tracing.pricing.cost_usd(
                ctx.model_provider or "",
                ctx.model_name or "",
                int(input_tokens or 0),
                int(output_tokens or 0),
                cached_tokens=int(cached_tokens or 0),
                cache_write_tokens=int(cache_write_tokens or 0),
                billable_input_tokens=(
                    int(billable_input_tokens)
                    if billable_input_tokens is not None
                    else None
                ),
            )
            if cost > 0:
                payload["cost_usd"] = round(cost, 6)

        if ttft_ms is not None:
            payload["ttft_ms"] = int(round(ttft_ms))
        if inference_time_ms is not None:
            payload["inference_time_ms"] = int(round(inference_time_ms))
        if queue_time_ms is not None:
            payload["queue_time_ms"] = int(round(queue_time_ms))
        if context_tokens is not None:
            payload["context_tokens"] = int(context_tokens)
        if retrieval_latency_ms is not None:
            payload["retrieval_latency_ms"] = int(round(retrieval_latency_ms))
        if rag_graph_used is not None:
            payload["rag_graph_used"] = rag_graph_used
        if rag_nodes_retrieved is not None:
            payload["rag_nodes_retrieved"] = int(rag_nodes_retrieved)
        if rag_chunks_used is not None:
            payload["rag_chunks_used"] = int(rag_chunks_used)
        if api_endpoint:
            payload["api_endpoint"] = api_endpoint
        if request_headers_sanitized:
            payload["request_headers_sanitized"] = request_headers_sanitized
        if provider_request_id:
            payload["provider_request_id"] = provider_request_id

        if error_message:
            payload["error_message"] = error_message
            payload["error_type"] = error_type or "unknown"

        if output_tokens is not None and output_tokens > 0:
            # Prefer inference_time_ms (actual generation time) over total
            # wall-clock duration, which includes network RTT and queue wait.
            gen_ms = inference_time_ms or duration_ms
            if gen_ms and gen_ms > 0:
                payload["tokens_per_second"] = round(output_tokens / (gen_ms / 1000), 1)

        self._dispatch(payload)

    def emit_child_span(
        self,
        parent: SpanContext,
        span_name: str,
        operation_type: str,
        start_iso: str,
        end_iso: str,
        duration_ms: float,
        status: str = "completed",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
        cache_write_tokens: Optional[int] = None,
        billable_input_tokens: Optional[int] = None,
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        reasoning: Optional[str] = None,
    ) -> None:
        """Emit a child span with explicit timestamps.

        Useful when sub-step timings are known after the parent completes.
        """

        duration_val = max(0, int(round(duration_ms)))
        payload: dict[str, Any] = {
            "trace_id": parent.trace_id,
            "span_id": f"span_{uuid.uuid4().hex[:16]}",
            "parent_trace_id": parent.span_id,
            "span_name": span_name,
            "operation_type": operation_type,
            "start_time": start_iso,
            "end_time": end_iso,
            "duration_ms": duration_val,
            "status": "failed" if error_message else status,
            "model_name": parent.model_name,
            "model_provider": parent.model_provider,
            "session_tag": parent.session_id,
            "metadata": {**(metadata or {})},
        }
        reasoning_text = _derive_reasoning_text(payload["metadata"], reasoning)
        if reasoning_text:
            payload["reasoning"] = reasoning_text

        if self._user_id:
            payload["user_id"] = self._user_id

        if input_data:
            payload["input_data"] = _truncate(input_data, 50_000)
        if output_data:
            payload["output_data"] = _truncate(output_data, 10_000)
        if input_tokens is not None:
            payload["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            payload["output_tokens"] = int(output_tokens)
        if input_tokens is not None or output_tokens is not None:
            payload["total_tokens"] = int(input_tokens or 0) + int(output_tokens or 0)
        if cached_tokens is not None:
            payload["cached_tokens"] = int(cached_tokens)
        if cache_write_tokens is not None:
            payload["cache_write_tokens"] = int(cache_write_tokens)
        if billable_input_tokens is not None:
            payload["billable_input_tokens"] = int(billable_input_tokens)

        # Calculate cost from parent span's model info
        if (
            input_tokens is not None
            or output_tokens is not None
            or cached_tokens is not None
            or cache_write_tokens is not None
        ) and parent.model_provider:
            cost = settings.ftl_tracing.pricing.cost_usd(
                parent.model_provider or "",
                parent.model_name or "",
                int(input_tokens or 0),
                int(output_tokens or 0),
                cached_tokens=int(cached_tokens or 0),
                cache_write_tokens=int(cache_write_tokens or 0),
                billable_input_tokens=(
                    int(billable_input_tokens)
                    if billable_input_tokens is not None
                    else None
                ),
            )
            if cost > 0:
                payload["cost_usd"] = round(cost, 6)

        if error_message:
            payload["error_message"] = error_message
            payload["error_type"] = error_type or "unknown"

        if output_tokens is not None and output_tokens > 0 and duration_val > 0:
            payload["tokens_per_second"] = round(output_tokens / (duration_val / 1000), 1)

        self._dispatch(payload)

    def _dispatch(self, payload: dict[str, Any]) -> None:
        """Fire-and-forget send; never block caller."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send(payload, use_shared_resources=True))
        except RuntimeError:
            # No running event loop (sync context) -- send in a thread
            import threading

            def _bg_send() -> None:
                try:
                    asyncio.run(self._send(payload, use_shared_resources=False))
                except Exception:
                    pass

            threading.Thread(target=_bg_send, daemon=True).start()

    async def _send(
        self,
        payload: dict[str, Any],
        *,
        use_shared_resources: bool,
    ) -> None:
        """POST trace to FTL API + store locally. Errors are logged, never raised."""
        span_id = payload.get("span_id", "?")

        # Store locally for cost dashboard queries (always, even without FTL)
        await self._store_local(payload, use_shared_pool=use_shared_resources)

        # Send to FTL remote API only when enabled
        if not self._enabled:
            return

        try:
            if use_shared_resources:
                client = await self._ensure_client()
                resp = await self._post_remote(client, payload)
            else:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await self._post_remote(client, payload)
            if resp.status_code >= 400:
                logger.warning(
                    "FTL trace rejected (%d) span=%s",
                    resp.status_code,
                    span_id,
                )
            else:
                logger.info(
                    "FTL trace sent span=%s status=%s tokens=%d+%d cost=$%.4f model=%s",
                    span_id,
                    payload.get("status"),
                    payload.get("input_tokens", 0),
                    payload.get("output_tokens", 0),
                    payload.get("cost_usd", 0),
                    payload.get("model_name"),
                )
        except Exception as e:
            logger.warning("FTL trace send failed span=%s: %s", span_id, type(e).__name__)

    async def _post_remote(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
    ) -> httpx.Response:
        """Send a trace payload to the remote tracing API."""
        return await client.post(
            f"{self._base_url}/api/analytics/traces",
            json=payload,
            headers={
                "X-API-Key": self._api_key,
                "Content-Type": "application/json",
            },
        )

    async def _store_local(
        self,
        payload: dict[str, Any],
        *,
        use_shared_pool: bool = True,
    ) -> None:
        """Insert usage row into local llm_usage table (best-effort)."""
        if not any(
            payload.get(key)
            for key in (
                "input_tokens",
                "output_tokens",
                "cached_tokens",
                "cache_write_tokens",
                "billable_input_tokens",
            )
        ):
            return
        # Extract high-value attribution fields from metadata for top-level columns.
        meta = payload.get("metadata") or {}
        vendor_name = _metadata_text_value(meta, "vendor_name")
        run_id = _metadata_text_value(meta, "run_id")
        source_name = _metadata_text_value(meta, "source_name")
        event_type = _metadata_text_value(meta, "event_type")
        entity_type = _metadata_text_value(meta, "entity_type")
        entity_id = _metadata_text_value(meta, "entity_id")
        # PR-D3: account_id rides in metadata (or sentinel for atlas's
        # internal pipeline). PR-D4's LLM Gateway router will set this
        # via metadata={"account_id": user.account_id, ...}.
        account_id_str = _metadata_text_value(meta, "account_id")

        try:
            from ..storage.database import get_db_pool

            pool = get_db_pool()
            if not pool.is_initialized:
                return
            # PR-D3: account_id read from metadata (set by PR-D4's
            # LLM Gateway router) or sentinel for atlas's internal
            # pipeline.
            account_id = account_id_str or "00000000-0000-0000-0000-000000000000"
            query = """INSERT INTO llm_usage
                       (span_name, operation_type, model_name, model_provider,
                        input_tokens, output_tokens, total_tokens, cost_usd,
                        duration_ms, ttft_ms, inference_time_ms, queue_time_ms,
                        tokens_per_second, billable_input_tokens, cached_tokens,
                        cache_write_tokens, api_endpoint, provider_request_id,
                        status, metadata, vendor_name, run_id, source_name,
                        event_type, entity_type, entity_id, account_id)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27)"""
            args = (
                payload.get("span_name", ""),
                payload.get("operation_type", "llm_call"),
                payload.get("model_name"),
                payload.get("model_provider"),
                payload.get("input_tokens", 0),
                payload.get("output_tokens", 0),
                payload.get("total_tokens", 0),
                payload.get("cost_usd", 0),
                payload.get("duration_ms", 0),
                payload.get("ttft_ms"),
                payload.get("inference_time_ms"),
                payload.get("queue_time_ms"),
                payload.get("tokens_per_second"),
                payload.get("billable_input_tokens", payload.get("input_tokens", 0)),
                payload.get("cached_tokens", 0),
                payload.get("cache_write_tokens", 0),
                payload.get("api_endpoint"),
                payload.get("provider_request_id"),
                payload.get("status", "completed"),
                json.dumps(payload.get("metadata", {})),
                vendor_name,
                run_id,
                source_name,
                event_type,
                entity_type,
                entity_id,
                account_id,
            )
            if use_shared_pool:
                await pool.execute(query, *args)
                return
            conn = await pool.acquire_raw()
            try:
                await conn.execute(query, *args)
            finally:
                await conn.close()
        except Exception as exc:
            logger.warning("_store_local failed for span=%s: %s", payload.get("span_name"), exc)


def _truncate(data: Any, max_chars: int) -> Any:
    """Truncate data to fit within size limits."""
    s = json.dumps(data, default=str)
    if len(s) <= max_chars:
        return data
    return {"_truncated": True, "preview": s[:max_chars]}


def _derive_reasoning_text(metadata: dict[str, Any], explicit_reasoning: Optional[str]) -> Optional[str]:
    """Promote a compact reasoning summary to the top-level trace field."""
    if explicit_reasoning:
        text = explicit_reasoning.strip()
        return text[: settings.ftl_tracing.max_reasoning_chars] if text else None
    reasoning_meta = metadata.get("reasoning")
    if not isinstance(reasoning_meta, dict):
        return None
    for key in ("summary", "triage", "raw_preview"):
        value = reasoning_meta.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text[: settings.ftl_tracing.max_reasoning_chars]
    return None


def build_business_trace_context(**kwargs: Any) -> dict[str, Any]:
    """Build a compact business-context payload for trace metadata."""
    if not settings.ftl_tracing.capture_business_context:
        return {}

    payload: dict[str, Any] = {}
    for key in _TRACE_BUSINESS_KEYS:
        value = kwargs.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
            payload[key] = value[:250]
            continue
        payload[key] = value
    return payload


def build_reasoning_trace_context(
    *,
    decision: Optional[dict[str, Any]] = None,
    evidence: Optional[dict[str, Any]] = None,
    triage_reasoning: Optional[str] = None,
    rationale: Optional[str] = None,
    raw_reasoning: Optional[str] = None,
) -> dict[str, Any]:
    """Build a structured reasoning artifact for trace metadata."""
    cfg = settings.ftl_tracing
    if not cfg.capture_reasoning_summaries:
        return {}

    payload: dict[str, Any] = {}
    if decision:
        payload["decision"] = _truncate(_clean_trace_value(decision), 3000)
    if evidence:
        payload["evidence"] = _truncate(_clean_trace_value(evidence), 3000)
    if triage_reasoning:
        payload["triage"] = triage_reasoning[:cfg.max_reasoning_chars]
    if rationale:
        payload["summary"] = rationale[:cfg.max_reasoning_chars]
    if cfg.capture_raw_reasoning and raw_reasoning:
        payload["raw_preview"] = raw_reasoning[:cfg.max_reasoning_chars]
    return payload


def _clean_trace_value(value: Any) -> Any:
    """Normalize trace values to compact JSON-safe structures."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            normalized = _clean_trace_value(item)
            if normalized is not None:
                cleaned[str(key)] = normalized
        return cleaned or None
    if isinstance(value, (list, tuple, set)):
        cleaned = []
        for item in value:
            normalized = _clean_trace_value(item)
            if normalized is not None:
                cleaned.append(normalized)
        return cleaned[:20] or None
    if isinstance(value, str):
        text = value.strip()
        return text[:1000] if text else None
    if isinstance(value, (bool, int, float)):
        return value
    with suppress(Exception):
        text = str(value).strip()
        if text:
            return text[:1000]
    return None


# --- Module-level singleton ---
tracer = FTLTracingClient()
