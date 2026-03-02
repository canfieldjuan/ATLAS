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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from ..config import settings

logger = logging.getLogger("atlas.tracing")


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
        context_tokens: Optional[int] = None,
        retrieval_latency_ms: Optional[float] = None,
        rag_graph_used: Optional[bool] = None,
        rag_nodes_retrieved: Optional[int] = None,
        rag_chunks_used: Optional[int] = None,
        api_endpoint: Optional[str] = None,
        request_headers_sanitized: Optional[dict[str, Any]] = None,
        provider_request_id: Optional[str] = None,
    ) -> None:
        """End a span and fire-and-forget the trace payload."""
        if not self._enabled:
            return

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

        if output_tokens is not None and output_tokens > 0 and duration_ms > 0:
            payload["tokens_per_second"] = int(round(output_tokens / (duration_ms / 1000)))

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
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit a child span with explicit timestamps.

        Useful when sub-step timings are known after the parent completes.
        """
        if not self._enabled:
            return

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
        if error_message:
            payload["error_message"] = error_message
            payload["error_type"] = error_type or "unknown"

        if output_tokens is not None and output_tokens > 0 and duration_val > 0:
            payload["tokens_per_second"] = int(round(output_tokens / (duration_val / 1000)))

        self._dispatch(payload)

    def _dispatch(self, payload: dict[str, Any]) -> None:
        """Fire-and-forget send; never block caller."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send(payload))
        except RuntimeError:
            # No event loop -- skip silently
            pass

    async def _send(self, payload: dict[str, Any]) -> None:
        """POST trace to FTL API. Errors are logged, never raised."""
        span_id = payload.get("span_id", "?")
        try:
            client = await self._ensure_client()
            resp = await client.post(
                f"{self._base_url}/api/analytics/traces",
                json=payload,
                headers={
                    "X-API-Key": self._api_key,
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code >= 400:
                logger.warning(
                    "FTL trace rejected (%d) span=%s",
                    resp.status_code,
                    span_id,
                )
            else:
                logger.info(
                    "FTL trace sent span=%s status=%s tokens=%d+%d model=%s",
                    span_id,
                    payload.get("status"),
                    payload.get("input_tokens", 0),
                    payload.get("output_tokens", 0),
                    payload.get("model_name"),
                )
        except Exception as e:
            logger.warning("FTL trace send failed span=%s: %s", span_id, type(e).__name__)


def _truncate(data: Any, max_chars: int) -> Any:
    """Truncate data to fit within size limits."""
    s = json.dumps(data, default=str)
    if len(s) <= max_chars:
        return data
    return {"_truncated": True, "preview": s[:max_chars]}


# --- Module-level singleton ---
tracer = FTLTracingClient()
