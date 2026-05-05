"""Tests for the LLM Gateway streaming + batch endpoints (PR-D4b).

Pure structural tests + SSE format pinning. DB-bound integration
tests (live submit to Anthropic, status polling) live alongside
other auth integration fixtures and are gated on a running
Postgres + a real BYOK key -- not in this file.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest


_MIG_DIR = Path(__file__).resolve().parent.parent / "atlas_brain" / "storage" / "migrations"


def _read_migration(filename: str) -> str:
    return (_MIG_DIR / filename).read_text(encoding="utf-8")


# ---- Migration ----------------------------------------------------------


def test_migration_317_creates_llm_gateway_batches_table():
    sql = _read_migration("317_llm_gateway_batches.sql")
    assert "CREATE TABLE IF NOT EXISTS llm_gateway_batches" in sql
    assert "REFERENCES saas_accounts(id) ON DELETE CASCADE" in sql
    # Provider batch id stored separately so a re-submit can't collide
    # with another tenant's batch.
    assert "provider_batch_id" in sql
    # Status reads need fast lookup by (account_id, created_at).
    assert "idx_llm_gateway_batches_account_created" in sql


def test_migration_317_partial_index_for_active_batches():
    """Status-poll workers can scan only active batches without
    touching terminal rows."""
    sql = _read_migration("317_llm_gateway_batches.sql")
    assert "idx_llm_gateway_batches_active" in sql
    assert "WHERE status IN" in sql
    assert "'queued'" in sql and "'in_progress'" in sql


# ---- Streaming endpoint -------------------------------------------------


def test_chat_stream_route_registered():
    from atlas_brain.api.llm_gateway import router

    paths = sorted({route.path for route in router.routes if hasattr(route, "path")})
    assert "/llm/chat/stream" in paths


def test_chat_stream_uses_plan_gate():
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat_stream)
    assert "require_llm_plan" in src


def test_chat_stream_returns_event_stream_media_type():
    """The response Content-Type must be ``text/event-stream`` so
    EventSource clients consume it correctly. Pin via source-text
    inspection -- the alternative is firing up a TestClient."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat_stream)
    assert 'media_type="text/event-stream"' in src


def test_chat_stream_disables_proxy_buffering():
    """``X-Accel-Buffering: no`` tells nginx to not buffer the
    response, which would otherwise hold all chunks until close --
    breaking the streaming UX."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat_stream)
    assert '"X-Accel-Buffering": "no"' in src


def test_sse_event_helper_format():
    """SSE event format: ``event: <name>\\ndata: <json>\\n\\n``.
    Standard EventSource clients reject malformed events silently."""
    from atlas_brain.api.llm_gateway import _sse_event

    out = _sse_event("content", {"id": "abc", "text": "hello"})
    assert isinstance(out, bytes)
    text = out.decode("utf-8")
    assert text.startswith("event: content\n")
    assert "data: " in text
    assert text.endswith("\n\n")
    # JSON payload is on a single line.
    data_line = [line for line in text.split("\n") if line.startswith("data: ")][0]
    assert "\n" not in data_line


def test_sse_event_helper_handles_special_chars():
    """JSON encoding must escape newlines + quotes in the text
    payload -- otherwise the event becomes malformed."""
    from atlas_brain.api.llm_gateway import _sse_event

    out = _sse_event("content", {"text": 'line one\nline "two"'})
    text = out.decode("utf-8")
    # Newline inside content must be JSON-escaped, not actually
    # broken across lines.
    assert "line one\nline" not in text  # Real newline -- bad
    assert '"line one\\nline \\"two\\""' in text


# ---- Batch endpoints ----------------------------------------------------


def test_batch_submit_route_registered():
    from atlas_brain.api.llm_gateway import router

    paths = sorted({route.path for route in router.routes if hasattr(route, "path")})
    assert "/llm/batch" in paths


def test_batch_status_route_registered():
    from atlas_brain.api.llm_gateway import router

    paths = sorted({route.path for route in router.routes if hasattr(route, "path")})
    assert "/llm/batch/{batch_id}" in paths


def test_batch_submit_returns_202():
    """Submit is async-accept -- the batch is enqueued, customer
    polls /batch/{id} for status. 202 Accepted is the right code."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.submit_batch)
    assert "status_code=202" in src or "status_code = 202" in src


def test_batch_uses_starter_plan_gate():
    """Plan tier: ``llm_starter`` minimum so the trial tier cannot
    abuse the 50% discount for free volume."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.submit_batch)
    assert "require_llm_plan(\"llm_starter\")" in src or "require_llm_plan('llm_starter')" in src

    src = inspect.getsource(llm_gateway.get_batch)
    assert "require_llm_plan(\"llm_starter\")" in src or "require_llm_plan('llm_starter')" in src


def test_batch_enforces_batch_enabled_plan_limit():
    """Even within llm_starter+, the ``batch_enabled`` flag in
    LLM_PLAN_LIMITS gates feature access -- so a future plan
    redesign can disable batch on a specific tier without breaking
    the gate."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway._require_batch_enabled)
    assert "batch_enabled" in src
    assert "status_code=403" in src


def test_batch_status_404_on_invalid_uuid():
    """Malformed batch_id strings 404 instead of 500; a customer
    pasting a typo gets a clear error."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.get_batch)
    assert "ValueError" in src
    assert "Batch not found" in src


def test_batch_status_returns_404_cross_account():
    """Account A querying B's batch_id must NOT leak existence.
    ``get_customer_batch`` filters on account_id; the route 404s
    when None."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.get_batch)
    assert "if record is None" in src
    assert 'detail="Batch not found"' in src


# ---- Service helpers ---------------------------------------------------


def test_submit_customer_batch_signature():
    import inspect as _inspect
    from atlas_brain.services.llm_gateway_batch import submit_customer_batch

    assert _inspect.iscoroutinefunction(submit_customer_batch)
    sig = _inspect.signature(submit_customer_batch)
    for p in ("account_id", "api_key", "model", "items"):
        assert p in sig.parameters


def test_get_customer_batch_scopes_by_account():
    """The service-level helper MUST filter on account_id so an
    accidental router-side oversight cannot leak cross-account."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.get_customer_batch)
    assert "WHERE id = $1 AND account_id = $2" in src


def test_refresh_customer_batch_status_skips_terminal():
    """Terminal batches don't get re-polled -- avoids round-trip
    to Anthropic for completed work and keeps the status endpoint
    cheap when the customer is just paginating their batch list."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.refresh_customer_batch_status)
    assert "TERMINAL_STATUSES" in src or '"ended"' in src


def test_refresh_customer_batch_skips_when_no_provider_id():
    """If the submit failed before Anthropic returned an id (e.g.
    network error), we have no provider_batch_id to poll. The
    refresh path must return the row as-is, not crash."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.refresh_customer_batch_status)
    assert "if not record.provider_batch_id" in src


def test_terminal_status_set_includes_anthropic_terminal_states():
    """Anthropic's batch API has 3 terminal states: ended, canceled,
    expired. All three must short-circuit polling."""
    from atlas_brain.services.llm_gateway_batch import TERMINAL_STATUSES

    assert "ended" in TERMINAL_STATUSES
    assert "canceled" in TERMINAL_STATUSES
    assert "expired" in TERMINAL_STATUSES


# ---- Schema shape ------------------------------------------------------


def test_batch_submit_request_validates():
    from atlas_brain.api.llm_gateway import BatchSubmitRequest

    req = BatchSubmitRequest(
        provider="anthropic",
        model="claude-haiku-4-5",
        items=[
            {"custom_id": "req1", "messages": [{"role": "user", "content": "hi"}]},
            {"custom_id": "req2", "messages": [{"role": "user", "content": "hello"}]},
        ],
    )
    assert len(req.items) == 2


def test_batch_submit_request_caps_items():
    """Anthropic's batch API has practical limits; we cap at 10k to
    bound atlas's tracking-row bloat."""
    from atlas_brain.api.llm_gateway import BatchSubmitRequest

    huge_items = [
        {"custom_id": f"r{i}", "messages": [{"role": "user", "content": "x"}]}
        for i in range(10_001)
    ]
    with pytest.raises(Exception):
        BatchSubmitRequest(provider="anthropic", model="claude-haiku-4-5", items=huge_items)


def test_batch_submit_request_rejects_empty_items():
    from atlas_brain.api.llm_gateway import BatchSubmitRequest

    with pytest.raises(Exception):
        BatchSubmitRequest(provider="anthropic", model="claude-haiku-4-5", items=[])


def test_batch_view_omits_results_jsonl():
    """The schema must NOT expose ``results_jsonl`` -- that field
    is large and customers fetch results via a separate endpoint
    in PR-D4c (TBD). For PR-D4b, only metadata is returned."""
    from atlas_brain.api.llm_gateway import BatchView

    fields = set(BatchView.model_fields.keys())
    assert "results_jsonl" not in fields


# ---- Codex P2 review (resource cleanup via async-with) ----------------


def test_submit_customer_batch_uses_async_with():
    """Codex P2: ``AsyncAnthropic`` opens an httpx connection pool
    that must be released after each batch submit. Source-text pin
    that we use ``async with`` (not bare construction)."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.submit_customer_batch)
    assert "async with AsyncAnthropic(" in src
    # The bare ``client = AsyncAnthropic(...)`` pattern must NOT
    # remain in the file.
    full_src = inspect.getsource(llm_gateway_batch)
    assert "client = AsyncAnthropic(" not in full_src


def test_refresh_customer_batch_uses_async_with():
    """Same posture for the polling path -- /batch/{id} is hit
    repeatedly while a batch is in flight, so leaks compound."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.refresh_customer_batch_status)
    assert "async with AsyncAnthropic(" in src


def test_chat_handler_uses_async_with_anthropic():
    """The /chat handler used to access ``llm._async_client``
    directly -- AnthropicLLM is shaped for atlas's long-running
    pipeline (one instance held across calls), so per-request
    gateway use leaked the httpx pool. Now uses AsyncAnthropic
    in async-with directly."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    assert "async with AsyncAnthropic(api_key=api_key)" in src
    # Direct ``llm._async_client`` access must NOT remain.
    assert "llm._async_client" not in src
    # The unnecessary load() call is also gone (AsyncAnthropic is
    # constructed directly inside async-with).
    assert "llm.load()" not in src


def test_chat_stream_handler_uses_async_with_anthropic():
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway._stream_chat_chunks)
    assert "async with AsyncAnthropic(api_key=api_key)" in src
    assert "llm._async_client" not in src
    assert "llm.load()" not in src


# ---- Codex/Copilot 2nd pass review -------------------------------------


def test_submit_failed_status_is_distinct_from_ended():
    """Codex/Copilot fix: pre-submit failure used to write
    ``status='ended'``, conflating with provider-completed batches.
    The new ``submit_failed`` status is distinct so consumers can
    tell submit-time failure from completion."""
    from atlas_brain.services.llm_gateway_batch import TERMINAL_STATUSES

    # submit_failed is terminal (no polling can recover it).
    assert "submit_failed" in TERMINAL_STATUSES
    # Distinct from the Anthropic terminal states.
    assert "submit_failed" != "ended"


def test_submit_failure_returns_record_not_502():
    """Customer should get the persisted batch_id + error_text on
    submit failure, not an opaque 502 -- the row is already saved
    with status='submit_failed', so returning it lets the caller
    debug without a list endpoint."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.submit_batch)
    # The 502 path is gone; submit_customer_batch persists the
    # failure and returns the record.
    assert "Batch submit failed" not in src
    # The record-return is the only success path.
    assert "_batch_record_to_view(record)" in src


def test_submit_customer_batch_returns_failed_record_not_raises():
    """Service-level: submit failure persists ``status='submit_failed'``
    and returns the record. No raise. Customers polling /batch/{id}
    immediately see the error text."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.submit_customer_batch)
    assert "status = 'submit_failed'" in src
    assert "return _row_to_record(failed_row)" in src


def test_chat_message_body_threads_tool_fields():
    """Tool-use round-trip: assistant ``tool_calls`` and tool-role
    ``tool_call_id`` must reach the internal Message dataclass.
    Without this, batch requests that include tool transcripts get
    serialized into malformed Anthropic payloads."""
    from atlas_brain.api.llm_gateway import (
        ChatMessageBody,
        _to_internal_message,
    )

    body = ChatMessageBody(
        role="assistant",
        content="calling tool",
        tool_calls=[{"id": "t1", "function": {"name": "search", "arguments": {}}}],
    )
    msg = _to_internal_message(body)
    assert msg.tool_calls is not None
    assert msg.tool_calls[0]["id"] == "t1"

    body2 = ChatMessageBody(
        role="tool", content="result", tool_call_id="t1"
    )
    msg2 = _to_internal_message(body2)
    assert msg2.tool_call_id == "t1"


def test_chat_handler_uses_to_internal_message_helper():
    """All three sites that build Message lists now use the helper
    so tool fields don't get dropped silently."""
    from atlas_brain.api import llm_gateway

    chat_src = inspect.getsource(llm_gateway.chat)
    assert "_to_internal_message" in chat_src

    stream_src = inspect.getsource(llm_gateway._stream_chat_chunks)
    assert "_to_internal_message" in stream_src

    batch_src = inspect.getsource(llm_gateway.submit_batch)
    assert "_to_internal_message" in batch_src


def test_batch_normalizes_model_alias():
    """The /batch endpoint must apply AnthropicLLM's alias map so
    deprecated names work consistently with /chat. Codex review
    fix: ``claude-3-5-haiku-latest`` worked on chat, failed on
    batch before this."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.submit_batch)
    # Normalization happens via AnthropicLLM's alias map.
    assert "_AnthropicLLM(model=body.model).model" in src
    assert "normalized_model" in src
    # The submit call passes the normalized variant.
    assert "model=normalized_model" in src


def test_batch_provider_validator_uses_batch_specific_message():
    """Error from /batch should not say 'chat endpoint'."""
    from atlas_brain.api import llm_gateway

    assert hasattr(llm_gateway, "_validate_batch_provider")
    src = inspect.getsource(llm_gateway._validate_batch_provider)
    assert "/api/v1/llm/batch" in src


def test_failed_items_includes_canceled():
    """Codex fix: ``failed_items`` was ``errored + expired`` --
    missing canceled. For canceled batches, completed + failed
    didn't equal total_items."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.refresh_customer_batch_status)
    # The sum now covers all 3 non-success terminal states.
    assert 'getattr(counts, "canceled"' in src


def test_anthropic_sdk_calls_use_timeout():
    """Provider-stall protection: every AsyncAnthropic construction
    passes a bounded timeout so a slow upstream cannot tie up a
    FastAPI worker indefinitely."""
    from atlas_brain.services import llm_gateway_batch

    submit_src = inspect.getsource(llm_gateway_batch.submit_customer_batch)
    refresh_src = inspect.getsource(llm_gateway_batch.refresh_customer_batch_status)

    assert "timeout=ANTHROPIC_SDK_TIMEOUT_SECONDS" in submit_src
    assert "timeout=ANTHROPIC_SDK_TIMEOUT_SECONDS" in refresh_src


def test_batch_item_custom_id_capped_at_64():
    """Match atlas's existing batch path -- avoid local-pass + remote-
    fail. Codex fix on PR-D4b."""
    from atlas_brain.api.llm_gateway import BatchItemBody

    field = BatchItemBody.model_fields["custom_id"]
    assert field.metadata is not None
    # Pydantic stores constraints in metadata as MaxLen objects.
    max_len_constraints = [
        m for m in field.metadata if hasattr(m, "max_length")
    ]
    assert max_len_constraints
    assert max_len_constraints[0].max_length == 64


def test_batch_request_caps_dropped_to_safer_levels():
    """Body-size DoS surface bounded: per-message char cap dropped
    from 200k to 64k; per-item messages cap from 200 to 50; total
    items cap from 10k to 1k."""
    from atlas_brain.api.llm_gateway import (
        BatchItemBody,
        BatchSubmitRequest,
        ChatMessageBody,
    )

    # Per-message char cap.
    content_field = ChatMessageBody.model_fields["content"]
    max_len = next(
        (m.max_length for m in content_field.metadata if hasattr(m, "max_length")),
        None,
    )
    assert max_len == 64_000

    # Per-item messages cap.
    messages_field = BatchItemBody.model_fields["messages"]
    max_msg = next(
        (m.max_length for m in messages_field.metadata if hasattr(m, "max_length")),
        None,
    )
    assert max_msg == 50

    # Total items cap.
    items_field = BatchSubmitRequest.model_fields["items"]
    max_items = next(
        (m.max_length for m in items_field.metadata if hasattr(m, "max_length")),
        None,
    )
    assert max_items == 1_000


def test_batch_status_distinguishes_revoked_from_outage():
    """Codex fix: when ``lookup_provider_key_async`` returns None,
    the get_batch handler now probes for an active byok_keys row
    directly. If one exists but lookup failed (DB outage / decrypt
    drift), surface 503 instead of silently freezing the batch."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.get_batch)
    # Direct probe for active row.
    assert "FROM byok_keys" in src
    assert "AND revoked_at IS NULL" in src
    # 503 surfaced when row exists but lookup failed.
    assert "status_code=503" in src
