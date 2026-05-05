"""Customer-facing Anthropic Message Batches integration (PR-D4b).

Wraps the Anthropic SDK's batch surface so customers get the 50%
batch discount via ``POST /api/v1/llm/batch``. Distinct from atlas's
internal ``services/b2b/anthropic_batch.py`` which is shaped around
the B2B pipeline (vendor_name, stage_id, artifact_type) -- this
module is purely customer-shaped (custom_id, messages, model,
account_id).

Persistence: one row per customer batch in ``llm_gateway_batches``
(migration 317). Status reads always scope on ``account_id`` so
account A cannot see B's batch results.

Status mapping (Anthropic SDK -> our status field):
  in_progress    -> "in_progress"
  ended          -> "ended"          (terminal; results available)
  canceling      -> "canceling"
  canceled       -> "canceled"       (terminal; partial results may exist)
  expired        -> "expired"        (terminal; Anthropic 24h TTL hit)

Atlas-internal lifecycle states (NOT Anthropic):
  queued         -> pre-submit holding state (rare; insert succeeds but
                    we crash before calling Anthropic)
  submit_failed  -> Anthropic rejected the submit (validation error,
                    rate limit, etc.) Distinct from "ended" so
                    consumers polling by status can tell submit-time
                    failure from provider-completed batches.
"""

from __future__ import annotations

import json
import logging
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence

import asyncpg

from .llm.anthropic import convert_messages
from .protocols import Message

logger = logging.getLogger("atlas.services.llm_gateway_batch")


# How long to wait for Anthropic's batch SDK calls. Worker latency
# is bounded so a single slow upstream cannot stack blocked workers.
ANTHROPIC_SDK_TIMEOUT_SECONDS = 30.0

# Terminal statuses (no more polling needed). ``submit_failed`` is
# also terminal -- the row never made it to Anthropic, so polling
# would do nothing useful.
TERMINAL_STATUSES = ("ended", "canceled", "expired", "submit_failed")

# Provider statuses we treat as in-flight on the read path.
ACTIVE_STATUSES = ("queued", "in_progress", "canceling")

# Span name written into the ``span_name`` column of ``llm_usage``
# for batch-item rows. Stable identifier so /api/v1/llm/usage
# rollups can split batch traffic vs synchronous /chat traffic
# without joining ``llm_gateway_batches``.
_BATCH_ITEM_SPAN_NAME = "llm_gateway.batch_item"

# Endpoint marker for batch-item rows. Mirrors the ``endpoint``
# label written for sync /chat traffic so the same rollup queries
# can group both lanes.
_BATCH_ITEM_ENDPOINT = "llm_gateway.batch"


@dataclass(frozen=True)
class CustomerBatchItem:
    """One per customer-supplied item in a batch submit. ``custom_id``
    is the customer's correlation id; we echo it back in results so
    they can match outputs to inputs."""

    custom_id: str
    messages: Sequence[Message]
    max_tokens: int = 1024
    temperature: float = 0.7


@dataclass(frozen=True)
class CustomerBatchRecord:
    """Display-safe view of an llm_gateway_batches row."""

    id: _uuid.UUID
    account_id: _uuid.UUID
    provider: str
    provider_batch_id: Optional[str]
    model: str
    status: str
    total_items: int
    completed_items: int
    failed_items: int
    error_text: Optional[str]
    created_at: datetime
    updated_at: datetime
    submitted_at: Optional[datetime]
    completed_at: Optional[datetime]
    # PR-D4c Codex P1: surfaced so the refresh path can detect a
    # terminal row whose usage write never landed (results pre-fetch
    # failed before the atomic claim could flip the flag) and retry.
    usage_tracked: bool = False


def _row_to_record(row: Any) -> CustomerBatchRecord:
    return CustomerBatchRecord(
        id=row["id"],
        account_id=row["account_id"],
        provider=row["provider"],
        provider_batch_id=row["provider_batch_id"],
        model=row["model"],
        status=row["status"],
        total_items=int(row["total_items"] or 0),
        completed_items=int(row["completed_items"] or 0),
        failed_items=int(row["failed_items"] or 0),
        error_text=row["error_text"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        submitted_at=row["submitted_at"],
        completed_at=row["completed_at"],
        # ``usage_tracked`` was added in migration 318. Use ``.get``
        # via dict-like access so older callers / tests with shorter
        # row mocks still work (the column defaults FALSE in the
        # migration, so missing == not-yet-tracked).
        usage_tracked=bool(row["usage_tracked"]) if "usage_tracked" in row.keys() else False,
    )


# ---- Submit -------------------------------------------------------------


async def submit_customer_batch(
    pool,
    *,
    account_id: _uuid.UUID,
    api_key: str,
    model: str,
    items: Sequence[CustomerBatchItem],
    idempotency_key: Optional[str] = None,
) -> CustomerBatchRecord:
    """Submit a batch to Anthropic with the customer's BYOK key.

    Inserts a tracking row first (status="queued"), calls Anthropic's
    batches.create, persists ``provider_batch_id`` + status="in_progress"
    on success, persists ``error_text`` + status="submit_failed" on
    failure. Returns the persisted record.

    When ``idempotency_key`` is provided, a prior submit with the same
    key for the same account replays the original record -- no new
    Anthropic call. Closes the accepted-upstream-but-timeout-locally
    retry case where a customer would otherwise create duplicate paid
    batches. PR-D4c.
    """
    if not items:
        raise ValueError("submit_customer_batch: items list is empty")
    if not api_key:
        raise ValueError("submit_customer_batch: api_key is required")

    # Idempotency replay: if the customer already submitted under this
    # key for this account, return the existing record without calling
    # Anthropic again. Per-account UNIQUE constraint enforced at the
    # DB level (migration 318). The empty-string key is treated as
    # "no key supplied" so customers don't accidentally collide on it.
    normalized_key = idempotency_key.strip() if idempotency_key else None
    if not normalized_key:
        normalized_key = None
    row: Any = None
    if normalized_key:
        existing = await pool.fetchrow(
            """
            SELECT id, account_id, provider, provider_batch_id, model,
                   status, total_items, completed_items, failed_items,
                   error_text, created_at, updated_at, submitted_at,
                   completed_at, usage_tracked
            FROM llm_gateway_batches
            WHERE account_id = $1 AND idempotency_key = $2
            """,
            account_id,
            normalized_key,
        )
        if existing:
            # "Settled" = Anthropic accepted the submission. We have a
            # provider_batch_id to point the customer at, regardless
            # of whether the batch ended/canceled/expired. A retry
            # with the same key gets the prior result back.
            if existing["provider_batch_id"]:
                logger.info(
                    "llm_gateway_batch.submit replay account=%s key=%s id=%s",
                    account_id,
                    normalized_key,
                    existing["id"],
                )
                return _row_to_record(existing)

            # No provider_batch_id means the prior attempt never
            # made it to Anthropic. Two cases:
            #   a) status='queued', updated recently -- a concurrent
            #      retry is in flight right now; re-submitting would
            #      duplicate-pay. Replay so the customer polls.
            #   b) status='queued' but stale (updated_at >
            #      2x ANTHROPIC_SDK_TIMEOUT_SECONDS ago), OR
            #      status='submit_failed' -- the prior attempt
            #      crashed pre-submit (a) or Anthropic rejected /
            #      timed out (b). Either way, the customer's batch
            #      never landed; resume so the retry actually
            #      submits. Codex P1 rounds 3+4 on PR-D4c.
            #
            # The decision is made atomically in SQL so two
            # concurrent resumes can't both win. Threshold = 60s =
            # 2x ANTHROPIC_SDK_TIMEOUT_SECONDS so an in-flight call
            # has had time to either succeed or hit the SDK timeout
            # (which would have updated the row to submit_failed).
            resumed = await pool.fetchrow(
                """
                UPDATE llm_gateway_batches
                SET updated_at = NOW(),
                    status = 'queued',
                    error_text = NULL
                WHERE id = $1
                  AND account_id = $2
                  AND provider_batch_id IS NULL
                  AND (
                    status = 'submit_failed'
                    OR (status = 'queued'
                        AND updated_at < NOW() - INTERVAL '60 seconds')
                  )
                RETURNING id, account_id, provider, provider_batch_id, model,
                          status, total_items, completed_items, failed_items,
                          error_text, created_at, updated_at, submitted_at,
                          completed_at, usage_tracked
                """,
                existing["id"],
                account_id,
            )
            if resumed is None:
                # Recent queued (< 60s old) or another retry already
                # claimed it. Replay -- the customer polls /batch/{id}
                # either way.
                #
                # PR-D4d post-audit: re-read the row before returning.
                # The ``existing`` snapshot was taken before the resume
                # claim attempt; if a concurrent retry just succeeded
                # (set provider_batch_id, transitioned to in_progress),
                # ``existing`` is stale. Fetching fresh state means the
                # caller sees the current row instead of the snapshot.
                latest = await pool.fetchrow(
                    """
                    SELECT id, account_id, provider, provider_batch_id, model,
                           status, total_items, completed_items, failed_items,
                           error_text, created_at, updated_at, submitted_at,
                           completed_at, usage_tracked
                    FROM llm_gateway_batches
                    WHERE id = $1 AND account_id = $2
                    """,
                    existing["id"],
                    account_id,
                )
                replay_row = latest if latest is not None else existing
                logger.info(
                    "llm_gateway_batch.submit replay (in-flight) "
                    "account=%s key=%s id=%s status=%s",
                    account_id,
                    normalized_key,
                    replay_row["id"],
                    replay_row["status"],
                )
                return _row_to_record(replay_row)
            logger.info(
                "llm_gateway_batch.submit resume pre-submit "
                "account=%s key=%s id=%s prior_status=%s",
                account_id,
                normalized_key,
                existing["id"],
                existing["status"],
            )
            row = resumed

    # Insert pre-submit so a crash mid-call still leaves a queued row
    # the customer can see. The idempotency_key column is NULL when
    # no header was sent -- the partial UNIQUE index ignores NULL.
    #
    # Codex P1 on PR-D4c: the SELECT above is not atomic with this
    # INSERT, so two concurrent retries with the same key can both
    # miss the SELECT and race here. Catch the resulting unique
    # violation and re-read so the loser of the race still gets the
    # winner's record back (the contract callers expect from
    # idempotency).
    if row is None:
        try:
            async with pool.transaction() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO llm_gateway_batches (
                        account_id, provider, model, status, total_items, idempotency_key
                    ) VALUES (
                        $1, 'anthropic', $2, 'queued', $3, $4
                    )
                    RETURNING id, account_id, provider, provider_batch_id, model,
                              status, total_items, completed_items, failed_items,
                              error_text, created_at, updated_at, submitted_at,
                              completed_at, usage_tracked
                    """,
                    account_id,
                    model,
                    len(items),
                    normalized_key,
                )
        except asyncpg.exceptions.UniqueViolationError:
            if not normalized_key:
                # Should not happen -- the only unique constraint is the
                # idempotency partial index, which excludes NULL keys.
                raise
            existing = await pool.fetchrow(
                """
                SELECT id, account_id, provider, provider_batch_id, model,
                       status, total_items, completed_items, failed_items,
                       error_text, created_at, updated_at, submitted_at,
                       completed_at, usage_tracked
                FROM llm_gateway_batches
                WHERE account_id = $1 AND idempotency_key = $2
                """,
                account_id,
                normalized_key,
            )
            if existing is None:
                # Unique-violation but no matching row -- means the
                # winner committed and rolled back, or some other
                # column collided. Re-raise so the caller sees the real
                # 500 instead of a confusing "missing record".
                raise
            logger.info(
                "llm_gateway_batch.submit replay (race) account=%s key=%s id=%s",
                account_id,
                normalized_key,
                existing["id"],
            )
            return _row_to_record(existing)

    requests = []
    for item in items:
        system_prompt, api_messages = convert_messages(list(item.messages))
        params: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": item.max_tokens,
            "temperature": item.temperature,
        }
        if system_prompt:
            params["system"] = system_prompt
        requests.append({"custom_id": item.custom_id, "params": params})

    # Call Anthropic. Imported lazily so unit tests can stub the
    # client without dragging the SDK into module load.
    # ``async with`` ensures the underlying httpx connection pool is
    # released after each call -- /llm/batch is high-traffic per
    # customer, so leaking even one connection per call accumulates
    # quickly. Codex P2 fix on PR-D4b.
    from anthropic import AsyncAnthropic

    try:
        async with AsyncAnthropic(
            api_key=api_key,
            timeout=ANTHROPIC_SDK_TIMEOUT_SECONDS,
        ) as client:
            provider_batch = await client.messages.batches.create(requests=requests)
    except Exception as exc:
        # Codex review fix on PR-D4b: distinct ``submit_failed`` status
        # so consumers polling by status can tell submit-time failure
        # from provider-completed (``ended``). The router returns this
        # record (instead of raising) so customers get the batch_id of
        # their failed submission and can read ``error_text`` to
        # debug.
        logger.warning(
            "llm_gateway_batch.submit failed account=%s model=%s items=%d: %s",
            account_id,
            model,
            len(items),
            exc,
        )
        failed_row = await pool.fetchrow(
            """
            UPDATE llm_gateway_batches
            SET status = 'submit_failed',
                error_text = $2,
                updated_at = NOW(),
                completed_at = NOW()
            WHERE id = $1
            RETURNING id, account_id, provider, provider_batch_id, model,
                      status, total_items, completed_items, failed_items,
                      error_text, created_at, updated_at, submitted_at,
                      completed_at, usage_tracked
            """,
            row["id"],
            f"Anthropic batch submit failed: {exc}",
        )
        return _row_to_record(failed_row)

    provider_batch_id = getattr(provider_batch, "id", None)
    initial_status = getattr(provider_batch, "processing_status", None) or "in_progress"

    updated_row = await pool.fetchrow(
        """
        UPDATE llm_gateway_batches
        SET provider_batch_id = $2,
            status = $3,
            submitted_at = NOW(),
            updated_at = NOW()
        WHERE id = $1
        RETURNING id, account_id, provider, provider_batch_id, model,
                  status, total_items, completed_items, failed_items,
                  error_text, created_at, updated_at, submitted_at,
                  completed_at, usage_tracked
        """,
        row["id"],
        str(provider_batch_id) if provider_batch_id else None,
        str(initial_status),
    )

    return _row_to_record(updated_row)


# ---- Status / poll ------------------------------------------------------


async def get_customer_batch(
    pool,
    *,
    account_id: _uuid.UUID,
    batch_id: _uuid.UUID,
) -> Optional[CustomerBatchRecord]:
    """Fetch a batch row scoped to the calling account. Returns None
    when no row matches -- the router translates this to 404 (avoids
    leaking batch-id existence cross-account)."""
    row = await pool.fetchrow(
        """
        SELECT id, account_id, provider, provider_batch_id, model,
               status, total_items, completed_items, failed_items,
               error_text, created_at, updated_at, submitted_at,
               completed_at, usage_tracked
        FROM llm_gateway_batches
        WHERE id = $1 AND account_id = $2
        """,
        batch_id,
        account_id,
    )
    return _row_to_record(row) if row else None


async def refresh_customer_batch_status(
    pool,
    *,
    account_id: _uuid.UUID,
    batch_id: _uuid.UUID,
    api_key: str,
) -> Optional[CustomerBatchRecord]:
    """Poll Anthropic for the latest batch status, persist updates,
    return the refreshed record. Returns None when no row matches
    (cross-account or unknown id).

    Skips the API call when the row is already in a terminal state
    -- avoids round-tripping for completed batches.
    """
    record = await get_customer_batch(pool, account_id=account_id, batch_id=batch_id)
    if record is None:
        return None
    if record.status in TERMINAL_STATUSES:
        # Codex P1 on PR-D4c: a terminal row whose usage write has
        # not landed yet means the prior _persist_batch_usage call
        # failed during the Anthropic results pre-fetch (so the
        # atomic claim was never made and usage_tracked stayed
        # FALSE). We must retry here -- the only other place that
        # triggers a write is the non-terminal->terminal transition
        # below, and that won't fire again. Without this branch, a
        # single transient SDK error would silently lose the
        # customer's usage data forever.
        if (
            not record.usage_tracked
            and record.status in ("ended", "canceled", "expired")
            and record.provider_batch_id
        ):
            try:
                await _persist_batch_usage(
                    pool,
                    account_id=account_id,
                    batch_id=batch_id,
                    provider_batch_id=record.provider_batch_id,
                    model=record.model,
                    api_key=api_key,
                )
            except Exception:
                logger.exception(
                    "llm_gateway_batch.refresh: usage retry failed "
                    "account=%s batch=%s",
                    account_id,
                    batch_id,
                )
            # Re-read so usage_tracked / counts reflect the retry.
            return await get_customer_batch(
                pool, account_id=account_id, batch_id=batch_id
            )
        return record
    if not record.provider_batch_id:
        # Submit failed before persisting provider_batch_id; nothing
        # to poll. The row's ended/error state is already final.
        return record

    from anthropic import AsyncAnthropic

    # ``async with`` releases the httpx connection pool after each
    # poll -- /llm/batch/{id} is hit repeatedly while a batch is
    # processing, so leaks compound fast. Timeout bounds worker
    # blocking under provider stalls. Codex P2 fix on PR-D4b.
    try:
        async with AsyncAnthropic(
            api_key=api_key,
            timeout=ANTHROPIC_SDK_TIMEOUT_SECONDS,
        ) as client:
            provider_batch = await client.messages.batches.retrieve(record.provider_batch_id)
    except Exception as exc:
        logger.warning(
            "llm_gateway_batch.refresh failed account=%s batch=%s: %s",
            account_id,
            batch_id,
            exc,
        )
        return record  # Return last-known state on transient errors.

    new_status = str(getattr(provider_batch, "processing_status", record.status) or record.status)
    counts = getattr(provider_batch, "request_counts", None)
    if counts:
        completed = int(getattr(counts, "succeeded", 0) or 0)
        # Codex review on PR-D4b: include ``canceled`` in failed_items
        # so completed + failed adds up to total_items even when the
        # batch was canceled mid-flight.
        failed = (
            int(getattr(counts, "errored", 0) or 0)
            + int(getattr(counts, "expired", 0) or 0)
            + int(getattr(counts, "canceled", 0) or 0)
        )
    else:
        completed = record.completed_items
        failed = record.failed_items

    is_terminal = new_status in TERMINAL_STATUSES
    completed_at_clause = "NOW()" if is_terminal else "completed_at"
    updated = await pool.fetchrow(
        f"""
        UPDATE llm_gateway_batches
        SET status = $2,
            completed_items = $3,
            failed_items = $4,
            updated_at = NOW(),
            completed_at = {completed_at_clause}
        WHERE id = $1
        RETURNING id, account_id, provider, provider_batch_id, model,
                  status, total_items, completed_items, failed_items,
                  error_text, created_at, updated_at, submitted_at,
                  completed_at, usage_tracked
        """,
        batch_id,
        new_status,
        completed,
        failed,
    )
    refreshed = _row_to_record(updated)

    # PR-D4c: when a batch transitions to a real Anthropic terminal
    # state (ended/canceled/expired -- NOT submit_failed), fetch the
    # per-item results from Anthropic and write llm_usage rows so the
    # batch traffic shows up in /api/v1/llm/usage. Idempotent via the
    # ``usage_tracked`` column -- repeat polls don't double-count.
    if is_terminal and new_status in ("ended", "canceled", "expired"):
        try:
            await _persist_batch_usage(
                pool,
                account_id=account_id,
                batch_id=batch_id,
                provider_batch_id=record.provider_batch_id,
                model=record.model,
                api_key=api_key,
            )
        except Exception:
            logger.exception(
                "llm_gateway_batch.refresh: usage write failed account=%s batch=%s",
                account_id,
                batch_id,
            )

    return refreshed


# ---- Batch usage persistence -------------------------------------------


# Anthropic batch discount factor: tokens billed at 50% of the
# synchronous rate. Mirrors the same factor in atlas's existing
# ``services/b2b/anthropic_batch.py`` -- kept as a module-level
# constant so any future Anthropic pricing change updates one place.
_BATCH_DISCOUNT_FACTOR = 0.5


_BATCH_USAGE_INSERT_SQL = """
INSERT INTO llm_usage (
    span_name, operation_type, model_name, model_provider,
    input_tokens, output_tokens, total_tokens,
    billable_input_tokens, cached_tokens, cache_write_tokens,
    cost_usd, status, metadata,
    api_endpoint, provider_request_id, account_id,
    batch_id, custom_id
) VALUES (
    $1, 'llm_call', $2, 'anthropic',
    $3, $4, $5,
    $6, $7, $8,
    $9, 'completed', $10::jsonb,
    $11, $12, $13,
    $14, $15
)
ON CONFLICT (account_id, batch_id, custom_id)
WHERE batch_id IS NOT NULL
DO NOTHING
"""


async def _persist_batch_usage(
    pool,
    *,
    account_id: _uuid.UUID,
    batch_id: _uuid.UUID,
    provider_batch_id: str,
    model: str,
    api_key: str,
) -> None:
    """Write per-item ``llm_usage`` rows for a completed batch.

    Three-phase design (PR-D4d, post-audit):

    1. Pre-fetch all results from Anthropic into memory. No DB
       writes happen yet -- if the SDK iteration fails mid-stream,
       ``usage_tracked`` stays FALSE and the next poll retries
       from a clean slate.
    2. Direct INSERT each per-item row via ``pool.executemany``
       under a transaction. Each row uses ``ON CONFLICT
       (account_id, batch_id, custom_id) DO NOTHING`` (migration
       319) so retries are safe -- if a partial set of rows
       already landed, conflicts no-op silently.
    3. Flip ``usage_tracked = TRUE`` only AFTER the inserts
       commit. A concurrent poller racing on the same batch hits
       the same ON CONFLICT predicate (no double-write), and the
       UPDATE-at-end is account-scoped so cross-account batch_id
       reuse can never flip a foreign row.

    PR-D4c routed per-item writes through ``trace_llm_call``,
    which calls ``tracer.end_span`` -> ``loop.create_task`` --
    fire-and-forget. That meant the persist phase could not catch
    DB failures and ``usage_tracked = TRUE`` was set BEFORE writes
    actually landed. PR-D4d issues writes directly so failures
    surface synchronously and the flag flip honestly reflects
    "all rows landed."

    Per-item cost applies the 50% Anthropic batch discount.
    Customers see the discounted figure in
    ``/api/v1/llm/usage`` -- matches what Anthropic actually
    charges them.
    """
    # Phase 1: Pre-fetch results from Anthropic. Materialize the
    # async iterator into memory BEFORE we touch the DB. The SDK
    # call is the most likely failure point (network, provider
    # stall, rate limit), and surfacing the failure here means we
    # never write a partial set -- next poll retries cleanly.
    from anthropic import AsyncAnthropic

    items_to_persist: list[dict[str, Any]] = []
    async with AsyncAnthropic(
        api_key=api_key,
        timeout=ANTHROPIC_SDK_TIMEOUT_SECONDS,
    ) as client:
        results_iter = await client.messages.batches.results(provider_batch_id)
        async for entry in results_iter:
            custom_id = getattr(entry, "custom_id", "") or ""
            result = getattr(entry, "result", None)
            rtype = getattr(result, "type", None) if result else None
            if rtype != "succeeded":
                # Only successful items consumed tokens billed by
                # Anthropic. Errored / canceled / expired items
                # show in the failed_items counter but don't add
                # to llm_usage.
                continue
            message = getattr(result, "message", None)
            usage = getattr(message, "usage", None) if message else None
            if usage is None:
                continue
            items_to_persist.append({
                "custom_id": custom_id,
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "cached_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
                "cache_write_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
                "provider_request_id": getattr(message, "id", None),
            })

    # Phase 2: Direct INSERT each per-item row. Conflicts (a prior
    # retry that wrote the same (account_id, batch_id, custom_id))
    # no-op silently so we never double-count. The whole batch is
    # one transaction so a mid-iteration DB failure leaves no
    # partial state visible to other pollers.
    if items_to_persist:
        rows_args = []
        for item in items_to_persist:
            cost_usd = _BATCH_DISCOUNT_FACTOR * _estimate_cost_usd(
                model=model,
                input_tokens=item["input_tokens"],
                output_tokens=item["output_tokens"],
                cached_tokens=item["cached_tokens"],
                cache_write_tokens=item["cache_write_tokens"],
            )
            metadata = json.dumps({
                "account_id": str(account_id),
                "batch_id": str(batch_id),
                "custom_id": item["custom_id"],
                "endpoint": _BATCH_ITEM_ENDPOINT,
            })
            total_tokens = item["input_tokens"] + item["output_tokens"]
            rows_args.append((
                _BATCH_ITEM_SPAN_NAME,
                model,
                item["input_tokens"],
                item["output_tokens"],
                total_tokens,
                item["input_tokens"],  # billable_input_tokens
                item["cached_tokens"],
                item["cache_write_tokens"],
                cost_usd,
                metadata,
                _BATCH_ITEM_ENDPOINT,
                str(item["provider_request_id"]) if item["provider_request_id"] else None,
                str(account_id),
                batch_id,
                item["custom_id"],
            ))
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(_BATCH_USAGE_INSERT_SQL, rows_args)

    # Phase 3: Flip the flag. Account-scoped so cross-account
    # batch_id reuse cannot accidentally mark a foreign row.
    # ``usage_tracked = FALSE`` precondition makes the UPDATE
    # idempotent under concurrent pollers (loser sees no rows
    # to update; the inserts already succeeded via ON CONFLICT).
    await pool.execute(
        """
        UPDATE llm_gateway_batches
        SET usage_tracked = TRUE
        WHERE id = $1 AND account_id = $2 AND usage_tracked = FALSE
        """,
        batch_id,
        account_id,
    )


def _estimate_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Compute the synchronous-tier cost for a token mix using
    atlas's existing pricing config. The caller multiplies by
    ``_BATCH_DISCOUNT_FACTOR`` to get the actual batch cost.
    """
    from ..config import settings

    return float(
        settings.ftl_tracing.pricing.cost_usd(
            "anthropic",
            model,
            input_tokens,
            output_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            billable_input_tokens=input_tokens,
        )
    )
