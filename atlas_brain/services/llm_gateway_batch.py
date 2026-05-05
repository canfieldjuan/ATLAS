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
  in_progress -> "in_progress"
  ended       -> "ended"      (terminal; results available)
  canceling   -> "canceling"
  canceled    -> "canceled"   (terminal; partial results may exist)
  expired     -> "expired"    (terminal; Anthropic 24h TTL hit)

The "queued" status is our internal pre-submit state -- we insert
the row before calling the Anthropic API so failures during submit
are still tracked.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence

from .llm.anthropic import convert_messages
from .protocols import Message

logger = logging.getLogger("atlas.services.llm_gateway_batch")


# Terminal statuses (no more polling needed).
TERMINAL_STATUSES = ("ended", "canceled", "expired")

# Provider statuses we treat as in-flight on the read path.
ACTIVE_STATUSES = ("queued", "in_progress", "canceling")


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
    )


# ---- Submit -------------------------------------------------------------


async def submit_customer_batch(
    pool,
    *,
    account_id: _uuid.UUID,
    api_key: str,
    model: str,
    items: Sequence[CustomerBatchItem],
) -> CustomerBatchRecord:
    """Submit a batch to Anthropic with the customer's BYOK key.

    Inserts a tracking row first (status="queued"), calls Anthropic's
    batches.create, persists ``provider_batch_id`` + status="in_progress"
    on success, persists ``error_text`` + status="ended" on failure.
    Returns the persisted record.
    """
    if not items:
        raise ValueError("submit_customer_batch: items list is empty")
    if not api_key:
        raise ValueError("submit_customer_batch: api_key is required")

    # Insert pre-submit so a crash mid-call still leaves a queued row
    # the customer can see.
    async with pool.transaction() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO llm_gateway_batches (
                account_id, provider, model, status, total_items
            ) VALUES (
                $1, 'anthropic', $2, 'queued', $3
            )
            RETURNING id, account_id, provider, provider_batch_id, model,
                      status, total_items, completed_items, failed_items,
                      error_text, created_at, updated_at, submitted_at,
                      completed_at
            """,
            account_id,
            model,
            len(items),
        )

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
        async with AsyncAnthropic(api_key=api_key) as client:
            provider_batch = await client.messages.batches.create(requests=requests)
    except Exception as exc:
        logger.warning(
            "llm_gateway_batch.submit failed account=%s model=%s items=%d: %s",
            account_id,
            model,
            len(items),
            exc,
        )
        await pool.execute(
            """
            UPDATE llm_gateway_batches
            SET status = 'ended', error_text = $2, updated_at = NOW(),
                completed_at = NOW()
            WHERE id = $1
            """,
            row["id"],
            f"Anthropic batch submit failed: {exc}",
        )
        raise

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
                  completed_at
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
               completed_at
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
        return record
    if not record.provider_batch_id:
        # Submit failed before persisting provider_batch_id; nothing
        # to poll. The row's ended/error state is already final.
        return record

    from anthropic import AsyncAnthropic

    # ``async with`` releases the httpx connection pool after each
    # poll -- /llm/batch/{id} is hit repeatedly while a batch is
    # processing, so leaks compound fast. Codex P2 fix on PR-D4b.
    try:
        async with AsyncAnthropic(api_key=api_key) as client:
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
    completed = int(getattr(counts, "succeeded", 0) or 0) if counts else record.completed_items
    failed = int(getattr(counts, "errored", 0) or 0) + int(getattr(counts, "expired", 0) or 0) if counts else record.failed_items

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
                  completed_at
        """,
        batch_id,
        new_status,
        completed,
        failed,
    )
    return _row_to_record(updated)
