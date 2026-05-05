"""Tests for batch idempotency + per-item usage tracking (PR-D4c).

Pure structural + source-text tests (no live Anthropic calls).
DB-bound integration tests live with other auth integration
fixtures and are gated on a running Postgres.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest


_MIG_DIR = Path(__file__).resolve().parent.parent / "atlas_brain" / "storage" / "migrations"


def _read_migration(filename: str) -> str:
    return (_MIG_DIR / filename).read_text(encoding="utf-8")


# ---- Migration ----------------------------------------------------------


def test_migration_318_adds_idempotency_and_usage_columns():
    sql = _read_migration("318_llm_gateway_batches_idempotency_and_usage.sql")
    assert "ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(128)" in sql
    assert "ADD COLUMN IF NOT EXISTS usage_tracked   BOOLEAN NOT NULL DEFAULT FALSE" in sql


def test_migration_318_partial_unique_per_account():
    """Each account has its own idempotency namespace -- two
    accounts can independently use the same key without colliding.
    NULL keys (the optional case) skip the constraint."""
    sql = _read_migration("318_llm_gateway_batches_idempotency_and_usage.sql")
    assert "uq_llm_gateway_batches_idempotency" in sql
    assert "(account_id, idempotency_key)" in sql
    assert "WHERE idempotency_key IS NOT NULL" in sql


def test_migration_318_index_for_pending_usage_writes():
    """Catches batches that ended but didn't get usage written
    (atlas crashed mid-write, etc.). A future cron worker can
    scan this index to retry."""
    sql = _read_migration("318_llm_gateway_batches_idempotency_and_usage.sql")
    assert "idx_llm_gateway_batches_usage_pending" in sql
    assert "WHERE usage_tracked = FALSE" in sql
    assert "AND status IN ('ended', 'canceled', 'expired')" in sql


# ---- submit_customer_batch idempotency ----------------------------------


def test_submit_customer_batch_signature_accepts_idempotency_key():
    from atlas_brain.services.llm_gateway_batch import submit_customer_batch

    sig = inspect.signature(submit_customer_batch)
    assert "idempotency_key" in sig.parameters
    assert sig.parameters["idempotency_key"].default is None


def test_submit_customer_batch_replay_check_in_source():
    """Source-text pin: the replay path queries the existing row by
    (account_id, idempotency_key) BEFORE inserting a new one. Without
    this, the partial UNIQUE index would 500 on conflict instead of
    returning the prior record."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.submit_customer_batch)
    assert "WHERE account_id = $1 AND idempotency_key = $2" in src
    # Replay is a no-op aside from logging + return.
    assert 'logger.info(\n                "llm_gateway_batch.submit replay' in src


def test_submit_customer_batch_normalizes_empty_idempotency():
    """Empty-string and whitespace-only idempotency keys are
    treated as 'not supplied' so customers don't accidentally
    collide on them."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.submit_customer_batch)
    # The function strips and normalizes empty -> None.
    assert "normalized_key = idempotency_key.strip() if idempotency_key else None" in src
    assert "if not normalized_key:" in src


def test_submit_customer_batch_inserts_idempotency_column():
    """The INSERT statement must include the idempotency_key column
    so the unique-per-account constraint can deduplicate retries."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.submit_customer_batch)
    # Column listed explicitly in INSERT.
    assert "idempotency_key" in src.split("INSERT INTO llm_gateway_batches")[1].split("RETURNING")[0]


# ---- Router Idempotency-Key header --------------------------------------


def test_submit_batch_route_accepts_idempotency_header():
    """The router must declare ``Idempotency-Key`` as a Header
    parameter so FastAPI parses it (otherwise the value never
    reaches the service-level dedup logic)."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.submit_batch)
    assert 'alias="Idempotency-Key"' in src
    assert "Header(" in src
    # And the value is threaded into the service call.
    assert "idempotency_key=idempotency_key" in src


def test_submit_batch_idempotency_key_caps_at_128():
    """Bound the header length so a malicious caller can't ship
    arbitrarily large headers."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.submit_batch)
    assert "max_length=128" in src


# ---- Batch usage persistence -------------------------------------------


def test_persist_batch_usage_helper_exists():
    from atlas_brain.services import llm_gateway_batch

    assert hasattr(llm_gateway_batch, "_persist_batch_usage")
    assert inspect.iscoroutinefunction(llm_gateway_batch._persist_batch_usage)


def test_persist_batch_usage_idempotent_via_atomic_claim():
    """The function must atomically transition usage_tracked
    FALSE->TRUE before doing the work. A concurrent poller losing
    the race is a no-op (claim returns no rows)."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    assert "WHERE id = $1 AND usage_tracked = FALSE" in src
    assert "RETURNING id" in src
    # Loser path returns cleanly.
    assert "if claim is None:" in src


def test_persist_batch_usage_rolls_back_on_error():
    """If results-fetch raises (transient SDK error mid-iteration),
    the usage_tracked flag must roll back so a future poll retries.
    Otherwise the customer's usage would be lost permanently."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    # Rollback path on exception.
    assert "SET usage_tracked = FALSE WHERE id = $1" in src


def test_persist_batch_usage_only_counts_succeeded_items():
    """Errored / canceled / expired items don't consume billable
    tokens at Anthropic, so they shouldn't generate llm_usage rows.
    Only succeeded items get tracked."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    assert 'rtype != "succeeded"' in src
    assert "continue" in src


def test_persist_batch_usage_applies_50_percent_discount():
    """Anthropic charges 50% of synchronous rate for batch.
    /api/v1/llm/usage must show the discounted figure -- matches
    what the customer pays."""
    from atlas_brain.services import llm_gateway_batch

    assert llm_gateway_batch._BATCH_DISCOUNT_FACTOR == 0.5
    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    assert "_BATCH_DISCOUNT_FACTOR" in src


def test_persist_batch_usage_threads_account_id_into_metadata():
    """Same metadata pattern as /chat -- per-account scoping in
    llm_usage requires account_id in the trace metadata so PR-D3's
    INSERT picks it up."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    assert '"account_id": str(account_id)' in src
    assert '"batch_id": str(batch_id)' in src
    assert '"custom_id": custom_id' in src


def test_persist_batch_usage_uses_async_with():
    """Same async-with posture as the rest of the module -- close
    the httpx connection pool after the results-fetch."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    assert "async with AsyncAnthropic(" in src
    assert "timeout=ANTHROPIC_SDK_TIMEOUT_SECONDS" in src


# ---- Refresh -> usage persistence wiring -------------------------------


def test_refresh_calls_persist_usage_on_terminal_transition():
    """Polling /batch/{id} must trigger the usage write when the
    batch transitions to a real Anthropic terminal state. Excludes
    submit_failed which is an atlas-internal terminal state with
    no Anthropic-side results to fetch."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.refresh_customer_batch_status)
    assert "_persist_batch_usage" in src
    # submit_failed must NOT trigger the results-fetch (no batch on
    # Anthropic's side).
    assert 'new_status in ("ended", "canceled", "expired")' in src


def test_refresh_does_not_propagate_usage_write_failure():
    """A usage-write failure (transient SDK error fetching results)
    must NOT block /batch/{id} from returning the new status.
    Customer keeps polling; the usage_tracked rollback ensures the
    next poll retries."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.refresh_customer_batch_status)
    # try/except around _persist_batch_usage call.
    assert "try:\n            await _persist_batch_usage(" in src
    assert 'logger.exception(\n                "llm_gateway_batch.refresh: usage write failed' in src


# ---- Cost helper -------------------------------------------------------


def test_estimate_cost_uses_atlas_pricing_config():
    """The cost helper reuses ``settings.ftl_tracing.pricing`` so
    pricing changes update one place (not duplicated here)."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._estimate_cost_usd)
    assert "settings.ftl_tracing.pricing.cost_usd" in src
    assert '"anthropic"' in src
