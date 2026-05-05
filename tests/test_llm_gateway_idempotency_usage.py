"""Tests for batch idempotency + per-item usage tracking (PR-D4c).

Pure structural + source-text tests (no live Anthropic calls).
DB-bound integration tests live with other auth integration
fixtures and are gated on a running Postgres.
"""

from __future__ import annotations

import inspect
from pathlib import Path


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


def test_submit_customer_batch_handles_unique_violation_race():
    """Codex P1 fix: SELECT-then-INSERT is not atomic. Two concurrent
    retries with the same key can both miss the SELECT and race on
    the partial UNIQUE index. The loser must re-read and return the
    winner's record -- otherwise the loser gets a 500 and the
    idempotency contract is broken precisely in the case it exists
    to handle."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.submit_customer_batch)
    # Handler exists.
    assert "asyncpg.exceptions.UniqueViolationError" in src
    # Replay path re-reads by (account_id, idempotency_key).
    assert (
        "WHERE account_id = $1 AND idempotency_key = $2" in src
    )
    # Race-replay log line distinguishes from the normal-replay path.
    assert 'logger.info(\n                "llm_gateway_batch.submit replay (race)' in src


def test_submit_customer_batch_resumes_stale_pre_submit_atomically():
    """Codex P1 rounds 3+4 on PR-D4c: a row without provider_batch_id
    means the prior attempt never landed on Anthropic. Either it
    crashed pre-submit (status='queued', stale) or Anthropic
    rejected/timed out (status='submit_failed'). Both are resumable
    -- naive replay would loop forever returning the dead row, so
    the customer's batch never actually submits. Resume via an
    atomic SQL claim; concurrent in-flight queued calls are
    distinguished from stuck queued rows via updated_at age (60s =
    2x SDK timeout). Two concurrent resumes can't both win because
    the UPDATE is atomic."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.submit_customer_batch)
    # Atomic SQL claim covers BOTH submit_failed and stale-queued.
    assert "AND provider_batch_id IS NULL" in src
    assert "status = 'submit_failed'" in src
    assert "OR (status = 'queued'" in src
    assert "AND updated_at < NOW() - INTERVAL '60 seconds'" in src
    # account_id is in the WHERE (defense-in-depth, same pattern
    # as _persist_batch_usage's atomic claim).
    assert "WHERE id = $1\n                  AND account_id = $2" in src
    # Resume clears stale error_text (submit_failed sets it) and
    # bumps status back to queued so the row looks freshly-INSERTed.
    assert "SET updated_at = NOW(),\n                    status = 'queued',\n                    error_text = NULL" in src
    # Distinct log lines for each branch so ops can spot leaks.
    assert "submit replay (in-flight)" in src
    assert "submit resume pre-submit" in src


def test_submit_customer_batch_replay_only_when_provider_batch_id_set():
    """The fast-path replay must check provider_batch_id, not just
    status. A retry with the same key against a settled row
    (Anthropic accepted the submission) returns immediately;
    everything else falls through to the resume claim. Codex P1
    round 4 on PR-D4c -- the prior `status != 'queued'` check
    incorrectly treated submit_failed as settled."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.submit_customer_batch)
    # Replay guard keys on provider_batch_id only.
    assert 'if existing["provider_batch_id"]:' in src
    # No spurious status check before the replay return.
    assert 'or existing["status"] != "queued"' not in src


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
    the race is a no-op (claim returns no rows). Codex P1 + Copilot
    on PR-D4c: claim is also account-scoped so a misrouted batch_id
    can never flip the flag on a row outside the caller's account."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    assert "WHERE id = $1 AND account_id = $2 AND usage_tracked = FALSE" in src
    assert "RETURNING id" in src
    # Loser path returns cleanly.
    assert "if claim is None:" in src


def test_persist_batch_usage_pre_fetches_before_claiming():
    """Codex P1 + Copilot fix on PR-D4c: the SDK results fetch must
    happen BEFORE the atomic claim, not after. Otherwise a transient
    SDK error mid-iteration would happen with usage_tracked already
    flipped to TRUE -- the rollback-then-retry pattern would re-emit
    the items that already landed and double-count against the
    customer. Pre-fetching means failures here leave usage_tracked
    FALSE (no claim was made) so the next poll retries cleanly."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    fetch_idx = src.find("client.messages.batches.results")
    claim_idx = src.find("SET usage_tracked = TRUE")
    assert fetch_idx > 0 and claim_idx > 0, "Both phases must be present"
    assert fetch_idx < claim_idx, (
        "Results fetch must precede the atomic claim so SDK failures "
        "don't leave the row in an unrecoverable claimed state."
    )


def test_persist_batch_usage_does_not_roll_back_after_claim():
    """Codex P1 + Copilot on PR-D4c: once the claim flips
    usage_tracked to TRUE, we must NOT roll it back to FALSE on
    subsequent failures. trace_llm_call failures during the persist
    phase are logged loudly but the claim stays TRUE so retries
    can't double-emit the items that already landed. A pending-
    usage index in migration 318 lets ops scan for orphaned
    partial-write batches."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    # No rollback statement.
    assert "SET usage_tracked = FALSE" not in src
    # But trace_llm_call failures still log so ops can see them.
    assert (
        'logger.exception(\n                '
        '"llm_gateway_batch.persist_usage: trace_llm_call failed' in src
    )


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


def test_refresh_retries_persist_usage_on_terminal_untracked():
    """Codex P1 fix: a batch can be terminal but still have
    ``usage_tracked = FALSE`` if a previous _persist_batch_usage
    rolled back mid-iteration. The early-terminal short-circuit
    must NOT skip retrying in that case -- otherwise a single
    transient SDK error permanently loses the customer's usage."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.refresh_customer_batch_status)
    # Retry condition explicit.
    assert "not record.usage_tracked" in src
    assert 'record.status in ("ended", "canceled", "expired")' in src
    # Re-read after retry so the caller sees usage_tracked = TRUE.
    assert "return await get_customer_batch(" in src


def test_customer_batch_record_exposes_usage_tracked():
    """The dataclass must surface ``usage_tracked`` so the refresh
    path can decide whether to retry. Defaults FALSE so test mocks
    that don't include the column still construct cleanly."""
    from atlas_brain.services.llm_gateway_batch import CustomerBatchRecord

    fields = {f.name for f in CustomerBatchRecord.__dataclass_fields__.values()}
    assert "usage_tracked" in fields
    field = CustomerBatchRecord.__dataclass_fields__["usage_tracked"]
    assert field.default is False


# ---- Cost helper -------------------------------------------------------


def test_estimate_cost_uses_atlas_pricing_config():
    """The cost helper reuses ``settings.ftl_tracing.pricing`` so
    pricing changes update one place (not duplicated here)."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._estimate_cost_usd)
    assert "settings.ftl_tracing.pricing.cost_usd" in src
    assert '"anthropic"' in src
