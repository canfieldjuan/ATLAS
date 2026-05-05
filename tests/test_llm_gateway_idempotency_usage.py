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


# ---- Migration 319: per-item llm_usage uniqueness ----------------------


def test_migration_319_adds_batch_columns_and_unique_index():
    """PR-D4d post-audit: per-item idempotency at the DB level.
    batch_id + custom_id columns + partial UNIQUE index on
    (account_id, batch_id, custom_id) WHERE batch_id IS NOT NULL
    so retries of _persist_batch_usage can ON CONFLICT DO NOTHING
    on rows that already landed."""
    sql = _read_migration("319_llm_usage_batch_uniqueness.sql")
    # New columns nullable so existing /chat traffic still inserts cleanly.
    assert "ADD COLUMN IF NOT EXISTS batch_id  UUID" in sql
    assert "ADD COLUMN IF NOT EXISTS custom_id TEXT" in sql
    # Partial UNIQUE index scoped to batch rows only.
    assert "uq_llm_usage_batch_item" in sql
    assert "(account_id, batch_id, custom_id)" in sql
    assert "WHERE batch_id IS NOT NULL" in sql


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


def test_in_flight_replay_re_reads_row_before_returning():
    """PR-D4d post-audit: when the resume claim returns None
    (recent in-flight call or another retry already claimed it),
    we must re-read the row before returning. The original
    ``existing`` snapshot was taken before the claim attempt,
    so a concurrent retry that just succeeded (set
    provider_batch_id, transitioned to in_progress) would be
    invisible to the snapshot. Customer would see stale data."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch.submit_customer_batch)
    in_flight_idx = src.find("if resumed is None:")
    re_read_idx = src.find("latest = await pool.fetchrow(")
    log_idx = src.find('"llm_gateway_batch.submit replay (in-flight) "')
    return_idx = src.rfind("return _row_to_record(replay_row)")
    assert in_flight_idx > 0 and re_read_idx > 0
    assert in_flight_idx < re_read_idx < log_idx
    assert log_idx < return_idx
    # The returned record uses the freshly-read row (or the
    # snapshot as a defensive fallback if the row vanished --
    # which shouldn't happen under normal flow).
    assert "replay_row = latest if latest is not None else existing" in src


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


def test_persist_batch_usage_idempotent_via_on_conflict():
    """PR-D4d post-audit: per-item idempotency comes from the
    UNIQUE (account_id, batch_id, custom_id) partial index in
    migration 319 + ``ON CONFLICT DO NOTHING`` on the INSERT.
    A retry that hits an already-written row no-ops at the DB
    level instead of double-writing. The ``usage_tracked`` flag
    flip at the end is account-scoped so cross-account batch_id
    reuse can never flip a foreign row."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    # Per-item dedup at DB level.
    assert "ON CONFLICT (account_id, batch_id, custom_id)" in llm_gateway_batch._BATCH_USAGE_INSERT_SQL
    assert "WHERE batch_id IS NOT NULL" in llm_gateway_batch._BATCH_USAGE_INSERT_SQL
    assert "DO NOTHING" in llm_gateway_batch._BATCH_USAGE_INSERT_SQL
    # Account-scoped flag flip.
    assert "WHERE id = $1 AND account_id = $2 AND usage_tracked = FALSE" in src


def test_persist_batch_usage_pre_fetches_before_writing():
    """PR-D4d: the Anthropic results fetch must happen BEFORE any
    DB writes. SDK failures during the fetch leave usage_tracked
    FALSE so the next poll retries cleanly with no partial DB
    state visible."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    fetch_idx = src.find("client.messages.batches.results")
    insert_idx = src.find("conn.executemany(")
    flag_idx = src.find("SET usage_tracked = TRUE")
    assert fetch_idx > 0 and insert_idx > 0 and flag_idx > 0
    assert fetch_idx < insert_idx < flag_idx, (
        "Order must be: fetch results -> insert rows -> flip flag. "
        "Any other ordering re-introduces the partial-write loss the "
        "PR-D4d audit found."
    )


def test_persist_batch_usage_writes_directly_via_pool():
    """PR-D4d post-audit fix: PR-D4c routed per-item writes through
    ``trace_llm_call`` -> tracer.end_span -> loop.create_task --
    fire-and-forget, so DB failures could not be caught and the
    flag was flipped before writes actually landed. PR-D4d issues
    the INSERT directly via pool.executemany under a transaction
    so failures surface synchronously."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    # Direct executemany under explicit transaction.
    assert "async with pool.acquire() as conn:" in src
    assert "async with conn.transaction():" in src
    assert "await conn.executemany(_BATCH_USAGE_INSERT_SQL" in src
    # No more fire-and-forget tracer for batch items.
    # (docstring may reference the old approach by name; what we
    # care about is the absence of an actual call site or import).
    assert "trace_llm_call(" not in src
    assert "from ..pipelines.llm import trace_llm_call" not in src


def test_persist_batch_usage_flips_flag_only_after_inserts():
    """PR-D4d: the usage_tracked flip happens AFTER conn.executemany
    returns successfully. If the inserts raise (transient asyncpg
    error, schema drift), the flag stays FALSE so a retry can
    re-attempt. Conflicts on already-written items are absorbed
    by ON CONFLICT DO NOTHING, not by skipping the flip."""
    from atlas_brain.services import llm_gateway_batch

    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    insert_idx = src.find("conn.executemany(_BATCH_USAGE_INSERT_SQL")
    flag_update_idx = src.find("SET usage_tracked = TRUE")
    assert insert_idx > 0 and flag_update_idx > 0
    assert insert_idx < flag_update_idx, (
        "The flag must flip only after inserts commit. Otherwise the "
        "round-trip through the fire-and-forget tracer reintroduces "
        "the partial-write loss the audit caught."
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


def test_persist_batch_usage_threads_account_id_into_row():
    """PR-D3 per-account scoping requires account_id on each
    llm_usage row. PR-D4d writes the column directly (rather than
    through tracer metadata) and also records batch_id + custom_id
    natively so the partial UNIQUE index can dedup retries."""
    from atlas_brain.services import llm_gateway_batch

    sql = llm_gateway_batch._BATCH_USAGE_INSERT_SQL
    src = inspect.getsource(llm_gateway_batch._persist_batch_usage)
    # account_id, batch_id, custom_id are explicit columns now.
    assert "account_id" in sql
    assert "batch_id" in sql
    assert "custom_id" in sql
    # Args tuple threads them into the INSERT.
    assert "str(account_id)" in src
    assert "batch_id" in src
    assert 'item["custom_id"]' in src
    # Metadata payload still carries the same triplet so /api/v1/llm/usage
    # consumers that read metadata.endpoint / metadata.batch_id keep
    # working (non-breaking change for any downstream rollups).
    assert '"account_id": str(account_id)' in src
    assert '"batch_id": str(batch_id)' in src
    assert '"custom_id": item["custom_id"]' in src


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
