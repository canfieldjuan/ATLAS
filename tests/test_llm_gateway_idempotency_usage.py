"""Tests for batch idempotency + per-item usage tracking (PR-D4c).

Pure structural + source-text tests (no live Anthropic calls).
DB-bound integration tests live with other auth integration
fixtures and are gated on a running Postgres.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid as _uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch


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


def test_migration_319_check_constraint_blocks_null_or_blank_custom_id():
    """Copilot on PR-D4d: a NULL custom_id is treated as distinct
    in UNIQUE indexes, so the partial UNIQUE alone doesn't prevent
    double-write. CHECK constraint enforces the invariant at the
    DB level: any row with batch_id set MUST have a non-empty
    custom_id."""
    sql = _read_migration("319_llm_usage_batch_uniqueness.sql")
    assert "llm_usage_batch_requires_custom_id" in sql
    assert (
        "CHECK (\n"
        "            batch_id IS NULL\n"
        "            OR (custom_id IS NOT NULL AND custom_id <> '')\n"
        "        )"
    ) in sql


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
    # Direct executemany under pool.transaction() -- atlas's
    # DatabasePool.acquire is async-def-returning-connection, NOT
    # an async context manager, so async-with-acquire would raise
    # at runtime. pool.transaction() handles the acquire + tx scope
    # in one block. Codex P1 fix on PR-D4d.
    assert "async with pool.transaction() as conn:" in src
    assert "await conn.executemany(_BATCH_USAGE_INSERT_SQL" in src
    # The buggy acquire() pattern must not return.
    assert "async with pool.acquire()" not in src
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


# ---- Runtime smoke (catches "wrong primitive" class of bugs) -----------


class _FakePoolConn:
    """Records executemany calls so the smoke test can assert the
    INSERT was issued with the expected args. Mirrors asyncpg's
    Connection.executemany signature (just enough for our path)."""

    def __init__(self) -> None:
        self.executemany_calls: list[tuple[str, list[tuple[Any, ...]]]] = []

    async def executemany(self, query: str, args: list[tuple[Any, ...]]) -> None:
        self.executemany_calls.append((query, args))


class _FakePool:
    """Minimal stand-in for atlas's DatabasePool used to runtime-
    exercise _persist_batch_usage. Implements:
      - ``transaction()`` -- async context manager yielding a conn
        (matches storage/database.py:144).
      - ``execute()`` -- awaitable for the post-insert flag flip.
      - ``fetchval()`` -- awaitable for the short-circuit check
        on ``usage_tracked`` (PR-D4d Copilot).
      - DOES NOT implement ``acquire()``, so any code that tries
        the wrong primitive blows up immediately."""

    def __init__(self, usage_tracked: bool = False) -> None:
        self.conn = _FakePoolConn()
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self._usage_tracked = usage_tracked

    @asynccontextmanager
    async def transaction(self):
        yield self.conn

    async def execute(self, query: str, *args: Any) -> None:
        self.execute_calls.append((query, args))

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchval_calls.append((query, args))
        # Mirror real DB behaviour: SELECT usage_tracked returns the
        # column's bool value (or None if no matching row).
        if "SELECT usage_tracked" in query:
            return self._usage_tracked
        return None


class _FakeAnthropicResultEntry:
    def __init__(self, custom_id: str, input_tokens: int, output_tokens: int):
        self.custom_id = custom_id

        class _Usage:
            pass
        u = _Usage()
        u.input_tokens = input_tokens
        u.output_tokens = output_tokens
        u.cache_read_input_tokens = 0
        u.cache_creation_input_tokens = 0

        class _Message:
            pass
        m = _Message()
        m.usage = u
        m.id = f"msg_{custom_id}"

        class _Result:
            pass
        r = _Result()
        r.type = "succeeded"
        r.message = m

        self.result = r


class _FakeAsyncIter:
    def __init__(self, entries):
        self._entries = list(entries)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._entries:
            raise StopAsyncIteration
        return self._entries.pop(0)


class _FakeBatchesResource:
    def __init__(self, entries):
        self._entries = entries

    async def results(self, provider_batch_id: str):
        return _FakeAsyncIter(self._entries)


class _FakeMessagesResource:
    def __init__(self, entries):
        self.batches = _FakeBatchesResource(entries)


class _FakeAsyncAnthropic:
    """Stands in for AsyncAnthropic. Supports async-with for the
    connection-pool teardown the real client does."""

    def __init__(self, *args, entries=None, **kwargs):
        self.messages = _FakeMessagesResource(entries or [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_persist_batch_usage_runtime_smoke():
    """Catches the "wrong primitive" class of bug: source-text
    pins passed for `async with pool.acquire()` but it would have
    raised at runtime because atlas's DatabasePool.acquire is a
    plain async-def. This smoke test exercises the full path with
    a fake pool that only implements the correct interface, so
    any future regression is caught at test time."""
    from atlas_brain.services import llm_gateway_batch

    pool = _FakePool()
    account_id = _uuid.uuid4()
    batch_id = _uuid.uuid4()
    entries = [
        _FakeAnthropicResultEntry("item-1", 100, 50),
        _FakeAnthropicResultEntry("item-2", 200, 75),
    ]

    def _fake_anthropic_factory(*args, **kwargs):
        return _FakeAsyncAnthropic(entries=entries)

    with patch("anthropic.AsyncAnthropic", _fake_anthropic_factory):
        asyncio.run(llm_gateway_batch._persist_batch_usage(
            pool,
            account_id=account_id,
            batch_id=batch_id,
            provider_batch_id="msgbatch_test",
            model="claude-haiku-4-5-20251001",
            api_key="test-key",
        ))

    # executemany was called once with both rows.
    assert len(pool.conn.executemany_calls) == 1
    sql, args = pool.conn.executemany_calls[0]
    assert "INSERT INTO llm_usage" in sql
    assert "ON CONFLICT (account_id, batch_id, custom_id)" in sql
    assert len(args) == 2
    # args tuple positions: span_name, model, input_tok, output_tok, total,
    # billable_in, cached, cache_write, cost, metadata, endpoint,
    # provider_request_id, account_id, batch_id, custom_id
    assert args[0][0] == "llm_gateway.batch_item"  # span_name
    assert args[0][1] == "claude-haiku-4-5-20251001"  # model
    assert args[0][2] == 100  # input_tokens
    assert args[0][3] == 50   # output_tokens
    assert args[0][4] == 150  # total_tokens
    assert args[0][13] == batch_id  # batch_id
    assert args[0][14] == "item-1"  # custom_id
    assert args[1][14] == "item-2"  # custom_id of second item

    # Flag flip ran AFTER the insert -- check both ordering and the
    # account-scoped predicate.
    assert len(pool.execute_calls) == 1
    flag_sql, flag_args = pool.execute_calls[0]
    assert "SET usage_tracked = TRUE" in flag_sql
    assert "WHERE id = $1 AND account_id = $2 AND usage_tracked = FALSE" in flag_sql
    assert flag_args == (batch_id, account_id)


def test_persist_batch_usage_short_circuits_when_already_tracked():
    """Copilot on PR-D4d: when a concurrent poller already
    persisted usage and flipped the flag, _persist_batch_usage
    should skip the (potentially expensive) Anthropic results
    fetch and exit immediately. Saves provider calls under
    polling races; correctness is preserved either way (the flag
    flip is idempotent + account-scoped)."""
    from atlas_brain.services import llm_gateway_batch

    pool = _FakePool(usage_tracked=True)

    # Anthropic factory shouldn't be touched because the short-
    # circuit fires first. Wire one that asserts if called.
    sentinel: dict[str, bool] = {"sdk_called": False}

    def _exploding_anthropic_factory(*args, **kwargs):
        sentinel["sdk_called"] = True
        return _FakeAsyncAnthropic(entries=[])

    with patch("anthropic.AsyncAnthropic", _exploding_anthropic_factory):
        asyncio.run(llm_gateway_batch._persist_batch_usage(
            pool,
            account_id=_uuid.uuid4(),
            batch_id=_uuid.uuid4(),
            provider_batch_id="msgbatch_already_tracked",
            model="claude-haiku-4-5-20251001",
            api_key="test-key",
        ))

    # Pre-check ran.
    assert len(pool.fetchval_calls) == 1
    assert "SELECT usage_tracked" in pool.fetchval_calls[0][0]
    # SDK was NOT called.
    assert sentinel["sdk_called"] is False
    # No inserts, no flag flip.
    assert pool.conn.executemany_calls == []
    assert pool.execute_calls == []


def test_persist_batch_usage_rejects_blank_custom_id():
    """Copilot on PR-D4d: a blank custom_id from the provider
    would be silently absorbed by ON CONFLICT (NULL/'' treated as
    distinct in the UNIQUE index), but we'd still flip
    usage_tracked TRUE and undercount. Raise instead so the flag
    stays FALSE for retry / ops triage."""
    from atlas_brain.services import llm_gateway_batch

    pool = _FakePool()
    entries = [_FakeAnthropicResultEntry("", 100, 50)]

    def _fake_anthropic_factory(*args, **kwargs):
        return _FakeAsyncAnthropic(entries=entries)

    with patch("anthropic.AsyncAnthropic", _fake_anthropic_factory):
        try:
            asyncio.run(llm_gateway_batch._persist_batch_usage(
                pool,
                account_id=_uuid.uuid4(),
                batch_id=_uuid.uuid4(),
                provider_batch_id="msgbatch_blank",
                model="claude-haiku-4-5-20251001",
                api_key="test-key",
            ))
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "blank custom_id" in str(exc)

    # No inserts, no flag flip -- usage_tracked stays FALSE so retry works.
    assert pool.conn.executemany_calls == []
    assert pool.execute_calls == []


def test_persist_batch_usage_rejects_duplicate_custom_id():
    """Copilot on PR-D4d: duplicate custom_id within a single
    batch result set would have the second row absorbed by
    ON CONFLICT, again undercounting silently. Detect at the
    Python layer before any DB write."""
    from atlas_brain.services import llm_gateway_batch

    pool = _FakePool()
    entries = [
        _FakeAnthropicResultEntry("dup", 100, 50),
        _FakeAnthropicResultEntry("dup", 200, 75),
    ]

    def _fake_anthropic_factory(*args, **kwargs):
        return _FakeAsyncAnthropic(entries=entries)

    with patch("anthropic.AsyncAnthropic", _fake_anthropic_factory):
        try:
            asyncio.run(llm_gateway_batch._persist_batch_usage(
                pool,
                account_id=_uuid.uuid4(),
                batch_id=_uuid.uuid4(),
                provider_batch_id="msgbatch_dup",
                model="claude-haiku-4-5-20251001",
                api_key="test-key",
            ))
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "duplicate custom_id" in str(exc)

    assert pool.conn.executemany_calls == []
    assert pool.execute_calls == []


def test_persist_batch_usage_skips_db_when_no_succeeded_items():
    """If the batch had only errored / canceled items, items_to_persist
    is empty -- we should NOT issue an executemany (asyncpg raises on
    empty rows-args) but we SHOULD still flip the flag so the
    refresh path doesn't keep retrying."""
    from atlas_brain.services import llm_gateway_batch

    pool = _FakePool()
    # Empty entries iterator -> nothing to persist.
    def _fake_anthropic_factory(*args, **kwargs):
        return _FakeAsyncAnthropic(entries=[])

    with patch("anthropic.AsyncAnthropic", _fake_anthropic_factory):
        asyncio.run(llm_gateway_batch._persist_batch_usage(
            pool,
            account_id=_uuid.uuid4(),
            batch_id=_uuid.uuid4(),
            provider_batch_id="msgbatch_empty",
            model="claude-haiku-4-5-20251001",
            api_key="test-key",
        ))

    # No executemany call (no rows to write).
    assert pool.conn.executemany_calls == []
    # But flag flip still ran -- otherwise refresh keeps retrying.
    assert len(pool.execute_calls) == 1
    assert "SET usage_tracked = TRUE" in pool.execute_calls[0][0]
