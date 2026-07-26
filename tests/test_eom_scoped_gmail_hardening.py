"""Regression tests for the four #2200 review findings.

Each test proves the failure direction per AGENTS.md 3i: the fixture reproduces
the reviewed defect's exact path, and defeating the fix (where practical) is
shown to flip the result in the paired probe test.
"""
from __future__ import annotations

import json
import os
import threading
import uuid

import asyncio
import logging
from contextlib import asynccontextmanager

import pytest

import atlas_brain.services.customer_context as context_mod
import atlas_brain.services.crm_provider as crm_provider_mod
import atlas_brain.services.email_provider as email_provider_mod
import atlas_brain.storage.repositories.scoped_mailbox_credential as repo_mod
from atlas_brain.storage.migrations import MIGRATIONS_DIR
from atlas_brain.storage.config import db_settings

DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"

TEST_KEK = "test:DEj0-fNH6mOs5JYXn3Uv6ejEfP4PQ6XIqWla36eIR_U="

TENANT = "effingham_maids"
CONTACT_ID = "11111111-1111-1111-1111-111111111111"


class _CRM:
    async def get_contact(self, contact_id):
        return {
            "id": str(contact_id),
            "email": "customer@example.com",
            "full_name": "Customer",
            "business_context_id": TENANT,
        }

    async def get_interactions(self, _contact_id, *, limit, business_context_id):
        return []

    async def get_contact_appointments(self, _contact_id, *, business_context_id):
        return []


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(crm_provider_mod, "get_crm_provider", lambda: _CRM())
    previous = context_mod._customer_context_service
    context_mod._customer_context_service = None
    yield context_mod.get_customer_context_service()
    context_mod._customer_context_service = previous


# --- finding 1: setup failures keep a sanitized, diagnosable reason ---------


@pytest.mark.asyncio
async def test_provider_setup_failure_logs_exception_class(
    service, monkeypatch, caplog
):
    async def _boom(_context):
        raise TimeoutError("refresh_token=SECRET-DO-NOT-LOG")

    monkeypatch.setattr(email_provider_mod, "get_scoped_inbox_provider", _boom)
    with caplog.at_level(logging.WARNING, logger=context_mod.logger.name):
        ctx = await service.get_context(
            CONTACT_ID, business_context_id=TENANT
        )
    assert ctx.inbox_email_source_omitted is True
    setup_lines = [
        r.getMessage() for r in caplog.records if "setup failed" in r.getMessage()
    ]
    assert setup_lines, "setup failure must be logged"
    # The CLASS is the diagnosable reason; the MESSAGE may carry credential
    # text and must never reach the log line.
    assert any("TimeoutError" in line for line in setup_lines)
    assert all("SECRET-DO-NOT-LOG" not in line for line in setup_lines)


# --- finding 2: late revocation is an omitted source, not an empty inbox ----


class _RevokedMidReadProvider:
    async def list_messages(self, *, query, max_results):
        raise repo_mod.ScopedMailboxCredentialUnavailable(
            "scoped_gmail_credentials_unavailable"
        )


class _BrokenProvider:
    async def list_messages(self, *, query, max_results):
        raise ValueError("transient provider failure")


@pytest.mark.asyncio
async def test_late_revocation_marks_source_omitted(service, monkeypatch):
    async def _resolver(_context):
        return _RevokedMidReadProvider()

    monkeypatch.setattr(
        email_provider_mod, "get_scoped_inbox_provider", _resolver
    )
    ctx = await service.get_context(CONTACT_ID, business_context_id=TENANT)
    assert ctx.inbox_emails == []
    assert ctx.inbox_email_source_omitted is True


@pytest.mark.asyncio
async def test_ordinary_read_failure_is_not_reported_as_omitted(
    service, monkeypatch
):
    """The classification boundary: a transient failure stays an empty read.

    Without this, finding 2's fix could over-correct and report every provider
    hiccup as a revoked authorization.
    """

    async def _resolver(_context):
        return _BrokenProvider()

    monkeypatch.setattr(
        email_provider_mod, "get_scoped_inbox_provider", _resolver
    )
    ctx = await service.get_context(CONTACT_ID, business_context_id=TENANT)
    assert ctx.inbox_emails == []
    assert ctx.inbox_email_source_omitted is False


# --- finding 3: the standalone CRM server migrates at startup ---------------


class _Pool:
    def __init__(self, initialized):
        self.is_initialized = initialized


def _lifespan_fakes(pool):
    ran = []
    closed = []

    async def _init():
        return None

    async def _close():
        closed.append(True)

    async def _migrate(p, *, only=None):
        ran.append((p, only))

    return {
        "init_database_fn": _init,
        "get_db_pool_fn": lambda: pool,
        "run_migrations_fn": _migrate,
        "close_database_fn": _close,
    }, ran, closed


@pytest.mark.asyncio
async def test_crm_lifespan_runs_migrations_when_pool_is_up():
    import atlas_brain.mcp.crm_server as crm_server

    pool = _Pool(initialized=True)
    fns, ran, closed = _lifespan_fakes(pool)
    async with crm_server._database_lifespan(**fns):
        pass
    assert closed == [True]
    assert len(ran) == 1
    applied_pool, only = ran[0]
    assert applied_pool is pool
    # NOT the whole chain: 076+ reference an out-of-band product_metadata table
    # that no migration creates, so a fresh database dies partway through and
    # would take this server's whole lifespan with it.
    assert only == crm_server.SCOPED_MAILBOX_MIGRATIONS
    assert "350_scoped_mailbox_credentials" in only


@pytest.mark.asyncio
async def test_crm_lifespan_skips_migrations_loudly_when_pool_is_down(caplog):
    import atlas_brain.mcp.crm_server as crm_server

    pool = _Pool(initialized=False)
    fns, ran, closed = _lifespan_fakes(pool)
    with caplog.at_level(logging.WARNING):
        async with crm_server._database_lifespan(**fns):
            pass
    assert ran == []
    assert closed == [True]
    assert any("migrations skipped" in r.getMessage() for r in caplog.records)


# --- finding 4: refresh waiters queue without holding pool connections ------


class _RowLockPool:
    """Reproduces the FOR UPDATE profile: a connection is HELD while blocked."""

    def __init__(self, row):
        self.row = row
        self.open = 0
        self.max_open = 0
        self.holder_done = asyncio.Event()
        self.has_holder = False

    @asynccontextmanager
    async def transaction(self):
        self.open += 1
        self.max_open = max(self.max_open, self.open)
        try:
            yield _RowLockConn(self)
        finally:
            self.open -= 1

    async def fetchrow(self, query, *args):
        """Pool-level query: asyncpg holds a connection for its duration, and
        an UPDATE against a FOR UPDATE-locked row blocks while holding it."""
        self.open += 1
        self.max_open = max(self.max_open, self.open)
        try:
            if "UPDATE scoped_mailbox_credentials" in query and self.has_holder:
                await self.holder_done.wait()
            return {"generation": 2}
        finally:
            self.open -= 1


class _RowLockConn:
    def __init__(self, pool):
        self.pool = pool

    async def fetchrow(self, query, *args):
        if "UPDATE scoped_mailbox_credentials" in query:
            return {"generation": 2}
        if not self.pool.has_holder:
            self.pool.has_holder = True
            return dict(self.pool.row)
        # Later entrant on a locked row: block, still holding the connection,
        # exactly as FOR UPDATE does.
        await self.pool.holder_done.wait()
        return dict(self.pool.row)


@pytest.fixture
def row_lock_repo(monkeypatch):
    import atlas_brain.config as config_mod

    monkeypatch.setattr(
        config_mod.settings.saas_auth, "byok_encryption_kek", TEST_KEK
    )
    encrypted, kid = repo_mod._encrypt_bundle(
        client_id="cid", client_secret="csec", refresh_token="rtok"
    )
    pool = _RowLockPool(
        {"encrypted_credentials": encrypted, "encryption_kid": kid, "generation": 1}
    )
    yield repo_mod.ScopedMailboxCredentialRepository(pool=pool), pool


@pytest.mark.asyncio
async def test_concurrent_refreshes_hold_at_most_one_connection(row_lock_repo):
    repo, pool = row_lock_repo
    release = asyncio.Event()
    entered = asyncio.Event()

    async def _worker():
        async with repo.locked_gmail(TENANT):
            entered.set()
            await release.wait()

    workers = [asyncio.create_task(_worker()) for _ in range(10)]
    await asyncio.wait_for(entered.wait(), timeout=5)
    # Let every other worker advance as far as it can while the first holds
    # the locked section.
    for _ in range(20):
        await asyncio.sleep(0)
    assert pool.max_open == 1, (
        f"waiters must queue on the in-process gate WITHOUT a pool "
        f"connection; saw {pool.max_open} concurrently-held connections"
    )
    pool.holder_done.set()
    release.set()
    await asyncio.wait_for(asyncio.gather(*workers), timeout=5)


@pytest.mark.asyncio
async def test_cross_context_refresh_draw_is_bounded_by_the_pool_budget(
    row_lock_repo,
):
    """3i probe, no internals patched: workers on DISTINCT contexts take
    DISTINCT per-context gates, so that gate cannot bound them -- each one
    would otherwise enter a transaction and hold its connection through the
    token exchange, taking the whole default pool. The global slot caps the
    portfolio-wide draw and leaves the rest of the pool for unrelated work."""
    repo, pool = row_lock_repo
    release = asyncio.Event()
    entered = asyncio.Event()

    async def _worker(context):
        async with repo.locked_gmail(context):
            entered.set()
            await release.wait()

    workers = [
        asyncio.create_task(_worker(f"context-{i}")) for i in range(10)
    ]
    await asyncio.wait_for(entered.wait(), timeout=5)
    for _ in range(20):
        await asyncio.sleep(0)
    budget = repo_mod._refresh_budget()
    assert pool.max_open == budget, (
        f"concurrent refreshes across distinct contexts must hold at most "
        f"{budget} connections; saw {pool.max_open}"
    )
    assert budget < db_settings.max_pool_size, (
        "the budget must leave pool headroom for non-refresh work"
    )
    pool.holder_done.set()
    release.set()
    await asyncio.wait_for(asyncio.gather(*workers), timeout=5)


@pytest.mark.asyncio
async def test_refresh_budget_tracks_the_configured_pool_size(row_lock_repo):
    """Proven-failure companion (3i): move the ONLY input the cap derives from
    -- the budget -- through the constructor seam, and the observed draw moves
    with it. The bound is the slot, not the fixture: given headroom for all
    ten, all ten run, which is the pre-fix profile."""
    _, pool = row_lock_repo
    repo = repo_mod.ScopedMailboxCredentialRepository(pool=pool, refresh_budget=10)
    release = asyncio.Event()
    entered = asyncio.Event()

    async def _worker(context):
        async with repo.locked_gmail(context):
            entered.set()
            await release.wait()

    workers = [
        asyncio.create_task(_worker(f"context-{i}")) for i in range(10)
    ]
    await asyncio.wait_for(entered.wait(), timeout=5)
    for _ in range(20):
        await asyncio.sleep(0)
    assert pool.max_open == 10, (
        "with budget for all ten, none should be held back -- the cap must "
        f"follow the stated budget; saw {pool.max_open}"
    )
    pool.holder_done.set()
    release.set()
    await asyncio.wait_for(asyncio.gather(*workers), timeout=5)


@pytest.mark.asyncio
async def test_revoke_shares_the_refresh_gate(row_lock_repo):
    """A same-context revoke must queue on the gate, not on a pool connection.

    Reviewed case: a refresh holds the row through Google's token exchange
    while same-context rebind/revoke retries arrive. Ungated, each occupies a
    connection blocked on the row lock.
    """
    repo, pool = row_lock_repo
    release = asyncio.Event()
    entered = asyncio.Event()

    async def _refresher():
        async with repo.locked_gmail(TENANT):
            entered.set()
            await release.wait()

    refresher = asyncio.create_task(_refresher())
    await asyncio.wait_for(entered.wait(), timeout=5)

    revokes = [
        asyncio.create_task(repo.revoke_gmail(TENANT)) for _ in range(9)
    ]
    for _ in range(20):
        await asyncio.sleep(0)
    assert pool.max_open == 1, (
        f"same-context revokes must wait on the gate without a pool "
        f"connection; saw {pool.max_open}"
    )

    pool.holder_done.set()
    release.set()
    await asyncio.wait_for(refresher, timeout=5)
    for task in revokes:
        task.cancel()
    await asyncio.gather(*revokes, return_exceptions=True)


class _HydrationClient:
    """Gmail client whose envelope reads fail on a chosen exception class."""

    def __init__(self, exc):
        self._exc = exc
        self.closed = False

    async def list_messages(self, query, max_results):
        return [{"id": "m1"}, {"id": "m2"}]

    async def get_message_envelope(self, message_id):
        if message_id == "m2":
            raise self._exc
        return {"id": message_id, "from": "a@example.com"}

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_revocation_during_hydration_is_not_swallowed():
    """F2: revocation landing AFTER the credential read but DURING metadata
    hydration must reach the caller. Dropping it returns a short inbox that
    reads exactly like a genuinely short one."""
    from atlas_brain.services.email_provider import ScopedGmailEmailProvider
    from atlas_brain.storage.repositories.scoped_mailbox_credential import (
        ScopedMailboxCredentialUnavailable,
    )

    client = _HydrationClient(
        ScopedMailboxCredentialUnavailable("scoped_gmail_credentials_unavailable")
    )
    provider = ScopedGmailEmailProvider(client)

    with pytest.raises(ScopedMailboxCredentialUnavailable):
        await provider.list_messages()
    assert client.closed, "the client must still be closed on the raising path"


@pytest.mark.asyncio
async def test_ordinary_hydration_failure_still_drops_only_that_message():
    """Both directions: the re-raise must be scoped to revocation. A transient
    per-message read failure keeps its drop-one-message behaviour."""
    from atlas_brain.services.email_provider import ScopedGmailEmailProvider

    provider = _HydrationClient(RuntimeError("transient metadata read failure"))
    messages = await ScopedGmailEmailProvider(provider).list_messages()

    assert [m["id"] for m in messages] == ["m1"]


@pytest.mark.asyncio
async def test_crm_lifespan_survives_a_failing_prerequisite_migration(caplog):
    """R4/R12: a migration failure must degrade the Gmail binding to its
    documented fail-closed state, not abort the CRM server. Contacts, tickets
    and appointments do not depend on migration 350."""
    import atlas_brain.mcp.crm_server as crm_server

    pool = _Pool(initialized=True)
    fns, ran, closed = _lifespan_fakes(pool)
    served = []

    async def _boom(p, *, only=None):
        raise RuntimeError("relation \"product_metadata\" does not exist")

    fns["run_migrations_fn"] = _boom

    with caplog.at_level(logging.WARNING):
        async with crm_server._database_lifespan(**fns):
            served.append(True)

    assert served == [True], "the server must still come up and serve tools"
    assert closed == [True], "shutdown must still run"
    messages = [r.getMessage() for r in caplog.records]
    assert any("scoped Gmail" in m for m in messages), (
        "the operator must be told which capability is degraded"
    )
    assert not any("product_metadata" in m for m in messages), (
        "log the exception class, not a message that may carry schema detail"
    )


class _RealPoolAdapter:
    """Gives a real asyncpg pool the DatabasePool surface the repository uses.

    Everything underneath is genuine: real connections, real transactions, real
    row locks. Only the method names are bridged.
    """

    def __init__(self, pool):
        self._pool = pool

    @asynccontextmanager
    async def transaction(self):
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def fetchrow(self, *args):
        return await self._pool.fetchrow(*args)

    async def fetch(self, *args):
        return await self._pool.fetch(*args)

    async def fetchval(self, *args):
        return await self._pool.fetchval(*args)

    async def execute(self, *args):
        return await self._pool.execute(*args)


@pytest.mark.asyncio
async def test_real_postgres_row_lock_serializes_independent_sessions(monkeypatch):
    """R2/R8: the Review Contract claims PostgreSQL serializes refresh-token
    rotation. Every single-loop test of that claim is masked by the in-process
    `_refresh_gate` added for the pool-exhaustion finding -- it serializes the
    actors before either reaches `FOR UPDATE`, so the row lock could be deleted
    and those tests would stay green.

    This one puts the second actor on its own event loop in its own thread.
    `_refresh_gate` keys by running loop, so the two actors take DIFFERENT
    gates, hold DIFFERENT pooled connections, and the row lock is the only
    thing left that can order them.
    """
    import asyncpg

    import atlas_brain.config as config_mod
    from atlas_brain.auth import encryption
    from atlas_brain.storage.repositories import scoped_mailbox_credential as smc

    monkeypatch.setattr(
        config_mod.settings.saas_auth, "byok_encryption_kek", TEST_KEK
    )

    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    context = "eom_rowlock_probe"
    schema = f"atlas_rowlock_{uuid.uuid4().hex}"
    admin = await asyncpg.connect(database_url)
    pool_a = None
    try:
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        await admin.execute(f'SET search_path TO "{schema}", public')
        await admin.execute(
            (MIGRATIONS_DIR / "350_scoped_mailbox_credentials.sql").read_text()
        )

        ss = {"search_path": f"{schema},public"}
        pool_a = await asyncpg.create_pool(
            database_url, min_size=1, max_size=2, server_settings=ss
        )

        repo_a = smc.ScopedMailboxCredentialRepository(
            pool=_RealPoolAdapter(pool_a)
        )
        await repo_a.bind_gmail(
            business_context_id=context,
            client_id="client-1",
            client_secret="secret-1",
            refresh_token="token-initial",
        )

        first_holder_in = threading.Event()
        second_may_start = threading.Event()
        observed: list[str] = []

        async def _second_actor():
            # Its own loop -> its own _refresh_gate namespace. The pool must be
            # created HERE too: an asyncpg pool is bound to the loop that made
            # it, and that binding is what makes this a genuinely independent
            # backend session rather than a borrowed one.
            pool = await asyncpg.create_pool(
                database_url, min_size=1, max_size=2, server_settings=ss
            )
            try:
                repo_b = smc.ScopedMailboxCredentialRepository(
                    pool=_RealPoolAdapter(pool)
                )
                async with repo_b.locked_gmail(context) as locked:
                    observed.append(locked.credentials.refresh_token)
                    await locked.persist_refresh_token("token-from-b")
            finally:
                await pool.close()

        errors: list[BaseException] = []

        def _run_second():
            second_may_start.wait(timeout=10)
            try:
                asyncio.run(_second_actor())
            except BaseException as exc:  # surfaced below, never swallowed
                errors.append(exc)

        thread = threading.Thread(target=_run_second, daemon=True)
        thread.start()

        async with repo_a.locked_gmail(context) as locked:
            assert locked.credentials.refresh_token == "token-initial"
            first_holder_in.set()
            # Release the second actor while THIS transaction still holds the
            # row. It must block on FOR UPDATE rather than read the stale row.
            second_may_start.set()
            await asyncio.sleep(1.0)
            await locked.persist_refresh_token("token-from-a")

        await asyncio.get_running_loop().run_in_executor(None, thread.join, 20)
        assert not thread.is_alive(), "the second session never completed"
        assert not errors, f"second session raised: {errors[0]!r}"

        assert observed == ["token-from-a"], (
            "the second independent session must observe the token the first "
            f"one committed, not the pre-rotation row; saw {observed}"
        )

        row = await admin.fetchrow(
            "SELECT encryption_kid, encrypted_credentials, generation "
            "FROM scoped_mailbox_credentials WHERE business_context_id = $1",
            context,
        )
        payload = json.loads(
            encryption.decrypt_secret(
                bytes(row["encrypted_credentials"]), str(row["encryption_kid"])
            )
        )
        assert payload["refresh_token"] == "token-from-b", (
            "the last committed rotation must win"
        )
        assert row["generation"] == 3, (
            f"one bind plus two serialized rotations; saw {row['generation']}"
        )
    finally:
        if pool_a is not None:
            await pool_a.close()
        await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin.close()


def test_overlong_kek_identifier_fails_with_an_actionable_error(monkeypatch):
    """R4/R11: parse_kek_string applies no length bound, but encryption_kid is
    VARCHAR(64). An over-long kid would otherwise reach PostgreSQL and fail
    with StringDataRightTruncationError -- on a bind, or on the rotation write
    during a refresh, taking scoped Gmail down right after a valid KEK
    rotation. Fail early, naming the limit and the variable to change."""
    import atlas_brain.config as config_mod

    long_kid = "k" * (repo_mod._MAX_ENCRYPTION_KID_LENGTH + 1)
    _, key = TEST_KEK.split(":", 1)
    monkeypatch.setattr(
        config_mod.settings.saas_auth,
        "byok_encryption_kek",
        f"{long_kid}:{key}",
    )

    with pytest.raises(ValueError) as excinfo:
        repo_mod._encrypt_bundle(
            client_id="cid", client_secret="csec", refresh_token="rtok"
        )

    message = str(excinfo.value)
    assert str(repo_mod._MAX_ENCRYPTION_KID_LENGTH) in message
    assert "ATLAS_SAAS_BYOK_ENCRYPTION_KEK" in message
    assert key not in message, "the error must not echo the key material"


def test_kid_at_the_column_limit_is_accepted(monkeypatch):
    """Boundary second side: exactly 64 characters must still work, or the
    guard has moved the usable limit rather than matched the column."""
    import atlas_brain.config as config_mod

    exact_kid = "k" * repo_mod._MAX_ENCRYPTION_KID_LENGTH
    _, key = TEST_KEK.split(":", 1)
    monkeypatch.setattr(
        config_mod.settings.saas_auth,
        "byok_encryption_kek",
        f"{exact_kid}:{key}",
    )

    _, kid = repo_mod._encrypt_bundle(
        client_id="cid", client_secret="csec", refresh_token="rtok"
    )
    assert kid == exact_kid


class _DrainTrackingClient:
    """Gmail client that fails one envelope read and records whether any
    hydration was still running when close() landed."""

    def __init__(self, exc, candidate_count=12):
        self._exc = exc
        self._candidates = [{"id": f"m{i}"} for i in range(candidate_count)]
        self.closed = False
        self.in_flight = 0
        self.touched_after_close = []

    async def list_messages(self, query, max_results):
        return list(self._candidates)

    async def get_message_envelope(self, message_id):
        if self.closed:
            self.touched_after_close.append(message_id)
        self.in_flight += 1
        try:
            if message_id == "m0":
                raise self._exc
            # Yield repeatedly so slower siblings are demonstrably still
            # mid-flight when the first one raises.
            for _ in range(5):
                await asyncio.sleep(0)
            if self.closed:
                self.touched_after_close.append(message_id)
            return {"id": message_id, "from": "a@example.com"}
        finally:
            self.in_flight -= 1

    async def close(self):
        assert self.in_flight == 0, (
            f"close() ran with {self.in_flight} hydration(s) still active -- "
            "a survivor can rebuild the client and leak it"
        )
        self.closed = True


@pytest.mark.asyncio
async def test_hydration_siblings_are_drained_before_the_client_closes():
    """R8: gather() propagates the first exception without waiting for its
    siblings, so the finally closed the shared client underneath up to 49 live
    hydrations. A survivor that refreshes rebuilds an AsyncClient after close()
    nulled it and leaves the replacement unclosed."""
    from atlas_brain.services.email_provider import ScopedGmailEmailProvider

    client = _DrainTrackingClient(
        repo_mod.ScopedMailboxCredentialUnavailable("revoked")
    )
    provider = ScopedGmailEmailProvider(client)

    with pytest.raises(repo_mod.ScopedMailboxCredentialUnavailable):
        await provider.list_messages()

    assert client.closed, "the client must still be closed"
    assert client.in_flight == 0
    assert client.touched_after_close == [], (
        f"hydrations touched the client after close: {client.touched_after_close}"
    )


@pytest.mark.asyncio
async def test_refresh_is_refused_when_the_pool_cannot_spare_a_connection(
    row_lock_repo, caplog
):
    """R7/R8: max(1, 1 // 2) == 1, so a min=max=1 deployment computed a budget
    of one and a refresh still took the whole pool across Google's token call.
    Refuse instead, which degrades scoped reads to the documented fail-closed
    state rather than stalling every unrelated query."""
    _, pool = row_lock_repo
    # Derived budget only -- an explicit budget is a stated reservation.
    repo = repo_mod.ScopedMailboxCredentialRepository(pool=pool, pool_capacity=1)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(repo_mod.ScopedMailboxCredentialUnavailable) as exc:
            async with repo.locked_gmail(TENANT):
                pass

    assert "pool_headroom" in str(exc.value)
    assert pool.max_open == 0, "it must refuse WITHOUT taking a connection"
    assert any("max_pool_size=1" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_two_connection_pool_still_permits_refresh(row_lock_repo):
    """Boundary second side: exactly the minimum must work, or the guard has
    moved the supported floor instead of matching it."""
    _, pool = row_lock_repo
    repo = repo_mod.ScopedMailboxCredentialRepository(
        pool=pool, pool_capacity=repo_mod._MIN_POOL_FOR_SCOPED_REFRESH
    )

    async with repo.locked_gmail(TENANT) as locked:
        assert locked.credentials.refresh_token == "rtok"
    pool.holder_done.set()


class _FencedClient:
    """Gmail client double carrying the generation fence, so the provider's
    result-fencing can be exercised without a live Google exchange."""

    def __init__(self, source, generation):
        self._source = source
        self._generation = generation
        self.closed = False

    async def list_messages(self, query, max_results):
        return [{"id": "m1"}]

    async def get_message_envelope(self, message_id):
        return {"id": message_id, "from": "a@example.com"}

    async def assert_credentials_unchanged(self):
        from atlas_brain.autonomous.tasks.gmail_digest import GmailClient

        # Reuse the real implementation against this double's state.
        return await GmailClient.assert_credentials_unchanged(self)

    @property
    def _credential_source(self):
        return self._source

    @property
    def _credential_generation(self):
        return self._generation

    async def close(self):
        self.closed = True


class _FenceSource:
    def __init__(self, repository, business_context_id):
        self.repository = repository
        self.business_context_id = business_context_id


class _GenerationRepo:
    """Repository double returning whatever the row currently looks like."""

    def __init__(self, credentials):
        self._credentials = credentials

    async def get_active_gmail(self, business_context_id):
        return self._credentials


@pytest.mark.asyncio
async def test_revoke_during_read_is_not_delivered_from_a_cached_token():
    """R3/R8: a successful refresh caches the access token AFTER releasing the
    lease, and _refresh_token then short-circuits on it for ~an hour without
    rereading the row. A revoke committing mid-read would otherwise return the
    revoked mailbox's data with the source reported as present."""
    from atlas_brain.services.email_provider import ScopedGmailEmailProvider

    source = _FenceSource(_GenerationRepo(None), TENANT)  # row now gone
    client = _FencedClient(source, generation=3)

    with pytest.raises(repo_mod.ScopedMailboxCredentialUnavailable) as exc:
        await ScopedGmailEmailProvider(client).list_messages()

    assert "revoked_during_read" in str(exc.value)
    assert client.closed


@pytest.mark.asyncio
async def test_rebind_during_read_is_not_delivered_from_a_cached_token():
    """Same fence, other mutation: a rebind advances generation, so the cached
    token belongs to the PREVIOUS mailbox."""
    from atlas_brain.services.email_provider import ScopedGmailEmailProvider

    rebound = repo_mod.ScopedGmailCredentials(
        client_id="cid", client_secret="csec", refresh_token="rtok", generation=4
    )
    source = _FenceSource(_GenerationRepo(rebound), TENANT)
    client = _FencedClient(source, generation=3)

    with pytest.raises(repo_mod.ScopedMailboxCredentialUnavailable) as exc:
        await ScopedGmailEmailProvider(client).list_messages()

    assert "rebound_during_read" in str(exc.value)


@pytest.mark.asyncio
async def test_unchanged_credentials_deliver_normally():
    """Boundary second side: an untouched row must still deliver, or the fence
    has turned every scoped read into an omission."""
    from atlas_brain.services.email_provider import ScopedGmailEmailProvider

    same = repo_mod.ScopedGmailCredentials(
        client_id="cid", client_secret="csec", refresh_token="rtok", generation=3
    )
    source = _FenceSource(_GenerationRepo(same), TENANT)
    client = _FencedClient(source, generation=3)

    messages = await ScopedGmailEmailProvider(client).list_messages()
    assert [m["id"] for m in messages] == ["m1"]


class _EmptyInboxClient(_FencedClient):
    """Gmail returns no message IDs -- still a delivered answer."""

    async def list_messages(self, query, max_results):
        return []


@pytest.mark.asyncio
async def test_empty_result_is_fenced_too():
    """R5/R8: the empty-candidate early return bypassed the fence, so a revoke
    committing during the list request reported an empty but PRESENT inbox
    instead of an omitted source."""
    from atlas_brain.services.email_provider import ScopedGmailEmailProvider

    source = _FenceSource(_GenerationRepo(None), TENANT)  # revoked mid-list
    client = _EmptyInboxClient(source, generation=3)

    with pytest.raises(repo_mod.ScopedMailboxCredentialUnavailable):
        await ScopedGmailEmailProvider(client).list_messages()


@pytest.mark.asyncio
async def test_genuinely_empty_inbox_still_delivers_empty():
    """Boundary second side: an untouched row with no messages is an empty
    inbox, not an omission."""
    from atlas_brain.services.email_provider import ScopedGmailEmailProvider

    same = repo_mod.ScopedGmailCredentials(
        client_id="cid", client_secret="csec", refresh_token="rtok", generation=3
    )
    client = _EmptyInboxClient(_FenceSource(_GenerationRepo(same), TENANT), 3)

    assert await ScopedGmailEmailProvider(client).list_messages() == []


@pytest.mark.asyncio
async def test_short_lived_token_is_still_served_from_cache():
    """R7: a fixed 60s early-refresh margin exceeds a short issued lifetime, so
    the cached token is never accepted and each of up to 50 hydrations takes the
    row lock and exchanges again -- ~51 serialized refreshes for one read.

    Drives the REAL _refresh_token cache decision rather than restating the
    formula: with a 60-second token, the second call must be served from cache.
    """
    from atlas_brain.autonomous.tasks.gmail_digest import GmailClient

    exchanges = []

    async def _fake_exchange(*_args, **kwargs):
        exchanges.append(kwargs.get("refresh_token"))
        return {"access_token": "short-lived", "expires_in": 60}

    creds = repo_mod.ScopedGmailCredentials(
        client_id="cid", client_secret="csec", refresh_token="rtok", generation=7
    )

    class _Lease:
        credentials = creds

        async def persist_refresh_token(self, _token):
            raise AssertionError("no rotation in this scenario")

    class _Source:
        business_context_id = TENANT
        repository = None

        @asynccontextmanager
        async def locked_credentials(self):
            yield _Lease()

    client = GmailClient(credential_source=_Source(), token_exchange=_fake_exchange)
    try:
        first = await client._refresh_token()
        second = await client._refresh_token()
    finally:
        await client.close()

    assert first == "short-lived" and second == "short-lived"
    assert len(exchanges) == 1, (
        f"a 60s token must remain usable for part of its life; the cache was "
        f"bypassed and the exchange ran {len(exchanges)} times"
    )
    assert client._credential_generation == 7


@pytest.mark.asyncio
async def test_revocation_does_not_queue_behind_the_refresh_budget(row_lock_repo):
    """R1/R7/R8: revoke performs no external I/O, so it must not wait on the
    portfolio slot that reserves headroom against token-endpoint calls. An
    operator revoking must not queue behind unrelated contexts' refreshes.

    The budget is stated outright so the test saturates the SAME semaphore the
    repository will reach for -- saturating a differently-keyed one would make
    this pass no matter how revoke behaves.
    """
    _, pool = row_lock_repo
    repo = repo_mod.ScopedMailboxCredentialRepository(pool=pool, refresh_budget=1)

    slot = repo._slot()
    await slot.acquire()  # the one permit is now held, as a refresh would
    try:
        # A different context: only the portfolio slot could couple them.
        await asyncio.wait_for(
            repo.revoke_gmail("some-other-context"), timeout=2
        )
    finally:
        slot.release()
