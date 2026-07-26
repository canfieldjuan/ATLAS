"""Regression tests for the four #2200 review findings.

Each test proves the failure direction per AGENTS.md 3i: the fixture reproduces
the reviewed defect's exact path, and defeating the fix (where practical) is
shown to flip the result in the paired probe test.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import pytest

import atlas_brain.services.customer_context as context_mod
import atlas_brain.services.crm_provider as crm_provider_mod
import atlas_brain.services.email_provider as email_provider_mod
import atlas_brain.storage.repositories.scoped_mailbox_credential as repo_mod

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

    async def _migrate(p):
        ran.append(p)

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
    assert ran == [pool], "lifespan must apply migrations to the live pool"
    assert closed == [True]


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
async def test_ungated_waiters_hold_connections_while_blocked(row_lock_repo):
    """3i probe, no internals patched: workers on DISTINCT contexts take
    distinct gates, so the gate cannot serialize them -- every one enters a
    transaction and blocks on the contended row while holding its connection.
    This is the exact pre-fix profile for one context, proving the bounded
    test above is measuring the gate rather than the fixture."""
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
    assert pool.max_open == 10, (
        "ungated waiters must each hold a connection blocked on the row "
        "lock -- the defect the per-context gate exists to prevent"
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
