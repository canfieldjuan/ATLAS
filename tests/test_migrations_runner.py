import pytest
from pathlib import Path


class FakeMigrationPool:
    def __init__(self, records=None):
        self.records = list(records or [])
        self.inserted = []

    async def fetchval(self, query, *args):
        normalized = " ".join(query.split())
        if "WHERE name = $1" in normalized:
            name = args[0]
            for version, record_name in self.records:
                if record_name == name:
                    return version
            return None
        if "WHERE version = $1" in normalized:
            version = args[0]
            for record_version, name in self.records:
                if record_version == version:
                    return name
            return None
        if "MIN(version)" in normalized:
            if not self.records:
                return -1
            min_version = min(version for version, _ in self.records)
            return min_version - 1 if min_version < 0 else -1
        raise AssertionError(f"Unexpected fetchval query: {query}")

    async def execute(self, query, *args):
        normalized = " ".join(query.split())
        if normalized.startswith("INSERT INTO schema_migrations"):
            record = (args[0], args[1])
            self.records.append(record)
            self.inserted.append(record)
            return
        raise AssertionError(f"Unexpected execute query: {query}")


@pytest.mark.asyncio
async def test_record_migration_uses_prefix_version_when_available():
    from atlas_brain.storage.migrations import _record_migration

    pool = FakeMigrationPool(records=[(1, "001_initial_schema")])

    await _record_migration(pool, "247_b2b_vendor_witness_packets.sql")

    assert pool.inserted == [(247, "247_b2b_vendor_witness_packets")]


@pytest.mark.asyncio
async def test_record_migration_uses_negative_version_on_prefix_collision():
    from atlas_brain.storage.migrations import _record_migration

    pool = FakeMigrationPool(records=[
        (76, "076_saas_accounts"),
        (230, "230_scrape_target_checkpoints"),
    ])

    await _record_migration(pool, "076_consumer_analytics_views.sql")
    await _record_migration(pool, "230_b2b_reasoning_synthesis.sql")

    assert pool.inserted == [
        (-1, "076_consumer_analytics_views"),
        (-2, "230_b2b_reasoning_synthesis"),
    ]


def test_find_duplicate_migration_prefixes_detects_repo_collisions():
    from atlas_brain.storage.migrations import _find_duplicate_migration_prefixes

    duplicates = _find_duplicate_migration_prefixes([
        Path("270_b2b_watchlist_views.sql"),
        Path("271_b2b_watchlist_view_alert_thresholds.sql"),
        Path("272_b2b_opportunity_dispositions.sql"),
        Path("272_b2b_watchlist_alert_events.sql"),
    ])

    assert duplicates == {
        272: [
            "272_b2b_opportunity_dispositions.sql",
            "272_b2b_watchlist_alert_events.sql",
        ]
    }


def test_repo_migration_prefix_collisions_are_only_historical_exceptions():
    from atlas_brain.storage.migrations import MIGRATIONS_DIR, _find_duplicate_migration_prefixes

    duplicates = _find_duplicate_migration_prefixes(sorted(MIGRATIONS_DIR.glob("*.sql")))

    # Accepted historical prefix collisions: each pair already shipped on main
    # with the same numeric prefix (parallel-session migrations that landed
    # together). The allowlist documents them so this test still catches a NEW
    # collision; 281/282/283/298 were added after the allowlist was last updated.
    assert duplicates == {
        76: [
            "076_consumer_analytics_views.sql",
            "076_saas_accounts.sql",
        ],
        230: [
            "230_b2b_reasoning_synthesis.sql",
            "230_scrape_target_checkpoints.sql",
        ],
        281: [
            "281_b2b_report_subscription_filter_payload.sql",
            "281_b2b_watchlist_alert_reopen_count.sql",
        ],
        282: [
            "282_b2b_enrichment_stage_runs.sql",
            "282_b2b_materialization_run_id.sql",
        ],
        283: [
            "283_b2b_enrichment_stage_runs_work_fingerprint.sql",
            "283_b2b_report_subscription_delivery_blocked_freshness.sql",
        ],
        298: [
            "298_b2b_review_vendor_mentions_drop_duplicate_rows.sql",
            "298_b2b_watchlist_preview_alert_policy.sql",
        ],
    }


def test_contact_lead_pipeline_migration_is_additive_and_indexed():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "346_contact_lead_pipeline.sql"
    ).read_text()

    assert "ADD COLUMN IF NOT EXISTS lead_stage VARCHAR(64)" in migration
    assert "ADD COLUMN IF NOT EXISTS lead_owner VARCHAR(128)" in migration
    assert "ADD COLUMN IF NOT EXISTS next_follow_up_at TIMESTAMPTZ" in migration
    assert "idx_contacts_lead_follow_up" in migration
    assert "WHERE contact_type = 'lead'" in migration
    assert "DROP " not in migration.upper()


def test_customer_service_ticket_migration_is_additive_tenant_scoped_and_indexed():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "347_customer_service_tickets.sql"
    ).read_text()

    assert "CREATE TABLE IF NOT EXISTS customer_service_tickets" in migration
    assert "contact_id UUID NOT NULL REFERENCES contacts(id)" in migration
    assert "business_context_id VARCHAR(64) NOT NULL" in migration
    assert "CHECK (btrim(business_context_id) <> '')" in migration
    assert "CHECK (btrim(summary) <> '')" in migration
    assert "CHECK (status IN ('open', 'closed'))" in migration
    assert "NULLIF(btrim(resolution), '') IS NOT NULL" in migration
    assert "idx_customer_service_tickets_open_queue" in migration
    assert "WHERE status = 'open'" in migration
    assert "DROP " not in migration.upper()


def test_appointment_operating_fields_migration_is_additive_and_constrained():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "348_appointment_operating_fields.sql"
    ).read_text()

    assert "ADD COLUMN IF NOT EXISTS recurrence_interval SMALLINT" in migration
    assert "ADD COLUMN IF NOT EXISTS recurrence_unit VARCHAR(16)" in migration
    assert "ADD COLUMN IF NOT EXISTS assigned_cleaner VARCHAR(128)" in migration
    assert "ADD COLUMN IF NOT EXISTS per_visit_price NUMERIC(12,2)" in migration
    assert "chk_appointments_recurrence_pair" in migration
    assert "recurrence_interval BETWEEN 1 AND 365" in migration
    assert "recurrence_unit IN ('day', 'week', 'month')" in migration
    assert "chk_appointments_assigned_cleaner" in migration
    assert "chk_appointments_per_visit_price" in migration
    assert "DROP " not in migration.upper()


def test_sent_email_tenant_migration_is_additive_replay_safe_and_unclassified():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "349_sent_emails_business_context.sql"
    ).read_text()
    upper = migration.upper()

    assert "ADD COLUMN IF NOT EXISTS business_context_id VARCHAR(64)" in migration
    assert "chk_sent_emails_business_context_nonblank" in migration
    assert "business_context_id IS NULL" in migration
    assert "btrim(business_context_id) <> ''" in migration
    assert "idx_sent_emails_context_sent_at" in migration
    assert "WHERE business_context_id IS NOT NULL" in migration
    assert "UPDATE sent_emails" not in migration
    assert "SET DEFAULT" not in upper
    assert "SET NOT NULL" not in upper
    assert "DROP " not in upper


def test_scoped_mailbox_credential_migration_is_additive_and_constrained():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "350_scoped_mailbox_credentials.sql"
    ).read_text()
    upper = migration.upper()
    normalized = " ".join(migration.split())

    assert "CREATE TABLE IF NOT EXISTS scoped_mailbox_credentials" in migration
    assert "PRIMARY KEY (business_context_id, provider)" in migration
    assert "encrypted_credentials BYTEA NOT NULL" in normalized
    assert "encryption_kid" in migration
    assert "generation BIGINT NOT NULL DEFAULT 1" in normalized
    assert "CHECK (btrim(business_context_id) <> '')" in migration
    assert "CHECK (provider = 'gmail')" in migration
    assert "CHECK (generation > 0)" in migration
    assert "WHERE revoked_at IS NULL" in migration
    assert "DROP " not in upper


class _SerializingPool(FakeMigrationPool):
    """transaction() + pg_advisory_xact_lock with real blocking semantics.

    ``honor_lock=False`` makes the advisory lock a no-op so the paired probe
    below shows the single-application result comes from the lock statement,
    not from an accident of the fixture.
    """

    def __init__(self, *, honor_lock=True):
        super().__init__()
        self.honor_lock = honor_lock
        self.applied_sql = []
        self.acquired = 0
        self.max_acquired = 0
        self._gate = None

    def _lock(self):
        import asyncio

        if self._gate is None:
            self._gate = asyncio.Lock()
        return self._gate

    async def fetch(self, query, *args):
        assert "FROM schema_migrations" in query
        # Snapshot BEFORE yielding, like a real query that reads at statement
        # start and then awaits I/O. Snapshotting after the yield would let a
        # concurrent runner's write leak into this result and mask the race.
        import asyncio

        snapshot = [{"name": name} for _version, name in self.records]
        await asyncio.sleep(0)
        return snapshot

    async def execute(self, query, *args):
        normalized = " ".join(query.split())
        if normalized.startswith("CREATE TABLE IF NOT EXISTS schema_migrations"):
            return
        if normalized.startswith("INSERT INTO schema_migrations"):
            return await super().execute(query, *args)
        self.applied_sql.append(normalized)

    async def acquire(self):
        self.acquired += 1
        self.max_acquired = max(self.max_acquired, self.acquired)
        return _Conn(self)

    async def release(self, conn):
        self.acquired -= 1


class _Conn:
    """Single acquired connection: the lock, bookkeeping and migration SQL all
    run here, which is what proves a one-connection pool cannot deadlock."""

    def __init__(self, pool):
        self.pool = pool
        self.held = False

    async def execute(self, query, *args):
        if "pg_advisory_lock" in query:
            if self.pool.honor_lock:
                await self.pool._lock().acquire()
                self.held = True
            return
        if "pg_advisory_unlock" in query:
            if self.pool.honor_lock and self.held:
                self.pool._lock().release()
                self.held = False
            return
        return await self.pool.execute(query, *args)

    async def fetch(self, query, *args):
        return await self.pool.fetch(query, *args)

    async def fetchval(self, query, *args):
        return await self.pool.fetchval(query, *args)


@pytest.mark.asyncio
async def test_concurrent_runners_apply_each_migration_once(tmp_path):
    """Two simultaneous startups (main app + standalone MCP, or two replicas)
    must not both apply the same pending migration."""
    import asyncio

    from atlas_brain.storage.migrations import run_migrations

    (tmp_path / "900_probe.sql").write_text("SELECT 900")
    pool = _SerializingPool(honor_lock=True)

    await asyncio.gather(
        run_migrations(pool, migrations_dir=tmp_path),
        run_migrations(pool, migrations_dir=tmp_path),
    )
    assert pool.applied_sql == ["SELECT 900"], (
        "the second entrant must re-snapshot under the lock and find nothing "
        "pending"
    )
    assert pool.inserted == [(900, "900_probe")]


@pytest.mark.asyncio
async def test_without_the_advisory_lock_both_runners_apply(tmp_path):
    """3i probe: with the lock a no-op the same fixture double-applies,
    proving the test above measures the lock."""
    import asyncio

    from atlas_brain.storage.migrations import run_migrations

    (tmp_path / "900_probe.sql").write_text("SELECT 900")
    pool = _SerializingPool(honor_lock=False)

    await asyncio.gather(
        run_migrations(pool, migrations_dir=tmp_path),
        run_migrations(pool, migrations_dir=tmp_path),
    )
    assert len(pool.applied_sql) == 2, (
        "no-op lock must reproduce the double-apply race the real lock prevents"
    )


class _SingleConnectionPool:
    """A pool configured min=max=1: the second concurrent acquire() waits
    forever, which is precisely what a transaction that then reaches back to
    the pool would do."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.applied_sql = []
        self.in_use = False
        self.autocommit_sql = []

    async def acquire(self):
        if self.in_use:
            # A real asyncpg pool would block here until timeout; failing loudly
            # keeps the deadlock from hanging the suite.
            raise AssertionError(
                "second connection requested while the migration run holds the "
                "only one -- this is the min=max=1 deadlock"
            )
        self.in_use = True
        return _SingleConn(self)

    async def release(self, conn):
        self.in_use = False

    # Any helper still calling the POOL mid-run re-enters acquire() and trips
    # the assertion above.
    async def execute(self, query, *args):
        conn = await self.acquire()
        try:
            return await conn.execute(query, *args)
        finally:
            await self.release(conn)

    async def fetch(self, query, *args):
        conn = await self.acquire()
        try:
            return await conn.fetch(query, *args)
        finally:
            await self.release(conn)

    async def fetchval(self, query, *args):
        conn = await self.acquire()
        try:
            return await conn.fetchval(query, *args)
        finally:
            await self.release(conn)


class _SingleConn:
    def __init__(self, pool):
        self.pool = pool

    async def execute(self, query, *args):
        if "pg_advisory" in query:
            return
        self.pool.applied_sql.append(query)

    async def fetch(self, query, *args):
        return []

    async def fetchval(self, query, *args):
        return None


@pytest.mark.asyncio
async def test_migrations_run_on_a_single_connection_pool(tmp_path):
    """F3: the whole run -- advisory lock, bookkeeping and migration SQL --
    must occupy exactly ONE connection, or a deployment with
    min_pool_size == max_pool_size == 1 deadlocks at startup."""
    from atlas_brain.storage.migrations import run_migrations

    (tmp_path / "001_first.sql").write_text("CREATE TABLE a (id int);")
    (tmp_path / "002_second.sql").write_text("CREATE TABLE b (id int);")

    pool = _SingleConnectionPool(tmp_path)
    await run_migrations(pool, migrations_dir=tmp_path)

    assert any("CREATE TABLE a" in sql for sql in pool.applied_sql)
    assert any("CREATE TABLE b" in sql for sql in pool.applied_sql)
    assert pool.in_use is False, "the connection must be released"


@pytest.mark.asyncio
async def test_migration_sql_does_not_run_inside_a_transaction(tmp_path):
    """F3 second side: five packaged migrations use CREATE INDEX
    CONCURRENTLY, which Postgres refuses inside a transaction block. The run
    must therefore hold a SESSION-level lock, not a transaction-scoped one."""
    from atlas_brain.storage.migrations import run_migrations

    (tmp_path / "001_concurrent.sql").write_text(
        "CREATE INDEX CONCURRENTLY idx_x ON t (c);"
    )
    pool = _SingleConnectionPool(tmp_path)
    started = []

    class _TxnBanningConn(_SingleConn):
        async def execute(self, query, *args):
            if query.strip().upper().startswith(("BEGIN", "START TRANSACTION")):
                started.append(query)
            if "pg_advisory_xact_lock" in query:
                raise AssertionError(
                    "a transaction-scoped advisory lock implies an open "
                    "transaction; CREATE INDEX CONCURRENTLY cannot run there"
                )
            return await super().execute(query, *args)

    pool.acquire = lambda: _acquire_banning(pool, _TxnBanningConn)
    await run_migrations(pool, migrations_dir=tmp_path)

    assert not started, f"migration run opened a transaction: {started}"
    assert any("CONCURRENTLY" in sql for sql in pool.applied_sql)


async def _acquire_banning(pool, conn_cls):
    if pool.in_use:
        raise AssertionError("second connection requested during the run")
    pool.in_use = True
    return conn_cls(pool)
