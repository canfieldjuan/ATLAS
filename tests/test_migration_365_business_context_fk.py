"""Real-PostgreSQL apply check for migration 365 (ATLAS #2318).

The mocked ``create_contact`` unit tests cannot verify the DB behaviors of the
migration -- seed-before-FK ordering, existing-row preservation, FK enforcement,
voice-config neutralization, or rerun idempotence. This applies 365 (over its
prerequisites 040 + 035) to a disposable database and asserts them.

Skipped unless ``ATLAS_MIGRATION_TEST_DATABASE_URL`` points at a disposable DB
(the CI migration-tests service database sets it). Locally, point it at a scratch
DB -- never prod.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = ROOT / "atlas_brain" / "storage" / "migrations"
PREREQS = ("040_business_contexts.sql", "035_contacts.sql")
MIG_365 = "365_contacts_business_context_registry_fk.sql"


def _database_url() -> str | None:
    return os.environ.get("ATLAS_MIGRATION_TEST_DATABASE_URL")


async def _apply(conn, *names: str) -> None:
    for name in names:
        await conn.execute((MIGRATIONS_DIR / name).read_text())


async def _reset(conn) -> None:
    # Shared CI service DB: drop what these tests create so each starts clean.
    await conn.execute(
        "DROP TABLE IF EXISTS contact_interactions, contacts, business_contexts, "
        "appointments CASCADE"
    )


async def _prereqs(conn) -> None:
    """Create business_contexts + contacts (no FK yet). 035 only ALTERs an existing
    `appointments` table (adds contact_id + an index), so a minimal stub satisfies
    that dependency without pulling the whole migration chain."""
    await conn.execute("CREATE TABLE appointments (id UUID PRIMARY KEY)")
    await _apply(conn, *PREREQS)


async def _fk_on_contacts(conn) -> bool:
    return await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM pg_constraint "
        "WHERE conname = 'contacts_business_context_id_fkey' "
        "AND conrelid = 'contacts'::regclass)"
    )


async def _insert_contact(conn, tenant):
    await conn.execute(
        "INSERT INTO contacts (id, full_name, business_context_id) VALUES ($1, $2, $3)",
        uuid.uuid4(),
        "Test Contact",
        tenant,
    )


@pytest.mark.asyncio
async def test_365_fresh_db_seeds_neutralized_registry_and_adds_scoped_fk() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    url = _database_url()
    if not url:
        pytest.skip("ATLAS_MIGRATION_TEST_DATABASE_URL not set")

    conn = await asyncpg.connect(url)
    try:
        await _reset(conn)
        await _prereqs(conn)
        await _apply(conn, MIG_365)

        ids = {r["id"] for r in await conn.fetch("SELECT id FROM business_contexts")}
        # personal is seeded too: it is the atlas_comms ContextRouter fallback that
        # inbound SMS resolves to, and the FK governs that writer -- without the seed
        # the first unmatched SMS would FK-violate and drop the contact.
        assert {"effingham_maids", "churnsignals", "personal"} <= ids

        # Voice-product config is NEUTRALIZED, not left to migration 040's active
        # defaults (Atlas voice, 9-5 hours, scheduling/SMS/messages ENABLED).
        row = await conn.fetchrow(
            "SELECT enabled, scheduling_enabled, sms_enabled, sms_auto_reply, "
            "take_messages, voice_name, greeting, monday_open, timezone "
            "FROM business_contexts WHERE id = 'effingham_maids'"
        )
        # enabled = FALSE keeps the registry rows OUT of the voice loader
        # (list_enabled filters on enabled = TRUE), so their NULL voice fields never
        # reach the non-optional Pydantic model that would ValidationError the load.
        assert row["enabled"] is False
        assert row["scheduling_enabled"] is False
        assert row["sms_enabled"] is False
        assert row["sms_auto_reply"] is False
        assert row["take_messages"] is False
        assert row["voice_name"] is None
        assert row["greeting"] is None
        assert row["monday_open"] is None
        assert row["timezone"] is None

        # The voice loader's query (enabled = TRUE) returns NONE of the seed rows.
        loader_visible = await conn.fetch(
            "SELECT id FROM business_contexts WHERE enabled = TRUE"
        )
        assert loader_visible == []

        assert await _fk_on_contacts(conn) is True
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_365_prepopulated_validates_enforces_and_is_idempotent() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    url = _database_url()
    if not url:
        pytest.skip("ATLAS_MIGRATION_TEST_DATABASE_URL not set")

    conn = await asyncpg.connect(url)
    try:
        await _reset(conn)
        await _prereqs(conn)  # tables exist; no FK yet

        # Prepopulate contacts BEFORE 365 -- seed-before-FK must let these validate.
        await _insert_contact(conn, "effingham_maids")
        # A tenant present ONLY on contacts: the dynamic backstop must seed it so the
        # FK can validate rather than fail the migration.
        await _insert_contact(conn, "oddtenant")

        await _apply(conn, MIG_365)  # must NOT raise

        # Existing rows preserved and the backstop tenant is now a registry row.
        assert await conn.fetchval(
            "SELECT count(*) FROM contacts WHERE business_context_id = 'effingham_maids'"
        ) == 1
        assert await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM business_contexts WHERE id = 'oddtenant')"
        ) is True
        assert await _fk_on_contacts(conn) is True

        # FK ENFORCEMENT: an unknown tenant is rejected at the INSERT.
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await _insert_contact(conn, "nonexistent_tenant")

        # NULL tenant is allowed by the FK (the D1 guard forbids NULL on the agent path).
        await _insert_contact(conn, None)

        # IDEMPOTENT: reapplying does not raise or duplicate the constraint.
        await _apply(conn, MIG_365)
        assert await _fk_on_contacts(conn) is True
        assert await conn.fetchval(
            "SELECT count(*) FROM pg_constraint "
            "WHERE conname = 'contacts_business_context_id_fkey' "
            "AND conrelid = 'contacts'::regclass"
        ) == 1
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_real_admission_check_gates_on_fk_not_table_occupancy(monkeypatch) -> None:
    """Drive the REAL BusinessContextRepository.admission_check against the
    disposable DB (not a hand-copied query): it must report enforced iff migration
    365 has run (the FK exists), NOT merely because business_contexts holds an
    unrelated voice row. Reverting the method to table-occupancy gating breaks the
    first assertion here."""
    asyncpg = pytest.importorskip("asyncpg")
    url = _database_url()
    if not url:
        pytest.skip("ATLAS_MIGRATION_TEST_DATABASE_URL not set")

    from atlas_brain.storage import database as db_module
    from atlas_brain.storage.repositories.business_context import (
        BusinessContextRepository,
    )

    setup = await asyncpg.connect(url)
    try:
        await _reset(setup)
        await _prereqs(setup)  # tables exist; 365 NOT applied -> no FK
        # A stray, unrelated voice row makes the table non-empty before 365 runs.
        await setup.execute(
            "INSERT INTO business_contexts (id, name, phone_numbers) "
            "VALUES ('some_voice_context', 'Voice', '{}')"
        )
    finally:
        await setup.close()

    # Point the process-global pool at the disposable DB so the REAL repository
    # method executes here rather than being monkeypatched.
    real_pool = db_module.DatabasePool()
    real_pool._pool = await asyncpg.create_pool(dsn=url)
    real_pool._initialized = True
    monkeypatch.setattr(db_module, "_db_pool", real_pool)
    repo = BusinessContextRepository()
    try:
        # Table occupied, but 365 has NOT run -> NOT enforced; churnsignals unknown.
        enforced, known = await repo.admission_check("churnsignals")
        assert enforced is False
        assert known is False

        # Apply 365, then the real method must flip enforced True and know the tenant.
        apply_conn = await asyncpg.connect(url)
        try:
            await apply_conn.execute((MIGRATIONS_DIR / MIG_365).read_text())
        finally:
            await apply_conn.close()

        enforced, known = await repo.admission_check("churnsignals")
        assert enforced is True
        assert known is True

        # Enforced + unknown tenant -> the guard would reject.
        enforced_u, known_u = await repo.admission_check("nonexistent_tenant")
        assert enforced_u is True
        assert known_u is False

        # The voice loader (real list_enabled) must EXCLUDE the enabled=FALSE
        # registry seeds -- otherwise their NULL voice fields abort the whole
        # comms-startup registration -- while still returning a genuine enabled
        # voice context (some_voice_context), proving the loader is not broken.
        loaded = {r["id"] for r in await repo.list_enabled()}
        assert "effingham_maids" not in loaded
        assert "churnsignals" not in loaded
        assert "personal" not in loaded
        assert "some_voice_context" in loaded
    finally:
        await real_pool._pool.close()


@pytest.mark.asyncio
async def test_migration_365_serializes_a_concurrent_writer() -> None:
    """Drive the REAL migration 365 concurrently with an in-flight contact writer --
    not a hand-copied lock. The DO block takes `LOCK TABLE contacts IN SHARE MODE`
    FIRST; SHARE conflicts with the ROW EXCLUSIVE lock an uncommitted INSERT holds,
    so the migration must WAIT for the writer instead of snapshotting tenants around
    it. This exercises the protocol end to end:

    * with a writer holding an uncommitted NEW-tenant insert, the real migration
      blocks -- and pg_locks shows it waiting for a **ShareLock** on `contacts`, the
      mode its first statement requests. A migration that moved the lock below the
      seed or out of the DO block would not be waiting on a ShareLock here (it would
      seed without blocking, then block/fail only at the ALTER), so the mode + timing
      assertion fails for a faked protocol.
    * once the writer commits, the migration completes: the dynamic backstop seeds
      the now-committed tenant, the FK lands with no orphan, and an unseeded tenant
      is subsequently rejected."""
    asyncpg = pytest.importorskip("asyncpg")
    url = _database_url()
    if not url:
        pytest.skip("ATLAS_MIGRATION_TEST_DATABASE_URL not set")

    # Cheap structural complement (not the proof): the lock lives in the migration.
    assert "LOCK TABLE contacts IN SHARE MODE" in (MIGRATIONS_DIR / MIG_365).read_text()

    setup = await asyncpg.connect(url)
    try:
        await _reset(setup)
        await _prereqs(setup)  # tables exist; 365 NOT applied yet -> no FK
    finally:
        await setup.close()

    writer = await asyncpg.connect(url)
    migrator = await asyncpg.connect(url)
    observer = await asyncpg.connect(url)
    mig_task = None
    try:
        migrator_pid = await migrator.fetchval("SELECT pg_backend_pid()")

        # Writer holds an UNCOMMITTED insert of a brand-new tenant -> it owns a
        # ROW EXCLUSIVE lock on contacts, which conflicts with the migration's SHARE.
        wtx = writer.transaction()
        await wtx.start()
        await writer.execute(
            "INSERT INTO contacts (id, full_name, business_context_id) "
            "VALUES ($1, 'Racer', 'racer_tenant')",
            uuid.uuid4(),
        )

        # Launch the ACTUAL migration file concurrently; it must block on its lock.
        migration_sql = (MIGRATIONS_DIR / MIG_365).read_text()
        mig_task = asyncio.create_task(migrator.execute(migration_sql))

        # Poll pg_locks until the migrator is BLOCKED (granted=false) on contacts.
        blocked = None
        for _ in range(200):  # up to ~10s; the block is near-instant in practice
            blocked = await observer.fetchrow(
                "SELECT mode, granted FROM pg_locks "
                "WHERE pid = $1 AND relation = 'contacts'::regclass "
                "AND locktype = 'relation' AND NOT granted",
                migrator_pid,
            )
            if blocked is not None:
                break
            if mig_task.done():
                # Migration finished WITHOUT ever blocking on the writer -> it did not
                # serialize against the in-flight write. Surface the real error if any.
                await mig_task
                raise AssertionError(
                    "migration 365 completed without blocking on the in-flight writer "
                    "-- its lock does not span the seed/FK against concurrent writes"
                )
            await asyncio.sleep(0.05)

        assert blocked is not None, (
            "migration 365 never blocked on a contacts lock while a writer was in flight"
        )
        # The first conflicting lock it waits for is the SHARE lock of its DO block,
        # NOT the AccessExclusiveLock a lock-less migration would only reach at ALTER.
        assert blocked["mode"] == "ShareLock", (
            f"expected the migration to wait on ShareLock (LOCK TABLE ... IN SHARE "
            f"MODE) before seeding; got {blocked['mode']} -- the lock is misplaced"
        )

        # Release the writer: its new tenant commits; the migration now proceeds.
        await wtx.commit()
        await asyncio.wait_for(mig_task, timeout=15)  # completes without error

        # Final state: FK present; the raced tenant was captured by the backstop
        # (no orphan, no FK violation); an unseeded tenant is now FK-rejected.
        assert await _fk_on_contacts(migrator) is True
        assert await migrator.fetchval(
            "SELECT EXISTS (SELECT 1 FROM business_contexts WHERE id = 'racer_tenant')"
        ) is True
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await _insert_contact(migrator, "never_seeded_tenant")
    finally:
        if mig_task is not None and not mig_task.done():
            mig_task.cancel()
        await writer.close()
        await migrator.close()
        await observer.close()
