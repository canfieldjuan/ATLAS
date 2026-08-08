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
        assert {"effingham_maids", "churnsignals"} <= ids

        # Voice-product config is NEUTRALIZED, not left to migration 040's active
        # defaults (Atlas voice, 9-5 hours, scheduling/SMS/messages ENABLED).
        row = await conn.fetchrow(
            "SELECT scheduling_enabled, sms_enabled, sms_auto_reply, take_messages, "
            "voice_name, greeting, monday_open, timezone "
            "FROM business_contexts WHERE id = 'effingham_maids'"
        )
        assert row["scheduling_enabled"] is False
        assert row["sms_enabled"] is False
        assert row["sms_auto_reply"] is False
        assert row["take_messages"] is False
        assert row["voice_name"] is None
        assert row["greeting"] is None
        assert row["monday_open"] is None
        assert row["timezone"] is None

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
