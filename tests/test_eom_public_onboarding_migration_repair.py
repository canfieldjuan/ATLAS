"""Real-PostgreSQL regression proof for the public-onboarding schema repair."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.storage.migrations import run_migrations  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
MIGRATION_STEM = "384_eom_public_onboarding_tokens_schema_repair"

_EXPECTED_REPAIRED_COLUMNS = {
    "signing_key_fingerprint": ("character varying", 64, "NO"),
    "prefill_full_name": ("character varying", 256, "NO"),
    "prefill_email": ("character varying", 256, "YES"),
    "prefill_phone": ("character varying", 32, "YES"),
    "prefill_address": ("text", None, "YES"),
    "prefill_city": ("character varying", 128, "YES"),
    "prefill_state": ("character varying", 64, "YES"),
    "prefill_zip": ("character varying", 16, "YES"),
    "prefill_customer_type": ("character varying", 32, "NO"),
}


def _database_url_or_skip() -> str:
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")
    return database_url


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class _MigrationPool:
    def __init__(self, conn) -> None:
        self._conn = conn

    async def acquire(self):
        return self._conn

    async def release(self, released) -> None:
        assert released is self._conn


async def _prepare_known_legacy_schema(conn, schema: str) -> None:
    """Create the exact pre-383 token shape observed in production.

    The negative ledger record models the old prefix collision. The normal 383
    ledger record models why the production runner will not revisit its
    `CREATE TABLE IF NOT EXISTS` body.
    """
    schema_ident = _quote_ident(schema)
    await conn.execute(f"CREATE SCHEMA {schema_ident}")
    await conn.execute(f"SET search_path TO {schema_ident}, public")
    await conn.execute("""
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        );
        INSERT INTO schema_migrations (version, name) VALUES
            (-11, '382_eom_public_onboarding_tokens'),
            (383, '383_eom_public_onboarding_tokens');

        CREATE TABLE contacts (id UUID PRIMARY KEY);
        CREATE TABLE eom_onboarding_email_drafts (id UUID PRIMARY KEY);

        CREATE TABLE eom_public_onboarding_tokens (
            id UUID NOT NULL,
            draft_id UUID NOT NULL REFERENCES eom_onboarding_email_drafts(id)
                ON DELETE RESTRICT,
            contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT,
            approval_key VARCHAR(128) NOT NULL,
            status VARCHAR(16) NOT NULL DEFAULT 'issued',
            approved_by_employee_id BIGINT NOT NULL
                CHECK (approved_by_employee_id > 0),
            approved_by_name VARCHAR(128) NOT NULL,
            issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            redeemed_at TIMESTAMPTZ,
            revoked_at TIMESTAMPTZ,
            handoff_id UUID,
            CONSTRAINT pk_eom_public_onboarding_tokens PRIMARY KEY (id),
            CONSTRAINT uq_eom_public_onboarding_tokens_draft UNIQUE (draft_id),
            CONSTRAINT uq_eom_public_onboarding_tokens_approval UNIQUE (approval_key),
            CONSTRAINT uq_eom_public_onboarding_tokens_handoff UNIQUE (handoff_id),
            CONSTRAINT ck_eom_public_onboarding_tokens_status
                CHECK (status IN ('issued', 'redeemed', 'revoked')),
            CONSTRAINT ck_eom_public_onboarding_tokens_terminal_state CHECK (
                (status = 'issued'
                    AND redeemed_at IS NULL
                    AND revoked_at IS NULL
                    AND handoff_id IS NULL)
                OR (status = 'redeemed'
                    AND redeemed_at IS NOT NULL
                    AND revoked_at IS NULL
                    AND handoff_id IS NOT NULL)
                OR (status = 'revoked'
                    AND redeemed_at IS NULL
                    AND revoked_at IS NOT NULL
                    AND handoff_id IS NULL)
            )
        );
        CREATE UNIQUE INDEX uq_eom_public_onboarding_tokens_issued_contact
            ON eom_public_onboarding_tokens (contact_id)
            WHERE status = 'issued';
        CREATE INDEX idx_eom_public_onboarding_tokens_status
            ON eom_public_onboarding_tokens (status, issued_at DESC);
        """)


async def _prepare_complete_schema(conn, schema: str) -> None:
    """Create the normal, complete 383 relation through the migration runner."""
    schema_ident = _quote_ident(schema)
    await conn.execute(f"CREATE SCHEMA {schema_ident}")
    await conn.execute(f"SET search_path TO {schema_ident}, public")
    await conn.execute("""
        CREATE TABLE contacts (id UUID PRIMARY KEY);
        CREATE TABLE eom_onboarding_email_drafts (id UUID PRIMARY KEY);
        """)
    await run_migrations(
        _MigrationPool(conn),
        migrations_dir=MIGRATIONS,
        only={"383_eom_public_onboarding_tokens"},
    )


async def _run_schema_repair(conn) -> None:
    await run_migrations(
        _MigrationPool(conn),
        migrations_dir=MIGRATIONS,
        only={MIGRATION_STEM},
    )


async def _legacy_columns(conn, schema: str) -> set[str]:
    rows = await conn.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = $1
          AND table_name = 'eom_public_onboarding_tokens'
        ORDER BY column_name
        """,
        schema,
    )
    return {row["column_name"] for row in rows}


@pytest.mark.asyncio
async def test_schema_repair_adds_383_immutable_projection_to_empty_legacy_table():
    database_url = _database_url_or_skip()
    schema = f"eom_public_repair_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_known_legacy_schema(conn, schema)
        await _run_schema_repair(conn)

        assert (
            await conn.fetchval(
                """
            SELECT version
            FROM schema_migrations
            WHERE name = $1
            """,
                MIGRATION_STEM,
            )
            == 384
        )

        rows = await conn.fetch(
            """
            SELECT column_name, data_type, character_maximum_length, is_nullable
            FROM information_schema.columns
            WHERE table_schema = $1
              AND table_name = 'eom_public_onboarding_tokens'
              AND column_name = ANY($2::text[])
            """,
            schema,
            list(_EXPECTED_REPAIRED_COLUMNS),
        )
        actual_columns = {
            row["column_name"]: (
                row["data_type"],
                row["character_maximum_length"],
                row["is_nullable"],
            )
            for row in rows
        }
        assert actual_columns == _EXPECTED_REPAIRED_COLUMNS

        check_definitions = await conn.fetch(
            """
            SELECT pg_get_constraintdef(oid) AS definition
            FROM pg_constraint
            WHERE conrelid = 'eom_public_onboarding_tokens'::regclass
              AND contype = 'c'
            """
        )
        assert any(
            "signing_key_fingerprint" in row["definition"]
            and "^[0-9a-f]{64}$" in row["definition"]
            for row in check_definitions
        )

        contact_id = uuid.uuid4()
        draft_id = uuid.uuid4()
        await conn.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await conn.execute(
            "INSERT INTO eom_onboarding_email_drafts (id) VALUES ($1)", draft_id
        )
        valid_values = (
            uuid.uuid4(),
            draft_id,
            contact_id,
            "a" * 64,
            "Immutable Person",
            "residential",
            "approval-valid",
            1,
            "Office User",
        )
        await conn.execute(
            """
            INSERT INTO eom_public_onboarding_tokens (
                id, draft_id, contact_id, signing_key_fingerprint,
                prefill_full_name, prefill_customer_type, approval_key,
                approved_by_employee_id, approved_by_name
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            *valid_values,
        )

        invalid_contact_id = uuid.uuid4()
        invalid_draft_id = uuid.uuid4()
        await conn.execute("INSERT INTO contacts (id) VALUES ($1)", invalid_contact_id)
        await conn.execute(
            "INSERT INTO eom_onboarding_email_drafts (id) VALUES ($1)",
            invalid_draft_id,
        )
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO eom_public_onboarding_tokens (
                    id, draft_id, contact_id, signing_key_fingerprint,
                    prefill_full_name, prefill_customer_type, approval_key,
                    approved_by_employee_id, approved_by_name
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                uuid.uuid4(),
                invalid_draft_id,
                invalid_contact_id,
                "g" * 64,
                "Immutable Person",
                "residential",
                "approval-invalid",
                1,
                "Office User",
            )
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_schema_repair_fails_without_mutating_nonempty_legacy_table():
    database_url = _database_url_or_skip()
    schema = f"eom_public_repair_nonempty_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_known_legacy_schema(conn, schema)
        contact_id = uuid.uuid4()
        draft_id = uuid.uuid4()
        await conn.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await conn.execute(
            "INSERT INTO eom_onboarding_email_drafts (id) VALUES ($1)", draft_id
        )
        await conn.execute(
            """
            INSERT INTO eom_public_onboarding_tokens (
                id, draft_id, contact_id, approval_key,
                approved_by_employee_id, approved_by_name
            ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
            uuid.uuid4(),
            draft_id,
            contact_id,
            "legacy-approval",
            1,
            "Office User",
        )
        columns_before = await _legacy_columns(conn, schema)

        with pytest.raises(
            asyncpg.RaiseError,
            match="cannot safely repair nonempty eom_public_onboarding_tokens",
        ):
            await _run_schema_repair(conn)

        assert await _legacy_columns(conn, schema) == columns_before
        assert (
            await conn.fetchval("SELECT COUNT(*) FROM eom_public_onboarding_tokens")
            == 1
        )
        assert (
            await conn.fetchval(
                """
            SELECT COUNT(*)
            FROM schema_migrations
            WHERE name = $1
            """,
                MIGRATION_STEM,
            )
            == 0
        )
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_schema_repair_is_a_noop_for_complete_relation_with_token_rows():
    database_url = _database_url_or_skip()
    schema = f"eom_public_repair_compatible_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_complete_schema(conn, schema)
        contact_id = uuid.uuid4()
        draft_id = uuid.uuid4()
        token_id = uuid.uuid4()
        await conn.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await conn.execute(
            "INSERT INTO eom_onboarding_email_drafts (id) VALUES ($1)", draft_id
        )
        await conn.execute(
            """
            INSERT INTO eom_public_onboarding_tokens (
                id, draft_id, contact_id, signing_key_fingerprint,
                prefill_full_name, prefill_customer_type, approval_key,
                approved_by_employee_id, approved_by_name
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            token_id,
            draft_id,
            contact_id,
            "a" * 64,
            "Immutable Person",
            "residential",
            "complete-approval",
            1,
            "Office User",
        )
        before = dict(
            await conn.fetchrow(
                """
                SELECT id, draft_id, contact_id, signing_key_fingerprint,
                       prefill_full_name, prefill_customer_type, approval_key,
                       approved_by_employee_id, approved_by_name, status
                FROM eom_public_onboarding_tokens
                WHERE id = $1
                """,
                token_id,
            )
        )

        await _run_schema_repair(conn)

        assert (
            await conn.fetchval(
                "SELECT version FROM schema_migrations WHERE name = $1",
                MIGRATION_STEM,
            )
            == 384
        )
        after = dict(
            await conn.fetchrow(
                """
                SELECT id, draft_id, contact_id, signing_key_fingerprint,
                       prefill_full_name, prefill_customer_type, approval_key,
                       approved_by_employee_id, approved_by_name, status
                FROM eom_public_onboarding_tokens
                WHERE id = $1
                """,
                token_id,
            )
        )
        assert after == before
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()
