from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"


def _migration(name: str) -> str:
    return (MIGRATIONS_DIR / name).read_text()


def test_b2b_campaign_updated_at_migration_runs_before_sequence_unique_migration():
    migration_names = [path.name for path in sorted(MIGRATIONS_DIR.glob("*.sql"))]

    updated_at_index = migration_names.index("067_b2b_campaigns_updated_at.sql")
    sequence_setup_index = migration_names.index("068_campaign_sequences.sql")
    unique_recipient_index = migration_names.index(
        "309_campaign_sequences_unique_active_recipient.sql"
    )

    assert migration_names.index("066_b2b_campaigns.sql") < updated_at_index
    assert updated_at_index < sequence_setup_index
    assert updated_at_index < unique_recipient_index


def test_b2b_campaign_updated_at_migration_fills_column_required_by_309():
    updated_at_sql = _migration("067_b2b_campaigns_updated_at.sql")
    sequence_unique_sql = _migration("309_campaign_sequences_unique_active_recipient.sql")

    assert "ALTER TABLE b2b_campaigns" in updated_at_sql
    assert "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()" in (
        updated_at_sql
    )
    assert "UPDATE b2b_campaigns bc" in sequence_unique_sql
    assert "SET status = 'cancelled', updated_at = NOW()" in sequence_unique_sql


@pytest.mark.asyncio
async def test_b2b_campaign_updated_at_migration_repairs_066_table_for_309_update():
    asyncpg = pytest.importorskip("asyncpg")

    import os
    import uuid

    database_url = os.getenv(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema_name = f"atlas_migration_test_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        await conn.execute(f'CREATE SCHEMA "{schema_name}"')
        await conn.execute(f'SET search_path TO "{schema_name}", public')

        await conn.execute(_migration("066_b2b_campaigns.sql"))
        await conn.execute(
            """
            INSERT INTO b2b_campaigns
                (id, company_name, vendor_name, channel, body, status)
            VALUES
                ('11111111-1111-1111-1111-111111111111',
                 'Atlas Demo', 'Zendesk', 'email_cold', 'Body', 'draft')
            """
        )

        with pytest.raises(asyncpg.UndefinedColumnError):
            await conn.execute(
                "UPDATE b2b_campaigns SET status = 'expired', updated_at = NOW()"
            )

        await conn.execute(_migration("067_b2b_campaigns_updated_at.sql"))

        column = await conn.fetchrow(
            """
            SELECT data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = $1
              AND table_name = 'b2b_campaigns'
              AND column_name = 'updated_at'
            """,
            schema_name,
        )
        assert column is not None
        assert column["data_type"] == "timestamp with time zone"
        assert column["is_nullable"] == "NO"
        assert await conn.fetchval(
            "SELECT count(*) FROM b2b_campaigns WHERE updated_at IS NULL"
        ) == 0

        await conn.execute(
            "UPDATE b2b_campaigns SET status = 'expired', updated_at = NOW()"
        )
        await conn.execute(_migration("067_b2b_campaigns_updated_at.sql"))
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
        await conn.close()
