"""Real-Postgres verification for the Content Ops calibration library (#1497).

These tests prove the migration/repository contract that fake asyncpg pools
cannot: migration 334 -> 335 applies against Postgres, CHECK constraints and the
partial unique index fire, cascade delete works, and the repo functions
round-trip through asyncpg.
"""

from __future__ import annotations

import os
from pathlib import Path
import uuid

import pytest

from atlas_brain import _content_ops_calibration_library as calib
from extracted_content_pipeline.calibration_library import CalibrationLabel
from extracted_content_pipeline.campaign_ports import TenantScope


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"


def _database_url() -> str | None:
    return os.environ.get(DATABASE_URL_ENV)


def _migration(name: str) -> str:
    return (MIGRATIONS_DIR / name).read_text(encoding="utf-8")


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


async def _apply_calibration_chain(conn) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS saas_accounts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid()
        )
        """
    )
    await conn.execute(_migration("334_content_ops_claim_registry.sql"))
    await conn.execute(_migration("335_content_ops_calibration_library.sql"))


async def _new_schema(conn) -> str:
    schema_name = f"atlas_calibration_live_{uuid.uuid4().hex}"
    quoted = _quote_ident(schema_name)
    await conn.execute(f"CREATE SCHEMA {quoted}")
    await conn.execute(f"SET search_path TO {quoted}, public")
    return schema_name


@pytest.mark.asyncio
async def test_calibration_library_migration_and_repo_round_trip_live_db() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = _database_url()
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    conn = await asyncpg.connect(database_url)
    schema_name = await _new_schema(conn)
    try:
        await _apply_calibration_chain(conn)

        account_id = await conn.fetchval("INSERT INTO saas_accounts DEFAULT VALUES RETURNING id")
        other_account_id = await conn.fetchval(
            "INSERT INTO saas_accounts DEFAULT VALUES RETURNING id"
        )
        pool = conn

        created = await calib.create_calibration_example(
            pool,
            account_id=account_id,
            payload={
                "example_id": " Overclaim-001 ",
                "label": CalibrationLabel.OVERCLAIM.value,
                "excerpt": "Guaranteed 99.99% uptime.",
                "reasoning": "No published SLA supports that claim.",
                "source": "live-db-test",
                "metadata": {"case": "create"},
            },
        )
        assert created.example_id == "Overclaim-001"
        assert created.metadata == {"case": "create"}

        reader = calib.ContentOpsCalibrationLibraryRepository(pool)
        examples = await reader.list_calibration_examples(
            scope=TenantScope(account_id=str(account_id))
        )
        assert tuple(item.example_id for item in examples) == ("Overclaim-001",)
        assert examples[0].label is CalibrationLabel.OVERCLAIM

        records = await calib.list_calibration_example_records(pool, account_id=account_id)
        assert [record.id for record in records] == [created.id]

        updated = await calib.update_calibration_example(
            pool,
            account_id=account_id,
            example_row_id=created.id,
            payload={
                "example_id": "Voice-001",
                "label": CalibrationLabel.VOICE_DRIFT.value,
                "excerpt": "Crush your quota instantly.",
                "reasoning": "Tone does not match the brand voice.",
                "metadata": {"case": "update"},
            },
        )
        assert updated is not None
        assert updated.example_id == "Voice-001"
        assert updated.label == CalibrationLabel.VOICE_DRIFT.value

        same_id_other_tenant = await calib.create_calibration_example(
            pool,
            account_id=other_account_id,
            payload={
                "example_id": "voice-001",
                "label": CalibrationLabel.VOICE_DRIFT.value,
                "excerpt": "Other tenant can reuse the same logical id.",
                "reasoning": "The unique index is tenant scoped.",
            },
        )
        assert same_id_other_tenant.account_id == other_account_id

        with pytest.raises(asyncpg.UniqueViolationError):
            await calib.create_calibration_example(
                pool,
                account_id=account_id,
                payload={
                    "example_id": " voice-001 ",
                    "label": CalibrationLabel.VOICE_DRIFT.value,
                    "excerpt": "Duplicate active id.",
                    "reasoning": "Should violate lower/btrim partial unique index.",
                },
            )

        assert await calib.archive_calibration_example(
            pool, account_id=account_id, example_row_id=created.id
        )
        assert await reader.list_calibration_examples(
            scope=TenantScope(account_id=str(account_id))
        ) == ()

        reused = await calib.create_calibration_example(
            pool,
            account_id=account_id,
            payload={
                "example_id": "voice-001",
                "label": CalibrationLabel.GOOD_VOICE.value,
                "excerpt": "Clear, specific, and on brand.",
                "reasoning": "Archived rows do not participate in the active unique index.",
            },
        )
        assert reused.id != created.id

        await conn.execute("DELETE FROM saas_accounts WHERE id = $1", account_id)
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM content_ops_calibration_library WHERE account_id = $1",
                account_id,
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM content_ops_calibration_library WHERE account_id = $1",
                other_account_id,
            )
            == 1
        )
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema_name)} CASCADE")
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        ("label", "not_a_label", "label"),
        ("excerpt", "   ", "excerpt"),
        ("reasoning", "", "reasoning"),
    ),
)
async def test_calibration_library_live_db_constraints_reject_bad_rows(
    column: str, value: str, message: str
) -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = _database_url()
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    conn = await asyncpg.connect(database_url)
    schema_name = await _new_schema(conn)
    try:
        await _apply_calibration_chain(conn)
        account_id = await conn.fetchval("INSERT INTO saas_accounts DEFAULT VALUES RETURNING id")
        row = {
            "example_id": "constraint-001",
            "label": CalibrationLabel.OVERCLAIM.value,
            "excerpt": "Example excerpt.",
            "reasoning": "Example reasoning.",
        }
        row[column] = value

        with pytest.raises(asyncpg.CheckViolationError, match=message):
            await conn.execute(
                """
                INSERT INTO content_ops_calibration_library
                    (account_id, example_id, label, excerpt, reasoning)
                VALUES ($1, $2, $3, $4, $5)
                """,
                account_id,
                row["example_id"],
                row["label"],
                row["excerpt"],
                row["reasoning"],
            )
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema_name)} CASCADE")
        await conn.close()
