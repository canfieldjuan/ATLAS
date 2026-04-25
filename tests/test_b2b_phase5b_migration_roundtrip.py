"""Phase 5b migration + persistence integration coverage.

Three tests:

  1. test_migration_303_sql_declares_expected_columns
       Pure parse-level check on the migration file. Always runs in CI.
       Catches typos or schema drift in the SQL itself before any DB
       round-trip.

  2. test_migration_303_columns_exist_in_live_db
       Connects to the configured Postgres. Skips when unreachable.
       Asserts each new column exists on b2b_vendor_witnesses with the
       expected data_type. Verifies the migration was actually applied
       to the dev DB and matches the SQL file.

  3. test_persist_packet_artifacts_roundtrips_phase5b_columns
       Connects to the configured Postgres. Skips when unreachable.
       Builds a minimal CompressedPacket with one witness whose Phase 5b
       fields are populated, calls _persist_packet_artifacts, then
       SELECTs the row back and asserts each column round-tripped.
       Cleans up sentinel-vendor rows before and after so we never
       pollute real data.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from atlas_brain.autonomous.tasks._b2b_pool_compression import CompressedPacket
from atlas_brain.autonomous.tasks._b2b_witnesses import compute_witness_hash


_MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain"
    / "storage"
    / "migrations"
    / "303_witness_phrase_metadata.sql"
)

_EXPECTED_COLUMNS: dict[str, str] = {
    # column_name -> expected information_schema data_type
    "phrase_polarity": "text",
    "phrase_subject": "text",
    "phrase_role": "text",
    "phrase_verbatim": "boolean",
    "pain_confidence": "text",
}


_SENTINEL_VENDOR = "_phase5b_roundtrip_test_vendor"
_SENTINEL_AS_OF = date(2099, 1, 1)
_SENTINEL_WINDOW = 90


# ---------------------------------------------------------------------------
# 1. Parse-level test (always runs)
# ---------------------------------------------------------------------------


def test_migration_303_sql_declares_expected_columns():
    """The migration file must contain ADD COLUMN IF NOT EXISTS for each
    of the 5 Phase 5b columns with the expected type. Catches SQL typos
    before they hit a real database."""
    assert _MIGRATION_PATH.exists(), f"missing migration: {_MIGRATION_PATH}"
    sql = _MIGRATION_PATH.read_text()

    # Each column declaration looks like:
    #   ADD COLUMN IF NOT EXISTS <name> <type> NULL
    # case-insensitive, whitespace-tolerant.
    expected_types_for_sql = {
        "phrase_polarity": "text",
        "phrase_subject": "text",
        "phrase_role": "text",
        "phrase_verbatim": "boolean",
        "pain_confidence": "text",
    }
    for column, sql_type in expected_types_for_sql.items():
        pattern = (
            rf"ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+{re.escape(column)}\s+"
            rf"{re.escape(sql_type)}\b"
        )
        assert re.search(pattern, sql, re.IGNORECASE), (
            f"migration 303 missing column declaration: {column} {sql_type}"
        )

    # Sanity: an index on pain_confidence so the API banner-rendering
    # filter stays cheap.
    assert "idx_b2b_vendor_witnesses_pain_confidence" in sql, (
        "migration 303 missing pain_confidence index"
    )


# ---------------------------------------------------------------------------
# DB-integration fixture (skips when Postgres unreachable)
# ---------------------------------------------------------------------------


@pytest.fixture
async def live_pool():
    """Real asyncpg pool. Skips the test cleanly if the configured DB is
    unreachable (e.g. local dev DB not running, CI without Postgres).

    We do NOT call init_database()/get_db_pool() because those install a
    process-wide singleton that other tests in the run may rely on.
    Build a private pool here and dispose of it at end of test."""
    asyncpg = pytest.importorskip("asyncpg")
    try:
        from atlas_brain.storage.config import db_settings
    except Exception as e:  # pragma: no cover -- defensive
        pytest.skip(f"cannot import db_settings: {e}")

    try:
        pool = await asyncpg.create_pool(
            host=db_settings.host,
            port=db_settings.port,
            database=db_settings.database,
            user=db_settings.user,
            password=db_settings.password,
            min_size=1,
            max_size=2,
            timeout=2.0,
            command_timeout=5.0,
        )
    except Exception as e:
        pytest.skip(f"Postgres unreachable, skipping integration test: {e}")
    try:
        yield pool
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# 2. information_schema.columns presence test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_303_columns_exist_in_live_db(live_pool):
    """Verify the 5 Phase 5b columns are actually applied on the
    b2b_vendor_witnesses table. If this fails, the dev DB hasn't run
    migration 303."""
    rows = await live_pool.fetch(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'b2b_vendor_witnesses'
          AND column_name = ANY($1::text[])
        """,
        list(_EXPECTED_COLUMNS),
    )
    found = {r["column_name"]: r["data_type"] for r in rows}
    missing = sorted(set(_EXPECTED_COLUMNS) - set(found))
    assert not missing, (
        f"migration 303 not applied to live DB; missing columns: {missing}"
    )
    for column, expected_type in _EXPECTED_COLUMNS.items():
        assert found[column] == expected_type, (
            f"{column}: expected {expected_type}, got {found[column]}"
        )


# ---------------------------------------------------------------------------
# 3. INSERT round-trip
# ---------------------------------------------------------------------------


def _make_witness(*, witness_id: str, **overrides) -> dict:
    """Build a minimal witness dict matching the persistence contract."""
    base = {
        "witness_id": witness_id,
        "_sid": witness_id,
        "review_id": "00000000-0000-0000-0000-000000000000",
        "witness_type": "common_pattern",
        "excerpt_text": "pricing keeps going up unexpectedly",
        "source": "g2",
        "reviewed_at": datetime(2026, 4, 1, tzinfo=timezone.utc),
        "reviewer_company": "Globex",
        "reviewer_title": "VP Engineering",
        "pain_category": "pricing",
        "competitor": None,
        "salience_score": 1.5,
        "selection_reason": "selected_for_common_pattern",
        "signal_tags": ["named_org"],
        "specificity_score": 2.0,
        "generic_reason": None,
        "source_span_id": "review:r1:span:0-32",
        "grounding_status": "grounded",
        # Phase 5b columns under test:
        "phrase_polarity": "negative",
        "phrase_subject": "subject_vendor",
        "phrase_role": "primary_driver",
        "phrase_verbatim": True,
        "pain_confidence": "weak",
    }
    base.update(overrides)
    base["witness_hash"] = compute_witness_hash(base)
    return base


async def _delete_sentinel_rows(pool) -> None:
    await pool.execute(
        """
        DELETE FROM b2b_vendor_witnesses
        WHERE vendor_name = $1
          AND as_of_date = $2
          AND analysis_window_days = $3
        """,
        _SENTINEL_VENDOR,
        _SENTINEL_AS_OF,
        _SENTINEL_WINDOW,
    )
    await pool.execute(
        """
        DELETE FROM b2b_vendor_reasoning_packets
        WHERE vendor_name = $1
          AND as_of_date = $2
          AND analysis_window_days = $3
        """,
        _SENTINEL_VENDOR,
        _SENTINEL_AS_OF,
        _SENTINEL_WINDOW,
    )


@pytest.mark.asyncio
async def test_persist_packet_artifacts_roundtrips_phase5b_columns(live_pool):
    """End-to-end: persist one witness with all five Phase 5b fields
    populated, SELECT it back, and assert each value round-tripped."""
    from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
        _persist_packet_artifacts,
    )

    await _delete_sentinel_rows(live_pool)
    try:
        witness = _make_witness(witness_id="phase5b:roundtrip:1")
        packet = CompressedPacket(
            vendor_name=_SENTINEL_VENDOR,
            pools={},
            aggregates=[],
            witness_pack=[witness],
        )

        await _persist_packet_artifacts(
            live_pool,
            vendor_name=_SENTINEL_VENDOR,
            as_of_date=_SENTINEL_AS_OF,
            analysis_window_days=_SENTINEL_WINDOW,
            evidence_hash="phase5b_roundtrip_hash",
            packet=packet,
        )

        row = await live_pool.fetchrow(
            """
            SELECT phrase_polarity, phrase_subject, phrase_role,
                   phrase_verbatim, pain_confidence,
                   grounding_status, witness_hash
            FROM b2b_vendor_witnesses
            WHERE vendor_name = $1
              AND as_of_date = $2
              AND analysis_window_days = $3
              AND witness_id = $4
            """,
            _SENTINEL_VENDOR,
            _SENTINEL_AS_OF,
            _SENTINEL_WINDOW,
            "phase5b:roundtrip:1",
        )
        assert row is not None, "row not persisted"
        assert row["phrase_polarity"] == "negative"
        assert row["phrase_subject"] == "subject_vendor"
        assert row["phrase_role"] == "primary_driver"
        assert row["phrase_verbatim"] is True
        assert row["pain_confidence"] == "weak"
        # Sanity: Phase 1b column unaffected; Phase 1b classifier ran on
        # write so this should be one of the real enum values, not pending.
        assert row["grounding_status"] in ("grounded", "not_grounded")
        # Sanity: hash not empty (means compute_witness_hash ran)
        assert row["witness_hash"] and len(row["witness_hash"]) >= 8

        # Update the same witness with new Phase 5b values; verify the
        # ON CONFLICT DO UPDATE branch refreshes them too.
        updated_witness = _make_witness(
            witness_id="phase5b:roundtrip:1",
            phrase_polarity="mixed",
            phrase_subject="subject_vendor",
            phrase_role="supporting_context",
            phrase_verbatim=False,
            pain_confidence="strong",
        )
        packet2 = CompressedPacket(
            vendor_name=_SENTINEL_VENDOR,
            pools={},
            aggregates=[],
            witness_pack=[updated_witness],
        )
        await _persist_packet_artifacts(
            live_pool,
            vendor_name=_SENTINEL_VENDOR,
            as_of_date=_SENTINEL_AS_OF,
            analysis_window_days=_SENTINEL_WINDOW,
            evidence_hash="phase5b_roundtrip_hash_v2",
            packet=packet2,
        )

        row2 = await live_pool.fetchrow(
            """
            SELECT phrase_polarity, phrase_subject, phrase_role,
                   phrase_verbatim, pain_confidence
            FROM b2b_vendor_witnesses
            WHERE vendor_name = $1
              AND as_of_date = $2
              AND analysis_window_days = $3
              AND witness_id = $4
            """,
            _SENTINEL_VENDOR,
            _SENTINEL_AS_OF,
            _SENTINEL_WINDOW,
            "phase5b:roundtrip:1",
        )
        assert row2["phrase_polarity"] == "mixed"
        assert row2["phrase_role"] == "supporting_context"
        assert row2["phrase_verbatim"] is False
        assert row2["pain_confidence"] == "strong"
    finally:
        await _delete_sentinel_rows(live_pool)
