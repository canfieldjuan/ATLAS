from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atlas_brain.storage.migrations import reconciliation as reconciliation_mod
from atlas_brain.storage import recurring_invoice_schema


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_migration_content_integrity.py"
SPEC = importlib.util.spec_from_file_location("check_migration_content_integrity", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class FakeReadOnlyTransaction:
    def __init__(self, connection: "FakeConnection", readonly: bool):
        self.connection = connection
        self.readonly = readonly

    async def __aenter__(self) -> "FakeReadOnlyTransaction":
        self.connection.transaction_readonly.append(self.readonly)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class FakeConnection:
    def __init__(
        self,
        records: list[tuple[str, str | None]],
        *,
        reconciliation_rows: list[dict[str, object]] | None = None,
        public_onboarding_reconciliation_rows: list[dict[str, object]] | None = None,
        public_onboarding_columns: list[dict[str, object]] | None = None,
        public_onboarding_constraints: list[dict[str, object]] | None = None,
        public_onboarding_indexes: dict[str, dict[str, object] | None] | None = None,
        recurring_schema_ready: bool = True,
        zero_active_null_period_rows: bool = True,
    ):
        self.records = records
        self.reconciliation_rows = reconciliation_rows or []
        self.public_onboarding_reconciliation_rows = (
            public_onboarding_reconciliation_rows or []
        )
        self.public_onboarding_columns = (
            _default_public_onboarding_columns()
            if public_onboarding_columns is None
            else public_onboarding_columns
        )
        self.public_onboarding_constraints = (
            _default_public_onboarding_constraints()
            if public_onboarding_constraints is None
            else public_onboarding_constraints
        )
        self.public_onboarding_indexes = (
            _default_public_onboarding_indexes()
            if public_onboarding_indexes is None
            else public_onboarding_indexes
        )
        self.recurring_schema_ready = recurring_schema_ready
        self.zero_active_null_period_rows = zero_active_null_period_rows
        self.queries: list[str] = []
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchval_calls: list[tuple[str, tuple[object, ...]]] = []
        self.transaction_readonly: list[bool] = []
        self.execute_calls: list[str] = []
        self.closed = False

    def transaction(self, *, readonly: bool = False) -> FakeReadOnlyTransaction:
        return FakeReadOnlyTransaction(self, readonly)

    async def fetch(self, query: str, *args: object):
        self.queries.append(query)
        self.fetch_calls.append((query, args))
        if query == "SELECT name, content_sha256 FROM schema_migrations":
            assert args == ()
            return [
                {"name": name, "content_sha256": content_sha256}
                for name, content_sha256 in self.records
            ]
        if query == (
            "SELECT content_sha256, applied_at FROM schema_migrations "
            "WHERE name = $1 LIMIT 2"
        ):
            if args == (reconciliation_mod.MIGRATION_387_RECONCILIATION.migration_name,):
                return self.reconciliation_rows
            assert args == (
                reconciliation_mod.MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION.migration_name,
            )
            return self.public_onboarding_reconciliation_rows
        if "FROM information_schema.columns AS actual" in query:
            assert args == (
                list(reconciliation_mod._PUBLIC_ONBOARDING_TOKEN_IMMUTABLE_COLUMNS),
            )
            return self.public_onboarding_columns
        if "eom_public_onboarding_tokens" in query:
            assert "FROM pg_constraint AS actual" in query
            assert args == (
                list(
                    reconciliation_mod._PUBLIC_ONBOARDING_TOKEN_CONSTRAINT_EXPRESSIONS
                ),
            )
            return self.public_onboarding_constraints
        assert "FROM pg_constraint AS actual" in query
        assert args == (list(recurring_invoice_schema._RECURRING_INVOICE_DEDUP_CONSTRAINTS),)
        return [
            {"conname": name, "definition": definition}
            for name, definition in (
                recurring_invoice_schema._RECURRING_INVOICE_DEDUP_CONSTRAINT_EXPRESSIONS.items()
            )
        ]

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        self.fetchrow_calls.append((query, args))
        if "eom_public_onboarding_tokens" in query:
            assert len(args) == 1
            return self.public_onboarding_indexes.get(str(args[0]))
        assert "FROM pg_index AS index_state" in query
        assert args == (recurring_invoice_schema._RECURRING_INVOICE_DEDUP_INDEX,)
        return {
            "indisunique": True,
            "indisvalid": True,
            "indisready": True,
            "indnkeyatts": 2,
            "key_column_1": "contact_id",
            "key_column_2": "billing_period",
            "predicate": (
                "(billing_period IS NOT NULL) AND "
                "(source = ANY (ARRAY['monthly_auto', 'eom_commercial_billing'])) "
                "AND (status <> 'void')"
            ),
        }

    async def fetchval(self, query: str, *args: object) -> bool:
        self.fetchval_calls.append((query, args))
        assert args == ()
        if "information_schema.columns AS actual" in query:
            return self.recurring_schema_ready
        assert "FROM invoices" in query
        return self.zero_active_null_period_rows

    async def execute(self, query: str, *args) -> None:
        self.execute_calls.append(query)
        raise AssertionError("read-only provenance preflight must not execute SQL")

    async def close(self) -> None:
        self.closed = True


def _default_public_onboarding_columns() -> list[dict[str, object]]:
    return [
        {
            "column_name": name,
            "data_type": data_type,
            "character_maximum_length": maximum_length,
            "is_nullable": nullable,
        }
        for name, (data_type, maximum_length, nullable) in (
            reconciliation_mod._PUBLIC_ONBOARDING_TOKEN_IMMUTABLE_COLUMNS.items()
        )
    ]


def _default_public_onboarding_constraints() -> list[dict[str, object]]:
    return [
        {"conname": name, "definition": definition}
        for name, definition in (
            reconciliation_mod._PUBLIC_ONBOARDING_TOKEN_CONSTRAINT_EXPRESSIONS.items()
        )
    ]


def _default_public_onboarding_indexes() -> dict[str, dict[str, object]]:
    return {
        reconciliation_mod._PUBLIC_ONBOARDING_TOKEN_ISSUED_CONTACT_INDEX: {
            "indisunique": True,
            "indisvalid": True,
            "indisready": True,
            "indnkeyatts": 1,
            "key_column_1": "contact_id",
            "definition": (
                "CREATE UNIQUE INDEX uq_eom_public_onboarding_tokens_issued_contact "
                "ON eom_public_onboarding_tokens USING btree (contact_id) "
                "WHERE ((status)::text = 'issued'::text)"
            ),
            "predicate": "((status)::text = 'issued'::text)",
        },
        reconciliation_mod._PUBLIC_ONBOARDING_TOKEN_STATUS_INDEX: {
            "indisunique": False,
            "indisvalid": True,
            "indisready": True,
            "indnkeyatts": 2,
            "key_column_1": "status",
            "key_column_2": "issued_at",
            "definition": (
                "CREATE INDEX idx_eom_public_onboarding_tokens_status "
                "ON eom_public_onboarding_tokens USING btree (status, issued_at DESC)"
            ),
            "predicate": None,
        },
    }


def _write_migration(directory: Path, name: str, content: bytes) -> Path:
    path = directory / f"{name}.sql"
    path.write_bytes(content)
    return path


def _migration_387_source() -> bytes:
    return (
        ROOT
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "387_eom_recurring_invoice_dedup_recovery.sql"
    ).read_bytes()


def _migration_387_connection(
    *,
    ledger_digest: str | None = None,
    applied_at: object | None = None,
    recurring_schema_ready: bool = True,
    zero_active_null_period_rows: bool = True,
) -> FakeConnection:
    record = reconciliation_mod.MIGRATION_387_RECONCILIATION
    actual_digest = ledger_digest or record.historical_ledger_sha256
    actual_applied_at = record.observed_applied_at if applied_at is None else applied_at
    reconciliation_row = {
        "content_sha256": actual_digest,
        "applied_at": actual_applied_at,
    }
    return FakeConnection(
        [(record.migration_name, actual_digest)],
        reconciliation_rows=[reconciliation_row],
        recurring_schema_ready=recurring_schema_ready,
        zero_active_null_period_rows=zero_active_null_period_rows,
    )


def _migration_382_connection(
    *,
    ledger_digest: str | None = None,
    applied_at: object | None = None,
    reconciliation_rows: list[dict[str, object]] | None = None,
) -> FakeConnection:
    record = (
        reconciliation_mod.MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION
    )
    actual_applied_at = record.observed_applied_at if applied_at is None else applied_at
    actual_rows = reconciliation_rows
    if actual_rows is None:
        actual_rows = [{
            "content_sha256": ledger_digest,
            "applied_at": actual_applied_at,
        }]
    return FakeConnection(
        [(record.migration_name, ledger_digest)],
        public_onboarding_reconciliation_rows=actual_rows,
    )


@pytest.mark.asyncio
async def test_preflight_reports_unresolved_drift_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verified = _write_migration(tmp_path, "900_verified", b"SELECT 'verified';\n")
    _write_migration(tmp_path, "901_legacy", b"SELECT 'legacy';\n")
    _write_migration(tmp_path, "902_mismatched", b"SELECT 'mismatched';\n")
    connection = FakeConnection([
        ("900_verified", hashlib.sha256(verified.read_bytes()).hexdigest()),
        ("901_legacy", None),
        ("902_mismatched", "not-the-current-digest"),
        ("903_missing_source", "f" * 64),
    ])

    async def connect_read_only() -> FakeConnection:
        return connection

    monkeypatch.setattr(module, "_connect_read_only", connect_read_only)

    code = await module._main(migrations_dir=tmp_path)

    payload = json.loads(capsys.readouterr().out)
    assert code == module.UNRESOLVED_DRIFT_EXIT
    assert payload == {
        "check_completed": True,
        "counts": {
            "legacy_unverified": 1,
            "mismatched": 1,
            "missing_source": 1,
            "verified": 1,
        },
        "database_target": module.db_settings.target_label,
        "exit_code": module.UNRESOLVED_DRIFT_EXIT,
        "report": {
            "legacy_unverified": ["901_legacy"],
            "mismatched": ["902_mismatched"],
            "missing_source": ["903_missing_source"],
            "verified": ["900_verified"],
        },
        "status": "unresolved_drift",
    }
    assert connection.queries == ["SELECT name, content_sha256 FROM schema_migrations"]
    assert connection.transaction_readonly == [True]
    assert connection.execute_calls == []
    assert connection.closed is True


@pytest.mark.asyncio
async def test_preflight_keeps_legacy_evidence_visible_without_treating_it_as_drift(
    tmp_path: Path,
) -> None:
    _write_migration(tmp_path, "901_legacy", b"SELECT 'legacy';\n")
    connection = FakeConnection([("901_legacy", None)])

    code, payload = await module.run_migration_content_integrity_preflight(
        connection,
        migrations_dir=tmp_path,
    )

    assert code == 0
    assert payload["status"] == "legacy_unverified"
    assert payload["report"] == {
        "verified": [],
        "legacy_unverified": ["901_legacy"],
        "mismatched": [],
        "missing_source": [],
    }
    assert connection.transaction_readonly == [True]
    assert connection.execute_calls == []


def test_migration_387_reconciliation_record_matches_checked_in_final_source() -> None:
    record = reconciliation_mod.MIGRATION_387_RECONCILIATION

    assert record.source_verification == reconciliation_mod.HISTORICAL_SOURCE_UNAVAILABLE
    assert record.historical_ledger_sha256 != record.final_packaged_sha256
    assert hashlib.sha256(_migration_387_source()).hexdigest() == record.final_packaged_sha256
    assert record.observed_applied_at == datetime(
        2026,
        8,
        21,
        1,
        30,
        46,
        82_989,
        tzinfo=timezone.utc,
    )
    assert record.observed_applied_at < record.earliest_retained_source_commit_at


def test_migration_382_reconciliation_record_is_closed_legacy_source_evidence() -> None:
    record = (
        reconciliation_mod.MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION
    )

    assert record.source_verification == reconciliation_mod.HISTORICAL_SOURCE_UNAVAILABLE
    assert record.historical_ledger_sha256 is None
    assert record.observed_applied_at == datetime(
        2026,
        8,
        17,
        19,
        18,
        7,
        242_686,
        tzinfo=timezone.utc,
    )
    assert reconciliation_mod.known_historical_missing_source_reconciliation_names() == {
        record.migration_name,
    }
    assert reconciliation_mod.known_historical_reconciliation_names() == {
        record.migration_name,
        reconciliation_mod.MIGRATION_387_RECONCILIATION.migration_name,
    }


@pytest.mark.asyncio
async def test_default_known_reconciliation_attestation_preserves_387_only_behavior(
    tmp_path: Path,
) -> None:
    record = reconciliation_mod.MIGRATION_387_RECONCILIATION
    _write_migration(tmp_path, record.migration_name, _migration_387_source())

    attestations = await reconciliation_mod.attest_known_historical_migration_reconciliations(
        _migration_387_connection(),
        sorted(tmp_path.glob("*.sql")),
    )

    assert [attestation.migration_name for attestation in attestations] == [
        record.migration_name,
    ]


@pytest.mark.asyncio
async def test_known_387_reconciliation_attests_catalog_without_verifying_source(
    tmp_path: Path,
) -> None:
    record = reconciliation_mod.MIGRATION_387_RECONCILIATION
    _write_migration(tmp_path, record.migration_name, _migration_387_source())
    connection = _migration_387_connection()

    code, payload = await module.run_migration_content_integrity_preflight(
        connection,
        migrations_dir=tmp_path,
        attest_known_reconciliations=True,
    )

    assert code == module.UNRESOLVED_DRIFT_EXIT
    assert payload["status"] == "unresolved_drift"
    assert payload["report"]["mismatched"] == [record.migration_name]
    assert payload["known_reconciliation_evidence"] == [{
        "reconciliation_id": record.reconciliation_id,
        "migration_name": record.migration_name,
        "source_verification": reconciliation_mod.HISTORICAL_SOURCE_UNAVAILABLE,
        "exactly_one_ledger_row": True,
        "ledger_digest_matches_record": True,
        "packaged_digest_matches_record": True,
        "applied_at_matches_record": True,
        "applied_before_retained_source": True,
        "recurring_schema_ready": True,
        "zero_active_null_period_recurring_rows": True,
        "status": "attested",
    }]
    assert connection.fetch_calls[0] == (
        "SELECT name, content_sha256 FROM schema_migrations",
        (),
    )
    assert connection.fetch_calls[1] == (
        "SELECT content_sha256, applied_at FROM schema_migrations WHERE name = $1 LIMIT 2",
        (record.migration_name,),
    )
    assert len(connection.fetchrow_calls) == 1
    assert len(connection.fetchval_calls) == 2
    assert connection.transaction_readonly == [True]
    assert connection.execute_calls == []


@pytest.mark.asyncio
async def test_known_382_reconciliation_attests_complete_catalog_without_source(
    tmp_path: Path,
) -> None:
    record = (
        reconciliation_mod.MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION
    )
    connection = _migration_382_connection()

    code, payload = await module.run_migration_content_integrity_preflight(
        connection,
        migrations_dir=tmp_path,
        attest_known_reconciliations=True,
    )

    assert code == module.UNRESOLVED_DRIFT_EXIT
    assert payload["status"] == "unresolved_drift"
    assert payload["report"]["missing_source"] == [record.migration_name]
    assert payload["known_reconciliation_evidence"] == [{
        "reconciliation_id": record.reconciliation_id,
        "migration_name": record.migration_name,
        "source_verification": reconciliation_mod.HISTORICAL_SOURCE_UNAVAILABLE,
        "exactly_one_ledger_row": True,
        "ledger_digest_is_null": True,
        "applied_at_matches_record": True,
        "immutable_projection_ready": True,
        "fingerprint_check_ready": True,
        "terminal_state_check_ready": True,
        "issued_contact_index_ready": True,
        "status_index_ready": True,
        "status": "attested",
    }]
    assert connection.fetch_calls[0] == (
        "SELECT name, content_sha256 FROM schema_migrations",
        (),
    )
    assert connection.fetch_calls[1] == (
        "SELECT content_sha256, applied_at FROM schema_migrations WHERE name = $1 LIMIT 2",
        (record.migration_name,),
    )
    assert len(connection.fetchrow_calls) == 2
    assert connection.transaction_readonly == [True]
    assert connection.execute_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "field"),
    [
        ("non-null ledger digest", "ledger_digest_is_null"),
        ("truncated applied timestamp", "applied_at_matches_record"),
        ("duplicate ledger rows", "exactly_one_ledger_row"),
        ("missing immutable column", "immutable_projection_ready"),
        ("weakened fingerprint check", "fingerprint_check_ready"),
        ("weakened terminal-state check", "terminal_state_check_ready"),
        ("missing issued-contact index", "issued_contact_index_ready"),
        ("missing status index", "status_index_ready"),
    ],
)
async def test_known_382_reconciliation_rejects_each_required_evidence_field(
    case: str,
    field: str,
    tmp_path: Path,
) -> None:
    record = (
        reconciliation_mod.MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION
    )
    connection = _migration_382_connection()
    if case == "non-null ledger digest":
        connection = _migration_382_connection(ledger_digest="a" * 64)
    elif case == "truncated applied timestamp":
        connection = _migration_382_connection(
            applied_at=record.observed_applied_at.replace(microsecond=0)
        )
    elif case == "duplicate ledger rows":
        connection = _migration_382_connection(reconciliation_rows=[
            {
                "content_sha256": None,
                "applied_at": record.observed_applied_at,
            },
            {
                "content_sha256": None,
                "applied_at": record.observed_applied_at,
            },
        ])
    elif case == "missing immutable column":
        connection.public_onboarding_columns = connection.public_onboarding_columns[1:]
    elif case == "weakened fingerprint check":
        connection.public_onboarding_constraints[0] = {
            "conname": "eom_public_onboarding_tokens_signing_key_fingerprint_check",
            "definition": "((signing_key_fingerprint)::text ~ '^[0-9a-f]{63}$'::text)",
        }
    elif case == "weakened terminal-state check":
        connection.public_onboarding_constraints[1] = {
            "conname": "ck_eom_public_onboarding_tokens_terminal_state",
            "definition": "(status = 'issued')",
        }
    elif case == "missing issued-contact index":
        connection.public_onboarding_indexes[
            reconciliation_mod._PUBLIC_ONBOARDING_TOKEN_ISSUED_CONTACT_INDEX
        ] = None
    elif case == "missing status index":
        connection.public_onboarding_indexes[
            reconciliation_mod._PUBLIC_ONBOARDING_TOKEN_STATUS_INDEX
        ] = None
    else:  # pragma: no cover - parametrize keeps this exhaustive.
        raise AssertionError(f"unexpected evidence case: {case}")

    code, payload = await module.run_migration_content_integrity_preflight(
        connection,
        migrations_dir=tmp_path,
        attest_known_reconciliations=True,
    )

    evidence = payload["known_reconciliation_evidence"][0]
    assert code == module.UNRESOLVED_DRIFT_EXIT, case
    assert evidence["source_verification"] == reconciliation_mod.HISTORICAL_SOURCE_UNAVAILABLE
    assert evidence[field] is False, case
    assert evidence["status"] == "not_attested", case
    assert connection.execute_calls == []


@pytest.mark.asyncio
async def test_known_387_reconciliation_rejects_truncated_observed_timestamp(
    tmp_path: Path,
) -> None:
    """A seconds-only timestamp must not silently attest target evidence."""
    record = reconciliation_mod.MIGRATION_387_RECONCILIATION
    _write_migration(tmp_path, record.migration_name, _migration_387_source())
    connection = _migration_387_connection(
        applied_at=record.observed_applied_at.replace(microsecond=0),
    )

    code, payload = await module.run_migration_content_integrity_preflight(
        connection,
        migrations_dir=tmp_path,
        attest_known_reconciliations=True,
    )

    evidence = payload["known_reconciliation_evidence"][0]
    assert code == module.UNRESOLVED_DRIFT_EXIT
    assert evidence["applied_at_matches_record"] is False
    assert evidence["status"] == "not_attested"
    assert connection.execute_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "case",
        "ledger_digest",
        "applied_at",
        "source",
        "schema_is_ready",
        "zero_rows",
        "fields",
    ),
    [
        (
            "ledger digest changed",
            "a" * 64,
            None,
            None,
            True,
            True,
            ("ledger_digest_matches_record",),
        ),
        (
            "application time changed",
            None,
            reconciliation_mod.MIGRATION_387_RECONCILIATION.earliest_retained_source_commit_at,
            None,
            True,
            True,
            ("applied_at_matches_record", "applied_before_retained_source"),
        ),
        (
            "packaged source changed",
            None,
            None,
            _migration_387_source() + b"\n-- changed after historical evidence\n",
            True,
            True,
            ("packaged_digest_matches_record",),
        ),
        (
            "recurring schema no longer ready",
            None,
            None,
            None,
            False,
            True,
            ("recurring_schema_ready",),
        ),
        (
            "active null-period recurring row exists",
            None,
            None,
            None,
            True,
            False,
            ("zero_active_null_period_recurring_rows",),
        ),
    ],
)
async def test_known_387_reconciliation_remains_not_attested_when_evidence_changes(
    case: str,
    ledger_digest: str | None,
    applied_at: object | None,
    source: bytes | None,
    schema_is_ready: bool,
    zero_rows: bool,
    fields: tuple[str, ...],
    tmp_path: Path,
) -> None:
    record = reconciliation_mod.MIGRATION_387_RECONCILIATION
    _write_migration(
        tmp_path,
        record.migration_name,
        _migration_387_source() if source is None else source,
    )
    connection = _migration_387_connection(
        ledger_digest=ledger_digest,
        applied_at=applied_at,
        recurring_schema_ready=schema_is_ready,
        zero_active_null_period_rows=zero_rows,
    )

    code, payload = await module.run_migration_content_integrity_preflight(
        connection,
        migrations_dir=tmp_path,
        attest_known_reconciliations=True,
    )

    evidence = payload["known_reconciliation_evidence"][0]
    assert code == module.UNRESOLVED_DRIFT_EXIT, case
    assert evidence["source_verification"] == reconciliation_mod.HISTORICAL_SOURCE_UNAVAILABLE
    assert all(evidence[field] is False for field in fields), case
    assert evidence["status"] == "not_attested", case
    assert connection.execute_calls == []
    if not schema_is_ready:
        assert len(connection.fetchval_calls) == 1


@pytest.mark.asyncio
async def test_known_387_reconciliation_rejects_duplicate_ledger_rows(
    tmp_path: Path,
) -> None:
    record = reconciliation_mod.MIGRATION_387_RECONCILIATION
    _write_migration(tmp_path, record.migration_name, _migration_387_source())
    expected_row = {
        "content_sha256": record.historical_ledger_sha256,
        "applied_at": record.observed_applied_at,
    }
    conflicting_row = {
        "content_sha256": "a" * 64,
        "applied_at": record.earliest_retained_source_commit_at,
    }
    connection = FakeConnection(
        [
            (record.migration_name, expected_row["content_sha256"]),
            (record.migration_name, conflicting_row["content_sha256"]),
        ],
        reconciliation_rows=[expected_row, conflicting_row],
    )

    code, payload = await module.run_migration_content_integrity_preflight(
        connection,
        migrations_dir=tmp_path,
        attest_known_reconciliations=True,
    )

    evidence = payload["known_reconciliation_evidence"][0]
    assert code == module.UNRESOLVED_DRIFT_EXIT
    assert evidence["exactly_one_ledger_row"] is False
    assert evidence["ledger_digest_matches_record"] is False
    assert evidence["applied_at_matches_record"] is False
    assert evidence["status"] == "not_attested"
    assert connection.execute_calls == []


@pytest.mark.asyncio
async def test_main_redacts_database_failure_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def connect_read_only() -> FakeConnection:
        raise RuntimeError("connection details must not appear in preflight output")

    monkeypatch.setattr(module, "_connect_read_only", connect_read_only)

    code = await module._main()

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert code == module.COULD_NOT_DETERMINE_EXIT
    assert payload == {
        "check_completed": False,
        "database_target": module.db_settings.target_label,
        "error_type": "RuntimeError",
        "exit_code": module.COULD_NOT_DETERMINE_EXIT,
        "status": "could_not_determine",
    }
    assert "connection details" not in output


@pytest.mark.asyncio
async def test_connection_defaults_to_read_only_without_printing_connection_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}
    sentinel = object()

    async def connect(**kwargs):
        captured_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setitem(sys.modules, "asyncpg", types.SimpleNamespace(connect=connect))

    connection = await module._connect_read_only()

    assert connection is sentinel
    server_settings = captured_kwargs["server_settings"]
    assert isinstance(server_settings, dict)
    assert server_settings["default_transaction_read_only"] == "on"


def test_main_displays_the_safe_target_without_opening_a_connection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def should_not_connect() -> FakeConnection:
        raise AssertionError("--show-target must not connect")

    monkeypatch.setattr(module, "_connect_read_only", should_not_connect)

    code = module.main(["--show-target"])

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "database_target": module.db_settings.target_label,
        "status": "target_displayed",
    }


@pytest.mark.parametrize(
    ("argv", "status"),
    [
        ([], "target_confirmation_required"),
        (["--attest-known-reconciliations"], "target_confirmation_required"),
        (["--expected-target", "other-safe-target"], "target_confirmation_mismatch"),
    ],
)
def test_main_rejects_unconfirmed_or_mismatched_target_before_connection(
    argv: list[str],
    status: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def should_not_connect() -> FakeConnection:
        raise AssertionError("target admission must happen before connection")

    monkeypatch.setattr(module, "_connect_read_only", should_not_connect)

    code = module.main(argv)

    assert code == module.COULD_NOT_DETERMINE_EXIT
    assert json.loads(capsys.readouterr().out) == {
        "check_completed": False,
        "database_target": module.db_settings.target_label,
        "exit_code": module.COULD_NOT_DETERMINE_EXIT,
        "status": status,
    }


def test_main_passes_a_matching_target_to_the_async_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, bool]] = []

    async def fake_main(
        *,
        database_target: str,
        attest_known_reconciliations: bool,
    ) -> int:
        observed.append((database_target, attest_known_reconciliations))
        return 0

    monkeypatch.setattr(module, "_main", fake_main)

    code = module.main(["--expected-target", module.db_settings.target_label])

    assert code == 0
    assert observed == [(module.db_settings.target_label, False)]


def test_main_passes_explicit_reconciliation_attestation_to_async_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, bool]] = []

    async def fake_main(
        *,
        database_target: str,
        attest_known_reconciliations: bool,
    ) -> int:
        observed.append((database_target, attest_known_reconciliations))
        return 0

    monkeypatch.setattr(module, "_main", fake_main)

    code = module.main([
        "--expected-target",
        module.db_settings.target_label,
        "--attest-known-reconciliations",
    ])

    assert code == 0
    assert observed == [(module.db_settings.target_label, True)]


def test_main_rejects_attestation_with_show_target_before_connection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def should_not_connect() -> FakeConnection:
        raise AssertionError("incompatible target modes must not connect")

    monkeypatch.setattr(module, "_connect_read_only", should_not_connect)

    with pytest.raises(SystemExit) as exc_info:
        module.main(["--show-target", "--attest-known-reconciliations"])

    assert exc_info.value.code == 2
    assert "requires --expected-target" in capsys.readouterr().err
