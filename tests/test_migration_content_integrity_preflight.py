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
        b2b_campaign_partner_reconciliation_rows: list[dict[str, object]] | None = None,
        b2b_company_signal_promotion_reconciliation_rows: list[dict[str, object]] | None = None,
        b2b_watchlist_alert_events_reconciliation_rows: list[dict[str, object]] | None = None,
        presence_unknown_count_reconciliation_rows: list[dict[str, object]] | None = None,
        public_onboarding_columns: list[dict[str, object]] | None = None,
        public_onboarding_constraints: list[dict[str, object]] | None = None,
        public_onboarding_indexes: dict[str, dict[str, object] | None] | None = None,
        presence_unknown_count_columns: list[dict[str, object]] | None = None,
        presence_events_is_ordinary_table: bool = True,
        presence_events_is_leaf_partition: bool = False,
        presence_unknown_count_has_constraint: bool = False,
        b2b_campaign_partner_catalog_row: dict[str, object] | None = None,
        b2b_company_signal_promotion_catalog_row: dict[str, object] | None = None,
        b2b_watchlist_alert_events_catalog_row: dict[str, object] | None = None,
        recurring_schema_ready: bool = True,
        zero_active_null_period_rows: bool = True,
    ):
        self.records = records
        self.reconciliation_rows = reconciliation_rows or []
        self.public_onboarding_reconciliation_rows = (
            public_onboarding_reconciliation_rows or []
        )
        self.b2b_campaign_partner_reconciliation_rows = (
            b2b_campaign_partner_reconciliation_rows or []
        )
        self.b2b_company_signal_promotion_reconciliation_rows = (
            b2b_company_signal_promotion_reconciliation_rows or []
        )
        self.b2b_watchlist_alert_events_reconciliation_rows = (
            b2b_watchlist_alert_events_reconciliation_rows or []
        )
        self.presence_unknown_count_reconciliation_rows = (
            presence_unknown_count_reconciliation_rows or []
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
        self.presence_unknown_count_columns = (
            _default_presence_unknown_count_columns()
            if presence_unknown_count_columns is None
            else presence_unknown_count_columns
        )
        self.presence_events_is_ordinary_table = presence_events_is_ordinary_table
        self.presence_events_is_leaf_partition = presence_events_is_leaf_partition
        self.presence_unknown_count_has_constraint = presence_unknown_count_has_constraint
        self.b2b_campaign_partner_catalog_row = (
            _default_b2b_campaign_partner_catalog_row()
            if b2b_campaign_partner_catalog_row is None
            else b2b_campaign_partner_catalog_row
        )
        self.b2b_company_signal_promotion_catalog_row = (
            _default_b2b_company_signal_promotion_catalog_row()
            if b2b_company_signal_promotion_catalog_row is None
            else b2b_company_signal_promotion_catalog_row
        )
        self.b2b_watchlist_alert_events_catalog_row = (
            _default_b2b_watchlist_alert_events_catalog_row()
            if b2b_watchlist_alert_events_catalog_row is None
            else b2b_watchlist_alert_events_catalog_row
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
            if args == (
                reconciliation_mod.MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION.migration_name,
            ):
                return self.public_onboarding_reconciliation_rows
            assert args == (
                reconciliation_mod.MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION.migration_name,
            )
            return self.presence_unknown_count_reconciliation_rows
        if query == (
            "SELECT version, content_sha256, applied_at FROM schema_migrations "
            "WHERE name = $1 LIMIT 2"
        ):
            if args == (
                reconciliation_mod.MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION.migration_name,
            ):
                return self.b2b_campaign_partner_reconciliation_rows
            if args == (
                reconciliation_mod.MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION.migration_name,
            ):
                return self.b2b_company_signal_promotion_reconciliation_rows
            assert args == (
                reconciliation_mod.MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION.migration_name,
            )
            return self.b2b_watchlist_alert_events_reconciliation_rows
        if "FROM information_schema.columns AS actual" in query:
            assert args == (
                list(reconciliation_mod._PUBLIC_ONBOARDING_TOKEN_COLUMNS),
            )
            return self.public_onboarding_columns
        if "eom_public_onboarding_tokens" in query:
            assert "FROM pg_constraint AS actual" in query
            assert args == (
                list(reconciliation_mod._PUBLIC_ONBOARDING_TOKEN_REQUIRED_CONSTRAINTS),
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
        if "b2b_watchlist_alert_events" in query:
            assert args == (
                list(reconciliation_mod._B2B_WATCHLIST_ALERT_EVENT_ALLOWED_COLUMNS),
                list(reconciliation_mod._B2B_WATCHLIST_ALERT_EVENT_CONSTRAINTS),
                list(reconciliation_mod._B2B_WATCHLIST_ALERT_EVENT_INDEXES),
            )
            assert "WITH target_relation AS" in query
            assert "relation_state.relpersistence" in query
            assert "JOIN pg_attribute AS attribute_state" in query
            assert "JOIN pg_constraint AS actual" in query
            assert "JOIN pg_index AS index_state" in query
            return self.b2b_watchlist_alert_events_catalog_row
        if "b2b_campaigns" in query:
            assert args == ()
            assert "WITH target_relation AS" in query
            assert "JOIN pg_attribute AS attribute_state" in query
            assert "JOIN pg_constraint AS actual" in query
            assert "JOIN pg_index AS index_state" in query
            return self.b2b_campaign_partner_catalog_row
        if "b2b_company_signals" in query:
            assert args == ()
            assert "WITH target_relation AS" in query
            assert "relation_state.relpersistence" in query
            assert "JOIN pg_attribute AS attribute_state" in query
            assert "JOIN pg_type AS type_state" in query
            assert "FROM pg_constraint AS constraint_state" in query
            assert "JOIN pg_index AS index_state" in query
            return self.b2b_company_signal_promotion_catalog_row
        if "WITH target_relation AS" in query:
            assert args == ()
            assert "relation_state.relkind" in query
            assert "AND NOT relispartition" in query
            assert "information_schema.columns AS actual" in query
            assert "FROM pg_constraint AS actual" in query
            columns = self.presence_unknown_count_columns
            column = columns[0] if len(columns) == 1 else {}
            return {
                "presence_events_is_ordinary_table": (
                    self.presence_events_is_ordinary_table
                    and not self.presence_events_is_leaf_partition
                ),
                "unknown_count_has_no_constraints": (
                    not self.presence_unknown_count_has_constraint
                ),
                "column_name": column.get("column_name"),
                "data_type": column.get("data_type"),
                "is_nullable": column.get("is_nullable"),
                "column_default": column.get("column_default"),
            }
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
            "column_default": default,
        }
        for name, (data_type, maximum_length, nullable, default) in (
            reconciliation_mod._PUBLIC_ONBOARDING_TOKEN_COLUMNS.items()
        )
    ]


def _default_presence_unknown_count_columns() -> list[dict[str, object]]:
    data_type, nullable, default = reconciliation_mod._PRESENCE_UNKNOWN_COUNT_COLUMN
    return [
        {
            "column_name": "unknown_count",
            "data_type": data_type,
            "is_nullable": nullable,
            "column_default": default,
        }
    ]


def _default_b2b_campaign_partner_catalog_row() -> dict[str, object]:
    foreign_key = reconciliation_mod._B2B_CAMPAIGN_PARTNER_FOREIGN_KEY
    index = reconciliation_mod._B2B_CAMPAIGN_PARTNER_INDEX
    return {
        "b2b_campaigns_is_ordinary_table": True,
        "partner_id_column_name": "partner_id",
        "partner_id_data_type": "uuid",
        "partner_id_is_nullable": True,
        "partner_id_has_default": False,
        "partner_foreign_key_constraint_type": foreign_key["constraint_type"],
        "partner_foreign_key_columns": list(foreign_key["key_columns"]),
        "partner_foreign_key_referenced_table": foreign_key["referenced_table"],
        "partner_foreign_key_references_current_schema": True,
        "partner_foreign_key_referenced_columns": list(
            foreign_key["referenced_columns"]
        ),
        "partner_foreign_key_delete_action": foreign_key["delete_action"],
        "partner_foreign_key_update_action": foreign_key["update_action"],
        "partner_foreign_key_match_type": foreign_key["match_type"],
        "partner_foreign_key_is_deferrable": False,
        "partner_foreign_key_is_initially_deferred": False,
        "partner_foreign_key_is_validated": True,
        "partner_index_relation_kind": "i",
        "partner_index_is_partition": False,
        "partner_index_is_unique": False,
        "partner_index_is_valid": True,
        "partner_index_is_ready": True,
        "partner_index_key_attribute_count": 1,
        "partner_index_attribute_count": 1,
        "partner_index_key_column": index["key_column"],
        "partner_index_predicate": "(partner_id IS NOT NULL)",
    }


def _default_b2b_company_signal_promotion_catalog_row() -> dict[str, object]:
    index = reconciliation_mod._B2B_COMPANY_SIGNAL_PROMOTION_INDEX
    return {
        "b2b_company_signals_is_ordinary_table": True,
        "canonical_promotion_type_column_name": "canonical_promotion_type",
        "canonical_promotion_type_data_type": "text",
        "canonical_promotion_type_is_nullable": True,
        "canonical_promotion_type_has_default": False,
        "canonical_promotion_type_is_generated": False,
        "canonical_promotion_type_is_identity": False,
        "canonical_promotion_type_uses_type_default_collation": True,
        "canonical_promotion_type_has_no_constraints": True,
        "promotion_index_relation_kind": "i",
        "promotion_index_is_partition": False,
        "promotion_index_is_unique": False,
        "promotion_index_is_valid": True,
        "promotion_index_is_ready": True,
        "promotion_index_key_attribute_count": 1,
        "promotion_index_attribute_count": 1,
        "promotion_index_key_column": index["key_column"],
        "promotion_index_predicate": "(canonical_promotion_type IS NOT NULL)",
    }


def _default_b2b_watchlist_alert_events_catalog_row() -> dict[str, object]:
    """Return one complete metadata-only 272 catalog evidence fixture."""
    return {
        "catalog_evidence": {
            "watchlist_alert_events_is_ordinary_table": True,
            "watchlist_alert_events_has_permanent_storage": True,
            "columns": {
                name: {
                    "exists": True,
                    "data_type": data_type,
                    "is_nullable": is_nullable,
                    "is_generated": False,
                    "is_identity": False,
                    "uses_type_default_collation": True,
                    "column_default": default,
                }
                for name, (data_type, is_nullable, default) in (
                    reconciliation_mod._B2B_WATCHLIST_ALERT_EVENT_ALLOWED_COLUMNS.items()
                )
            },
            "no_unlisted_alert_event_columns": True,
            "constraints": {
                name: {
                    "constraint_type": constraint.constraint_type,
                    "key_columns": list(constraint.key_columns),
                    "referenced_table": constraint.referenced_table,
                    "references_current_schema": (
                        constraint.referenced_table is not None
                    ),
                    "referenced_columns": list(constraint.referenced_columns),
                    "delete_action": constraint.delete_action,
                    "update_action": constraint.update_action,
                    "match_type": constraint.match_type,
                    "is_deferrable": False,
                    "is_initially_deferred": False,
                    "is_validated": True,
                    "internal_trigger_count": (
                        constraint.expected_internal_trigger_count
                    ),
                    "origin_enabled_internal_trigger_count": (
                        constraint.expected_internal_trigger_count
                    ),
                    "expression": constraint.expression,
                }
                for name, constraint in (
                    reconciliation_mod._B2B_WATCHLIST_ALERT_EVENT_CONSTRAINTS.items()
                )
            },
            "no_unlisted_alert_event_constraints": True,
            "indexes": {
                name: {
                    "relation_kind": "i",
                    "is_partition": False,
                    "is_unique": index.unique,
                    "is_valid": True,
                    "is_ready": True,
                    "key_attribute_count": len(index.key_columns),
                    "attribute_count": len(index.key_columns),
                    "key_columns": list(index.key_columns),
                    "definition": index.definition_fragment,
                    "predicate": index.predicate,
                }
                for name, index in (
                    reconciliation_mod._B2B_WATCHLIST_ALERT_EVENT_INDEXES.items()
                )
            },
            "no_unlisted_alert_event_indexes": True,
            "no_unreviewed_alert_event_write_interceptors": True,
        }
    }


def _default_public_onboarding_constraints() -> list[dict[str, object]]:
    return [
        {
            "conname": name,
            "constraint_type": constraint.constraint_type,
            "key_columns": list(constraint.key_columns),
            "referenced_table": constraint.referenced_table,
            "references_current_schema": constraint.referenced_table is not None,
            "referenced_columns": list(constraint.referenced_columns),
            "delete_action": constraint.delete_action or " ",
            "update_action": constraint.update_action or " ",
            "match_type": constraint.match_type or " ",
            "is_deferrable": False,
            "is_initially_deferred": False,
            "is_validated": True,
            "expression": constraint.expression,
        }
        for name, constraint in (
            reconciliation_mod._PUBLIC_ONBOARDING_TOKEN_REQUIRED_CONSTRAINTS.items()
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


def _migration_022b_source() -> bytes:
    return (
        ROOT
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "027_presence_unknown_count.sql"
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


def _migration_022b_connection(
    *,
    ledger_digest: str | None = None,
    applied_at: object | None = None,
    reconciliation_rows: list[dict[str, object]] | None = None,
) -> FakeConnection:
    record = reconciliation_mod.MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION
    actual_applied_at = record.observed_applied_at if applied_at is None else applied_at
    actual_rows = reconciliation_rows
    if actual_rows is None:
        actual_rows = [
            {
                "content_sha256": ledger_digest,
                "applied_at": actual_applied_at,
            }
        ]
    return FakeConnection(
        [(record.migration_name, ledger_digest)],
        presence_unknown_count_reconciliation_rows=actual_rows,
    )


def _migration_067_connection(
    *,
    ledger_digest: str | None = None,
    applied_at: object | None = None,
    reconciliation_rows: list[dict[str, object]] | None = None,
) -> FakeConnection:
    record = reconciliation_mod.MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION
    actual_applied_at = record.observed_applied_at if applied_at is None else applied_at
    actual_rows = reconciliation_rows
    if actual_rows is None:
        actual_rows = [{
            "version": record.migration_version,
            "content_sha256": ledger_digest,
            "applied_at": actual_applied_at,
        }]
    return FakeConnection(
        [(record.migration_name, ledger_digest)],
        b2b_campaign_partner_reconciliation_rows=actual_rows,
    )


def _migration_297_connection(
    *,
    ledger_digest: str | None = None,
    applied_at: object | None = None,
    reconciliation_rows: list[dict[str, object]] | None = None,
) -> FakeConnection:
    record = (
        reconciliation_mod.MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION
    )
    actual_applied_at = record.observed_applied_at if applied_at is None else applied_at
    actual_rows = reconciliation_rows
    if actual_rows is None:
        actual_rows = [{
            "version": record.migration_version,
            "content_sha256": ledger_digest,
            "applied_at": actual_applied_at,
        }]
    return FakeConnection(
        [(record.migration_name, ledger_digest)],
        b2b_company_signal_promotion_reconciliation_rows=actual_rows,
    )


def _migration_272_connection(
    *,
    ledger_digest: str | None = None,
    applied_at: object | None = None,
    reconciliation_rows: list[dict[str, object]] | None = None,
) -> FakeConnection:
    record = reconciliation_mod.MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION
    actual_applied_at = record.observed_applied_at if applied_at is None else applied_at
    actual_rows = reconciliation_rows
    if actual_rows is None:
        actual_rows = [{
            "version": record.migration_version,
            "content_sha256": ledger_digest,
            "applied_at": actual_applied_at,
        }]
    return FakeConnection(
        [(record.migration_name, ledger_digest)],
        b2b_watchlist_alert_events_reconciliation_rows=actual_rows,
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
        reconciliation_mod.MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION.migration_name,
        reconciliation_mod.MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION.migration_name,
        reconciliation_mod.MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION.migration_name,
        reconciliation_mod.MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION.migration_name,
    }
    assert reconciliation_mod.known_historical_reconciliation_names() == {
        record.migration_name,
        reconciliation_mod.MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION.migration_name,
        reconciliation_mod.MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION.migration_name,
        reconciliation_mod.MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION.migration_name,
        reconciliation_mod.MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION.migration_name,
        reconciliation_mod.MIGRATION_387_RECONCILIATION.migration_name,
    }


def test_migration_067_reconciliation_record_is_closed_target_evidence() -> None:
    record = reconciliation_mod.MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION

    assert record.source_verification == reconciliation_mod.HISTORICAL_SOURCE_UNAVAILABLE
    assert record.migration_version == 67
    assert record.historical_ledger_sha256 is None
    assert record.observed_applied_at == datetime(
        2026,
        3,
        1,
        4,
        58,
        0,
        789_236,
        tzinfo=timezone.utc,
    )


def test_migration_297_reconciliation_record_is_closed_target_evidence() -> None:
    record = (
        reconciliation_mod.MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION
    )

    assert record.source_verification == reconciliation_mod.HISTORICAL_SOURCE_UNAVAILABLE
    assert record.migration_version == 297
    assert record.historical_ledger_sha256 is None
    assert record.observed_applied_at == datetime(
        2026,
        4,
        12,
        19,
        28,
        13,
        742_305,
        tzinfo=timezone.utc,
    )


def test_migration_272_reconciliation_record_is_closed_target_evidence() -> None:
    record = reconciliation_mod.MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION

    assert record.source_verification == reconciliation_mod.HISTORICAL_SOURCE_UNAVAILABLE
    assert record.migration_version == -3
    assert record.historical_ledger_sha256 is None
    assert record.observed_applied_at == datetime(
        2026,
        4,
        8,
        3,
        34,
        24,
        452_014,
        tzinfo=timezone.utc,
    )


def test_migration_022b_reconciliation_record_preserves_renamed_source_limits() -> None:
    record = reconciliation_mod.MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION

    assert (
        record.source_verification
        == reconciliation_mod.HISTORICAL_LEDGER_DIGEST_UNAVAILABLE
    )
    assert record.historical_ledger_sha256 is None
    assert record.current_packaged_migration_name == "027_presence_unknown_count"
    assert (
        hashlib.sha256(_migration_022b_source()).hexdigest()
        == record.retained_source_sha256
    )
    assert record.observed_applied_at == datetime(
        2026,
        2,
        17,
        23,
        34,
        17,
        949_845,
        tzinfo=timezone.utc,
    )
    assert record.retained_source_history_commit_ids == (
        "72c008b40d134bf1e0432e5c586bf0e156a1780b",
        "5df3f12fa9aa1721983b37cceca913803db27722",
        "2ec15ce4d8f159dd773405dfb311da4219d531aa",
    )


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
        "base_token_contract_ready": True,
        "required_constraints_ready": True,
        "complete_token_schema_ready": True,
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
        ("missing ledger row", "exactly_one_ledger_row"),
        ("missing immutable column", "immutable_projection_ready"),
        ("missing original column", "base_token_contract_ready"),
        ("missing status default", "base_token_contract_ready"),
        ("missing required constraint", "required_constraints_ready"),
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
    elif case == "missing ledger row":
        connection = _migration_382_connection(reconciliation_rows=[])
    elif case == "missing immutable column":
        connection.public_onboarding_columns = connection.public_onboarding_columns[1:]
    elif case == "missing original column":
        connection.public_onboarding_columns = [
            row
            for row in connection.public_onboarding_columns
            if row["column_name"] != "approval_key"
        ]
    elif case == "missing status default":
        for row in connection.public_onboarding_columns:
            if row["column_name"] == "status":
                row["column_default"] = None
    elif case == "missing required constraint":
        connection.public_onboarding_constraints = [
            row
            for row in connection.public_onboarding_constraints
            if row["conname"] != "eom_public_onboarding_tokens_contact_id_fkey"
        ]
    elif case == "weakened fingerprint check":
        for row in connection.public_onboarding_constraints:
            if row["conname"] == (
                "eom_public_onboarding_tokens_signing_key_fingerprint_check"
            ):
                row["expression"] = (
                    "((signing_key_fingerprint)::text ~ '^[0-9a-f]{63}$'::text)"
                )
    elif case == "weakened terminal-state check":
        for row in connection.public_onboarding_constraints:
            if row["conname"] == "ck_eom_public_onboarding_tokens_terminal_state":
                row["expression"] = "(status = 'issued')"
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
    assert evidence["complete_token_schema_ready"] is False or field in {
        "ledger_digest_is_null",
        "applied_at_matches_record",
        "exactly_one_ledger_row",
    }, case
    assert evidence["status"] == "not_attested", case
    assert connection.execute_calls == []


@pytest.mark.asyncio
async def test_known_067_reconciliation_attests_catalog_without_source(
    tmp_path: Path,
) -> None:
    record = reconciliation_mod.MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION
    connection = _migration_067_connection()

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
        "ledger_version_matches_record": True,
        "ledger_digest_is_null": True,
        "applied_at_matches_record": True,
        "b2b_campaigns_is_ordinary_table": True,
        "partner_id_column_ready": True,
        "partner_foreign_key_ready": True,
        "partner_partial_index_ready": True,
        "status": "attested",
    }]
    assert connection.fetch_calls[0] == (
        "SELECT name, content_sha256 FROM schema_migrations",
        (),
    )
    assert connection.fetch_calls[1] == (
        "SELECT version, content_sha256, applied_at FROM schema_migrations "
        "WHERE name = $1 LIMIT 2",
        (record.migration_name,),
    )
    assert len(connection.fetch_calls) == 2
    assert len(connection.fetchrow_calls) == 1
    assert connection.fetchval_calls == []
    catalog_query = connection.fetchrow_calls[0][0]
    assert "WITH target_relation AS" in catalog_query
    assert "JOIN pg_attribute AS attribute_state" in catalog_query
    assert "JOIN pg_constraint AS actual" in catalog_query
    assert "JOIN pg_index AS index_state" in catalog_query
    assert all(
        "FROM b2b_campaigns" not in query
        for query, _args in (
            connection.fetch_calls
            + connection.fetchrow_calls
            + connection.fetchval_calls
        )
    )
    assert connection.transaction_readonly == [True]
    assert connection.execute_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "field"),
    [
        ("wrong ledger version", "ledger_version_matches_record"),
        ("non-null ledger digest", "ledger_digest_is_null"),
        ("truncated applied timestamp", "applied_at_matches_record"),
        ("duplicate ledger rows", "exactly_one_ledger_row"),
        ("missing detailed ledger row", "exactly_one_ledger_row"),
        ("campaign relation is not an ordinary table", "b2b_campaigns_is_ordinary_table"),
        ("partner id has the wrong type", "partner_id_column_ready"),
        ("partner id is not nullable", "partner_id_column_ready"),
        ("partner id has a default", "partner_id_column_ready"),
        ("wrong partner foreign-key parent", "partner_foreign_key_ready"),
        ("wrong partner foreign-key delete action", "partner_foreign_key_ready"),
        ("missing partner index", "partner_partial_index_ready"),
        ("partner index is unready", "partner_partial_index_ready"),
        ("partner index predicate changed", "partner_partial_index_ready"),
    ],
)
async def test_known_067_reconciliation_rejects_each_required_evidence_field(
    case: str,
    field: str,
    tmp_path: Path,
) -> None:
    record = reconciliation_mod.MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION
    connection = _migration_067_connection()
    if case == "wrong ledger version":
        connection = _migration_067_connection(reconciliation_rows=[{
            "version": record.migration_version + 1,
            "content_sha256": None,
            "applied_at": record.observed_applied_at,
        }])
    elif case == "non-null ledger digest":
        connection = _migration_067_connection(ledger_digest="a" * 64)
    elif case == "truncated applied timestamp":
        connection = _migration_067_connection(
            applied_at=record.observed_applied_at.replace(microsecond=0)
        )
    elif case == "duplicate ledger rows":
        connection = _migration_067_connection(reconciliation_rows=[
            {
                "version": record.migration_version,
                "content_sha256": None,
                "applied_at": record.observed_applied_at,
            },
            {
                "version": record.migration_version,
                "content_sha256": None,
                "applied_at": record.observed_applied_at,
            },
        ])
    elif case == "missing detailed ledger row":
        connection = _migration_067_connection(reconciliation_rows=[])
    elif case == "campaign relation is not an ordinary table":
        connection.b2b_campaign_partner_catalog_row[
            "b2b_campaigns_is_ordinary_table"
        ] = False
    elif case == "partner id has the wrong type":
        connection.b2b_campaign_partner_catalog_row["partner_id_data_type"] = "text"
    elif case == "partner id is not nullable":
        connection.b2b_campaign_partner_catalog_row["partner_id_is_nullable"] = False
    elif case == "partner id has a default":
        connection.b2b_campaign_partner_catalog_row["partner_id_has_default"] = True
    elif case == "wrong partner foreign-key parent":
        connection.b2b_campaign_partner_catalog_row[
            "partner_foreign_key_referenced_table"
        ] = "wrong_partners"
    elif case == "wrong partner foreign-key delete action":
        connection.b2b_campaign_partner_catalog_row[
            "partner_foreign_key_delete_action"
        ] = "a"
    elif case == "missing partner index":
        connection.b2b_campaign_partner_catalog_row[
            "partner_index_relation_kind"
        ] = None
    elif case == "partner index is unready":
        connection.b2b_campaign_partner_catalog_row["partner_index_is_ready"] = False
    elif case == "partner index predicate changed":
        connection.b2b_campaign_partner_catalog_row[
            "partner_index_predicate"
        ] = "(partner_id IS NULL)"
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
async def test_known_297_reconciliation_attests_catalog_without_source(
    tmp_path: Path,
) -> None:
    record = (
        reconciliation_mod.MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION
    )
    connection = _migration_297_connection()

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
        "ledger_version_matches_record": True,
        "ledger_digest_is_null": True,
        "applied_at_matches_record": True,
        "b2b_company_signals_is_ordinary_table": True,
        "canonical_promotion_type_column_ready": True,
        "canonical_promotion_type_has_no_constraints": True,
        "canonical_promotion_type_partial_index_ready": True,
        "status": "attested",
    }]
    assert connection.fetch_calls == [
        ("SELECT name, content_sha256 FROM schema_migrations", ()),
        (
            "SELECT version, content_sha256, applied_at FROM schema_migrations "
            "WHERE name = $1 LIMIT 2",
            (record.migration_name,),
        ),
    ]
    assert len(connection.fetchrow_calls) == 1
    assert connection.fetchval_calls == []
    catalog_query = connection.fetchrow_calls[0][0]
    assert "WITH target_relation AS" in catalog_query
    assert "relation_state.relpersistence" in catalog_query
    assert "JOIN pg_attribute AS attribute_state" in catalog_query
    assert "JOIN pg_type AS type_state" in catalog_query
    assert "attribute_state.attgenerated <> ''::\"char\"" in catalog_query
    assert "attribute_state.attidentity <> ''::\"char\"" in catalog_query
    assert "FROM pg_constraint AS constraint_state" in catalog_query
    assert "pg_get_constraintdef" in catalog_query
    assert "JOIN pg_index AS index_state" in catalog_query
    assert "pg_get_indexdef" in catalog_query
    assert all(
        "FROM b2b_company_signals" not in query
        for query, _args in (
            connection.fetch_calls
            + connection.fetchrow_calls
            + connection.fetchval_calls
        )
    )
    assert connection.transaction_readonly == [True]
    assert connection.execute_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "field"),
    [
        ("wrong ledger version", "ledger_version_matches_record"),
        ("non-null ledger digest", "ledger_digest_is_null"),
        ("truncated applied timestamp", "applied_at_matches_record"),
        ("duplicate ledger rows", "exactly_one_ledger_row"),
        ("missing detailed ledger row", "exactly_one_ledger_row"),
        (
            "company-signal relation is not a permanent ordinary table",
            "b2b_company_signals_is_ordinary_table",
        ),
        ("promotion column has the wrong type", "canonical_promotion_type_column_ready"),
        ("promotion column is not nullable", "canonical_promotion_type_column_ready"),
        ("promotion column has a default", "canonical_promotion_type_column_ready"),
        ("promotion column is generated", "canonical_promotion_type_column_ready"),
        ("promotion column is identity", "canonical_promotion_type_column_ready"),
        (
            "promotion column has nondefault collation",
            "canonical_promotion_type_column_ready",
        ),
        (
            "promotion column has a constraint",
            "canonical_promotion_type_has_no_constraints",
        ),
        (
            "missing promotion partial index",
            "canonical_promotion_type_partial_index_ready",
        ),
        (
            "promotion partial index is unready",
            "canonical_promotion_type_partial_index_ready",
        ),
        (
            "promotion partial index is unique",
            "canonical_promotion_type_partial_index_ready",
        ),
        (
            "promotion partial index has extra attributes",
            "canonical_promotion_type_partial_index_ready",
        ),
        (
            "promotion partial index has wrong key",
            "canonical_promotion_type_partial_index_ready",
        ),
        (
            "promotion partial index predicate changed",
            "canonical_promotion_type_partial_index_ready",
        ),
    ],
)
async def test_known_297_reconciliation_rejects_each_required_evidence_field(
    case: str,
    field: str,
    tmp_path: Path,
) -> None:
    record = (
        reconciliation_mod.MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION
    )
    connection = _migration_297_connection()
    catalog = connection.b2b_company_signal_promotion_catalog_row
    if case == "wrong ledger version":
        connection = _migration_297_connection(reconciliation_rows=[{
            "version": record.migration_version + 1,
            "content_sha256": None,
            "applied_at": record.observed_applied_at,
        }])
    elif case == "non-null ledger digest":
        connection = _migration_297_connection(ledger_digest="a" * 64)
    elif case == "truncated applied timestamp":
        connection = _migration_297_connection(
            applied_at=record.observed_applied_at.replace(microsecond=0)
        )
    elif case == "duplicate ledger rows":
        connection = _migration_297_connection(reconciliation_rows=[
            {
                "version": record.migration_version,
                "content_sha256": None,
                "applied_at": record.observed_applied_at,
            },
            {
                "version": record.migration_version,
                "content_sha256": None,
                "applied_at": record.observed_applied_at,
            },
        ])
    elif case == "missing detailed ledger row":
        connection = _migration_297_connection(reconciliation_rows=[])
    elif case == "company-signal relation is not a permanent ordinary table":
        catalog["b2b_company_signals_is_ordinary_table"] = False
    elif case == "promotion column has the wrong type":
        catalog["canonical_promotion_type_data_type"] = "character varying"
    elif case == "promotion column is not nullable":
        catalog["canonical_promotion_type_is_nullable"] = False
    elif case == "promotion column has a default":
        catalog["canonical_promotion_type_has_default"] = True
    elif case == "promotion column is generated":
        catalog["canonical_promotion_type_is_generated"] = True
    elif case == "promotion column is identity":
        catalog["canonical_promotion_type_is_identity"] = True
    elif case == "promotion column has nondefault collation":
        catalog["canonical_promotion_type_uses_type_default_collation"] = False
    elif case == "promotion column has a constraint":
        catalog["canonical_promotion_type_has_no_constraints"] = False
    elif case == "missing promotion partial index":
        catalog["promotion_index_relation_kind"] = None
    elif case == "promotion partial index is unready":
        catalog["promotion_index_is_ready"] = False
    elif case == "promotion partial index is unique":
        catalog["promotion_index_is_unique"] = True
    elif case == "promotion partial index has extra attributes":
        catalog["promotion_index_attribute_count"] = 2
    elif case == "promotion partial index has wrong key":
        catalog["promotion_index_key_column"] = "other_column"
    elif case == "promotion partial index predicate changed":
        catalog["promotion_index_predicate"] = "(canonical_promotion_type IS NULL)"
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
    assert connection.transaction_readonly == [True]
    assert connection.execute_calls == []


@pytest.mark.asyncio
async def test_known_272_reconciliation_attests_catalog_without_source(
    tmp_path: Path,
) -> None:
    record = reconciliation_mod.MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION
    connection = _migration_272_connection()

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
        "ledger_version_matches_record": True,
        "ledger_digest_is_null": True,
        "applied_at_matches_record": True,
        "watchlist_alert_events_is_ordinary_table": True,
        "watchlist_alert_events_has_permanent_storage": True,
        "base_alert_event_columns_ready": True,
        "known_later_alert_event_columns_ready": True,
        "no_unlisted_alert_event_columns": True,
        "required_alert_event_constraints_ready": True,
        "no_unlisted_alert_event_constraints": True,
        "required_alert_event_indexes_ready": True,
        "no_unlisted_alert_event_indexes": True,
        "no_unreviewed_alert_event_write_interceptors": True,
        "status": "attested",
    }]
    assert connection.fetch_calls == [
        ("SELECT name, content_sha256 FROM schema_migrations", ()),
        (
            "SELECT version, content_sha256, applied_at FROM schema_migrations "
            "WHERE name = $1 LIMIT 2",
            (record.migration_name,),
        ),
    ]
    assert len(connection.fetchrow_calls) == 1
    assert connection.fetchval_calls == []
    catalog_query = connection.fetchrow_calls[0][0]
    assert "WITH target_relation AS" in catalog_query
    assert "relation_state.relpersistence" in catalog_query
    assert "JOIN pg_attribute AS attribute_state" in catalog_query
    assert "JOIN pg_type AS type_state" in catalog_query
    assert "attribute_state.attcollation = type_state.typcollation" in catalog_query
    assert "JOIN pg_attrdef AS default_state" in catalog_query
    assert "unlisted_columns AS" in catalog_query
    assert "JOIN pg_constraint AS actual" in catalog_query
    assert "unlisted_constraints AS" in catalog_query
    assert "constraint_trigger.tgconstraint = actual.oid" in catalog_query
    assert "constraint_trigger.tgisinternal" in catalog_query
    assert "constraint_trigger.tgenabled = 'O'::\"char\"" in catalog_query
    assert "JOIN pg_index AS index_state" in catalog_query
    assert "unlisted_indexes AS" in catalog_query
    assert "FROM unnest(index_state.indkey)" in catalog_query
    assert "WITH ORDINALITY AS key_state(attnum, ordinality)" in catalog_query
    assert "index_state.indisunique OR index_state.indisexclusion" not in catalog_query
    assert "unreviewed_write_interceptors AS" in catalog_query
    assert "FROM pg_trigger AS trigger_state" in catalog_query
    assert "FROM pg_rewrite AS rule_state" in catalog_query
    assert "FROM pg_policy AS policy_state" in catalog_query
    assert "relation_state.relrowsecurity" in catalog_query
    assert "relation_state.relforcerowsecurity" in catalog_query
    assert "pg_get_indexdef" in catalog_query
    assert all(
        "FROM b2b_watchlist_alert_events" not in query
        for query, _args in (
            connection.fetch_calls
            + connection.fetchrow_calls
            + connection.fetchval_calls
        )
    )
    assert connection.transaction_readonly == [True]
    assert connection.execute_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "field"),
    [
        ("wrong ledger version", "ledger_version_matches_record"),
        ("non-null ledger digest", "ledger_digest_is_null"),
        ("truncated applied timestamp", "applied_at_matches_record"),
        ("duplicate ledger rows", "exactly_one_ledger_row"),
        ("missing detailed ledger row", "exactly_one_ledger_row"),
        (
            "watchlist relation is not an ordinary table",
            "watchlist_alert_events_is_ordinary_table",
        ),
        ("missing source-era column", "base_alert_event_columns_ready"),
        ("wrong source-era column default", "base_alert_event_columns_ready"),
        (
            "missing known later column",
            "known_later_alert_event_columns_ready",
        ),
        ("missing named constraint", "required_alert_event_constraints_ready"),
        ("wrong foreign-key action", "required_alert_event_constraints_ready"),
        ("weakened check constraint", "required_alert_event_constraints_ready"),
        ("missing named index", "required_alert_event_indexes_ready"),
        ("unready named index", "required_alert_event_indexes_ready"),
        ("wrong index direction", "required_alert_event_indexes_ready"),
    ],
)
async def test_known_272_reconciliation_rejects_each_required_evidence_field(
    case: str,
    field: str,
    tmp_path: Path,
) -> None:
    record = reconciliation_mod.MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION
    connection = _migration_272_connection()
    catalog = connection.b2b_watchlist_alert_events_catalog_row["catalog_evidence"]
    columns = catalog["columns"]
    constraints = catalog["constraints"]
    indexes = catalog["indexes"]
    if case == "wrong ledger version":
        connection = _migration_272_connection(reconciliation_rows=[{
            "version": record.migration_version - 1,
            "content_sha256": None,
            "applied_at": record.observed_applied_at,
        }])
    elif case == "non-null ledger digest":
        connection = _migration_272_connection(ledger_digest="a" * 64)
    elif case == "truncated applied timestamp":
        connection = _migration_272_connection(
            applied_at=record.observed_applied_at.replace(microsecond=0)
        )
    elif case == "duplicate ledger rows":
        connection = _migration_272_connection(reconciliation_rows=[
            {
                "version": record.migration_version,
                "content_sha256": None,
                "applied_at": record.observed_applied_at,
            },
            {
                "version": record.migration_version,
                "content_sha256": None,
                "applied_at": record.observed_applied_at,
            },
        ])
    elif case == "missing detailed ledger row":
        connection = _migration_272_connection(reconciliation_rows=[])
    elif case == "watchlist relation is not an ordinary table":
        catalog["watchlist_alert_events_is_ordinary_table"] = False
    elif case == "missing source-era column":
        columns.pop("payload")
    elif case == "wrong source-era column default":
        columns["status"]["column_default"] = "closed"
    elif case == "missing known later column":
        columns.pop("reopen_count")
    elif case == "missing named constraint":
        constraints.pop("chk_b2b_watchlist_alert_events_status")
    elif case == "wrong foreign-key action":
        constraints["b2b_watchlist_alert_events_account_id_fkey"][
            "delete_action"
        ] = "a"
    elif case == "weakened check constraint":
        constraints["chk_b2b_watchlist_alert_events_status"][
            "expression"
        ] = "(status = any (array[open]))"
    elif case == "missing named index":
        indexes.pop("idx_b2b_watchlist_alert_events_view_entity")
    elif case == "unready named index":
        indexes["idx_b2b_watchlist_alert_events_account_status"]["is_ready"] = False
    elif case == "wrong index direction":
        indexes["idx_b2b_watchlist_alert_events_view_status"]["definition"] = (
            "usingbtree(watchlist_view_id,status,last_seen_at)"
        )
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
@pytest.mark.parametrize(
    ("case", "field"),
    [
        (
            "ordinary relation has the wrong kind",
            "watchlist_alert_events_is_ordinary_table",
        ),
        (
            "ordinary table is not permanent",
            "watchlist_alert_events_has_permanent_storage",
        ),
        ("column is absent", "base_alert_event_columns_ready"),
        ("column has the wrong type", "base_alert_event_columns_ready"),
        ("column has the wrong nullability", "base_alert_event_columns_ready"),
        ("column has the wrong default", "base_alert_event_columns_ready"),
        (
            "source-era status column is generated",
            "base_alert_event_columns_ready",
        ),
        (
            "source-era payload column is generated",
            "base_alert_event_columns_ready",
        ),
        (
            "source-era account ID column is identity",
            "base_alert_event_columns_ready",
        ),
        (
            "source-era summary column is identity",
            "base_alert_event_columns_ready",
        ),
        (
            "known later column is generated",
            "known_later_alert_event_columns_ready",
        ),
        (
            "known later column is identity",
            "known_later_alert_event_columns_ready",
        ),
        (
            "source-era entity key has a non-default collation",
            "base_alert_event_columns_ready",
        ),
        (
            "source-era vendor name has a non-default collation",
            "base_alert_event_columns_ready",
        ),
        (
            "source-era status has a non-default collation",
            "base_alert_event_columns_ready",
        ),
        (
            "source-era summary has a non-default collation",
            "base_alert_event_columns_ready",
        ),
        (
            "source-era company name has a non-default collation",
            "base_alert_event_columns_ready",
        ),
        (
            "known later column is absent",
            "known_later_alert_event_columns_ready",
        ),
        (
            "known later column has the wrong type",
            "known_later_alert_event_columns_ready",
        ),
        (
            "known later column has the wrong nullability",
            "known_later_alert_event_columns_ready",
        ),
        (
            "known later column has the wrong default",
            "known_later_alert_event_columns_ready",
        ),
        (
            "unlisted required column can reject writes",
            "no_unlisted_alert_event_columns",
        ),
        ("constraint is absent", "required_alert_event_constraints_ready"),
        ("constraint has the wrong type", "required_alert_event_constraints_ready"),
        ("constraint has the wrong key columns", "required_alert_event_constraints_ready"),
        (
            "foreign key has the wrong referenced table",
            "required_alert_event_constraints_ready",
        ),
        (
            "foreign key has the wrong referenced schema",
            "required_alert_event_constraints_ready",
        ),
        (
            "foreign key has the wrong referenced columns",
            "required_alert_event_constraints_ready",
        ),
        (
            "foreign key has the wrong delete action",
            "required_alert_event_constraints_ready",
        ),
        (
            "foreign key has the wrong update action",
            "required_alert_event_constraints_ready",
        ),
        (
            "foreign key has the wrong match type",
            "required_alert_event_constraints_ready",
        ),
        (
            "constraint is deferrable",
            "required_alert_event_constraints_ready",
        ),
        (
            "constraint is initially deferred",
            "required_alert_event_constraints_ready",
        ),
        (
            "constraint is unvalidated",
            "required_alert_event_constraints_ready",
        ),
        (
            "foreign key has a missing internal enforcement trigger",
            "required_alert_event_constraints_ready",
        ),
        (
            "foreign key has a disabled internal enforcement trigger",
            "required_alert_event_constraints_ready",
        ),
        (
            "check constraint is weakened",
            "required_alert_event_constraints_ready",
        ),
        (
            "check constraint has a literal collision",
            "required_alert_event_constraints_ready",
        ),
        (
            "check constraint has a literal case collision",
            "required_alert_event_constraints_ready",
        ),
        (
            "unlisted constraint can reject writes",
            "no_unlisted_alert_event_constraints",
        ),
        ("index is absent", "required_alert_event_indexes_ready"),
        ("index has the wrong relation kind", "required_alert_event_indexes_ready"),
        ("index is a partition", "required_alert_event_indexes_ready"),
        ("index has the wrong uniqueness", "required_alert_event_indexes_ready"),
        ("index is invalid", "required_alert_event_indexes_ready"),
        ("index is unready", "required_alert_event_indexes_ready"),
        (
            "index has the wrong key attribute count",
            "required_alert_event_indexes_ready",
        ),
        (
            "index has the wrong attribute count",
            "required_alert_event_indexes_ready",
        ),
        ("index has the wrong key columns", "required_alert_event_indexes_ready"),
        (
            "index definition has the wrong direction",
            "required_alert_event_indexes_ready",
        ),
        ("index has an unexpected predicate", "required_alert_event_indexes_ready"),
        (
            "unlisted index can reject writes",
            "no_unlisted_alert_event_indexes",
        ),
        (
            "unreviewed write interceptor can reject writes",
            "no_unreviewed_alert_event_write_interceptors",
        ),
    ],
)
async def test_known_272_reconciliation_rejects_each_catalog_guard_independently(
    case: str,
    field: str,
    tmp_path: Path,
) -> None:
    """One mutated catalog value must close exactly its named receipt group."""
    connection = _migration_272_connection()
    catalog = connection.b2b_watchlist_alert_events_catalog_row["catalog_evidence"]
    columns = catalog["columns"]
    constraints = catalog["constraints"]
    indexes = catalog["indexes"]
    foreign_key = constraints["b2b_watchlist_alert_events_account_id_fkey"]
    status_check = constraints["chk_b2b_watchlist_alert_events_status"]
    account_status_index = indexes["idx_b2b_watchlist_alert_events_account_status"]

    if case == "ordinary relation has the wrong kind":
        catalog["watchlist_alert_events_is_ordinary_table"] = False
    elif case == "ordinary table is not permanent":
        catalog["watchlist_alert_events_has_permanent_storage"] = False
    elif case == "column is absent":
        columns["payload"]["exists"] = False
    elif case == "column has the wrong type":
        columns["status"]["data_type"] = "character varying"
    elif case == "column has the wrong nullability":
        columns["status"]["is_nullable"] = True
    elif case == "column has the wrong default":
        columns["status"]["column_default"] = "'closed'"
    elif case == "source-era status column is generated":
        columns["status"]["is_generated"] = True
    elif case == "source-era payload column is generated":
        columns["payload"]["is_generated"] = True
    elif case == "source-era account ID column is identity":
        columns["account_id"]["is_identity"] = True
    elif case == "source-era summary column is identity":
        columns["summary"]["is_identity"] = True
    elif case == "known later column is generated":
        columns["reopen_count"]["is_generated"] = True
    elif case == "known later column is identity":
        columns["reopen_count"]["is_identity"] = True
    elif case == "source-era entity key has a non-default collation":
        columns["entity_key"]["uses_type_default_collation"] = False
    elif case == "source-era vendor name has a non-default collation":
        columns["vendor_name"]["uses_type_default_collation"] = False
    elif case == "source-era status has a non-default collation":
        columns["status"]["uses_type_default_collation"] = False
    elif case == "source-era summary has a non-default collation":
        columns["summary"]["uses_type_default_collation"] = False
    elif case == "source-era company name has a non-default collation":
        columns["company_name"]["uses_type_default_collation"] = False
    elif case == "known later column is absent":
        columns["reopen_count"]["exists"] = False
    elif case == "known later column has the wrong type":
        columns["reopen_count"]["data_type"] = "bigint"
    elif case == "known later column has the wrong nullability":
        columns["reopen_count"]["is_nullable"] = True
    elif case == "known later column has the wrong default":
        columns["reopen_count"]["column_default"] = None
    elif case == "unlisted required column can reject writes":
        catalog["no_unlisted_alert_event_columns"] = False
    elif case == "constraint is absent":
        constraints.pop("b2b_watchlist_alert_events_pkey")
    elif case == "constraint has the wrong type":
        constraints["b2b_watchlist_alert_events_pkey"]["constraint_type"] = "u"
    elif case == "constraint has the wrong key columns":
        constraints["b2b_watchlist_alert_events_pkey"]["key_columns"] = ["status"]
    elif case == "foreign key has the wrong referenced table":
        foreign_key["referenced_table"] = "other_accounts"
    elif case == "foreign key has the wrong referenced schema":
        foreign_key["references_current_schema"] = False
    elif case == "foreign key has the wrong referenced columns":
        foreign_key["referenced_columns"] = ["other_id"]
    elif case == "foreign key has the wrong delete action":
        foreign_key["delete_action"] = "r"
    elif case == "foreign key has the wrong update action":
        foreign_key["update_action"] = "r"
    elif case == "foreign key has the wrong match type":
        foreign_key["match_type"] = "f"
    elif case == "constraint is deferrable":
        foreign_key["is_deferrable"] = True
    elif case == "constraint is initially deferred":
        foreign_key["is_initially_deferred"] = True
    elif case == "constraint is unvalidated":
        foreign_key["is_validated"] = False
    elif case == "foreign key has a missing internal enforcement trigger":
        foreign_key["internal_trigger_count"] = 3
        foreign_key["origin_enabled_internal_trigger_count"] = 3
    elif case == "foreign key has a disabled internal enforcement trigger":
        foreign_key["origin_enabled_internal_trigger_count"] = 3
    elif case == "check constraint is weakened":
        status_check["expression"] = "(status = any (array['open']))"
    elif case == "check constraint has a literal collision":
        status_check["expression"] = "(status = any (array['o''pen', 'resolved']))"
    elif case == "check constraint has a literal case collision":
        status_check["expression"] = "(status = any (array['OPEN', 'resolved']))"
    elif case == "unlisted constraint can reject writes":
        catalog["no_unlisted_alert_event_constraints"] = False
    elif case == "index is absent":
        indexes.pop("idx_b2b_watchlist_alert_events_account_status")
    elif case == "index has the wrong relation kind":
        account_status_index["relation_kind"] = "r"
    elif case == "index is a partition":
        account_status_index["is_partition"] = True
    elif case == "index has the wrong uniqueness":
        account_status_index["is_unique"] = True
    elif case == "index is invalid":
        account_status_index["is_valid"] = False
    elif case == "index is unready":
        account_status_index["is_ready"] = False
    elif case == "index has the wrong key attribute count":
        account_status_index["key_attribute_count"] = 2
    elif case == "index has the wrong attribute count":
        account_status_index["attribute_count"] = 4
    elif case == "index has the wrong key columns":
        account_status_index["key_columns"] = ["account_id", "last_seen_at", "status"]
    elif case == "index definition has the wrong direction":
        account_status_index["definition"] = (
            "usingbtree(account_id,status,last_seen_at)"
        )
    elif case == "index has an unexpected predicate":
        account_status_index["predicate"] = "(status = 'open')"
    elif case == "unlisted index can reject writes":
        catalog["no_unlisted_alert_event_indexes"] = False
    elif case == "unreviewed write interceptor can reject writes":
        catalog["no_unreviewed_alert_event_write_interceptors"] = False
    else:  # pragma: no cover - parametrize keeps this exhaustive.
        raise AssertionError(f"unexpected catalog guard case: {case}")

    code, payload = await module.run_migration_content_integrity_preflight(
        connection,
        migrations_dir=tmp_path,
        attest_known_reconciliations=True,
    )

    evidence = payload["known_reconciliation_evidence"][0]
    assert code == module.UNRESOLVED_DRIFT_EXIT, case
    assert evidence[field] is False, case
    for other_field in (
        "watchlist_alert_events_is_ordinary_table",
        "watchlist_alert_events_has_permanent_storage",
        "base_alert_event_columns_ready",
        "known_later_alert_event_columns_ready",
        "no_unlisted_alert_event_columns",
        "required_alert_event_constraints_ready",
        "no_unlisted_alert_event_constraints",
        "required_alert_event_indexes_ready",
        "no_unlisted_alert_event_indexes",
        "no_unreviewed_alert_event_write_interceptors",
    ):
        if other_field != field:
            assert evidence[other_field] is True, case
    assert evidence["status"] == "not_attested", case
    assert connection.execute_calls == []


@pytest.mark.asyncio
async def test_known_022b_reconciliation_attests_named_renamed_source_without_rows(
    tmp_path: Path,
) -> None:
    record = reconciliation_mod.MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION
    _write_migration(
        tmp_path,
        record.current_packaged_migration_name,
        _migration_022b_source(),
    )
    connection = _migration_022b_connection()

    code, payload = await module.run_migration_content_integrity_preflight(
        connection,
        migrations_dir=tmp_path,
        attest_known_reconciliations=True,
    )

    assert code == module.UNRESOLVED_DRIFT_EXIT
    assert payload["status"] == "unresolved_drift"
    assert payload["report"]["missing_source"] == [record.migration_name]
    assert payload["known_reconciliation_evidence"] == [
        {
            "reconciliation_id": record.reconciliation_id,
            "migration_name": record.migration_name,
            "source_verification": reconciliation_mod.HISTORICAL_LEDGER_DIGEST_UNAVAILABLE,
            "exactly_one_ledger_row": True,
            "ledger_digest_is_null": True,
            "applied_at_matches_record": True,
            "retained_packaged_digest_matches_record": True,
            "presence_events_is_ordinary_table": True,
            "unknown_count_column_ready": True,
            "unknown_count_has_no_constraints": True,
            "status": "attested",
        }
    ]
    assert connection.fetch_calls[0] == (
        "SELECT name, content_sha256 FROM schema_migrations",
        (),
    )
    assert connection.fetch_calls[1] == (
        "SELECT content_sha256, applied_at FROM schema_migrations WHERE name = $1 LIMIT 2",
        (record.migration_name,),
    )
    assert len(connection.fetch_calls) == 2
    assert len(connection.fetchrow_calls) == 1
    assert connection.fetchval_calls == []
    catalog_query = connection.fetchrow_calls[0][0]
    assert "WITH target_relation AS" in catalog_query
    assert "relation_state.relkind" in catalog_query
    assert "AND NOT relispartition" in catalog_query
    assert "information_schema.columns AS actual" in catalog_query
    assert "FROM pg_constraint AS actual" in catalog_query
    assert all(
        "FROM presence_events" not in query
        for query, _args in (
            connection.fetch_calls
            + connection.fetchrow_calls
            + connection.fetchval_calls
        )
    )
    assert connection.transaction_readonly == [True]
    assert connection.execute_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "field"),
    [
        ("non-null ledger digest", "ledger_digest_is_null"),
        ("truncated applied timestamp", "applied_at_matches_record"),
        ("duplicate ledger rows", "exactly_one_ledger_row"),
        ("missing detailed ledger row", "exactly_one_ledger_row"),
        ("changed retained package", "retained_packaged_digest_matches_record"),
        ("presence-events relation is not an ordinary table", "presence_events_is_ordinary_table"),
        ("presence-events relation is a leaf partition", "presence_events_is_ordinary_table"),
        ("missing unknown-count column", "unknown_count_column_ready"),
        ("wrong unknown-count type", "unknown_count_column_ready"),
        ("wrong unknown-count nullability", "unknown_count_column_ready"),
        ("wrong unknown-count default", "unknown_count_column_ready"),
        ("unknown-count constraint present", "unknown_count_has_no_constraints"),
    ],
)
async def test_known_022b_reconciliation_rejects_each_required_evidence_field(
    case: str,
    field: str,
    tmp_path: Path,
) -> None:
    record = reconciliation_mod.MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION
    source = _migration_022b_source()
    connection = _migration_022b_connection()
    if case == "non-null ledger digest":
        connection = _migration_022b_connection(ledger_digest="a" * 64)
    elif case == "truncated applied timestamp":
        connection = _migration_022b_connection(
            applied_at=record.observed_applied_at.replace(microsecond=0)
        )
    elif case == "duplicate ledger rows":
        connection = _migration_022b_connection(
            reconciliation_rows=[
                {
                    "content_sha256": None,
                    "applied_at": record.observed_applied_at,
                },
                {
                    "content_sha256": None,
                    "applied_at": record.observed_applied_at,
                },
            ]
        )
    elif case == "missing detailed ledger row":
        connection = _migration_022b_connection(reconciliation_rows=[])
    elif case == "changed retained package":
        source += b"\n-- changed after reviewed rename evidence\n"
    elif case == "presence-events relation is not an ordinary table":
        connection.presence_events_is_ordinary_table = False
    elif case == "presence-events relation is a leaf partition":
        connection.presence_events_is_leaf_partition = True
    elif case == "missing unknown-count column":
        connection.presence_unknown_count_columns = []
    elif case == "wrong unknown-count type":
        connection.presence_unknown_count_columns[0]["data_type"] = "bigint"
    elif case == "wrong unknown-count nullability":
        connection.presence_unknown_count_columns[0]["is_nullable"] = "NO"
    elif case == "wrong unknown-count default":
        connection.presence_unknown_count_columns[0]["column_default"] = "1"
    elif case == "unknown-count constraint present":
        connection.presence_unknown_count_has_constraint = True
    else:  # pragma: no cover - parametrize keeps this exhaustive.
        raise AssertionError(f"unexpected evidence case: {case}")
    _write_migration(tmp_path, record.current_packaged_migration_name, source)

    code, payload = await module.run_migration_content_integrity_preflight(
        connection,
        migrations_dir=tmp_path,
        attest_known_reconciliations=True,
    )

    evidence = payload["known_reconciliation_evidence"][0]
    assert code == module.UNRESOLVED_DRIFT_EXIT, case
    assert (
        evidence["source_verification"]
        == reconciliation_mod.HISTORICAL_LEDGER_DIGEST_UNAVAILABLE
    )
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
