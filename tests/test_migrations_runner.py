import asyncio
import hashlib
import itertools
import logging
import os
import re
import uuid
from pathlib import Path

import pytest

from atlas_brain.storage import recurring_invoice_schema

DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"


def _migration_387_source() -> bytes:
    return (
        Path(__file__).resolve().parents[1]
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "387_eom_recurring_invoice_dedup_recovery.sql"
    ).read_bytes()


def _migration_386_source() -> bytes:
    return (
        Path(__file__).resolve().parents[1]
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "386_eom_won_loss_nocodb_fence.sql"
    ).read_bytes()


def _migration_390_source() -> bytes:
    return (
        Path(__file__).resolve().parents[1]
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "390_eom_won_loss_direct_sql_fence_recovery.sql"
    ).read_bytes()


def _migration_function_body(source: bytes) -> str:
    match = re.search(
        r"AS\s+\$function\$(.*?)\$function\s*\$;",
        source.decode("utf-8"),
        re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _migration_022b_source() -> bytes:
    return (
        Path(__file__).resolve().parents[1]
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "027_presence_unknown_count.sql"
    ).read_bytes()


def test_migration_runner_workflow_enrolls_alert_writer_on_pr_and_main_push() -> None:
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "atlas_migrations_runner_checks.yml"
    ).read_text()
    trigger_block = workflow.split("\njobs:", 1)[0]
    pull_request_paths, push_paths = trigger_block.split("  push:", 1)
    writer_path = '"atlas_brain/services/b2b/watchlist_alerts.py"'

    assert writer_path in pull_request_paths
    assert writer_path in push_paths


def _default_b2b_watchlist_alert_events_catalog_row() -> dict[str, object]:
    """Return complete metadata-only evidence for the named 272 receipt."""
    from atlas_brain.storage.migrations.reconciliation import (
        _B2B_WATCHLIST_ALERT_EVENT_ALLOWED_COLUMNS,
        _B2B_WATCHLIST_ALERT_EVENT_CONSTRAINTS,
        _B2B_WATCHLIST_ALERT_EVENT_INDEXES,
    )

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
                    _B2B_WATCHLIST_ALERT_EVENT_ALLOWED_COLUMNS.items()
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
                    _B2B_WATCHLIST_ALERT_EVENT_CONSTRAINTS.items()
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
                for name, index in _B2B_WATCHLIST_ALERT_EVENT_INDEXES.items()
            },
            "no_unlisted_alert_event_indexes": True,
            "no_unreviewed_alert_event_write_interceptors": True,
        }
    }


class FakeMigrationPool:
    def __init__(self, records=None):
        self.records = [
            (record[0], record[1], record[2] if len(record) > 2 else None)
            for record in (records or [])
        ]
        self.inserted = []
        self.inserted_with_digest = []
        self.updated = []

    async def fetchval(self, query, *args):
        normalized = " ".join(query.split())
        if normalized == "SELECT content_sha256 FROM schema_migrations WHERE name = $1":
            name = args[0]
            for _version, record_name, content_sha256 in self.records:
                if record_name == name:
                    return content_sha256
            return None
        if "WHERE name = $1" in normalized:
            name = args[0]
            for version, record_name, _content_sha256 in self.records:
                if record_name == name:
                    return version
            return None
        if "WHERE version = $1" in normalized:
            version = args[0]
            for record_version, name, _content_sha256 in self.records:
                if record_version == version:
                    return name
            return None
        if "MIN(version)" in normalized:
            if not self.records:
                return -1
            min_version = min(version for version, _name, _digest in self.records)
            return min_version - 1 if min_version < 0 else -1
        raise AssertionError(f"Unexpected fetchval query: {query}")

    async def fetch(self, query, *args):
        assert query == "SELECT name, content_sha256 FROM schema_migrations"
        return [
            {"name": name, "content_sha256": content_sha256}
            for _version, name, content_sha256 in self.records
        ]

    async def execute(self, query, *args):
        normalized = " ".join(query.split())
        if normalized.startswith("UPDATE schema_migrations SET content_sha256"):
            name, content_sha256 = args
            self.records = [
                (version, record_name, content_sha256)
                if record_name == name
                else (version, record_name, existing_digest)
                for version, record_name, existing_digest in self.records
            ]
            self.updated.append((name, content_sha256))
            return
        if normalized.startswith("INSERT INTO schema_migrations"):
            record = (args[0], args[1], args[2])
            self.records.append(record)
            self.inserted.append(record[:2])
            self.inserted_with_digest.append(record)
            return
        raise AssertionError(f"Unexpected execute query: {query}")


@pytest.mark.asyncio
async def test_record_migration_uses_prefix_version_when_available():
    from atlas_brain.storage.migrations import _record_migration

    pool = FakeMigrationPool(records=[(1, "001_initial_schema")])
    content_sha256 = "a" * 64

    await _record_migration(
        pool,
        "247_b2b_vendor_witness_packets.sql",
        content_sha256,
    )

    assert pool.inserted == [(247, "247_b2b_vendor_witness_packets")]
    assert pool.inserted_with_digest == [
        (247, "247_b2b_vendor_witness_packets", content_sha256)
    ]


@pytest.mark.asyncio
async def test_record_migration_uses_negative_version_on_prefix_collision():
    from atlas_brain.storage.migrations import _record_migration

    pool = FakeMigrationPool(records=[
        (76, "076_saas_accounts"),
        (230, "230_scrape_target_checkpoints"),
    ])

    await _record_migration(pool, "076_consumer_analytics_views.sql", "b" * 64)
    await _record_migration(pool, "230_b2b_reasoning_synthesis.sql", "c" * 64)

    assert pool.inserted == [
        (-1, "076_consumer_analytics_views"),
        (-2, "230_b2b_reasoning_synthesis"),
    ]


@pytest.mark.asyncio
async def test_record_migration_leaves_existing_legacy_row_unchanged_without_pending_proof():
    from atlas_brain.storage.migrations import _record_migration

    pool = FakeMigrationPool(records=[(247, "247_probe", None)])

    await _record_migration(pool, "247_probe.sql", "d" * 64)

    assert pool.records == [(247, "247_probe", None)]
    assert pool.updated == []


@pytest.mark.asyncio
async def test_migration_content_integrity_report_classifies_all_evidence_states(
    tmp_path,
    monkeypatch,
):
    from atlas_brain.storage.migrations import _migration_content_integrity_report

    verified = tmp_path / "900_verified.sql"
    legacy = tmp_path / "901_legacy.sql"
    mismatched = tmp_path / "902_mismatched.sql"
    unreadable = tmp_path / "903_unreadable.sql"
    verified.write_bytes(b"SELECT 'verified';\n")
    legacy.write_bytes(b"SELECT 'legacy';\n")
    mismatched.write_bytes(b"SELECT 'mismatched';\n")
    unreadable.write_bytes(b"SELECT 'unreadable';\n")
    pool = FakeMigrationPool(records=[
        (900, "900_verified", hashlib.sha256(verified.read_bytes()).hexdigest()),
        (901, "901_legacy", None),
        (902, "902_mismatched", "not-a-sha256"),
        (903, "903_unreadable", None),
        (904, "904_missing_source", "f" * 64),
    ])

    original_read_bytes = Path.read_bytes

    def read_bytes_or_raise(path):
        if path == unreadable:
            raise PermissionError("migration source is unreadable")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes_or_raise)

    report = await _migration_content_integrity_report(
        pool,
        [verified, legacy, mismatched, unreadable],
    )

    assert report.verified == ("900_verified",)
    assert report.legacy_unverified == ("901_legacy",)
    assert report.mismatched == ("902_mismatched",)
    assert report.missing_source == ("903_unreadable", "904_missing_source")


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("INSERT INTO schema_migrations (version, name) VALUES (1, '001');", True),
        ('INSERT INTO "schema_migrations" (version, name) VALUES (1, \'001\');', True),
        ("INSERT INTO public.schema_migrations (version, name) VALUES (1, '001');", True),
        ('INSERT INTO "public"."schema_migrations" (version, name) VALUES (1, \'001\');', True),
        ('INSERT INTO "SCHEMA_MIGRATIONS" (version, name) VALUES (1, \'001\');', False),
        ("-- INSERT INTO schema_migrations (version, name) VALUES (1, '001');", False),
        ("SELECT 'INSERT INTO schema_migrations (version, name)';", False),
        ("INSERT INTO migration_audit (name) VALUES ('schema_migrations');", False),
    ],
)
def test_self_recording_detector_only_admits_executable_ledger_inserts(sql, expected):
    from atlas_brain.storage.migrations import _contains_executable_self_recording_insert

    assert _contains_executable_self_recording_insert(sql) is expected


def test_self_recording_detector_matches_static_sql_grammar_matrix():
    """Derive expected admission from SQL token, context, and target axes."""
    from atlas_brain.storage.migrations import _contains_executable_self_recording_insert

    grammar_axes = {
        "tokens": (("INSERT", True), ("insert", True), ("UPDATE", False)),
        "containers": ("direct", "line-comment", "literal"),
        "keys": (
            ("schema_migrations", True),
            ("SCHEMA_MIGRATIONS", True),
            ('"schema_migrations"', True),
            ('"SCHEMA_MIGRATIONS"', False),
            ("public.schema_migrations", True),
            ('"public"."schema_migrations"', True),
            ('public."SCHEMA_MIGRATIONS"', False),
            ("migration_audit", False),
        ),
    }

    for (token, is_insert), container, (target, is_ledger) in itertools.product(
        grammar_axes["tokens"],
        grammar_axes["containers"],
        grammar_axes["keys"],
    ):
        statement = f"{token} INTO ONLY {target} (version, name) VALUES (1, '001');"
        if container == "direct":
            sql = statement
        elif container == "line-comment":
            sql = f"-- {statement}"
        else:
            literal = statement.replace("'", "''")
            sql = f"SELECT '{literal}'"

        expected = is_insert and is_ledger and container == "direct"
        assert _contains_executable_self_recording_insert(sql) is expected


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


def test_eom_lead_review_queue_index_matches_keyset_order():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "355_eom_lead_review_queue_index.sql"
    ).read_text()

    assert (
        "DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue"
        in migration
    )
    assert "CREATE INDEX CONCURRENTLY idx_contacts_eom_lead_review_queue" in migration
    assert "ON contacts (created_at DESC, id DESC)" in migration
    assert "business_context_id = 'effingham_maids'" in migration
    assert "status = 'active'" in migration
    assert "contact_type = 'lead'" in migration
    assert "lead_stage = 'new'" in migration


def test_eom_lead_review_queue_booked_stage_index_matches_provider_filter():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "356_eom_lead_review_queue_booked_stage.sql"
    ).read_text()

    assert (
        "DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue"
        in migration
    )
    assert "CREATE INDEX CONCURRENTLY idx_contacts_eom_lead_review_queue" in migration
    assert "ON contacts (created_at DESC, id DESC)" in migration
    assert "business_context_id = 'effingham_maids'" in migration
    assert "status = 'active'" in migration
    assert "contact_type = 'lead'" in migration
    assert "lead_stage IN ('new', 'estimate_booked')" in migration
    assert "Rollback evidence:" in migration
    assert "lead_stage = 'new'" in migration
    assert "old code still filters lead_stage = 'new' at query time" in migration


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


def test_eom_estimate_booking_operation_key_index_is_additive_and_leading_key():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "357_eom_estimate_booking_operation_key_index.sql"
    ).read_text()
    upper = migration.upper()

    # Replay-safe pattern (same as migration 355): a canceled concurrent build
    # leaves an INVALID same-named index, so IF NOT EXISTS on the create would
    # record a broken index as applied. The drop must be real (not just the
    # rollback-evidence comment) and must precede the recreate.
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" not in migration
    assert (
        migration.count(
            "DROP INDEX CONCURRENTLY IF EXISTS "
            "idx_eom_lead_lifecycle_booking_operation_key"
        )
        >= 2
    )  # rollback-evidence comment + the executable statement
    assert (
        "CREATE INDEX CONCURRENTLY idx_eom_lead_lifecycle_booking_operation_key"
        in migration
    )
    assert migration.index(
        "DROP INDEX CONCURRENTLY IF EXISTS "
        "idx_eom_lead_lifecycle_booking_operation_key"
    ) < migration.index(
        "CREATE INDEX CONCURRENTLY idx_eom_lead_lifecycle_booking_operation_key"
    )
    assert "ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)" in migration
    assert "operation_key IS NOT NULL" in migration
    assert "'estimate_booking_requested'" in migration
    assert "'estimate_booking_calendar_failed'" in migration
    assert "'estimate_booking_calendar_ambiguous'" in migration
    assert "'estimate_booked'" in migration
    assert "DROP TABLE" not in upper
    assert "ALTER TABLE" not in upper


def test_eom_lead_review_queue_won_stage_index_matches_provider_filter():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "358_eom_lead_review_queue_won_stage.sql"
    ).read_text()

    # Replay-safe pattern (same as migrations 355/356/357): the drop must be
    # real and must precede the recreate; IF NOT EXISTS on the create would
    # record an INVALID leftover index as applied.
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" not in migration
    assert (
        migration.count(
            "DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue"
        )
        >= 2
    )  # rollback-evidence comment + the executable statement
    assert "CREATE INDEX CONCURRENTLY idx_contacts_eom_lead_review_queue" in migration
    assert migration.index(
        "DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue"
    ) < migration.index(
        "CREATE INDEX CONCURRENTLY idx_contacts_eom_lead_review_queue"
    )
    assert "ON contacts (created_at DESC, id DESC)" in migration
    assert "business_context_id = 'effingham_maids'" in migration
    assert "status = 'active'" in migration
    assert "contact_type = 'lead'" in migration
    assert "lead_stage IN ('new', 'estimate_booked', 'won')" in migration
    assert "Rollback evidence:" in migration
    assert "lead_stage IN ('new', 'estimate_booked')" in migration
    # Application rollback keeps persisted won leads operable under old
    # code (which admits only new/estimate_booked to review and handoff):
    # the documented data step reverts stage state while the append-only
    # ledger keeps the first_clean_booked evidence.
    assert "Application rollback" in migration
    assert "UPDATE contacts SET lead_stage = 'estimate_booked'" in migration
    assert "AND lead_stage = 'won'" in migration


def test_eom_booking_operation_key_index_covers_first_clean_events():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "359_eom_booking_operation_key_index_first_clean.sql"
    ).read_text()
    upper = migration.upper()

    # Replay-safe pattern (same as migration 357): real drop preceding the
    # recreate, no IF NOT EXISTS on the create.
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" not in migration
    assert (
        migration.count(
            "DROP INDEX CONCURRENTLY IF EXISTS "
            "idx_eom_lead_lifecycle_booking_operation_key"
        )
        >= 2
    )  # rollback-evidence comment + the executable statement
    assert (
        "CREATE INDEX CONCURRENTLY idx_eom_lead_lifecycle_booking_operation_key"
        in migration
    )
    assert migration.index(
        "DROP INDEX CONCURRENTLY IF EXISTS "
        "idx_eom_lead_lifecycle_booking_operation_key"
    ) < migration.index(
        "CREATE INDEX CONCURRENTLY idx_eom_lead_lifecycle_booking_operation_key"
    )
    assert "ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)" in migration
    assert "operation_key IS NOT NULL" in migration
    # Both booking families must stay covered so cross-family ownership
    # checks keep the leading-key access path.
    assert "'estimate_booking_requested'" in migration
    assert "'estimate_booking_calendar_failed'" in migration
    assert "'estimate_booking_calendar_ambiguous'" in migration
    assert "'estimate_booked'" in migration
    assert "'first_clean_booking_requested'" in migration
    assert "'first_clean_booking_calendar_failed'" in migration
    assert "'first_clean_booking_calendar_ambiguous'" in migration
    assert "'first_clean_booked'" in migration
    assert "Rollback evidence:" in migration
    assert "DROP TABLE" not in upper
    assert "ALTER TABLE" not in upper


def test_eom_lead_disposition_operation_key_index_covers_lost_and_reopen():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "362_eom_lead_disposition_operation_key_index.sql"
    ).read_text()
    upper = migration.upper()

    # Replay-safe pattern (same as 357/359): real drop preceding the recreate,
    # no IF NOT EXISTS on the create.
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" not in migration
    assert (
        migration.count(
            "DROP INDEX CONCURRENTLY IF EXISTS "
            "idx_eom_lead_lifecycle_disposition_operation_key"
        )
        >= 2
    )  # rollback-evidence comment + the executable statement
    assert (
        "CREATE INDEX CONCURRENTLY "
        "idx_eom_lead_lifecycle_disposition_operation_key" in migration
    )
    assert migration.index(
        "DROP INDEX CONCURRENTLY IF EXISTS "
        "idx_eom_lead_lifecycle_disposition_operation_key"
    ) < migration.index(
        "CREATE INDEX CONCURRENTLY "
        "idx_eom_lead_lifecycle_disposition_operation_key"
    )
    # operation_key must lead so the cross-contact key-ownership probe in
    # mark_eom_lead_lost / reopen_eom_lead is index-backed, and the predicate
    # must cover exactly the two disposition events (not the booking families).
    assert "ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)" in migration
    assert "operation_key IS NOT NULL" in migration
    assert "'lead_lost'" in migration
    assert "'lead_reopened'" in migration
    assert "'estimate_booked'" not in migration
    assert "Rollback evidence:" in migration
    assert "DROP TABLE" not in upper
    assert "ALTER TABLE" not in upper


def test_eom_operator_contact_operation_key_index_covers_contact_mutations():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "364_eom_operator_contact_operation_key_index.sql"
    ).read_text()
    upper = migration.upper()

    # Replay-safe pattern (same as 357/359/362): real drop preceding the
    # recreate, no IF NOT EXISTS on the create.
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" not in migration
    assert (
        migration.count(
            "DROP INDEX CONCURRENTLY IF EXISTS "
            "idx_eom_lead_lifecycle_operator_contact_operation_key"
        )
        >= 2
    )  # rollback-evidence comment + the executable statement
    assert (
        "CREATE INDEX CONCURRENTLY "
        "idx_eom_lead_lifecycle_operator_contact_operation_key" in migration
    )
    assert migration.index(
        "DROP INDEX CONCURRENTLY IF EXISTS "
        "idx_eom_lead_lifecycle_operator_contact_operation_key"
    ) < migration.index(
        "CREATE INDEX CONCURRENTLY "
        "idx_eom_lead_lifecycle_operator_contact_operation_key"
    )
    # operation_key must lead so idempotent replay/conflict checks in the
    # operator contact mutation path are index-backed, and the predicate must
    # cover exactly the two operator contact receipt events.
    assert "ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)" in migration
    assert "operation_key IS NOT NULL" in migration
    assert "'contact_created'" in migration
    assert "'contact_updated'" in migration
    assert "'lead_lost'" not in migration
    assert "'estimate_booked'" not in migration
    assert "Rollback evidence:" in migration
    assert "DROP TABLE" not in upper
    assert "ALTER TABLE" not in upper


def test_eom_lead_lifecycle_sequence_is_db_owned_and_writer_compatible():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "363_eom_lead_lifecycle_sequence.sql"
    ).read_text()
    upper = migration.upper()

    assert "CREATE SEQUENCE IF NOT EXISTS eom_lead_lifecycle_events_sequence_seq" in migration
    assert "ADD COLUMN IF NOT EXISTS lifecycle_sequence BIGINT" in migration
    assert (
        "ALTER COLUMN lifecycle_sequence SET DEFAULT "
        "nextval('eom_lead_lifecycle_events_sequence_seq'::regclass)"
        in migration
    )
    assert (
        "OWNED BY eom_lead_lifecycle_events.lifecycle_sequence"
        in migration
    )
    assert "compatible with old app writers" in migration
    assert "Rollback evidence:" in migration
    assert "UPDATE eom_lead_lifecycle_events" not in migration
    assert "DELETE FROM eom_lead_lifecycle_events" not in migration
    assert "DROP TABLE" not in upper


def test_eom_onboarding_email_drafts_migration_is_additive_and_single_send_safe():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "360_eom_onboarding_email_drafts.sql"
    ).read_text()
    upper = migration.upper()

    assert "CREATE TABLE IF NOT EXISTS eom_onboarding_email_drafts" in migration
    assert "contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT" in migration
    assert "operation_key VARCHAR(128) NOT NULL UNIQUE" in migration
    # Claim ownership ('sending') is modeled separately from confirmed
    # delivery ('sent') so a crashed send can never be recorded as sent.
    assert "CHECK (status IN ('pending', 'sending', 'sent', 'revoked'))" in migration
    assert "claimed_at TIMESTAMPTZ" in migration
    assert "subject TEXT NOT NULL" in migration
    assert "body TEXT NOT NULL" in migration
    # At most one live draft per contact, across pending AND sending.
    assert "uq_eom_onboarding_email_drafts_live_contact" in migration
    assert "WHERE status IN ('pending', 'sending')" in migration
    # The A3 approval surface inherits the atomic claim from this header:
    # single winner, readiness predicate built in (no blocked or
    # recipient-less row is claimable), delivery confirmed separately.
    assert "SET status = 'sending', claimed_at = NOW()" in migration
    assert "AND status = 'pending'" in migration
    assert "AND blocker IS NULL" in migration
    assert "AND recipient_email IS NOT NULL" in migration
    assert "RETURNING" in migration
    assert "SET status = 'sent', sent_at = NOW()" in migration
    assert "WHERE id = $1 AND status = 'sending'" in migration
    assert "idempotency key" in migration
    assert "Rollback evidence:" in migration
    assert "ALTER TABLE" not in upper
    assert "CONCURRENTLY" not in upper


def test_eom_onboarding_draft_actor_bigint_migration_is_atomic_and_value_preserving():
    migration = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "storage"
        / "migrations"
        / "361_eom_onboarding_draft_actor_bigint.sql"
    ).read_text()
    upper = migration.upper()
    first_line = next(
        line.strip() for line in migration.splitlines() if line.strip()
    )

    # Atomic bookkeeping: the widening and its ledger row commit together.
    assert first_line == "-- atlas: atomic-bookkeeping"
    # The funnel actor boundary admits signed 64-bit ids and the handoff
    # table already stores BIGINT; the draft approver column must match.
    assert (
        "ALTER COLUMN approved_by_employee_id TYPE BIGINT" in migration
    )
    assert "ALTER TABLE eom_onboarding_email_drafts" in migration
    assert "Rollback evidence:" in migration
    assert "TYPE INTEGER" in migration
    assert "CONCURRENTLY" not in upper
    assert "DROP " not in upper


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
        self.atomic_transactions = 0
        self.atomic_transaction_errors = 0
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

        snapshot = [
            {"name": name, "content_sha256": content_sha256}
            for _version, name, content_sha256 in self.records
        ]
        await asyncio.sleep(0)
        return snapshot

    async def execute(self, query, *args):
        normalized = " ".join(query.split())
        if normalized.startswith("CREATE TABLE IF NOT EXISTS schema_migrations"):
            return
        if normalized.startswith("ALTER TABLE schema_migrations ADD COLUMN"):
            return
        if normalized.startswith("INSERT INTO schema_migrations"):
            return await super().execute(query, *args)
        if normalized.startswith("UPDATE schema_migrations SET content_sha256"):
            return await super().execute(query, *args)
        self.applied_sql.append(normalized)

    async def acquire(self):
        self.acquired += 1
        self.max_acquired = max(self.max_acquired, self.acquired)
        return _Conn(self)

    async def release(self, conn):
        self.acquired -= 1

    def transaction(self):
        return _RollbackMigrationTransaction(self)


class _AttestedReconciliationPool(_SerializingPool):
    """Runner fixture that answers the real named catalog-attestation queries."""

    def __init__(
        self,
        *,
        recurring_schema_ready: bool = True,
        zero_active_null_period_rows: bool = True,
    ):
        super().__init__()
        self.recurring_schema_ready = recurring_schema_ready
        self.zero_active_null_period_rows = zero_active_null_period_rows
        self.reconciliation_rows: list[dict[str, object]] = []
        self.public_onboarding_reconciliation_rows: list[dict[str, object]] = []
        self.b2b_campaign_partner_reconciliation_rows: list[dict[str, object]] = []
        self.b2b_watchlist_alert_events_reconciliation_rows: list[dict[str, object]] = []
        self.presence_unknown_count_reconciliation_rows: list[dict[str, object]] = []
        self.presence_unknown_count_columns = [
            {
                "column_name": "unknown_count",
                "data_type": "integer",
                "is_nullable": "YES",
                "column_default": "0",
            }
        ]
        self.presence_events_is_ordinary_table = True
        self.presence_events_is_leaf_partition = False
        self.presence_unknown_count_has_constraint = False
        self.b2b_campaign_partner_catalog_row = {
            "b2b_campaigns_is_ordinary_table": True,
            "partner_id_column_name": "partner_id",
            "partner_id_data_type": "uuid",
            "partner_id_is_nullable": True,
            "partner_id_has_default": False,
            "partner_foreign_key_constraint_type": "f",
            "partner_foreign_key_columns": ["partner_id"],
            "partner_foreign_key_referenced_table": "affiliate_partners",
            "partner_foreign_key_references_current_schema": True,
            "partner_foreign_key_referenced_columns": ["id"],
            "partner_foreign_key_delete_action": "n",
            "partner_foreign_key_update_action": "a",
            "partner_foreign_key_match_type": "s",
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
            "partner_index_key_column": "partner_id",
            "partner_index_predicate": "(partner_id IS NOT NULL)",
        }
        self.b2b_watchlist_alert_events_catalog_row = (
            _default_b2b_watchlist_alert_events_catalog_row()
        )

    async def fetch(self, query, *args):
        from atlas_brain.storage.migrations.reconciliation import (
            MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION,
            MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION,
            MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION,
            MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION,
            MIGRATION_387_RECONCILIATION,
            _PUBLIC_ONBOARDING_TOKEN_COLUMNS,
            _PUBLIC_ONBOARDING_TOKEN_REQUIRED_CONSTRAINTS,
        )

        normalized = " ".join(query.split())
        if normalized == (
            "SELECT content_sha256, applied_at FROM schema_migrations "
            "WHERE name = $1 LIMIT 2"
        ):
            if args == (MIGRATION_387_RECONCILIATION.migration_name,):
                return list(self.reconciliation_rows)
            if args == (
                MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION.migration_name,
            ):
                return list(self.public_onboarding_reconciliation_rows)
            assert args == (
                MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION.migration_name,
            )
            return list(self.presence_unknown_count_reconciliation_rows)
        if normalized == (
            "SELECT version, content_sha256, applied_at FROM schema_migrations "
            "WHERE name = $1 LIMIT 2"
        ):
            if args == (
                MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION.migration_name,
            ):
                return list(self.b2b_campaign_partner_reconciliation_rows)
            assert args == (
                MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION.migration_name,
            )
            return list(self.b2b_watchlist_alert_events_reconciliation_rows)
        if "FROM information_schema.columns AS actual" in query:
            assert args == (list(_PUBLIC_ONBOARDING_TOKEN_COLUMNS),)
            return [
                {
                    "column_name": name,
                    "data_type": data_type,
                    "character_maximum_length": maximum_length,
                    "is_nullable": nullable,
                    "column_default": default,
                }
                for name, (data_type, maximum_length, nullable, default) in (
                    _PUBLIC_ONBOARDING_TOKEN_COLUMNS.items()
                )
            ]
        if "eom_public_onboarding_tokens" in query:
            assert "FROM pg_constraint AS actual" in query
            assert args == (list(_PUBLIC_ONBOARDING_TOKEN_REQUIRED_CONSTRAINTS),)
            return [
                {
                    "conname": name,
                    "constraint_type": constraint.constraint_type,
                    "key_columns": list(constraint.key_columns),
                    "referenced_table": constraint.referenced_table,
                    "references_current_schema": (
                        constraint.referenced_table is not None
                    ),
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
                    _PUBLIC_ONBOARDING_TOKEN_REQUIRED_CONSTRAINTS.items()
                )
            ]
        if "FROM pg_constraint AS actual" in query:
            assert args == (
                list(recurring_invoice_schema._RECURRING_INVOICE_DEDUP_CONSTRAINTS),
            )
            return [
                {"conname": name, "definition": definition}
                for name, definition in (
                    recurring_invoice_schema._RECURRING_INVOICE_DEDUP_CONSTRAINT_EXPRESSIONS.items()
                )
            ]
        return await super().fetch(query, *args)

    async def fetchrow(self, query, *args):
        if "b2b_watchlist_alert_events" in query:
            from atlas_brain.storage.migrations.reconciliation import (
                _B2B_WATCHLIST_ALERT_EVENT_ALLOWED_COLUMNS,
                _B2B_WATCHLIST_ALERT_EVENT_CONSTRAINTS,
                _B2B_WATCHLIST_ALERT_EVENT_INDEXES,
            )

            assert args == (
                list(_B2B_WATCHLIST_ALERT_EVENT_ALLOWED_COLUMNS),
                list(_B2B_WATCHLIST_ALERT_EVENT_CONSTRAINTS),
                list(_B2B_WATCHLIST_ALERT_EVENT_INDEXES),
            )
            assert "relation_state.relpersistence" in query
            assert "JOIN pg_type AS type_state" in query
            assert "attribute_state.attcollation = type_state.typcollation" in query
            assert "unlisted_indexes AS" in query
            assert "FROM unnest(index_state.indkey)" in query
            assert "index_state.indisunique OR index_state.indisexclusion" not in query
            assert "unreviewed_write_interceptors AS" in query
            assert "constraint_trigger.tgconstraint = actual.oid" in query
            assert "constraint_trigger.tgisinternal" in query
            assert "constraint_trigger.tgenabled = 'O'::\"char\"" in query
            assert "FROM pg_trigger AS trigger_state" in query
            assert "FROM pg_rewrite AS rule_state" in query
            assert "FROM pg_policy AS policy_state" in query
            assert "relation_state.relrowsecurity" in query
            assert "relation_state.relforcerowsecurity" in query
            return dict(self.b2b_watchlist_alert_events_catalog_row)
        if "b2b_campaigns" in query:
            assert args == ()
            assert "WITH target_relation AS" in query
            assert "JOIN pg_attribute AS attribute_state" in query
            assert "JOIN pg_constraint AS actual" in query
            assert "JOIN pg_index AS index_state" in query
            return dict(self.b2b_campaign_partner_catalog_row)
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
            from atlas_brain.storage.migrations.reconciliation import (
                _PUBLIC_ONBOARDING_TOKEN_ISSUED_CONTACT_INDEX,
                _PUBLIC_ONBOARDING_TOKEN_STATUS_INDEX,
            )

            if args == (_PUBLIC_ONBOARDING_TOKEN_ISSUED_CONTACT_INDEX,):
                return {
                    "indisunique": True,
                    "indisvalid": True,
                    "indisready": True,
                    "indnkeyatts": 1,
                    "key_column_1": "contact_id",
                    "definition": (
                        "CREATE UNIQUE INDEX "
                        "uq_eom_public_onboarding_tokens_issued_contact ON "
                        "eom_public_onboarding_tokens USING btree (contact_id) "
                        "WHERE ((status)::text = 'issued'::text)"
                    ),
                    "predicate": "((status)::text = 'issued'::text)",
                }
            assert args == (_PUBLIC_ONBOARDING_TOKEN_STATUS_INDEX,)
            return {
                "indisunique": False,
                "indisvalid": True,
                "indisready": True,
                "indnkeyatts": 2,
                "key_column_1": "status",
                "key_column_2": "issued_at",
                "definition": (
                    "CREATE INDEX idx_eom_public_onboarding_tokens_status ON "
                    "eom_public_onboarding_tokens USING btree (status, issued_at DESC)"
                ),
                "predicate": None,
            }
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

    async def fetchval(self, query, *args):
        if "information_schema.columns AS actual" in query:
            assert args == ()
            return self.recurring_schema_ready
        if "FROM invoices" in query:
            assert args == ()
            return self.zero_active_null_period_rows
        return await super().fetchval(query, *args)


class _RollbackMigrationTransaction:
    """In-memory transaction seam that restores migration effects on failure."""

    def __init__(self, pool):
        self.pool = pool
        self.snapshot = None

    async def __aenter__(self):
        self.pool.atomic_transactions += 1
        self.snapshot = (
            list(self.pool.records),
            list(self.pool.inserted),
            list(self.pool.inserted_with_digest),
            list(self.pool.updated),
            list(self.pool.applied_sql),
        )
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.pool.atomic_transaction_errors += 1
            (
                self.pool.records,
                self.pool.inserted,
                self.pool.inserted_with_digest,
                self.pool.updated,
                self.pool.applied_sql,
            ) = self.snapshot
        return False


class _Conn:
    """Single acquired connection: the lock, bookkeeping and migration SQL all
    run here, which is what proves a one-connection pool cannot deadlock."""

    def __init__(self, pool):
        self.pool = pool
        self.held = False

    async def execute(self, query, *args):
        if "pg_advisory_unlock" in query:
            if self.pool.honor_lock and self.held:
                self.pool._lock().release()
                self.held = False
            return
        return await self.pool.execute(query, *args)

    async def fetch(self, query, *args):
        return await self.pool.fetch(query, *args)

    async def fetchval(self, query, *args):
        if "pg_try_advisory_lock" in query:
            # Non-blocking, like the real thing: hand back False rather than
            # waiting, so the runner polls instead of sitting in a transaction.
            if not self.pool.honor_lock:
                return True
            gate = self.pool._lock()
            if gate.locked():
                return False
            await gate.acquire()
            self.held = True
            return True
        return await self.pool.fetchval(query, *args)

    async def fetchrow(self, query, *args):
        return await self.pool.fetchrow(query, *args)

    def transaction(self):
        return self.pool.transaction()


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
    assert pool.inserted_with_digest == [
        (900, "900_probe", hashlib.sha256(b"SELECT 900").hexdigest())
    ]


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


@pytest.mark.asyncio
async def test_pending_migration_hashes_and_executes_one_source_read(
    tmp_path,
    monkeypatch,
):
    from atlas_brain.storage.migrations import run_migrations

    source_path = tmp_path / "900_one_read.sql"
    source = b"SELECT 'one source read'"
    expected_digest = hashlib.sha256(source).hexdigest()
    source_path.write_bytes(source)
    original_read_bytes = Path.read_bytes
    reads = []

    def read_bytes_once(path):
        if path == source_path:
            reads.append(path)
            if len(reads) > 1:
                raise AssertionError("pending migration source was read more than once")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes_once)
    pool = _SerializingPool()

    await run_migrations(pool, migrations_dir=tmp_path)

    assert reads == [source_path]
    assert pool.applied_sql == [source.decode("utf-8")]
    assert pool.inserted_with_digest == [(900, "900_one_read", expected_digest)]


@pytest.mark.asyncio
async def test_pending_self_recording_migration_receives_its_digest(tmp_path):
    from atlas_brain.storage.migrations import run_migrations

    source = (
        "INSERT INTO schema_migrations (version, name, content_sha256) "
        "VALUES (900, '900_self_recording', 'wrong-digest');"
    )
    (tmp_path / "900_self_recording.sql").write_text(source)

    class _SelfRecordingPool(_SerializingPool):
        async def execute(self, query, *args):
            if query == source:
                self.records.append((900, "900_self_recording", "wrong-digest"))
                self.applied_sql.append(query)
                return
            return await super().execute(query, *args)

    pool = _SelfRecordingPool()
    await run_migrations(pool, migrations_dir=tmp_path)

    expected_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert pool.records == [(900, "900_self_recording", expected_digest)]
    assert pool.updated == [("900_self_recording", expected_digest)]
    assert pool.atomic_transactions == 1
    assert pool.atomic_transaction_errors == 0


@pytest.mark.asyncio
async def test_self_recording_digest_must_persist_before_transaction_commits(tmp_path):
    """A silent UPDATE 0 cannot commit a stale self-recorded digest."""
    from atlas_brain.storage.migrations import run_migrations

    source = (
        "CREATE TABLE self_recording_persistence_probe (id int);\n"
        "INSERT INTO schema_migrations (version, name) "
        "VALUES (900, '900_self_recording_persistence');"
    )
    (tmp_path / "900_self_recording_persistence.sql").write_text(source)

    class _SuppressOnceDigestUpdatePool(_SerializingPool):
        def __init__(self):
            super().__init__()
            self.suppress_next_digest_update = True

        async def execute(self, query, *args):
            if query == source:
                self.records.append((900, "900_self_recording_persistence", None))
                self.applied_sql.append(query)
                return
            normalized = " ".join(query.split())
            if (
                normalized.startswith("UPDATE schema_migrations SET content_sha256")
                and self.suppress_next_digest_update
            ):
                self.suppress_next_digest_update = False
                return "UPDATE 0"
            return await super().execute(query, *args)

    pool = _SuppressOnceDigestUpdatePool()
    with pytest.raises(RuntimeError, match="did not persist its expected content SHA-256"):
        await run_migrations(pool, migrations_dir=tmp_path)

    assert pool.records == []
    assert pool.applied_sql == []
    assert pool.atomic_transactions == 1
    assert pool.atomic_transaction_errors == 1

    await run_migrations(pool, migrations_dir=tmp_path)

    expected_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert pool.records == [
        (900, "900_self_recording_persistence", expected_digest)
    ]
    assert pool.atomic_transactions == 2


@pytest.mark.asyncio
async def test_self_recording_digest_failure_rolls_back_and_retry_records_once(tmp_path):
    """A failed digest update cannot leave a self-recorded row permanently legacy."""
    from atlas_brain.storage.migrations import run_migrations

    source = (
        "CREATE TABLE self_recording_retry_probe (id int);\n"
        "INSERT INTO schema_migrations (version, name) "
        "VALUES (900, '900_self_recording_retry');"
    )
    (tmp_path / "900_self_recording_retry.sql").write_text(source)

    class _FailOnceDigestUpdatePool(_SerializingPool):
        def __init__(self):
            super().__init__()
            self.fail_next_digest_update = True

        async def execute(self, query, *args):
            if query == source:
                self.records.append((900, "900_self_recording_retry", None))
                self.applied_sql.append(query)
                return
            normalized = " ".join(query.split())
            if (
                normalized.startswith("UPDATE schema_migrations SET content_sha256")
                and self.fail_next_digest_update
            ):
                self.fail_next_digest_update = False
                raise RuntimeError("injected digest update failure")
            return await super().execute(query, *args)

    pool = _FailOnceDigestUpdatePool()
    with pytest.raises(RuntimeError, match="injected digest update failure"):
        await run_migrations(pool, migrations_dir=tmp_path)

    assert pool.records == []
    assert pool.applied_sql == []
    assert pool.updated == []
    assert pool.atomic_transactions == 1
    assert pool.atomic_transaction_errors == 1

    await run_migrations(pool, migrations_dir=tmp_path)

    expected_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert pool.records == [(900, "900_self_recording_retry", expected_digest)]
    assert pool.updated == [("900_self_recording_retry", expected_digest)]
    assert pool.atomic_transactions == 2


@pytest.mark.asyncio
async def test_comment_or_literal_ledger_insert_does_not_force_a_transaction(tmp_path):
    """Mentions are not self-recording SQL and preserve autocommit behavior."""
    from atlas_brain.storage.migrations import run_migrations

    (tmp_path / "900_mention_only.sql").write_text(
        "CREATE TABLE mention_only_probe (id int);\n"
        "-- INSERT INTO schema_migrations (version, name) VALUES (900, 'noop');\n"
        "SELECT 'INSERT INTO schema_migrations (version, name)';"
    )
    pool = _SingleConnectionPool(tmp_path)

    await run_migrations(pool, migrations_dir=tmp_path)

    assert getattr(pool, "atomic_transactions", 0) == 0


@pytest.mark.asyncio
async def test_self_recording_concurrent_ddl_is_rejected_before_any_write(tmp_path):
    """A self-recording concurrent-DDL migration cannot reopen the crash window."""
    from atlas_brain.storage.migrations import run_migrations

    source = (
        "CREATE INDEX CONCURRENTLY idx_self_recording_probe ON probe (id);\n"
        "INSERT INTO schema_migrations (version, name) "
        "VALUES (900, '900_self_recording_concurrent');"
    )
    (tmp_path / "900_self_recording_concurrent.sql").write_text(source)
    pool = _SingleConnectionPool(tmp_path)

    with pytest.raises(
        RuntimeError, match="atomic-bookkeeping migration cannot use CONCURRENTLY"
    ):
        await run_migrations(pool, migrations_dir=tmp_path)

    assert getattr(pool, "atomic_transactions", 0) == 0
    assert source not in pool.applied_sql


@pytest.mark.asyncio
async def test_mixed_case_quoted_ledger_lookalike_preserves_concurrent_autocommit(tmp_path):
    """A distinct quoted table does not select the atomic/concurrent guard."""
    from atlas_brain.storage.migrations import run_migrations

    source = (
        "CREATE INDEX CONCURRENTLY idx_quoted_ledger_lookalike ON probe (id);\n"
        'INSERT INTO "SCHEMA_MIGRATIONS" (version, name) '
        "VALUES (900, '900_quoted_ledger_lookalike');"
    )
    (tmp_path / "900_quoted_ledger_lookalike.sql").write_text(source)
    pool = _SingleConnectionPool(tmp_path)

    await run_migrations(pool, migrations_dir=tmp_path)

    assert getattr(pool, "atomic_transactions", 0) == 0
    assert source not in pool.applied_sql
    assert any("CREATE INDEX CONCURRENTLY" in sql for sql in pool.applied_sql)


def _stage_historical_387_mismatch(tmp_path, pool):
    from atlas_brain.storage.migrations.reconciliation import (
        MIGRATION_387_RECONCILIATION,
    )

    record = MIGRATION_387_RECONCILIATION
    (tmp_path / f"{record.migration_name}.sql").write_bytes(_migration_387_source())
    pool.records.append((387, record.migration_name, record.historical_ledger_sha256))
    if hasattr(pool, "reconciliation_rows"):
        pool.reconciliation_rows = [{
            "content_sha256": record.historical_ledger_sha256,
            "applied_at": record.observed_applied_at,
        }]
    return record


def _legacy_386_function_body() -> str:
    return (
        _migration_function_body(_migration_386_source())
        .replace(
            "IF OLD.business_context_id = 'effingham_maids'",
            "IF session_user = 'atlas_nocodb'\n"
            "               AND OLD.business_context_id = 'effingham_maids'",
        )
        .replace(
            "before direct contact mutation",
            "before NocoDB can change the contact",
        )
    )


class _ForwardRecoveryTransaction(_RollbackMigrationTransaction):
    """Extend the in-memory atomic seam to include the function/trigger state."""

    async def __aenter__(self):
        await super().__aenter__()
        self.catalog_snapshot = dict(self.pool.won_loss_catalog)
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        result = await super().__aexit__(exc_type, exc, traceback)
        if exc_type is not None:
            self.pool.won_loss_catalog = self.catalog_snapshot
        return result


class _AsyncpgRecordLike:
    """Exercise the catalog boundary where asyncpg.Record is not a Mapping."""

    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values.items())


class _ForwardRecoveryPool(_SerializingPool):
    """Runner fixture whose catalog transitions only when the 390 SQL succeeds."""

    def __init__(self, *, fail_recovery: bool = False):
        super().__init__(honor_lock=True)
        self.fail_recovery = fail_recovery
        self.recovery_attempts = 0
        self.won_loss_catalog = {
            "schema_name": "migration_probe",
            "contacts_relation_ready": True,
            "function_ready": True,
            "function_security_definer": True,
            "function_proconfig": ["search_path=pg_catalog, migration_probe"],
            "function_public_execute_revoked": True,
            "function_body": _legacy_386_function_body(),
            "trigger_ready": True,
            "trigger_enabled": "O",
            "trigger_is_before_row_update_delete": True,
            "trigger_has_no_when_clause": True,
            "trigger_update_columns": ["status"],
        }

    async def fetch(self, query, *args):
        from atlas_brain.storage.migrations.reconciliation import (
            MIGRATION_386_WON_LOSS_FENCE_FORWARD_RECOVERY,
        )

        record = MIGRATION_386_WON_LOSS_FENCE_FORWARD_RECOVERY
        normalized = " ".join(query.split())
        if normalized == (
            "SELECT version, content_sha256, applied_at FROM schema_migrations "
            "WHERE name = $1 LIMIT 2"
        ):
            assert args == (record.migration_name,)
            return [{
                "version": version,
                "content_sha256": digest,
                "applied_at": record.observed_applied_at,
            } for version, name, digest in self.records if name == record.migration_name]
        if normalized == (
            "SELECT version, content_sha256 FROM schema_migrations "
            "WHERE name = $1 LIMIT 2"
        ):
            assert args == (record.recovery_migration_name,)
            return [{
                "version": version,
                "content_sha256": digest,
            } for version, name, digest in self.records if name == record.recovery_migration_name]
        return await super().fetch(query, *args)

    async def fetchrow(self, query, *args):
        if "reject_nocodb_eom_won_loss_mutation" in query:
            assert args == ()
            assert "pg_catalog.pg_trigger AS trigger_state" in query
            assert "trigger_state.tgqual" in query
            return _AsyncpgRecordLike(self.won_loss_catalog)
        raise AssertionError(f"Unexpected fetchrow query: {query}")

    async def execute(self, query, *args):
        if "Forward-only recovery for targets" in query:
            self.recovery_attempts += 1
            if self.fail_recovery:
                raise RuntimeError("injected 390 recovery failure")
            self.won_loss_catalog.update({
                "function_body": _migration_function_body(_migration_386_source()),
                # PostgreSQL exposes tgattr in physical column order rather
                # than the CREATE TRIGGER declaration order.
                "trigger_update_columns": ["contact_type", "status"],
            })
        return await super().execute(query, *args)

    def transaction(self):
        return _ForwardRecoveryTransaction(self)


def _stage_historical_386_forward_recovery(tmp_path, pool):
    from atlas_brain.storage.migrations.reconciliation import (
        MIGRATION_386_WON_LOSS_FENCE_FORWARD_RECOVERY,
    )

    record = MIGRATION_386_WON_LOSS_FENCE_FORWARD_RECOVERY
    (tmp_path / f"{record.migration_name}.sql").write_bytes(_migration_386_source())
    (tmp_path / f"{record.recovery_migration_name}.sql").write_bytes(
        _migration_390_source()
    )
    pool.records.append((386, record.migration_name, record.historical_ledger_sha256))
    return record


def _stage_historical_382_missing_source(pool):
    from atlas_brain.storage.migrations.reconciliation import (
        MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION,
    )

    record = MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION
    pool.records.append((-11, record.migration_name, None))
    if hasattr(pool, "public_onboarding_reconciliation_rows"):
        pool.public_onboarding_reconciliation_rows = [{
            "content_sha256": None,
            "applied_at": record.observed_applied_at,
        }]
    return record


def _stage_historical_067_missing_source(pool):
    from atlas_brain.storage.migrations.reconciliation import (
        MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION,
    )

    record = MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION
    pool.records.append((record.migration_version, record.migration_name, None))
    if hasattr(pool, "b2b_campaign_partner_reconciliation_rows"):
        pool.b2b_campaign_partner_reconciliation_rows = [{
            "version": record.migration_version,
            "content_sha256": None,
            "applied_at": record.observed_applied_at,
        }]
    return record


def _stage_historical_272_missing_source(pool):
    from atlas_brain.storage.migrations.reconciliation import (
        MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION,
    )

    record = MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION
    pool.records.append((record.migration_version, record.migration_name, None))
    if hasattr(pool, "b2b_watchlist_alert_events_reconciliation_rows"):
        pool.b2b_watchlist_alert_events_reconciliation_rows = [{
            "version": record.migration_version,
            "content_sha256": None,
            "applied_at": record.observed_applied_at,
        }]
    return record


def _stage_historical_022b_missing_source(tmp_path, pool):
    from atlas_brain.storage.migrations.reconciliation import (
        MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION,
    )

    record = MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION
    (tmp_path / f"{record.current_packaged_migration_name}.sql").write_bytes(
        _migration_022b_source()
    )
    pool.records.append((22, record.migration_name, None))
    if hasattr(pool, "presence_unknown_count_reconciliation_rows"):
        pool.presence_unknown_count_reconciliation_rows = [
            {
                "content_sha256": None,
                "applied_at": record.observed_applied_at,
            }
        ]
    return record


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "source_present", "expected_category"),
    [
        ("digest mismatch", True, "mismatched=900_recorded"),
        ("missing packaged source", False, "missing_source=900_recorded"),
    ],
)
async def test_unresolved_content_evidence_blocks_pending_migration_before_sql(
    tmp_path,
    case,
    source_present,
    expected_category,
):
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    if source_present:
        (tmp_path / "900_recorded.sql").write_text("SELECT 900")
    (tmp_path / "901_pending.sql").write_text("SELECT 901")
    pool = _SerializingPool(honor_lock=True)
    pool.records.append((900, "900_recorded", "f" * 64))

    with pytest.raises(PendingMigrationContentIntegrityError) as exc_info:
        await run_migrations(pool, migrations_dir=tmp_path)

    assert expected_category in str(exc_info.value), case
    assert pool.applied_sql == [], case
    assert pool.records == [(900, "900_recorded", "f" * 64)], case
    assert pool.inserted_with_digest == [], case


@pytest.mark.asyncio
async def test_386_forward_recovery_runs_before_a_lower_numbered_pending_migration(
    tmp_path,
):
    """The named prelude is an explicit dependency edge, not numeric replay."""
    from atlas_brain.storage.migrations import run_migrations

    pool = _ForwardRecoveryPool()
    record = _stage_historical_386_forward_recovery(tmp_path, pool)
    (tmp_path / "389_later_pending.sql").write_text("SELECT 389")

    await run_migrations(
        pool,
        migrations_dir=tmp_path,
        only={record.recovery_migration_name, "389_later_pending"},
    )

    assert "Forward-only recovery for targets" in pool.applied_sql[0]
    assert pool.applied_sql[1] == "SELECT 389"
    assert pool.inserted_with_digest == [
        (
            record.recovery_migration_version,
            record.recovery_migration_name,
            record.recovery_packaged_sha256,
        ),
        (389, "389_later_pending", hashlib.sha256(b"SELECT 389").hexdigest()),
    ]
    assert pool.atomic_transactions == 1
    assert pool.atomic_transaction_errors == 0


@pytest.mark.asyncio
async def test_386_forward_recovery_stays_closed_when_only_omits_recovery(tmp_path):
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _ForwardRecoveryPool()
    _stage_historical_386_forward_recovery(tmp_path, pool)
    (tmp_path / "389_later_pending.sql").write_text("SELECT 389")

    with pytest.raises(PendingMigrationContentIntegrityError, match="mismatched="):
        await run_migrations(
            pool,
            migrations_dir=tmp_path,
            only={"389_later_pending"},
        )

    assert pool.applied_sql == []
    assert pool.inserted_with_digest == []


@pytest.mark.asyncio
async def test_386_forward_recovery_stays_closed_with_another_unresolved_record(
    tmp_path,
):
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _ForwardRecoveryPool()
    record = _stage_historical_386_forward_recovery(tmp_path, pool)
    pool.records.append((900, "900_other_recorded", "f" * 64))
    (tmp_path / "900_other_recorded.sql").write_text("SELECT 900")
    (tmp_path / "389_later_pending.sql").write_text("SELECT 389")

    with pytest.raises(PendingMigrationContentIntegrityError) as exc_info:
        await run_migrations(
            pool,
            migrations_dir=tmp_path,
            only={record.recovery_migration_name, "389_later_pending"},
        )

    assert "mismatched=386_eom_won_loss_nocodb_fence,900_other_recorded" in str(
        exc_info.value
    )
    assert pool.applied_sql == []
    assert pool.inserted_with_digest == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ("altered catalog", "wrong historical version", "recorded weak recovery"),
)
async def test_386_forward_recovery_requires_the_exact_unrecovered_state(
    tmp_path,
    case,
):
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _ForwardRecoveryPool()
    record = _stage_historical_386_forward_recovery(tmp_path, pool)
    if case == "altered catalog":
        pool.won_loss_catalog["trigger_has_no_when_clause"] = False
    elif case == "wrong historical version":
        pool.records[0] = (
            record.historical_migration_version - 1,
            record.migration_name,
            record.historical_ledger_sha256,
        )
    else:
        pool.records.append((
            record.recovery_migration_version,
            record.recovery_migration_name,
            record.recovery_packaged_sha256,
        ))
    (tmp_path / "389_later_pending.sql").write_text("SELECT 389")

    with pytest.raises(PendingMigrationContentIntegrityError, match="mismatched="):
        await run_migrations(
            pool,
            migrations_dir=tmp_path,
            only={record.recovery_migration_name, "389_later_pending"},
        )

    assert pool.applied_sql == []
    assert pool.inserted_with_digest == []


@pytest.mark.asyncio
async def test_386_forward_recovery_failure_rolls_back_then_retry_applies_once(
    tmp_path,
):
    from atlas_brain.storage.migrations import run_migrations

    pool = _ForwardRecoveryPool(fail_recovery=True)
    record = _stage_historical_386_forward_recovery(tmp_path, pool)
    (tmp_path / "389_later_pending.sql").write_text("SELECT 389")
    requested = {record.recovery_migration_name, "389_later_pending"}

    with pytest.raises(RuntimeError, match="injected 390 recovery failure"):
        await run_migrations(pool, migrations_dir=tmp_path, only=requested)

    assert pool.atomic_transactions == 1
    assert pool.atomic_transaction_errors == 1
    assert pool.applied_sql == []
    assert pool.inserted_with_digest == []
    assert pool.won_loss_catalog["function_body"] == _legacy_386_function_body()

    pool.fail_recovery = False
    await run_migrations(pool, migrations_dir=tmp_path, only=requested)

    assert pool.recovery_attempts == 2
    assert [record[1] for record in pool.inserted_with_digest] == [
        record.recovery_migration_name,
        "389_later_pending",
    ]
    assert pool.atomic_transactions == 2
    assert pool.atomic_transaction_errors == 1
@pytest.mark.asyncio
async def test_attested_historical_mismatch_admits_targeted_pending_migration(tmp_path):
    from atlas_brain.storage.migrations import run_migrations

    pool = _AttestedReconciliationPool()
    _stage_historical_387_mismatch(tmp_path, pool)
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    await run_migrations(
        pool,
        migrations_dir=tmp_path,
        only={"901_pending"},
    )

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_attested_missing_source_admits_targeted_pending_migration(tmp_path):
    from atlas_brain.storage.migrations import run_migrations

    pool = _AttestedReconciliationPool()
    _stage_historical_382_missing_source(pool)
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    await run_migrations(
        pool,
        migrations_dir=tmp_path,
        only={"901_pending"},
    )

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_attested_067_missing_source_admits_targeted_pending_migration(tmp_path):
    """The source-unavailable receipt admits only its exact ledger name."""
    from atlas_brain.storage.migrations import run_migrations

    pool = _AttestedReconciliationPool()
    _stage_historical_067_missing_source(pool)
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    await run_migrations(
        pool,
        migrations_dir=tmp_path,
        only={"901_pending"},
    )

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_attested_272_missing_source_admits_targeted_pending_migration(tmp_path):
    """The synthetic-version receipt admits only its exact ledger name."""
    from atlas_brain.storage.migrations import run_migrations

    pool = _AttestedReconciliationPool()
    _stage_historical_272_missing_source(pool)
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    await run_migrations(
        pool,
        migrations_dir=tmp_path,
        only={"901_pending"},
    )

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_attested_022b_missing_source_admits_targeted_pending_migration(tmp_path):
    """The renamed-source receipt admits only its exact recorded name."""
    from atlas_brain.storage.migrations import run_migrations

    pool = _AttestedReconciliationPool()
    _stage_historical_022b_missing_source(tmp_path, pool)
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    await run_migrations(
        pool,
        migrations_dir=tmp_path,
        only={"901_pending"},
    )

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_failed_historical_attestation_blocks_then_retry_applies_once(tmp_path):
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool(zero_active_null_period_rows=False)
    _stage_historical_387_mismatch(tmp_path, pool)
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(PendingMigrationContentIntegrityError, match="mismatched="):
        await run_migrations(pool, migrations_dir=tmp_path)

    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []

    pool.zero_active_null_period_rows = True
    await run_migrations(pool, migrations_dir=tmp_path)

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_failed_missing_source_attestation_blocks_then_retry_applies_once(tmp_path):
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool()
    record = _stage_historical_382_missing_source(pool)
    pool.public_onboarding_reconciliation_rows = [{
        "content_sha256": "a" * 64,
        "applied_at": record.observed_applied_at,
    }]
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(
        PendingMigrationContentIntegrityError,
        match=f"missing_source={record.migration_name}",
    ):
        await run_migrations(pool, migrations_dir=tmp_path)

    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []

    pool.public_onboarding_reconciliation_rows = [{
        "content_sha256": None,
        "applied_at": record.observed_applied_at,
    }]
    await run_migrations(pool, migrations_dir=tmp_path)

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_failed_067_attestation_blocks_then_retry_applies_once(tmp_path):
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool()
    record = _stage_historical_067_missing_source(pool)
    pool.b2b_campaign_partner_catalog_row["partner_index_is_ready"] = False
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(
        PendingMigrationContentIntegrityError,
        match=f"missing_source={record.migration_name}",
    ):
        await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []

    pool.b2b_campaign_partner_catalog_row["partner_index_is_ready"] = True
    await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_failed_272_unlisted_index_attestation_blocks_then_retry_applies_once(
    tmp_path,
):
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool()
    record = _stage_historical_272_missing_source(pool)
    catalog = pool.b2b_watchlist_alert_events_catalog_row["catalog_evidence"]
    catalog["no_unlisted_alert_event_indexes"] = False
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(
        PendingMigrationContentIntegrityError,
        match=f"missing_source={record.migration_name}",
    ):
        await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []

    catalog["no_unlisted_alert_event_indexes"] = True
    await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_failed_272_write_interceptor_attestation_blocks_then_retry_applies_once(
    tmp_path,
):
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool()
    record = _stage_historical_272_missing_source(pool)
    catalog = pool.b2b_watchlist_alert_events_catalog_row["catalog_evidence"]
    catalog["no_unreviewed_alert_event_write_interceptors"] = False
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(
        PendingMigrationContentIntegrityError,
        match=f"missing_source={record.migration_name}",
    ):
        await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []

    catalog["no_unreviewed_alert_event_write_interceptors"] = True
    await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_failed_272_internal_fk_trigger_attestation_blocks_then_retry_applies_once(
    tmp_path,
):
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool()
    record = _stage_historical_272_missing_source(pool)
    catalog = pool.b2b_watchlist_alert_events_catalog_row["catalog_evidence"]
    account_fk = catalog["constraints"][
        "b2b_watchlist_alert_events_account_id_fkey"
    ]
    expected_trigger_count = int(
        account_fk["origin_enabled_internal_trigger_count"]
    )
    account_fk["origin_enabled_internal_trigger_count"] = expected_trigger_count - 1
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(
        PendingMigrationContentIntegrityError,
        match=f"missing_source={record.migration_name}",
    ):
        await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []

    account_fk["origin_enabled_internal_trigger_count"] = expected_trigger_count
    await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_failed_022b_attestation_blocks_then_retry_applies_once(tmp_path):
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool()
    record = _stage_historical_022b_missing_source(tmp_path, pool)
    pool.presence_unknown_count_reconciliation_rows = [
        {
            "content_sha256": "a" * 64,
            "applied_at": record.observed_applied_at,
        }
    ]
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(
        PendingMigrationContentIntegrityError,
        match=f"missing_source={record.migration_name}",
    ):
        await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []

    pool.presence_unknown_count_reconciliation_rows = [
        {
            "content_sha256": None,
            "applied_at": record.observed_applied_at,
        }
    ]
    await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_non_table_022b_evidence_blocks_then_retry_applies_once(tmp_path):
    """A view or foreign relation can never satisfy the ALTER TABLE receipt."""
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool()
    record = _stage_historical_022b_missing_source(tmp_path, pool)
    pool.presence_events_is_ordinary_table = False
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(
        PendingMigrationContentIntegrityError,
        match=f"missing_source={record.migration_name}",
    ):
        await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []

    pool.presence_events_is_ordinary_table = True
    await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_missing_382_ledger_row_blocks_pending_sql_then_retries_once(tmp_path):
    """The named receipt fails closed when the detailed ledger read is absent."""
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool()
    record = _stage_historical_382_missing_source(pool)
    pool.public_onboarding_reconciliation_rows = []
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(
        PendingMigrationContentIntegrityError,
        match=f"missing_source={record.migration_name}",
    ):
        await run_migrations(pool, migrations_dir=tmp_path)

    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []

    pool.public_onboarding_reconciliation_rows = [{
        "content_sha256": None,
        "applied_at": record.observed_applied_at,
    }]
    await run_migrations(pool, migrations_dir=tmp_path)

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_missing_022b_ledger_row_blocks_pending_sql_then_retries_once(tmp_path):
    """The receipt fails closed when the detailed NULL-digest row is absent."""
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool()
    record = _stage_historical_022b_missing_source(tmp_path, pool)
    pool.presence_unknown_count_reconciliation_rows = []
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(
        PendingMigrationContentIntegrityError,
        match=f"missing_source={record.migration_name}",
    ):
        await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []

    pool.presence_unknown_count_reconciliation_rows = [
        {
            "content_sha256": None,
            "applied_at": record.observed_applied_at,
        }
    ]
    await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]


@pytest.mark.asyncio
async def test_attestation_transport_failure_blocks_then_retry_applies_once(tmp_path):
    """Attestation transport failures fail closed and release the runner lock."""
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    class _FailingAttestationTransportPool(_AttestedReconciliationPool):
        def __init__(self):
            super().__init__()
            self.fail_attestation_transport = True

        async def fetchrow(self, query, *args):
            if self.fail_attestation_transport:
                raise RuntimeError("injected 387 attestation transport failure")
            return await super().fetchrow(query, *args)

    pool = _FailingAttestationTransportPool()
    _stage_historical_387_mismatch(tmp_path, pool)
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(
        PendingMigrationContentIntegrityError,
        match="known historical migration evidence could not be attested",
    ):
        await run_migrations(pool, migrations_dir=tmp_path)

    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []
    assert not pool._lock().locked()
    assert pool.acquired == 0

    pool.fail_attestation_transport = False
    await run_migrations(pool, migrations_dir=tmp_path)

    assert pool.applied_sql == ["SELECT 901"]
    assert pool.inserted_with_digest == [
        (901, "901_pending", hashlib.sha256(b"SELECT 901").hexdigest())
    ]
    assert not pool._lock().locked()
    assert pool.acquired == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "other_source_present", "expected_category"),
    [
        ("unknown digest mismatch", True, "mismatched=900_other_recorded"),
        ("missing packaged source", False, "missing_source=900_other_recorded"),
    ],
)
async def test_attested_known_mismatch_cannot_clear_other_content_evidence(
    tmp_path,
    case,
    other_source_present,
    expected_category,
):
    """A successful 387 attestation clears only its exact reviewed mismatch."""
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool()
    record = _stage_historical_387_mismatch(tmp_path, pool)
    pool.records.append((900, "900_other_recorded", "f" * 64))
    if other_source_present:
        (tmp_path / "900_other_recorded.sql").write_text("SELECT 900")
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(PendingMigrationContentIntegrityError) as exc_info:
        await run_migrations(pool, migrations_dir=tmp_path)

    message = str(exc_info.value)
    assert expected_category in message, case
    assert record.migration_name not in message, case
    assert pool.applied_sql == [], case
    assert pool.inserted == [], case
    assert pool.inserted_with_digest == [], case
    assert pool.updated == [], case
    assert not pool._lock().locked(), case
    assert pool.acquired == 0, case


@pytest.mark.asyncio
async def test_attested_known_missing_source_cannot_clear_other_missing_source(tmp_path):
    """The 382 receipt cannot become a generic missing-source allowlist."""
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool()
    record = _stage_historical_382_missing_source(pool)
    pool.records.append((900, "900_other_recorded", "f" * 64))
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(PendingMigrationContentIntegrityError) as exc_info:
        await run_migrations(pool, migrations_dir=tmp_path)

    message = str(exc_info.value)
    assert "missing_source=900_other_recorded" in message
    assert record.migration_name not in message
    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []
    assert not pool._lock().locked()
    assert pool.acquired == 0


@pytest.mark.asyncio
async def test_attested_022b_missing_source_cannot_clear_other_missing_source(tmp_path):
    """The renamed-source receipt is not a general legacy-name allowlist."""
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool()
    record = _stage_historical_022b_missing_source(tmp_path, pool)
    pool.records.append((900, "900_other_recorded", "f" * 64))
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(PendingMigrationContentIntegrityError) as exc_info:
        await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    message = str(exc_info.value)
    assert "missing_source=900_other_recorded" in message
    assert record.migration_name not in message
    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []
    assert not pool._lock().locked()
    assert pool.acquired == 0


@pytest.mark.asyncio
async def test_attested_272_missing_source_cannot_clear_other_missing_source(tmp_path):
    """The named 272 receipt is not a generic missing-source allowlist."""
    from atlas_brain.storage.migrations import (
        PendingMigrationContentIntegrityError,
        run_migrations,
    )

    pool = _AttestedReconciliationPool()
    record = _stage_historical_272_missing_source(pool)
    pool.records.append((900, "900_other_recorded", "f" * 64))
    (tmp_path / "901_pending.sql").write_text("SELECT 901")

    with pytest.raises(PendingMigrationContentIntegrityError) as exc_info:
        await run_migrations(pool, migrations_dir=tmp_path, only={"901_pending"})

    message = str(exc_info.value)
    assert "missing_source=900_other_recorded" in message
    assert record.migration_name not in message
    assert pool.applied_sql == []
    assert pool.inserted == []
    assert pool.inserted_with_digest == []
    assert pool.updated == []
    assert not pool._lock().locked()
    assert pool.acquired == 0


@pytest.mark.asyncio
async def test_historical_mismatch_without_pending_migration_remains_available(
    tmp_path,
    caplog,
):
    from atlas_brain.storage.migrations import run_migrations

    pool = _SerializingPool(honor_lock=True)
    _stage_historical_387_mismatch(tmp_path, pool)
    caplog.set_level(logging.ERROR, logger="atlas.storage.migrations")

    await run_migrations(pool, migrations_dir=tmp_path)

    assert any(
        "mismatched=387_eom_recurring_invoice_dedup_recovery" in message
        for message in caplog.messages
    )
    assert pool.applied_sql == []
    assert pool.inserted_with_digest == []


@pytest.mark.asyncio
async def test_legacy_null_row_is_reported_but_never_backfilled_by_a_later_run(
    tmp_path,
    caplog,
):
    from atlas_brain.storage.migrations import run_migrations

    (tmp_path / "900_legacy.sql").write_text("SELECT 900")
    (tmp_path / "901_pending.sql").write_text("SELECT 901")
    pool = _SerializingPool(honor_lock=True)
    pool.records.append((900, "900_legacy", None))
    caplog.set_level(logging.INFO, logger="atlas.storage.migrations")

    await run_migrations(pool, migrations_dir=tmp_path)

    assert any(
        "unavailable for 1 legacy ledger rows" in message
        for message in caplog.messages
    )
    assert pool.records[0] == (900, "900_legacy", None)
    assert pool.updated == []


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
        if "pg_try_advisory_lock" in query:
            return True
        return None

    def transaction(self):
        return _RecordingTransaction(self.pool)


class _RecordingTransaction:
    """Small asyncpg transaction stand-in for the marked-migration proof."""

    def __init__(self, pool):
        self.pool = pool

    async def __aenter__(self):
        self.pool.atomic_transactions = getattr(self.pool, "atomic_transactions", 0) + 1
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.pool.atomic_transaction_errors = getattr(
            self.pool, "atomic_transaction_errors", 0
        ) + int(exc_type is not None)
        return False


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
async def test_marked_migration_records_its_ledger_entry_in_one_transaction(tmp_path):
    """An atomic-bookkeeping migration has no SQL-to-ledger crash window."""
    from atlas_brain.storage.migrations import run_migrations

    (tmp_path / "900_atomic.sql").write_text(
        "-- atlas: atomic-bookkeeping\nCREATE TABLE atomic_probe (id int);\n"
    )
    pool = _SingleConnectionPool(tmp_path)

    await run_migrations(pool, migrations_dir=tmp_path)

    assert pool.atomic_transactions == 1
    assert pool.atomic_transaction_errors == 0
    assert any("CREATE TABLE atomic_probe" in sql for sql in pool.applied_sql)


def test_dynamic_self_recording_requires_the_first_line_atomic_marker():
    """Dynamic ledger SQL remains opaque until its exact marker opts it in."""
    from atlas_brain.storage.migrations import (
        _contains_executable_self_recording_insert,
        _requires_atomic_bookkeeping,
    )

    dynamic_source = (
        "DO $$\n"
        "BEGIN\n"
        "    EXECUTE 'INSERT INTO schema_migrations (version, name) "
        "VALUES (900, ''900_dynamic_self_recording'')';\n"
        "END $$;\n"
    )
    late_marker_source = (
        "-- authoring note precedes the marker\n"
        "-- atlas: atomic-bookkeeping\n"
        f"{dynamic_source}"
    )
    marked_source = f"-- atlas: atomic-bookkeeping\n{dynamic_source}"

    assert not _contains_executable_self_recording_insert(dynamic_source)
    assert not _requires_atomic_bookkeeping(late_marker_source)
    assert _requires_atomic_bookkeeping(marked_source)


@pytest.mark.asyncio
async def test_marked_migration_rejects_concurrently_ddl_before_a_transaction(tmp_path):
    """The opt-in cannot silently break concurrent-index migration safety."""
    from atlas_brain.storage.migrations import run_migrations

    (tmp_path / "900_atomic_concurrent.sql").write_text(
        "-- atlas: atomic-bookkeeping\n"
        "CREATE INDEX CONCURRENTLY idx_atomic_probe ON atomic_probe (id);\n"
    )
    pool = _SingleConnectionPool(tmp_path)

    with pytest.raises(
        RuntimeError, match="atomic-bookkeeping migration cannot use CONCURRENTLY"
    ):
        await run_migrations(pool, migrations_dir=tmp_path)

    assert getattr(pool, "atomic_transactions", 0) == 0


@pytest.mark.asyncio
async def test_marked_migration_ignores_concurrently_mentions_in_comments(tmp_path):
    """The atomic guard keys off executable SQL, not rollback notes."""
    from atlas_brain.storage.migrations import run_migrations

    (tmp_path / "900_atomic_comment.sql").write_text(
        "-- atlas: atomic-bookkeeping\n"
        "CREATE TABLE atomic_probe (id int);\n"
        "-- Rollback with CREATE INDEX CONCURRENTLY in a later migration if needed.\n"
    )
    pool = _SingleConnectionPool(tmp_path)

    await run_migrations(pool, migrations_dir=tmp_path)

    assert pool.atomic_transactions == 1
    assert any("CREATE TABLE atomic_probe" in sql for sql in pool.applied_sql)


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
            if "pg_advisory_lock(" in query:
                raise AssertionError(
                    "a BLOCKING advisory lock holds an open transaction while "
                    "it waits, which deadlocks against a holder running "
                    "CREATE INDEX CONCURRENTLY; poll with try-lock instead"
                )
            return await super().execute(query, *args)

    pool.acquire = lambda: _acquire_banning(pool, _TxnBanningConn)
    await run_migrations(pool, migrations_dir=tmp_path)

    assert not started, f"migration run opened a transaction: {started}"
    assert any("CONCURRENTLY" in sql for sql in pool.applied_sql)


@pytest.mark.asyncio
async def test_non_atomic_migration_batches_run_statement_by_statement(tmp_path):
    """Concurrent index repair needs separate autocommit statements.

    PostgreSQL rejects ``DROP/CREATE INDEX CONCURRENTLY`` if the runner batches
    them into one query string. The single acquired connection is preserved; only
    the SQL execution granularity changes.
    """
    from atlas_brain.storage.migrations import run_migrations

    (tmp_path / "001_concurrent_repair.sql").write_text(
        "DROP INDEX CONCURRENTLY IF EXISTS idx_x;\n"
        "CREATE INDEX CONCURRENTLY idx_x ON t (c);\n"
    )
    pool = _SingleConnectionPool(tmp_path)

    await run_migrations(pool, migrations_dir=tmp_path)

    migration_sql = [
        " ".join(sql.split())
        for sql in pool.applied_sql
        if "CONCURRENTLY" in sql
    ]
    assert migration_sql == [
        "DROP INDEX CONCURRENTLY IF EXISTS idx_x;",
        "CREATE INDEX CONCURRENTLY idx_x ON t (c);",
    ]


@pytest.mark.asyncio
async def test_non_concurrent_unmarked_migration_still_runs_as_one_batch(tmp_path):
    """Ordinary migrations keep their previous implicit all-or-nothing batch."""
    from atlas_brain.storage.migrations import run_migrations

    (tmp_path / "001_constraints.sql").write_text(
        "ALTER TABLE tasks ADD CONSTRAINT chk_task_type CHECK (task_type <> '');\n"
        "ALTER TABLE tasks ADD CONSTRAINT chk_task_status CHECK (status <> '');\n"
    )
    pool = _SingleConnectionPool(tmp_path)

    await run_migrations(pool, migrations_dir=tmp_path)

    migration_sql = [
        " ".join(sql.split())
        for sql in pool.applied_sql
        if "ALTER TABLE tasks ADD CONSTRAINT" in sql
    ]
    assert migration_sql == [
        "ALTER TABLE tasks ADD CONSTRAINT chk_task_type CHECK (task_type <> ''); "
        "ALTER TABLE tasks ADD CONSTRAINT chk_task_status CHECK (status <> '');"
    ]


@pytest.mark.asyncio
async def test_non_concurrent_unmarked_migration_ignores_concurrently_comments(tmp_path):
    """Comments must not opt ordinary migrations into autocommit splitting."""
    from atlas_brain.storage.migrations import run_migrations

    (tmp_path / "001_comment_only_concurrent.sql").write_text(
        "ALTER TABLE llm_usage ADD CONSTRAINT chk_usage CHECK (tokens >= 0);\n"
        "DO $$\n"
        "BEGIN\n"
        "    RAISE NOTICE 'no concurrent DDL here';\n"
        "END $$;\n"
        "-- Later replacement can use CREATE INDEX CONCURRENTLY.\n"
        "/* CREATE INDEX CONCURRENTLY cannot run in this migration. */\n"
    )
    pool = _SingleConnectionPool(tmp_path)

    await run_migrations(pool, migrations_dir=tmp_path)

    migration_sql = [
        " ".join(sql.split())
        for sql in pool.applied_sql
        if "ALTER TABLE llm_usage ADD CONSTRAINT" in sql
    ]
    assert migration_sql == [
        "ALTER TABLE llm_usage ADD CONSTRAINT chk_usage CHECK (tokens >= 0); "
        "DO $$ BEGIN RAISE NOTICE 'no concurrent DDL here'; END $$; "
        "-- Later replacement can use CREATE INDEX CONCURRENTLY. "
        "/* CREATE INDEX CONCURRENTLY cannot run in this migration. */"
    ]


def test_sql_statement_splitter_preserves_plpgsql_and_comments():
    from atlas_brain.storage.migrations import _split_sql_statements

    sql = """
    -- leading comment; not a split
    DO $$
    BEGIN
        RAISE NOTICE 'inside; body';
    END $$;
    CREATE INDEX CONCURRENTLY idx_x ON t ("semi;colon");
    """

    statements = _split_sql_statements(sql)

    assert len(statements) == 2
    assert "RAISE NOTICE 'inside; body';" in statements[0]
    assert statements[1] == 'CREATE INDEX CONCURRENTLY idx_x ON t ("semi;colon");'


async def _acquire_banning(pool, conn_cls):
    if pool.in_use:
        raise AssertionError("second connection requested during the run")
    pool.in_use = True
    return conn_cls(pool)


@pytest.mark.asyncio
async def test_real_postgres_concurrent_runners_apply_each_migration_once(tmp_path):
    """R2/R8: the fake-pool proofs above model asyncpg and advisory-lock
    semantics; this one uses them. Two concurrent run_migrations() against a
    real asyncpg pool sized min=max=1 must apply each migration exactly once
    and must not deadlock -- the two properties the single-connection
    session-lock design exists to provide. Skips when no database is wired,
    same as the other real-PostgreSQL probes in this repo."""
    import asyncpg
    from atlas_brain.storage.migrations import run_migrations

    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema = f"atlas_migration_lock_{uuid.uuid4().hex}"
    admin = await asyncpg.connect(database_url)
    pool = None
    other = None
    try:
        await admin.execute(f'CREATE SCHEMA "{schema}"')

        first_source = b"CREATE TABLE migration_lock_probe_a (id integer primary key);"
        second_source = b"CREATE TABLE migration_lock_probe_b (id integer primary key);"
        concurrent_index_source = (
            b"CREATE INDEX CONCURRENTLY migration_lock_probe_idx "
            b"ON migration_lock_probe_a (id);"
        )
        self_recording_source = (
            b"CREATE TABLE migration_lock_probe_self_recording "
            b"(id integer primary key);\n"
            b"INSERT INTO schema_migrations (version, name, content_sha256) "
            b"VALUES (4, '004_self_recording', 'wrong-digest');"
        )
        (tmp_path / "001_first.sql").write_bytes(first_source)
        (tmp_path / "002_second.sql").write_bytes(second_source)
        # Statement Postgres refuses inside a transaction block: it passes only
        # because the run holds a SESSION-level lock on an autocommit
        # connection, not a transaction-scoped one.
        (tmp_path / "003_concurrent_index.sql").write_bytes(concurrent_index_source)
        (tmp_path / "004_self_recording.sql").write_bytes(self_recording_source)

        # TWO INDEPENDENT pools -> two independent backend sessions. A single
        # max_size=1 pool would serialize the runners in acquire() before they
        # ever reached the lock, so the advisory lock could be deleted outright
        # and the test would still pass. Contention has to be between separate
        # sessions or it is not the lock being measured.
        pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=1,
            server_settings={"search_path": f"{schema},public"},
        )
        other = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=1,
            server_settings={"search_path": f"{schema},public"},
        )

        await asyncio.wait_for(
            asyncio.gather(
                run_migrations(pool, migrations_dir=tmp_path),
                run_migrations(other, migrations_dir=tmp_path),
            ),
            timeout=60,
        )

        await admin.execute(f'SET search_path TO "{schema}", public')
        applied = await admin.fetch(
            "SELECT name, content_sha256 FROM schema_migrations ORDER BY name"
        )
        assert {
            row["name"]: row["content_sha256"] for row in applied
        } == {
            "001_first": hashlib.sha256(first_source).hexdigest(),
            "002_second": hashlib.sha256(second_source).hexdigest(),
            "003_concurrent_index": hashlib.sha256(concurrent_index_source).hexdigest(),
            "004_self_recording": hashlib.sha256(self_recording_source).hexdigest(),
        }, "each migration must be recorded exactly once with its exact source identity"

        index_rows = await admin.fetch(
            "SELECT indexname FROM pg_indexes "
            "WHERE schemaname = $1 AND indexname = $2",
            schema,
            "migration_lock_probe_idx",
        )
        assert index_rows, (
            "CREATE INDEX CONCURRENTLY must have run -- if the runner opened a "
            "transaction, Postgres would have rejected it"
        )

        dynamic_retry_source = (
            b"-- atlas: atomic-bookkeeping\n"
            b"CREATE TABLE migration_lock_probe_dynamic_retry "
            b"(id integer primary key);\n"
            b"DO $$\n"
            b"BEGIN\n"
            b"    EXECUTE 'INSERT INTO schema_migrations (version, name) "
            b"VALUES (5, ''005_dynamic_self_recording_retry'')';\n"
            b"END $$;\n"
        )
        (tmp_path / "005_dynamic_self_recording_retry.sql").write_bytes(
            dynamic_retry_source
        )
        await admin.execute(
            """
            CREATE OR REPLACE FUNCTION suppress_dynamic_self_recording_retry_digest()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF NEW.name = '005_dynamic_self_recording_retry'
                   AND NEW.content_sha256 IS NOT NULL THEN
                    RETURN NULL;
                END IF;
                RETURN NEW;
            END;
            $$
            """
        )
        await admin.execute(
            """
            CREATE TRIGGER suppress_dynamic_self_recording_retry_digest_trigger
            BEFORE UPDATE OF content_sha256 ON schema_migrations
            FOR EACH ROW
            EXECUTE FUNCTION suppress_dynamic_self_recording_retry_digest()
            """
        )

        with pytest.raises(RuntimeError, match="did not persist its expected content SHA-256"):
            await run_migrations(
                pool,
                migrations_dir=tmp_path,
                only={"005_dynamic_self_recording_retry"},
            )

        assert await admin.fetchval(
            "SELECT to_regclass($1)", "migration_lock_probe_dynamic_retry"
        ) is None
        assert await admin.fetch(
            "SELECT name FROM schema_migrations WHERE name = $1",
            "005_dynamic_self_recording_retry",
        ) == []

        await admin.execute(
            "DROP TRIGGER suppress_dynamic_self_recording_retry_digest_trigger "
            "ON schema_migrations"
        )
        await admin.execute("DROP FUNCTION suppress_dynamic_self_recording_retry_digest()")
        await run_migrations(
            pool,
            migrations_dir=tmp_path,
            only={"005_dynamic_self_recording_retry"},
        )

        assert await admin.fetchval(
            "SELECT to_regclass($1)", "migration_lock_probe_dynamic_retry"
        ) is not None
        persisted_dynamic_row = await admin.fetchrow(
            "SELECT version, content_sha256 FROM schema_migrations WHERE name = $1",
            "005_dynamic_self_recording_retry",
        )
        assert persisted_dynamic_row["version"] == 5
        assert persisted_dynamic_row["content_sha256"] == hashlib.sha256(
            dynamic_retry_source
        ).hexdigest()

        # The SESSION lock outlives its transaction, so it must be released
        # explicitly or the pooled connection carries it into unrelated work.
        # Scoped to THIS pool's backend: the running app uses the same key, so
        # a global count would be a coin flip.
        backend_pid = await pool.fetchval("SELECT pg_backend_pid()")
        held = await admin.fetchval(
            "SELECT count(*) FROM pg_locks "
            "WHERE locktype = 'advisory' AND objid = $1 AND pid = $2",
            0x41544C41,
            backend_pid,
        )
        assert held == 0, (
            "the session advisory lock must be released back to the pooled "
            "connection, not carried into the next borrower"
        )
    finally:
        for p in (pool, other):
            if p is not None:
                await p.close()
        await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin.close()
