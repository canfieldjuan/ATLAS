"""Immutable evidence for named historical migration-source gaps.

This module records what Atlas can prove about a known historical mismatch
without rewriting the migration ledger or treating unavailable source bytes as
verified. It is intentionally narrow: every additional historical mismatch
requires its own reviewed evidence record and catalog predicate.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HISTORICAL_SOURCE_UNAVAILABLE = "historical_source_unavailable"


@dataclass(frozen=True)
class HistoricalMigrationReconciliation:
    """Reviewed facts for one historical migration-source discrepancy."""

    reconciliation_id: str
    migration_name: str
    historical_ledger_sha256: str
    final_packaged_sha256: str
    observed_applied_at: datetime
    earliest_retained_source_commit_at: datetime

    @property
    def source_verification(self) -> str:
        """Keep this evidence record permanently distinct from source verification."""
        return HISTORICAL_SOURCE_UNAVAILABLE


@dataclass(frozen=True)
class HistoricalMissingSourceReconciliation:
    """Reviewed facts for one legacy migration whose source bytes are absent."""

    reconciliation_id: str
    migration_name: str
    historical_ledger_sha256: None
    observed_applied_at: datetime

    @property
    def source_verification(self) -> str:
        """No catalog predicate can recover unavailable historical source bytes."""
        return HISTORICAL_SOURCE_UNAVAILABLE


@dataclass(frozen=True)
class MigrationReconciliationAttestation:
    """Current read-only checks against one immutable evidence record."""

    reconciliation_id: str
    migration_name: str
    exactly_one_ledger_row: bool
    ledger_digest_matches_record: bool
    packaged_digest_matches_record: bool
    applied_at_matches_record: bool
    applied_before_retained_source: bool
    recurring_schema_ready: bool
    zero_active_null_period_recurring_rows: bool | None

    @property
    def source_verification(self) -> str:
        """Catalog attestation never upgrades unavailable source evidence."""
        return HISTORICAL_SOURCE_UNAVAILABLE

    @property
    def status(self) -> str:
        if all((
            self.exactly_one_ledger_row,
            self.ledger_digest_matches_record,
            self.packaged_digest_matches_record,
            self.applied_at_matches_record,
            self.applied_before_retained_source,
            self.recurring_schema_ready,
            self.zero_active_null_period_recurring_rows,
        )):
            return "attested"
        return "not_attested"

    def as_payload(self) -> dict[str, object]:
        """Return an operator-safe result without exposing invoice rows."""
        return {
            "reconciliation_id": self.reconciliation_id,
            "migration_name": self.migration_name,
            "source_verification": self.source_verification,
            "exactly_one_ledger_row": self.exactly_one_ledger_row,
            "ledger_digest_matches_record": self.ledger_digest_matches_record,
            "packaged_digest_matches_record": self.packaged_digest_matches_record,
            "applied_at_matches_record": self.applied_at_matches_record,
            "applied_before_retained_source": self.applied_before_retained_source,
            "recurring_schema_ready": self.recurring_schema_ready,
            "zero_active_null_period_recurring_rows": (
                self.zero_active_null_period_recurring_rows
            ),
            "status": self.status,
        }


@dataclass(frozen=True)
class MissingSourceMigrationReconciliationAttestation:
    """Current read-only proof for one named legacy missing-source record."""

    reconciliation_id: str
    migration_name: str
    exactly_one_ledger_row: bool
    ledger_digest_is_null: bool
    applied_at_matches_record: bool
    immutable_projection_ready: bool
    fingerprint_check_ready: bool
    terminal_state_check_ready: bool
    issued_contact_index_ready: bool
    status_index_ready: bool

    @property
    def source_verification(self) -> str:
        """Admission evidence never upgrades missing source bytes to verified."""
        return HISTORICAL_SOURCE_UNAVAILABLE

    @property
    def status(self) -> str:
        if all((
            self.exactly_one_ledger_row,
            self.ledger_digest_is_null,
            self.applied_at_matches_record,
            self.immutable_projection_ready,
            self.fingerprint_check_ready,
            self.terminal_state_check_ready,
            self.issued_contact_index_ready,
            self.status_index_ready,
        )):
            return "attested"
        return "not_attested"

    def as_payload(self) -> dict[str, object]:
        """Return structural catalog evidence without token or customer data."""
        return {
            "reconciliation_id": self.reconciliation_id,
            "migration_name": self.migration_name,
            "source_verification": self.source_verification,
            "exactly_one_ledger_row": self.exactly_one_ledger_row,
            "ledger_digest_is_null": self.ledger_digest_is_null,
            "applied_at_matches_record": self.applied_at_matches_record,
            "immutable_projection_ready": self.immutable_projection_ready,
            "fingerprint_check_ready": self.fingerprint_check_ready,
            "terminal_state_check_ready": self.terminal_state_check_ready,
            "issued_contact_index_ready": self.issued_contact_index_ready,
            "status_index_ready": self.status_index_ready,
            "status": self.status,
        }


MIGRATION_387_RECONCILIATION = HistoricalMigrationReconciliation(
    reconciliation_id="eom-migration-387-recurring-invoice-dedup-recovery",
    migration_name="387_eom_recurring_invoice_dedup_recovery",
    historical_ledger_sha256=(
        "1dae95d216bfdc836461943af1c6ce382ff7dd21b92eff41d4c94088f72315b2"
    ),
    final_packaged_sha256=(
        "f6382a07d807f7b38772e9823c66f1e47e4118841611e259220d9ab654c84f3d"
    ),
    # The production ledger stores ``CURRENT_TIMESTAMP`` at microsecond
    # precision.  This record is immutable target evidence, so retaining the
    # observed precision is required for the exact equality check below; a
    # seconds-only literal would make the known record permanently unattestable.
    observed_applied_at=datetime(
        2026,
        8,
        21,
        1,
        30,
        46,
        82_989,
        tzinfo=timezone.utc,
    ),
    earliest_retained_source_commit_at=datetime(
        2026, 8, 21, 3, 21, 35, tzinfo=timezone.utc
    ),
)


MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION = (
    HistoricalMissingSourceReconciliation(
        reconciliation_id="eom-migration-382-public-onboarding-source-absence",
        migration_name="382_eom_public_onboarding_tokens",
        historical_ledger_sha256=None,
        observed_applied_at=datetime(
            2026,
            8,
            17,
            19,
            18,
            7,
            242_686,
            tzinfo=timezone.utc,
        ),
    )
)


_HISTORICAL_MISMATCH_RECONCILIATIONS = (MIGRATION_387_RECONCILIATION,)
_HISTORICAL_MISSING_SOURCE_RECONCILIATIONS = (
    MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION,
)


_PUBLIC_ONBOARDING_TOKEN_IMMUTABLE_COLUMNS = {
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
_PUBLIC_ONBOARDING_TOKEN_CONSTRAINT_EXPRESSIONS = {
    "eom_public_onboarding_tokens_signing_key_fingerprint_check": (
        "((signing_key_fingerprint) ~ ^[0-9a-f]{64}$)"
    ),
    "ck_eom_public_onboarding_tokens_terminal_state": (
        "((((status) = issued) and (redeemed_at is null) and "
        "(revoked_at is null) and (handoff_id is null)) or (((status) = "
        "redeemed) and (redeemed_at is not null) and (revoked_at is null) "
        "and (handoff_id is not null)) or (((status) = revoked) and "
        "(redeemed_at is null) and (revoked_at is not null) and "
        "(handoff_id is null)))"
    ),
}
_PUBLIC_ONBOARDING_TOKEN_ISSUED_CONTACT_INDEX = (
    "uq_eom_public_onboarding_tokens_issued_contact"
)
_PUBLIC_ONBOARDING_TOKEN_STATUS_INDEX = "idx_eom_public_onboarding_tokens_status"
_PUBLIC_ONBOARDING_TOKEN_ISSUED_CONTACT_PREDICATE = "((status) = issued)"
_PUBLIC_ONBOARDING_TOKEN_ISSUED_CONTACT_INDEX_FRAGMENT = "usingbtree(contact_id)"
_PUBLIC_ONBOARDING_TOKEN_STATUS_INDEX_FRAGMENT = "usingbtree(status,issued_atdesc)"


def known_historical_migration_reconciliation_names() -> frozenset[str]:
    """Return reviewed mismatch names eligible for current attestation.

    The runner derives this closed set from the source-controlled evidence
    module instead of maintaining a second exception list. Adding another name
    still requires its own reviewed record and catalog predicate here.
    """
    return frozenset(
        record.migration_name for record in _HISTORICAL_MISMATCH_RECONCILIATIONS
    )


def known_historical_missing_source_reconciliation_names() -> frozenset[str]:
    """Return the closed set of missing-source names with reviewed receipts."""
    return frozenset(
        record.migration_name
        for record in _HISTORICAL_MISSING_SOURCE_RECONCILIATIONS
    )


def known_historical_reconciliation_names() -> frozenset[str]:
    """Return every reviewed name without creating a second caller allowlist."""
    return (
        known_historical_migration_reconciliation_names()
        | known_historical_missing_source_reconciliation_names()
    )


def _packaged_migration_digest(
    migration_files: Collection[Path],
    migration_name: str,
) -> str | None:
    """Return a packaged migration digest, or no evidence when it is unreadable."""
    migration_file = next(
        (path for path in migration_files if path.stem == migration_name),
        None,
    )
    if migration_file is None:
        return None
    try:
        return hashlib.sha256(migration_file.read_bytes()).hexdigest()
    except OSError:
        return None


def _normalize_utc(value: object) -> datetime | None:
    """Accept only timezone-aware ledger timestamps as provenance evidence."""
    if not isinstance(value, datetime) or value.tzinfo is None:
        return None
    return value.astimezone(timezone.utc)


async def _migration_387_catalog_evidence(executor: Any) -> tuple[bool, bool | None]:
    """Read the current final catalog without returning any invoice data."""
    recurring_schema_ready = await _recurring_invoice_dedup_schema_ready(executor)
    if not recurring_schema_ready:
        return False, None

    zero_active_null_period_rows = bool(
        await executor.fetchval(
            """
            SELECT NOT EXISTS (
                SELECT 1
                FROM invoices
                WHERE source IN ('monthly_auto', 'eom_commercial_billing')
                  AND status <> 'void'
                  AND billing_period IS NULL
            )
            """
        )
    )
    return True, zero_active_null_period_rows


async def _recurring_invoice_dedup_schema_ready(executor: Any) -> bool:
    """Load the writer-owned readiness predicate only for the opted-in probe."""
    from ..recurring_invoice_schema import recurring_invoice_dedup_schema_ready

    return await recurring_invoice_dedup_schema_ready(executor)


def _normalize_schema_definition(definition: object) -> str:
    """Return stable lower-case PostgreSQL catalog text for exact comparison."""
    return " ".join(str(definition or "").lower().split())


def _canonicalize_catalog_constraint_expression(expression: object) -> str:
    """Normalize PostgreSQL casts without weakening an exact check predicate."""
    normalized = _normalize_schema_definition(expression)
    normalized = re.sub(
        r"::(?:character varying|varchar|text|name)(?:\[\])?",
        "",
        normalized,
    )
    return " ".join(normalized.replace("'", "").split())


def _canonicalize_catalog_index_definition(definition: object) -> str:
    """Compact a catalog index definition before checking keys and direction."""
    return re.sub(r"\s+", "", _normalize_schema_definition(definition))


async def _public_onboarding_token_index_ready(
    executor: Any,
    *,
    index_name: str,
    unique: bool,
    key_columns: tuple[str, ...],
    predicate: str | None,
    definition_fragment: str,
) -> bool:
    """Require the exact named final index without reading token rows."""
    index_row = await executor.fetchrow(
        """
        SELECT
            index_state.indisunique,
            index_state.indisvalid,
            index_state.indisready,
            index_state.indnkeyatts,
            pg_get_indexdef(index_state.indexrelid, 1, true) AS key_column_1,
            pg_get_indexdef(index_state.indexrelid, 2, true) AS key_column_2,
            pg_get_indexdef(index_state.indexrelid) AS definition,
            pg_get_expr(index_state.indpred, index_state.indrelid) AS predicate
        FROM pg_index AS index_state
        JOIN pg_class AS table_class
          ON table_class.oid = index_state.indrelid
        JOIN pg_namespace AS table_namespace
          ON table_namespace.oid = table_class.relnamespace
        JOIN pg_class AS index_class
          ON index_class.oid = index_state.indexrelid
        WHERE table_namespace.nspname = current_schema()
          AND table_class.relname = 'eom_public_onboarding_tokens'
          AND index_class.relname = $1
        """,
        index_name,
    )
    if not index_row:
        return False
    observed_columns = tuple(
        str(index_row[f"key_column_{position}"] or "")
        for position in range(1, len(key_columns) + 1)
    )
    observed_predicate = index_row["predicate"]
    predicate_ready = (
        observed_predicate is None
        if predicate is None
        else _canonicalize_catalog_constraint_expression(observed_predicate) == predicate
    )
    return (
        bool(index_row["indisunique"]) is unique
        and bool(index_row["indisvalid"])
        and bool(index_row["indisready"])
        and int(index_row["indnkeyatts"] or 0) == len(key_columns)
        and observed_columns == key_columns
        and definition_fragment
        in _canonicalize_catalog_index_definition(index_row["definition"])
        and predicate_ready
    )


async def _migration_382_catalog_evidence(
    executor: Any,
) -> tuple[bool, bool, bool, bool, bool]:
    """Read only final immutable token catalog metadata, never token values."""
    column_rows = await executor.fetch(
        """
        SELECT
            actual.column_name,
            actual.data_type,
            actual.character_maximum_length,
            actual.is_nullable
        FROM information_schema.columns AS actual
        WHERE actual.table_schema = current_schema()
          AND actual.table_name = 'eom_public_onboarding_tokens'
          AND actual.column_name = ANY($1::text[])
        """,
        list(_PUBLIC_ONBOARDING_TOKEN_IMMUTABLE_COLUMNS),
    )
    observed_columns = {
        row["column_name"]: (
            row["data_type"],
            row["character_maximum_length"],
            row["is_nullable"],
        )
        for row in column_rows
    }
    immutable_projection_ready = (
        observed_columns == _PUBLIC_ONBOARDING_TOKEN_IMMUTABLE_COLUMNS
    )

    constraint_rows = await executor.fetch(
        """
        SELECT actual.conname, pg_get_expr(actual.conbin, actual.conrelid) AS definition
        FROM pg_constraint AS actual
        JOIN pg_class AS table_class
          ON table_class.oid = actual.conrelid
        JOIN pg_namespace AS table_namespace
          ON table_namespace.oid = table_class.relnamespace
        WHERE table_namespace.nspname = current_schema()
          AND table_class.relname = 'eom_public_onboarding_tokens'
          AND actual.conname = ANY($1::text[])
        """,
        list(_PUBLIC_ONBOARDING_TOKEN_CONSTRAINT_EXPRESSIONS),
    )
    observed_constraints = {
        row["conname"]: row["definition"]
        for row in constraint_rows
    }
    fingerprint_check_ready = (
        _canonicalize_catalog_constraint_expression(
            observed_constraints.get(
                "eom_public_onboarding_tokens_signing_key_fingerprint_check"
            )
        )
        == _PUBLIC_ONBOARDING_TOKEN_CONSTRAINT_EXPRESSIONS[
            "eom_public_onboarding_tokens_signing_key_fingerprint_check"
        ]
    )
    terminal_state_check_ready = (
        _canonicalize_catalog_constraint_expression(
            observed_constraints.get(
                "ck_eom_public_onboarding_tokens_terminal_state"
            )
        )
        == _PUBLIC_ONBOARDING_TOKEN_CONSTRAINT_EXPRESSIONS[
            "ck_eom_public_onboarding_tokens_terminal_state"
        ]
    )
    issued_contact_index_ready = await _public_onboarding_token_index_ready(
        executor,
        index_name=_PUBLIC_ONBOARDING_TOKEN_ISSUED_CONTACT_INDEX,
        unique=True,
        key_columns=("contact_id",),
        predicate=_PUBLIC_ONBOARDING_TOKEN_ISSUED_CONTACT_PREDICATE,
        definition_fragment=_PUBLIC_ONBOARDING_TOKEN_ISSUED_CONTACT_INDEX_FRAGMENT,
    )
    status_index_ready = await _public_onboarding_token_index_ready(
        executor,
        index_name=_PUBLIC_ONBOARDING_TOKEN_STATUS_INDEX,
        unique=False,
        key_columns=("status", "issued_at"),
        predicate=None,
        definition_fragment=_PUBLIC_ONBOARDING_TOKEN_STATUS_INDEX_FRAGMENT,
    )
    return (
        immutable_projection_ready,
        fingerprint_check_ready,
        terminal_state_check_ready,
        issued_contact_index_ready,
        status_index_ready,
    )


async def _attest_migration_387(
    executor: Any,
    migration_files: Collection[Path],
) -> MigrationReconciliationAttestation:
    """Attest the one reviewed 387 discrepancy without source verification."""
    record = MIGRATION_387_RECONCILIATION
    ledger_rows = await executor.fetch(
        "SELECT content_sha256, applied_at FROM schema_migrations WHERE name = $1 LIMIT 2",
        record.migration_name,
    )
    exactly_one_ledger_row = len(ledger_rows) == 1
    ledger_row = ledger_rows[0] if exactly_one_ledger_row else None
    recorded_digest = ledger_row["content_sha256"] if ledger_row is not None else None
    applied_at = _normalize_utc(ledger_row["applied_at"] if ledger_row is not None else None)
    packaged_digest = _packaged_migration_digest(migration_files, record.migration_name)
    recurring_schema_ready, zero_active_null_period_rows = await _migration_387_catalog_evidence(
        executor
    )

    return MigrationReconciliationAttestation(
        reconciliation_id=record.reconciliation_id,
        migration_name=record.migration_name,
        exactly_one_ledger_row=exactly_one_ledger_row,
        ledger_digest_matches_record=recorded_digest == record.historical_ledger_sha256,
        packaged_digest_matches_record=packaged_digest == record.final_packaged_sha256,
        applied_at_matches_record=applied_at == record.observed_applied_at,
        applied_before_retained_source=(
            applied_at is not None
            and applied_at < record.earliest_retained_source_commit_at
        ),
        recurring_schema_ready=recurring_schema_ready,
        zero_active_null_period_recurring_rows=zero_active_null_period_rows,
    )


async def _attest_migration_382(
    executor: Any,
) -> MissingSourceMigrationReconciliationAttestation:
    """Attest only the reviewed 382 source absence from ledger and catalog facts."""
    record = MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION
    ledger_rows = await executor.fetch(
        "SELECT content_sha256, applied_at FROM schema_migrations WHERE name = $1 LIMIT 2",
        record.migration_name,
    )
    exactly_one_ledger_row = len(ledger_rows) == 1
    ledger_row = ledger_rows[0] if exactly_one_ledger_row else None
    recorded_digest = ledger_row["content_sha256"] if ledger_row is not None else None
    applied_at = _normalize_utc(ledger_row["applied_at"] if ledger_row is not None else None)
    (
        immutable_projection_ready,
        fingerprint_check_ready,
        terminal_state_check_ready,
        issued_contact_index_ready,
        status_index_ready,
    ) = await _migration_382_catalog_evidence(executor)

    return MissingSourceMigrationReconciliationAttestation(
        reconciliation_id=record.reconciliation_id,
        migration_name=record.migration_name,
        exactly_one_ledger_row=exactly_one_ledger_row,
        ledger_digest_is_null=(
            exactly_one_ledger_row
            and recorded_digest == record.historical_ledger_sha256
        ),
        applied_at_matches_record=applied_at == record.observed_applied_at,
        immutable_projection_ready=immutable_projection_ready,
        fingerprint_check_ready=fingerprint_check_ready,
        terminal_state_check_ready=terminal_state_check_ready,
        issued_contact_index_ready=issued_contact_index_ready,
        status_index_ready=status_index_ready,
    )


async def attest_known_historical_migration_reconciliations(
    executor: Any,
    migration_files: Collection[Path],
    *,
    candidate_names: Collection[str] | None = None,
) -> tuple[
    MigrationReconciliationAttestation | MissingSourceMigrationReconciliationAttestation,
    ...,
]:
    """Return read-only attestation for reported, reviewed source gaps.

    This is intentionally not a generic exception mechanism. Every additional
    gap requires a reviewed evidence record and a named catalog probe here.
    Omitting ``candidate_names`` preserves the historical 387-only helper
    behavior for existing callers; the runner and read-only preflight pass
    their exact report-derived candidate names explicitly.
    """
    requested_names = (
        known_historical_migration_reconciliation_names()
        if candidate_names is None
        else frozenset(candidate_names)
    )
    attestations: list[
        MigrationReconciliationAttestation
        | MissingSourceMigrationReconciliationAttestation
    ] = []
    if MIGRATION_387_RECONCILIATION.migration_name in requested_names:
        attestations.append(await _attest_migration_387(executor, migration_files))
    if (
        MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION.migration_name
        in requested_names
    ):
        attestations.append(await _attest_migration_382(executor))
    return tuple(attestations)
