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
HISTORICAL_LEDGER_DIGEST_UNAVAILABLE = "historical_ledger_digest_unavailable"


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
class HistoricalRenamedMissingSourceReconciliation:
    """Facts for one NULL-digest ledger name whose source was later renamed.

    The retained package bytes make the source-history claim reviewable, but a
    NULL historical ledger digest means Atlas cannot prove those exact bytes
    were the bytes executed at the old ledger name.  This record therefore
    remains a narrow admission receipt rather than a recovered checksum.
    """

    reconciliation_id: str
    migration_name: str
    current_packaged_migration_name: str
    historical_ledger_sha256: None
    retained_source_sha256: str
    observed_applied_at: datetime
    retained_source_history_commit_ids: tuple[str, ...]

    @property
    def source_verification(self) -> str:
        """A retained rename chain cannot fill the historical NULL digest."""
        return HISTORICAL_LEDGER_DIGEST_UNAVAILABLE


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
    base_token_contract_ready: bool
    required_constraints_ready: bool
    fingerprint_check_ready: bool
    terminal_state_check_ready: bool
    issued_contact_index_ready: bool
    status_index_ready: bool

    @property
    def complete_token_schema_ready(self) -> bool:
        """Require every writer-used token field and constraint, not a projection."""
        return all((
            self.immutable_projection_ready,
            self.base_token_contract_ready,
            self.required_constraints_ready,
            self.fingerprint_check_ready,
            self.terminal_state_check_ready,
            self.issued_contact_index_ready,
            self.status_index_ready,
        ))

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
            self.complete_token_schema_ready,
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
            "base_token_contract_ready": self.base_token_contract_ready,
            "required_constraints_ready": self.required_constraints_ready,
            "complete_token_schema_ready": self.complete_token_schema_ready,
            "fingerprint_check_ready": self.fingerprint_check_ready,
            "terminal_state_check_ready": self.terminal_state_check_ready,
            "issued_contact_index_ready": self.issued_contact_index_ready,
            "status_index_ready": self.status_index_ready,
            "status": self.status,
        }


@dataclass(frozen=True)
class RenamedMissingSourceMigrationReconciliationAttestation:
    """Read-only evidence for one named NULL-digest source rename."""

    reconciliation_id: str
    migration_name: str
    exactly_one_ledger_row: bool
    ledger_digest_is_null: bool
    applied_at_matches_record: bool
    retained_packaged_digest_matches_record: bool
    presence_events_is_ordinary_table: bool
    unknown_count_column_ready: bool
    unknown_count_has_no_constraints: bool

    @property
    def source_verification(self) -> str:
        """Keep retained package evidence distinct from a ledger checksum."""
        return HISTORICAL_LEDGER_DIGEST_UNAVAILABLE

    @property
    def status(self) -> str:
        if all(
            (
                self.exactly_one_ledger_row,
                self.ledger_digest_is_null,
                self.applied_at_matches_record,
                self.retained_packaged_digest_matches_record,
                self.presence_events_is_ordinary_table,
                self.unknown_count_column_ready,
                self.unknown_count_has_no_constraints,
            )
        ):
            return "attested"
        return "not_attested"

    def as_payload(self) -> dict[str, object]:
        """Return structural evidence without reading presence-event rows."""
        return {
            "reconciliation_id": self.reconciliation_id,
            "migration_name": self.migration_name,
            "source_verification": self.source_verification,
            "exactly_one_ledger_row": self.exactly_one_ledger_row,
            "ledger_digest_is_null": self.ledger_digest_is_null,
            "applied_at_matches_record": self.applied_at_matches_record,
            "retained_packaged_digest_matches_record": (
                self.retained_packaged_digest_matches_record
            ),
            "presence_events_is_ordinary_table": (
                self.presence_events_is_ordinary_table
            ),
            "unknown_count_column_ready": self.unknown_count_column_ready,
            "unknown_count_has_no_constraints": self.unknown_count_has_no_constraints,
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


MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION = (
    HistoricalRenamedMissingSourceReconciliation(
        reconciliation_id="presence-migration-022b-unknown-count-source-rename",
        migration_name="022b_presence_unknown_count",
        current_packaged_migration_name="027_presence_unknown_count",
        historical_ledger_sha256=None,
        retained_source_sha256=(
            "30ae96a2b8f85aea912bdd55599c7f27cb972e5a2cb9f20572798feb52d1b0aa"
        ),
        observed_applied_at=datetime(
            2026,
            2,
            17,
            23,
            34,
            17,
            949_845,
            tzinfo=timezone.utc,
        ),
        retained_source_history_commit_ids=(
            "72c008b40d134bf1e0432e5c586bf0e156a1780b",
            "5df3f12fa9aa1721983b37cceca913803db27722",
            "2ec15ce4d8f159dd773405dfb311da4219d531aa",
        ),
    )
)


_HISTORICAL_MISMATCH_RECONCILIATIONS = (MIGRATION_387_RECONCILIATION,)
_HISTORICAL_MISSING_SOURCE_RECONCILIATIONS = (
    MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION,
    MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION,
)


_PRESENCE_UNKNOWN_COUNT_COLUMN = ("integer", "YES", "0")


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
_PUBLIC_ONBOARDING_TOKEN_BASE_COLUMNS = {
    "id": ("uuid", None, "NO", None),
    "draft_id": ("uuid", None, "NO", None),
    "contact_id": ("uuid", None, "NO", None),
    "approval_key": ("character varying", 128, "NO", None),
    "status": ("character varying", 16, "NO", "issued"),
    "approved_by_employee_id": ("bigint", None, "NO", None),
    "approved_by_name": ("character varying", 128, "NO", None),
    "issued_at": ("timestamp with time zone", None, "NO", "now()"),
    "redeemed_at": ("timestamp with time zone", None, "YES", None),
    "revoked_at": ("timestamp with time zone", None, "YES", None),
    "handoff_id": ("uuid", None, "YES", None),
}
_PUBLIC_ONBOARDING_TOKEN_COLUMNS = {
    **{
        name: (*signature, None)
        for name, signature in _PUBLIC_ONBOARDING_TOKEN_IMMUTABLE_COLUMNS.items()
    },
    **_PUBLIC_ONBOARDING_TOKEN_BASE_COLUMNS,
}
_PUBLIC_ONBOARDING_TOKEN_CONSTRAINT_EXPRESSIONS = {
    "ck_eom_public_onboarding_tokens_status": (
        "((status) = any ((array[issued, redeemed, revoked])))"
    ),
    "eom_public_onboarding_tokens_approved_by_employee_id_check": (
        "(approved_by_employee_id > 0)"
    ),
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


@dataclass(frozen=True)
class _PublicOnboardingTokenConstraint:
    """Exact structural requirement for one final token-table constraint."""

    constraint_type: str
    key_columns: tuple[str, ...]
    referenced_table: str | None = None
    referenced_columns: tuple[str, ...] = ()
    delete_action: str | None = None
    update_action: str | None = None
    match_type: str | None = None
    expression: str | None = None


_PUBLIC_ONBOARDING_TOKEN_REQUIRED_CONSTRAINTS = {
    "pk_eom_public_onboarding_tokens": _PublicOnboardingTokenConstraint(
        "p", ("id",)
    ),
    "uq_eom_public_onboarding_tokens_draft": _PublicOnboardingTokenConstraint(
        "u", ("draft_id",)
    ),
    "uq_eom_public_onboarding_tokens_approval": _PublicOnboardingTokenConstraint(
        "u", ("approval_key",)
    ),
    "uq_eom_public_onboarding_tokens_handoff": _PublicOnboardingTokenConstraint(
        "u", ("handoff_id",)
    ),
    "eom_public_onboarding_tokens_draft_id_fkey": _PublicOnboardingTokenConstraint(
        "f",
        ("draft_id",),
        referenced_table="eom_onboarding_email_drafts",
        referenced_columns=("id",),
        delete_action="r",
        update_action="a",
        match_type="s",
    ),
    "eom_public_onboarding_tokens_contact_id_fkey": _PublicOnboardingTokenConstraint(
        "f",
        ("contact_id",),
        referenced_table="contacts",
        referenced_columns=("id",),
        delete_action="r",
        update_action="a",
        match_type="s",
    ),
} | {
    name: _PublicOnboardingTokenConstraint(
        "c",
        {
            "ck_eom_public_onboarding_tokens_status": ("status",),
            "eom_public_onboarding_tokens_approved_by_employee_id_check": (
                "approved_by_employee_id",
            ),
            "eom_public_onboarding_tokens_signing_key_fingerprint_check": (
                "signing_key_fingerprint",
            ),
            "ck_eom_public_onboarding_tokens_terminal_state": (
                "status",
                "redeemed_at",
                "revoked_at",
                "handoff_id",
            ),
        }[name],
        expression=expression,
    )
    for name, expression in _PUBLIC_ONBOARDING_TOKEN_CONSTRAINT_EXPRESSIONS.items()
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


def _catalog_char(value: object) -> str:
    """Normalize PostgreSQL's single-character catalog values for comparison."""
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value or "")


def _catalog_column_names(value: object) -> tuple[str, ...]:
    """Return catalog array values as an immutable, order-preserving tuple."""
    if value is None:
        return ()
    return tuple(str(name) for name in value)


def _public_onboarding_token_constraint_ready(
    row: Any,
    expected: _PublicOnboardingTokenConstraint,
) -> bool:
    """Check one named constraint's full writer-facing structural contract."""
    referenced_table_ready = (
        expected.referenced_table is None
        or (
            row["referenced_table"] == expected.referenced_table
            and bool(row["references_current_schema"])
        )
    )
    delete_action_ready = (
        expected.delete_action is None
        or _catalog_char(row["delete_action"]) == expected.delete_action
    )
    update_action_ready = (
        expected.update_action is None
        or _catalog_char(row["update_action"]) == expected.update_action
    )
    match_type_ready = (
        expected.match_type is None
        or _catalog_char(row["match_type"]) == expected.match_type
    )
    expression_ready = (
        expected.expression is None
        or _canonicalize_catalog_constraint_expression(row["expression"])
        == expected.expression
    )
    return all((
        _catalog_char(row["constraint_type"]) == expected.constraint_type,
        _catalog_column_names(row["key_columns"]) == expected.key_columns,
        referenced_table_ready,
        _catalog_column_names(row["referenced_columns"])
        == expected.referenced_columns,
        delete_action_ready,
        update_action_ready,
        match_type_ready,
        not bool(row["is_deferrable"]),
        not bool(row["is_initially_deferred"]),
        bool(row["is_validated"]),
        expression_ready,
    ))


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
) -> tuple[bool, bool, bool, bool, bool, bool, bool]:
    """Read only the full final token contract, never token values."""
    column_rows = await executor.fetch(
        """
        SELECT
            actual.column_name,
            actual.data_type,
            actual.character_maximum_length,
            actual.is_nullable,
            actual.column_default
        FROM information_schema.columns AS actual
        WHERE actual.table_schema = current_schema()
          AND actual.table_name = 'eom_public_onboarding_tokens'
          AND actual.column_name = ANY($1::text[])
        """,
        list(_PUBLIC_ONBOARDING_TOKEN_COLUMNS),
    )
    observed_columns = {
        row["column_name"]: (
            row["data_type"],
            row["character_maximum_length"],
            row["is_nullable"],
            _canonicalize_catalog_constraint_expression(row["column_default"])
            if row["column_default"] is not None
            else None,
        )
        for row in column_rows
    }
    immutable_projection_ready = all(
        observed_columns.get(name) == (*signature, None)
        for name, signature in _PUBLIC_ONBOARDING_TOKEN_IMMUTABLE_COLUMNS.items()
    )
    base_token_contract_ready = all(
        observed_columns.get(name) == signature
        for name, signature in _PUBLIC_ONBOARDING_TOKEN_BASE_COLUMNS.items()
    )

    constraint_rows = await executor.fetch(
        """
        SELECT
            actual.conname,
            actual.contype AS constraint_type,
            ARRAY(
                SELECT attribute_state.attname
                FROM unnest(actual.conkey)
                     WITH ORDINALITY AS key_state(attnum, ordinality)
                JOIN pg_attribute AS attribute_state
                  ON attribute_state.attrelid = actual.conrelid
                 AND attribute_state.attnum = key_state.attnum
                ORDER BY key_state.ordinality
            ) AS key_columns,
            referenced_table.relname AS referenced_table,
            (referenced_namespace.nspname = current_schema())
                AS references_current_schema,
            ARRAY(
                SELECT attribute_state.attname
                FROM unnest(actual.confkey)
                     WITH ORDINALITY AS key_state(attnum, ordinality)
                JOIN pg_attribute AS attribute_state
                  ON attribute_state.attrelid = actual.confrelid
                 AND attribute_state.attnum = key_state.attnum
                ORDER BY key_state.ordinality
            ) AS referenced_columns,
            actual.confdeltype AS delete_action,
            actual.confupdtype AS update_action,
            actual.confmatchtype AS match_type,
            actual.condeferrable AS is_deferrable,
            actual.condeferred AS is_initially_deferred,
            actual.convalidated AS is_validated,
            pg_get_expr(actual.conbin, actual.conrelid) AS expression
        FROM pg_constraint AS actual
        JOIN pg_class AS table_class
          ON table_class.oid = actual.conrelid
        JOIN pg_namespace AS table_namespace
          ON table_namespace.oid = table_class.relnamespace
        LEFT JOIN pg_class AS referenced_table
          ON referenced_table.oid = actual.confrelid
        LEFT JOIN pg_namespace AS referenced_namespace
          ON referenced_namespace.oid = referenced_table.relnamespace
        WHERE table_namespace.nspname = current_schema()
          AND table_class.relname = 'eom_public_onboarding_tokens'
          AND actual.conname = ANY($1::text[])
        """,
        list(_PUBLIC_ONBOARDING_TOKEN_REQUIRED_CONSTRAINTS),
    )
    observed_constraints = {
        row["conname"]: row
        for row in constraint_rows
    }
    required_constraints_ready = (
        set(observed_constraints) == set(_PUBLIC_ONBOARDING_TOKEN_REQUIRED_CONSTRAINTS)
        and all(
            _public_onboarding_token_constraint_ready(
                observed_constraints[name], expected
            )
            for name, expected in _PUBLIC_ONBOARDING_TOKEN_REQUIRED_CONSTRAINTS.items()
        )
    )
    fingerprint_constraint = observed_constraints.get(
        "eom_public_onboarding_tokens_signing_key_fingerprint_check"
    )
    terminal_state_constraint = observed_constraints.get(
        "ck_eom_public_onboarding_tokens_terminal_state"
    )
    fingerprint_check_ready = (
        _canonicalize_catalog_constraint_expression(
            fingerprint_constraint["expression"]
            if fingerprint_constraint is not None
            else None
        )
        == _PUBLIC_ONBOARDING_TOKEN_CONSTRAINT_EXPRESSIONS[
            "eom_public_onboarding_tokens_signing_key_fingerprint_check"
        ]
    )
    terminal_state_check_ready = (
        _canonicalize_catalog_constraint_expression(
            terminal_state_constraint["expression"]
            if terminal_state_constraint is not None
            else None
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
        base_token_contract_ready,
        required_constraints_ready,
        fingerprint_check_ready,
        terminal_state_check_ready,
        issued_contact_index_ready,
        status_index_ready,
    )


async def _migration_022b_catalog_evidence(
    executor: Any,
) -> tuple[bool, bool, bool]:
    """Read only the exact ordinary-table and column contract, never rows."""
    presence_events_is_ordinary_table = bool(
        await executor.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_class AS relation_state
                JOIN pg_namespace AS schema_state
                  ON schema_state.oid = relation_state.relnamespace
                WHERE schema_state.nspname = current_schema()
                  AND relation_state.relname = 'presence_events'
                  AND relation_state.relkind = 'r'
            )
            """
        )
    )
    column_rows = await executor.fetch(
        """
        SELECT
            actual.column_name,
            actual.data_type,
            actual.is_nullable,
            actual.column_default
        FROM information_schema.columns AS actual
        WHERE actual.table_schema = current_schema()
          AND actual.table_name = 'presence_events'
          AND actual.column_name = 'unknown_count'
        """
    )
    observed_columns = {
        row["column_name"]: (
            row["data_type"],
            row["is_nullable"],
            _canonicalize_catalog_constraint_expression(row["column_default"])
            if row["column_default"] is not None
            else None,
        )
        for row in column_rows
    }
    unknown_count_column_ready = observed_columns == {
        "unknown_count": _PRESENCE_UNKNOWN_COUNT_COLUMN
    }
    unknown_count_has_no_constraints = not bool(
        await executor.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_constraint AS actual
                JOIN pg_class AS table_class
                  ON table_class.oid = actual.conrelid
                JOIN pg_namespace AS table_namespace
                  ON table_namespace.oid = table_class.relnamespace
                WHERE table_namespace.nspname = current_schema()
                  AND table_class.relname = 'presence_events'
                  AND EXISTS (
                      SELECT 1
                      FROM unnest(
                          COALESCE(actual.conkey, ARRAY[]::smallint[])
                      ) AS key_state(attnum)
                      JOIN pg_attribute AS attribute_state
                        ON attribute_state.attrelid = actual.conrelid
                       AND attribute_state.attnum = key_state.attnum
                      WHERE attribute_state.attname = 'unknown_count'
                  )
            )
            """
        )
    )
    return (
        presence_events_is_ordinary_table,
        unknown_count_column_ready,
        unknown_count_has_no_constraints,
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
        base_token_contract_ready,
        required_constraints_ready,
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
        base_token_contract_ready=base_token_contract_ready,
        required_constraints_ready=required_constraints_ready,
        fingerprint_check_ready=fingerprint_check_ready,
        terminal_state_check_ready=terminal_state_check_ready,
        issued_contact_index_ready=issued_contact_index_ready,
        status_index_ready=status_index_ready,
    )


async def _attest_migration_022b(
    executor: Any,
    migration_files: Collection[Path],
) -> RenamedMissingSourceMigrationReconciliationAttestation:
    """Attest the named 022b receipt without backfilling its NULL digest."""
    record = MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION
    ledger_rows = await executor.fetch(
        "SELECT content_sha256, applied_at FROM schema_migrations WHERE name = $1 LIMIT 2",
        record.migration_name,
    )
    exactly_one_ledger_row = len(ledger_rows) == 1
    ledger_row = ledger_rows[0] if exactly_one_ledger_row else None
    recorded_digest = ledger_row["content_sha256"] if ledger_row is not None else None
    applied_at = _normalize_utc(
        ledger_row["applied_at"] if ledger_row is not None else None
    )
    retained_packaged_digest = _packaged_migration_digest(
        migration_files,
        record.current_packaged_migration_name,
    )
    (
        presence_events_is_ordinary_table,
        unknown_count_column_ready,
        unknown_count_has_no_constraints,
    ) = await _migration_022b_catalog_evidence(executor)

    return RenamedMissingSourceMigrationReconciliationAttestation(
        reconciliation_id=record.reconciliation_id,
        migration_name=record.migration_name,
        exactly_one_ledger_row=exactly_one_ledger_row,
        ledger_digest_is_null=(
            exactly_one_ledger_row
            and recorded_digest == record.historical_ledger_sha256
        ),
        applied_at_matches_record=applied_at == record.observed_applied_at,
        retained_packaged_digest_matches_record=(
            retained_packaged_digest == record.retained_source_sha256
        ),
        presence_events_is_ordinary_table=presence_events_is_ordinary_table,
        unknown_count_column_ready=unknown_count_column_ready,
        unknown_count_has_no_constraints=unknown_count_has_no_constraints,
    )


async def attest_known_historical_migration_reconciliations(
    executor: Any,
    migration_files: Collection[Path],
    *,
    candidate_names: Collection[str] | None = None,
) -> tuple[
    MigrationReconciliationAttestation
    | MissingSourceMigrationReconciliationAttestation
    | RenamedMissingSourceMigrationReconciliationAttestation,
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
        | RenamedMissingSourceMigrationReconciliationAttestation
    ] = []
    if MIGRATION_387_RECONCILIATION.migration_name in requested_names:
        attestations.append(await _attest_migration_387(executor, migration_files))
    if (
        MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION.migration_name
        in requested_names
    ):
        attestations.append(await _attest_migration_382(executor))
    if (
        MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION.migration_name
        in requested_names
    ):
        attestations.append(await _attest_migration_022b(executor, migration_files))
    return tuple(attestations)
