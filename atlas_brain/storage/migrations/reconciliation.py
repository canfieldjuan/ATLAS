"""Immutable evidence for named historical migration-source gaps.

This module records what Atlas can prove about a known historical mismatch
without rewriting the migration ledger or treating unavailable source bytes as
verified. It is intentionally narrow: every additional historical mismatch
requires its own reviewed evidence record and catalog predicate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Mapping
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
class HistoricalMigrationForwardRecoveryReconciliation:
    """Immutable evidence for one target that needs a named forward recovery.

    Unlike an ordinary historical reconciliation, the exact old catalog state
    is unsafe to admit. It is a narrow precondition for one additive migration,
    and becomes attested only after that migration's independent ledger receipt
    and stronger catalog state both exist.
    """

    reconciliation_id: str
    migration_name: str
    historical_migration_version: int
    historical_ledger_sha256: str
    final_packaged_sha256: str
    observed_applied_at: datetime
    legacy_function_body_sha256: str
    recovered_function_body_template_sha256: str
    recovery_migration_name: str
    recovery_migration_version: int
    recovery_packaged_sha256: str

    @property
    def source_verification(self) -> str:
        """Catalog recovery never changes the historical source-evidence limit."""
        return HISTORICAL_SOURCE_UNAVAILABLE


@dataclass(frozen=True)
class HistoricalNullDigestMigrationReceipt:
    """One exact NULL-digest receipt required by a forward recovery."""

    migration_name: str
    migration_version: int
    observed_applied_at: datetime


@dataclass(frozen=True)
class HistoricalMissingSourceForwardRecoveryReconciliation:
    """Immutable evidence for a missing source whose catalog needs recovery.

    This is deliberately distinct from a historical digest mismatch. The
    missing source cannot be reconstructed, so Atlas admits no later SQL until
    an independently-recorded, exact catalog recovery attests the known target.
    """

    reconciliation_id: str
    migration_name: str
    historical_migration_version: int
    historical_ledger_sha256: None
    observed_applied_at: datetime
    successor_receipts: tuple[HistoricalNullDigestMigrationReceipt, ...]
    legacy_function_body_sha256: str
    recovered_function_body_template_sha256: str
    review_decision_default_function_body_sha256: str
    review_decision_history_guard_function_body_sha256: str
    override_history_guard_function_body_sha256: str
    recovery_migration_name: str
    recovery_migration_version: int
    recovery_packaged_sha256: str
    schema_binding_migration_name: str
    schema_binding_migration_version: int
    schema_binding_packaged_sha256: str

    @property
    def source_verification(self) -> str:
        """The unavailable historical bytes remain unavailable after recovery."""
        return HISTORICAL_SOURCE_UNAVAILABLE


class HistoricalForwardRecoveryAtomicPreflightError(RuntimeError):
    """Raised when a selected recovery changes before its receipt can commit."""


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
class HistoricalVersionedMissingSourceReconciliation:
    """Reviewed receipt for a missing source with immutable version evidence."""

    reconciliation_id: str
    migration_name: str
    migration_version: int
    historical_ledger_sha256: None
    observed_applied_at: datetime

    @property
    def source_verification(self) -> str:
        """A ledger row and catalog cannot reconstruct unavailable source bytes."""
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
class ForwardRecoveryMigrationReconciliationAttestation:
    """Read-only state for the one 386 forward-only recovery route."""

    reconciliation_id: str
    migration_name: str
    historical_receipt_ready: bool
    recovery_source_ready: bool
    no_recovery_ledger_row: bool
    recovery_receipt_ready: bool
    trusted_guard_role_ready: bool
    recovered_function_guard_owner_ready: bool
    recovered_function_guard_lifecycle_read_ready: bool
    legacy_catalog_ready: bool
    recovered_catalog_ready: bool

    @property
    def source_verification(self) -> str:
        """Recovery evidence still cannot recover the original source bytes."""
        return HISTORICAL_SOURCE_UNAVAILABLE

    @property
    def status(self) -> str:
        if all((
            self.historical_receipt_ready,
            self.recovery_source_ready,
            self.legacy_catalog_ready,
            self.no_recovery_ledger_row,
        )):
            return "recovery_required"
        if all((
            self.historical_receipt_ready,
            self.recovery_source_ready,
            self.recovered_catalog_ready,
            self.recovery_receipt_ready,
        )):
            return "attested"
        return "not_attested"

    def as_payload(self) -> dict[str, object]:
        """Expose catalog facts and state without contact or event rows."""
        return {
            "reconciliation_id": self.reconciliation_id,
            "migration_name": self.migration_name,
            "source_verification": self.source_verification,
            "historical_receipt_ready": self.historical_receipt_ready,
            "recovery_source_ready": self.recovery_source_ready,
            "no_recovery_ledger_row": self.no_recovery_ledger_row,
            "legacy_catalog_ready": self.legacy_catalog_ready,
            "recovered_catalog_ready": self.recovered_catalog_ready,
            "recovery_receipt_ready": self.recovery_receipt_ready,
            "trusted_guard_role_ready": self.trusted_guard_role_ready,
            "recovered_function_guard_owner_ready": (
                self.recovered_function_guard_owner_ready
            ),
            "recovered_function_guard_lifecycle_read_ready": (
                self.recovered_function_guard_lifecycle_read_ready
            ),
            "status": self.status,
        }


@dataclass(frozen=True)
class MissingSourceForwardRecoveryMigrationReconciliationAttestation:
    """Read-only state for the one run-scoped commercial billing recovery."""

    reconciliation_id: str
    migration_name: str
    historical_receipt_ready: bool
    successor_receipts_ready: bool
    recovery_source_ready: bool
    no_recovery_ledger_row: bool
    recovery_receipt_ready: bool
    schema_binding_source_ready: bool
    no_schema_binding_ledger_row: bool
    schema_binding_receipt_ready: bool
    reviewed_billing_catalog_ready: bool
    required_billing_columns_ready: bool
    no_unreviewed_billing_columns: bool
    no_unreviewed_billing_read_interceptors: bool
    no_unreviewed_billing_write_interceptors: bool
    review_decision_default_trigger_ready: bool
    review_decision_default_function_body_ready: bool
    history_guard_function_bodies_ready: bool
    required_billing_constraints_ready: bool
    foreign_key_enforcement_ready: bool
    no_unreviewed_billing_constraints: bool
    required_billing_indexes_ready: bool
    no_unreviewed_billing_indexes: bool
    invoice_fence_trigger_ready: bool
    no_unreviewed_invoice_insert_interceptors: bool
    no_unreviewed_invoice_rewrite_interceptors: bool
    trigger_function_execution_metadata_ready: bool
    invoice_fence_function_schema_binding_ready: bool
    legacy_function_body_matches: bool
    recovered_function_body_matches: bool

    @property
    def source_verification(self) -> str:
        """The recovery's receipt never claims the original source was found."""
        return HISTORICAL_SOURCE_UNAVAILABLE

    @property
    def legacy_catalog_ready(self) -> bool:
        """Return whether the exact unsafe catalog can receive only 391."""
        return all((
            self.reviewed_billing_catalog_ready,
            self.invoice_fence_trigger_ready,
            self.legacy_function_body_matches,
        ))

    @property
    def recovered_catalog_ready(self) -> bool:
        """Return whether the current run-scoped fence is now visible."""
        return all((
            self.reviewed_billing_catalog_ready,
            self.invoice_fence_trigger_ready,
            self.recovered_function_body_matches,
            self.invoice_fence_function_schema_binding_ready,
        ))

    @property
    def schema_binding_required(self) -> bool:
        """Return whether exact recovered 391 state needs only its schema pin."""
        return all((
            self.historical_receipt_ready,
            self.successor_receipts_ready,
            self.recovery_source_ready,
            self.schema_binding_source_ready,
            self.recovery_receipt_ready,
            self.no_schema_binding_ledger_row,
            self.reviewed_billing_catalog_ready,
            self.invoice_fence_trigger_ready,
            self.recovered_function_body_matches,
        ))

    @property
    def status(self) -> str:
        if all((
            self.historical_receipt_ready,
            self.successor_receipts_ready,
            self.recovery_source_ready,
            self.schema_binding_source_ready,
            self.no_recovery_ledger_row,
            self.no_schema_binding_ledger_row,
            self.legacy_catalog_ready,
        )):
            return "recovery_required"
        if self.schema_binding_required:
            return "schema_binding_required"
        if all((
            self.historical_receipt_ready,
            self.successor_receipts_ready,
            self.recovery_source_ready,
            self.schema_binding_source_ready,
            self.recovery_receipt_ready,
            self.schema_binding_receipt_ready,
            self.recovered_catalog_ready,
        )):
            return "attested"
        return "not_attested"

    def as_payload(self) -> dict[str, object]:
        """Expose only state evidence, never billing rows or customer data."""
        return {
            "reconciliation_id": self.reconciliation_id,
            "migration_name": self.migration_name,
            "source_verification": self.source_verification,
            "historical_receipt_ready": self.historical_receipt_ready,
            "successor_receipts_ready": self.successor_receipts_ready,
            "recovery_source_ready": self.recovery_source_ready,
            "no_recovery_ledger_row": self.no_recovery_ledger_row,
            "recovery_receipt_ready": self.recovery_receipt_ready,
            "schema_binding_source_ready": self.schema_binding_source_ready,
            "no_schema_binding_ledger_row": self.no_schema_binding_ledger_row,
            "schema_binding_receipt_ready": self.schema_binding_receipt_ready,
            "reviewed_billing_catalog_ready": self.reviewed_billing_catalog_ready,
            "required_billing_columns_ready": self.required_billing_columns_ready,
            "no_unreviewed_billing_columns": self.no_unreviewed_billing_columns,
            "no_unreviewed_billing_read_interceptors": (
                self.no_unreviewed_billing_read_interceptors
            ),
            "no_unreviewed_billing_write_interceptors": (
                self.no_unreviewed_billing_write_interceptors
            ),
            "review_decision_default_trigger_ready": (
                self.review_decision_default_trigger_ready
            ),
            "review_decision_default_function_body_ready": (
                self.review_decision_default_function_body_ready
            ),
            "history_guard_function_bodies_ready": (
                self.history_guard_function_bodies_ready
            ),
            "required_billing_constraints_ready": (
                self.required_billing_constraints_ready
            ),
            "foreign_key_enforcement_ready": self.foreign_key_enforcement_ready,
            "no_unreviewed_billing_constraints": (
                self.no_unreviewed_billing_constraints
            ),
            "required_billing_indexes_ready": self.required_billing_indexes_ready,
            "no_unreviewed_billing_indexes": self.no_unreviewed_billing_indexes,
            "invoice_fence_trigger_ready": self.invoice_fence_trigger_ready,
            "no_unreviewed_invoice_insert_interceptors": (
                self.no_unreviewed_invoice_insert_interceptors
            ),
            "no_unreviewed_invoice_rewrite_interceptors": (
                self.no_unreviewed_invoice_rewrite_interceptors
            ),
            "trigger_function_execution_metadata_ready": (
                self.trigger_function_execution_metadata_ready
            ),
            "invoice_fence_function_schema_binding_ready": (
                self.invoice_fence_function_schema_binding_ready
            ),
            "legacy_function_body_matches": self.legacy_function_body_matches,
            "recovered_function_body_matches": (
                self.recovered_function_body_matches
            ),
            "legacy_catalog_ready": self.legacy_catalog_ready,
            "recovered_catalog_ready": self.recovered_catalog_ready,
            "schema_binding_required": self.schema_binding_required,
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


@dataclass(frozen=True)
class B2BCampaignPartnerMissingSourceMigrationReconciliationAttestation:
    """Read-only target proof for one legacy B2B campaign-partner receipt."""

    reconciliation_id: str
    migration_name: str
    exactly_one_ledger_row: bool
    ledger_version_matches_record: bool
    ledger_digest_is_null: bool
    applied_at_matches_record: bool
    b2b_campaigns_is_ordinary_table: bool
    partner_id_column_ready: bool
    partner_foreign_key_ready: bool
    partner_partial_index_ready: bool

    @property
    def source_verification(self) -> str:
        """Catalog evidence must not pretend to verify source bytes."""
        return HISTORICAL_SOURCE_UNAVAILABLE

    @property
    def status(self) -> str:
        if all((
            self.exactly_one_ledger_row,
            self.ledger_version_matches_record,
            self.ledger_digest_is_null,
            self.applied_at_matches_record,
            self.b2b_campaigns_is_ordinary_table,
            self.partner_id_column_ready,
            self.partner_foreign_key_ready,
            self.partner_partial_index_ready,
        )):
            return "attested"
        return "not_attested"

    def as_payload(self) -> dict[str, object]:
        """Expose only target metadata and booleans, never campaign rows."""
        return {
            "reconciliation_id": self.reconciliation_id,
            "migration_name": self.migration_name,
            "source_verification": self.source_verification,
            "exactly_one_ledger_row": self.exactly_one_ledger_row,
            "ledger_version_matches_record": self.ledger_version_matches_record,
            "ledger_digest_is_null": self.ledger_digest_is_null,
            "applied_at_matches_record": self.applied_at_matches_record,
            "b2b_campaigns_is_ordinary_table": (
                self.b2b_campaigns_is_ordinary_table
            ),
            "partner_id_column_ready": self.partner_id_column_ready,
            "partner_foreign_key_ready": self.partner_foreign_key_ready,
            "partner_partial_index_ready": self.partner_partial_index_ready,
            "status": self.status,
        }


@dataclass(frozen=True)
class B2BCompanySignalPromotionMissingSourceMigrationReconciliationAttestation:
    """Read-only target proof for the legacy company-signal 297 receipt."""

    reconciliation_id: str
    migration_name: str
    exactly_one_ledger_row: bool
    ledger_version_matches_record: bool
    ledger_digest_is_null: bool
    applied_at_matches_record: bool
    b2b_company_signals_is_ordinary_table: bool
    canonical_promotion_type_column_ready: bool
    canonical_promotion_type_has_no_constraints: bool
    canonical_promotion_type_partial_index_ready: bool

    @property
    def source_verification(self) -> str:
        """Catalog evidence must not pretend to verify source bytes."""
        return HISTORICAL_SOURCE_UNAVAILABLE

    @property
    def status(self) -> str:
        if all((
            self.exactly_one_ledger_row,
            self.ledger_version_matches_record,
            self.ledger_digest_is_null,
            self.applied_at_matches_record,
            self.b2b_company_signals_is_ordinary_table,
            self.canonical_promotion_type_column_ready,
            self.canonical_promotion_type_has_no_constraints,
            self.canonical_promotion_type_partial_index_ready,
        )):
            return "attested"
        return "not_attested"

    def as_payload(self) -> dict[str, object]:
        """Expose structural evidence without reading company-signal rows."""
        return {
            "reconciliation_id": self.reconciliation_id,
            "migration_name": self.migration_name,
            "source_verification": self.source_verification,
            "exactly_one_ledger_row": self.exactly_one_ledger_row,
            "ledger_version_matches_record": self.ledger_version_matches_record,
            "ledger_digest_is_null": self.ledger_digest_is_null,
            "applied_at_matches_record": self.applied_at_matches_record,
            "b2b_company_signals_is_ordinary_table": (
                self.b2b_company_signals_is_ordinary_table
            ),
            "canonical_promotion_type_column_ready": (
                self.canonical_promotion_type_column_ready
            ),
            "canonical_promotion_type_has_no_constraints": (
                self.canonical_promotion_type_has_no_constraints
            ),
            "canonical_promotion_type_partial_index_ready": (
                self.canonical_promotion_type_partial_index_ready
            ),
            "status": self.status,
        }


@dataclass(frozen=True)
class B2BWatchlistAlertEventsMissingSourceMigrationReconciliationAttestation:
    """Read-only target proof for the legacy synthetic-version alert receipt."""

    reconciliation_id: str
    migration_name: str
    exactly_one_ledger_row: bool
    ledger_version_matches_record: bool
    ledger_digest_is_null: bool
    applied_at_matches_record: bool
    watchlist_alert_events_is_ordinary_table: bool
    watchlist_alert_events_has_permanent_storage: bool
    base_alert_event_columns_ready: bool
    known_later_alert_event_columns_ready: bool
    no_unlisted_alert_event_columns: bool
    required_alert_event_constraints_ready: bool
    no_unlisted_alert_event_constraints: bool
    required_alert_event_indexes_ready: bool
    no_unlisted_alert_event_indexes: bool
    no_unreviewed_alert_event_write_interceptors: bool

    @property
    def source_verification(self) -> str:
        """Catalog evidence must not pretend to verify missing source bytes."""
        return HISTORICAL_SOURCE_UNAVAILABLE

    @property
    def status(self) -> str:
        if all((
            self.exactly_one_ledger_row,
            self.ledger_version_matches_record,
            self.ledger_digest_is_null,
            self.applied_at_matches_record,
            self.watchlist_alert_events_is_ordinary_table,
            self.watchlist_alert_events_has_permanent_storage,
            self.base_alert_event_columns_ready,
            self.known_later_alert_event_columns_ready,
            self.no_unlisted_alert_event_columns,
            self.required_alert_event_constraints_ready,
            self.no_unlisted_alert_event_constraints,
            self.required_alert_event_indexes_ready,
            self.no_unlisted_alert_event_indexes,
            self.no_unreviewed_alert_event_write_interceptors,
        )):
            return "attested"
        return "not_attested"

    def as_payload(self) -> dict[str, object]:
        """Expose metadata booleans without reading tenant alert-event rows."""
        return {
            "reconciliation_id": self.reconciliation_id,
            "migration_name": self.migration_name,
            "source_verification": self.source_verification,
            "exactly_one_ledger_row": self.exactly_one_ledger_row,
            "ledger_version_matches_record": self.ledger_version_matches_record,
            "ledger_digest_is_null": self.ledger_digest_is_null,
            "applied_at_matches_record": self.applied_at_matches_record,
            "watchlist_alert_events_is_ordinary_table": (
                self.watchlist_alert_events_is_ordinary_table
            ),
            "watchlist_alert_events_has_permanent_storage": (
                self.watchlist_alert_events_has_permanent_storage
            ),
            "base_alert_event_columns_ready": self.base_alert_event_columns_ready,
            "known_later_alert_event_columns_ready": (
                self.known_later_alert_event_columns_ready
            ),
            "no_unlisted_alert_event_columns": (
                self.no_unlisted_alert_event_columns
            ),
            "required_alert_event_constraints_ready": (
                self.required_alert_event_constraints_ready
            ),
            "no_unlisted_alert_event_constraints": (
                self.no_unlisted_alert_event_constraints
            ),
            "required_alert_event_indexes_ready": (
                self.required_alert_event_indexes_ready
            ),
            "no_unlisted_alert_event_indexes": self.no_unlisted_alert_event_indexes,
            "no_unreviewed_alert_event_write_interceptors": (
                self.no_unreviewed_alert_event_write_interceptors
            ),
            "status": self.status,
        }


MIGRATION_386_WON_LOSS_FENCE_FORWARD_RECOVERY = (
    HistoricalMigrationForwardRecoveryReconciliation(
        reconciliation_id="eom-migration-386-won-loss-direct-sql-fence-forward-recovery",
        migration_name="386_eom_won_loss_nocodb_fence",
        # The canonical target recorded this original NocoDB-only source before
        # the packaged migration was strengthened in place. Retain its exact
        # ledger receipt and timestamp; do not rewrite either historical fact.
        historical_migration_version=386,
        historical_ledger_sha256=(
            "2055264e1a819b968935bc901aee0175d99ed1ec15465a1203a58a2cf7aa40ea"
        ),
        final_packaged_sha256=(
            "3bcef3e3a6b5564bd0f1f0c38dfbe38e4f65a9c03d27afd242b317902042c982"
        ),
        observed_applied_at=datetime(
            2026,
            8,
            20,
            19,
            5,
            51,
            230_103,
            tzinfo=timezone.utc,
        ),
        legacy_function_body_sha256=(
            "40ec9678638905a797cc62cb58aa1c43c354d10c3de14cbda543b0dd6d8258c4"
        ),
        recovered_function_body_template_sha256=(
            "f482b2af7e028c058f83dc1288d6c05b04dcaf377dc507392d4a8ca04c58cf1a"
        ),
        recovery_migration_name="390_eom_won_loss_direct_sql_fence_recovery",
        recovery_migration_version=390,
        recovery_packaged_sha256=(
            "e65d61e16cfea9974df6b765522459d7ddd7e50332346914a9d0a58ca6e8f6d0"
        ),
    )
)


MIGRATION_379_COMMERCIAL_BILLING_RUN_FENCE_FORWARD_RECOVERY = (
    HistoricalMissingSourceForwardRecoveryReconciliation(
        reconciliation_id=(
            "eom-migration-379-commercial-billing-run-fence-forward-recovery"
        ),
        migration_name="379_commercial_billing_candidate_review_decisions",
        # The target records this unavailable source under a synthetic version.
        # Preserve that historical fact; 391 is a separate forward-only receipt.
        historical_migration_version=-10,
        historical_ledger_sha256=None,
        observed_applied_at=datetime(
            2026,
            8,
            16,
            18,
            4,
            47,
            984_357,
            tzinfo=timezone.utc,
        ),
        successor_receipts=(
            HistoricalNullDigestMigrationReceipt(
                migration_name="380_commercial_billing_candidate_review_decisions",
                migration_version=380,
                observed_applied_at=datetime(
                    2026,
                    8,
                    16,
                    18,
                    22,
                    56,
                    919_633,
                    tzinfo=timezone.utc,
                ),
            ),
            HistoricalNullDigestMigrationReceipt(
                migration_name=(
                    "381_commercial_billing_candidate_review_decisions_recovery"
                ),
                migration_version=381,
                observed_applied_at=datetime(
                    2026,
                    8,
                    16,
                    23,
                    18,
                    24,
                    384_279,
                    tzinfo=timezone.utc,
                ),
            ),
            HistoricalNullDigestMigrationReceipt(
                migration_name="382_commercial_billing_candidate_overrides",
                migration_version=382,
                observed_applied_at=datetime(
                    2026,
                    8,
                    17,
                    19,
                    9,
                    25,
                    208_581,
                    tzinfo=timezone.utc,
                ),
            ),
        ),
        legacy_function_body_sha256=(
            "b71db37ee1906ca26788be21deb716092052fc3197d4b72762d57892fbc77851"
        ),
        recovered_function_body_template_sha256=(
            "04b99e4a3ff2b18f2d58d3e1e610a4b2079fcbbd0d5ce51d97c212daaefd0477"
        ),
        review_decision_default_function_body_sha256=(
            "a07d01aa1ea28b8817d6e0f8d26195f653cfb39328aea0cbb54db2a44b7b4d54"
        ),
        review_decision_history_guard_function_body_sha256=(
            "a417f49d8bd7c62ee4dbc80348014fb2d251809c79c4a190f2b17c34182c896c"
        ),
        override_history_guard_function_body_sha256=(
            "be37fd47a94998ebe16fb8e08fc25542330af76780f2d830a10b779891058002"
        ),
        recovery_migration_name="391_eom_commercial_billing_run_fence_recovery",
        recovery_migration_version=391,
        recovery_packaged_sha256=(
            "117cdd2c509cd89ffaae2efbc4732caf9aea7e155114910a5e5bbe1b5f7d66b7"
        ),
        schema_binding_migration_name=(
            "392_eom_commercial_billing_run_fence_schema_binding"
        ),
        schema_binding_migration_version=392,
        schema_binding_packaged_sha256=(
            "737a5ef0a8c035821c108f245bfe048717d4f064daf6da8732f2756da5d67c58"
        ),
    )
)


_RECOVERED_386_TRIGGER_UPDATE_COLUMNS = frozenset(
    {"business_context_id", "contact_type", "lead_stage", "status"}
)


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


MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION = (
    HistoricalVersionedMissingSourceReconciliation(
        reconciliation_id="b2b-migration-067-campaign-partner-source-absence",
        migration_name="067_b2b_campaign_partner",
        migration_version=67,
        historical_ledger_sha256=None,
        observed_applied_at=datetime(
            2026,
            3,
            1,
            4,
            58,
            0,
            789_236,
            tzinfo=timezone.utc,
        ),
    )
)


MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION = (
    HistoricalVersionedMissingSourceReconciliation(
        reconciliation_id="b2b-migration-297-company-signal-promotion-source-absence",
        migration_name="297_b2b_company_signal_canonical_promotion_type",
        migration_version=297,
        historical_ledger_sha256=None,
        observed_applied_at=datetime(
            2026,
            4,
            12,
            19,
            28,
            13,
            742_305,
            tzinfo=timezone.utc,
        ),
    )
)


MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION = (
    HistoricalVersionedMissingSourceReconciliation(
        reconciliation_id="b2b-migration-272-watchlist-alert-events-source-absence",
        migration_name="272_b2b_watchlist_alert_events",
        # The historical runner assigned a synthetic negative version because
        # its numeric prefix collided. The target also retains a separate
        # later 273 receipt, so this is not a source rename.
        migration_version=-3,
        historical_ledger_sha256=None,
        observed_applied_at=datetime(
            2026,
            4,
            8,
            3,
            34,
            24,
            452_014,
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


_HISTORICAL_MISSING_SOURCE_FORWARD_RECOVERY_RECONCILIATIONS = (
    MIGRATION_379_COMMERCIAL_BILLING_RUN_FENCE_FORWARD_RECOVERY,
)
_HISTORICAL_MISMATCH_FORWARD_RECOVERY_RECONCILIATIONS = (
    MIGRATION_386_WON_LOSS_FENCE_FORWARD_RECOVERY,
)
_HISTORICAL_FORWARD_RECOVERY_RECONCILIATIONS = (
    *_HISTORICAL_MISSING_SOURCE_FORWARD_RECOVERY_RECONCILIATIONS,
    *_HISTORICAL_MISMATCH_FORWARD_RECOVERY_RECONCILIATIONS,
)
_HISTORICAL_MISMATCH_RECONCILIATIONS = (
    *_HISTORICAL_MISMATCH_FORWARD_RECOVERY_RECONCILIATIONS,
    MIGRATION_387_RECONCILIATION,
)
_HISTORICAL_MISSING_SOURCE_RECONCILIATIONS = (
    *_HISTORICAL_MISSING_SOURCE_FORWARD_RECOVERY_RECONCILIATIONS,
    MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION,
    MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION,
    MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION,
    MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION,
    MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION,
)


def historical_forward_recovery_migration_names() -> frozenset[str]:
    """Return migration stems reserved for attested recovery preludes only."""

    migration_names = {
        record.recovery_migration_name
        for record in _HISTORICAL_FORWARD_RECOVERY_RECONCILIATIONS
    }
    migration_names.update(
        record.schema_binding_migration_name
        for record in _HISTORICAL_MISSING_SOURCE_FORWARD_RECOVERY_RECONCILIATIONS
    )
    return frozenset(migration_names)


_PRESENCE_UNKNOWN_COUNT_COLUMN = ("integer", "YES", "0")

_B2B_CAMPAIGN_PARTNER_COLUMN = ("uuid", True, False)
_B2B_CAMPAIGN_PARTNER_FOREIGN_KEY = {
    "constraint_type": "f",
    "key_columns": ("partner_id",),
    "referenced_table": "affiliate_partners",
    "referenced_columns": ("id",),
    "delete_action": "n",
    "update_action": "a",
    "match_type": "s",
}
_B2B_CAMPAIGN_PARTNER_INDEX = {
    "key_column": "partner_id",
    "predicate": "(partner_id is not null)",
}

_B2B_COMPANY_SIGNAL_PROMOTION_COLUMN = ("text", True, False)
_B2B_COMPANY_SIGNAL_PROMOTION_INDEX = {
    "access_method": "btree",
    "key_column": "canonical_promotion_type",
    "predicate": "(canonical_promotion_type is not null)",
}


@dataclass(frozen=True)
class _B2BWatchlistAlertEventConstraint:
    """One source-era, named alert-event constraint requirement."""

    constraint_type: str
    key_columns: tuple[str, ...]
    referenced_table: str | None = None
    referenced_columns: tuple[str, ...] = ()
    delete_action: str | None = None
    update_action: str | None = None
    match_type: str | None = None
    expression: str | None = None
    expected_internal_trigger_count: int = 0


@dataclass(frozen=True)
class _B2BWatchlistAlertEventIndex:
    """One source-era, named alert-event index requirement."""

    unique: bool
    key_columns: tuple[str, ...]
    definition_fragment: str
    predicate: str | None = None


_B2B_WATCHLIST_ALERT_EVENT_BASE_COLUMNS = {
    "id": ("uuid", False, None),
    "account_id": ("uuid", False, None),
    "watchlist_view_id": ("uuid", False, None),
    "event_type": ("text", False, None),
    "threshold_field": ("text", False, None),
    "entity_type": ("text", False, None),
    "entity_key": ("text", False, None),
    "vendor_name": ("text", True, None),
    "company_name": ("text", True, None),
    "category": ("text", True, None),
    "source": ("text", True, None),
    "threshold_value": ("numeric(6,2)", True, None),
    "summary": ("text", False, None),
    "payload": ("jsonb", False, "'{}'::jsonb"),
    "status": ("text", False, "'open'"),
    "first_seen_at": ("timestamp with time zone", False, "now()"),
    "last_seen_at": ("timestamp with time zone", False, "now()"),
    "resolved_at": ("timestamp with time zone", True, None),
    "created_at": ("timestamp with time zone", False, "now()"),
    "updated_at": ("timestamp with time zone", False, "now()"),
}

# Retained migration 281 is the only later source that changes the live table.
# The writer references this column in its conflict update, so it is part of the
# closed compatibility receipt rather than an arbitrary additive extension.
_B2B_WATCHLIST_ALERT_EVENT_KNOWN_LATER_COLUMNS = {
    "reopen_count": ("integer", False, "0"),
}
_B2B_WATCHLIST_ALERT_EVENT_ALLOWED_COLUMNS = {
    **_B2B_WATCHLIST_ALERT_EVENT_BASE_COLUMNS,
    **_B2B_WATCHLIST_ALERT_EVENT_KNOWN_LATER_COLUMNS,
}

# PostgreSQL implements each ordinary non-deferrable foreign key with two
# internal constraint triggers on each participating relation. The source-era
# receipt needs all four in origin mode, not just a validated pg_constraint row.
_B2B_WATCHLIST_ALERT_EVENT_FOREIGN_KEY_INTERNAL_TRIGGER_COUNT = 4

_B2B_WATCHLIST_ALERT_EVENT_CONSTRAINTS = {
    "b2b_watchlist_alert_events_pkey": _B2BWatchlistAlertEventConstraint(
        "p", ("id",)
    ),
    "b2b_watchlist_alert_events_account_id_fkey": (
        _B2BWatchlistAlertEventConstraint(
            "f", ("account_id",),
            referenced_table="saas_accounts",
            referenced_columns=("id",),
            delete_action="c",
            update_action="a",
            match_type="s",
            expected_internal_trigger_count=(
                _B2B_WATCHLIST_ALERT_EVENT_FOREIGN_KEY_INTERNAL_TRIGGER_COUNT
            ),
        )
    ),
    "b2b_watchlist_alert_events_watchlist_view_id_fkey": (
        _B2BWatchlistAlertEventConstraint(
            "f", ("watchlist_view_id",),
            referenced_table="b2b_watchlist_views",
            referenced_columns=("id",),
            delete_action="c",
            update_action="a",
            match_type="s",
            expected_internal_trigger_count=(
                _B2B_WATCHLIST_ALERT_EVENT_FOREIGN_KEY_INTERNAL_TRIGGER_COUNT
            ),
        )
    ),
    "chk_b2b_watchlist_alert_events_event_type": (
        _B2BWatchlistAlertEventConstraint(
            "c", ("event_type",),
            expression=(
                "(event_type=any(array['vendor_alert','account_alert',"
                "'stale_data']))"
            ),
        )
    ),
    "chk_b2b_watchlist_alert_events_threshold_field": (
        _B2BWatchlistAlertEventConstraint(
            "c", ("threshold_field",),
            expression=(
                "(threshold_field=any(array['vendor_alert_threshold',"
                "'account_alert_threshold','stale_days_threshold']))"
            ),
        )
    ),
    "chk_b2b_watchlist_alert_events_entity_type": (
        _B2BWatchlistAlertEventConstraint(
            "c", ("entity_type",),
            expression=(
                "(entity_type=any(array['vendor','account',"
                "'signal_cluster']))"
            ),
        )
    ),
    "chk_b2b_watchlist_alert_events_status": _B2BWatchlistAlertEventConstraint(
        "c", ("status",),
        expression="(status=any(array['open','resolved']))",
    ),
}

_B2B_WATCHLIST_ALERT_EVENT_INDEXES = {
    "idx_b2b_watchlist_alert_events_view_entity": _B2BWatchlistAlertEventIndex(
        unique=True,
        key_columns=("watchlist_view_id", "event_type", "entity_key"),
        definition_fragment="usingbtree(watchlist_view_id,event_type,entity_key)",
    ),
    "idx_b2b_watchlist_alert_events_account_status": (
        _B2BWatchlistAlertEventIndex(
            unique=False,
            key_columns=("account_id", "status", "last_seen_at"),
            definition_fragment="usingbtree(account_id,status,last_seen_atdesc)",
        )
    ),
    "idx_b2b_watchlist_alert_events_view_status": _B2BWatchlistAlertEventIndex(
        unique=False,
        key_columns=("watchlist_view_id", "status", "last_seen_at"),
        definition_fragment="usingbtree(watchlist_view_id,status,last_seen_atdesc)",
    ),
}


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


_PLPGSQL_FUNCTION_BODY_RE = re.compile(
    r"AS\s+\$function\$(.*?)\$function\s*\$;",
    re.DOTALL,
)


def _packaged_migration_function_body(
    migration_files: Collection[Path],
    migration_name: str,
) -> str | None:
    """Return the named packaged PL/pgSQL body without inventing source facts."""
    migration_file = next(
        (path for path in migration_files if path.stem == migration_name),
        None,
    )
    if migration_file is None:
        return None
    try:
        source = migration_file.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _PLPGSQL_FUNCTION_BODY_RE.search(source)
    return None if match is None else match.group(1)


def _packaged_migration_function_body_sha256(
    migration_files: Collection[Path],
    migration_name: str,
) -> str | None:
    """Return the exact named PL/pgSQL body digest from packaged source."""
    function_body = _packaged_migration_function_body(migration_files, migration_name)
    if function_body is None:
        return None
    return hashlib.sha256(function_body.encode("utf-8")).hexdigest()


def _quoted_sql_identifier(identifier: str) -> str:
    """Return the forced identifier quoting used by migration 390's body."""
    return '"' + identifier.replace('"', '""') + '"'


def _rendered_packaged_migration_function_body_sha256(
    migration_files: Collection[Path],
    migration_name: str,
    *,
    schema_name: str,
) -> str | None:
    """Digest a package's schema-bound PL/pgSQL body exactly as 390 renders it."""
    function_body = _packaged_migration_function_body(migration_files, migration_name)
    if function_body is None:
        return None
    rendered_body = function_body.replace("%2$s", _quoted_sql_identifier(schema_name))
    return hashlib.sha256(rendered_body.encode("utf-8")).hexdigest()


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


_WATCHLIST_ALERT_EVENT_SQL_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")
_WATCHLIST_ALERT_EVENT_REMOVABLE_CAST_RE = re.compile(
    r"::(?:character varying|varchar|text|name)(?:\[\])?",
    re.IGNORECASE,
)


def _canonicalize_watchlist_alert_event_expression(expression: object) -> str:
    """Normalize only unquoted SQL while preserving every literal exactly.

    The named 272 receipt compares source-era defaults and check constraints.
    Tokenizing literals first prevents the comparator from collapsing either
    distinct contents (``'open'`` versus ``'o''pen'``) or distinct case
    (``'open'`` versus ``'OPEN'``). Keep this narrower than the older generic
    normalizer so established historical receipts retain their contract.
    """
    raw_expression = str(expression or "")
    fragments: list[str] = []
    cursor = 0
    for literal in _WATCHLIST_ALERT_EVENT_SQL_LITERAL_RE.finditer(raw_expression):
        fragments.append(
            _canonicalize_watchlist_alert_event_unquoted_sql(
                raw_expression[cursor : literal.start()]
            )
        )
        fragments.append(literal.group(0))
        cursor = literal.end()
    fragments.append(
        _canonicalize_watchlist_alert_event_unquoted_sql(raw_expression[cursor:])
    )
    return "".join(fragments)


def _canonicalize_watchlist_alert_event_unquoted_sql(fragment: str) -> str:
    """Normalize only SQL syntax surrounding an already-isolated literal."""
    without_removable_casts = _WATCHLIST_ALERT_EVENT_REMOVABLE_CAST_RE.sub(
        "",
        fragment.lower(),
    )
    return re.sub(r"\s+", "", without_removable_casts)


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


def _catalog_text_values(value: object) -> tuple[str, ...]:
    """Return a PostgreSQL text array as an immutable exact-value tuple."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (str(value),)
    return tuple(str(item) for item in value)


def _catalog_function_body_sha256(value: object) -> str | None:
    """Hash only catalog function source text, failing closed on unexpected data."""
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _catalog_json_mapping(value: object) -> Mapping[str, object]:
    """Decode a catalog JSON object, failing closed on malformed metadata."""
    decoded = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return decoded if isinstance(decoded, Mapping) else {}


def _watchlist_alert_event_column_ready(
    observed: Mapping[str, object],
    expected: tuple[str, bool, str | None],
) -> bool:
    """Require one explicitly approved alert-event column signature."""
    expected_type, expected_is_nullable, expected_default = expected
    observed_default = observed.get("column_default")
    canonical_default = (
        _canonicalize_watchlist_alert_event_expression(observed_default)
        if observed_default is not None
        else None
    )
    return all((
        observed.get("exists") is True,
        observed.get("data_type") == expected_type,
        bool(observed.get("is_nullable")) is expected_is_nullable,
        observed.get("is_generated") is False,
        observed.get("is_identity") is False,
        observed.get("uses_type_default_collation") is True,
        canonical_default == expected_default,
    ))


def _watchlist_alert_event_constraint_ready(
    observed: Mapping[str, object],
    expected: _B2BWatchlistAlertEventConstraint,
) -> bool:
    """Require one named source-era constraint without reading alert rows."""
    referenced_table_ready = (
        expected.referenced_table is None
        or (
            observed.get("referenced_table") == expected.referenced_table
            and bool(observed.get("references_current_schema"))
        )
    )
    delete_action_ready = (
        expected.delete_action is None
        or _catalog_char(observed.get("delete_action")) == expected.delete_action
    )
    update_action_ready = (
        expected.update_action is None
        or _catalog_char(observed.get("update_action")) == expected.update_action
    )
    match_type_ready = (
        expected.match_type is None
        or _catalog_char(observed.get("match_type")) == expected.match_type
    )
    expression_ready = (
        expected.expression is None
        or _canonicalize_watchlist_alert_event_expression(
            observed.get("expression")
        )
        == expected.expression
    )
    internal_trigger_enforcement_ready = all((
        int(observed.get("internal_trigger_count") or 0)
        == expected.expected_internal_trigger_count,
        int(observed.get("origin_enabled_internal_trigger_count") or 0)
        == expected.expected_internal_trigger_count,
    ))
    return all((
        _catalog_char(observed.get("constraint_type")) == expected.constraint_type,
        _catalog_column_names(observed.get("key_columns")) == expected.key_columns,
        referenced_table_ready,
        _catalog_column_names(observed.get("referenced_columns"))
        == expected.referenced_columns,
        delete_action_ready,
        update_action_ready,
        match_type_ready,
        not bool(observed.get("is_deferrable")),
        not bool(observed.get("is_initially_deferred")),
        bool(observed.get("is_validated")),
        expression_ready,
        internal_trigger_enforcement_ready,
    ))


def _watchlist_alert_event_index_ready(
    observed: Mapping[str, object],
    expected: _B2BWatchlistAlertEventIndex,
) -> bool:
    """Require one named source-era index with its key order and readiness."""
    observed_key_columns = tuple(
        _normalize_schema_definition(column)
        for column in _catalog_column_names(observed.get("key_columns"))
    )
    observed_predicate = observed.get("predicate")
    predicate_ready = (
        observed_predicate is None
        if expected.predicate is None
        else _canonicalize_watchlist_alert_event_expression(observed_predicate)
        == expected.predicate
    )
    return all((
        _catalog_char(observed.get("relation_kind")) == "i",
        not bool(observed.get("is_partition")),
        bool(observed.get("is_unique")) is expected.unique,
        bool(observed.get("is_valid")),
        bool(observed.get("is_ready")),
        int(observed.get("key_attribute_count") or 0) == len(expected.key_columns),
        int(observed.get("attribute_count") or 0) == len(expected.key_columns),
        observed_key_columns == expected.key_columns,
        expected.definition_fragment
        in _canonicalize_catalog_index_definition(observed.get("definition")),
        predicate_ready,
    ))


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
    """Read identity, column, and constraint metadata from one snapshot only."""
    evidence_row = await executor.fetchrow(
        """
        WITH target_relation AS (
            SELECT
                relation_state.oid,
                relation_state.relkind,
                relation_state.relispartition
            FROM pg_class AS relation_state
            JOIN pg_namespace AS schema_state
              ON schema_state.oid = relation_state.relnamespace
            WHERE schema_state.nspname = current_schema()
              AND relation_state.relname = 'presence_events'
        ), relation_evidence AS (
            SELECT EXISTS (
                SELECT 1
                FROM target_relation
                WHERE relkind = 'r'
                  AND NOT relispartition
            ) AS presence_events_is_ordinary_table
        ), constraint_evidence AS (
            SELECT NOT EXISTS (
                SELECT 1
                FROM pg_constraint AS actual
                JOIN target_relation
                  ON target_relation.oid = actual.conrelid
                WHERE EXISTS (
                    SELECT 1
                    FROM unnest(
                        COALESCE(actual.conkey, ARRAY[]::smallint[])
                    ) AS key_state(attnum)
                    JOIN pg_attribute AS attribute_state
                      ON attribute_state.attrelid = actual.conrelid
                     AND attribute_state.attnum = key_state.attnum
                    WHERE attribute_state.attname = 'unknown_count'
                )
            ) AS unknown_count_has_no_constraints
        )
        SELECT
            relation_evidence.presence_events_is_ordinary_table,
            constraint_evidence.unknown_count_has_no_constraints,
            actual.column_name,
            actual.data_type,
            actual.is_nullable,
            actual.column_default
        FROM relation_evidence
        CROSS JOIN constraint_evidence
        LEFT JOIN information_schema.columns AS actual
          ON actual.table_schema = current_schema()
         AND actual.table_name = 'presence_events'
         AND actual.column_name = 'unknown_count'
        """
    )
    if evidence_row is None:
        return False, False, False

    presence_events_is_ordinary_table = bool(
        evidence_row["presence_events_is_ordinary_table"]
    )
    unknown_count_column_ready = (
        evidence_row["column_name"] == "unknown_count"
        and evidence_row["data_type"] == _PRESENCE_UNKNOWN_COUNT_COLUMN[0]
        and evidence_row["is_nullable"] == _PRESENCE_UNKNOWN_COUNT_COLUMN[1]
        and (
            _canonicalize_catalog_constraint_expression(evidence_row["column_default"])
            if evidence_row["column_default"] is not None
            else None
        )
        == _PRESENCE_UNKNOWN_COUNT_COLUMN[2]
    )
    unknown_count_has_no_constraints = bool(
        evidence_row["unknown_count_has_no_constraints"]
    )
    return (
        presence_events_is_ordinary_table,
        unknown_count_column_ready,
        unknown_count_has_no_constraints,
    )


async def _migration_067_catalog_evidence(
    executor: Any,
) -> tuple[bool, bool, bool, bool]:
    """Read one immutable campaign-partner schema receipt from one snapshot."""
    evidence_row = await executor.fetchrow(
        """
        WITH target_relation AS (
            SELECT
                relation_state.oid,
                relation_state.relkind,
                relation_state.relispartition
            FROM pg_class AS relation_state
            JOIN pg_namespace AS schema_state
              ON schema_state.oid = relation_state.relnamespace
            WHERE schema_state.nspname = current_schema()
              AND relation_state.relname = 'b2b_campaigns'
        ), partner_column AS (
            SELECT
                attribute_state.attname AS column_name,
                format_type(
                    attribute_state.atttypid,
                    attribute_state.atttypmod
                ) AS data_type,
                NOT attribute_state.attnotnull AS is_nullable,
                default_state.oid IS NOT NULL AS has_default
            FROM target_relation
            JOIN pg_attribute AS attribute_state
              ON attribute_state.attrelid = target_relation.oid
            LEFT JOIN pg_attrdef AS default_state
              ON default_state.adrelid = attribute_state.attrelid
             AND default_state.adnum = attribute_state.attnum
            WHERE attribute_state.attname = 'partner_id'
              AND attribute_state.attnum > 0
              AND NOT attribute_state.attisdropped
        ), partner_foreign_key AS (
            SELECT
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
                (referenced_schema.nspname = current_schema())
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
                actual.convalidated AS is_validated
            FROM target_relation
            JOIN pg_constraint AS actual
              ON actual.conrelid = target_relation.oid
            LEFT JOIN pg_class AS referenced_table
              ON referenced_table.oid = actual.confrelid
            LEFT JOIN pg_namespace AS referenced_schema
              ON referenced_schema.oid = referenced_table.relnamespace
            WHERE actual.conname = 'b2b_campaigns_partner_id_fkey'
        ), partner_index AS (
            SELECT
                index_relation.relkind AS relation_kind,
                index_relation.relispartition AS is_partition,
                index_state.indisunique AS is_unique,
                index_state.indisvalid AS is_valid,
                index_state.indisready AS is_ready,
                index_state.indnkeyatts AS key_attribute_count,
                index_state.indnatts AS attribute_count,
                pg_get_indexdef(index_state.indexrelid, 1, true) AS key_column,
                pg_get_expr(
                    index_state.indpred,
                    index_state.indrelid
                ) AS predicate
            FROM target_relation
            JOIN pg_index AS index_state
              ON index_state.indrelid = target_relation.oid
            JOIN pg_class AS index_relation
              ON index_relation.oid = index_state.indexrelid
            WHERE index_relation.relname = 'idx_b2b_campaigns_partner'
        )
        SELECT
            EXISTS (
                SELECT 1
                FROM target_relation
                WHERE relkind = 'r'
                  AND NOT relispartition
            ) AS b2b_campaigns_is_ordinary_table,
            (SELECT column_name FROM partner_column LIMIT 1)
                AS partner_id_column_name,
            (SELECT data_type FROM partner_column LIMIT 1)
                AS partner_id_data_type,
            (SELECT is_nullable FROM partner_column LIMIT 1)
                AS partner_id_is_nullable,
            (SELECT has_default FROM partner_column LIMIT 1)
                AS partner_id_has_default,
            (SELECT constraint_type FROM partner_foreign_key LIMIT 1)
                AS partner_foreign_key_constraint_type,
            (SELECT key_columns FROM partner_foreign_key LIMIT 1)
                AS partner_foreign_key_columns,
            (SELECT referenced_table FROM partner_foreign_key LIMIT 1)
                AS partner_foreign_key_referenced_table,
            (SELECT references_current_schema FROM partner_foreign_key LIMIT 1)
                AS partner_foreign_key_references_current_schema,
            (SELECT referenced_columns FROM partner_foreign_key LIMIT 1)
                AS partner_foreign_key_referenced_columns,
            (SELECT delete_action FROM partner_foreign_key LIMIT 1)
                AS partner_foreign_key_delete_action,
            (SELECT update_action FROM partner_foreign_key LIMIT 1)
                AS partner_foreign_key_update_action,
            (SELECT match_type FROM partner_foreign_key LIMIT 1)
                AS partner_foreign_key_match_type,
            (SELECT is_deferrable FROM partner_foreign_key LIMIT 1)
                AS partner_foreign_key_is_deferrable,
            (SELECT is_initially_deferred FROM partner_foreign_key LIMIT 1)
                AS partner_foreign_key_is_initially_deferred,
            (SELECT is_validated FROM partner_foreign_key LIMIT 1)
                AS partner_foreign_key_is_validated,
            (SELECT relation_kind FROM partner_index LIMIT 1)
                AS partner_index_relation_kind,
            (SELECT is_partition FROM partner_index LIMIT 1)
                AS partner_index_is_partition,
            (SELECT is_unique FROM partner_index LIMIT 1)
                AS partner_index_is_unique,
            (SELECT is_valid FROM partner_index LIMIT 1)
                AS partner_index_is_valid,
            (SELECT is_ready FROM partner_index LIMIT 1)
                AS partner_index_is_ready,
            (SELECT key_attribute_count FROM partner_index LIMIT 1)
                AS partner_index_key_attribute_count,
            (SELECT attribute_count FROM partner_index LIMIT 1)
                AS partner_index_attribute_count,
            (SELECT key_column FROM partner_index LIMIT 1)
                AS partner_index_key_column,
            (SELECT predicate FROM partner_index LIMIT 1)
                AS partner_index_predicate
        """
    )
    if evidence_row is None:
        return False, False, False, False

    b2b_campaigns_is_ordinary_table = bool(
        evidence_row["b2b_campaigns_is_ordinary_table"]
    )
    expected_column_type, expected_is_nullable, expected_has_default = (
        _B2B_CAMPAIGN_PARTNER_COLUMN
    )
    partner_id_column_ready = all((
        evidence_row["partner_id_column_name"] == "partner_id",
        evidence_row["partner_id_data_type"] == expected_column_type,
        evidence_row["partner_id_is_nullable"] is expected_is_nullable,
        evidence_row["partner_id_has_default"] is expected_has_default,
    ))
    foreign_key = _B2B_CAMPAIGN_PARTNER_FOREIGN_KEY
    partner_foreign_key_ready = all((
        _catalog_char(evidence_row["partner_foreign_key_constraint_type"])
        == foreign_key["constraint_type"],
        _catalog_column_names(evidence_row["partner_foreign_key_columns"])
        == foreign_key["key_columns"],
        evidence_row["partner_foreign_key_referenced_table"]
        == foreign_key["referenced_table"],
        bool(evidence_row["partner_foreign_key_references_current_schema"]),
        _catalog_column_names(evidence_row["partner_foreign_key_referenced_columns"])
        == foreign_key["referenced_columns"],
        _catalog_char(evidence_row["partner_foreign_key_delete_action"])
        == foreign_key["delete_action"],
        _catalog_char(evidence_row["partner_foreign_key_update_action"])
        == foreign_key["update_action"],
        _catalog_char(evidence_row["partner_foreign_key_match_type"])
        == foreign_key["match_type"],
        not bool(evidence_row["partner_foreign_key_is_deferrable"]),
        not bool(evidence_row["partner_foreign_key_is_initially_deferred"]),
        bool(evidence_row["partner_foreign_key_is_validated"]),
    ))
    index = _B2B_CAMPAIGN_PARTNER_INDEX
    partner_partial_index_ready = all((
        _catalog_char(evidence_row["partner_index_relation_kind"]) == "i",
        not bool(evidence_row["partner_index_is_partition"]),
        evidence_row["partner_index_is_unique"] is False,
        bool(evidence_row["partner_index_is_valid"]),
        bool(evidence_row["partner_index_is_ready"]),
        int(evidence_row["partner_index_key_attribute_count"] or 0) == 1,
        int(evidence_row["partner_index_attribute_count"] or 0) == 1,
        evidence_row["partner_index_key_column"] == index["key_column"],
        _canonicalize_catalog_constraint_expression(
            evidence_row["partner_index_predicate"]
        )
        == index["predicate"],
    ))
    return (
        b2b_campaigns_is_ordinary_table,
        partner_id_column_ready,
        partner_foreign_key_ready,
        partner_partial_index_ready,
    )


async def _migration_297_catalog_evidence(
    executor: Any,
) -> tuple[bool, bool, bool, bool]:
    """Read the named 297 column/index receipt in one catalog-only snapshot."""
    evidence_row = await executor.fetchrow(
        """
        WITH target_relation AS (
            SELECT
                relation_state.oid,
                relation_state.relkind,
                relation_state.relpersistence,
                relation_state.relispartition
            FROM pg_class AS relation_state
            JOIN pg_namespace AS schema_state
              ON schema_state.oid = relation_state.relnamespace
            WHERE schema_state.nspname = current_schema()
              AND relation_state.relname = 'b2b_company_signals'
        ), promotion_column AS (
            SELECT
                attribute_state.attnum AS column_number,
                attribute_state.attname AS column_name,
                format_type(
                    attribute_state.atttypid,
                    attribute_state.atttypmod
                ) AS data_type,
                NOT attribute_state.attnotnull AS is_nullable,
                default_state.oid IS NOT NULL AS has_default,
                attribute_state.attgenerated <> ''::"char" AS is_generated,
                attribute_state.attidentity <> ''::"char" AS is_identity,
                attribute_state.attcollation = type_state.typcollation
                    AS uses_type_default_collation
            FROM target_relation
            JOIN pg_attribute AS attribute_state
              ON attribute_state.attrelid = target_relation.oid
            JOIN pg_type AS type_state
              ON type_state.oid = attribute_state.atttypid
            LEFT JOIN pg_attrdef AS default_state
              ON default_state.adrelid = attribute_state.attrelid
             AND default_state.adnum = attribute_state.attnum
            WHERE attribute_state.attname = 'canonical_promotion_type'
              AND attribute_state.attnum > 0
              AND NOT attribute_state.attisdropped
        ), promotion_constraints AS (
            SELECT NOT EXISTS (
                SELECT 1
                FROM pg_constraint AS constraint_state
                WHERE constraint_state.conrelid = target_relation.oid
                  AND (
                      EXISTS (
                          SELECT 1
                          FROM promotion_column
                          WHERE promotion_column.column_number
                              = ANY(constraint_state.conkey)
                      )
                      OR position(
                          'canonical_promotion_type'
                          IN lower(
                              pg_get_constraintdef(constraint_state.oid, true)
                          )
                      ) > 0
                  )
            ) AS has_no_constraints
            FROM target_relation
        ), promotion_index AS (
            SELECT
                index_relation.relkind AS relation_kind,
                index_relation.relispartition AS is_partition,
                access_method.amname AS access_method,
                index_state.indisunique AS is_unique,
                index_state.indisvalid AS is_valid,
                index_state.indisready AS is_ready,
                index_state.indnkeyatts AS key_attribute_count,
                index_state.indnatts AS attribute_count,
                pg_get_indexdef(index_state.indexrelid, 1, true) AS key_column,
                pg_get_expr(
                    index_state.indpred,
                    index_state.indrelid
                ) AS predicate
            FROM target_relation
            JOIN pg_index AS index_state
              ON index_state.indrelid = target_relation.oid
            JOIN pg_class AS index_relation
              ON index_relation.oid = index_state.indexrelid
            JOIN pg_am AS access_method
              ON access_method.oid = index_relation.relam
            WHERE index_relation.relname =
                'idx_b2b_company_signals_canonical_promotion_type'
        )
        SELECT
            EXISTS (
                SELECT 1
                FROM target_relation
                WHERE relkind = 'r'
                  AND relpersistence = 'p'::"char"
                  AND NOT relispartition
            ) AS b2b_company_signals_is_ordinary_table,
            (SELECT column_name FROM promotion_column LIMIT 1)
                AS canonical_promotion_type_column_name,
            (SELECT data_type FROM promotion_column LIMIT 1)
                AS canonical_promotion_type_data_type,
            (SELECT is_nullable FROM promotion_column LIMIT 1)
                AS canonical_promotion_type_is_nullable,
            (SELECT has_default FROM promotion_column LIMIT 1)
                AS canonical_promotion_type_has_default,
            (SELECT is_generated FROM promotion_column LIMIT 1)
                AS canonical_promotion_type_is_generated,
            (SELECT is_identity FROM promotion_column LIMIT 1)
                AS canonical_promotion_type_is_identity,
            (SELECT uses_type_default_collation FROM promotion_column LIMIT 1)
                AS canonical_promotion_type_uses_type_default_collation,
            (SELECT has_no_constraints FROM promotion_constraints LIMIT 1)
                AS canonical_promotion_type_has_no_constraints,
            (SELECT relation_kind FROM promotion_index LIMIT 1)
                AS promotion_index_relation_kind,
            (SELECT is_partition FROM promotion_index LIMIT 1)
                AS promotion_index_is_partition,
            (SELECT access_method FROM promotion_index LIMIT 1)
                AS promotion_index_access_method,
            (SELECT is_unique FROM promotion_index LIMIT 1)
                AS promotion_index_is_unique,
            (SELECT is_valid FROM promotion_index LIMIT 1)
                AS promotion_index_is_valid,
            (SELECT is_ready FROM promotion_index LIMIT 1)
                AS promotion_index_is_ready,
            (SELECT key_attribute_count FROM promotion_index LIMIT 1)
                AS promotion_index_key_attribute_count,
            (SELECT attribute_count FROM promotion_index LIMIT 1)
                AS promotion_index_attribute_count,
            (SELECT key_column FROM promotion_index LIMIT 1)
                AS promotion_index_key_column,
            (SELECT predicate FROM promotion_index LIMIT 1)
                AS promotion_index_predicate
        """
    )
    if evidence_row is None:
        return False, False, False, False

    b2b_company_signals_is_ordinary_table = bool(
        evidence_row["b2b_company_signals_is_ordinary_table"]
    )
    expected_column_type, expected_is_nullable, expected_has_default = (
        _B2B_COMPANY_SIGNAL_PROMOTION_COLUMN
    )
    canonical_promotion_type_column_ready = all((
        evidence_row["canonical_promotion_type_column_name"]
        == "canonical_promotion_type",
        evidence_row["canonical_promotion_type_data_type"] == expected_column_type,
        evidence_row["canonical_promotion_type_is_nullable"] is expected_is_nullable,
        evidence_row["canonical_promotion_type_has_default"] is expected_has_default,
        evidence_row["canonical_promotion_type_is_generated"] is False,
        evidence_row["canonical_promotion_type_is_identity"] is False,
        evidence_row["canonical_promotion_type_uses_type_default_collation"] is True,
    ))
    canonical_promotion_type_has_no_constraints = bool(
        evidence_row["canonical_promotion_type_has_no_constraints"]
    )
    index = _B2B_COMPANY_SIGNAL_PROMOTION_INDEX
    canonical_promotion_type_partial_index_ready = all((
        _catalog_char(evidence_row["promotion_index_relation_kind"]) == "i",
        not bool(evidence_row["promotion_index_is_partition"]),
        evidence_row["promotion_index_access_method"] == index["access_method"],
        evidence_row["promotion_index_is_unique"] is False,
        bool(evidence_row["promotion_index_is_valid"]),
        bool(evidence_row["promotion_index_is_ready"]),
        int(evidence_row["promotion_index_key_attribute_count"] or 0) == 1,
        int(evidence_row["promotion_index_attribute_count"] or 0) == 1,
        evidence_row["promotion_index_key_column"] == index["key_column"],
        _canonicalize_catalog_constraint_expression(
            evidence_row["promotion_index_predicate"]
        )
        == index["predicate"],
    ))
    return (
        b2b_company_signals_is_ordinary_table,
        canonical_promotion_type_column_ready,
        canonical_promotion_type_has_no_constraints,
        canonical_promotion_type_partial_index_ready,
    )


async def _migration_272_catalog_evidence(
    executor: Any,
) -> tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]:
    """Read the named 272 base-table receipt in one catalog-only snapshot."""
    evidence_row = await executor.fetchrow(
        """
        WITH target_relation AS (
            SELECT
                relation_state.oid,
                relation_state.relkind,
                relation_state.relpersistence,
                relation_state.relispartition,
                relation_state.relrowsecurity,
                relation_state.relforcerowsecurity
            FROM pg_class AS relation_state
            JOIN pg_namespace AS schema_state
              ON schema_state.oid = relation_state.relnamespace
            WHERE schema_state.nspname = current_schema()
              AND relation_state.relname = 'b2b_watchlist_alert_events'
        ), requested_columns AS (
            SELECT requested.name
            FROM unnest($1::text[]) AS requested(name)
        ), column_evidence AS (
            SELECT
                requested.name,
                jsonb_build_object(
                    'exists', attribute_state.attname IS NOT NULL,
                    'data_type', format_type(
                        attribute_state.atttypid,
                        attribute_state.atttypmod
                    ),
                    'is_nullable', NOT attribute_state.attnotnull,
                    'is_generated', attribute_state.attgenerated <> ''::"char",
                    'is_identity', attribute_state.attidentity <> ''::"char",
                    'uses_type_default_collation',
                        attribute_state.attcollation = type_state.typcollation,
                    'column_default', CASE
                        WHEN default_state.oid IS NULL THEN NULL
                        ELSE pg_get_expr(
                            default_state.adbin,
                            default_state.adrelid
                        )
                    END
                ) AS evidence
            FROM requested_columns AS requested
            LEFT JOIN target_relation ON TRUE
            LEFT JOIN pg_attribute AS attribute_state
              ON attribute_state.attrelid = target_relation.oid
             AND attribute_state.attname = requested.name
             AND attribute_state.attnum > 0
             AND NOT attribute_state.attisdropped
            LEFT JOIN pg_type AS type_state
              ON type_state.oid = attribute_state.atttypid
            LEFT JOIN pg_attrdef AS default_state
              ON default_state.adrelid = attribute_state.attrelid
             AND default_state.adnum = attribute_state.attnum
        ), requested_constraints AS (
            SELECT requested.name
            FROM unnest($2::text[]) AS requested(name)
        ), constraint_evidence AS (
            SELECT
                requested.name,
                jsonb_build_object(
                    'constraint_type', actual.contype,
                    'key_columns', ARRAY(
                        SELECT attribute_state.attname
                        FROM unnest(actual.conkey)
                             WITH ORDINALITY AS key_state(attnum, ordinality)
                        JOIN pg_attribute AS attribute_state
                          ON attribute_state.attrelid = actual.conrelid
                         AND attribute_state.attnum = key_state.attnum
                        ORDER BY key_state.ordinality
                    ),
                    'referenced_table', referenced_table.relname,
                    'references_current_schema',
                        referenced_schema.nspname = current_schema(),
                    'referenced_columns', ARRAY(
                        SELECT attribute_state.attname
                        FROM unnest(actual.confkey)
                             WITH ORDINALITY AS key_state(attnum, ordinality)
                        JOIN pg_attribute AS attribute_state
                          ON attribute_state.attrelid = actual.confrelid
                         AND attribute_state.attnum = key_state.attnum
                        ORDER BY key_state.ordinality
                    ),
                    'delete_action', actual.confdeltype,
                    'update_action', actual.confupdtype,
                    'match_type', actual.confmatchtype,
                    'is_deferrable', actual.condeferrable,
                    'is_initially_deferred', actual.condeferred,
                    'is_validated', actual.convalidated,
                    'internal_trigger_count', (
                        SELECT COUNT(*)
                        FROM pg_trigger AS constraint_trigger
                        WHERE constraint_trigger.tgconstraint = actual.oid
                          AND constraint_trigger.tgisinternal
                    ),
                    'origin_enabled_internal_trigger_count', (
                        SELECT COUNT(*)
                        FROM pg_trigger AS constraint_trigger
                        WHERE constraint_trigger.tgconstraint = actual.oid
                          AND constraint_trigger.tgisinternal
                          AND constraint_trigger.tgenabled = 'O'::"char"
                    ),
                    'expression', CASE
                        WHEN actual.oid IS NULL THEN NULL
                        ELSE pg_get_expr(actual.conbin, actual.conrelid)
                    END
                ) AS evidence
            FROM requested_constraints AS requested
            LEFT JOIN target_relation ON TRUE
            LEFT JOIN pg_constraint AS actual
              ON actual.conrelid = target_relation.oid
             AND actual.conname = requested.name
            LEFT JOIN pg_class AS referenced_table
              ON referenced_table.oid = actual.confrelid
            LEFT JOIN pg_namespace AS referenced_schema
              ON referenced_schema.oid = referenced_table.relnamespace
        ), requested_indexes AS (
            SELECT requested.name
            FROM unnest($3::text[]) AS requested(name)
        ), index_evidence AS (
            SELECT
                requested.name,
                jsonb_build_object(
                    'relation_kind', index_relation.relkind,
                    'is_partition', index_relation.relispartition,
                    'is_unique', index_state.indisunique,
                    'is_valid', index_state.indisvalid,
                    'is_ready', index_state.indisready,
                    'key_attribute_count', index_state.indnkeyatts,
                    'attribute_count', index_state.indnatts,
                    'definition', CASE
                        WHEN index_state.indexrelid IS NULL THEN NULL
                        ELSE pg_get_indexdef(index_state.indexrelid)
                    END,
                    'key_columns', ARRAY(
                        SELECT attribute_state.attname
                        FROM unnest(index_state.indkey)
                             WITH ORDINALITY AS key_state(attnum, ordinality)
                        JOIN pg_attribute AS attribute_state
                          ON attribute_state.attrelid = index_state.indrelid
                         AND attribute_state.attnum = key_state.attnum
                        WHERE key_state.ordinality <= index_state.indnkeyatts
                        ORDER BY key_state.ordinality
                    ),
                    'predicate', CASE
                        WHEN index_state.indexrelid IS NULL THEN NULL
                        ELSE pg_get_expr(
                            index_state.indpred,
                            index_state.indrelid
                        )
                    END
                ) AS evidence
            FROM requested_indexes AS requested
            LEFT JOIN target_relation ON TRUE
            LEFT JOIN pg_class AS index_relation
              ON index_relation.relnamespace = (
                    SELECT oid
                    FROM pg_namespace
                    WHERE nspname = current_schema()
                )
             AND index_relation.relname = requested.name
            LEFT JOIN pg_index AS index_state
              ON index_state.indexrelid = index_relation.oid
             AND index_state.indrelid = target_relation.oid
        ), unlisted_columns AS (
            SELECT
                EXISTS (SELECT 1 FROM target_relation)
                AND NOT EXISTS (
                    SELECT 1
                    FROM pg_attribute AS actual
                    JOIN target_relation
                      ON target_relation.oid = actual.attrelid
                    WHERE actual.attnum > 0
                      AND NOT actual.attisdropped
                      AND actual.attname <> ALL($1::text[])
                ) AS no_unlisted_alert_event_columns
        ), unlisted_constraints AS (
            SELECT
                EXISTS (SELECT 1 FROM target_relation)
                AND NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint AS actual
                    JOIN target_relation
                      ON target_relation.oid = actual.conrelid
                    WHERE actual.conname <> ALL($2::text[])
                ) AS no_unlisted_alert_event_constraints
        ), unlisted_indexes AS (
            SELECT
                EXISTS (SELECT 1 FROM target_relation)
                AND NOT EXISTS (
                    SELECT 1
                    FROM pg_index AS index_state
                    JOIN target_relation
                      ON target_relation.oid = index_state.indrelid
                    JOIN pg_class AS index_relation
                      ON index_relation.oid = index_state.indexrelid
                    LEFT JOIN pg_constraint AS backing_constraint
                      ON backing_constraint.conindid = index_state.indexrelid
                    WHERE index_relation.relname <> ALL($3::text[])
                      AND (
                          backing_constraint.oid IS NULL
                          OR backing_constraint.conname <> ALL($2::text[])
                      )
                ) AS no_unlisted_alert_event_indexes
        ), unreviewed_write_interceptors AS (
            SELECT
                EXISTS (SELECT 1 FROM target_relation)
                AND NOT EXISTS (
                    SELECT 1
                    FROM pg_trigger AS trigger_state
                    JOIN target_relation
                      ON target_relation.oid = trigger_state.tgrelid
                    WHERE NOT trigger_state.tgisinternal
                    UNION ALL
                    SELECT 1
                    FROM pg_rewrite AS rule_state
                    JOIN target_relation
                      ON target_relation.oid = rule_state.ev_class
                    WHERE rule_state.rulename <> '_RETURN'
                    UNION ALL
                    SELECT 1
                    FROM target_relation
                    WHERE relrowsecurity OR relforcerowsecurity
                    UNION ALL
                    SELECT 1
                    FROM pg_policy AS policy_state
                    JOIN target_relation
                      ON target_relation.oid = policy_state.polrelid
                ) AS no_unreviewed_alert_event_write_interceptors
        )
        SELECT jsonb_build_object(
            'watchlist_alert_events_is_ordinary_table', EXISTS (
                SELECT 1
                FROM target_relation
                WHERE relkind = 'r'
                  AND NOT relispartition
            ),
            'watchlist_alert_events_has_permanent_storage', EXISTS (
                SELECT 1
                FROM target_relation
                WHERE relpersistence = 'p'
            ),
            'columns', (
                SELECT jsonb_object_agg(name, evidence)
                FROM column_evidence
            ),
            'no_unlisted_alert_event_columns', (
                SELECT no_unlisted_alert_event_columns
                FROM unlisted_columns
            ),
            'constraints', (
                SELECT jsonb_object_agg(name, evidence)
                FROM constraint_evidence
            ),
            'no_unlisted_alert_event_constraints', (
                SELECT no_unlisted_alert_event_constraints
                FROM unlisted_constraints
            ),
            'indexes', (
                SELECT jsonb_object_agg(name, evidence)
                FROM index_evidence
            ),
            'no_unlisted_alert_event_indexes', (
                SELECT no_unlisted_alert_event_indexes
                FROM unlisted_indexes
            ),
            'no_unreviewed_alert_event_write_interceptors', (
                SELECT no_unreviewed_alert_event_write_interceptors
                FROM unreviewed_write_interceptors
            )
        ) AS catalog_evidence
        """,
        list(_B2B_WATCHLIST_ALERT_EVENT_ALLOWED_COLUMNS),
        list(_B2B_WATCHLIST_ALERT_EVENT_CONSTRAINTS),
        list(_B2B_WATCHLIST_ALERT_EVENT_INDEXES),
    )
    if evidence_row is None:
        return False, False, False, False, False, False, False, False, False, False

    catalog = _catalog_json_mapping(evidence_row["catalog_evidence"])
    observed_columns = _catalog_json_mapping(catalog.get("columns"))
    observed_constraints = _catalog_json_mapping(catalog.get("constraints"))
    observed_indexes = _catalog_json_mapping(catalog.get("indexes"))
    base_alert_event_columns_ready = all(
        _watchlist_alert_event_column_ready(
            _catalog_json_mapping(observed_columns.get(name)),
            expected,
        )
        for name, expected in _B2B_WATCHLIST_ALERT_EVENT_BASE_COLUMNS.items()
    )
    known_later_alert_event_columns_ready = all(
        _watchlist_alert_event_column_ready(
            _catalog_json_mapping(observed_columns.get(name)),
            expected,
        )
        for name, expected in _B2B_WATCHLIST_ALERT_EVENT_KNOWN_LATER_COLUMNS.items()
    )
    required_alert_event_constraints_ready = all(
        _watchlist_alert_event_constraint_ready(
            _catalog_json_mapping(observed_constraints.get(name)),
            expected,
        )
        for name, expected in _B2B_WATCHLIST_ALERT_EVENT_CONSTRAINTS.items()
    )
    required_alert_event_indexes_ready = all(
        _watchlist_alert_event_index_ready(
            _catalog_json_mapping(observed_indexes.get(name)),
            expected,
        )
        for name, expected in _B2B_WATCHLIST_ALERT_EVENT_INDEXES.items()
    )
    return (
        bool(catalog.get("watchlist_alert_events_is_ordinary_table")),
        bool(catalog.get("watchlist_alert_events_has_permanent_storage")),
        base_alert_event_columns_ready,
        known_later_alert_event_columns_ready,
        bool(catalog.get("no_unlisted_alert_event_columns")),
        required_alert_event_constraints_ready,
        bool(catalog.get("no_unlisted_alert_event_constraints")),
        required_alert_event_indexes_ready,
        bool(catalog.get("no_unlisted_alert_event_indexes")),
        bool(catalog.get("no_unreviewed_alert_event_write_interceptors")),
    )


async def _migration_379_catalog_evidence(executor: Any) -> Mapping[str, object]:
    """Read only the reviewed commercial-billing fence metadata in one schema."""
    evidence_row = await executor.fetchrow(
        """
        WITH target_relations AS (
            SELECT relation_state.oid, relation_state.relname,
                   relation_state.relkind, relation_state.relpersistence,
                   relation_state.relispartition,
                   relation_state.relrowsecurity,
                   relation_state.relforcerowsecurity
            FROM pg_catalog.pg_class AS relation_state
            JOIN pg_catalog.pg_namespace AS namespace_state
              ON namespace_state.oid = relation_state.relnamespace
            WHERE namespace_state.nspname = pg_catalog.current_schema()
              AND relation_state.relname = ANY (
                  ARRAY[
                      'commercial_billing_candidate_review_decisions',
                      'commercial_billing_candidate_overrides',
                      'commercial_billing_run_candidates',
                      'invoices'
                  ]::text[]
              )
        ),
        required_columns AS (
            SELECT *
            FROM (
                VALUES
                    ('commercial_billing_run_candidates', 'id', 'uuid', -1, TRUE, TRUE),
                    ('commercial_billing_run_candidates', 'billing_run_id', 'uuid', -1, TRUE, TRUE),
                    ('commercial_billing_run_candidates', 'candidate_key', 'varchar', 516, TRUE, TRUE),
                    ('commercial_billing_run_candidates', 'source_fingerprint', 'varchar', 68, TRUE, TRUE),
                    ('commercial_billing_run_candidates', 'display_order', 'int4', -1, TRUE, TRUE),
                    ('commercial_billing_run_candidates', 'snapshot', 'jsonb', -1, TRUE, TRUE),
                    ('commercial_billing_run_candidates', 'created_at', 'timestamptz', -1, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'id', 'uuid', -1, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'billing_run_id', 'uuid', -1, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'candidate_key', 'varchar', 516, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'source_fingerprint', 'varchar', 68, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'revision', 'int4', -1, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'decision', 'varchar', 20, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'reason', 'varchar', 1004, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'source', 'varchar', 36, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'idempotency_key', 'varchar', 132, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'request_fingerprint', 'varchar', 68, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'decided_by', 'varchar', 132, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'decided_at', 'timestamptz', -1, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'created_at', 'timestamptz', -1, TRUE, TRUE),
                    ('commercial_billing_candidate_review_decisions', 'review_fingerprint', 'varchar', 68, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'id', 'uuid', -1, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'billing_run_id', 'uuid', -1, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'candidate_key', 'varchar', 516, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'source_fingerprint', 'varchar', 68, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'revision', 'int4', -1, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'review_fingerprint', 'varchar', 68, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'effective_snapshot', 'jsonb', -1, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'reason_code', 'varchar', 68, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'reason', 'varchar', 1004, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'source', 'varchar', 36, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'idempotency_key', 'varchar', 132, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'request_fingerprint', 'varchar', 68, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'overridden_by', 'varchar', 132, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'overridden_at', 'timestamptz', -1, TRUE, TRUE),
                    ('commercial_billing_candidate_overrides', 'created_at', 'timestamptz', -1, TRUE, TRUE)
            ) AS expected_column(
                relation_name,
                column_name,
                type_name,
                type_modifier,
                not_null,
                uses_type_default_collation
            )
        ),
        target_columns AS (
            SELECT relation_state.relname AS relation_name,
                   attribute_state.attname AS column_name,
                   type_state.typname AS type_name,
                   attribute_state.atttypmod AS type_modifier,
                   attribute_state.attnotnull,
                   attribute_state.attcollation = type_state.typcollation
                       AS uses_type_default_collation
            FROM target_relations AS relation_state
            JOIN pg_catalog.pg_attribute AS attribute_state
              ON attribute_state.attrelid = relation_state.oid
             AND attribute_state.attnum > 0
             AND NOT attribute_state.attisdropped
            JOIN pg_catalog.pg_type AS type_state
              ON type_state.oid = attribute_state.atttypid
        ),
        billing_catalog_relations AS (
            SELECT *
            FROM target_relations
            WHERE relname = ANY (
                ARRAY[
                    'commercial_billing_candidate_review_decisions',
                    'commercial_billing_candidate_overrides',
                    'commercial_billing_run_candidates'
                ]::text[]
            )
        ),
        unreviewed_columns AS (
            SELECT NOT EXISTS (
                SELECT 1
                FROM target_columns AS actual_column
                JOIN billing_catalog_relations AS billing_relation
                  ON billing_relation.relname = actual_column.relation_name
                LEFT JOIN required_columns AS expected_column
                  ON expected_column.relation_name = actual_column.relation_name
                 AND expected_column.column_name = actual_column.column_name
                WHERE expected_column.column_name IS NULL
            ) AS no_unreviewed_billing_columns
        ),
        unreviewed_billing_read_interceptors AS (
            SELECT NOT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_rewrite AS rule_state
                JOIN billing_catalog_relations AS relation_state
                  ON relation_state.oid = rule_state.ev_class
                WHERE rule_state.rulename <> '_RETURN'
                UNION ALL
                SELECT 1
                FROM billing_catalog_relations AS relation_state
                WHERE relation_state.relrowsecurity
                   OR relation_state.relforcerowsecurity
                UNION ALL
                -- A parent-table read includes traditional inheritance children,
                -- whose rows and mutation guards are outside this closed catalog.
                SELECT 1
                FROM pg_catalog.pg_inherits AS inheritance_state
                JOIN billing_catalog_relations AS relation_state
                  ON relation_state.oid = inheritance_state.inhparent
                UNION ALL
                SELECT 1
                FROM pg_catalog.pg_policy AS policy_state
                JOIN billing_catalog_relations AS relation_state
                  ON relation_state.oid = policy_state.polrelid
            ) AS no_unreviewed_billing_read_interceptors
        ),
        declared_constraints AS (
            SELECT *
            FROM (
                -- The long review-decision FK was auto-derived while preserving
                -- its `_fkey` suffix. Other auto names below are already
                -- physical; required_constraints projects explicit names to 63.
                VALUES
                    ('commercial_billing_run_candidates', 'commercial_billing_run_candidates_pkey', 'p', ARRAY['id']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_run_candidates', 'commercial_billing_run_candidates_billing_run_id_fkey', 'f', ARRAY['billing_run_id']::text[], 'commercial_billing_runs', ARRAY['id']::text[]),
                    ('commercial_billing_run_candidates', 'commercial_billing_run_candidates_source_fingerprint_check', 'c', ARRAY['source_fingerprint']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_run_candidates', 'commercial_billing_run_candidates_display_order_check', 'c', ARRAY['display_order']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_run_candidates', 'commercial_billing_run_candidates_run_key_key', 'u', ARRAY['billing_run_id', 'candidate_key']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_run_candidates', 'commercial_billing_run_candidates_run_order_key', 'u', ARRAY['billing_run_id', 'display_order']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_pkey', 'p', ARRAY['id']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisio_billing_run_id_fkey', 'f', ARRAY['billing_run_id']::text[], 'commercial_billing_runs', ARRAY['id']::text[]),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_fingerprint_check', 'c', ARRAY['source_fingerprint']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_revision_check', 'c', ARRAY['revision']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_decision_check', 'c', ARRAY['decision']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_reason_check', 'c', ARRAY['reason']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_request_fingerprint_check', 'c', ARRAY['request_fingerprint']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_actor_check', 'c', ARRAY['decided_by']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_source_key', 'u', ARRAY['source', 'idempotency_key']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_revision_key', 'u', ARRAY['candidate_key', 'source_fingerprint', 'revision']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_snapshot_fkey', 'f', ARRAY['billing_run_id', 'candidate_key', 'source_fingerprint']::text[], 'commercial_billing_run_candidates', ARRAY['billing_run_id', 'candidate_key', 'source_fingerprint']::text[]),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_review_fingerprint_check', 'c', ARRAY['review_fingerprint']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_pkey', 'p', ARRAY['id']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_billing_run_id_fkey', 'f', ARRAY['billing_run_id']::text[], 'commercial_billing_runs', ARRAY['id']::text[]),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_source_fingerprint_check', 'c', ARRAY['source_fingerprint']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_review_fingerprint_check', 'c', ARRAY['review_fingerprint']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_revision_check', 'c', ARRAY['revision']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_reason_code_check', 'c', ARRAY['reason_code']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_reason_check', 'c', ARRAY['reason']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_request_fingerprint_check', 'c', ARRAY['request_fingerprint']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_actor_check', 'c', ARRAY['overridden_by']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_source_key', 'u', ARRAY['source', 'idempotency_key']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_revision_key', 'u', ARRAY['billing_run_id', 'candidate_key', 'source_fingerprint', 'revision']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_review_key', 'u', ARRAY['review_fingerprint']::text[], NULL::text, ARRAY[]::text[]),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_snapshot_fkey', 'f', ARRAY['billing_run_id', 'candidate_key', 'source_fingerprint']::text[], 'commercial_billing_run_candidates', ARRAY['billing_run_id', 'candidate_key', 'source_fingerprint']::text[])
            ) AS expected_constraint(
                relation_name,
                constraint_name,
                constraint_type,
                key_columns,
                referenced_relation_name,
                referenced_columns
            )
        ),
        required_constraints AS (
            SELECT
                relation_name,
                pg_catalog.left(constraint_name, 63) AS constraint_name,
                constraint_type,
                key_columns,
                referenced_relation_name,
                referenced_columns
            FROM declared_constraints
        ),
        required_check_expressions AS (
            SELECT
                relation_name,
                pg_catalog.left(constraint_name, 63) AS constraint_name,
                normalized_expression
            FROM (
                VALUES
                    ('commercial_billing_run_candidates', 'commercial_billing_run_candidates_source_fingerprint_check', '((source_fingerprint)::text~''^[0-9a-f]{64}$''::text)'),
                    ('commercial_billing_run_candidates', 'commercial_billing_run_candidates_display_order_check', '(display_order>=0)'),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_fingerprint_check', '((source_fingerprint)::text~''^[0-9a-f]{64}$''::text)'),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_revision_check', '(revision>0)'),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_decision_check', '((decision)::text=any((array[''included''::charactervarying,''excluded''::charactervarying])::text[]))'),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_reason_check', '(length(btrim((reason)::text))>0)'),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_request_fingerprint_check', '((request_fingerprint)::text~''^[0-9a-f]{64}$''::text)'),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_actor_check', '(length(btrim((decided_by)::text))>0)'),
                    ('commercial_billing_candidate_review_decisions', 'commercial_billing_candidate_review_decisions_review_fingerprint_check', '((review_fingerprint)::text~''^[0-9a-f]{64}$''::text)'),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_source_fingerprint_check', '((source_fingerprint)::text~''^[0-9a-f]{64}$''::text)'),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_review_fingerprint_check', '((review_fingerprint)::text~''^[0-9a-f]{64}$''::text)'),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_revision_check', '(revision>0)'),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_reason_code_check', '((reason_code)::text=any((array[''one_time_service_variation''::charactervarying,''partial_or_missed_service''::charactervarying,''approved_pricing_exception''::charactervarying,''customer_credit''::charactervarying,''additional_charge''::charactervarying,''source_correction_pending''::charactervarying,''billing_delivery_exception''::charactervarying])::text[]))'),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_reason_check', '(length(btrim((reason)::text))>0)'),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_request_fingerprint_check', '((request_fingerprint)::text~''^[0-9a-f]{64}$''::text)'),
                    ('commercial_billing_candidate_overrides', 'commercial_billing_candidate_overrides_actor_check', '(length(btrim((overridden_by)::text))>0)')
            ) AS expected_check(relation_name, constraint_name, normalized_expression)
        ),
        target_constraints AS (
            SELECT
                relation_state.relname AS relation_name,
                constraint_state.conname AS constraint_name,
                constraint_state.contype::text AS constraint_type,
                ARRAY(
                    SELECT attribute_state.attname::text
                    FROM unnest(constraint_state.conkey)
                         WITH ORDINALITY AS key_state(attnum, ordinality)
                    JOIN pg_catalog.pg_attribute AS attribute_state
                      ON attribute_state.attrelid = constraint_state.conrelid
                     AND attribute_state.attnum = key_state.attnum
                    ORDER BY key_state.ordinality
                ) AS key_columns,
                referenced_relation.relname AS referenced_relation_name,
                referenced_schema.nspname AS referenced_schema_name,
                ARRAY(
                    SELECT attribute_state.attname::text
                    FROM unnest(constraint_state.confkey)
                         WITH ORDINALITY AS key_state(attnum, ordinality)
                    JOIN pg_catalog.pg_attribute AS attribute_state
                      ON attribute_state.attrelid = constraint_state.confrelid
                     AND attribute_state.attnum = key_state.attnum
                    ORDER BY key_state.ordinality
                ) AS referenced_columns,
                constraint_state.confdeltype::text AS confdeltype,
                constraint_state.confupdtype::text AS confupdtype,
                constraint_state.confmatchtype::text AS confmatchtype,
                constraint_state.condeferrable,
                constraint_state.condeferred,
                constraint_state.convalidated,
                CASE
                    WHEN constraint_state.contype = 'f' THEN COALESCE(
                        (
                            SELECT COUNT(*) = 4
                               AND bool_and(constraint_trigger.tgisinternal)
                               AND bool_and(
                                   constraint_trigger.tgenabled = 'O'::"char"
                               )
                            FROM pg_catalog.pg_trigger AS constraint_trigger
                            WHERE constraint_trigger.tgconstraint = constraint_state.oid
                        ),
                        FALSE
                    )
                    ELSE TRUE
                END AS foreign_key_enforcement_ready,
                CASE
                    WHEN constraint_state.contype = 'c' THEN regexp_replace(
                        lower(
                            pg_catalog.pg_get_expr(
                                constraint_state.conbin,
                                constraint_state.conrelid
                            )
                        ),
                        '\\s+',
                        '',
                        'g'
                    )
                END AS normalized_check_expression
            FROM pg_catalog.pg_constraint AS constraint_state
            JOIN billing_catalog_relations AS relation_state
              ON relation_state.oid = constraint_state.conrelid
            LEFT JOIN pg_catalog.pg_class AS referenced_relation
              ON referenced_relation.oid = constraint_state.confrelid
            LEFT JOIN pg_catalog.pg_namespace AS referenced_schema
              ON referenced_schema.oid = referenced_relation.relnamespace
        ),
        required_indexes AS (
            SELECT *
            FROM (
                VALUES
                    ('commercial_billing_run_candidates', 'idx_commercial_billing_run_candidates_run_order', FALSE, 2, 'usingbtree(billing_run_id,display_order)'),
                    ('commercial_billing_run_candidates', 'idx_commercial_billing_run_candidates_exact_source', TRUE, 3, 'usingbtree(billing_run_id,candidate_key,source_fingerprint)'),
                    ('commercial_billing_run_candidates', 'idx_commercial_billing_run_candidates_identity', FALSE, 2, 'usingbtree(candidate_key,source_fingerprint)'),
                    ('commercial_billing_candidate_review_decisions', 'idx_commercial_billing_candidate_review_decisions_run_candidate', FALSE, 4, 'usingbtree(billing_run_id,candidate_key,source_fingerprint,revisiondesc)'),
                    ('commercial_billing_candidate_review_decisions', 'idx_commercial_billing_candidate_review_decisions_review', FALSE, 4, 'usingbtree(candidate_key,source_fingerprint,review_fingerprint,revisiondesc)'),
                    ('commercial_billing_candidate_overrides', 'idx_commercial_billing_candidate_overrides_active', FALSE, 4, 'usingbtree(billing_run_id,candidate_key,source_fingerprint,revisiondesc)')
            ) AS expected_index(
                relation_name,
                index_name,
                is_unique,
                key_attribute_count,
                definition_fragment
            )
        ),
        target_indexes AS (
            SELECT
                relation_state.relname AS relation_name,
                index_relation.relname AS index_name,
                index_state.indisunique AS is_unique,
                index_state.indisvalid AS is_valid,
                index_state.indisready AS is_ready,
                index_state.indnkeyatts AS key_attribute_count,
                index_state.indnatts AS attribute_count,
                index_state.indpred IS NULL AS has_no_predicate,
                regexp_replace(
                    lower(pg_catalog.pg_get_indexdef(index_state.indexrelid)),
                    '\\s+',
                    '',
                    'g'
                ) AS normalized_definition,
                backing_constraint.conname AS backing_constraint_name
            FROM pg_catalog.pg_index AS index_state
            JOIN billing_catalog_relations AS relation_state
              ON relation_state.oid = index_state.indrelid
            JOIN pg_catalog.pg_class AS index_relation
              ON index_relation.oid = index_state.indexrelid
            LEFT JOIN pg_catalog.pg_constraint AS backing_constraint
              ON backing_constraint.conindid = index_state.indexrelid
             AND backing_constraint.conrelid = index_state.indrelid
        ),
        unreviewed_constraints AS (
            SELECT NOT EXISTS (
                SELECT 1
                FROM target_constraints AS actual_constraint
                LEFT JOIN required_constraints AS expected_constraint
                  ON expected_constraint.relation_name = actual_constraint.relation_name
                 AND expected_constraint.constraint_name = actual_constraint.constraint_name
                WHERE expected_constraint.constraint_name IS NULL
            ) AS no_unreviewed_billing_constraints
        ),
        unreviewed_indexes AS (
            SELECT NOT EXISTS (
                SELECT 1
                FROM target_indexes AS actual_index
                LEFT JOIN required_indexes AS expected_index
                  ON expected_index.relation_name = actual_index.relation_name
                 AND expected_index.index_name = actual_index.index_name
                LEFT JOIN required_constraints AS expected_backing_constraint
                  ON expected_backing_constraint.relation_name = actual_index.relation_name
                 AND expected_backing_constraint.constraint_name = actual_index.backing_constraint_name
                WHERE (
                    actual_index.backing_constraint_name IS NULL
                    AND expected_index.index_name IS NULL
                )
                   OR (
                    actual_index.backing_constraint_name IS NOT NULL
                    AND expected_backing_constraint.constraint_name IS NULL
                )
            ) AS no_unreviewed_billing_indexes
        ),
        target_functions AS (
            SELECT function_state.oid,
                   function_state.proname,
                   function_state.prosrc,
                   function_state.prokind,
                   function_state.provolatile,
                   function_state.proisstrict,
                   function_state.prosecdef,
                   function_state.proleakproof,
                   function_state.proparallel,
                   function_state.prosupport,
                   function_state.proconfig,
                   language_state.lanname AS language_name
            FROM pg_catalog.pg_proc AS function_state
            JOIN pg_catalog.pg_namespace AS namespace_state
              ON namespace_state.oid = function_state.pronamespace
            JOIN pg_catalog.pg_language AS language_state
              ON language_state.oid = function_state.prolang
            WHERE namespace_state.nspname = pg_catalog.current_schema()
              AND function_state.proname IN (
                  'default_commercial_billing_review_fingerprint',
                  'prevent_commercial_billing_invoice_for_excluded_candidate',
                  'prevent_commercial_billing_review_decision_mutation',
                  'prevent_commercial_billing_candidate_override_mutation'
              )
              AND function_state.pronargs = 0
              AND function_state.prorettype = 'trigger'::pg_catalog.regtype
        ),
        expected_invoice_fence_config AS (
            SELECT ARRAY[
                pg_catalog.format(
                    'search_path=pg_catalog, %I, pg_temp',
                    pg_catalog.current_schema()
                )
            ]::text[] AS function_proconfig
        ),
        reviewed_trigger_function_execution_metadata AS (
            SELECT
                COUNT(*) = 4
                AND NOT EXISTS (
                    SELECT 1
                    FROM target_functions AS function_state
                    WHERE function_state.prokind <> 'f'
                       OR function_state.language_name <> 'plpgsql'
                       OR function_state.provolatile <> 'v'
                       OR function_state.proisstrict
                       OR function_state.prosecdef
                       OR function_state.proleakproof
                       OR function_state.proparallel <> 'u'
                       OR function_state.prosupport IS DISTINCT FROM 0::pg_catalog.oid
                       OR (
                           function_state.proname =
                               'prevent_commercial_billing_invoice_for_excluded_candidate'
                           AND COALESCE(
                               function_state.proconfig,
                               ARRAY[]::text[]
                           ) <> ARRAY[]::text[]
                           AND COALESCE(
                               function_state.proconfig,
                               ARRAY[]::text[]
                           ) <> (
                               SELECT function_proconfig
                               FROM expected_invoice_fence_config
                           )
                       )
                       OR (
                           function_state.proname <>
                               'prevent_commercial_billing_invoice_for_excluded_candidate'
                           AND COALESCE(
                               function_state.proconfig,
                               ARRAY[]::text[]
                           ) <> ARRAY[]::text[]
                       )
                ) AS trigger_function_execution_metadata_ready
            FROM target_functions AS function_state
        ),
        target_function AS (
            SELECT function_state.oid,
                   function_state.prosrc,
                   function_state.proconfig
            FROM target_functions AS function_state
            WHERE function_state.proname =
                'prevent_commercial_billing_invoice_for_excluded_candidate'
            LIMIT 1
        ),
        invoice_fence_schema_binding AS (
            SELECT COALESCE(
                (
                    SELECT COALESCE(
                        function_state.proconfig,
                        ARRAY[]::text[]
                    ) = expected_config.function_proconfig
                    FROM target_function AS function_state
                    JOIN expected_invoice_fence_config AS expected_config ON TRUE
                ),
                FALSE
            ) AS invoice_fence_function_schema_binding_ready
        ),
        target_triggers AS (
            SELECT relation_state.relname AS relation_name,
                   trigger_state.tgname AS trigger_name,
                   function_state.proname AS function_name,
                   trigger_state.tgfoid,
                   trigger_state.tgtype,
                   trigger_state.tgenabled,
                   trigger_state.tgqual,
                   (trigger_state.tgtype::integer & 6) = 6
                       AS is_before_insert
            FROM pg_catalog.pg_trigger AS trigger_state
            JOIN target_relations AS relation_state
              ON relation_state.oid = trigger_state.tgrelid
            JOIN pg_catalog.pg_proc AS function_state
              ON function_state.oid = trigger_state.tgfoid
            WHERE NOT trigger_state.tgisinternal
        ),
        required_history_triggers AS (
            SELECT expected_trigger.relation_name,
                   expected_trigger.trigger_name,
                   expected_trigger.function_name,
                   expected_function.oid AS function_oid,
                   expected_trigger.trigger_type
            FROM (
                VALUES
                    (
                        'commercial_billing_candidate_review_decisions',
                        'trg_prevent_commercial_billing_review_decision_mutation',
                        'prevent_commercial_billing_review_decision_mutation',
                        27
                    ),
                    (
                        'commercial_billing_candidate_review_decisions',
                        'trg_prevent_commercial_billing_review_decision_truncate',
                        'prevent_commercial_billing_review_decision_mutation',
                        34
                    ),
                    (
                        'commercial_billing_candidate_overrides',
                        'trg_prevent_commercial_billing_candidate_override_mutation',
                        'prevent_commercial_billing_candidate_override_mutation',
                        27
                    ),
                    (
                        'commercial_billing_candidate_overrides',
                        'trg_prevent_commercial_billing_candidate_override_truncate',
                        'prevent_commercial_billing_candidate_override_mutation',
                        34
                    )
            ) AS expected_trigger(
                relation_name, trigger_name, function_name, trigger_type
            )
            LEFT JOIN target_functions AS expected_function
              ON expected_function.proname = expected_trigger.function_name
        ),
        required_billing_write_triggers AS (
            SELECT relation_name,
                   trigger_name,
                   function_name,
                   function_oid,
                   trigger_type
            FROM required_history_triggers
            UNION ALL
            SELECT
                'commercial_billing_candidate_review_decisions',
                'trg_default_commercial_billing_review_decision_fingerprint',
                'default_commercial_billing_review_fingerprint',
                expected_function.oid,
                7
            FROM target_functions AS expected_function
            WHERE expected_function.proname =
                'default_commercial_billing_review_fingerprint'
        ),
        unreviewed_billing_write_interceptors AS (
            SELECT NOT EXISTS (
                SELECT 1
                FROM target_triggers AS interceptor
                JOIN billing_catalog_relations AS billing_relation
                  ON billing_relation.relname = interceptor.relation_name
                LEFT JOIN required_billing_write_triggers AS expected_trigger
                  ON expected_trigger.relation_name = interceptor.relation_name
                 AND expected_trigger.trigger_name = interceptor.trigger_name
                 AND expected_trigger.function_name = interceptor.function_name
                 AND expected_trigger.function_oid = interceptor.tgfoid
                 AND expected_trigger.trigger_type = interceptor.tgtype
                 AND interceptor.tgenabled = 'O'
                 AND interceptor.tgqual IS NULL
                WHERE expected_trigger.trigger_name IS NULL
            ) AS no_unreviewed_billing_write_interceptors
        ),
        unreviewed_invoice_insert_interceptors AS (
            SELECT NOT EXISTS (
                SELECT 1
                FROM target_triggers AS interceptor
                WHERE interceptor.relation_name = 'invoices'
                  AND interceptor.is_before_insert
                  AND interceptor.trigger_name <>
                      'trg_prevent_commercial_billing_invoice_for_excluded_candidate'
            ) AS no_unreviewed_invoice_insert_interceptors
        ),
        unreviewed_invoice_rewrite_interceptors AS (
            SELECT NOT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_rewrite AS rule_state
                JOIN target_relations AS relation_state
                  ON relation_state.oid = rule_state.ev_class
                WHERE relation_state.relname = 'invoices'
                  AND rule_state.rulename <> '_RETURN'
            ) AS no_unreviewed_invoice_rewrite_interceptors
        )
        SELECT
            (
                SELECT COUNT(*) = 4
                FROM target_relations AS relation_state
                WHERE relation_state.relkind = 'r'
                  AND relation_state.relpersistence = 'p'
                  AND NOT relation_state.relispartition
            ) AS relations_ready,
            NOT EXISTS (
                SELECT 1
                FROM required_columns AS expected_column
                LEFT JOIN target_columns AS actual_column
                  ON actual_column.relation_name = expected_column.relation_name
                 AND actual_column.column_name = expected_column.column_name
                WHERE actual_column.relation_name IS NULL
                   OR actual_column.type_name <> expected_column.type_name
                   OR actual_column.type_modifier
                      IS DISTINCT FROM expected_column.type_modifier
                   OR actual_column.attnotnull
                      IS DISTINCT FROM expected_column.not_null
                   OR actual_column.uses_type_default_collation
                      IS DISTINCT FROM expected_column.uses_type_default_collation
            ) AS required_columns_ready,
            (
                SELECT no_unreviewed_billing_columns
                FROM unreviewed_columns
            ) AS no_unreviewed_billing_columns,
            (
                SELECT no_unreviewed_billing_read_interceptors
                FROM unreviewed_billing_read_interceptors
            ) AS no_unreviewed_billing_read_interceptors,
            (
                SELECT no_unreviewed_billing_write_interceptors
                FROM unreviewed_billing_write_interceptors
            ) AS no_unreviewed_billing_write_interceptors,
            (
                NOT EXISTS (
                    SELECT 1
                    FROM required_constraints AS expected_constraint
                    LEFT JOIN target_constraints AS actual_constraint
                      ON actual_constraint.relation_name = expected_constraint.relation_name
                     AND actual_constraint.constraint_name = expected_constraint.constraint_name
                    WHERE actual_constraint.constraint_name IS NULL
                       OR actual_constraint.constraint_type
                          IS DISTINCT FROM expected_constraint.constraint_type
                       OR actual_constraint.key_columns
                          IS DISTINCT FROM expected_constraint.key_columns
                       OR (
                           expected_constraint.referenced_relation_name IS NOT NULL
                           AND (
                               actual_constraint.referenced_relation_name
                                   IS DISTINCT FROM expected_constraint.referenced_relation_name
                               OR actual_constraint.referenced_schema_name
                                   IS DISTINCT FROM pg_catalog.current_schema()
                               OR actual_constraint.referenced_columns
                                   IS DISTINCT FROM expected_constraint.referenced_columns
                               OR actual_constraint.confdeltype IS DISTINCT FROM 'r'
                               OR actual_constraint.confupdtype IS DISTINCT FROM 'a'
                               OR actual_constraint.confmatchtype IS DISTINCT FROM 's'
                           )
                       )
                       OR actual_constraint.condeferrable
                       OR actual_constraint.condeferred
                       OR NOT actual_constraint.convalidated
                       OR NOT actual_constraint.foreign_key_enforcement_ready
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM required_check_expressions AS expected_check
                    LEFT JOIN target_constraints AS actual_constraint
                      ON actual_constraint.relation_name = expected_check.relation_name
                     AND actual_constraint.constraint_name = expected_check.constraint_name
                    WHERE actual_constraint.constraint_name IS NULL
                       OR actual_constraint.normalized_check_expression
                          IS DISTINCT FROM expected_check.normalized_expression
                )
            ) AS required_billing_constraints_ready,
            (
                SELECT no_unreviewed_billing_constraints
                FROM unreviewed_constraints
            ) AS no_unreviewed_billing_constraints,
            NOT EXISTS (
                SELECT 1
                FROM required_indexes AS expected_index
                LEFT JOIN target_indexes AS actual_index
                  ON actual_index.relation_name = expected_index.relation_name
                 AND actual_index.index_name = expected_index.index_name
                WHERE actual_index.index_name IS NULL
                   OR actual_index.backing_constraint_name IS NOT NULL
                   OR actual_index.is_unique IS DISTINCT FROM expected_index.is_unique
                   OR NOT actual_index.is_valid
                   OR NOT actual_index.is_ready
                   OR actual_index.key_attribute_count
                      IS DISTINCT FROM expected_index.key_attribute_count
                   OR actual_index.attribute_count
                      IS DISTINCT FROM expected_index.key_attribute_count
                   OR NOT actual_index.has_no_predicate
                   OR actual_index.normalized_definition NOT LIKE
                      '%' || expected_index.definition_fragment || '%'
            ) AS required_billing_indexes_ready,
            (
                SELECT no_unreviewed_billing_indexes
                FROM unreviewed_indexes
            ) AS no_unreviewed_billing_indexes,
            NOT EXISTS (
                SELECT 1
                FROM required_history_triggers AS expected_trigger
                LEFT JOIN target_triggers AS actual_trigger
                  ON actual_trigger.relation_name = expected_trigger.relation_name
                 AND actual_trigger.trigger_name = expected_trigger.trigger_name
                 AND actual_trigger.function_name = expected_trigger.function_name
                 AND actual_trigger.tgfoid = expected_trigger.function_oid
                 AND actual_trigger.tgtype = expected_trigger.trigger_type
                 AND actual_trigger.tgenabled = 'O'
                 AND actual_trigger.tgqual IS NULL
                WHERE actual_trigger.trigger_name IS NULL
            ) AS immutable_history_guards_ready,
            EXISTS (
                SELECT 1
                FROM target_triggers AS trigger_state
                JOIN target_functions AS function_state
                  ON trigger_state.tgfoid = function_state.oid
                WHERE trigger_state.relation_name =
                    'commercial_billing_candidate_review_decisions'
                  AND trigger_state.trigger_name =
                      'trg_default_commercial_billing_review_decision_fingerprint'
                  AND trigger_state.function_name =
                      'default_commercial_billing_review_fingerprint'
                  AND trigger_state.tgtype = 7
                  AND trigger_state.tgenabled = 'O'
                  AND trigger_state.tgqual IS NULL
            ) AS review_decision_default_trigger_ready,
            EXISTS (
                SELECT 1
                FROM target_triggers AS trigger_state
                JOIN target_function AS function_state
                  ON trigger_state.tgfoid = function_state.oid
                WHERE trigger_state.relation_name = 'invoices'
                  AND trigger_state.trigger_name =
                      'trg_prevent_commercial_billing_invoice_for_excluded_candidate'
                  AND trigger_state.function_name =
                      'prevent_commercial_billing_invoice_for_excluded_candidate'
                  AND trigger_state.tgtype = 7
                  AND trigger_state.tgenabled = 'O'
                  AND trigger_state.tgqual IS NULL
            ) AS invoice_fence_trigger_ready,
            (
                SELECT no_unreviewed_invoice_insert_interceptors
                FROM unreviewed_invoice_insert_interceptors
            ) AS no_unreviewed_invoice_insert_interceptors,
            (
                SELECT no_unreviewed_invoice_rewrite_interceptors
                FROM unreviewed_invoice_rewrite_interceptors
            ) AS no_unreviewed_invoice_rewrite_interceptors,
            (
                SELECT trigger_function_execution_metadata_ready
                FROM reviewed_trigger_function_execution_metadata
            ) AS trigger_function_execution_metadata_ready,
            (
                SELECT invoice_fence_function_schema_binding_ready
                FROM invoice_fence_schema_binding
            ) AS invoice_fence_function_schema_binding_ready,
            (
                SELECT function_state.prosrc
                FROM target_functions AS function_state
                WHERE function_state.proname =
                    'default_commercial_billing_review_fingerprint'
            ) AS review_decision_default_function_body,
            (
                SELECT function_state.prosrc
                FROM target_functions AS function_state
                WHERE function_state.proname =
                    'prevent_commercial_billing_review_decision_mutation'
            ) AS review_decision_history_guard_function_body,
            (
                SELECT function_state.prosrc
                FROM target_functions AS function_state
                WHERE function_state.proname =
                    'prevent_commercial_billing_candidate_override_mutation'
            ) AS override_history_guard_function_body,
            (SELECT function_state.prosrc FROM target_function AS function_state)
                AS function_body
        """
    )
    # asyncpg.Record is iterable but is not registered as collections.abc.Mapping.
    return {} if evidence_row is None else dict(evidence_row)


async def _attest_migration_379(
    executor: Any,
    migration_files: Collection[Path],
) -> MissingSourceForwardRecoveryMigrationReconciliationAttestation:
    """Classify only the exact missing-379 legacy fence state."""
    record = MIGRATION_379_COMMERCIAL_BILLING_RUN_FENCE_FORWARD_RECOVERY
    historical_ledger_rows = await executor.fetch(
        "SELECT version, content_sha256, applied_at FROM schema_migrations WHERE name = $1 LIMIT 2",
        record.migration_name,
    )
    successor_ledger_rows = await executor.fetch(
        "SELECT name, version, content_sha256, applied_at "
        "FROM schema_migrations WHERE name = ANY($1::text[]) ORDER BY name",
        [receipt.migration_name for receipt in record.successor_receipts],
    )
    recovery_ledger_rows = await executor.fetch(
        "SELECT version, content_sha256 FROM schema_migrations WHERE name = $1 LIMIT 2",
        record.recovery_migration_name,
    )
    schema_binding_ledger_rows = await executor.fetch(
        "SELECT version, content_sha256 FROM schema_migrations WHERE name = $1 LIMIT 2",
        record.schema_binding_migration_name,
    )
    exactly_one_historical_ledger_row = len(historical_ledger_rows) == 1
    historical_ledger_row = (
        historical_ledger_rows[0] if exactly_one_historical_ledger_row else None
    )
    exactly_one_recovery_ledger_row = len(recovery_ledger_rows) == 1
    recovery_ledger_row = (
        recovery_ledger_rows[0] if exactly_one_recovery_ledger_row else None
    )
    exactly_one_schema_binding_ledger_row = len(schema_binding_ledger_rows) == 1
    schema_binding_ledger_row = (
        schema_binding_ledger_rows[0]
        if exactly_one_schema_binding_ledger_row
        else None
    )
    expected_successors = {
        receipt.migration_name: receipt for receipt in record.successor_receipts
    }
    observed_successors = {
        str(row["name"]): row for row in successor_ledger_rows
    }
    successor_receipts_ready = (
        len(successor_ledger_rows) == len(expected_successors)
        and set(observed_successors) == set(expected_successors)
        and all(
            row["version"] == expected.migration_version
            and row["content_sha256"] is None
            and _normalize_utc(row["applied_at"]) == expected.observed_applied_at
            for name, expected in expected_successors.items()
            for row in (observed_successors[name],)
        )
    )
    catalog = await _migration_379_catalog_evidence(executor)
    function_body_sha256 = _catalog_function_body_sha256(
        catalog.get("function_body")
    )
    review_decision_default_function_body_ready = (
        _catalog_function_body_sha256(
            catalog.get("review_decision_default_function_body")
        ) == record.review_decision_default_function_body_sha256
    )
    history_guard_function_bodies_ready = all((
        _catalog_function_body_sha256(
            catalog.get("review_decision_history_guard_function_body")
        ) == record.review_decision_history_guard_function_body_sha256,
        _catalog_function_body_sha256(
            catalog.get("override_history_guard_function_body")
        ) == record.override_history_guard_function_body_sha256,
    ))
    historical_receipt_ready = all((
        exactly_one_historical_ledger_row,
        historical_ledger_row is not None
        and historical_ledger_row["version"] == record.historical_migration_version,
        historical_ledger_row is not None
        and historical_ledger_row["content_sha256"] == record.historical_ledger_sha256,
        historical_ledger_row is not None
        and _normalize_utc(historical_ledger_row["applied_at"])
        == record.observed_applied_at,
    ))
    recovery_source_ready = all((
        _packaged_migration_digest(migration_files, record.recovery_migration_name)
        == record.recovery_packaged_sha256,
        _packaged_migration_function_body_sha256(
            migration_files, record.recovery_migration_name
        )
        == record.recovered_function_body_template_sha256,
    ))
    recovery_receipt_ready = all((
        exactly_one_recovery_ledger_row,
        recovery_ledger_row is not None
        and recovery_ledger_row["version"] == record.recovery_migration_version,
        recovery_ledger_row is not None
        and recovery_ledger_row["content_sha256"] == record.recovery_packaged_sha256,
    ))
    schema_binding_source_ready = (
        _packaged_migration_digest(
            migration_files, record.schema_binding_migration_name
        )
        == record.schema_binding_packaged_sha256
    )
    schema_binding_receipt_ready = all((
        exactly_one_schema_binding_ledger_row,
        schema_binding_ledger_row is not None
        and schema_binding_ledger_row["version"]
        == record.schema_binding_migration_version,
        schema_binding_ledger_row is not None
        and schema_binding_ledger_row["content_sha256"]
        == record.schema_binding_packaged_sha256,
    ))

    return MissingSourceForwardRecoveryMigrationReconciliationAttestation(
        reconciliation_id=record.reconciliation_id,
        migration_name=record.migration_name,
        historical_receipt_ready=historical_receipt_ready,
        successor_receipts_ready=successor_receipts_ready,
        recovery_source_ready=recovery_source_ready,
        no_recovery_ledger_row=not recovery_ledger_rows,
        recovery_receipt_ready=recovery_receipt_ready,
        schema_binding_source_ready=schema_binding_source_ready,
        no_schema_binding_ledger_row=not schema_binding_ledger_rows,
        schema_binding_receipt_ready=schema_binding_receipt_ready,
        reviewed_billing_catalog_ready=all((
            bool(catalog.get("relations_ready")),
            bool(catalog.get("required_columns_ready")),
            bool(catalog.get("no_unreviewed_billing_columns")),
            bool(catalog.get("no_unreviewed_billing_read_interceptors")),
            bool(catalog.get("no_unreviewed_billing_write_interceptors")),
            bool(catalog.get("review_decision_default_trigger_ready")),
            review_decision_default_function_body_ready,
            bool(catalog.get("required_billing_constraints_ready")),
            bool(catalog.get("foreign_key_enforcement_ready")),
            bool(catalog.get("no_unreviewed_billing_constraints")),
            bool(catalog.get("required_billing_indexes_ready")),
            bool(catalog.get("no_unreviewed_billing_indexes")),
            bool(catalog.get("immutable_history_guards_ready")),
            history_guard_function_bodies_ready,
            bool(catalog.get("no_unreviewed_invoice_insert_interceptors")),
            bool(catalog.get("no_unreviewed_invoice_rewrite_interceptors")),
            bool(catalog.get("trigger_function_execution_metadata_ready")),
        )),
        required_billing_columns_ready=bool(
            catalog.get("required_columns_ready")
        ),
        no_unreviewed_billing_columns=bool(
            catalog.get("no_unreviewed_billing_columns")
        ),
        no_unreviewed_billing_read_interceptors=bool(
            catalog.get("no_unreviewed_billing_read_interceptors")
        ),
        no_unreviewed_billing_write_interceptors=bool(
            catalog.get("no_unreviewed_billing_write_interceptors")
        ),
        review_decision_default_trigger_ready=bool(
            catalog.get("review_decision_default_trigger_ready")
        ),
        review_decision_default_function_body_ready=(
            review_decision_default_function_body_ready
        ),
        history_guard_function_bodies_ready=history_guard_function_bodies_ready,
        required_billing_constraints_ready=bool(
            catalog.get("required_billing_constraints_ready")
        ),
        foreign_key_enforcement_ready=bool(
            catalog.get("foreign_key_enforcement_ready")
        ),
        no_unreviewed_billing_constraints=bool(
            catalog.get("no_unreviewed_billing_constraints")
        ),
        required_billing_indexes_ready=bool(
            catalog.get("required_billing_indexes_ready")
        ),
        no_unreviewed_billing_indexes=bool(
            catalog.get("no_unreviewed_billing_indexes")
        ),
        invoice_fence_trigger_ready=bool(
            catalog.get("invoice_fence_trigger_ready")
        ),
        no_unreviewed_invoice_insert_interceptors=bool(
            catalog.get("no_unreviewed_invoice_insert_interceptors")
        ),
        no_unreviewed_invoice_rewrite_interceptors=bool(
            catalog.get("no_unreviewed_invoice_rewrite_interceptors")
        ),
        trigger_function_execution_metadata_ready=bool(
            catalog.get("trigger_function_execution_metadata_ready")
        ),
        invoice_fence_function_schema_binding_ready=bool(
            catalog.get("invoice_fence_function_schema_binding_ready")
        ),
        legacy_function_body_matches=(
            function_body_sha256 == record.legacy_function_body_sha256
        ),
        recovered_function_body_matches=(
            function_body_sha256 == record.recovered_function_body_template_sha256
        ),
    )


_MIGRATION_379_ATOMIC_FUNCTION_NAMES = (
    "default_commercial_billing_review_fingerprint",
    "prevent_commercial_billing_candidate_override_mutation",
    "prevent_commercial_billing_invoice_for_excluded_candidate",
    "prevent_commercial_billing_review_decision_mutation",
)


async def _migration_379_atomic_function_definitions(
    executor: Any,
    *,
    expected_invoice_fence_body_sha256: str,
) -> tuple[str, ...]:
    """Return only owner-replayable, catalog-attested trigger definitions.

    PostgreSQL deliberately does not grant a normal application/migration role
    write-class ``LOCK TABLE`` access to ``pg_catalog.pg_proc``. Replaying the
    exact, already-validated definition instead takes the function row lock
    that conflicts with ``CREATE OR REPLACE FUNCTION`` and ``ALTER FUNCTION``.
    The definition is read and validated before replay, so an untrusted catalog
    change can never be executed as migration SQL. If concurrent function DDL
    commits after this read but before the replay, PostgreSQL rejects the stale
    catalog tuple update; the surrounding atomic recovery then rolls back with
    no migration receipt instead of overwriting that newer definition.
    """

    rows = [
        dict(row)
        for row in await executor.fetch(
            """
            WITH expected_invoice_fence_config AS (
                SELECT ARRAY[
                    pg_catalog.format(
                        'search_path=pg_catalog, %I, pg_temp',
                        pg_catalog.current_schema()
                    )
                ]::text[] AS function_proconfig
            )
            SELECT function_state.proname AS function_name,
                   function_state.prosrc AS function_body,
                   pg_catalog.pg_get_functiondef(function_state.oid)
                       AS function_definition,
                   pg_catalog.pg_has_role(
                       CURRENT_USER,
                       function_state.proowner,
                       'MEMBER'
                   ) AS current_role_can_replace
            FROM pg_catalog.pg_proc AS function_state
            JOIN pg_catalog.pg_namespace AS namespace_state
              ON namespace_state.oid = function_state.pronamespace
            JOIN pg_catalog.pg_language AS language_state
              ON language_state.oid = function_state.prolang
            WHERE namespace_state.nspname = pg_catalog.current_schema()
              AND function_state.proname IN (
                  'default_commercial_billing_review_fingerprint',
                  'prevent_commercial_billing_invoice_for_excluded_candidate',
                  'prevent_commercial_billing_review_decision_mutation',
                  'prevent_commercial_billing_candidate_override_mutation'
              )
              AND function_state.pronargs = 0
              AND function_state.prorettype = 'trigger'::pg_catalog.regtype
              AND function_state.prokind = 'f'
              AND language_state.lanname = 'plpgsql'
              AND function_state.provolatile = 'v'
              AND NOT function_state.proisstrict
              AND NOT function_state.prosecdef
              AND NOT function_state.proleakproof
              AND function_state.proparallel = 'u'
              AND function_state.prosupport IS NOT DISTINCT FROM 0::pg_catalog.oid
              AND (
                  (
                      function_state.proname =
                          'prevent_commercial_billing_invoice_for_excluded_candidate'
                      AND (
                          COALESCE(function_state.proconfig, ARRAY[]::text[])
                              = ARRAY[]::text[]
                          OR COALESCE(function_state.proconfig, ARRAY[]::text[])
                              = (
                                  SELECT function_proconfig
                                  FROM expected_invoice_fence_config
                              )
                      )
                  )
                  OR (
                      function_state.proname <>
                          'prevent_commercial_billing_invoice_for_excluded_candidate'
                      AND COALESCE(function_state.proconfig, ARRAY[]::text[])
                          = ARRAY[]::text[]
                  )
              )
            ORDER BY function_state.proname
            """
        )
    ]
    observed_names = tuple(row["function_name"] for row in rows)
    if observed_names != _MIGRATION_379_ATOMIC_FUNCTION_NAMES:
        raise HistoricalForwardRecoveryAtomicPreflightError(
            "the exact owner-replayable 379 trigger-function set was not present"
        )

    record = MIGRATION_379_COMMERCIAL_BILLING_RUN_FENCE_FORWARD_RECOVERY
    expected_body_sha256 = {
        "default_commercial_billing_review_fingerprint": (
            record.review_decision_default_function_body_sha256
        ),
        "prevent_commercial_billing_candidate_override_mutation": (
            record.override_history_guard_function_body_sha256
        ),
        "prevent_commercial_billing_invoice_for_excluded_candidate": (
            expected_invoice_fence_body_sha256
        ),
        "prevent_commercial_billing_review_decision_mutation": (
            record.review_decision_history_guard_function_body_sha256
        ),
    }
    function_definitions: list[str] = []
    for row in rows:
        function_name = row["function_name"]
        if not row["current_role_can_replace"]:
            raise HistoricalForwardRecoveryAtomicPreflightError(
                "the migration role cannot replay the attested 379 trigger functions"
            )
        if (
            _catalog_function_body_sha256(row["function_body"])
            != expected_body_sha256[function_name]
        ):
            raise HistoricalForwardRecoveryAtomicPreflightError(
                "the 379 trigger-function body changed before its atomic receipt"
            )
        function_definition = row["function_definition"]
        if not isinstance(function_definition, str) or not function_definition.strip():
            raise HistoricalForwardRecoveryAtomicPreflightError(
                "the 379 trigger-function definition could not be safely replayed"
            )
        function_definitions.append(function_definition)
    return tuple(function_definitions)


async def reattest_historical_forward_recovery_in_atomic_transaction(
    executor: Any,
    *,
    migration_name: str,
    migration_files: Collection[Path],
) -> None:
    """Close the selected 391/392 catalog race inside its receipt transaction.

    ``run_migrations`` holds the process-wide migration advisory lock, but that
    lock intentionally does not govern a separate session's direct catalog DDL.
    Relation locks stabilize tables, constraints, indexes, rules, triggers, and
    receipts. Exact owner-replayed trigger definitions stabilize the associated
    ``pg_proc`` rows without requiring a superuser-only system-catalog lock.
    The canonical 379 predicate is then re-attested before the selected SQL or
    its irreversible receipt may run.
    """

    record = MIGRATION_379_COMMERCIAL_BILLING_RUN_FENCE_FORWARD_RECOVERY
    expected_recovery_state = {
        record.recovery_migration_name: (
            "recovery_required",
            record.legacy_function_body_sha256,
        ),
        record.schema_binding_migration_name: (
            "schema_binding_required",
            record.recovered_function_body_template_sha256,
        ),
    }.get(migration_name)
    if expected_recovery_state is None:
        return
    expected_status, expected_invoice_fence_body_sha256 = expected_recovery_state

    await executor.execute(
        """
        DO $migration_379_catalog_lock$
        DECLARE
            schema_name TEXT := pg_catalog.current_schema();
        BEGIN
            IF schema_name IS NULL THEN
                RAISE EXCEPTION
                    'Cannot re-attest commercial billing fence without an active schema';
            END IF;

            -- SHARE ROW EXCLUSIVE excludes concurrent relation DDL and writes
            -- through the final receipt while remaining self-compatible with
            -- this atomic recovery's own function/trigger/ledger changes.
            EXECUTE pg_catalog.format(
                'LOCK TABLE '
                || '%1$I.schema_migrations, '
                || '%1$I.commercial_billing_candidate_overrides, '
                || '%1$I.commercial_billing_candidate_review_decisions, '
                || '%1$I.commercial_billing_run_candidates, '
                || '%1$I.invoices '
                || 'IN SHARE ROW EXCLUSIVE MODE',
                schema_name
            );
        END;
        $migration_379_catalog_lock$;
        """
    )

    function_definitions = await _migration_379_atomic_function_definitions(
        executor,
        expected_invoice_fence_body_sha256=expected_invoice_fence_body_sha256,
    )
    for function_definition in function_definitions:
        await executor.execute(function_definition)

    attestation = await _attest_migration_379(executor, migration_files)
    if attestation.status != expected_status:
        raise HistoricalForwardRecoveryAtomicPreflightError(
            "the exact 379 recovery state was not present after acquiring the "
            f"atomic catalog locks (expected={expected_status}, "
            f"observed={attestation.status})"
        )

    if migration_name == record.schema_binding_migration_name:
        await executor.execute(
            "SELECT pg_catalog.set_config("
            "'atlas.migration_379_catalog_attestation_schema', "
            "pg_catalog.current_schema(), TRUE)"
        )


async def _migration_386_catalog_evidence(executor: Any) -> Mapping[str, object]:
    """Read only the named function and trigger metadata in the active schema."""
    evidence_row = await executor.fetchrow(
        """
        WITH target_relation AS (
            SELECT relation_state.oid
            FROM pg_catalog.pg_class AS relation_state
            JOIN pg_catalog.pg_namespace AS namespace_state
              ON namespace_state.oid = relation_state.relnamespace
            WHERE namespace_state.nspname = pg_catalog.current_schema()
              AND relation_state.relname = 'contacts'
              AND relation_state.relkind = 'r'
              AND NOT relation_state.relispartition
            LIMIT 1
        ),
        target_lifecycle_relation AS (
            SELECT relation_state.oid
            FROM pg_catalog.pg_class AS relation_state
            JOIN pg_catalog.pg_namespace AS namespace_state
              ON namespace_state.oid = relation_state.relnamespace
            WHERE namespace_state.nspname = pg_catalog.current_schema()
              AND relation_state.relname = 'eom_lead_lifecycle_events'
              AND relation_state.relkind = 'r'
              AND NOT relation_state.relispartition
            LIMIT 1
        ),
        target_function AS (
            SELECT
                function_state.oid,
                function_state.prosrc,
                function_state.prosecdef,
                function_state.proconfig,
                function_state.proacl,
                function_state.proowner
            FROM pg_catalog.pg_proc AS function_state
            JOIN pg_catalog.pg_namespace AS namespace_state
              ON namespace_state.oid = function_state.pronamespace
            WHERE namespace_state.nspname = pg_catalog.current_schema()
              AND function_state.proname = 'reject_nocodb_eom_won_loss_mutation'
              AND function_state.pronargs = 0
              AND function_state.prorettype = 'trigger'::pg_catalog.regtype
            LIMIT 1
        ),
        target_guard_role AS (
            SELECT
                guard_role.oid,
                NOT guard_role.rolcanlogin
                AND NOT guard_role.rolinherit
                AND NOT guard_role.rolsuper
                AND NOT guard_role.rolcreaterole
                AND NOT guard_role.rolcreatedb
                AND NOT guard_role.rolreplication
                AND NOT guard_role.rolbypassrls
                AND pg_catalog.has_schema_privilege(
                    guard_role.oid,
                    pg_catalog.current_schema(),
                    'USAGE'
                )
                AND pg_catalog.has_schema_privilege(
                    guard_role.oid,
                    pg_catalog.current_schema(),
                    'CREATE'
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_roles AS member_role
                    WHERE member_role.rolcanlogin
                      AND NOT member_role.rolsuper
                      AND pg_catalog.pg_has_role(
                          member_role.oid,
                          guard_role.oid,
                          'MEMBER'
                      )
                ) AS trusted_guard_role_ready
            FROM pg_catalog.pg_roles AS guard_role
            WHERE guard_role.rolname = 'atlas_eom_handoff_owner'
            LIMIT 1
        ),
        target_trigger AS (
            SELECT
                trigger_state.tgtype,
                trigger_state.tgenabled,
                trigger_state.tgattr,
                trigger_state.tgqual
            FROM pg_catalog.pg_trigger AS trigger_state
            JOIN target_relation AS relation_state
              ON relation_state.oid = trigger_state.tgrelid
            JOIN target_function AS function_state
              ON function_state.oid = trigger_state.tgfoid
            WHERE trigger_state.tgname = 'trg_reject_nocodb_eom_won_loss_mutation'
              AND NOT trigger_state.tgisinternal
            LIMIT 1
        )
        SELECT
            pg_catalog.current_schema()::text AS schema_name,
            EXISTS (SELECT 1 FROM target_relation) AS contacts_relation_ready,
            EXISTS (SELECT 1 FROM target_function) AS function_ready,
            COALESCE(
                (SELECT function_state.prosecdef FROM target_function AS function_state),
                FALSE
            ) AS function_security_definer,
            COALESCE(
                (SELECT function_state.proconfig FROM target_function AS function_state),
                ARRAY[]::text[]
            ) AS function_proconfig,
            COALESCE(
                (
                    SELECT NOT EXISTS (
                        SELECT 1
                        FROM pg_catalog.aclexplode(
                            COALESCE(
                                function_state.proacl,
                                pg_catalog.acldefault('f', function_state.proowner)
                            )
                        ) AS privilege_state
                        WHERE privilege_state.grantee = 0
                          AND privilege_state.privilege_type = 'EXECUTE'
                    )
                    FROM target_function AS function_state
                ),
                FALSE
            ) AS function_public_execute_revoked,
            COALESCE(
                (
                    SELECT guard_role.trusted_guard_role_ready
                    FROM target_guard_role AS guard_role
                ),
                FALSE
            ) AS trusted_guard_role_ready,
            COALESCE(
                (
                    SELECT function_state.proowner = guard_role.oid
                    FROM target_function AS function_state
                    JOIN target_guard_role AS guard_role ON TRUE
                ),
                FALSE
            ) AS recovered_function_guard_owner_ready,
            COALESCE(
                (
                    SELECT pg_catalog.has_table_privilege(
                        guard_role.oid,
                        lifecycle_relation.oid,
                        'SELECT'
                    )
                    FROM target_guard_role AS guard_role
                    JOIN target_lifecycle_relation AS lifecycle_relation ON TRUE
                ),
                FALSE
            ) AS recovered_function_guard_lifecycle_read_ready,
            (SELECT function_state.prosrc FROM target_function AS function_state)
                AS function_body,
            EXISTS (SELECT 1 FROM target_trigger) AS trigger_ready,
            COALESCE(
                (SELECT trigger_state.tgenabled::text FROM target_trigger AS trigger_state),
                ''
            ) AS trigger_enabled,
            COALESCE(
                (SELECT trigger_state.tgtype::integer FROM target_trigger AS trigger_state),
                0
            ) = 27 AS trigger_is_before_row_update_delete,
            COALESCE(
                (
                    SELECT trigger_state.tgqual IS NULL
                    FROM target_trigger AS trigger_state
                ),
                FALSE
            ) AS trigger_has_no_when_clause,
            COALESCE(
                (
                    SELECT array_agg(attribute_state.attname::text ORDER BY attribute_state.attnum)
                    FROM target_trigger AS trigger_state
                    JOIN target_relation AS relation_state ON TRUE
                    JOIN LATERAL unnest(trigger_state.tgattr::smallint[])
                        AS updated_attribute(attnum) ON TRUE
                    JOIN pg_catalog.pg_attribute AS attribute_state
                      ON attribute_state.attrelid = relation_state.oid
                     AND attribute_state.attnum = updated_attribute.attnum
                ),
                ARRAY[]::text[]
            ) AS trigger_update_columns
        """
    )
    # asyncpg.Record is iterable but is not registered as collections.abc.Mapping.
    # Convert it explicitly rather than letting the generic JSON-map helper erase
    # valid catalog evidence and turn an exact target into a false negative.
    return {} if evidence_row is None else dict(evidence_row)


async def _attest_migration_386(
    executor: Any,
    migration_files: Collection[Path],
) -> ForwardRecoveryMigrationReconciliationAttestation:
    """Classify the exact weak 386 target without declaring it safe to admit."""
    record = MIGRATION_386_WON_LOSS_FENCE_FORWARD_RECOVERY
    ledger_rows = await executor.fetch(
        "SELECT version, content_sha256, applied_at FROM schema_migrations WHERE name = $1 LIMIT 2",
        record.migration_name,
    )
    recovery_ledger_rows = await executor.fetch(
        "SELECT version, content_sha256 FROM schema_migrations WHERE name = $1 LIMIT 2",
        record.recovery_migration_name,
    )
    exactly_one_ledger_row = len(ledger_rows) == 1
    ledger_row = ledger_rows[0] if exactly_one_ledger_row else None
    exactly_one_recovery_ledger_row = len(recovery_ledger_rows) == 1
    recovery_ledger_row = (
        recovery_ledger_rows[0] if exactly_one_recovery_ledger_row else None
    )
    catalog = await _migration_386_catalog_evidence(executor)
    schema_name = str(catalog.get("schema_name") or "")
    function_body_sha256 = _catalog_function_body_sha256(
        catalog.get("function_body")
    )

    historical_receipt_ready = all((
        exactly_one_ledger_row,
        ledger_row is not None
        and ledger_row["version"] == record.historical_migration_version,
        ledger_row is not None
        and ledger_row["content_sha256"] == record.historical_ledger_sha256,
        ledger_row is not None
        and _normalize_utc(ledger_row["applied_at"]) == record.observed_applied_at,
        _packaged_migration_digest(migration_files, record.migration_name)
        == record.final_packaged_sha256,
    ))
    recovery_source_ready = all((
        _packaged_migration_digest(migration_files, record.recovery_migration_name)
        == record.recovery_packaged_sha256,
        _packaged_migration_function_body_sha256(
            migration_files, record.recovery_migration_name
        )
        == record.recovered_function_body_template_sha256,
    ))
    function_proconfig = _catalog_text_values(catalog.get("function_proconfig"))
    legacy_function_search_path_ready = function_proconfig == (
        f"search_path=pg_catalog, {schema_name}",
    )
    recovered_function_search_path_ready = function_proconfig == (
        f"search_path=pg_catalog, {schema_name}, pg_temp",
    )
    shared_catalog_ready = all((
        bool(catalog.get("contacts_relation_ready")),
        bool(catalog.get("function_ready")),
        bool(catalog.get("function_security_definer")),
        bool(catalog.get("function_public_execute_revoked")),
        bool(catalog.get("trigger_ready")),
        _catalog_char(catalog.get("trigger_enabled")) == "O",
        bool(catalog.get("trigger_is_before_row_update_delete")),
        bool(catalog.get("trigger_has_no_when_clause")),
    ))
    trusted_guard_role_ready = bool(catalog.get("trusted_guard_role_ready"))
    recovered_function_guard_owner_ready = bool(
        catalog.get("recovered_function_guard_owner_ready")
    )
    recovered_function_guard_lifecycle_read_ready = bool(
        catalog.get("recovered_function_guard_lifecycle_read_ready")
    )
    trigger_update_columns = _catalog_column_names(
        catalog.get("trigger_update_columns")
    )
    recovered_trigger_columns_ready = (
        len(trigger_update_columns) == len(_RECOVERED_386_TRIGGER_UPDATE_COLUMNS)
        and frozenset(trigger_update_columns) == _RECOVERED_386_TRIGGER_UPDATE_COLUMNS
    )
    recovered_function_body_sha256 = (
        _rendered_packaged_migration_function_body_sha256(
            migration_files,
            record.recovery_migration_name,
            schema_name=schema_name,
        )
    )
    recovery_receipt_ready = all((
        exactly_one_recovery_ledger_row,
        recovery_ledger_row is not None
        and recovery_ledger_row["version"] == record.recovery_migration_version,
        recovery_ledger_row is not None
        and recovery_ledger_row["content_sha256"] == record.recovery_packaged_sha256,
    ))

    return ForwardRecoveryMigrationReconciliationAttestation(
        reconciliation_id=record.reconciliation_id,
        migration_name=record.migration_name,
        historical_receipt_ready=historical_receipt_ready,
        recovery_source_ready=recovery_source_ready,
        no_recovery_ledger_row=not recovery_ledger_rows,
        recovery_receipt_ready=recovery_receipt_ready,
        trusted_guard_role_ready=trusted_guard_role_ready,
        recovered_function_guard_owner_ready=(
            recovered_function_guard_owner_ready
        ),
        recovered_function_guard_lifecycle_read_ready=(
            recovered_function_guard_lifecycle_read_ready
        ),
        legacy_catalog_ready=(
            shared_catalog_ready
            and legacy_function_search_path_ready
            and trusted_guard_role_ready
            and function_body_sha256 == record.legacy_function_body_sha256
            and trigger_update_columns == ("status",)
        ),
        recovered_catalog_ready=(
            shared_catalog_ready
            and recovered_function_search_path_ready
            and trusted_guard_role_ready
            and recovered_function_guard_owner_ready
            and recovered_function_guard_lifecycle_read_ready
            and function_body_sha256 == recovered_function_body_sha256
            and recovered_trigger_columns_ready
        ),
    )


async def pending_historical_forward_recovery_migration(
    executor: Any,
    migration_files: Collection[Path],
    *,
    unresolved_mismatched: Collection[str],
    unresolved_missing_source: Collection[str],
    pending_migration_names: Collection[str],
) -> str | None:
    """Return the sole permitted prelude for an exact named recovery state.

    The caller provides the already-unresolved report names and its selected
    pending names. This keeps recovery source-controlled and prevents an
    `only=` caller from accidentally running an unrequested prerequisite.
    """
    mismatched_names = frozenset(unresolved_mismatched)
    missing_source_names = frozenset(unresolved_missing_source)
    pending_names = frozenset(pending_migration_names)
    commercial_record = MIGRATION_379_COMMERCIAL_BILLING_RUN_FENCE_FORWARD_RECOVERY
    won_loss_record = MIGRATION_386_WON_LOSS_FENCE_FORWARD_RECOVERY

    # The target that needs 391 also retains the independently-reviewed weak
    # 386 fence. Apply only 391 first, then force a fresh runner invocation so
    # each recovery has a committed receipt and an observable re-attestation.
    if commercial_record.migration_name in missing_source_names:
        if missing_source_names != {commercial_record.migration_name}:
            return None
        if mismatched_names - {won_loss_record.migration_name}:
            return None
        commercial_attestation = await _attest_migration_379(
            executor, migration_files
        )
        recovery_name = {
            "recovery_required": commercial_record.recovery_migration_name,
            "schema_binding_required": (
                commercial_record.schema_binding_migration_name
            ),
        }.get(commercial_attestation.status)
        if recovery_name is None or recovery_name not in pending_names:
            return None
        if mismatched_names:
            won_loss_attestation = await _attest_migration_386(
                executor, migration_files
            )
            if won_loss_attestation.status != "recovery_required":
                return None
        return recovery_name

    if mismatched_names != {won_loss_record.migration_name}:
        return None
    if missing_source_names:
        return None
    if won_loss_record.recovery_migration_name not in pending_names:
        return None

    attestation = await _attest_migration_386(executor, migration_files)
    if attestation.status != "recovery_required":
        return None
    return won_loss_record.recovery_migration_name


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


async def _attest_migration_067(
    executor: Any,
) -> B2BCampaignPartnerMissingSourceMigrationReconciliationAttestation:
    """Attest only the named 067 receipt without re-creating its source."""
    record = MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION
    ledger_rows = await executor.fetch(
        "SELECT version, content_sha256, applied_at FROM schema_migrations "
        "WHERE name = $1 LIMIT 2",
        record.migration_name,
    )
    exactly_one_ledger_row = len(ledger_rows) == 1
    ledger_row = ledger_rows[0] if exactly_one_ledger_row else None
    recorded_digest = ledger_row["content_sha256"] if ledger_row is not None else None
    applied_at = _normalize_utc(
        ledger_row["applied_at"] if ledger_row is not None else None
    )
    (
        b2b_campaigns_is_ordinary_table,
        partner_id_column_ready,
        partner_foreign_key_ready,
        partner_partial_index_ready,
    ) = await _migration_067_catalog_evidence(executor)

    return B2BCampaignPartnerMissingSourceMigrationReconciliationAttestation(
        reconciliation_id=record.reconciliation_id,
        migration_name=record.migration_name,
        exactly_one_ledger_row=exactly_one_ledger_row,
        ledger_version_matches_record=(
            exactly_one_ledger_row
            and ledger_row["version"] == record.migration_version
        ),
        ledger_digest_is_null=(
            exactly_one_ledger_row
            and recorded_digest == record.historical_ledger_sha256
        ),
        applied_at_matches_record=applied_at == record.observed_applied_at,
        b2b_campaigns_is_ordinary_table=b2b_campaigns_is_ordinary_table,
        partner_id_column_ready=partner_id_column_ready,
        partner_foreign_key_ready=partner_foreign_key_ready,
        partner_partial_index_ready=partner_partial_index_ready,
    )


async def _attest_migration_297(
    executor: Any,
) -> B2BCompanySignalPromotionMissingSourceMigrationReconciliationAttestation:
    """Attest only the named 297 receipt without reconstructing its source."""
    record = MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION
    ledger_rows = await executor.fetch(
        "SELECT version, content_sha256, applied_at FROM schema_migrations "
        "WHERE name = $1 LIMIT 2",
        record.migration_name,
    )
    exactly_one_ledger_row = len(ledger_rows) == 1
    ledger_row = ledger_rows[0] if exactly_one_ledger_row else None
    recorded_digest = ledger_row["content_sha256"] if ledger_row is not None else None
    applied_at = _normalize_utc(
        ledger_row["applied_at"] if ledger_row is not None else None
    )
    (
        b2b_company_signals_is_ordinary_table,
        canonical_promotion_type_column_ready,
        canonical_promotion_type_has_no_constraints,
        canonical_promotion_type_partial_index_ready,
    ) = await _migration_297_catalog_evidence(executor)

    return B2BCompanySignalPromotionMissingSourceMigrationReconciliationAttestation(
        reconciliation_id=record.reconciliation_id,
        migration_name=record.migration_name,
        exactly_one_ledger_row=exactly_one_ledger_row,
        ledger_version_matches_record=(
            exactly_one_ledger_row
            and ledger_row["version"] == record.migration_version
        ),
        ledger_digest_is_null=(
            exactly_one_ledger_row
            and recorded_digest == record.historical_ledger_sha256
        ),
        applied_at_matches_record=applied_at == record.observed_applied_at,
        b2b_company_signals_is_ordinary_table=(
            b2b_company_signals_is_ordinary_table
        ),
        canonical_promotion_type_column_ready=(
            canonical_promotion_type_column_ready
        ),
        canonical_promotion_type_has_no_constraints=(
            canonical_promotion_type_has_no_constraints
        ),
        canonical_promotion_type_partial_index_ready=(
            canonical_promotion_type_partial_index_ready
        ),
    )


async def _attest_migration_272(
    executor: Any,
) -> B2BWatchlistAlertEventsMissingSourceMigrationReconciliationAttestation:
    """Attest only the named synthetic-version alert-event receipt."""
    record = MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION
    ledger_rows = await executor.fetch(
        "SELECT version, content_sha256, applied_at FROM schema_migrations "
        "WHERE name = $1 LIMIT 2",
        record.migration_name,
    )
    exactly_one_ledger_row = len(ledger_rows) == 1
    ledger_row = ledger_rows[0] if exactly_one_ledger_row else None
    recorded_digest = ledger_row["content_sha256"] if ledger_row is not None else None
    applied_at = _normalize_utc(
        ledger_row["applied_at"] if ledger_row is not None else None
    )
    (
        watchlist_alert_events_is_ordinary_table,
        watchlist_alert_events_has_permanent_storage,
        base_alert_event_columns_ready,
        known_later_alert_event_columns_ready,
        no_unlisted_alert_event_columns,
        required_alert_event_constraints_ready,
        no_unlisted_alert_event_constraints,
        required_alert_event_indexes_ready,
        no_unlisted_alert_event_indexes,
        no_unreviewed_alert_event_write_interceptors,
    ) = await _migration_272_catalog_evidence(executor)

    return B2BWatchlistAlertEventsMissingSourceMigrationReconciliationAttestation(
        reconciliation_id=record.reconciliation_id,
        migration_name=record.migration_name,
        exactly_one_ledger_row=exactly_one_ledger_row,
        ledger_version_matches_record=(
            exactly_one_ledger_row
            and ledger_row["version"] == record.migration_version
        ),
        ledger_digest_is_null=(
            exactly_one_ledger_row
            and recorded_digest == record.historical_ledger_sha256
        ),
        applied_at_matches_record=applied_at == record.observed_applied_at,
        watchlist_alert_events_is_ordinary_table=(
            watchlist_alert_events_is_ordinary_table
        ),
        watchlist_alert_events_has_permanent_storage=(
            watchlist_alert_events_has_permanent_storage
        ),
        base_alert_event_columns_ready=base_alert_event_columns_ready,
        known_later_alert_event_columns_ready=(
            known_later_alert_event_columns_ready
        ),
        no_unlisted_alert_event_columns=no_unlisted_alert_event_columns,
        required_alert_event_constraints_ready=(
            required_alert_event_constraints_ready
        ),
        no_unlisted_alert_event_constraints=(
            no_unlisted_alert_event_constraints
        ),
        required_alert_event_indexes_ready=required_alert_event_indexes_ready,
        no_unlisted_alert_event_indexes=no_unlisted_alert_event_indexes,
        no_unreviewed_alert_event_write_interceptors=(
            no_unreviewed_alert_event_write_interceptors
        ),
    )


async def attest_known_historical_migration_reconciliations(
    executor: Any,
    migration_files: Collection[Path],
    *,
    candidate_names: Collection[str] | None = None,
) -> tuple[
    MigrationReconciliationAttestation
    | ForwardRecoveryMigrationReconciliationAttestation
    | MissingSourceForwardRecoveryMigrationReconciliationAttestation
    | MissingSourceMigrationReconciliationAttestation
    | RenamedMissingSourceMigrationReconciliationAttestation
    | B2BCampaignPartnerMissingSourceMigrationReconciliationAttestation
    | B2BCompanySignalPromotionMissingSourceMigrationReconciliationAttestation
    | B2BWatchlistAlertEventsMissingSourceMigrationReconciliationAttestation,
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
        frozenset({MIGRATION_387_RECONCILIATION.migration_name})
        if candidate_names is None
        else frozenset(candidate_names)
    )
    attestations: list[
        MigrationReconciliationAttestation
        | ForwardRecoveryMigrationReconciliationAttestation
        | MissingSourceForwardRecoveryMigrationReconciliationAttestation
        | MissingSourceMigrationReconciliationAttestation
        | RenamedMissingSourceMigrationReconciliationAttestation
        | B2BCampaignPartnerMissingSourceMigrationReconciliationAttestation
        | B2BCompanySignalPromotionMissingSourceMigrationReconciliationAttestation
        | B2BWatchlistAlertEventsMissingSourceMigrationReconciliationAttestation
    ] = []
    if (
        MIGRATION_379_COMMERCIAL_BILLING_RUN_FENCE_FORWARD_RECOVERY.migration_name
        in requested_names
    ):
        attestations.append(await _attest_migration_379(executor, migration_files))
    if (
        MIGRATION_386_WON_LOSS_FENCE_FORWARD_RECOVERY.migration_name
        in requested_names
    ):
        attestations.append(await _attest_migration_386(executor, migration_files))
    if MIGRATION_387_RECONCILIATION.migration_name in requested_names:
        attestations.append(await _attest_migration_387(executor, migration_files))
    if (
        MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION.migration_name
        in requested_names
    ):
        attestations.append(await _attest_migration_382(executor))
    if (
        MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION.migration_name
        in requested_names
    ):
        attestations.append(await _attest_migration_067(executor))
    if (
        MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION.migration_name
        in requested_names
    ):
        attestations.append(await _attest_migration_297(executor))
    if (
        MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION.migration_name
        in requested_names
    ):
        attestations.append(await _attest_migration_272(executor))
    if (
        MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION.migration_name
        in requested_names
    ):
        attestations.append(await _attest_migration_022b(executor, migration_files))
    return tuple(attestations)
