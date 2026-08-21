"""Immutable evidence for named historical migration-source gaps.

This module records what Atlas can prove about a known historical mismatch
without rewriting the migration ledger or treating unavailable source bytes as
verified. It is intentionally narrow: every additional historical mismatch
requires its own reviewed evidence record and catalog predicate.
"""

from __future__ import annotations

import hashlib
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


MIGRATION_387_RECONCILIATION = HistoricalMigrationReconciliation(
    reconciliation_id="eom-migration-387-recurring-invoice-dedup-recovery",
    migration_name="387_eom_recurring_invoice_dedup_recovery",
    historical_ledger_sha256=(
        "1dae95d216bfdc836461943af1c6ce382ff7dd21b92eff41d4c94088f72315b2"
    ),
    final_packaged_sha256=(
        "f6382a07d807f7b38772e9823c66f1e47e4118841611e259220d9ab654c84f3d"
    ),
    observed_applied_at=datetime(2026, 8, 21, 1, 30, 46, tzinfo=timezone.utc),
    earliest_retained_source_commit_at=datetime(
        2026, 8, 21, 3, 21, 35, tzinfo=timezone.utc
    ),
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


async def attest_known_historical_migration_reconciliations(
    executor: Any,
    migration_files: Collection[Path],
) -> tuple[MigrationReconciliationAttestation, ...]:
    """Return read-only attestation for the reviewed 387 source gap.

    This is intentionally not a generic exception mechanism. Every additional
    gap requires a reviewed evidence record and a named catalog probe here.
    """
    return (await _attest_migration_387(executor, migration_files),)
