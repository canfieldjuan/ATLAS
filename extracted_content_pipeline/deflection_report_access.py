"""Storage boundary for paid-gated FAQ deflection reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from .deflection_delta import compute_deflection_delta
from .faq_deflection_report import DEFLECTION_REPORT_SCHEMA_VERSION
from .storage._jsonb_helpers import (
    decode_jsonb_field,
    json_dump_jsonb,
    parse_command_tag,
    row_to_dict,
)


@dataclass(frozen=True)
class DeflectionReportAccessRecord:
    """Persisted report access state for one request."""

    account_id: str
    request_id: str
    snapshot: dict[str, Any]
    artifact: dict[str, Any] | None
    paid: bool
    payment_reference: str | None = None
    delivery_email: str | None = None

    def report_model(self) -> dict[str, Any] | None:
        """Return the supported persisted structured report model, if present."""

        return stored_deflection_report_model(self.artifact)


@dataclass(frozen=True)
class DeflectionReportListRecord:
    """Unpaid-safe report listing row for one tenant/request pair."""

    account_id: str
    request_id: str
    snapshot: dict[str, Any]
    paid: bool
    delivery_email: str | None = None
    created_at: Any = None
    updated_at: Any = None


@dataclass(frozen=True)
class DeflectionDeltaAccessRecord:
    """Persisted delta for one tenant/current/baseline report pair."""

    account_id: str
    current_request_id: str
    baseline_request_id: str
    delta: dict[str, Any]
    created_at: Any = None
    updated_at: Any = None


@dataclass(frozen=True)
class DeflectionDeltaBatchSummary:
    """Result of a tenant-scoped batch delta generation run."""

    accounts_scanned: int
    reports_scanned: int
    deltas_saved: int
    skipped_no_delta: int
    failed: int = 0


class DeflectionDeltaReadError(ValueError):
    """Raised when a persisted delta is unavailable to a paid read surface."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DeflectionReportArtifactStore(Protocol):
    """Host-owned persistence for snapshots, full artifacts, and paid flags."""

    async def save_report(
        self,
        *,
        account_id: str,
        request_id: str,
        snapshot: Mapping[str, Any],
        artifact: Mapping[str, Any],
        delivery_email: str | None = None,
    ) -> None:
        """Persist a generated report while keeping it locked by default."""

    async def get_snapshot(
        self,
        *,
        account_id: str,
        request_id: str,
    ) -> dict[str, Any] | None:
        """Return the free snapshot for a tenant/request pair."""

    async def list_reports(
        self,
        *,
        account_id: str,
        limit: int | None = 25,
        paid: bool | None = None,
    ) -> tuple[DeflectionReportListRecord, ...]:
        """List free report snapshots for one tenant, newest first."""

    async def list_paid_report_accounts(
        self,
        *,
        limit: int | None = 100,
    ) -> tuple[str, ...]:
        """List tenants with paid reports, newest paid activity first."""

    async def get_artifact_record(
        self,
        *,
        account_id: str,
        request_id: str,
    ) -> DeflectionReportAccessRecord | None:
        """Return paid state and full artifact for a tenant/request pair."""

    async def mark_paid(
        self,
        *,
        account_id: str,
        request_id: str,
        payment_reference: str | None = None,
    ) -> bool:
        """Mark a report paid. Returns False when no tenant/request row exists."""

    async def mark_unpaid(
        self,
        *,
        account_id: str,
        request_id: str,
        payment_reference: str | None = None,
    ) -> bool:
        """Relock a paid report. Returns False when no matching row exists."""

    async def count_reports_older_than(self, *, cutoff: datetime) -> int:
        """Count report rows older than the retention cutoff."""

    async def delete_reports_older_than(
        self,
        *,
        cutoff: datetime,
        limit: int | None = None,
    ) -> int:
        """Delete report rows older than the retention cutoff."""

    async def select_previous_paid_report(
        self,
        *,
        account_id: str,
        current_request_id: str,
    ) -> DeflectionReportAccessRecord | None:
        """Return the newest paid report before the current paid report."""

    async def save_deflection_delta(
        self,
        *,
        account_id: str,
        current_request_id: str,
        baseline_request_id: str,
        delta: Mapping[str, Any],
    ) -> None:
        """Persist one computed deflection delta for a report pair."""

    async def get_deflection_delta(
        self,
        *,
        account_id: str,
        current_request_id: str,
        baseline_request_id: str,
    ) -> DeflectionDeltaAccessRecord | None:
        """Return one persisted deflection delta for a report pair."""

    async def get_paid_deflection_delta(
        self,
        *,
        account_id: str,
        current_request_id: str,
        baseline_request_id: str,
    ) -> DeflectionDeltaAccessRecord | None:
        """Return one persisted delta only while both source reports are paid."""


_DEFLECTION_DELTA_METADATA_FIELDS = (
    "schema_version",
    "title",
    "source_date_start",
    "source_date_end",
    "source_window_days",
)
_DEFLECTION_DELTA_SUMMARY_FIELDS = (
    "current_item_count",
    "baseline_item_count",
    "matched_item_count",
    "support_cost_delta",
    "new_count",
    "resolved_count",
    "resurfaced_count",
    "growing_count",
    "shrinking_count",
    "still_unresolved_count",
    "status_changed_count",
    "cost_changed_count",
    "csat_changed_count",
    "low_confidence_identity_count",
    "stable_count",
)
_DEFLECTION_DELTA_ITEM_FIELDS = (
    "identity_key",
    "repeat_key",
    "cluster_id",
    "identity_basis",
    "identity_confidence",
    "question",
    "owner_lane",
    "fix_type",
    "current_status",
    "baseline_status",
    "current_ticket_count",
    "baseline_ticket_count",
    "ticket_count_delta",
    "current_estimated_support_cost",
    "baseline_estimated_support_cost",
    "support_cost_delta",
    "current_csat_signal",
    "baseline_csat_signal",
    "change_types",
)
_DEFLECTION_DELTA_CSAT_FIELDS = (
    "status",
    "csat_present_count",
    "negative_csat_ticket_count",
    "numeric_average",
)


class InMemoryDeflectionReportArtifactStore:
    """Test store with the same tenant/request semantics as the Postgres store."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DeflectionReportAccessRecord] = {}
        self._created_at_by_key: dict[tuple[str, str], datetime] = {}
        self._deltas: dict[
            tuple[str, str, str],
            DeflectionDeltaAccessRecord,
        ] = {}

    async def save_report(
        self,
        *,
        account_id: str,
        request_id: str,
        snapshot: Mapping[str, Any],
        artifact: Mapping[str, Any],
        delivery_email: str | None = None,
    ) -> None:
        key = (
            _required_text(account_id, "account_id"),
            _required_text(request_id, "request_id"),
        )
        resolved_account, resolved_request = key
        existing = self._rows.get(key)
        self._created_at_by_key.setdefault(key, datetime.now(timezone.utc))
        cleaned_delivery_email = _clean(delivery_email)
        self._rows[key] = DeflectionReportAccessRecord(
            account_id=resolved_account,
            request_id=resolved_request,
            snapshot=dict(snapshot),
            artifact=dict(artifact),
            paid=bool(existing.paid) if existing else False,
            payment_reference=existing.payment_reference if existing else None,
            delivery_email=cleaned_delivery_email
            or (existing.delivery_email if existing else None),
        )

    async def get_snapshot(
        self,
        *,
        account_id: str,
        request_id: str,
    ) -> dict[str, Any] | None:
        row = self._rows.get((account_id, request_id))
        return dict(row.snapshot) if row else None

    async def list_reports(
        self,
        *,
        account_id: str,
        limit: int | None = 25,
        paid: bool | None = None,
    ) -> tuple[DeflectionReportListRecord, ...]:
        resolved_account = _required_text(account_id, "account_id")
        rows = [
            row
            for (row_account, _request_id), row in self._rows.items()
            if row_account == resolved_account and (paid is None or row.paid is paid)
        ]
        selected_rows = rows if limit is None else rows[-_bounded_limit(limit):]
        out: list[DeflectionReportListRecord] = []
        for row in reversed(selected_rows):
            out.append(
                DeflectionReportListRecord(
                    account_id=row.account_id,
                    request_id=row.request_id,
                    snapshot=dict(row.snapshot),
                    paid=row.paid,
                    delivery_email=row.delivery_email,
                    created_at=self._created_at_by_key.get(
                        (row.account_id, row.request_id)
                    ),
                )
            )
        return tuple(out)

    async def list_paid_report_accounts(
        self,
        *,
        limit: int | None = 100,
    ) -> tuple[str, ...]:
        newest_by_account: dict[str, datetime] = {}
        fallback = datetime.min.replace(tzinfo=timezone.utc)
        for key, row in self._rows.items():
            account_id, _request_id = key
            if not row.paid:
                continue
            created_at = self._created_at_by_key.get(key, fallback)
            if created_at > newest_by_account.get(account_id, fallback):
                newest_by_account[account_id] = created_at
        ordered = sorted(
            newest_by_account,
            key=lambda account_id: newest_by_account[account_id],
            reverse=True,
        )
        if limit is None:
            return tuple(ordered)
        return tuple(ordered[:_bounded_limit(limit)])

    async def get_artifact_record(
        self,
        *,
        account_id: str,
        request_id: str,
    ) -> DeflectionReportAccessRecord | None:
        row = self._rows.get((account_id, request_id))
        if row is None:
            return None
        return DeflectionReportAccessRecord(
            account_id=row.account_id,
            request_id=row.request_id,
            snapshot=dict(row.snapshot),
            artifact=dict(row.artifact or {}) if row.artifact is not None else None,
            paid=row.paid,
            payment_reference=row.payment_reference,
            delivery_email=row.delivery_email,
        )

    async def mark_paid(
        self,
        *,
        account_id: str,
        request_id: str,
        payment_reference: str | None = None,
    ) -> bool:
        key = (account_id, request_id)
        row = self._rows.get(key)
        if row is None:
            return False
        self._rows[key] = DeflectionReportAccessRecord(
            account_id=row.account_id,
            request_id=row.request_id,
            snapshot=dict(row.snapshot),
            artifact=dict(row.artifact or {}) if row.artifact is not None else None,
            paid=True,
            payment_reference=_clean(payment_reference) or row.payment_reference,
            delivery_email=row.delivery_email,
        )
        return True

    async def count_reports_older_than(self, *, cutoff: datetime) -> int:
        resolved_cutoff = _required_cutoff(cutoff)
        fallback = datetime.now(timezone.utc)
        return sum(
            1
            for key in self._rows
            if self._created_at_by_key.get(key, fallback) < resolved_cutoff
        )

    async def delete_reports_older_than(
        self,
        *,
        cutoff: datetime,
        limit: int | None = None,
    ) -> int:
        resolved_cutoff = _required_cutoff(cutoff)
        resolved_limit = _optional_positive_limit(limit)
        fallback = datetime.now(timezone.utc)
        keys = [
            key
            for key in self._rows
            if self._created_at_by_key.get(key, fallback) < resolved_cutoff
        ]
        keys.sort(key=lambda key: self._created_at_by_key.get(key, fallback))
        if resolved_limit is not None:
            keys = keys[:resolved_limit]
        for key in keys:
            self._rows.pop(key, None)
            self._created_at_by_key.pop(key, None)
        return len(keys)

    async def mark_unpaid(
        self,
        *,
        account_id: str,
        request_id: str,
        payment_reference: str | None = None,
    ) -> bool:
        key = (account_id, request_id)
        row = self._rows.get(key)
        if row is None:
            return False
        expected_reference = _clean(payment_reference)
        if (
            expected_reference
            and row.payment_reference
            and row.payment_reference != expected_reference
        ):
            return False
        self._rows[key] = DeflectionReportAccessRecord(
            account_id=row.account_id,
            request_id=row.request_id,
            snapshot=dict(row.snapshot),
            artifact=dict(row.artifact or {}) if row.artifact is not None else None,
            paid=False,
            payment_reference=row.payment_reference,
            delivery_email=row.delivery_email,
        )
        return True

    async def select_previous_paid_report(
        self,
        *,
        account_id: str,
        current_request_id: str,
    ) -> DeflectionReportAccessRecord | None:
        resolved_account = _required_text(account_id, "account_id")
        resolved_current = _required_text(current_request_id, "current_request_id")
        current_key = (resolved_account, resolved_current)
        current = self._rows.get(current_key)
        if current is None or not current.paid:
            return None
        current_created_at = self._created_at_by_key.get(current_key)
        if current_created_at is None:
            return None
        candidates = [
            (created_at, row.request_id)
            for (row_account, request_id), row in self._rows.items()
            if row_account == resolved_account
            and request_id != resolved_current
            and row.paid
            and (created_at := self._created_at_by_key.get((row_account, request_id)))
            is not None
            and created_at < current_created_at
        ]
        if not candidates:
            return None
        _selected_created_at, selected_request_id = max(candidates)
        return await self.get_artifact_record(
            account_id=resolved_account,
            request_id=selected_request_id,
        )

    async def save_deflection_delta(
        self,
        *,
        account_id: str,
        current_request_id: str,
        baseline_request_id: str,
        delta: Mapping[str, Any],
    ) -> None:
        key = _delta_key(account_id, current_request_id, baseline_request_id)
        resolved_account, resolved_current, resolved_baseline = key
        current_key = (resolved_account, resolved_current)
        baseline_key = (resolved_account, resolved_baseline)
        if current_key not in self._rows or baseline_key not in self._rows:
            raise ValueError("current and baseline reports must exist")
        now = datetime.now(timezone.utc)
        existing = self._deltas.get(key)
        self._deltas[key] = DeflectionDeltaAccessRecord(
            account_id=resolved_account,
            current_request_id=resolved_current,
            baseline_request_id=resolved_baseline,
            delta=dict(delta),
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )

    async def get_deflection_delta(
        self,
        *,
        account_id: str,
        current_request_id: str,
        baseline_request_id: str,
    ) -> DeflectionDeltaAccessRecord | None:
        key = _delta_key(account_id, current_request_id, baseline_request_id)
        row = self._deltas.get(key)
        if row is None:
            return None
        return DeflectionDeltaAccessRecord(
            account_id=row.account_id,
            current_request_id=row.current_request_id,
            baseline_request_id=row.baseline_request_id,
            delta=dict(row.delta),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def get_paid_deflection_delta(
        self,
        *,
        account_id: str,
        current_request_id: str,
        baseline_request_id: str,
    ) -> DeflectionDeltaAccessRecord | None:
        key = _delta_key(account_id, current_request_id, baseline_request_id)
        resolved_account, resolved_current, resolved_baseline = key
        current = self._rows.get((resolved_account, resolved_current))
        baseline = self._rows.get((resolved_account, resolved_baseline))
        if current is None or baseline is None or not current.paid or not baseline.paid:
            return None
        return await self.get_deflection_delta(
            account_id=resolved_account,
            current_request_id=resolved_current,
            baseline_request_id=resolved_baseline,
        )


@dataclass(frozen=True)
class PostgresDeflectionReportArtifactStore:
    """Async Postgres adapter for paid-gated deflection reports."""

    pool: Any

    async def save_report(
        self,
        *,
        account_id: str,
        request_id: str,
        snapshot: Mapping[str, Any],
        artifact: Mapping[str, Any],
        delivery_email: str | None = None,
    ) -> None:
        await self.pool.execute(
            """
            INSERT INTO content_ops_deflection_reports (
                account_id, request_id, snapshot, artifact, paid, delivery_email, updated_at
            )
            VALUES ($1, $2, $3::jsonb, $4::jsonb, false, $5, NOW())
            ON CONFLICT (account_id, request_id) DO UPDATE
            SET snapshot = EXCLUDED.snapshot,
                artifact = EXCLUDED.artifact,
                paid = content_ops_deflection_reports.paid,
                payment_reference = content_ops_deflection_reports.payment_reference,
                delivery_email = COALESCE(
                    EXCLUDED.delivery_email,
                    content_ops_deflection_reports.delivery_email
                ),
                updated_at = NOW()
            """,
            _required_text(account_id, "account_id"),
            _required_text(request_id, "request_id"),
            json_dump_jsonb(dict(snapshot)),
            json_dump_jsonb(dict(artifact)),
            _clean(delivery_email) or None,
        )

    async def get_snapshot(
        self,
        *,
        account_id: str,
        request_id: str,
    ) -> dict[str, Any] | None:
        row = await self.pool.fetchrow(
            """
            SELECT snapshot
            FROM content_ops_deflection_reports
            WHERE account_id = $1 AND request_id = $2
            """,
            account_id,
            request_id,
        )
        if row is None:
            return None
        snapshot = decode_jsonb_field(row_to_dict(row).get("snapshot"), default={})
        return dict(snapshot) if isinstance(snapshot, Mapping) else {}

    async def list_reports(
        self,
        *,
        account_id: str,
        limit: int | None = 25,
        paid: bool | None = None,
    ) -> tuple[DeflectionReportListRecord, ...]:
        resolved_account = _required_text(account_id, "account_id")
        paid_clause = "" if paid is None else "AND paid = $2"
        args: list[Any] = [resolved_account]
        if paid is not None:
            args.append(bool(paid))
        limit_clause = ""
        if limit is not None:
            args.append(_bounded_limit(limit))
            limit_arg = len(args)
            limit_clause = f"LIMIT ${limit_arg}"
        rows = await self.pool.fetch(
            f"""
            SELECT account_id, request_id, snapshot, paid, delivery_email, created_at, updated_at
            FROM content_ops_deflection_reports
            WHERE account_id = $1
              {paid_clause}
            ORDER BY created_at DESC
            {limit_clause}
            """,
            *args,
        )
        return tuple(_list_record_from_row(row_to_dict(row)) for row in rows)

    async def list_paid_report_accounts(
        self,
        *,
        limit: int | None = 100,
    ) -> tuple[str, ...]:
        args: list[Any] = []
        limit_clause = ""
        if limit is not None:
            args.append(_bounded_limit(limit))
            limit_clause = "LIMIT $1"
        rows = await self.pool.fetch(
            f"""
            SELECT account_id
            FROM content_ops_deflection_reports
            WHERE paid = true
            GROUP BY account_id
            ORDER BY MAX(created_at) DESC, account_id ASC
            {limit_clause}
            """,
            *args,
        )
        return tuple(
            account_id
            for row in rows
            if (account_id := _clean(row_to_dict(row).get("account_id")))
        )

    async def get_artifact_record(
        self,
        *,
        account_id: str,
        request_id: str,
    ) -> DeflectionReportAccessRecord | None:
        row = await self.pool.fetchrow(
            """
            SELECT account_id, request_id, snapshot, artifact, paid, payment_reference, delivery_email
            FROM content_ops_deflection_reports
            WHERE account_id = $1 AND request_id = $2
            """,
            account_id,
            request_id,
        )
        if row is None:
            return None
        return _record_from_row(row_to_dict(row))

    async def mark_paid(
        self,
        *,
        account_id: str,
        request_id: str,
        payment_reference: str | None = None,
    ) -> bool:
        result = await self.pool.execute(
            """
            UPDATE content_ops_deflection_reports
            SET paid = true,
                paid_at = COALESCE(paid_at, NOW()),
                payment_reference = COALESCE($3, payment_reference),
                updated_at = NOW()
            WHERE account_id = $1 AND request_id = $2
            """,
            account_id,
            request_id,
            _clean(payment_reference) or None,
        )
        return parse_command_tag(result)

    async def count_reports_older_than(self, *, cutoff: datetime) -> int:
        count = await self.pool.fetchval(
            """
            SELECT COUNT(*)
            FROM content_ops_deflection_reports
            WHERE created_at < $1
            """,
            _required_cutoff(cutoff),
        )
        return int(count or 0)

    async def delete_reports_older_than(
        self,
        *,
        cutoff: datetime,
        limit: int | None = None,
    ) -> int:
        resolved_cutoff = _required_cutoff(cutoff)
        resolved_limit = _optional_positive_limit(limit)
        if resolved_limit is None:
            result = await self.pool.execute(
                """
                DELETE FROM content_ops_deflection_reports
                WHERE created_at < $1
                """,
                resolved_cutoff,
            )
        else:
            result = await self.pool.execute(
                """
                WITH doomed AS (
                    SELECT account_id, request_id
                    FROM content_ops_deflection_reports
                    WHERE created_at < $1
                    ORDER BY created_at ASC
                    LIMIT $2
                )
                DELETE FROM content_ops_deflection_reports reports
                USING doomed
                WHERE reports.account_id = doomed.account_id
                  AND reports.request_id = doomed.request_id
                """,
                resolved_cutoff,
                resolved_limit,
            )
        return _parse_command_count(result)

    async def mark_unpaid(
        self,
        *,
        account_id: str,
        request_id: str,
        payment_reference: str | None = None,
    ) -> bool:
        result = await self.pool.execute(
            """
            UPDATE content_ops_deflection_reports
            SET paid = false,
                paid_at = NULL,
                updated_at = NOW()
            WHERE account_id = $1
              AND request_id = $2
              AND (
                $3::text IS NULL
                OR payment_reference IS NULL
                OR payment_reference = $3
              )
            """,
            account_id,
            request_id,
            _clean(payment_reference) or None,
        )
        return parse_command_tag(result)

    async def select_previous_paid_report(
        self,
        *,
        account_id: str,
        current_request_id: str,
    ) -> DeflectionReportAccessRecord | None:
        row = await self.pool.fetchrow(
            """
            WITH current_report AS (
                SELECT created_at
                FROM content_ops_deflection_reports
                WHERE account_id = $1
                  AND request_id = $2
                  AND paid = true
            )
            SELECT reports.account_id,
                   reports.request_id,
                   reports.snapshot,
                   reports.artifact,
                   reports.paid,
                   reports.payment_reference,
                   reports.delivery_email
            FROM content_ops_deflection_reports reports
            JOIN current_report ON true
            WHERE reports.account_id = $1
              AND reports.request_id <> $2
              AND reports.paid = true
              AND reports.created_at < current_report.created_at
            ORDER BY reports.created_at DESC
            LIMIT 1
            """,
            _required_text(account_id, "account_id"),
            _required_text(current_request_id, "current_request_id"),
        )
        return _record_from_row(row_to_dict(row)) if row is not None else None

    async def save_deflection_delta(
        self,
        *,
        account_id: str,
        current_request_id: str,
        baseline_request_id: str,
        delta: Mapping[str, Any],
    ) -> None:
        key = _delta_key(account_id, current_request_id, baseline_request_id)
        resolved_account, resolved_current, resolved_baseline = key
        await self.pool.execute(
            """
            INSERT INTO content_ops_deflection_deltas (
                account_id,
                current_request_id,
                baseline_request_id,
                delta,
                updated_at
            )
            VALUES ($1, $2, $3, $4::jsonb, NOW())
            ON CONFLICT (account_id, current_request_id, baseline_request_id)
            DO UPDATE
            SET delta = EXCLUDED.delta,
                updated_at = NOW()
            """,
            resolved_account,
            resolved_current,
            resolved_baseline,
            json_dump_jsonb(dict(delta)),
        )

    async def get_deflection_delta(
        self,
        *,
        account_id: str,
        current_request_id: str,
        baseline_request_id: str,
    ) -> DeflectionDeltaAccessRecord | None:
        key = _delta_key(account_id, current_request_id, baseline_request_id)
        resolved_account, resolved_current, resolved_baseline = key
        row = await self.pool.fetchrow(
            """
            SELECT account_id,
                   current_request_id,
                   baseline_request_id,
                   delta,
                   created_at,
                   updated_at
            FROM content_ops_deflection_deltas
            WHERE account_id = $1
              AND current_request_id = $2
              AND baseline_request_id = $3
            """,
            resolved_account,
            resolved_current,
            resolved_baseline,
        )
        return _delta_record_from_row(row_to_dict(row)) if row is not None else None

    async def get_paid_deflection_delta(
        self,
        *,
        account_id: str,
        current_request_id: str,
        baseline_request_id: str,
    ) -> DeflectionDeltaAccessRecord | None:
        key = _delta_key(account_id, current_request_id, baseline_request_id)
        resolved_account, resolved_current, resolved_baseline = key
        row = await self.pool.fetchrow(
            """
            SELECT deltas.account_id,
                   deltas.current_request_id,
                   deltas.baseline_request_id,
                   deltas.delta,
                   deltas.created_at,
                   deltas.updated_at
            FROM content_ops_deflection_deltas deltas
            JOIN content_ops_deflection_reports current_report
              ON current_report.account_id = deltas.account_id
             AND current_report.request_id = deltas.current_request_id
             AND current_report.paid = true
            JOIN content_ops_deflection_reports baseline_report
              ON baseline_report.account_id = deltas.account_id
             AND baseline_report.request_id = deltas.baseline_request_id
             AND baseline_report.paid = true
            WHERE deltas.account_id = $1
              AND deltas.current_request_id = $2
              AND deltas.baseline_request_id = $3
            """,
            resolved_account,
            resolved_current,
            resolved_baseline,
        )
        return _delta_record_from_row(row_to_dict(row)) if row is not None else None


_STORED_ACTION_SECTION_LIMIT_DEFAULTS = {
    "priority_fix_queue": {
        "result_page_limit": 3,
        "pdf_limit": 10,
        "backlog_limit": 25,
    },
    "top_unresolved_repeats": {
        "result_page_limit": 3,
        "pdf_limit": 10,
    },
    "drafted_resolutions": {
        "result_page_limit": 3,
        "pdf_limit": 10,
    },
    "already_covered_still_recurring": {
        "result_page_limit": 3,
        "pdf_limit": 10,
    },
}


def _record_from_row(row: Mapping[str, Any]) -> DeflectionReportAccessRecord:
    snapshot = decode_jsonb_field(row.get("snapshot"), default={})
    artifact = decode_jsonb_field(row.get("artifact"), default={})
    return DeflectionReportAccessRecord(
        account_id=str(row.get("account_id") or ""),
        request_id=str(row.get("request_id") or ""),
        snapshot=dict(snapshot) if isinstance(snapshot, Mapping) else {},
        artifact=dict(artifact) if isinstance(artifact, Mapping) else None,
        paid=bool(row.get("paid")),
        payment_reference=_clean(row.get("payment_reference")),
        delivery_email=_clean(row.get("delivery_email")) or None,
    )


def stored_deflection_report_model(
    artifact: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Project a safe persisted `deflection.v1` report model from an artifact."""

    if not isinstance(artifact, Mapping):
        return None
    raw_model = artifact.get("report_model")
    if not isinstance(raw_model, Mapping):
        return None
    if _clean(raw_model.get("schema_version")) != DEFLECTION_REPORT_SCHEMA_VERSION:
        return None

    raw_sections = raw_model.get("sections")
    if not _is_sequence(raw_sections):
        return None
    sections = [
        section
        for raw_section in raw_sections
        if (section := _stored_report_model_section(raw_section)) is not None
    ]
    if not sections:
        return None

    summary = raw_model.get("summary")
    sections.sort(key=lambda section: section["priority"])
    return {
        "schema_version": DEFLECTION_REPORT_SCHEMA_VERSION,
        "title": _clean(raw_model.get("title")) or "Support Ticket Deflection Report",
        "summary": dict(summary) if isinstance(summary, Mapping) else {},
        "sections": sections,
    }


def _stored_report_model_section(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    section_id = _clean(value.get("id"))
    if not section_id:
        return None
    priority = _required_int(value.get("priority"))
    if priority is None:
        return None
    raw_data = value.get("data")
    data = dict(raw_data) if isinstance(raw_data, Mapping) else {}
    required_data = _text_list(value.get("required_data"))
    required_data, data = _normalize_stored_action_section_limits(
        section_id,
        required_data,
        data,
    )
    if any(key not in data for key in required_data):
        return None
    return {
        "id": section_id,
        "title": _clean(value.get("title")) or section_id.replace("_", " ").title(),
        "priority": priority,
        "surfaces": _text_list(value.get("surfaces")),
        "default_limit": _optional_int(value.get("default_limit")),
        "required_data": required_data,
        "snapshot_safe_fields": _text_list(value.get("snapshot_safe_fields")),
        "data": data,
    }


def _normalize_stored_action_section_limits(
    section_id: str,
    required_data: list[str],
    data: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    limit_defaults = _STORED_ACTION_SECTION_LIMIT_DEFAULTS.get(section_id)
    if limit_defaults is None:
        return required_data, data

    normalized_data = dict(data)
    for key, default in limit_defaults.items():
        if _optional_int(normalized_data.get(key)) is None:
            normalized_data[key] = default

    normalized_required = list(required_data)
    for key in limit_defaults:
        if key not in normalized_required:
            normalized_required.append(key)
    return normalized_required, normalized_data


def _list_record_from_row(row: Mapping[str, Any]) -> DeflectionReportListRecord:
    snapshot = decode_jsonb_field(row.get("snapshot"), default={})
    return DeflectionReportListRecord(
        account_id=str(row.get("account_id") or ""),
        request_id=str(row.get("request_id") or ""),
        snapshot=dict(snapshot) if isinstance(snapshot, Mapping) else {},
        paid=bool(row.get("paid")),
        delivery_email=_clean(row.get("delivery_email")) or None,
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _delta_record_from_row(row: Mapping[str, Any]) -> DeflectionDeltaAccessRecord:
    delta = decode_jsonb_field(row.get("delta"), default={})
    return DeflectionDeltaAccessRecord(
        account_id=str(row.get("account_id") or ""),
        current_request_id=str(row.get("current_request_id") or ""),
        baseline_request_id=str(row.get("baseline_request_id") or ""),
        delta=dict(delta) if isinstance(delta, Mapping) else {},
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def deflection_delta_read_payload(
    record: DeflectionDeltaAccessRecord,
) -> dict[str, Any]:
    """Return the allowlisted customer-facing payload for a persisted delta."""

    return {
        "schema_version": "deflection_delta_read.v1",
        "current_request_id": record.current_request_id,
        "baseline_request_id": record.baseline_request_id,
        "delta": _allowlisted_deflection_delta(record.delta),
        "metadata": {
            "created_at": _timestamp(record.created_at),
            "updated_at": _timestamp(record.updated_at),
        },
    }


def _require_supported_delta_schema(delta: Mapping[str, Any]) -> None:
    if _clean(delta.get("schema_version")) != "deflection_delta.v1":
        raise DeflectionDeltaReadError(
            "unsupported_delta_schema",
            "Persisted deflection delta uses an unsupported schema.",
        )


def _bounded_limit(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 25
    return max(1, min(parsed, 100))


def _optional_positive_limit(value: Any) -> int | None:
    if value is None:
        return None
    parsed = _parse_int(value)
    if parsed is None or parsed < 1:
        raise ValueError("limit must be greater than 0")
    return parsed


def _required_cutoff(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("cutoff must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("cutoff must be timezone-aware")
    return value


def _parse_command_count(result: Any) -> int:
    if isinstance(result, bool):
        raise ValueError("database command count must not be boolean")
    if isinstance(result, int):
        if result < 0:
            raise ValueError("database command count must not be negative")
        return result
    if not isinstance(result, str):
        raise ValueError("database command tag must be a string or integer")
    text = result.strip()
    _command, separator, count_text = text.rpartition(" ")
    if not separator or not count_text.isdecimal():
        raise ValueError(f"could not parse database command count from {result!r}")
    return int(count_text)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return _parse_int(value)


def _required_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return _parse_int(value)


def _parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _text_list(value: Any) -> list[str]:
    if not _is_sequence(value):
        return []
    return [text for item in value if (text := _clean(item))]


def _required_text(value: Any, field: str) -> str:
    text = _clean(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _delta_key(
    account_id: Any,
    current_request_id: Any,
    baseline_request_id: Any,
) -> tuple[str, str, str]:
    account = _required_text(account_id, "account_id")
    current = _required_text(current_request_id, "current_request_id")
    baseline = _required_text(baseline_request_id, "baseline_request_id")
    if current == baseline:
        raise ValueError("current_request_id and baseline_request_id must differ")
    return account, current, baseline


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _allowlisted_deflection_delta(delta: Mapping[str, Any]) -> dict[str, Any]:
    _require_supported_delta_schema(delta)
    return {
        "schema_version": "deflection_delta.v1",
        "current": _allowlisted_mapping(
            delta.get("current"),
            _DEFLECTION_DELTA_METADATA_FIELDS,
        ),
        "baseline": _allowlisted_mapping(
            delta.get("baseline"),
            _DEFLECTION_DELTA_METADATA_FIELDS,
        ),
        "summary": _allowlisted_mapping(
            delta.get("summary"),
            _DEFLECTION_DELTA_SUMMARY_FIELDS,
        ),
        "items": _allowlisted_delta_items(delta.get("items")),
    }


def _allowlisted_mapping(value: Any, fields: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {field: value.get(field) for field in fields if field in value}


def _allowlisted_delta_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        projected = _allowlisted_mapping(item, _DEFLECTION_DELTA_ITEM_FIELDS)
        if isinstance(projected.get("current_csat_signal"), Mapping):
            projected["current_csat_signal"] = _allowlisted_mapping(
                projected["current_csat_signal"],
                _DEFLECTION_DELTA_CSAT_FIELDS,
            )
        if isinstance(projected.get("baseline_csat_signal"), Mapping):
            projected["baseline_csat_signal"] = _allowlisted_mapping(
                projected["baseline_csat_signal"],
                _DEFLECTION_DELTA_CSAT_FIELDS,
            )
        out.append(projected)
    return out


async def compute_and_save_previous_deflection_delta(
    store: DeflectionReportArtifactStore,
    *,
    account_id: str,
    current_request_id: str,
) -> DeflectionDeltaAccessRecord | None:
    """Compute and persist the previous-paid-report delta for one tenant."""

    current = await store.get_artifact_record(
        account_id=account_id,
        request_id=current_request_id,
    )
    if current is None or not current.paid:
        return None
    baseline = await store.select_previous_paid_report(
        account_id=account_id,
        current_request_id=current_request_id,
    )
    if baseline is None:
        return None
    current_model = current.report_model()
    baseline_model = baseline.report_model()
    if current_model is None or baseline_model is None:
        return None
    delta = compute_deflection_delta(current_model, baseline_model)
    await store.save_deflection_delta(
        account_id=account_id,
        current_request_id=current.request_id,
        baseline_request_id=baseline.request_id,
        delta=delta,
    )
    return await store.get_deflection_delta(
        account_id=account_id,
        current_request_id=current.request_id,
        baseline_request_id=baseline.request_id,
    )


async def compute_and_save_recent_deflection_deltas(
    store: DeflectionReportArtifactStore,
    *,
    account_limit: int | None = 100,
    reports_per_account: int | None = 25,
) -> DeflectionDeltaBatchSummary:
    """Persist previous-paid-report deltas for recent paid reports."""

    accounts = await store.list_paid_report_accounts(limit=account_limit)
    reports_scanned = 0
    deltas_saved = 0
    skipped_no_delta = 0
    failed = 0

    for account_id in accounts:
        reports = await store.list_reports(
            account_id=account_id,
            limit=reports_per_account,
            paid=True,
        )
        for report in reports:
            reports_scanned += 1
            try:
                record = await compute_and_save_previous_deflection_delta(
                    store,
                    account_id=account_id,
                    current_request_id=report.request_id,
                )
            except Exception:
                failed += 1
                continue
            if record is None:
                skipped_no_delta += 1
                continue
            deltas_saved += 1

    return DeflectionDeltaBatchSummary(
        accounts_scanned=len(accounts),
        reports_scanned=reports_scanned,
        deltas_saved=deltas_saved,
        skipped_no_delta=skipped_no_delta,
        failed=failed,
    )


async def fetch_paid_deflection_delta(
    store: DeflectionReportArtifactStore,
    *,
    account_id: str,
    current_request_id: str,
    baseline_request_id: str | None = None,
) -> DeflectionDeltaAccessRecord:
    """Fetch a stored delta only while both source reports remain paid."""

    resolved_account = _clean(account_id)
    if not resolved_account:
        raise DeflectionDeltaReadError(
            "account_id_required",
            "Content Ops account ID is required.",
        )
    resolved_current = _clean(current_request_id)
    if not resolved_current:
        raise DeflectionDeltaReadError(
            "current_request_id_required",
            "Current deflection report request ID is required.",
        )
    current = await store.get_artifact_record(
        account_id=resolved_account,
        request_id=resolved_current,
    )
    if current is None:
        raise DeflectionDeltaReadError(
            "current_report_not_found",
            "Current deflection report was not found.",
        )
    if not current.paid:
        raise DeflectionDeltaReadError(
            "current_report_locked",
            "Current deflection report is locked.",
        )

    requested_baseline = _clean(baseline_request_id)
    if requested_baseline:
        if requested_baseline == resolved_current:
            raise DeflectionDeltaReadError(
                "invalid_report_pair",
                "Current and baseline deflection reports must differ.",
            )
        baseline = await store.get_artifact_record(
            account_id=resolved_account,
            request_id=requested_baseline,
        )
        if baseline is None:
            raise DeflectionDeltaReadError(
                "baseline_report_not_found",
                "Baseline deflection report was not found.",
            )
        if not baseline.paid:
            raise DeflectionDeltaReadError(
                "baseline_report_locked",
                "Baseline deflection report is locked.",
            )
    else:
        baseline = await store.select_previous_paid_report(
            account_id=resolved_account,
            current_request_id=resolved_current,
        )
        if baseline is None:
            raise DeflectionDeltaReadError(
                "baseline_report_not_found",
                "No paid baseline deflection report is available.",
            )

    record = await store.get_paid_deflection_delta(
        account_id=resolved_account,
        current_request_id=current.request_id,
        baseline_request_id=baseline.request_id,
    )
    if record is None:
        raise DeflectionDeltaReadError(
            "delta_not_found",
            "Persisted deflection delta was not found for this report pair.",
        )
    _require_supported_delta_schema(record.delta)
    return record


__all__ = [
    "DeflectionDeltaAccessRecord",
    "DeflectionDeltaBatchSummary",
    "DeflectionDeltaReadError",
    "DeflectionReportAccessRecord",
    "DeflectionReportListRecord",
    "DeflectionReportArtifactStore",
    "InMemoryDeflectionReportArtifactStore",
    "PostgresDeflectionReportArtifactStore",
    "compute_and_save_previous_deflection_delta",
    "compute_and_save_recent_deflection_deltas",
    "deflection_delta_read_payload",
    "fetch_paid_deflection_delta",
    "stored_deflection_report_model",
]
