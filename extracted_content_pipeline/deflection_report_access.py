"""Storage boundary for paid-gated FAQ deflection reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

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


class InMemoryDeflectionReportArtifactStore:
    """Test store with the same tenant/request semantics as the Postgres store."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DeflectionReportAccessRecord] = {}
        self._created_at_by_key: dict[tuple[str, str], datetime] = {}

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
        existing = self._rows.get(key)
        self._created_at_by_key.setdefault(key, datetime.now(timezone.utc))
        cleaned_delivery_email = _clean(delivery_email)
        self._rows[key] = DeflectionReportAccessRecord(
            account_id=key[0],
            request_id=key[1],
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


def _clean(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "DeflectionReportAccessRecord",
    "DeflectionReportListRecord",
    "DeflectionReportArtifactStore",
    "InMemoryDeflectionReportArtifactStore",
    "PostgresDeflectionReportArtifactStore",
    "stored_deflection_report_model",
]
