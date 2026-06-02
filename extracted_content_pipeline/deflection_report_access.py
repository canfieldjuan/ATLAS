"""Storage boundary for paid-gated FAQ deflection reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

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


class InMemoryDeflectionReportArtifactStore:
    """Test store with the same tenant/request semantics as the Postgres store."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DeflectionReportAccessRecord] = {}

    async def save_report(
        self,
        *,
        account_id: str,
        request_id: str,
        snapshot: Mapping[str, Any],
        artifact: Mapping[str, Any],
        delivery_email: str | None = None,
    ) -> None:
        key = (_required_text(account_id, "account_id"), _required_text(request_id, "request_id"))
        existing = self._rows.get(key)
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
            _clean(payment_reference),
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


def _required_text(value: Any, field: str) -> str:
    text = _clean(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _clean(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "DeflectionReportAccessRecord",
    "DeflectionReportArtifactStore",
    "InMemoryDeflectionReportArtifactStore",
    "PostgresDeflectionReportArtifactStore",
]
