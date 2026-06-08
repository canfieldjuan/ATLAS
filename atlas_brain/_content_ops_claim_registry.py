"""Tenant claim and messaging registry persistence for Content Ops review."""

from __future__ import annotations

import json
import logging
import uuid as _uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Optional

from atlas_brain._content_ops_review_workflow import TenantClaimRegistryReadError
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.claims_map import RegistryClaim
from extracted_content_pipeline.review_contract import RiskTier

logger = logging.getLogger("atlas.content_ops_claim_registry")


@dataclass(frozen=True)
class ContentOpsClaimRegistryRecord:
    """Display-safe saved claim-registry row."""

    id: _uuid.UUID
    account_id: _uuid.UUID
    registry_id: str
    approved_wording: str
    risk_tier: Optional[RiskTier]
    expires_on: Optional[date]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    archived_at: Optional[datetime]

    def as_registry_claim(self) -> RegistryClaim:
        return RegistryClaim(
            id=self.registry_id,
            approved_wording=self.approved_wording,
            risk_tier=self.risk_tier,
            expiration=self.expires_on,
        )


@dataclass(frozen=True)
class _ValidatedClaim:
    registry_id: str
    approved_wording: str
    risk_tier: Optional[RiskTier]
    expires_on: Optional[date]
    metadata: dict[str, Any]


class ContentOpsClaimRegistryRepository:
    """Postgres-backed implementation of the review workflow registry reader."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def list_registry_claims(
        self,
        *,
        scope: TenantScope,
    ) -> Mapping[str, RegistryClaim]:
        account_id = _scope_account_uuid(scope)
        if account_id is None:
            logger.warning("claim registry read: invalid tenant scope")
            raise TenantClaimRegistryReadError("valid tenant scope required")

        try:
            records = await list_registry_claim_records(self._pool, account_id=account_id)
        except Exception as exc:
            logger.exception("claim registry read failed")
            raise TenantClaimRegistryReadError("claim registry read failed") from exc
        claims: dict[str, RegistryClaim] = {}
        for record in records:
            if not record.registry_id or record.registry_id in claims:
                continue
            claims[record.registry_id] = record.as_registry_claim()
        return claims


async def create_registry_claim(
    pool: Any,
    *,
    account_id: _uuid.UUID,
    payload: Mapping[str, Any],
) -> ContentOpsClaimRegistryRecord:
    """Create one active claim-registry row for an account."""

    claim = _validated_claim(payload)
    row = await pool.fetchrow(
        """
        INSERT INTO content_ops_claim_registry (
            account_id, registry_id, approved_wording, risk_tier,
            expires_on, metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        RETURNING id, account_id, registry_id, approved_wording, risk_tier,
                  expires_on, metadata, created_at, updated_at, archived_at
        """,
        account_id,
        claim.registry_id,
        claim.approved_wording,
        _risk_tier_value(claim.risk_tier),
        claim.expires_on,
        _json_dump(claim.metadata),
    )
    return _display_record(row)


async def update_registry_claim(
    pool: Any,
    *,
    account_id: _uuid.UUID,
    claim_id: _uuid.UUID,
    payload: Mapping[str, Any],
) -> ContentOpsClaimRegistryRecord | None:
    """Replace one active tenant claim row. Missing/cross-tenant rows return None."""

    claim = _validated_claim(payload)
    row = await pool.fetchrow(
        """
        UPDATE content_ops_claim_registry
           SET registry_id = $3,
               approved_wording = $4,
               risk_tier = $5,
               expires_on = $6,
               metadata = $7::jsonb,
               updated_at = NOW()
         WHERE id = $1
           AND account_id = $2
           AND archived_at IS NULL
        RETURNING id, account_id, registry_id, approved_wording, risk_tier,
                  expires_on, metadata, created_at, updated_at, archived_at
        """,
        claim_id,
        account_id,
        claim.registry_id,
        claim.approved_wording,
        _risk_tier_value(claim.risk_tier),
        claim.expires_on,
        _json_dump(claim.metadata),
    )
    return _display_record(row) if row is not None else None


async def list_registry_claim_records(
    pool: Any,
    *,
    account_id: _uuid.UUID,
) -> list[ContentOpsClaimRegistryRecord]:
    """Return active claim-registry rows for one tenant."""

    rows = await pool.fetch(
        """
        SELECT id, account_id, registry_id, approved_wording, risk_tier,
               expires_on, metadata, created_at, updated_at, archived_at
          FROM content_ops_claim_registry
         WHERE account_id = $1
           AND archived_at IS NULL
         ORDER BY updated_at DESC
        """,
        account_id,
    )
    return [_display_record(row) for row in rows]


async def expire_registry_claim(
    pool: Any,
    *,
    account_id: _uuid.UUID,
    claim_id: _uuid.UUID,
    expires_on: date | None = None,
) -> ContentOpsClaimRegistryRecord | None:
    """Set a tenant claim expiration date while keeping it readable."""

    expiration = expires_on or date.today()
    row = await pool.fetchrow(
        """
        UPDATE content_ops_claim_registry
           SET expires_on = $3,
               updated_at = NOW()
         WHERE id = $1
           AND account_id = $2
           AND archived_at IS NULL
        RETURNING id, account_id, registry_id, approved_wording, risk_tier,
                  expires_on, metadata, created_at, updated_at, archived_at
        """,
        claim_id,
        account_id,
        expiration,
    )
    return _display_record(row) if row is not None else None


async def archive_registry_claim(
    pool: Any,
    *,
    account_id: _uuid.UUID,
    claim_id: _uuid.UUID,
) -> bool:
    """Soft-delete one tenant claim-registry row."""

    row = await pool.fetchrow(
        """
        UPDATE content_ops_claim_registry
           SET archived_at = NOW(),
               updated_at = NOW()
         WHERE id = $1
           AND account_id = $2
           AND archived_at IS NULL
        RETURNING id
        """,
        claim_id,
        account_id,
    )
    return row is not None


def _validated_claim(payload: Mapping[str, Any]) -> _ValidatedClaim:
    if not isinstance(payload, Mapping):
        raise ValueError("Claim registry payload must be an object")

    registry_id = _clean_registry_id(payload.get("registry_id"))
    if not registry_id:
        raise ValueError("Claim registry id is required")

    approved_wording = _clean(payload.get("approved_wording"))
    if not approved_wording:
        raise ValueError("Approved wording is required")

    return _ValidatedClaim(
        registry_id=registry_id,
        approved_wording=approved_wording,
        risk_tier=_optional_risk_tier(payload.get("risk_tier")),
        expires_on=_optional_date(payload.get("expires_on")),
        metadata=_metadata(payload.get("metadata")),
    )


def _display_record(row: Any) -> ContentOpsClaimRegistryRecord:
    return ContentOpsClaimRegistryRecord(
        id=row["id"],
        account_id=row["account_id"],
        registry_id=_clean_registry_id(row["registry_id"]),
        approved_wording=_clean(row["approved_wording"]),
        risk_tier=_optional_risk_tier(row["risk_tier"]),
        expires_on=_optional_date(row["expires_on"]),
        metadata=_json_mapping(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _scope_account_uuid(scope: TenantScope | None) -> _uuid.UUID | None:
    value = getattr(scope, "account_id", None)
    if not isinstance(value, str):
        return None
    try:
        return _uuid.UUID(value.strip())
    except ValueError:
        return None


def _optional_risk_tier(value: Any) -> RiskTier | None:
    if not isinstance(value, str):
        return None
    cleaned = _clean(value).casefold()
    if not cleaned:
        return None
    for tier in RiskTier:
        if tier == cleaned:
            return tier
    raise ValueError(f"Invalid risk tier: {value}")


def _risk_tier_value(value: RiskTier | None) -> str | None:
    return value.value if value is not None else None


def _optional_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        cleaned = _clean(value)
        if not cleaned:
            return None
        try:
            return date.fromisoformat(cleaned)
        except ValueError as exc:
            raise ValueError(f"Invalid expiration date: {value}") from exc
    return None


def _metadata(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_mapping(value: Any) -> dict[str, Any]:
    decoded = _decode_json(value, default={})
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _decode_json(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _clean(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _clean_registry_id(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


__all__ = [
    "ContentOpsClaimRegistryRecord",
    "ContentOpsClaimRegistryRepository",
    "archive_registry_claim",
    "create_registry_claim",
    "expire_registry_claim",
    "list_registry_claim_records",
    "update_registry_claim",
]
