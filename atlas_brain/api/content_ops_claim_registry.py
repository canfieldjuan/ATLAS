"""Tenant-scoped admin CRUD for the Content Ops claim registry.

The marketer verify MCP surface is deliberately verify-only, so curation of a
tenant's approved-wording claim registry lives here as an authenticated admin
API, mirroring the calibration-library and brand-voice-profiles admin pattern.
The verify flow reads these rows server-side via
``ContentOpsClaimRegistryRepository``.
"""

from __future__ import annotations

import uuid as _uuid
from inspect import isawaitable
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .._content_ops_claim_registry import (
    ContentOpsClaimRegistryRecord,
    archive_registry_claim,
    create_registry_claim,
    expire_registry_claim,
    list_registry_claim_records,
    update_registry_claim,
)
from ..auth.dependencies import AuthUser

PoolProvider = Callable[[], Any | Awaitable[Any]]
AuthDependency = Callable[..., AuthUser | Awaitable[AuthUser]]


class UpsertClaimRegistryRequest(BaseModel):
    registry_id: str = Field(..., min_length=1, max_length=160)
    approved_wording: str = Field(..., min_length=1, max_length=4000)
    risk_tier: str | None = Field(default=None, max_length=40)
    expires_on: str | None = Field(default=None, max_length=40)
    metadata: dict[str, object] = Field(default_factory=dict)


class ExpireClaimRegistryRequest(BaseModel):
    expires_on: str | None = Field(default=None, max_length=40)


class ClaimRegistryView(BaseModel):
    id: str
    account_id: str
    registry_id: str
    approved_wording: str
    risk_tier: str | None = None
    expires_on: str | None = None
    metadata: dict[str, object]
    created_at: str
    updated_at: str
    archived_at: str | None = None


def _default_pool_provider() -> Any:
    from ..storage.database import get_db_pool

    return get_db_pool()


def create_content_ops_claim_registry_router(
    *,
    pool_provider: PoolProvider = _default_pool_provider,
    auth_dependency: AuthDependency,
) -> APIRouter:
    """Build the tenant-scoped claim-registry admin router."""

    router = APIRouter(
        prefix="/content-ops/claim-registry",
        tags=["content-ops"],
    )

    @router.get("", response_model=list[ClaimRegistryView])
    async def list_claims(
        user: AuthUser = Depends(auth_dependency),
    ) -> list[ClaimRegistryView]:
        account_id = _account_uuid(user)
        pool = await _resolve_ready_pool(pool_provider)
        records = await list_registry_claim_records(pool, account_id=account_id)
        return [_record_to_view(record) for record in records]

    @router.post("", response_model=ClaimRegistryView, status_code=201)
    async def add_claim(
        body: UpsertClaimRegistryRequest,
        user: AuthUser = Depends(auth_dependency),
    ) -> ClaimRegistryView:
        _require_claim_registry_admin(user)
        account_id = _account_uuid(user)
        pool = await _resolve_ready_pool(pool_provider)
        try:
            record = await create_registry_claim(
                pool, account_id=account_id, payload=body.model_dump()
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - translate DB unique conflict
            if exc.__class__.__name__ == "UniqueViolationError":
                raise HTTPException(
                    status_code=409,
                    detail="A claim with this registry id already exists",
                )
            raise
        return _record_to_view(record)

    @router.put("/{row_id}", response_model=ClaimRegistryView)
    async def update_claim(
        row_id: str,
        body: UpsertClaimRegistryRequest,
        user: AuthUser = Depends(auth_dependency),
    ) -> ClaimRegistryView:
        _require_claim_registry_admin(user)
        account_id = _account_uuid(user)
        pool = await _resolve_ready_pool(pool_provider)
        try:
            record = await update_registry_claim(
                pool,
                account_id=account_id,
                claim_id=_row_uuid(row_id),
                payload=body.model_dump(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - translate DB unique conflict
            if exc.__class__.__name__ == "UniqueViolationError":
                raise HTTPException(
                    status_code=409,
                    detail="A claim with this registry id already exists",
                )
            raise
        if record is None:
            raise HTTPException(status_code=404, detail="Claim not found")
        return _record_to_view(record)

    @router.post("/{row_id}/expire", response_model=ClaimRegistryView)
    async def expire_claim(
        row_id: str,
        body: ExpireClaimRegistryRequest | None = None,
        user: AuthUser = Depends(auth_dependency),
    ) -> ClaimRegistryView:
        _require_claim_registry_admin(user)
        account_id = _account_uuid(user)
        pool = await _resolve_ready_pool(pool_provider)
        try:
            record = await expire_registry_claim(
                pool,
                account_id=account_id,
                claim_id=_row_uuid(row_id),
                expires_on=_optional_expiration(body),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if record is None:
            raise HTTPException(status_code=404, detail="Claim not found")
        return _record_to_view(record)

    @router.delete("/{row_id}", status_code=204)
    async def delete_claim(
        row_id: str,
        user: AuthUser = Depends(auth_dependency),
    ) -> None:
        _require_claim_registry_admin(user)
        account_id = _account_uuid(user)
        pool = await _resolve_ready_pool(pool_provider)
        archived = await archive_registry_claim(
            pool, account_id=account_id, claim_id=_row_uuid(row_id)
        )
        if not archived:
            raise HTTPException(status_code=404, detail="Claim not found")

    return router


async def _resolve_ready_pool(pool_provider: PoolProvider) -> Any:
    pool = pool_provider()
    if isawaitable(pool):
        pool = await pool
    return pool


def _account_uuid(user: AuthUser) -> _uuid.UUID:
    try:
        return _uuid.UUID(str(user.account_id))
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid tenant scope")


def _row_uuid(row_id: str) -> _uuid.UUID:
    try:
        return _uuid.UUID(str(row_id))
    except (TypeError, ValueError):
        raise HTTPException(status_code=404, detail="Claim not found")


def _optional_expiration(body: ExpireClaimRegistryRequest | None):
    """Parse an optional expiration date string from the request body.

    A missing body or blank value defers to the repo default (today). A
    malformed date string is a 400 rather than a silent fallthrough.
    """

    from datetime import date

    raw = getattr(body, "expires_on", None) if body is not None else None
    if raw is None:
        return None
    cleaned = str(raw).strip()
    if not cleaned:
        return None
    try:
        return date.fromisoformat(cleaned)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid expiration date")


def _require_claim_registry_admin(user: AuthUser) -> None:
    role = str(getattr(user, "role", "") or "").strip().lower()
    if bool(getattr(user, "is_admin", False)) or role in {"owner", "admin"}:
        return
    raise HTTPException(status_code=403, detail="Admin access required")


def _record_to_view(record: ContentOpsClaimRegistryRecord) -> ClaimRegistryView:
    return ClaimRegistryView(
        id=str(record.id),
        account_id=str(record.account_id),
        registry_id=record.registry_id,
        approved_wording=record.approved_wording,
        risk_tier=_risk_tier_str(record.risk_tier),
        expires_on=_fmt_date(record.expires_on),
        metadata=dict(record.metadata),
        created_at=_fmt_time(record.created_at) or "",
        updated_at=_fmt_time(record.updated_at) or "",
        archived_at=_fmt_time(record.archived_at),
    )


def _risk_tier_str(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", None) or str(value)


def _fmt_date(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _fmt_time(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)
