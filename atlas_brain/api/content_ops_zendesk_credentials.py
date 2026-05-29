"""Content Ops Zendesk credential management routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import uuid as _uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .._content_ops_zendesk_credentials import (
    ContentOpsZendeskCredentialRecord,
    list_zendesk_credentials,
    revoke_zendesk_credentials,
    upsert_zendesk_credentials,
)
from ..auth.dependencies import AuthUser
from ..storage.database import get_db_pool


PoolProvider = Callable[[], Any | Awaitable[Any]]
AuthDependency = Callable[..., AuthUser | Awaitable[AuthUser]]


class UpsertZendeskCredentialRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    api_token: str = Field(..., min_length=1, max_length=4096)
    subdomain: str = Field(default="", max_length=128)
    base_url: str = Field(default="", max_length=512)
    label: str = Field(default="", max_length=128)


class ZendeskCredentialView(BaseModel):
    id: str
    account_id: str
    email: str
    api_token_prefix: str
    subdomain: str
    base_url: str
    label: str
    added_at: str
    last_used_at: str | None = None
    revoked_at: str | None = None


def create_content_ops_zendesk_credentials_router(
    *,
    pool_provider: PoolProvider = get_db_pool,
    auth_dependency: AuthDependency,
) -> APIRouter:
    """Create tenant-scoped Content Ops Zendesk credential routes."""

    router = APIRouter(
        prefix="/content-ops/zendesk-credentials",
        tags=["content-ops"],
    )

    @router.get("", response_model=list[ZendeskCredentialView])
    async def list_credentials(
        user: AuthUser = Depends(auth_dependency),
    ) -> list[ZendeskCredentialView]:
        pool = await _resolve_ready_pool(pool_provider)
        account_id = _account_uuid(user)
        records = await list_zendesk_credentials(pool, account_id=account_id)
        return [_record_to_view(record) for record in records]

    @router.post("", response_model=ZendeskCredentialView, status_code=201)
    async def add_or_rotate_credential(
        body: UpsertZendeskCredentialRequest,
        user: AuthUser = Depends(auth_dependency),
    ) -> ZendeskCredentialView:
        pool = await _resolve_ready_pool(pool_provider)
        account_id = _account_uuid(user)
        try:
            record = await upsert_zendesk_credentials(
                pool,
                account_id=account_id,
                email=body.email,
                api_token=body.api_token,
                subdomain=body.subdomain,
                base_url=body.base_url,
                label=body.label,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            if exc.__class__.__name__ in ("UniqueViolationError", "IntegrityError"):
                raise HTTPException(
                    status_code=409,
                    detail="Concurrent Zendesk credential write lost the race. Retry the request.",
                )
            raise
        return _record_to_view(record)

    @router.delete("/{credential_id}", status_code=204)
    async def revoke_credential(
        credential_id: str,
        user: AuthUser = Depends(auth_dependency),
    ) -> None:
        pool = await _resolve_ready_pool(pool_provider)
        account_id = _account_uuid(user)
        try:
            parsed_credential_id = _uuid.UUID(str(credential_id))
        except (TypeError, ValueError):
            raise HTTPException(status_code=404, detail="Credential not found")
        revoked = await revoke_zendesk_credentials(
            pool,
            account_id=account_id,
            credential_id=parsed_credential_id,
        )
        if not revoked:
            raise HTTPException(status_code=404, detail="Credential not found")
        return None

    return router


async def _resolve_ready_pool(pool_provider: PoolProvider) -> Any:
    pool = pool_provider()
    if hasattr(pool, "__await__"):
        pool = await pool
    if pool is None or getattr(pool, "is_initialized", True) is False:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


def _account_uuid(user: AuthUser) -> _uuid.UUID:
    try:
        return _uuid.UUID(str(user.account_id))
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid tenant scope")


def _record_to_view(record: ContentOpsZendeskCredentialRecord) -> ZendeskCredentialView:
    return ZendeskCredentialView(
        id=str(record.id),
        account_id=str(record.account_id),
        email=record.email,
        api_token_prefix=record.api_token_prefix,
        subdomain=record.subdomain,
        base_url=record.base_url,
        label=record.label,
        added_at=_fmt_time(record.added_at) or "",
        last_used_at=_fmt_time(record.last_used_at),
        revoked_at=_fmt_time(record.revoked_at),
    )


def _fmt_time(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


__all__ = [
    "UpsertZendeskCredentialRequest",
    "ZendeskCredentialView",
    "create_content_ops_zendesk_credentials_router",
]
