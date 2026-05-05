"""LLM Gateway customer API key management endpoints (PR-D1).

Customer-facing CRUD for API keys (``atls_live_*``). Routes:

  POST   /keys           -- create a new key (returns raw key ONCE)
  GET    /keys           -- list active keys for the calling account
  DELETE /keys/{key_id}  -- revoke (soft-delete) a key

All routes require dashboard JWT auth (``require_auth``); customers
issue keys from the dashboard, then use the keys themselves with
``require_api_key`` on the LLM endpoints. Account-scoped: account A
cannot see or revoke account B's keys.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth.api_keys import (
    APIKeyRecord,
    DEFAULT_SCOPES,
    insert_api_key,
    list_api_keys,
    revoke_api_key,
)
from ..auth.dependencies import AuthUser, require_auth
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.api_keys")

router = APIRouter(prefix="/keys", tags=["api-keys"])


# ---- Request / response schemas ------------------------------------------


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    scopes: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of scope strings. Defaults to the v1 catch-all "
            "'llm:*' when omitted. Future versions may support fine-grained "
            "scopes (llm:chat, llm:batch, llm:embed)."
        ),
    )


class APIKeyView(BaseModel):
    """Display-safe view -- never carries the raw key or hash."""

    id: str
    name: str
    key_prefix: str
    scopes: list[str]
    last_used_at: Optional[str] = None
    created_at: str
    revoked_at: Optional[str] = None


class CreateAPIKeyResponse(BaseModel):
    """Returned exactly once at creation time. ``raw_key`` is the
    value the customer must store; it is not recoverable later."""

    raw_key: str
    key: APIKeyView


def _record_to_view(record: APIKeyRecord) -> APIKeyView:
    def _fmt(value):
        if value is None:
            return None
        try:
            return value.isoformat()
        except AttributeError:
            return str(value)

    return APIKeyView(
        id=str(record.id),
        name=record.name,
        key_prefix=record.key_prefix,
        scopes=list(record.scopes),
        last_used_at=_fmt(record.last_used_at),
        created_at=_fmt(record.created_at),
        revoked_at=_fmt(record.revoked_at),
    )


# ---- Routes --------------------------------------------------------------


@router.post("", response_model=CreateAPIKeyResponse, status_code=201)
async def create_key(
    body: CreateAPIKeyRequest,
    user: AuthUser = Depends(require_auth),
) -> CreateAPIKeyResponse:
    """Mint a new API key for the calling account. The raw key is
    returned exactly once and is not recoverable later."""
    scopes = tuple(body.scopes) if body.scopes else DEFAULT_SCOPES
    pool = get_db_pool()
    result = await insert_api_key(
        pool,
        account_id=_uuid.UUID(user.account_id),
        user_id=_uuid.UUID(user.user_id) if user.user_id else None,
        name=body.name,
        scopes=scopes,
    )
    return CreateAPIKeyResponse(
        raw_key=result.raw_key,
        key=_record_to_view(result.record),
    )


@router.get("", response_model=list[APIKeyView])
async def list_keys(user: AuthUser = Depends(require_auth)) -> list[APIKeyView]:
    """List active API keys for the calling account."""
    pool = get_db_pool()
    records = await list_api_keys(pool, account_id=_uuid.UUID(user.account_id))
    return [_record_to_view(r) for r in records]


@router.delete("/{key_id}", status_code=204)
async def revoke_key(
    key_id: str,
    user: AuthUser = Depends(require_auth),
) -> None:
    """Revoke (soft-delete) an API key. Account-scoped -- 404 when the
    caller does not own the key, to avoid leaking key existence across
    accounts."""
    try:
        key_uuid = _uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Key not found")

    pool = get_db_pool()
    revoked = await revoke_api_key(
        pool,
        key_id=key_uuid,
        account_id=_uuid.UUID(user.account_id),
    )
    if not revoked:
        raise HTTPException(status_code=404, detail="Key not found")
    return None
