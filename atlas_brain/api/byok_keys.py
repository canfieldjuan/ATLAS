"""LLM Gateway BYOK provider key management endpoints (PR-D5).

Customer-facing CRUD for the customer's own LLM provider API keys
(Anthropic, OpenRouter, Together, Groq). Routes:

  POST   /api/v1/byok-keys           -- add (or rotate) a provider key
  GET    /api/v1/byok-keys           -- list active provider keys
  DELETE /api/v1/byok-keys/{key_id}  -- revoke (soft-delete)
  GET    /api/v1/byok-keys/providers -- list of supported providers

All routes require dual auth (JWT or API key) via
``require_auth_or_api_key`` (PR-D4 helper) so customers can manage
keys from the dashboard OR via script. Account-scoped: account A
cannot read or revoke B's keys.

Plaintext keys never leave the request boundary -- on POST we
encrypt and store; on GET we return only ``key_prefix`` (8-char
display hint).
"""

from __future__ import annotations

import logging
import uuid as _uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth.dependencies import AuthUser, require_auth_or_api_key
from ..services.byok_keys import (
    SUPPORTED_PROVIDERS,
    BYOKKeyRecord,
    insert_provider_key,
    list_provider_keys,
    revoke_provider_key,
)
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.byok_keys")

router = APIRouter(prefix="/byok-keys", tags=["byok-keys"])


# ---- Request / response schemas -----------------------------------------


class AddBYOKKeyRequest(BaseModel):
    provider: str = Field(..., description="anthropic | openrouter | together | groq")
    raw_key: str = Field(..., min_length=8, max_length=4096)
    label: str = Field(default="", max_length=128)


class BYOKKeyView(BaseModel):
    """Display-safe view -- never carries the ciphertext or plaintext."""

    id: str
    provider: str
    key_prefix: str
    label: str
    added_at: str
    last_used_at: Optional[str] = None
    revoked_at: Optional[str] = None


def _record_to_view(record: BYOKKeyRecord) -> BYOKKeyView:
    def _fmt(value):
        if value is None:
            return None
        try:
            return value.isoformat()
        except AttributeError:
            return str(value)

    return BYOKKeyView(
        id=str(record.id),
        provider=record.provider,
        key_prefix=record.key_prefix,
        label=record.label,
        added_at=_fmt(record.added_at) or "",
        last_used_at=_fmt(record.last_used_at),
        revoked_at=_fmt(record.revoked_at),
    )


# ---- Routes --------------------------------------------------------------


@router.get("/providers", response_model=list[str])
async def list_supported_providers(
    _user: AuthUser = Depends(require_auth_or_api_key),
) -> list[str]:
    """The set of provider strings ``POST /byok-keys`` accepts. Used
    by the dashboard to render provider dropdowns."""
    return list(SUPPORTED_PROVIDERS)


@router.post("", response_model=BYOKKeyView, status_code=201)
async def add_key(
    body: AddBYOKKeyRequest,
    user: AuthUser = Depends(require_auth_or_api_key),
) -> BYOKKeyView:
    """Add (or rotate) a customer's provider key.

    If the customer already has an active key for this provider, the
    old row is revoked and a new one is inserted -- one transaction.
    The raw key is encrypted before persist; only ``key_prefix`` is
    ever readable later.
    """
    if body.provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Provider '{body.provider}' is not supported. "
                f"Allowed: {sorted(SUPPORTED_PROVIDERS)}."
            ),
        )

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    try:
        record = await insert_provider_key(
            pool,
            account_id=_uuid.UUID(user.account_id),
            provider=body.provider,
            raw_key=body.raw_key,
            label=body.label,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return _record_to_view(record)


@router.get("", response_model=list[BYOKKeyView])
async def list_keys(
    user: AuthUser = Depends(require_auth_or_api_key),
) -> list[BYOKKeyView]:
    """List active BYOK keys for the calling account. Display-safe;
    plaintext keys never returned."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    records = await list_provider_keys(pool, account_id=_uuid.UUID(user.account_id))
    return [_record_to_view(r) for r in records]


@router.delete("/{key_id}", status_code=204)
async def revoke_key(
    key_id: str,
    user: AuthUser = Depends(require_auth_or_api_key),
) -> None:
    """Revoke (soft-delete) a BYOK key. Account-scoped -- 404 when
    the caller does not own the key, to avoid leaking key existence
    across accounts."""
    try:
        key_uuid = _uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Key not found")

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    revoked = await revoke_provider_key(
        pool,
        key_id=key_uuid,
        account_id=_uuid.UUID(user.account_id),
    )
    if not revoked:
        raise HTTPException(status_code=404, detail="Key not found")
    return None
