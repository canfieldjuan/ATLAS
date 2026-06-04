"""Content Ops brand voice profile management routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import uuid as _uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .._content_ops_brand_voice_profiles import (
    ContentOpsBrandVoiceProfileRecord,
    archive_brand_voice_profile,
    create_brand_voice_profile,
    list_brand_voice_profiles,
    update_brand_voice_profile,
)
from ..auth.dependencies import AuthUser


PoolProvider = Callable[[], Any | Awaitable[Any]]
AuthDependency = Callable[..., AuthUser | Awaitable[AuthUser]]


class UpsertBrandVoiceProfileRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=160)
    descriptors: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    exemplars: tuple[str, ...] = Field(default_factory=tuple, max_length=3)
    banned_terms: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    preferred_pov: str | None = Field(default=None, max_length=40)
    reading_level: str | None = Field(default=None, max_length=80)
    metadata: dict[str, object] = Field(default_factory=dict)


class BrandVoiceProfileView(BaseModel):
    id: str
    account_id: str
    name: str
    descriptors: list[str]
    exemplars: list[str]
    banned_terms: list[str]
    preferred_pov: str | None = None
    reading_level: str | None = None
    metadata: dict[str, object]
    created_at: str
    updated_at: str
    archived_at: str | None = None


def _default_pool_provider() -> Any:
    from ..storage.database import get_db_pool

    return get_db_pool()


def create_content_ops_brand_voice_profiles_router(
    *,
    pool_provider: PoolProvider = _default_pool_provider,
    auth_dependency: AuthDependency,
) -> APIRouter:
    """Create tenant-scoped Content Ops brand voice profile routes."""

    router = APIRouter(
        prefix="/content-ops/brand-voice-profiles",
        tags=["content-ops"],
    )

    @router.get("", response_model=list[BrandVoiceProfileView])
    async def list_profiles(
        user: AuthUser = Depends(auth_dependency),
    ) -> list[BrandVoiceProfileView]:
        pool = await _resolve_ready_pool(pool_provider)
        account_id = _account_uuid(user)
        records = await list_brand_voice_profiles(pool, account_id=account_id)
        return [_record_to_view(record) for record in records]

    @router.post("", response_model=BrandVoiceProfileView, status_code=201)
    async def add_profile(
        body: UpsertBrandVoiceProfileRequest,
        user: AuthUser = Depends(auth_dependency),
    ) -> BrandVoiceProfileView:
        _require_profile_admin(user)
        pool = await _resolve_ready_pool(pool_provider)
        account_id = _account_uuid(user)
        try:
            record = await create_brand_voice_profile(
                pool,
                account_id=account_id,
                payload=body.model_dump(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            if exc.__class__.__name__ in ("UniqueViolationError", "IntegrityError"):
                raise HTTPException(
                    status_code=409,
                    detail="A brand voice profile with that name already exists.",
                )
            raise
        return _record_to_view(record)

    @router.put("/{profile_id}", response_model=BrandVoiceProfileView)
    async def update_profile(
        profile_id: str,
        body: UpsertBrandVoiceProfileRequest,
        user: AuthUser = Depends(auth_dependency),
    ) -> BrandVoiceProfileView:
        _require_profile_admin(user)
        pool = await _resolve_ready_pool(pool_provider)
        account_id = _account_uuid(user)
        parsed_profile_id = _profile_uuid(profile_id)
        try:
            record = await update_brand_voice_profile(
                pool,
                account_id=account_id,
                profile_id=parsed_profile_id,
                payload=body.model_dump(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            if exc.__class__.__name__ in ("UniqueViolationError", "IntegrityError"):
                raise HTTPException(
                    status_code=409,
                    detail="A brand voice profile with that name already exists.",
                )
            raise
        if record is None:
            raise HTTPException(status_code=404, detail="Brand voice profile not found")
        return _record_to_view(record)

    @router.delete("/{profile_id}", status_code=204)
    async def delete_profile(
        profile_id: str,
        user: AuthUser = Depends(auth_dependency),
    ) -> None:
        _require_profile_admin(user)
        pool = await _resolve_ready_pool(pool_provider)
        account_id = _account_uuid(user)
        archived = await archive_brand_voice_profile(
            pool,
            account_id=account_id,
            profile_id=_profile_uuid(profile_id),
        )
        if not archived:
            raise HTTPException(status_code=404, detail="Brand voice profile not found")
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


def _profile_uuid(profile_id: str) -> _uuid.UUID:
    try:
        return _uuid.UUID(str(profile_id))
    except (TypeError, ValueError):
        raise HTTPException(status_code=404, detail="Brand voice profile not found")


def _require_profile_admin(user: AuthUser) -> None:
    role = str(getattr(user, "role", "") or "").strip().lower()
    if bool(getattr(user, "is_admin", False)) or role in {"owner", "admin"}:
        return
    raise HTTPException(status_code=403, detail="Admin access required")


def _record_to_view(record: ContentOpsBrandVoiceProfileRecord) -> BrandVoiceProfileView:
    return BrandVoiceProfileView(
        id=str(record.id),
        account_id=str(record.account_id),
        name=record.name,
        descriptors=list(record.descriptors),
        exemplars=list(record.exemplars),
        banned_terms=list(record.banned_terms),
        preferred_pov=record.preferred_pov,
        reading_level=record.reading_level,
        metadata=dict(record.metadata),
        created_at=_fmt_time(record.created_at) or "",
        updated_at=_fmt_time(record.updated_at) or "",
        archived_at=_fmt_time(record.archived_at),
    )


def _fmt_time(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


__all__ = [
    "BrandVoiceProfileView",
    "UpsertBrandVoiceProfileRequest",
    "create_content_ops_brand_voice_profiles_router",
]
