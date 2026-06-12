"""Tenant-scoped admin CRUD for the Content Ops calibration library.

The marketer verify MCP surface is deliberately verify-only, so curation of a
tenant's calibration anchors lives here as an authenticated admin API, mirroring
the brand-voice-profiles and claim-registry admin pattern. The verify flow reads
these rows server-side via ``ContentOpsCalibrationLibraryRepository``.
"""

from __future__ import annotations

import uuid as _uuid
from inspect import isawaitable
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .._content_ops_calibration_library import (
    ContentOpsCalibrationLibraryRecord,
    archive_calibration_example,
    create_calibration_example,
    list_calibration_example_records,
    update_calibration_example,
)
from ..auth.dependencies import AuthUser

PoolProvider = Callable[[], Any | Awaitable[Any]]
AuthDependency = Callable[..., AuthUser | Awaitable[AuthUser]]


class UpsertCalibrationExampleRequest(BaseModel):
    example_id: str = Field(..., min_length=1, max_length=160)
    label: str = Field(..., min_length=1, max_length=40)
    excerpt: str = Field(..., min_length=1, max_length=4000)
    reasoning: str = Field(..., min_length=1, max_length=4000)
    source: str = Field(default="curated", max_length=80)
    metadata: dict[str, object] = Field(default_factory=dict)


class CalibrationExampleView(BaseModel):
    id: str
    account_id: str
    example_id: str
    label: str
    excerpt: str
    reasoning: str
    source: str
    metadata: dict[str, object]
    created_at: str
    updated_at: str
    archived_at: str | None = None


def _default_pool_provider() -> Any:
    from ..storage.database import get_db_pool

    return get_db_pool()


def create_content_ops_calibration_library_router(
    *,
    pool_provider: PoolProvider = _default_pool_provider,
    auth_dependency: AuthDependency,
) -> APIRouter:
    """Build the tenant-scoped calibration-library admin router."""

    router = APIRouter(
        prefix="/content-ops/calibration-library",
        tags=["content-ops"],
    )

    @router.get("", response_model=list[CalibrationExampleView])
    async def list_examples(
        user: AuthUser = Depends(auth_dependency),
    ) -> list[CalibrationExampleView]:
        account_id = _account_uuid(user)
        pool = await _resolve_ready_pool(pool_provider)
        records = await list_calibration_example_records(pool, account_id=account_id)
        return [_record_to_view(record) for record in records]

    @router.post("", response_model=CalibrationExampleView, status_code=201)
    async def add_example(
        body: UpsertCalibrationExampleRequest,
        user: AuthUser = Depends(auth_dependency),
    ) -> CalibrationExampleView:
        _require_calibration_admin(user)
        account_id = _account_uuid(user)
        pool = await _resolve_ready_pool(pool_provider)
        try:
            record = await create_calibration_example(
                pool, account_id=account_id, payload=body.model_dump()
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - translate DB unique conflict
            if exc.__class__.__name__ == "UniqueViolationError":
                raise HTTPException(
                    status_code=409,
                    detail="A calibration example with this id already exists",
                )
            raise
        return _record_to_view(record)

    @router.put("/{row_id}", response_model=CalibrationExampleView)
    async def update_example(
        row_id: str,
        body: UpsertCalibrationExampleRequest,
        user: AuthUser = Depends(auth_dependency),
    ) -> CalibrationExampleView:
        _require_calibration_admin(user)
        account_id = _account_uuid(user)
        pool = await _resolve_ready_pool(pool_provider)
        try:
            record = await update_calibration_example(
                pool,
                account_id=account_id,
                example_row_id=_row_uuid(row_id),
                payload=body.model_dump(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - translate DB unique conflict
            if exc.__class__.__name__ == "UniqueViolationError":
                raise HTTPException(
                    status_code=409,
                    detail="A calibration example with this id already exists",
                )
            raise
        if record is None:
            raise HTTPException(status_code=404, detail="Calibration example not found")
        return _record_to_view(record)

    @router.delete("/{row_id}", status_code=204)
    async def delete_example(
        row_id: str,
        user: AuthUser = Depends(auth_dependency),
    ) -> None:
        _require_calibration_admin(user)
        account_id = _account_uuid(user)
        pool = await _resolve_ready_pool(pool_provider)
        archived = await archive_calibration_example(
            pool, account_id=account_id, example_row_id=_row_uuid(row_id)
        )
        if not archived:
            raise HTTPException(status_code=404, detail="Calibration example not found")

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
        raise HTTPException(status_code=404, detail="Calibration example not found")


def _require_calibration_admin(user: AuthUser) -> None:
    role = str(getattr(user, "role", "") or "").strip().lower()
    if bool(getattr(user, "is_admin", False)) or role in {"owner", "admin"}:
        return
    raise HTTPException(status_code=403, detail="Admin access required")


def _record_to_view(record: ContentOpsCalibrationLibraryRecord) -> CalibrationExampleView:
    return CalibrationExampleView(
        id=str(record.id),
        account_id=str(record.account_id),
        example_id=record.example_id,
        label=record.label,
        excerpt=record.excerpt,
        reasoning=record.reasoning,
        source=record.source,
        metadata=dict(record.metadata),
        created_at=_fmt_time(record.created_at) or "",
        updated_at=_fmt_time(record.updated_at) or "",
        archived_at=_fmt_time(record.archived_at),
    )


def _fmt_time(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)
