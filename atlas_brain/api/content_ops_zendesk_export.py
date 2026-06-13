"""Tenant-scoped Zendesk full-thread export routes for Content Ops."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
import logging
import uuid as _uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
import httpx
from pydantic import BaseModel, Field

from extracted_content_pipeline.faq_macro_writeback_zendesk import (
    ZendeskMacroCredentials,
)
from extracted_content_pipeline.support_ticket_zendesk_export import (
    DEFAULT_ZENDESK_EXPORT_LIMIT,
    MAX_ZENDESK_EXPORT_LIMIT,
    ZendeskTicketExportError,
    export_zendesk_full_thread_artifact,
)

from .._content_ops_zendesk_credentials import lookup_zendesk_credentials
from ..auth.dependencies import AuthUser


logger = logging.getLogger("atlas.content_ops_zendesk_export")

PoolProvider = Callable[[], Any | Awaitable[Any]]
AuthDependency = Callable[..., AuthUser | Awaitable[AuthUser]]
CredentialLookup = Callable[..., ZendeskMacroCredentials | None | Awaitable[ZendeskMacroCredentials | None]]
ZendeskExporter = Callable[..., Mapping[str, Any] | Awaitable[Mapping[str, Any]]]


class ZendeskFullThreadExportRequest(BaseModel):
    limit: int = Field(
        default=DEFAULT_ZENDESK_EXPORT_LIMIT,
        ge=1,
        le=MAX_ZENDESK_EXPORT_LIMIT,
    )
    start_time: int = Field(default=0, ge=0)


class ZendeskFullThreadExportResponse(BaseModel):
    importer_mode: str
    support_platform: str
    ticket_count: int
    limit: int
    start_time: int
    artifact: dict[str, object]


def _default_pool_provider() -> Any:
    from ..storage.database import get_db_pool

    return get_db_pool()


def create_content_ops_zendesk_export_router(
    *,
    pool_provider: PoolProvider = _default_pool_provider,
    auth_dependency: AuthDependency,
    credential_lookup: CredentialLookup = lookup_zendesk_credentials,
    exporter: ZendeskExporter = export_zendesk_full_thread_artifact,
) -> APIRouter:
    """Create tenant-scoped Zendesk full-thread export routes."""

    router = APIRouter(
        prefix="/content-ops/zendesk-export",
        tags=["content-ops"],
    )

    @router.post("/full-thread", response_model=ZendeskFullThreadExportResponse)
    async def export_full_thread(
        body: ZendeskFullThreadExportRequest,
        user: AuthUser = Depends(auth_dependency),
    ) -> ZendeskFullThreadExportResponse:
        pool = await _resolve_ready_pool(pool_provider)
        account_id = _account_uuid(user)
        credentials = await _lookup_credentials(
            credential_lookup,
            pool,
            account_id=account_id,
        )
        if credentials is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "reason": "zendesk_credentials_missing",
                    "message": "Zendesk credentials are not configured for this tenant.",
                },
            )

        try:
            artifact = await _maybe_await(exporter(
                credentials,
                limit=body.limit,
                start_time=body.start_time,
            ))
        except ZendeskTicketExportError as exc:
            raise _export_error_to_http(exc) from exc
        except httpx.RequestError as exc:
            logger.warning(
                "zendesk export transport failed account_id=%s",
                account_id,
                extra={"error_type": type(exc).__name__},
            )
            raise HTTPException(
                status_code=502,
                detail={"reason": "zendesk_export_unavailable"},
            ) from exc

        artifact_dict = _artifact_mapping(artifact)
        return ZendeskFullThreadExportResponse(
            importer_mode="full_thread",
            support_platform="zendesk",
            ticket_count=_ticket_count(artifact_dict),
            limit=body.limit,
            start_time=body.start_time,
            artifact=dict(artifact_dict),
        )

    return router


async def _resolve_ready_pool(pool_provider: PoolProvider) -> Any:
    pool = pool_provider()
    pool = await _maybe_await(pool)
    if pool is None or getattr(pool, "is_initialized", True) is False:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


def _account_uuid(user: AuthUser) -> _uuid.UUID:
    try:
        return _uuid.UUID(str(user.account_id))
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid tenant scope")


async def _lookup_credentials(
    credential_lookup: CredentialLookup,
    pool: Any,
    *,
    account_id: _uuid.UUID,
) -> ZendeskMacroCredentials | None:
    try:
        return await _maybe_await(credential_lookup(pool, account_id=account_id))
    except Exception as exc:
        logger.exception(
            "zendesk export credential lookup failed account_id=%s",
            account_id,
        )
        raise HTTPException(
            status_code=503,
            detail={"reason": "zendesk_credentials_unavailable"},
        ) from exc


def _export_error_to_http(exc: ZendeskTicketExportError) -> HTTPException:
    detail: dict[str, Any] = {"reason": exc.message}
    if exc.status_code is not None:
        detail["zendesk_status_code"] = exc.status_code
    status_code = 404 if exc.message == "zendesk_credentials_missing" else 502
    return HTTPException(status_code=status_code, detail=detail)


def _artifact_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HTTPException(
            status_code=502,
            detail={"reason": "zendesk_export_artifact_invalid"},
        )
    tickets = value.get("tickets")
    if not isinstance(tickets, Sequence) or isinstance(tickets, (str, bytes, bytearray)):
        raise HTTPException(
            status_code=502,
            detail={"reason": "zendesk_export_artifact_invalid"},
        )
    return value


def _ticket_count(artifact: Mapping[str, Any]) -> int:
    tickets = artifact.get("tickets")
    if isinstance(tickets, Sequence) and not isinstance(tickets, (str, bytes, bytearray)):
        return len(tickets)
    return 0


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


__all__ = [
    "ZendeskFullThreadExportRequest",
    "ZendeskFullThreadExportResponse",
    "create_content_ops_zendesk_export_router",
]
