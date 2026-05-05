"""FastAPI router factory for extracted Amazon seller campaign workflows."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:
    from fastapi import APIRouter, Body, HTTPException, Query, Response
except ImportError as exc:  # pragma: no cover - exercised in dependency-light CI.
    APIRouter = None
    Body = None
    HTTPException = None
    Query = None
    Response = Any
    _FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

from ..campaign_ports import TenantScope
from ..campaign_postgres_export import list_campaign_drafts
from ..campaign_postgres_review import review_campaign_drafts
from ..campaign_postgres_seller_targets import (
    create_seller_target,
    delete_seller_target,
    get_seller_target,
    list_seller_targets,
    update_seller_target,
)


PoolProvider = Callable[[], Any | Awaitable[Any]]
ScopeProvider = Callable[[], TenantScope | Mapping[str, Any] | None | Awaitable[Any]]


@dataclass(frozen=True)
class SellerCampaignApiConfig:
    """Host-owned API defaults for Amazon seller campaign routes."""

    prefix: str = "/seller"
    tags: tuple[str, ...] = ("seller-campaigns",)
    campaign_table: str = "b2b_campaigns"
    seller_targets_table: str = "seller_targets"
    target_mode: str = "amazon_seller"
    default_statuses: tuple[str, ...] = ("draft",)
    default_limit: int = 50
    max_limit: int = 200
    export_filename: str = "seller_campaign_drafts.csv"

    def __post_init__(self) -> None:
        if self.default_limit < 0:
            raise ValueError("default_limit must be non-negative")
        if self.max_limit <= 0:
            raise ValueError("max_limit must be positive")
        if self.default_limit > self.max_limit:
            raise ValueError("default_limit must be less than or equal to max_limit")
        if not _clean(self.target_mode):
            raise ValueError("target_mode is required")
        if not _clean(self.export_filename):
            raise ValueError("export_filename is required")


def _require_fastapi() -> None:
    if _FASTAPI_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "FastAPI is required to create seller campaign API routes. "
        "Install fastapi in the host application environment."
    ) from _FASTAPI_IMPORT_ERROR


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _resolve_pool(pool_provider: PoolProvider) -> Any:
    pool = await _maybe_await(pool_provider())
    if pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    if getattr(pool, "is_initialized", True) is False:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return pool


async def _resolve_scope(
    scope_provider: ScopeProvider | None,
) -> TenantScope | Mapping[str, Any] | None:
    if scope_provider is None:
        return None
    scope = await _maybe_await(scope_provider())
    if scope is None or isinstance(scope, (TenantScope, Mapping)):
        return scope
    raise HTTPException(status_code=500, detail="Invalid tenant scope")


def _parse_csv_values(
    value: str | Sequence[str] | None,
    *,
    default: Sequence[str] = (),
) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        raw_values = value.split(",")
    else:
        raw_values = []
        for item in value:
            raw_values.extend(str(item or "").split(","))
    cleaned = tuple(item for item in (_clean(item) for item in raw_values) if item)
    return cleaned or tuple(default)


def _parse_payload_list(value: Any, *, default: Sequence[str] = ()) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return _parse_csv_values(value, default=default)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return _parse_csv_values([str(item or "") for item in value], default=default)
    return tuple(default)


def _api_limit(value: int | None, config: SellerCampaignApiConfig) -> int:
    limit = config.default_limit if value is None else int(value)
    if limit < 0:
        raise HTTPException(status_code=400, detail="limit must be non-negative")
    return min(limit, config.max_limit)


def _api_offset(value: int | None) -> int:
    offset = 0 if value is None else int(value)
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be non-negative")
    return offset


def _bad_request(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=404, detail=detail)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def create_seller_campaign_router(
    *,
    pool_provider: PoolProvider,
    scope_provider: ScopeProvider | None = None,
    config: SellerCampaignApiConfig | None = None,
    dependencies: Sequence[Any] | None = None,
) -> APIRouter:
    """Create host-mounted Amazon seller campaign routes."""
    _require_fastapi()
    resolved_config = config or SellerCampaignApiConfig()
    router = APIRouter(
        prefix=resolved_config.prefix,
        tags=list(resolved_config.tags),
        dependencies=list(dependencies or ()),
    )

    @router.get("/targets")
    async def list_targets(
        status: str | None = Query(None),
        seller_type: str | None = Query(None),
        category: str | None = Query(None),
        limit: int | None = Query(None, ge=0),
        offset: int | None = Query(None, ge=0),
    ) -> dict[str, Any]:
        pool = await _resolve_pool(pool_provider)
        try:
            result = await list_seller_targets(
                pool,
                status=status,
                seller_type=seller_type,
                category=category,
                limit=_api_limit(limit, resolved_config),
                offset=_api_offset(offset),
                seller_targets_table=resolved_config.seller_targets_table,
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc
        return result.as_dict()

    @router.post("/targets")
    async def create_target(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        pool = await _resolve_pool(pool_provider)
        try:
            return await create_seller_target(
                pool,
                seller_name=payload.get("seller_name"),
                company_name=payload.get("company_name"),
                email=payload.get("email"),
                seller_type=_clean(payload.get("seller_type")) or "private_label",
                categories=_parse_payload_list(payload.get("categories")),
                storefront_url=payload.get("storefront_url"),
                notes=payload.get("notes"),
                source=_clean(payload.get("source")) or "manual",
                seller_targets_table=resolved_config.seller_targets_table,
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc

    @router.get("/targets/{target_id}")
    async def get_target(target_id: str) -> dict[str, Any]:
        pool = await _resolve_pool(pool_provider)
        try:
            row = await get_seller_target(
                pool,
                target_id=target_id,
                seller_targets_table=resolved_config.seller_targets_table,
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc
        if row is None:
            raise _not_found("Seller target not found")
        return row

    @router.patch("/targets/{target_id}")
    async def update_target(
        target_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        pool = await _resolve_pool(pool_provider)
        try:
            row = await update_seller_target(
                pool,
                target_id=target_id,
                values=payload,
                seller_targets_table=resolved_config.seller_targets_table,
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc
        if row is None:
            raise _not_found("Seller target not found")
        return row

    @router.delete("/targets/{target_id}")
    async def delete_target(target_id: str) -> dict[str, bool]:
        pool = await _resolve_pool(pool_provider)
        try:
            deleted = await delete_seller_target(
                pool,
                target_id=target_id,
                seller_targets_table=resolved_config.seller_targets_table,
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc
        if not deleted:
            raise _not_found("Seller target not found")
        return {"ok": True}

    @router.get("/campaigns/drafts")
    async def list_drafts(
        statuses: str | None = Query(None, description="Comma-separated statuses."),
        channel: str | None = Query(None),
        vendor_name: str | None = Query(None),
        company_name: str | None = Query(None),
        limit: int | None = Query(None, ge=0),
    ) -> dict[str, Any]:
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        try:
            result = await list_campaign_drafts(
                pool,
                scope=scope,
                campaign_table=resolved_config.campaign_table,
                statuses=_parse_csv_values(
                    statuses,
                    default=resolved_config.default_statuses,
                ),
                target_mode=resolved_config.target_mode,
                channel=channel,
                vendor_name=vendor_name,
                company_name=company_name,
                limit=_api_limit(limit, resolved_config),
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc
        return result.as_dict()

    @router.get("/campaigns/drafts/export", response_model=None)
    async def export_drafts(
        statuses: str | None = Query(None, description="Comma-separated statuses."),
        channel: str | None = Query(None),
        vendor_name: str | None = Query(None),
        company_name: str | None = Query(None),
        limit: int | None = Query(None, ge=0),
        format: str = Query("csv", description="csv or json"),
    ) -> Any:
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        try:
            result = await list_campaign_drafts(
                pool,
                scope=scope,
                campaign_table=resolved_config.campaign_table,
                statuses=_parse_csv_values(
                    statuses,
                    default=resolved_config.default_statuses,
                ),
                target_mode=resolved_config.target_mode,
                channel=channel,
                vendor_name=vendor_name,
                company_name=company_name,
                limit=_api_limit(limit, resolved_config),
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc
        if _clean(format).lower() == "json":
            return result.as_dict()
        if _clean(format).lower() != "csv":
            raise HTTPException(status_code=400, detail="format must be csv or json")
        return Response(
            content=result.as_csv(),
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{_clean(resolved_config.export_filename)}"'
                ),
            },
        )

    @router.post("/campaigns/drafts/review")
    async def review_drafts(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        try:
            result = await review_campaign_drafts(
                pool,
                campaign_ids=_parse_payload_list(payload.get("campaign_ids")),
                status=_clean(payload.get("status")) or "approved",
                scope=scope,
                campaign_table=resolved_config.campaign_table,
                target_mode=resolved_config.target_mode,
                from_statuses=_parse_payload_list(
                    payload.get("from_statuses"),
                    default=("draft",),
                ),
                from_email=_clean(payload.get("from_email")) or None,
                reason=_clean(payload.get("reason")) or None,
                reviewed_by=_clean(payload.get("reviewed_by")) or None,
                metadata=payload.get("metadata")
                if isinstance(payload.get("metadata"), Mapping)
                else None,
                dry_run=bool(payload.get("dry_run")),
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc
        return result.as_dict()

    return router


__all__ = [
    "SellerCampaignApiConfig",
    "create_seller_campaign_router",
]
