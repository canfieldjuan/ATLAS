"""FastAPI router factory for generated Content Ops asset review."""

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
from ..landing_page_export import export_landing_page_drafts
from ..landing_page_postgres import PostgresLandingPageRepository
from ..report_export import export_report_drafts
from ..report_postgres import PostgresReportRepository
from ..sales_brief_export import export_sales_brief_drafts
from ..sales_brief_postgres import PostgresSalesBriefRepository


PoolProvider = Callable[[], Any | Awaitable[Any]]
ScopeProvider = Callable[[], TenantScope | Mapping[str, Any] | None | Awaitable[Any]]

ASSET_CHOICES = ("report", "landing_page", "sales_brief")


def _require_fastapi() -> None:
    if _FASTAPI_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "FastAPI is required to create generated asset API routes. "
        "Install fastapi in the host application environment."
    ) from _FASTAPI_IMPORT_ERROR


@dataclass(frozen=True)
class GeneratedAssetApiConfig:
    """Host-owned API defaults for generated asset review routes."""

    prefix: str = "/content-assets"
    tags: tuple[str, ...] = ("content-assets",)
    default_status: str | None = "draft"
    default_limit: int = 20
    max_limit: int = 200
    export_filename_prefix: str = "content_assets"

    def __post_init__(self) -> None:
        if self.default_limit < 0:
            raise ValueError("default_limit must be non-negative")
        if self.max_limit <= 0:
            raise ValueError("max_limit must be positive")
        if self.default_limit > self.max_limit:
            raise ValueError("default_limit must be less than or equal to max_limit")
        if not _clean(self.export_filename_prefix):
            raise ValueError("export_filename_prefix is required")


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


def _api_limit(value: int | None, config: GeneratedAssetApiConfig) -> int:
    limit = config.default_limit if value is None else int(value)
    if limit < 0:
        raise HTTPException(status_code=400, detail="limit must be non-negative")
    return min(limit, config.max_limit)


def _status_filter(value: str | None, config: GeneratedAssetApiConfig) -> str | None:
    if value is None:
        return config.default_status
    status = _clean(value)
    return status or None


def _review_status(value: Any) -> str:
    status = _clean(value)
    if not status:
        raise HTTPException(status_code=400, detail="status is required")
    return status


def _asset_arg(value: str) -> str:
    asset = _clean(value)
    if asset not in ASSET_CHOICES:
        raise HTTPException(
            status_code=400,
            detail=f"asset must be one of {', '.join(ASSET_CHOICES)}",
        )
    return asset


def _clean(value: Any) -> str:
    return str(value or "").strip()


def create_generated_asset_router(
    *,
    pool_provider: PoolProvider,
    scope_provider: ScopeProvider | None = None,
    config: GeneratedAssetApiConfig | None = None,
    dependencies: Sequence[Any] | None = None,
) -> APIRouter:
    """Create host-mounted generated asset review routes."""
    _require_fastapi()
    resolved_config = config or GeneratedAssetApiConfig()
    router = APIRouter(
        prefix=resolved_config.prefix,
        tags=list(resolved_config.tags),
        dependencies=list(dependencies or ()),
    )

    @router.get("/{asset}/drafts")
    async def list_drafts(
        asset: str,
        status: str | None = Query(None, description="Status filter. Empty string means all statuses."),
        target_mode: str | None = Query(None),
        report_type: str | None = Query(None),
        campaign_name: str | None = Query(None),
        slug: str | None = Query(None),
        brief_type: str | None = Query(None),
        limit: int | None = Query(None, ge=0),
    ) -> dict[str, Any]:
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        result = await _export_for_asset(
            _asset_arg(asset),
            pool,
            scope=scope,
            status=_status_filter(status, resolved_config),
            target_mode=target_mode,
            report_type=report_type,
            campaign_name=campaign_name,
            slug=slug,
            brief_type=brief_type,
            limit=_api_limit(limit, resolved_config),
        )
        return result.as_dict()

    @router.get("/{asset}/drafts/export", response_model=None)
    async def export_drafts(
        asset: str,
        status: str | None = Query(None, description="Status filter. Empty string means all statuses."),
        target_mode: str | None = Query(None),
        report_type: str | None = Query(None),
        campaign_name: str | None = Query(None),
        slug: str | None = Query(None),
        brief_type: str | None = Query(None),
        limit: int | None = Query(None, ge=0),
        format: str = Query("csv", description="csv or json"),
    ) -> Any:
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        asset_name = _asset_arg(asset)
        result = await _export_for_asset(
            asset_name,
            pool,
            scope=scope,
            status=_status_filter(status, resolved_config),
            target_mode=target_mode,
            report_type=report_type,
            campaign_name=campaign_name,
            slug=slug,
            brief_type=brief_type,
            limit=_api_limit(limit, resolved_config),
        )
        format_name = _clean(format).lower()
        if format_name == "json":
            return result.as_dict()
        if format_name != "csv":
            raise HTTPException(status_code=400, detail="format must be csv or json")
        filename = f"{_clean(resolved_config.export_filename_prefix)}_{asset_name}.csv"
        return Response(
            content=result.as_csv(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @router.post("/{asset}/drafts/review")
    async def review_draft(
        asset: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        tenant = _tenant_scope(scope)
        asset_name = _asset_arg(asset)
        asset_id = _clean(payload.get("id") or payload.get("asset_id"))
        if not asset_id:
            raise HTTPException(status_code=400, detail="id is required")
        status = _review_status(payload.get("status"))
        updated = await _update_asset_status(
            asset_name,
            pool,
            asset_id=asset_id,
            status=status,
            scope=tenant,
        )
        return {
            "account_id": tenant.account_id,
            "asset": asset_name,
            "id": asset_id,
            "status": status,
            "updated": bool(updated),
        }

    return router


async def _export_for_asset(
    asset: str,
    pool: Any,
    *,
    scope: TenantScope | Mapping[str, Any] | None,
    status: str | None,
    target_mode: str | None,
    report_type: str | None,
    campaign_name: str | None,
    slug: str | None,
    brief_type: str | None,
    limit: int,
) -> Any:
    if asset == "report":
        return await export_report_drafts(
            PostgresReportRepository(pool),
            scope=scope,
            status=status,
            target_mode=target_mode,
            report_type=report_type,
            limit=limit,
        )
    if asset == "landing_page":
        return await export_landing_page_drafts(
            PostgresLandingPageRepository(pool),
            scope=scope,
            status=status,
            campaign_name=campaign_name,
            slug=slug,
            limit=limit,
        )
    if asset == "sales_brief":
        return await export_sales_brief_drafts(
            PostgresSalesBriefRepository(pool),
            scope=scope,
            status=status,
            target_mode=target_mode,
            brief_type=brief_type,
            limit=limit,
        )
    raise HTTPException(status_code=400, detail=f"unsupported asset: {asset}")


async def _update_asset_status(
    asset: str,
    pool: Any,
    *,
    asset_id: str,
    status: str,
    scope: TenantScope,
) -> bool:
    if asset == "report":
        return await PostgresReportRepository(pool).update_status(asset_id, status, scope=scope)
    if asset == "landing_page":
        return await PostgresLandingPageRepository(pool).update_status(asset_id, status, scope=scope)
    if asset == "sales_brief":
        return await PostgresSalesBriefRepository(pool).update_status(asset_id, status, scope=scope)
    raise HTTPException(status_code=400, detail=f"unsupported asset: {asset}")


def _tenant_scope(value: TenantScope | Mapping[str, Any] | None) -> TenantScope:
    if isinstance(value, TenantScope):
        return value
    if isinstance(value, Mapping):
        return TenantScope(
            account_id=str(value.get("account_id") or "") or None,
            user_id=str(value.get("user_id") or "") or None,
        )
    return TenantScope()


__all__ = [
    "GeneratedAssetApiConfig",
    "create_generated_asset_router",
]
