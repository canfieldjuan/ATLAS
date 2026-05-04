"""FastAPI router factory for extracted B2B campaign draft review."""

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


PoolProvider = Callable[[], Any | Awaitable[Any]]
ScopeProvider = Callable[[], TenantScope | Mapping[str, Any] | None | Awaitable[Any]]


def _require_fastapi() -> None:
    if _FASTAPI_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "FastAPI is required to create B2B campaign API routes. "
        "Install fastapi in the host application environment."
    ) from _FASTAPI_IMPORT_ERROR


@dataclass(frozen=True)
class B2BCampaignApiConfig:
    """Host-owned API defaults for B2B campaign review routes."""

    prefix: str = "/b2b/campaigns"
    tags: tuple[str, ...] = ("b2b-campaigns",)
    campaign_table: str = "b2b_campaigns"
    default_statuses: tuple[str, ...] = ("draft",)
    default_limit: int = 20
    max_limit: int = 200
    export_filename: str = "campaign_drafts.csv"

    def __post_init__(self) -> None:
        if self.default_limit < 0:
            raise ValueError("default_limit must be non-negative")
        if self.max_limit <= 0:
            raise ValueError("max_limit must be positive")
        if self.default_limit > self.max_limit:
            raise ValueError("default_limit must be less than or equal to max_limit")
        if not _clean(self.export_filename):
            raise ValueError("export_filename is required")


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


def _api_limit(value: int | None, config: B2BCampaignApiConfig) -> int:
    limit = config.default_limit if value is None else int(value)
    if limit < 0:
        raise HTTPException(status_code=400, detail="limit must be non-negative")
    return min(limit, config.max_limit)


def _bad_request(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _clean(value: Any) -> str:
    return str(value or "").strip()


def create_b2b_campaign_router(
    *,
    pool_provider: PoolProvider,
    scope_provider: ScopeProvider | None = None,
    config: B2BCampaignApiConfig | None = None,
    dependencies: Sequence[Any] | None = None,
) -> APIRouter:
    """Create host-mounted B2B campaign draft review routes."""
    _require_fastapi()
    resolved_config = config or B2BCampaignApiConfig()
    router = APIRouter(
        prefix=resolved_config.prefix,
        tags=list(resolved_config.tags),
        dependencies=list(dependencies or ()),
    )

    @router.get("/drafts")
    async def list_drafts(
        statuses: str | None = Query(None, description="Comma-separated statuses."),
        target_mode: str | None = Query(None),
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
                target_mode=target_mode,
                channel=channel,
                vendor_name=vendor_name,
                company_name=company_name,
                limit=_api_limit(limit, resolved_config),
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc
        return result.as_dict()

    @router.get("/drafts/export", response_model=None)
    async def export_drafts(
        statuses: str | None = Query(None, description="Comma-separated statuses."),
        target_mode: str | None = Query(None),
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
                target_mode=target_mode,
                channel=channel,
                vendor_name=vendor_name,
                company_name=company_name,
                limit=_api_limit(limit, resolved_config),
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc

        format_name = _clean(format).lower()
        if format_name == "json":
            return result.as_dict()
        if format_name != "csv":
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

    @router.post("/drafts/review")
    async def review_drafts(
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        try:
            result = await review_campaign_drafts(
                pool,
                campaign_ids=_parse_payload_list(payload.get("campaign_ids")),
                status=_clean(payload.get("status")) or "approved",
                scope=scope,
                campaign_table=resolved_config.campaign_table,
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
    "B2BCampaignApiConfig",
    "create_b2b_campaign_router",
]
