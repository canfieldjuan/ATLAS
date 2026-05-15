"""FastAPI router factory for campaign reasoning context administration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:
    from fastapi import APIRouter, Body, HTTPException, Query
except ImportError as exc:  # pragma: no cover - exercised in dependency-light CI.
    APIRouter = None
    Body = None
    HTTPException = None
    Query = None
    _FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

from ..campaign_ports import TenantScope
from ..campaign_reasoning_postgres import PostgresCampaignReasoningContextRepository


PoolProvider = Callable[[], Any | Awaitable[Any]]
ScopeProvider = Callable[[], TenantScope | Mapping[str, Any] | None | Awaitable[Any]]

_MATCH_KEYS = ("target_id", "id", "company", "company_name", "account", "account_name", "email", "contact_email", "vendor", "vendor_name")


def _require_fastapi() -> None:
    if _FASTAPI_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "FastAPI is required to create campaign reasoning context API routes. "
        "Install fastapi in the host application environment."
    ) from _FASTAPI_IMPORT_ERROR


@dataclass(frozen=True)
class ReasoningContextAdminApiConfig:
    prefix: str = "/campaign-reasoning-contexts"
    tags: tuple[str, ...] = ("campaign-reasoning-contexts",)
    table: str = "campaign_reasoning_contexts"
    default_limit: int = 20
    max_limit: int = 200

    def __post_init__(self) -> None:
        if self.default_limit < 0:
            raise ValueError("default_limit must be non-negative")
        if self.max_limit <= 0:
            raise ValueError("max_limit must be positive")
        if self.default_limit > self.max_limit:
            raise ValueError("default_limit must be less than or equal to max_limit")


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


async def _resolve_scope(scope_provider: ScopeProvider | None) -> TenantScope:
    if scope_provider is None:
        return TenantScope()
    scope = await _maybe_await(scope_provider())
    if scope is None:
        return TenantScope()
    if isinstance(scope, TenantScope):
        return scope
    if isinstance(scope, Mapping):
        return TenantScope(
            account_id=str(scope.get("account_id") or "") or None,
            user_id=str(scope.get("user_id") or "") or None,
        )
    raise HTTPException(status_code=500, detail="Invalid tenant scope")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _api_limit(value: int | None, config: ReasoningContextAdminApiConfig) -> int:
    limit = config.default_limit if value is None else int(value)
    if limit < 0:
        raise HTTPException(status_code=400, detail="limit must be non-negative")
    return min(limit, config.max_limit)


def _clean_values(values: Sequence[Any]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            cleaned.extend(_clean_values(value))
            continue
        text = _clean(value)
        if text:
            cleaned.append(text)
    return tuple(cleaned)


def _row_selectors(row: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[Any] = [row.get("selectors")]
    values.extend(row.get(key) for key in _MATCH_KEYS)
    return _clean_values(values)


def _row_context(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("context")
    return value if isinstance(value, Mapping) else {}


def create_reasoning_context_admin_router(
    *,
    pool_provider: PoolProvider,
    scope_provider: ScopeProvider | None = None,
    config: ReasoningContextAdminApiConfig | None = None,
    dependencies: Sequence[Any] | None = None,
    repository_factory: Callable[[Any, str], Any] | None = None,
) -> APIRouter:
    """Create host-mounted reasoning context admin routes.

    Pass auth dependencies from the host application. The default leaves the
    router unprotected for private mounts and tests.
    """

    _require_fastapi()
    resolved_config = config or ReasoningContextAdminApiConfig()
    repo_factory = repository_factory or (
        lambda pool, table: PostgresCampaignReasoningContextRepository(pool=pool, table=table)
    )
    router = APIRouter(
        prefix=resolved_config.prefix,
        tags=list(resolved_config.tags),
        dependencies=list(dependencies or ()),
    )

    @router.get("")
    async def list_contexts(
        target_mode: str | None = Query(None),
        selector: list[str] | None = Query(None),
        limit: int | None = Query(None, ge=0),
    ) -> dict[str, Any]:
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        result = await repo_factory(pool, resolved_config.table).list_contexts(
            scope=scope if scope.account_id else None,
            target_mode=target_mode,
            selectors=tuple(selector or ()),
            limit=_api_limit(limit, resolved_config),
        )
        return result.as_dict()

    @router.post("")
    async def upsert_context(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        account_id = scope.account_id or _clean(payload.get("account_id"))
        target_mode = _clean(payload.get("target_mode")).lower()
        selectors = _row_selectors(payload)
        if not selectors:
            raise HTTPException(status_code=400, detail="selectors are required")
        context = _row_context(payload)
        if not context:
            raise HTTPException(status_code=400, detail="context is required")
        context_id = await repo_factory(pool, resolved_config.table).save_context(
            scope=TenantScope(account_id=account_id or "", user_id=scope.user_id),
            selectors=selectors,
            context=context,
            target_mode=target_mode,
        )
        return {
            "status": "ok",
            "id": context_id,
            "account_id": account_id or "",
            "target_mode": target_mode,
            "selectors": list(selectors),
        }

    return router
