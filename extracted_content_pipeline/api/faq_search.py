"""FastAPI router factory for ticket FAQ deflection search."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
import asyncio
from dataclasses import dataclass
import logging
from typing import Any
from uuid import UUID

try:
    from fastapi import APIRouter, HTTPException, Query
except ImportError as exc:  # pragma: no cover - exercised in dependency-light CI.
    APIRouter = None
    HTTPException = None
    Query = None
    _FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

from ..campaign_ports import TenantScope
from ..ticket_faq_ports import TicketFAQDraft
from ..ticket_faq_postgres import PostgresTicketFAQRepository
from ..ticket_faq_search import PostgresTicketFAQSearchRepository


PoolProvider = Callable[[], Any | Awaitable[Any]]
ScopeProvider = Callable[[], TenantScope | Mapping[str, Any] | None | Awaitable[Any]]
SearchRepositoryFactory = Callable[[Any], PostgresTicketFAQSearchRepository]
FAQRepositoryFactory = Callable[[Any], PostgresTicketFAQRepository]
logger = logging.getLogger(__name__)


def _require_fastapi() -> None:
    if _FASTAPI_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "FastAPI is required to create FAQ deflection search routes. "
        "Install fastapi in the host application environment."
    ) from _FASTAPI_IMPORT_ERROR


@dataclass(frozen=True)
class FAQDeflectionSearchApiConfig:
    """Host-owned defaults for the FAQ deflection search route."""

    prefix: str = "/content-ops/faq-deflection-search"
    tags: tuple[str, ...] = ("content-ops", "faq-search")
    default_status: str | None = "approved"
    default_limit: int = 5
    max_limit: int = 10
    max_query_chars: int = 300
    search_timeout_seconds: float | None = 5.0

    def __post_init__(self) -> None:
        if self.default_limit <= 0:
            raise ValueError("default_limit must be positive")
        if self.max_limit <= 0:
            raise ValueError("max_limit must be positive")
        if self.default_limit > self.max_limit:
            raise ValueError("default_limit must be less than or equal to max_limit")
        if self.max_query_chars <= 0:
            raise ValueError("max_query_chars must be positive")
        if self.search_timeout_seconds is not None and self.search_timeout_seconds <= 0:
            raise ValueError("search_timeout_seconds must be positive")


def create_faq_deflection_search_router(
    *,
    pool_provider: PoolProvider,
    scope_provider: ScopeProvider | None = None,
    repository_factory: SearchRepositoryFactory = PostgresTicketFAQSearchRepository,
    faq_repository_factory: FAQRepositoryFactory = PostgresTicketFAQRepository,
    config: FAQDeflectionSearchApiConfig | None = None,
    dependencies: Sequence[Any] | None = None,
) -> APIRouter:
    """Create host-mounted FAQ deflection search routes."""

    _require_fastapi()
    resolved_config = config or FAQDeflectionSearchApiConfig()
    router = APIRouter(
        prefix=resolved_config.prefix,
        tags=list(resolved_config.tags),
        dependencies=list(dependencies or ()),
    )

    @router.get("")
    async def search_faq_deflection(
        q: str = Query(..., description="Customer question or search text."),
        corpus_id: str | None = Query(None),
        status: str | None = Query(None),
        limit: int | None = Query(None, ge=1),
    ) -> dict[str, Any]:
        query = _search_query(q, resolved_config)
        scoped_account_id = _required_account_id(await _resolve_scope(scope_provider))
        pool = await _resolve_pool(pool_provider)
        repository = repository_factory(pool)
        try:
            response = await _with_timeout(
                repository.search(
                    query=query,
                    account_id=scoped_account_id,
                    corpus_id=_optional_filter(corpus_id),
                    status=_status_filter(status, resolved_config),
                    limit=_api_limit(limit, resolved_config),
                ),
                resolved_config,
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="FAQ search timed out") from exc
        except Exception as exc:
            logger.exception("FAQ deflection search failed")
            raise HTTPException(status_code=503, detail="FAQ search unavailable") from exc
        return response.as_dict()

    @router.get("/{faq_id}")
    async def get_faq_deflection_detail(faq_id: str) -> dict[str, Any]:
        normalized_faq_id = _required_uuid_path_id(faq_id, name="faq_id")
        scope = await _resolve_scope(scope_provider)
        scoped_account_id = _required_account_id(scope)
        pool = await _resolve_pool(pool_provider)
        repository = faq_repository_factory(pool)
        try:
            draft = await _with_timeout(
                repository.get_draft(
                    normalized_faq_id,
                    scope=TenantScope(
                        account_id=scoped_account_id,
                        user_id=scope.user_id,
                    ),
                ),
                resolved_config,
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="FAQ detail timed out") from exc
        except Exception as exc:
            logger.exception("FAQ deflection detail failed")
            raise HTTPException(status_code=503, detail="FAQ detail unavailable") from exc
        if draft is None:
            raise HTTPException(status_code=404, detail="FAQ not found")
        return _faq_detail_payload(draft, account_id=scoped_account_id)

    return router


async def _with_timeout(awaitable: Awaitable[Any], config: FAQDeflectionSearchApiConfig) -> Any:
    if config.search_timeout_seconds is None:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=config.search_timeout_seconds)


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


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _required_account_id(scope: TenantScope) -> str:
    account_id = _clean(scope.account_id)
    if not account_id:
        raise HTTPException(status_code=400, detail="account_id is required")
    return account_id


def _search_query(value: str, config: FAQDeflectionSearchApiConfig) -> str:
    query = _clean(value)
    if not query:
        raise HTTPException(status_code=400, detail="q is required")
    if len(query) > config.max_query_chars:
        raise HTTPException(
            status_code=400,
            detail=f"q must be {config.max_query_chars} characters or fewer",
        )
    return query


def _api_limit(value: int | None, config: FAQDeflectionSearchApiConfig) -> int:
    limit = config.default_limit if value is None else int(value)
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be positive")
    return min(limit, config.max_limit)


def _status_filter(value: str | None, config: FAQDeflectionSearchApiConfig) -> str | None:
    if value is None:
        return config.default_status
    cleaned = _clean(value)
    return cleaned or None


def _optional_filter(value: str | None) -> str | None:
    cleaned = _clean(value)
    return cleaned or None


def _required_path_id(value: str, *, name: str) -> str:
    cleaned = _clean(value)
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{name} is required")
    return cleaned


def _required_uuid_path_id(value: str, *, name: str) -> str:
    cleaned = _required_path_id(value, name=name)
    try:
        UUID(cleaned)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be a valid UUID") from exc
    return cleaned


def _faq_detail_payload(draft: TicketFAQDraft, *, account_id: str) -> dict[str, Any]:
    return {
        "account_id": account_id,
        **draft.as_dict(),
    }


def _clean(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "FAQDeflectionSearchApiConfig",
    "create_faq_deflection_search_router",
]
