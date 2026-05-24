"""FastAPI router factory for generated Content Ops asset review."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from html import escape as html_escape
import logging
from typing import Any
from uuid import UUID, uuid4

try:
    from fastapi import APIRouter, Body, HTTPException, Query, Request, Response
except ImportError as exc:  # pragma: no cover - exercised in dependency-light CI.
    APIRouter = None
    Body = None
    HTTPException = None
    Query = None
    Request = Any
    Response = Any
    _FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

from ..campaign_ports import LLMClient, SkillStore, TenantScope
from ..blog_post_export import export_blog_post_drafts
from ..blog_post_postgres import PostgresBlogPostRepository
from ..landing_page_export import (
    export_landing_page_drafts,
    landing_page_draft_export_row,
    public_landing_page_draft_row,
    public_landing_page_robots,
)
from ..landing_page_generation import LandingPageGenerationService
from ..landing_page_postgres import (
    LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY,
    PostgresLandingPageRepository,
)
from ..landing_page_ports import LandingPageDraft, LandingPageSection
from ..report_export import export_report_drafts
from ..report_postgres import PostgresReportRepository
from ..sales_brief_export import export_sales_brief_drafts
from ..sales_brief_postgres import PostgresSalesBriefRepository
from ..ticket_faq_export import export_ticket_faq_drafts
from ..ticket_faq_postgres import PostgresTicketFAQRepository


PoolProvider = Callable[[], Any | Awaitable[Any]]
LLMProvider = Callable[[], LLMClient | Awaitable[LLMClient]]
ScopeProvider = Callable[[], TenantScope | Mapping[str, Any] | None | Awaitable[Any]]
SkillsProvider = Callable[[], SkillStore | Awaitable[SkillStore]]

ASSET_CHOICES = ("blog_post", "report", "landing_page", "sales_brief", "faq_markdown")
logger = logging.getLogger(__name__)


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
    max_batch_size: int = 200
    export_filename_prefix: str = "content_assets"
    public_landing_page_base_url: str | None = None
    public_sitemap_limit: int = 500

    def __post_init__(self) -> None:
        if self.default_limit < 0:
            raise ValueError("default_limit must be non-negative")
        if self.max_limit <= 0:
            raise ValueError("max_limit must be positive")
        if self.default_limit > self.max_limit:
            raise ValueError("default_limit must be less than or equal to max_limit")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if not _clean(self.export_filename_prefix):
            raise ValueError("export_filename_prefix is required")
        if self.public_sitemap_limit <= 0:
            raise ValueError("public_sitemap_limit must be positive")


@dataclass(frozen=True)
class _ClaimedLandingPageRepository:
    repository: PostgresLandingPageRepository
    repair_claim_token: str

    def __getattr__(self, name: str) -> Any:
        return getattr(self.repository, name)

    async def update_draft(
        self,
        landing_page_id: str,
        draft: LandingPageDraft,
        *,
        scope: TenantScope,
    ) -> LandingPageDraft | None:
        return await self.repository.update_draft(
            landing_page_id,
            draft,
            scope=scope,
            repair_claim_token=self.repair_claim_token,
        )


async def _release_landing_page_repair_claim(
    repository: PostgresLandingPageRepository,
    landing_page_id: str,
    *,
    token: str,
    scope: TenantScope,
    outcome: str,
) -> None:
    released = await repository.release_repair(
        landing_page_id,
        token=token,
        scope=scope,
    )
    if released:
        return
    logger.warning(
        "landing page repair claim release missed",
        extra={
            "account_id": scope.account_id,
            "landing_page_id": landing_page_id,
            "outcome": outcome,
        },
    )


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


async def _resolve_required_provider(
    provider: Callable[[], Any | Awaitable[Any]] | None,
    *,
    detail: str,
) -> Any:
    if provider is None:
        raise HTTPException(status_code=503, detail=detail)
    value = await _maybe_await(provider())
    if value is None:
        raise HTTPException(status_code=503, detail=detail)
    return value


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


def _review_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise HTTPException(status_code=400, detail="ids must be a non-empty list")
    ids: list[str] = []
    seen: set[str] = set()
    for item in value:
        asset_id = _clean(item)
        if not asset_id or asset_id in seen:
            continue
        seen.add(asset_id)
        ids.append(asset_id)
    if not ids:
        raise HTTPException(status_code=400, detail="ids must be a non-empty list")
    return tuple(ids)


def _partition_uuid_ids(ids: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    valid: list[str] = []
    invalid: list[str] = []
    for asset_id in ids:
        try:
            UUID(asset_id)
        except ValueError:
            invalid.append(asset_id)
        else:
            valid.append(asset_id)
    return tuple(valid), tuple(invalid)


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
    llm_provider: LLMProvider | None = None,
    skills_provider: SkillsProvider | None = None,
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
        topic_type: str | None = Query(None),
        brief_type: str | None = Query(None),
        ids: list[str] | None = Query(None, alias="id"),
        limit: int | None = Query(None, ge=0),
    ) -> dict[str, Any]:
        asset_name = _asset_arg(asset)
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        result = await _export_for_asset(
            asset_name,
            pool,
            scope=scope,
            status=_status_filter(status, resolved_config),
            target_mode=target_mode,
            report_type=report_type,
            campaign_name=campaign_name,
            slug=slug,
            topic_type=topic_type,
            brief_type=brief_type,
            ids=ids,
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
        topic_type: str | None = Query(None),
        brief_type: str | None = Query(None),
        ids: list[str] | None = Query(None, alias="id"),
        limit: int | None = Query(None, ge=0),
        format: str = Query("csv", description="csv or json"),
    ) -> Any:
        asset_name = _asset_arg(asset)
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        result = await _export_for_asset(
            asset_name,
            pool,
            scope=scope,
            status=_status_filter(status, resolved_config),
            target_mode=target_mode,
            report_type=report_type,
            campaign_name=campaign_name,
            slug=slug,
            topic_type=topic_type,
            brief_type=brief_type,
            ids=ids,
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
        asset_name = _asset_arg(asset)
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        tenant = _tenant_scope(scope)
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

    @router.post("/{asset}/drafts/review-batch")
    async def review_drafts_batch(
        asset: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        asset_name = _asset_arg(asset)
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        tenant = _tenant_scope(scope)
        ids = _review_ids(payload.get("ids") or payload.get("asset_ids"))
        if len(ids) > resolved_config.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    "batch size exceeds max_batch_size "
                    f"({resolved_config.max_batch_size})"
                ),
        )
        status = _review_status(payload.get("status"))
        valid_ids, invalid_ids = _partition_uuid_ids(ids)
        updated = set(await _update_asset_statuses(
            asset_name,
            pool,
            asset_ids=valid_ids,
            status=status,
            scope=tenant,
        ))
        updated_ids = [asset_id for asset_id in ids if asset_id in updated]
        missing = set(invalid_ids)
        missing.update(asset_id for asset_id in valid_ids if asset_id not in updated)
        missing_ids = [asset_id for asset_id in ids if asset_id in missing]
        return {
            "account_id": tenant.account_id,
            "asset": asset_name,
            "ids": list(ids),
            "status": status,
            "updated": len(updated_ids),
            "updated_ids": updated_ids,
            "missing_ids": missing_ids,
        }

    @router.patch("/{asset}/drafts/{draft_id}")
    async def update_landing_page_draft(
        asset: str,
        draft_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        asset_name = _asset_arg(asset)
        if asset_name != "landing_page":
            raise HTTPException(
                status_code=400,
                detail="only landing_page drafts can be edited",
            )
        try:
            landing_page_id = str(UUID(draft_id))
        except ValueError:
            raise HTTPException(status_code=404, detail="Landing page draft not found") from None
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        tenant = _tenant_scope(scope)
        repository = PostgresLandingPageRepository(pool)
        existing = await repository.get_draft(landing_page_id, scope=tenant)
        if existing is None:
            raise HTTPException(status_code=404, detail="Landing page draft not found")
        if existing.status == "approved":
            raise HTTPException(
                status_code=409,
                detail="approved landing pages cannot be edited",
            )
        updated_draft = _patched_landing_page_draft(existing, payload)
        updated = await repository.update_draft(
            landing_page_id,
            updated_draft,
            scope=tenant,
        )
        if updated is None:
            raise HTTPException(
                status_code=409,
                detail="landing page draft could not be edited",
            )
        return landing_page_draft_export_row(updated)

    @router.post("/{asset}/drafts/{draft_id}/repair")
    async def repair_landing_page_draft(
        asset: str,
        draft_id: str,
    ) -> dict[str, Any]:
        asset_name = _asset_arg(asset)
        if asset_name != "landing_page":
            raise HTTPException(
                status_code=400,
                detail="only landing_page drafts can be repaired",
            )
        try:
            landing_page_id = str(UUID(draft_id))
        except ValueError:
            raise HTTPException(status_code=404, detail="Landing page draft not found") from None
        pool = await _resolve_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        tenant = _tenant_scope(scope)
        repository = PostgresLandingPageRepository(pool)
        existing = await repository.get_draft(landing_page_id, scope=tenant)
        if existing is None:
            raise HTTPException(status_code=404, detail="Landing page draft not found")
        if existing.status == "approved":
            raise HTTPException(
                status_code=409,
                detail="approved landing pages cannot be repaired",
            )
        repair_claim_token = uuid4().hex
        claimed = await repository.claim_repair(
            landing_page_id,
            token=repair_claim_token,
            scope=tenant,
        )
        if claimed is None:
            raise HTTPException(
                status_code=409,
                detail="landing page draft repair already in progress",
            )
        release_claim = True
        try:
            llm = await _resolve_required_provider(llm_provider, detail="LLM unavailable")
            skills = await _resolve_required_provider(
                skills_provider,
                detail="Landing page generation skills unavailable",
            )
            claimed_repository = _ClaimedLandingPageRepository(
                repository=repository,
                repair_claim_token=repair_claim_token,
            )
            result = await LandingPageGenerationService(
                landing_pages=claimed_repository,
                llm=llm,
                skills=skills,
            ).repair_draft(scope=tenant, draft=claimed)
            result_payload = result.as_dict()
            if result.generated == 0 and result.skipped > 0:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "landing page draft could not be repaired",
                        "repair_result": result_payload,
                    },
                )
            await _release_landing_page_repair_claim(
                repository,
                landing_page_id,
                token=repair_claim_token,
                scope=tenant,
                outcome="success",
            )
            release_claim = False
            refreshed = await repository.get_draft(landing_page_id, scope=tenant)
            if refreshed is None:
                raise HTTPException(status_code=404, detail="Landing page draft not found")
            row = landing_page_draft_export_row(refreshed)
            row["repair_result"] = result_payload
            return row
        finally:
            if release_claim:
                await _release_landing_page_repair_claim(
                    repository,
                    landing_page_id,
                    token=repair_claim_token,
                    scope=tenant,
                    outcome="cleanup",
                )

    return router


def create_public_landing_page_router(
    *,
    pool_provider: PoolProvider,
    config: GeneratedAssetApiConfig | None = None,
) -> APIRouter:
    """Create unauthenticated public routes for approved landing pages."""
    _require_fastapi()
    resolved_config = config or GeneratedAssetApiConfig()
    router = APIRouter(
        prefix=resolved_config.prefix,
        tags=list(resolved_config.tags),
    )

    @router.get("/landing_page/public/sitemap.xml", response_model=None)
    async def public_landing_page_sitemap(request: Request) -> Response:
        pool = await _resolve_pool(pool_provider)
        candidates = await (
            PostgresLandingPageRepository(pool).list_public_sitemap_candidates()
        )
        base_url = _public_landing_page_base_url(resolved_config, request)
        urls: list[str] = []
        for candidate in candidates:
            if public_landing_page_robots(candidate.to_policy_draft()) != "index,follow":
                continue
            urls.append(_public_landing_page_url(base_url, candidate))
            if len(urls) >= resolved_config.public_sitemap_limit:
                break
        return Response(
            content=_sitemap_xml(urls),
            media_type="application/xml",
        )

    @router.get("/landing_page/public/{landing_page_id}")
    async def public_landing_page(
        landing_page_id: str,
    ) -> dict[str, Any]:
        try:
            public_id = str(UUID(landing_page_id))
        except ValueError:
            raise HTTPException(status_code=404, detail="Landing page not found") from None
        pool = await _resolve_pool(pool_provider)
        draft = await PostgresLandingPageRepository(pool).get_public_approved_draft(
            public_id
        )
        if draft is None:
            raise HTTPException(status_code=404, detail="Landing page not found")
        return public_landing_page_draft_row(draft)

    return router


def _public_landing_page_base_url(
    config: GeneratedAssetApiConfig,
    request: Request,
) -> str:
    configured = _clean(config.public_landing_page_base_url)
    if configured:
        return configured.rstrip("/")
    return str(request.base_url).rstrip("/")


def _public_landing_page_url(base_url: str, draft: Any) -> str:
    return f"{base_url}/lp/{draft.id}/{draft.slug}"


def _sitemap_xml(urls: Sequence[str]) -> str:
    today = date.today().isoformat()
    rows = "\n".join(
        "  <url>\n"
        f"    <loc>{html_escape(url, quote=True)}</loc>\n"
        f"    <lastmod>{today}</lastmod>\n"
        "    <changefreq>weekly</changefreq>\n"
        "    <priority>0.7</priority>\n"
        "  </url>"
        for url in urls
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{rows}\n"
        "</urlset>\n"
    )


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
    topic_type: str | None,
    brief_type: str | None,
    ids: Sequence[str] | None,
    limit: int,
) -> Any:
    if ids and asset not in {"blog_post", "landing_page"}:
        raise HTTPException(
            status_code=400,
            detail="id filters are only supported for blog_post and landing_page",
        )
    if asset == "blog_post":
        return await export_blog_post_drafts(
            PostgresBlogPostRepository(pool),
            scope=scope,
            status=status,
            topic_type=topic_type,
            ids=ids,
            limit=limit,
        )
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
            ids=ids,
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
    if asset == "faq_markdown":
        return await export_ticket_faq_drafts(
            PostgresTicketFAQRepository(pool),
            scope=scope,
            status=status,
            target_mode=target_mode,
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
    if asset == "blog_post":
        return await PostgresBlogPostRepository(pool).update_status(asset_id, status, scope=scope)
    if asset == "report":
        return await PostgresReportRepository(pool).update_status(asset_id, status, scope=scope)
    if asset == "landing_page":
        return await PostgresLandingPageRepository(pool).update_status(asset_id, status, scope=scope)
    if asset == "sales_brief":
        return await PostgresSalesBriefRepository(pool).update_status(asset_id, status, scope=scope)
    if asset == "faq_markdown":
        return await PostgresTicketFAQRepository(pool).update_status(asset_id, status, scope=scope)
    raise HTTPException(status_code=400, detail=f"unsupported asset: {asset}")


async def _update_asset_statuses(
    asset: str,
    pool: Any,
    *,
    asset_ids: Sequence[str],
    status: str,
    scope: TenantScope,
) -> Sequence[str]:
    if asset == "blog_post":
        return await PostgresBlogPostRepository(pool).update_statuses(asset_ids, status, scope=scope)
    if asset == "report":
        return await PostgresReportRepository(pool).update_statuses(asset_ids, status, scope=scope)
    if asset == "landing_page":
        return await PostgresLandingPageRepository(pool).update_statuses(asset_ids, status, scope=scope)
    if asset == "sales_brief":
        return await PostgresSalesBriefRepository(pool).update_statuses(asset_ids, status, scope=scope)
    if asset == "faq_markdown":
        return await PostgresTicketFAQRepository(pool).update_statuses(asset_ids, status, scope=scope)
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


_LANDING_PAGE_EDITABLE_FIELDS = frozenset({
    "title",
    "slug",
    "hero",
    "sections",
    "cta",
    "meta",
    "reference_ids",
})


def _patched_landing_page_draft(
    existing: LandingPageDraft,
    payload: Mapping[str, Any],
) -> LandingPageDraft:
    fields = set(payload).intersection(_LANDING_PAGE_EDITABLE_FIELDS)
    if not fields:
        raise HTTPException(
            status_code=400,
            detail=(
                "payload must include at least one editable landing page field: "
                + ", ".join(sorted(_LANDING_PAGE_EDITABLE_FIELDS))
            ),
        )
    return LandingPageDraft(
        id=existing.id,
        status="draft",
        campaign_name=existing.campaign_name,
        persona=existing.persona,
        value_prop=existing.value_prop,
        title=_patch_text(payload, "title", existing.title),
        slug=_patch_text(payload, "slug", existing.slug),
        hero=_patch_mapping(payload, "hero", existing.hero),
        sections=_patch_sections(payload, "sections", existing.sections),
        cta=_patch_mapping(payload, "cta", existing.cta),
        meta=_patch_mapping(payload, "meta", existing.meta),
        reference_ids=_patch_string_list(
            payload,
            "reference_ids",
            existing.reference_ids,
        ),
        metadata=dict(existing.metadata or {}),
    )


def _patch_text(payload: Mapping[str, Any], key: str, current: str) -> str:
    if key not in payload:
        return current
    value = payload.get(key)
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"{key} must be a string")
    return value.strip()


def _patch_mapping(
    payload: Mapping[str, Any],
    key: str,
    current: Mapping[str, Any],
) -> Mapping[str, Any]:
    if key not in payload:
        return dict(current or {})
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise HTTPException(status_code=400, detail=f"{key} must be an object")
    return dict(value)


def _patch_sections(
    payload: Mapping[str, Any],
    key: str,
    current: Sequence[LandingPageSection],
) -> Sequence[LandingPageSection]:
    if key not in payload:
        return tuple(current or ())
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise HTTPException(status_code=400, detail=f"{key} must be a list")
    sections: list[LandingPageSection] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise HTTPException(
                status_code=400,
                detail=f"{key}[{index}] must be an object",
            )
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise HTTPException(
                status_code=400,
                detail=f"{key}[{index}].metadata must be an object",
            )
        sections.append(
            LandingPageSection(
                id=str(item.get("id") or ""),
                title=str(item.get("title") or ""),
                body_markdown=str(
                    item.get("body_markdown") or item.get("body") or ""
                ),
                metadata=dict(metadata),
            )
        )
    return tuple(sections)


def _patch_string_list(
    payload: Mapping[str, Any],
    key: str,
    current: Sequence[str],
) -> Sequence[str]:
    if key not in payload:
        return tuple(str(item) for item in current)
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise HTTPException(status_code=400, detail=f"{key} must be a list")
    return tuple(str(item).strip() for item in value if str(item).strip())


__all__ = [
    "GeneratedAssetApiConfig",
    "create_generated_asset_router",
    "create_public_landing_page_router",
]
