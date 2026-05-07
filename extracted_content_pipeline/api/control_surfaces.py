"""FastAPI router factory for AI Content Ops control-surface previews."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from fastapi import APIRouter, Body, HTTPException
except ImportError as exc:  # pragma: no cover - exercised in dependency-light CI.
    APIRouter = None
    Body = None
    HTTPException = None
    _FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

from ..campaign_ports import TenantScope
from ..content_ops_execution import (
    ContentOpsExecutionServices,
    execute_content_ops_from_mapping,
)
from ..control_surfaces import OUTPUT_CATALOG, PRESETS, preview_from_mapping
from ..generation_plan import build_generation_plan_from_mapping

ExecutionServicesProvider = Callable[
    [],
    ContentOpsExecutionServices | Awaitable[ContentOpsExecutionServices],
]
ScopeProvider = Callable[
    [],
    TenantScope | Mapping[str, Any] | None | Awaitable[TenantScope | Mapping[str, Any] | None],
]


def _require_fastapi() -> None:
    if _FASTAPI_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "FastAPI is required to create AI Content Ops control-surface routes. "
        "Install fastapi in the host application environment."
    ) from _FASTAPI_IMPORT_ERROR


@dataclass(frozen=True)
class ContentOpsControlSurfaceApiConfig:
    """Host-owned API defaults for content-ops planning routes."""

    prefix: str = "/content-ops"
    tags: tuple[str, ...] = ("content-ops",)

    def __post_init__(self) -> None:
        if not str(self.prefix or "").strip().startswith("/"):
            raise ValueError("prefix must start with /")


def create_content_ops_control_surface_router(
    *,
    config: ContentOpsControlSurfaceApiConfig | None = None,
    execution_services_provider: ExecutionServicesProvider | None = None,
    scope_provider: ScopeProvider | None = None,
    dependencies: Sequence[Any] | None = None,
) -> APIRouter:
    """Create host-mounted AI Content Ops control-surface routes.

    Preview and plan routes are preflight-only. The execute route is opt-in
    and only calls host-injected services.
    """

    _require_fastapi()
    resolved_config = config or ContentOpsControlSurfaceApiConfig()
    router = APIRouter(
        prefix=resolved_config.prefix,
        tags=list(resolved_config.tags),
        dependencies=list(dependencies or ()),
    )

    @router.get("/control-surfaces")
    async def describe_control_surfaces() -> dict[str, Any]:
        return {
            "outputs": [
                {
                    "id": item.id,
                    "label": item.label,
                    "description": item.description,
                    "implemented": item.implemented,
                    "estimated_unit_cost_usd": item.estimated_unit_cost_usd,
                    "required_inputs": list(item.required_inputs),
                    "default_max_items": item.default_max_items,
                }
                for item in OUTPUT_CATALOG.values()
            ],
            "presets": [
                {
                    "id": item.id,
                    "label": item.label,
                    "description": item.description,
                    "outputs": list(item.outputs),
                }
                for item in PRESETS.values()
            ],
            "ingestion_profiles": [
                "domain_specific",
                "manual",
                "existing_evidence",
            ],
        }

    @router.post("/preview")
    async def preview_generation(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return preview_from_mapping(payload)

    @router.post("/plan")
    async def plan_generation(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return build_generation_plan_from_mapping(payload)

    @router.post("/execute")
    async def execute_generation(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if execution_services_provider is None:
            raise HTTPException(
                status_code=503,
                detail="Content Ops execution services are not configured.",
            )
        services = await _resolve_provider(execution_services_provider)
        scope = _scope_from_value(
            await _resolve_provider(scope_provider)
            if scope_provider is not None
            else None
        )
        result = await execute_content_ops_from_mapping(
            payload,
            services=services,
            scope=scope,
        )
        if result["status"] == "blocked":
            raise HTTPException(status_code=400, detail=result)
        return result

    return router


async def _resolve_provider(provider: Callable[[], Any] | None) -> Any:
    if provider is None:
        return None
    value = provider()
    if hasattr(value, "__await__"):
        return await value
    return value


def _scope_from_value(value: TenantScope | Mapping[str, Any] | None) -> TenantScope | None:
    if value is None or isinstance(value, TenantScope):
        return value
    return TenantScope(
        account_id=_clean(value.get("account_id")),
        user_id=_clean(value.get("user_id")),
        allowed_vendors=tuple(str(item) for item in value.get("allowed_vendors", ()) or ()),
        roles=tuple(str(item) for item in value.get("roles", ()) or ()),
    )


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


__all__ = [
    "ContentOpsControlSurfaceApiConfig",
    "create_content_ops_control_surface_router",
]
