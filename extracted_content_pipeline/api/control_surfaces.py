"""FastAPI router factory for AI Content Ops control-surface previews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

try:
    from fastapi import APIRouter, Body
except ImportError as exc:  # pragma: no cover - exercised in dependency-light CI.
    APIRouter = None
    Body = None
    _FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

from ..control_surfaces import OUTPUT_CATALOG, PRESETS, preview_from_mapping


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
    dependencies: Sequence[Any] | None = None,
) -> APIRouter:
    """Create host-mounted AI Content Ops control-surface routes.

    These routes are intentionally preflight-only. They do not invoke LLMs,
    mutate storage, or kick off autonomous jobs. Revolutionary, apparently.
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

    return router


__all__ = [
    "ContentOpsControlSurfaceApiConfig",
    "create_content_ops_control_surface_router",
]
