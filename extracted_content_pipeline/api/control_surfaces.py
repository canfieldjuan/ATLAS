"""FastAPI router factory for AI Content Ops control-surface previews."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from fastapi import APIRouter, Body, HTTPException
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
except ImportError as exc:  # pragma: no cover - exercised in dependency-light CI.
    APIRouter = None
    Body = None
    HTTPException = None
    BaseModel = None
    ConfigDict = None
    Field = None
    ValidationError = None
    field_validator = None
    _FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

from ..campaign_ports import TenantScope
from ..content_ops_execution import (
    ContentOpsExecutionServices,
    execute_content_ops_from_mapping,
)
from ..control_surfaces import (
    OUTPUT_CATALOG,
    PRESETS,
    preview_from_mapping,
    retry_adjusted_unit_cost_usd,
)
from ..generation_plan import build_generation_plan_from_mapping

ExecutionServicesProvider = Callable[
    [],
    ContentOpsExecutionServices | Awaitable[ContentOpsExecutionServices],
]
ScopeProvider = Callable[
    [],
    TenantScope | Mapping[str, Any] | None | Awaitable[TenantScope | Mapping[str, Any] | None],
]

logger = logging.getLogger(__name__)

_MAX_INPUT_KEYS = 50
_MAX_INPUT_DEPTH = 6
_MAX_INPUT_STRING_CHARS = 10000
_SAFE_EXECUTION_REASONS = {"plan_not_executable", "service_not_configured"}


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


if BaseModel is not None:

    class ContentOpsRequestModel(BaseModel):
        """Bounded API request body for Content Ops preview/plan/execute."""

        model_config = ConfigDict(extra="forbid")

        target_mode: str = Field("vendor_retention", min_length=1, max_length=80)
        preset: str | None = Field(default=None, max_length=80)
        outputs: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
        limit: int = Field(1, ge=1, le=1000)
        max_cost_usd: float | None = Field(default=None, gt=0)
        inputs: dict[str, Any] = Field(default_factory=dict, max_length=_MAX_INPUT_KEYS)
        ingestion_profile: str = Field("domain_specific", min_length=1, max_length=80)
        require_quality_gates: bool = True
        allow_unimplemented_outputs: bool = False

        @field_validator("outputs")
        @classmethod
        def _validate_outputs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
            for item in value:
                text = str(item or "").strip()
                if not text or len(text) > 80:
                    raise ValueError("outputs entries must be 1-80 characters")
            return value

        @field_validator("inputs")
        @classmethod
        def _validate_inputs(cls, value: dict[str, Any]) -> dict[str, Any]:
            _validate_input_shape(value, depth=0)
            return value

else:  # pragma: no cover - module import fallback when FastAPI is unavailable.
    ContentOpsRequestModel = Any


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
        execution_services = await _resolve_execution_services(execution_services_provider)
        configured_outputs = set(
            execution_services.configured_outputs()
            if execution_services is not None
            else ()
        )
        return {
            "outputs": [
                {
                    "id": item.id,
                    "label": item.label,
                    "description": item.description,
                    "implemented": item.implemented,
                    "execution_configured": item.id in configured_outputs,
                    "can_execute": item.implemented and item.id in configured_outputs,
                    "estimated_unit_cost_usd": item.estimated_unit_cost_usd,
                    "default_parse_retry_attempts": item.default_parse_retry_attempts,
                    "estimated_retry_adjusted_unit_cost_usd": round(
                        retry_adjusted_unit_cost_usd(item),
                        4,
                    ),
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
            "execution": {
                "configured": execution_services is not None,
                "configured_outputs": sorted(configured_outputs),
            },
            "ingestion_profiles": [
                "domain_specific",
                "manual",
                "existing_evidence",
            ],
        }

    @router.post("/preview")
    async def preview_generation(
        payload: ContentOpsRequestModel = Body(...),
    ) -> dict[str, Any]:
        return preview_from_mapping(_payload_to_mapping(payload))

    @router.post("/plan")
    async def plan_generation(
        payload: ContentOpsRequestModel = Body(...),
    ) -> dict[str, Any]:
        return build_generation_plan_from_mapping(_payload_to_mapping(payload))

    @router.post("/execute")
    async def execute_generation(
        payload: ContentOpsRequestModel = Body(...),
    ) -> dict[str, Any]:
        if execution_services_provider is None:
            raise HTTPException(
                status_code=503,
                detail="Content Ops execution services are not configured.",
            )
        services = await _resolve_execution_services(execution_services_provider)
        if services is None:
            raise HTTPException(
                status_code=503,
                detail="Content Ops execution services are not configured.",
            )
        scope = await _resolve_scope(scope_provider)
        result = await execute_content_ops_from_mapping(
            _payload_to_mapping(payload),
            services=services,
            scope=scope,
        )
        result = _sanitize_execution_result(result)
        if result["status"] == "blocked":
            raise HTTPException(status_code=400, detail=result)
        if result["status"] == "failed":
            raise HTTPException(status_code=502, detail=result)
        if result["status"] == "partial":
            raise HTTPException(status_code=207, detail=result)
        return result

    return router


async def _resolve_execution_services(
    provider: ExecutionServicesProvider | None,
) -> ContentOpsExecutionServices | None:
    try:
        value = await _resolve_provider(provider)
    except Exception as exc:
        logger.warning(
            "Content Ops execution services provider failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Content Ops execution services are unavailable.",
        ) from exc
    return value if isinstance(value, ContentOpsExecutionServices) else None


async def _resolve_scope(provider: ScopeProvider | None) -> TenantScope | None:
    try:
        value = await _resolve_provider(provider) if provider is not None else None
    except Exception as exc:
        logger.warning(
            "Content Ops scope provider failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Content Ops scope provider is unavailable.",
        ) from exc
    return _scope_from_value(value)


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
        allowed_vendors=_clean_sequence(value.get("allowed_vendors", ())),
        roles=_clean_sequence(value.get("roles", ())),
    )


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _clean_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        candidates = (value,)
    else:
        candidates = value or ()
    cleaned: list[str] = []
    for item in candidates:
        text = _clean(item)
        if text:
            cleaned.append(text)
    return tuple(cleaned)


def _payload_to_mapping(payload: Any) -> dict[str, Any]:
    if BaseModel is not None and isinstance(payload, ContentOpsRequestModel):
        return payload.model_dump()
    try:
        return ContentOpsRequestModel.model_validate(payload).model_dump()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_detail(exc)) from exc


def _validate_input_shape(value: Any, *, depth: int) -> None:
    if depth > _MAX_INPUT_DEPTH:
        raise ValueError("inputs are too deeply nested")
    if isinstance(value, Mapping):
        if len(value) > _MAX_INPUT_KEYS:
            raise ValueError("inputs has too many keys")
        for key, nested in value.items():
            if len(str(key)) > 100:
                raise ValueError("inputs keys must be 100 characters or fewer")
            _validate_input_shape(nested, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        if len(value) > _MAX_INPUT_KEYS:
            raise ValueError("inputs arrays are too large")
        for nested in value:
            _validate_input_shape(nested, depth=depth + 1)
    elif isinstance(value, str) and len(value) > _MAX_INPUT_STRING_CHARS:
        raise ValueError("inputs strings are too large")


def _sanitize_execution_result(result: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = dict(result)
    sanitized["errors"] = [
        _sanitize_error(error)
        for error in sanitized.get("errors", ()) or ()
    ]
    sanitized["steps"] = [
        _sanitize_step(step)
        for step in sanitized.get("steps", ()) or ()
    ]
    return sanitized


def _sanitize_error(error: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(error)
    cleaned["reason"] = _safe_execution_reason(cleaned.get("reason"))
    if cleaned.get("error"):
        cleaned["error"] = _safe_execution_reason(cleaned.get("error"))
    return cleaned


def _sanitize_step(step: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(step)
    if cleaned.get("error"):
        cleaned["error"] = _safe_execution_reason(cleaned.get("error"))
    return cleaned


def _safe_execution_reason(reason: Any) -> str:
    text = str(reason or "").strip()
    if text in _SAFE_EXECUTION_REASONS:
        return text
    return "execution_failed"


def _validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in error.items() if key != "ctx"}
        for error in exc.errors()
    ]


__all__ = [
    "ContentOpsControlSurfaceApiConfig",
    "ContentOpsRequestModel",
    "create_content_ops_control_surface_router",
]
