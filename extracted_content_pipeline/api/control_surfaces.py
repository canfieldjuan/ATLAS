"""FastAPI router factory for AI Content Ops control-surface previews."""

from __future__ import annotations

import logging
import math
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
    request_from_mapping,
    retry_adjusted_unit_cost_usd,
    resolve_outputs,
)
from ..generation_plan import build_generation_plan, build_generation_plan_from_mapping
from ..reasoning_policy import ReasoningPreset, resolve_reasoning_policy

ExecutionServicesProvider = Callable[
    [],
    ContentOpsExecutionServices | Awaitable[ContentOpsExecutionServices],
]
ScopeProvider = Callable[
    [],
    TenantScope | Mapping[str, Any] | None | Awaitable[TenantScope | Mapping[str, Any] | None],
]
# PR-ControlSurfaces-Reasoning-Provider: per-request reasoning provider
# resolved at /execute and merged into the services bundle via
# ContentOpsExecutionServices.with_reasoning_context().
ReasoningContextProvider = Callable[
    [],
    Any | Awaitable[Any],
]
ReasoningStatusProvider = Callable[
    [],
    Mapping[str, Any] | None | Awaitable[Mapping[str, Any] | None],
]
LLMProvider = Callable[[], Any | Awaitable[Any]]

logger = logging.getLogger(__name__)

_MAX_INPUT_KEYS = 50
_MAX_INPUT_DEPTH = 6
_MAX_INPUT_STRING_CHARS = 10000
_MAX_REASONING_STATUS_LIST_ITEMS = 20
_SAFE_EXECUTION_REASONS = {"plan_not_executable", "service_not_configured"}
_RUNTIME_STRUCTURED_REASONING_OUTPUTS = frozenset({"report", "sales_brief"})


def _build_static_catalog_payload() -> Mapping[str, Any]:
    # PR-Describe-Control-Surfaces-Cache: the outputs/presets metadata
    # is a pure function of the immutable OUTPUT_CATALOG and PRESETS
    # MappingProxyType globals. Computing it once at import lets the
    # GET /content-ops/control-surfaces hot path skip 6 + 5 per-item
    # dict constructions per request.
    return {
        "outputs": tuple(
            {
                "id": item.id,
                "label": item.label,
                "description": item.description,
                "implemented": item.implemented,
                "estimated_unit_cost_usd": item.estimated_unit_cost_usd,
                "default_parse_retry_attempts": item.default_parse_retry_attempts,
                "estimated_retry_adjusted_unit_cost_usd": round(
                    retry_adjusted_unit_cost_usd(item),
                    4,
                ),
                "required_inputs": tuple(item.required_inputs),
                "default_max_items": item.default_max_items,
                "reasoning_requirement": item.reasoning_requirement,
            }
            for item in OUTPUT_CATALOG.values()
        ),
        "presets": tuple(
            {
                "id": item.id,
                "label": item.label,
                "description": item.description,
                "outputs": tuple(item.outputs),
            }
            for item in PRESETS.values()
        ),
        "ingestion_profiles": (
            "domain_specific",
            "manual",
            "existing_evidence",
        ),
    }


_STATIC_CATALOG_PAYLOAD: Mapping[str, Any] = _build_static_catalog_payload()


def _compose_describe_response(
    *,
    static: Mapping[str, Any],
    configured_outputs: frozenset[str],
    execution_configured: bool,
    reasoning_status: Mapping[str, Any],
) -> dict[str, Any]:
    # Re-project the cached static template into a fresh dict tree so
    # the caller can serialize / mutate without aliasing the module-
    # level cache. Per-output flags are the only fields that depend
    # on the host-injected execution services.
    return {
        "outputs": [
            {
                **base,
                "required_inputs": list(base["required_inputs"]),
                "execution_configured": base["id"] in configured_outputs,
                "can_execute": base["implemented"]
                and base["id"] in configured_outputs,
            }
            for base in static["outputs"]
        ],
        "presets": [
            {**preset, "outputs": list(preset["outputs"])}
            for preset in static["presets"]
        ],
        "execution": {
            "configured": execution_configured,
            "configured_outputs": sorted(configured_outputs),
        },
        "reasoning": dict(reasoning_status),
        "ingestion_profiles": list(static["ingestion_profiles"]),
    }


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
    structured_reasoning_default_goal: str = "synthesize content reasoning context"
    structured_reasoning_depth: str = "L3"
    structured_reasoning_pack_name: str = "content_ops_structured"
    structured_reasoning_top_thesis_limit: int = 5
    structured_reasoning_max_continuations: int = 8
    structured_reasoning_require_citations: bool = True

    def __post_init__(self) -> None:
        if not str(self.prefix or "").strip().startswith("/"):
            raise ValueError("prefix must start with /")
        if not str(self.structured_reasoning_default_goal or "").strip():
            raise ValueError("structured_reasoning_default_goal is required")
        if not str(self.structured_reasoning_pack_name or "").strip():
            raise ValueError("structured_reasoning_pack_name is required")
        if self.structured_reasoning_depth not in {"L1", "L2", "L3", "L4", "L5"}:
            raise ValueError("structured_reasoning_depth must be L1-L5")
        if self.structured_reasoning_top_thesis_limit <= 0:
            raise ValueError("structured_reasoning_top_thesis_limit must be positive")
        if self.structured_reasoning_max_continuations < 0:
            raise ValueError("structured_reasoning_max_continuations must be non-negative")


@dataclass(frozen=True)
class _BlockingReasoningContextProvider:
    provider: Any

    async def read_campaign_reasoning_context(
        self,
        *,
        scope: TenantScope,
        target_id: str,
        target_mode: str,
        opportunity: Mapping[str, Any],
    ) -> Any:
        context = await self.provider.read_campaign_reasoning_context(
            scope=scope,
            target_id=target_id,
            target_mode=target_mode,
            opportunity=opportunity,
        )
        if context is None:
            raise RuntimeError("reasoning_validation_blocked")
        validation = _reasoning_validation_from_context(context)
        if validation is not None and validation.get("passed") is False:
            blockers = [
                str(item).strip()
                for item in _clean_status_scalar_sequence(validation.get("blockers"))
                if str(item).strip()
            ]
            suffix = f":{','.join(blockers)}" if blockers else ""
            raise RuntimeError(f"reasoning_validation_blocked{suffix}")
        return context


def _reasoning_validation_from_context(context: Any) -> Mapping[str, Any] | None:
    canonical = getattr(context, "canonical_reasoning", None)
    if isinstance(canonical, Mapping):
        validation = canonical.get("validation")
        if isinstance(validation, Mapping):
            return validation
    if isinstance(context, Mapping):
        validation = context.get("validation")
        if isinstance(validation, Mapping):
            return validation
        canonical = context.get("canonical_reasoning")
        if isinstance(canonical, Mapping):
            validation = canonical.get("validation")
            if isinstance(validation, Mapping):
                return validation
    return None


if BaseModel is not None:

    class ContentOpsRequestModel(BaseModel):
        """Bounded API request body for Content Ops preview/plan/execute."""

        model_config = ConfigDict(extra="forbid")

        target_mode: str = Field("vendor_retention", min_length=1, max_length=80)
        preset: str | None = Field(default=None, max_length=80)
        reasoning_preset: ReasoningPreset | None = Field(default=None)
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

        @field_validator("reasoning_preset", mode="before")
        @classmethod
        def _normalize_reasoning_preset(cls, value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, str):
                text = value.strip()
                return text or None
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
    reasoning_context_provider: ReasoningContextProvider | None = None,
    reasoning_status_provider: ReasoningStatusProvider | None = None,
    llm_provider: LLMProvider | None = None,
    dependencies: Sequence[Any] | None = None,
) -> APIRouter:
    """Create host-mounted AI Content Ops control-surface routes.

    Preview and plan routes are preflight-only. The execute route is opt-in
    and only calls host-injected services.

    PR-ControlSurfaces-Reasoning-Provider: when
    ``reasoning_context_provider`` is supplied, the /execute route
    resolves it per request and derives a reasoning-aware services
    bundle via ``ContentOpsExecutionServices.with_reasoning_context``
    before invoking the executor. The base services from
    ``execution_services_provider`` are not mutated. Mirrors the
    legacy ``api.campaign_operations`` reasoning seam.
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
        configured_outputs = frozenset(
            execution_services.configured_outputs()
            if execution_services is not None
            else ()
        )
        reasoning_status = await _resolve_reasoning_status(
            reasoning_status_provider,
            default_configured=reasoning_context_provider is not None,
        )
        return _compose_describe_response(
            static=_STATIC_CATALOG_PAYLOAD,
            configured_outputs=configured_outputs,
            execution_configured=execution_services is not None,
            reasoning_status=reasoning_status,
        )

    @router.post("/preview")
    async def preview_generation(
        payload: ContentOpsRequestModel = Body(...),
    ) -> dict[str, Any]:
        return preview_from_mapping(_payload_to_mapping(payload))

    @router.post("/plan")
    async def plan_generation(
        payload: ContentOpsRequestModel = Body(...),
    ) -> dict[str, Any]:
        try:
            return build_generation_plan_from_mapping(_payload_to_mapping(payload))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        # PR-ControlSurfaces-Reasoning-Provider: resolve the optional
        # per-request reasoning provider and derive a reasoning-aware
        # bundle. The base services from execution_services_provider
        # are not mutated; the derivation rebinds reasoning on each
        # opt-in service via with_reasoning_context().
        #
        # Gate on whether the kwarg was supplied at router construction,
        # not on the resolved value -- when a host wires a per-request
        # provider that returns None for tenant-policy reasons, the
        # bundle is derived with reasoning rebound to None (predictable,
        # no leak of construction-time reasoning). When the kwarg was
        # never supplied, leave the host-baked reasoning alone.
        payload_mapping = _payload_to_mapping(payload)
        if reasoning_context_provider is not None:
            if _clean(payload_mapping.get("reasoning_preset")):
                logger.info(
                    "Content Ops reasoning_preset ignored because host provider is configured"
                )
            reasoning_context = await _resolve_reasoning_context(
                reasoning_context_provider
            )
            services = services.with_reasoning_context(reasoning_context)
        else:
            reasoning_context, reasoning_outputs = await _structured_reasoning_context(
                payload_mapping,
                config=resolved_config,
                services=services,
                llm_provider=llm_provider,
            )
            if reasoning_context is not None:
                services = services.with_reasoning_context(
                    reasoning_context,
                    outputs=reasoning_outputs,
                )
        scope = await _resolve_scope(scope_provider)
        try:
            result = await execute_content_ops_from_mapping(
                payload_mapping,
                services=services,
                scope=scope,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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


async def _resolve_reasoning_context(
    provider: ReasoningContextProvider | None,
) -> Any | None:
    if provider is None:
        return None
    try:
        value = await _resolve_provider(provider)
    except Exception as exc:
        logger.warning(
            "Content Ops reasoning context provider failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Content Ops reasoning context provider is unavailable.",
        ) from exc
    return value


async def _structured_reasoning_context(
    payload: Mapping[str, Any],
    *,
    config: ContentOpsControlSurfaceApiConfig,
    services: ContentOpsExecutionServices,
    llm_provider: LLMProvider | None,
) -> tuple[Any | None, tuple[str, ...]]:
    preset = _clean(payload.get("reasoning_preset"))
    if not preset:
        return None, ()
    if preset in {"none", "context_only"}:
        return None, ()
    try:
        request = request_from_mapping(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        plan = build_generation_plan(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not plan.can_execute:
        return None, ()
    supported_outputs = [
        str(output)
        for output in resolve_outputs(request)
        if output in _RUNTIME_STRUCTURED_REASONING_OUTPUTS
    ]
    if not supported_outputs:
        raise HTTPException(
            status_code=400,
            detail="reasoning_preset currently applies only to report and sales_brief.",
        )
    reasoning_definition = None
    for output in supported_outputs:
        try:
            _policy, definition = resolve_reasoning_policy(output, preset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        reasoning_definition = definition
        if definition.id not in {"multi_pass_structured", "multi_pass_strict"}:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Content Ops packaged reasoning currently supports "
                    "multi_pass_structured or multi_pass_strict for report "
                    "and sales_brief."
                ),
            )
    selected_outputs = [
        output for output in supported_outputs if output in services.configured_outputs()
    ]
    if not selected_outputs:
        return None, ()
    if reasoning_definition is None:
        return None, ()
    for output in selected_outputs:
        service = services.for_output(output)
        if not callable(getattr(service, "with_reasoning_context", None)):
            raise HTTPException(
                status_code=503,
                detail=f"{output} service does not support structured reasoning.",
            )
    if llm_provider is None:
        raise HTTPException(
            status_code=503,
            detail="Content Ops structured reasoning LLM is unavailable.",
        )
    try:
        llm = await _resolve_provider(llm_provider)
    except Exception as exc:
        logger.exception("Content Ops structured reasoning LLM provider failed")
        raise HTTPException(
            status_code=503,
            detail="Content Ops structured reasoning LLM is unavailable.",
        ) from exc
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="Content Ops structured reasoning LLM is unavailable.",
        )
    # Lazy import keeps hosts without structured reasoning free of these deps.
    from extracted_reasoning_core.types import OutputPolicy, ReasoningPack, ReasoningPorts

    from ..services.multi_pass_reasoning_provider import (
        MultiPassCampaignReasoningProvider,
        MultiPassReasoningProviderConfig,
    )

    provider = MultiPassCampaignReasoningProvider(
        ports=ReasoningPorts(llm=llm),
        config=MultiPassReasoningProviderConfig(
            default_goal=config.structured_reasoning_default_goal,
            default_depth=config.structured_reasoning_depth,
            top_thesis_limit=config.structured_reasoning_top_thesis_limit,
            max_continuations=config.structured_reasoning_max_continuations,
            narrative_plan_pack=ReasoningPack(
                name=config.structured_reasoning_pack_name,
            ),
            output_policy=OutputPolicy(
                require_citations=config.structured_reasoning_require_citations,
            ),
            # Keep the inner provider nonblocking so strict wrappers can
            # surface validation blocker details instead of a generic None.
            block_on_validation_failure=False,
        ),
    )
    if reasoning_definition.blocking_validation:
        provider = _BlockingReasoningContextProvider(provider)
    return provider, tuple(selected_outputs)


async def _resolve_reasoning_status(
    provider: ReasoningStatusProvider | None,
    *,
    default_configured: bool,
) -> dict[str, Any]:
    if provider is None:
        return {"configured": default_configured}
    try:
        value = await _resolve_provider(provider)
    except Exception as exc:
        logger.warning(
            "Content Ops reasoning status provider failed",
            extra={"error_type": type(exc).__name__},
        )
        return {"configured": default_configured}
    return _sanitize_reasoning_status(value, default_configured=default_configured)


def _sanitize_reasoning_status(
    value: Mapping[str, Any] | None,
    *,
    default_configured: bool,
) -> dict[str, Any]:
    status = dict(value) if isinstance(value, Mapping) else {}
    status["configured"] = bool(status.get("configured", default_configured))
    for key in list(status):
        item = status[key]
        if key == "configured" or item is None:
            continue
        if key == "capabilities" and isinstance(item, Mapping):
            capabilities = _sanitize_reasoning_capabilities(item)
            if capabilities:
                status[key] = capabilities
                continue
            status.pop(key)
            continue
        if _is_status_scalar(item):
            continue
        scalar_items = _clean_status_scalar_sequence(item)
        if scalar_items:
            status[key] = scalar_items
            continue
        status.pop(key)
    return status


def _sanitize_reasoning_capabilities(value: Mapping[str, Any]) -> dict[str, Any]:
    capabilities: dict[str, Any] = {}
    for name, raw_status in value.items():
        capability_name = _clean(name)
        if not capability_name or not isinstance(raw_status, Mapping):
            continue
        item: dict[str, Any] = {}
        for flag in ("configured", "ready", "active"):
            if flag in raw_status:
                item[flag] = bool(raw_status.get(flag))
        missing = [
            str(entry).strip()
            for entry in _clean_status_scalar_sequence(raw_status.get("missing"))
            if str(entry).strip()
        ]
        if missing:
            item["missing"] = missing
        if item:
            capabilities[capability_name] = item
    return capabilities


def _clean_status_scalar_sequence(value: Any) -> list[str | int | float | bool]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return []
    if not isinstance(value, Sequence):
        return []
    cleaned: list[str | int | float | bool] = []
    for index, item in enumerate(value):
        if index >= _MAX_REASONING_STATUS_LIST_ITEMS:
            break
        if _is_status_scalar(item):
            cleaned.append(item)
    return cleaned


def _is_status_scalar(value: Any) -> bool:
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return isinstance(value, (str, int, float, bool))


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
