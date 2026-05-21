"""FastAPI router factory for AI Content Ops control-surface previews."""

from __future__ import annotations

import logging
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

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
from ..campaign_postgres_import import import_campaign_opportunities
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
from ..ingestion_diagnostics import inspect_ingestion_rows
from ..reasoning_policy import (
    PACKAGED_REASONING_RUNTIME_OUTPUTS,
    ReasoningPreset,
    packaged_reasoning_runtime_presets_for_output,
    resolve_reasoning_policy,
)
from ..reasoning_signals import reasoning_validation_blocked_reason

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
PoolProvider = Callable[[], Any | Awaitable[Any]]

logger = logging.getLogger(__name__)

_MAX_INPUT_KEYS = 50
_MAX_INPUT_DEPTH = 6
_MAX_INPUT_STRING_CHARS = 10000
_MAX_INGESTION_ROWS = 500
_MAX_INGESTION_SAMPLE_LIMIT = 25
_MAX_REASONING_STATUS_LIST_ITEMS = 20
_MAX_FALSIFICATION_RULES = 20
_SAFE_EXECUTION_REASONS = {"plan_not_executable", "service_not_configured"}


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
                "default_quality_repair_attempts": item.default_quality_repair_attempts,
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
    """Host-owned API defaults for content-ops planning routes.

    Falsification rules are opt-in and only apply to strict packaged reasoning.
    Each rule should be a mapping with a human-readable predicate, for example
    ``{"id": "renewal_completed", "predicate": "fresh evidence shows renewal completed"}``.
    Falsification is evaluated per generated claim, so enabling rules can add
    one LLM call per claim before any falsified claims are dropped.
    ``structured_reasoning_falsification_conservative=True`` means ambiguous or
    malformed falsification responses preserve the claim instead of invalidating
    it. ``structured_reasoning_drop_falsified=True`` validates the filtered
    surviving claim set, which can increase strict-mode validation blocks.
    """

    prefix: str = "/content-ops"
    tags: tuple[str, ...] = ("content-ops",)
    structured_reasoning_default_goal: str = "synthesize content reasoning context"
    structured_reasoning_depth: str = "L3"
    structured_reasoning_pack_name: str = "content_ops_structured"
    structured_reasoning_output_pack_names: Mapping[str, str] | None = None
    structured_reasoning_top_thesis_limit: int = 5
    structured_reasoning_max_continuations: int = 8
    structured_reasoning_require_citations: bool = True
    structured_reasoning_falsification_rules: Sequence[Mapping[str, Any]] = ()
    structured_reasoning_falsification_conservative: bool = True
    structured_reasoning_drop_falsified: bool = False
    ingestion_opportunity_table: str = "campaign_opportunities"

    def __post_init__(self) -> None:
        if not str(self.prefix or "").strip().startswith("/"):
            raise ValueError("prefix must start with /")
        if not str(self.structured_reasoning_default_goal or "").strip():
            raise ValueError("structured_reasoning_default_goal is required")
        if not str(self.structured_reasoning_pack_name or "").strip():
            raise ValueError("structured_reasoning_pack_name is required")
        output_pack_names = self.structured_reasoning_output_pack_names
        if output_pack_names is None:
            output_pack_names = {"blog_post": "content_ops_blog"}
        if not isinstance(output_pack_names, Mapping):
            raise ValueError("structured_reasoning_output_pack_names must be a mapping")
        normalized_pack_names: dict[str, str] = {}
        for output, pack_name in output_pack_names.items():
            output_key = str(output or "").strip()
            pack_value = str(pack_name or "").strip()
            if not output_key:
                raise ValueError(
                    "structured_reasoning_output_pack_names keys must be non-empty"
                )
            if not pack_value:
                raise ValueError(
                    "structured_reasoning_output_pack_names values must be non-empty"
                )
            normalized_pack_names[output_key] = pack_value
        object.__setattr__(
            self,
            "structured_reasoning_output_pack_names",
            MappingProxyType(normalized_pack_names),
        )
        if self.structured_reasoning_depth not in {"L1", "L2", "L3", "L4", "L5"}:
            raise ValueError("structured_reasoning_depth must be L1-L5")
        if self.structured_reasoning_top_thesis_limit <= 0:
            raise ValueError("structured_reasoning_top_thesis_limit must be positive")
        if self.structured_reasoning_max_continuations < 0:
            raise ValueError("structured_reasoning_max_continuations must be non-negative")
        if not isinstance(
            self.structured_reasoning_falsification_rules,
            Sequence,
        ) or isinstance(
            self.structured_reasoning_falsification_rules,
            (str, bytes, bytearray, Mapping),
        ):
            raise ValueError("structured_reasoning_falsification_rules must be a sequence")
        if (
            self.structured_reasoning_drop_falsified
            and not self.structured_reasoning_falsification_rules
        ):
            raise ValueError(
                "structured_reasoning_drop_falsified=True requires "
                "non-empty structured_reasoning_falsification_rules"
            )
        if len(self.structured_reasoning_falsification_rules) > _MAX_FALSIFICATION_RULES:
            raise ValueError(
                "structured_reasoning_falsification_rules cannot exceed "
                f"{_MAX_FALSIFICATION_RULES}"
            )
        for rule in self.structured_reasoning_falsification_rules:
            if not isinstance(rule, Mapping):
                raise ValueError(
                    "structured_reasoning_falsification_rules entries must be mappings"
                )
        if not str(self.ingestion_opportunity_table or "").strip():
            raise ValueError("ingestion_opportunity_table is required")


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
            raise RuntimeError(reasoning_validation_blocked_reason(()))
        validation = _reasoning_validation_from_context(context)
        if validation is not None and validation.get("passed") is False:
            blockers = [
                str(item).strip()
                for item in _clean_status_scalar_sequence(validation.get("blockers"))
                if str(item).strip()
            ]
            raise RuntimeError(reasoning_validation_blocked_reason(tuple(blockers)))
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

    class ContentOpsIngestionInspectModel(BaseModel):
        """Bounded API request body for offline ingestion diagnostics."""

        model_config = ConfigDict(extra="forbid")

        rows: tuple[dict[str, Any], ...] = Field(
            default_factory=tuple,
            max_length=_MAX_INGESTION_ROWS,
        )
        source_rows: bool = False
        source: str | None = Field(default="api", max_length=200)
        target_mode: str | None = Field(default="vendor_retention", max_length=80)
        max_source_text_chars: int = Field(1200, ge=1, le=_MAX_INPUT_STRING_CHARS)
        sample_limit: int = Field(3, ge=0, le=_MAX_INGESTION_SAMPLE_LIMIT)
        default_fields: dict[str, Any] = Field(default_factory=dict)

        @field_validator("rows")
        @classmethod
        def _validate_rows(
            cls,
            value: tuple[dict[str, Any], ...],
        ) -> tuple[dict[str, Any], ...]:
            for row in value:
                _validate_input_shape(row, depth=0)
            return value

        @field_validator("default_fields")
        @classmethod
        def _validate_default_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
            if len(value) > 50:
                raise ValueError("default_fields cannot contain more than 50 entries")
            _validate_input_shape(value, depth=0)
            return value

    class ContentOpsIngestionImportModel(ContentOpsIngestionInspectModel):
        """Bounded API request body for hosted opportunity import."""

        replace_existing: bool = False
        dry_run: bool = False

else:  # pragma: no cover - module import fallback when FastAPI is unavailable.
    ContentOpsRequestModel = Any
    ContentOpsIngestionInspectModel = Any
    ContentOpsIngestionImportModel = Any


def create_content_ops_control_surface_router(
    *,
    config: ContentOpsControlSurfaceApiConfig | None = None,
    execution_services_provider: ExecutionServicesProvider | None = None,
    scope_provider: ScopeProvider | None = None,
    reasoning_context_provider: ReasoningContextProvider | None = None,
    reasoning_status_provider: ReasoningStatusProvider | None = None,
    llm_provider: LLMProvider | None = None,
    opportunity_import_pool_provider: PoolProvider | None = None,
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

    @router.post("/ingestion/inspect")
    async def inspect_ingestion(
        payload: ContentOpsIngestionInspectModel = Body(...),
    ) -> dict[str, Any]:
        data = _ingestion_payload_to_mapping(payload)
        try:
            report = inspect_ingestion_rows(
                data["rows"],
                source_rows=bool(data.get("source_rows")),
                source=_clean(data.get("source")) or "api",
                target_mode=_clean(data.get("target_mode")),
                max_source_text_chars=int(data.get("max_source_text_chars") or 1200),
                sample_limit=int(data.get("sample_limit") or 0),
                default_fields=data.get("default_fields"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return report.as_dict()

    @router.post("/ingestion/import")
    async def import_ingestion(
        payload: ContentOpsIngestionImportModel = Body(...),
    ) -> dict[str, Any]:
        data = _ingestion_import_payload_to_mapping(payload)
        target_mode = _clean(data.get("target_mode")) or "vendor_retention"
        try:
            report = inspect_ingestion_rows(
                data["rows"],
                source_rows=bool(data.get("source_rows")),
                source=_clean(data.get("source")) or "api",
                target_mode=target_mode,
                max_source_text_chars=int(data.get("max_source_text_chars") or 1200),
                sample_limit=int(data.get("sample_limit") or 0),
                default_fields=data.get("default_fields"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        diagnostics = report.as_dict()
        if not report.ok:
            raise HTTPException(
                status_code=400,
                detail={
                    "reason": "ingestion_not_ready",
                    "diagnostics": diagnostics,
                },
            )
        dry_run = bool(data.get("dry_run"))
        if dry_run:
            pool = object()
        else:
            pool = await _resolve_import_pool(opportunity_import_pool_provider)
        scope = await _resolve_scope(scope_provider)
        result = await _import_campaign_opportunities_for_route(
            pool,
            report.opportunities,
            scope=scope,
            target_mode=target_mode,
            opportunity_table=resolved_config.ingestion_opportunity_table,
            replace_existing=bool(data.get("replace_existing")),
            dry_run=dry_run,
            source=report.source,
        )
        return {
            "diagnostics": diagnostics,
            "import": result.as_dict(),
        }

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
            reasoning_groups = await _structured_reasoning_contexts(
                payload_mapping,
                config=resolved_config,
                services=services,
                llm_provider=llm_provider,
            )
            for reasoning_context, reasoning_outputs in reasoning_groups:
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


async def _structured_reasoning_contexts(
    payload: Mapping[str, Any],
    *,
    config: ContentOpsControlSurfaceApiConfig,
    services: ContentOpsExecutionServices,
    llm_provider: LLMProvider | None,
) -> tuple[tuple[Any, tuple[str, ...]], ...]:
    preset = _clean(payload.get("reasoning_preset"))
    if not preset:
        return ()
    if preset in {"none", "context_only"}:
        return ()
    try:
        request = request_from_mapping(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        plan = build_generation_plan(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not plan.can_execute:
        return ()
    supported_outputs = [
        str(output)
        for output in resolve_outputs(request)
        if output in PACKAGED_REASONING_RUNTIME_OUTPUTS
    ]
    if not supported_outputs:
        raise HTTPException(
            status_code=400,
            detail=(
                "reasoning_preset currently applies only to email_campaign, "
                "blog_post, report, landing_page, and sales_brief."
            ),
        )
    reasoning_definitions: dict[str, Any] = {}
    for output in supported_outputs:
        try:
            _policy, definition = resolve_reasoning_policy(output, preset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        reasoning_definitions[output] = definition
        runtime_presets = packaged_reasoning_runtime_presets_for_output(output)
        if definition.id not in runtime_presets:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Content Ops packaged reasoning currently supports "
                    "multi_pass_structured for email_campaign, blog_post, "
                    "and landing_page, and multi_pass_structured or "
                    "multi_pass_strict for report and sales_brief."
                ),
            )
    selected_outputs = [
        output for output in supported_outputs if output in services.configured_outputs()
    ]
    if not selected_outputs:
        return ()
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
    from extracted_reasoning_core.types import (
        FalsificationPolicy,
        OutputPolicy,
        ReasoningPack,
        ReasoningPorts,
    )

    from ..services.multi_pass_reasoning_provider import (
        MultiPassCampaignReasoningProvider,
        MultiPassReasoningProviderConfig,
    )

    falsification_policy = None
    definitions = tuple(reasoning_definitions[output] for output in selected_outputs)
    if any(item.falsification for item in definitions) and config.structured_reasoning_falsification_rules:
        falsification_policy = FalsificationPolicy(
            rules=tuple(
                dict(rule)
                for rule in config.structured_reasoning_falsification_rules
            ),
            conservative=config.structured_reasoning_falsification_conservative,
        )

    groups: dict[str, list[str]] = {}
    for output in selected_outputs:
        groups.setdefault(_structured_reasoning_pack_name(output, config), []).append(output)

    provider_groups: list[tuple[Any, tuple[str, ...]]] = []
    for pack_name, outputs in groups.items():
        # The reasoning preset is request-level, so every output in this group
        # resolves to the same ReasoningPresetDefinition.
        group_definition = reasoning_definitions[outputs[0]]
        provider = MultiPassCampaignReasoningProvider(
            ports=ReasoningPorts(llm=llm),
            config=MultiPassReasoningProviderConfig(
                default_goal=config.structured_reasoning_default_goal,
                default_depth=config.structured_reasoning_depth,
                top_thesis_limit=config.structured_reasoning_top_thesis_limit,
                max_continuations=config.structured_reasoning_max_continuations,
                narrative_plan_pack=ReasoningPack(name=pack_name),
                output_policy=OutputPolicy(
                    require_citations=config.structured_reasoning_require_citations,
                ),
                falsification_policy=falsification_policy
                if group_definition.falsification else None,
                drop_falsified=bool(
                    group_definition.falsification
                    and falsification_policy is not None
                    and config.structured_reasoning_drop_falsified
                ),
                # Keep the inner provider nonblocking so strict wrappers can
                # surface validation blocker details instead of a generic None.
                block_on_validation_failure=False,
            ),
        )
        if group_definition.blocking_validation:
            provider = _BlockingReasoningContextProvider(provider)
        provider_groups.append((provider, tuple(outputs)))
    return tuple(provider_groups)


def _structured_reasoning_pack_name(
    output: str,
    config: ContentOpsControlSurfaceApiConfig,
) -> str:
    configured = config.structured_reasoning_output_pack_names or {}
    pack_name = configured.get(output)
    if pack_name:
        return str(pack_name)
    return config.structured_reasoning_pack_name


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


async def _resolve_import_pool(provider: PoolProvider | None) -> Any:
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail="Content Ops ingestion import database is not configured.",
        )
    try:
        pool = await _resolve_provider(provider)
    except Exception as exc:
        logger.warning(
            "Content Ops ingestion import pool provider failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Content Ops ingestion import database is unavailable.",
        ) from exc
    if pool is None or getattr(pool, "is_initialized", True) is False:
        raise HTTPException(
            status_code=503,
            detail="Content Ops ingestion import database is unavailable.",
        )
    return pool


async def _import_campaign_opportunities_for_route(
    db: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: TenantScope | None,
    target_mode: str,
    opportunity_table: str,
    replace_existing: bool,
    dry_run: bool,
    source: str | None,
) -> Any:
    try:
        return await _import_campaign_opportunities_atomic(
            db,
            rows,
            scope=scope,
            target_mode=target_mode,
            opportunity_table=opportunity_table,
            replace_existing=replace_existing,
            dry_run=dry_run,
            source=source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning(
            "Content Ops ingestion import failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Content Ops ingestion import failed.",
        ) from exc


async def _import_campaign_opportunities_atomic(
    db: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: TenantScope | None,
    target_mode: str,
    opportunity_table: str,
    replace_existing: bool,
    dry_run: bool,
    source: str | None,
) -> Any:
    acquire = getattr(db, "acquire", None)
    if callable(acquire):
        async with acquire() as connection:
            transaction = getattr(connection, "transaction", None)
            if callable(transaction):
                async with transaction():
                    return await _import_campaign_opportunities_direct(
                        connection,
                        rows,
                        scope=scope,
                        target_mode=target_mode,
                        opportunity_table=opportunity_table,
                        replace_existing=replace_existing,
                        dry_run=dry_run,
                        source=source,
                    )
            return await _import_campaign_opportunities_direct(
                connection,
                rows,
                scope=scope,
                target_mode=target_mode,
                opportunity_table=opportunity_table,
                replace_existing=replace_existing,
                dry_run=dry_run,
                source=source,
            )

    transaction = getattr(db, "transaction", None)
    if callable(transaction):
        async with transaction():
            return await _import_campaign_opportunities_direct(
                db,
                rows,
                scope=scope,
                target_mode=target_mode,
                opportunity_table=opportunity_table,
                replace_existing=replace_existing,
                dry_run=dry_run,
                source=source,
            )
    return await _import_campaign_opportunities_direct(
        db,
        rows,
        scope=scope,
        target_mode=target_mode,
        opportunity_table=opportunity_table,
        replace_existing=replace_existing,
        dry_run=dry_run,
        source=source,
    )


async def _import_campaign_opportunities_direct(
    db: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: TenantScope | None,
    target_mode: str,
    opportunity_table: str,
    replace_existing: bool,
    dry_run: bool,
    source: str | None,
) -> Any:
    return await import_campaign_opportunities(
        db,
        rows,
        scope=scope,
        target_mode=target_mode,
        opportunity_table=opportunity_table,
        replace_existing=replace_existing,
        dry_run=dry_run,
        normalize=False,
        source=source,
    )


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


def _ingestion_payload_to_mapping(payload: Any) -> dict[str, Any]:
    if BaseModel is not None and isinstance(payload, ContentOpsIngestionInspectModel):
        return payload.model_dump()
    try:
        return ContentOpsIngestionInspectModel.model_validate(payload).model_dump()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_detail(exc)) from exc


def _ingestion_import_payload_to_mapping(payload: Any) -> dict[str, Any]:
    if BaseModel is not None and isinstance(payload, ContentOpsIngestionImportModel):
        return payload.model_dump()
    try:
        return ContentOpsIngestionImportModel.model_validate(payload).model_dump()
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
    "ContentOpsIngestionImportModel",
    "ContentOpsRequestModel",
    "create_content_ops_control_surface_router",
]
