"""FastAPI router factory for AI Content Ops control-surface previews."""

from __future__ import annotations

import asyncio
import http.client
import json
import logging
import math
import re
import ssl
import tempfile
import ipaddress
import socket
import urllib.error
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

try:
    from fastapi import APIRouter, Body, File, Form, HTTPException, Path as PathParam, Query, Request, UploadFile
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
except ImportError as exc:  # pragma: no cover - exercised in dependency-light CI.
    APIRouter = None
    Body = None
    HTTPException = None
    File = None
    Form = None
    Query = None
    Request = None
    UploadFile = None
    BaseModel = None
    ConfigDict = None
    Field = None
    ValidationError = None
    field_validator = None
    _FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

from ..campaign_ports import TenantScope
from ..campaign_customer_data import CsvCustomerDataParseError
from ..brand_voice import BrandVoiceProfile, brand_voice_profile_from_mapping
from ..campaign_postgres_import import import_campaign_opportunities
from ..campaign_source_adapters import load_csv_source_rows_result_from_file
from ..campaign_source_adapters import source_material_to_source_rows
from ..content_ops_execution import (
    ContentOpsExecutionServices,
    execute_content_ops_from_mapping,
)
from ..content_ops_usage_summary import summarize_content_ops_llm_usage
from ..content_ops_input_provider import (
    ContentOpsInputProvider,
    merge_content_ops_input_package,
)
from ..content_ops_cache_policy import normalize_content_ops_cache_policy
from ..deflection_report_access import (
    DeflectionDeltaReadError,
    DeflectionReportArtifactStore,
    deflection_delta_read_payload,
    fetch_paid_deflection_delta,
)
from ..control_surfaces import (
    OUTPUT_CATALOG,
    PRESETS,
    UsageBudgetEvaluation,
    evaluate_usage_budget,
    preview_from_mapping,
    request_from_mapping,
    retry_adjusted_unit_cost_usd,
    resolve_outputs,
)
from ..generation_plan import build_generation_plan, build_generation_plan_from_mapping
from ..faq_deflection_report import (
    DEFAULT_DEFLECTION_SNAPSHOT_TOP_N,
    DEFAULT_DEFLECTION_TEASER_PREVIEW_COUNT,
    DEFLECTION_EVIDENCE_EXPORT_SCHEMA_VERSION,
    DEFLECTION_REPORT_SCHEMA_VERSION,
    build_deflection_snapshot,
    deflection_report_model_contract_shape,
    scrub_deflection_report_payload,
)
from ..landing_page_input_contract import landing_page_seo_geo_aeo_input_contracts
from ..ingestion_diagnostics import inspect_ingestion_file, inspect_ingestion_rows
from ..landing_page_repair_contract import (
    LANDING_PAGE_QUALITY_REPAIR_INPUT,
    landing_page_quality_repair_input_contract,
)
from ..reasoning_policy import (
    PACKAGED_REASONING_RUNTIME_OUTPUTS,
    ReasoningPreset,
    packaged_reasoning_runtime_presets_for_output,
    resolve_reasoning_policy,
)
from ..reasoning_signals import reasoning_validation_blocked_reason
from ..ticket_faq_input_contract import ticket_faq_input_contracts
from ..support_ticket_input_package import build_support_ticket_input_package
from ..support_ticket_zendesk_thread import (
    load_zendesk_full_thread_rows_from_json_bytes,
    load_zendesk_full_thread_rows_from_json_file,
)

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

_DEFLECTION_REPORT_RETENTION_DAYS = 30
_DEFLECTION_CHECKOUT_OPEN_SESSION_GRACE = timedelta(hours=24)
ReasoningStatusProvider = Callable[
    [],
    Mapping[str, Any] | None | Awaitable[Mapping[str, Any] | None],
]
LLMProvider = Callable[[], Any | Awaitable[Any]]
PoolProvider = Callable[[], Any | Awaitable[Any]]
DeflectionReportStoreProvider = Callable[
    [],
    DeflectionReportArtifactStore | Awaitable[DeflectionReportArtifactStore],
]
ImportAdmissionProvider = Callable[[], Any | Awaitable[Any]]
CachePolicyDefaultProvider = Callable[
    [TenantScope],
    str | None | Awaitable[str | None],
]
BrandVoiceProfileProvider = Callable[
    [TenantScope, str],
    (
        BrandVoiceProfile
        | Mapping[str, Any]
        | None
        | Awaitable[BrandVoiceProfile | Mapping[str, Any] | None]
    ),
]
InputProvider = ContentOpsInputProvider

logger = logging.getLogger(__name__)

DEFLECTION_PROCESS_CONTRACT_SCHEMA_VERSION = "deflection_report_process.v1"
DEFLECTION_PROCESS_CONTRACT_PATH = "/deflection-reports/process-contract"
_MAX_INPUT_KEYS = 50
_MAX_INPUT_DEPTH = 6
_MAX_INPUT_STRING_CHARS = 10000
_MAX_INGESTION_ROWS = 1000
_MAX_FILE_INGESTION_ROWS = 10000
_MAX_INGESTION_FILE_BYTES = 25 * 1024 * 1024
_MAX_DEFLECTION_SUBMIT_BLOB_BYTES = 50 * 1024 * 1024
_MAX_DEFLECTION_SUBMIT_ROWS = _MAX_DEFLECTION_SUBMIT_BLOB_BYTES
_MAX_DEFLECTION_SUBMIT_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
_DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_INGESTION_SAMPLE_LIMIT = 25
_DEFLECTION_SUBMIT_FETCH_TIMEOUT_SECONDS = 15
_DEFLECTION_SUBMIT_OUTPUTS = ("faq_deflection_report",)
_DEFLECTION_SUBMIT_INTERNAL_TOKEN_KEY = "_deflection_submit_internal_token"
_DEFLECTION_SUBMIT_INTERNAL_TOKEN = object()
_DEFLECTION_SUBMIT_PLATFORMS = frozenset({
    "zendesk",
    "intercom",
    "help_scout",
    "other",
})
_DEFLECTION_SUBMIT_IMPORTER_MODES = frozenset({"csv", "full_thread"})
_DEFLECTION_REPORT_SEARCH_DEFAULT_LIMIT = 5
_DEFLECTION_REPORT_SEARCH_MAX_LIMIT = 10
_DEFLECTION_REPORT_SEARCH_MAX_QUERY_CHARS = 300
_DEFLECTION_REPORT_SEARCH_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DEFLECTION_REPORT_SEARCH_TEXT_FIELDS = (
    "topic",
    "question",
    "answer",
    "when_to_contact_support",
    "answer_evidence_status",
)
_DEFLECTION_REPORT_SEARCH_NUMBER_FIELDS = (
    "ticket_count",
    "opportunity_score",
)
_DEFLECTION_REPORT_SEARCH_STRING_ARRAY_FIELDS = (
    "steps",
    "action_items",
    "source_ids",
    "source_labels",
)


@dataclass(frozen=True)
class _DeflectionSubmitRowsLoad:
    rows: list[Any]
    byte_count: int
    warnings: tuple[dict[str, Any], ...]
    source_row_count: int | None = None

    def __iter__(self):
        yield self.rows
        yield self.byte_count
        yield self.warnings


_UPLOAD_FILE_FORMATS = ("auto", "json", "jsonl", "csv")
_MAX_REASONING_STATUS_LIST_ITEMS = 20
_MAX_FALSIFICATION_RULES = 20
_SAFE_EXECUTION_REASONS = {"plan_not_executable", "service_not_configured"}
_INPUT_PROVIDER_RESPONSE_METADATA_KEYS = frozenset({
    "source",
    "source_period",
    "source_row_count",
    "included_row_count",
    "skipped_row_count",
    "truncated_row_count",
    "support_ticket_resolution_evidence_present",
    "support_ticket_resolution_evidence_count",
})
_FAQ_SOURCE_MATERIAL_LIMITED_OUTPUTS = frozenset({
    "faq_markdown",
    "faq_deflection_report",
})
_SOURCE_MATERIAL_ROW_LIST_KEYS = {
    "sources",
    "opportunities",
    "complaints",
    "search_logs",
    "search_queries",
    "support_tickets",
    "tickets",
    "cases",
    "conversations",
    "feedback",
    "rows",
    "items",
}

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
        "ingestion_limits": {
            "inline_rows": {
                "max_rows": _MAX_INGESTION_ROWS,
                "deprecated": True,
            },
            "file_upload": {
                "max_file_bytes": _MAX_INGESTION_FILE_BYTES,
                "max_rows": _MAX_FILE_INGESTION_ROWS,
                "supported_formats": _UPLOAD_FILE_FORMATS,
            },
            "max_source_text_chars": _MAX_INPUT_STRING_CHARS,
            "max_sample_limit": _MAX_INGESTION_SAMPLE_LIMIT,
        },
        "execute_limits": {
            "max_source_material_rows": _MAX_INGESTION_ROWS,
            "large_upload_strategy": "background_or_offline",
        },
        "input_contracts": {
            LANDING_PAGE_QUALITY_REPAIR_INPUT: landing_page_quality_repair_input_contract(),
            **landing_page_seo_geo_aeo_input_contracts(),
            **ticket_faq_input_contracts(),
        },
    }


_STATIC_CATALOG_PAYLOAD: Mapping[str, Any] = _build_static_catalog_payload()


def _compose_describe_response(
    *,
    static: Mapping[str, Any],
    configured_outputs: frozenset[str],
    execution_configured: bool,
    execute_max_concurrency: int,
    faq_execute_max_source_material_rows: int,
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
            "limits": {
                "max_concurrency": execute_max_concurrency,
                "max_source_material_rows": static["execute_limits"][
                    "max_source_material_rows"
                ],
                "faq_max_source_material_rows": faq_execute_max_source_material_rows,
                "large_upload_strategy": static["execute_limits"][
                    "large_upload_strategy"
                ],
            },
        },
        "reasoning": dict(reasoning_status),
        "ingestion_profiles": list(static["ingestion_profiles"]),
        "ingestion_limits": {
            "inline_rows": dict(static["ingestion_limits"]["inline_rows"]),
            "file_upload": {
                **static["ingestion_limits"]["file_upload"],
                "supported_formats": list(
                    static["ingestion_limits"]["file_upload"]["supported_formats"]
                ),
            },
            "max_source_text_chars": static["ingestion_limits"]["max_source_text_chars"],
            "max_sample_limit": static["ingestion_limits"]["max_sample_limit"],
        },
        "input_contracts": {
            key: dict(value)
            for key, value in static["input_contracts"].items()
        },
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
    execute_max_concurrency: int = 8
    faq_execute_max_source_material_rows: int = _MAX_INGESTION_ROWS
    deflection_snapshot_top_n: int = DEFAULT_DEFLECTION_SNAPSHOT_TOP_N
    deflection_snapshot_teaser_preview_count: int = (
        DEFAULT_DEFLECTION_TEASER_PREVIEW_COUNT
    )
    deflection_checkout_amount_cents: int = 150000
    deflection_checkout_allowed_amount_cents: str = ""
    deflection_checkout_currency: str = "usd"
    deflection_checkout_price_id: str = ""
    ingestion_import_max_concurrency: int = 8

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
        if self.execute_max_concurrency <= 0:
            raise ValueError("execute_max_concurrency must be positive")
        if self.faq_execute_max_source_material_rows <= 0:
            raise ValueError("faq_execute_max_source_material_rows must be positive")
        if self.faq_execute_max_source_material_rows > _MAX_INGESTION_ROWS:
            raise ValueError(
                "faq_execute_max_source_material_rows cannot exceed "
                f"{_MAX_INGESTION_ROWS}"
            )
        if self.deflection_snapshot_top_n <= 0:
            raise ValueError("deflection_snapshot_top_n must be positive")
        if self.deflection_snapshot_teaser_preview_count < 0:
            raise ValueError(
                "deflection_snapshot_teaser_preview_count must be non-negative"
            )
        if self.deflection_checkout_amount_cents < 0:
            raise ValueError("deflection_checkout_amount_cents cannot be negative")
        if self.ingestion_import_max_concurrency <= 0:
            raise ValueError("ingestion_import_max_concurrency must be positive")
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


class _ExecuteConcurrencyGate:
    def __init__(self, max_concurrency: int) -> None:
        self.max_concurrency = int(max_concurrency)
        self._in_flight = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            if self._in_flight >= self.max_concurrency:
                return False
            self._in_flight += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1


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
        variant_count: int = Field(1, ge=1)
        max_cost_usd: float | None = Field(default=None, gt=0)
        account_usage_budget_usd: float | None = Field(default=None, gt=0)
        account_usage_budget_days: int = Field(7, ge=1, le=90)
        content_ops_cache_policy: str | None = Field(default=None, max_length=40)
        brand_voice_profile_id: str | None = Field(default=None, max_length=120)
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

        @field_validator("content_ops_cache_policy")
        @classmethod
        def _normalize_content_ops_cache_policy(cls, value: Any) -> str | None:
            try:
                return normalize_content_ops_cache_policy(value)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc

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
        include_source_material: bool = False

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

    class DeflectionReportPaidModel(BaseModel):
        """Authenticated paid-release marker for a generated report."""

        model_config = ConfigDict(extra="forbid")

        payment_reference: str | None = Field(default=None, max_length=200)

    class DeflectionReportSearchModel(BaseModel):
        """Request-scoped uploaded report search body."""

        model_config = ConfigDict(extra="forbid")

        q: str = Field(
            ...,
            min_length=1,
            max_length=_DEFLECTION_REPORT_SEARCH_MAX_QUERY_CHARS,
        )
        limit: int | None = Field(default=None, ge=1)

        @field_validator("q")
        @classmethod
        def _validate_query(cls, value: Any) -> str:
            text = _clean(value)
            if not text:
                raise ValueError("q is required")
            return str(text)

    class _DeflectionReportSubmitFieldsModel(BaseModel):
        """Shared portfolio submit metadata fields."""

        model_config = ConfigDict(extra="forbid")

        support_platform: str = Field(..., min_length=1, max_length=80)
        company_name: str = Field(..., min_length=1, max_length=200)
        contact_email: str = Field(..., min_length=3, max_length=320)
        limit: int | None = Field(default=None, ge=1, le=_MAX_DEFLECTION_SUBMIT_ROWS)
        importer_mode: str = Field(default="csv", min_length=1, max_length=40)

        @field_validator("support_platform")
        @classmethod
        def _validate_support_platform(cls, value: Any) -> str:
            text = (_clean(value) or "").lower().replace("-", "_")
            if text not in _DEFLECTION_SUBMIT_PLATFORMS:
                raise ValueError(
                    "support_platform must be one of: zendesk, intercom, "
                    "help_scout, other"
                )
            return text

        @field_validator("importer_mode")
        @classmethod
        def _validate_importer_mode(cls, value: Any) -> str:
            text = (_clean(value) or "csv").lower().replace("-", "_")
            if text not in _DEFLECTION_SUBMIT_IMPORTER_MODES:
                raise ValueError("importer_mode must be one of: csv, full_thread")
            return text

        @field_validator("company_name", "contact_email")
        @classmethod
        def _validate_required_text(cls, value: Any) -> str:
            text = _clean(value)
            if not text:
                raise ValueError("field is required")
            return str(text)

        @field_validator("contact_email")
        @classmethod
        def _validate_contact_email(cls, value: str) -> str:
            if "@" not in value or value.startswith("@") or value.endswith("@"):
                raise ValueError("contact_email must be an email address")
            return value

    class DeflectionReportSubmitModel(_DeflectionReportSubmitFieldsModel):
        """Portfolio-owned URL handoff for a paid FAQ deflection report."""

        blob_url: str = Field(..., min_length=1, max_length=2048)

        @field_validator("blob_url")
        @classmethod
        def _validate_blob_url(cls, value: Any) -> str:
            return _validate_https_blob_url(value)

    class DeflectionReportSubmitFieldsModel(_DeflectionReportSubmitFieldsModel):
        """Portfolio-owned multipart handoff fields for a paid FAQ deflection report."""

        pass

else:  # pragma: no cover - module import fallback when FastAPI is unavailable.
    ContentOpsRequestModel = Any
    ContentOpsIngestionInspectModel = Any
    ContentOpsIngestionImportModel = Any
    DeflectionReportPaidModel = Any
    DeflectionReportSearchModel = Any
    DeflectionReportSubmitModel = Any
    DeflectionReportSubmitFieldsModel = Any


def create_content_ops_control_surface_router(
    *,
    config: ContentOpsControlSurfaceApiConfig | None = None,
    execution_services_provider: ExecutionServicesProvider | None = None,
    scope_provider: ScopeProvider | None = None,
    reasoning_context_provider: ReasoningContextProvider | None = None,
    reasoning_status_provider: ReasoningStatusProvider | None = None,
    llm_provider: LLMProvider | None = None,
    input_provider: InputProvider | None = None,
    deflection_report_store_provider: DeflectionReportStoreProvider | None = None,
    opportunity_import_pool_provider: PoolProvider | None = None,
    usage_pool_provider: PoolProvider | None = None,
    usage_dependencies: Sequence[Any] | None = None,
    deflection_report_public_dependencies: Sequence[Any] | None = None,
    deflection_report_paid_dependencies: Sequence[Any] | None = None,
    ingestion_import_admission_provider: ImportAdmissionProvider | None = None,
    cache_policy_default_provider: CachePolicyDefaultProvider | None = None,
    brand_voice_profile_provider: BrandVoiceProfileProvider | None = None,
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
    execute_gate = _ExecuteConcurrencyGate(resolved_config.execute_max_concurrency)
    ingestion_import_gate = _ExecuteConcurrencyGate(
        resolved_config.ingestion_import_max_concurrency
    )
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
            execute_max_concurrency=execute_gate.max_concurrency,
            faq_execute_max_source_material_rows=(
                resolved_config.faq_execute_max_source_material_rows
            ),
            reasoning_status=reasoning_status,
        )

    @router.get("/usage/summary", dependencies=list(usage_dependencies or ()))
    async def usage_summary(
        days: int = Query(default=7, ge=1, le=90),
        asset_type: str | None = Query(default=None, max_length=80),
        run_id: str | None = Query(default=None, max_length=200),
        request_id: str | None = Query(default=None, max_length=200),
    ) -> dict[str, Any]:
        pool = await _resolve_usage_pool(usage_pool_provider)
        return await summarize_content_ops_llm_usage(
            pool,
            days=days,
            asset_type=asset_type,
            run_id=run_id,
            request_id=request_id,
        )

    @router.get("/usage/summary/tenant")
    async def tenant_usage_summary(
        days: int = Query(default=7, ge=1, le=90),
        asset_type: str | None = Query(default=None, max_length=80),
        run_id: str | None = Query(default=None, max_length=200),
        request_id: str | None = Query(default=None, max_length=200),
    ) -> dict[str, Any]:
        pool = await _resolve_usage_pool(usage_pool_provider)
        scope = await _resolve_scope(scope_provider)
        return await summarize_content_ops_llm_usage(
            pool,
            days=days,
            account_id=_required_scope_account_id(scope),
            asset_type=asset_type,
            run_id=run_id,
            request_id=request_id,
        )

    public_deflection_dependencies = list(deflection_report_public_dependencies or ())

    @router.get(
        DEFLECTION_PROCESS_CONTRACT_PATH,
        dependencies=public_deflection_dependencies,
    )
    async def deflection_report_process_contract() -> dict[str, Any]:
        return _deflection_report_process_contract_payload(resolved_config.prefix)

    @router.get(
        "/deflection-reports/{request_id}/snapshot",
        dependencies=public_deflection_dependencies,
    )
    async def deflection_report_snapshot(
        request_id: str = PathParam(..., min_length=1, max_length=200),
    ) -> dict[str, Any]:
        store = await _resolve_deflection_report_store(deflection_report_store_provider)
        scope = await _resolve_scope(scope_provider)
        snapshot = await store.get_snapshot(
            account_id=_required_scope_account_id(scope),
            request_id=request_id,
        )
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Deflection report not found.")
        return snapshot

    @router.get(
        "/deflection-reports/{request_id}/artifact",
        dependencies=public_deflection_dependencies,
    )
    async def deflection_report_artifact(
        request_id: str = PathParam(..., min_length=1, max_length=200),
    ) -> dict[str, Any]:
        store = await _resolve_deflection_report_store(deflection_report_store_provider)
        scope = await _resolve_scope(scope_provider)
        record = await store.get_artifact_record(
            account_id=_required_scope_account_id(scope),
            request_id=request_id,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Deflection report not found.")
        if not record.paid:
            raise HTTPException(status_code=403, detail="Deflection report is locked.")
        return dict(record.artifact or {})

    @router.get(
        "/deflection-reports/{request_id}/report-model",
        dependencies=public_deflection_dependencies,
    )
    async def deflection_report_model(
        request_id: str = PathParam(..., min_length=1, max_length=200),
    ) -> dict[str, Any]:
        store = await _resolve_deflection_report_store(deflection_report_store_provider)
        scope = await _resolve_scope(scope_provider)
        record = await store.get_artifact_record(
            account_id=_required_scope_account_id(scope),
            request_id=request_id,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Deflection report not found.")
        if not record.paid:
            raise HTTPException(status_code=403, detail="Deflection report is locked.")
        model = record.report_model()
        if model is None:
            raise HTTPException(
                status_code=404,
                detail="Deflection report model is not available.",
            )
        return model

    @router.get(
        "/deflection-reports/{request_id}/delta",
        dependencies=public_deflection_dependencies,
    )
    async def deflection_report_delta(
        request_id: str = PathParam(..., min_length=1, max_length=200),
        baseline_request_id: str | None = Query(default=None, max_length=200),
    ) -> dict[str, Any]:
        store = await _resolve_deflection_report_store(deflection_report_store_provider)
        scope = await _resolve_scope(scope_provider)
        try:
            record = await fetch_paid_deflection_delta(
                store,
                account_id=_required_scope_account_id(scope),
                current_request_id=request_id,
                baseline_request_id=baseline_request_id,
            )
        except DeflectionDeltaReadError as exc:
            status = 403 if exc.code.endswith("_locked") else 404
            if exc.code in {
                "account_id_required",
                "current_request_id_required",
                "invalid_report_pair",
            }:
                status = 400
            if exc.code == "unsupported_delta_schema":
                status = 409
            raise HTTPException(status_code=status, detail=exc.message) from exc
        return deflection_delta_read_payload(record)

    @router.post(
        "/deflection-reports/{request_id}/search",
        dependencies=public_deflection_dependencies,
    )
    async def search_deflection_report(
        payload: DeflectionReportSearchModel = Body(...),
        request_id: str = PathParam(..., min_length=1, max_length=200),
    ) -> dict[str, Any]:
        query = _deflection_report_search_query(payload.q)
        limit = _deflection_report_search_limit(payload.limit)
        store = await _resolve_deflection_report_store(deflection_report_store_provider)
        scope = await _resolve_scope(scope_provider)
        record = await store.get_artifact_record(
            account_id=_required_scope_account_id(scope),
            request_id=request_id,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Deflection report not found.")
        if not record.paid:
            raise HTTPException(status_code=403, detail="Deflection report is locked.")
        return _search_deflection_report_artifact(
            record.artifact,
            query=query,
            limit=limit,
        )

    @router.post(
        "/deflection-reports/{request_id}/checkout-authorization",
        dependencies=public_deflection_dependencies,
    )
    async def deflection_report_checkout_authorization(
        request_id: str = PathParam(..., min_length=1, max_length=200),
    ) -> dict[str, Any]:
        checkout = _deflection_checkout_terms(resolved_config)
        store = await _resolve_deflection_report_store(deflection_report_store_provider)
        scope = await _resolve_scope(scope_provider)
        record = await store.get_artifact_record(
            account_id=_required_scope_account_id(scope),
            request_id=request_id,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Deflection report not found.")
        if record.paid:
            raise HTTPException(
                status_code=409,
                detail="Deflection report is already paid.",
            )
        if not record.artifact:
            raise HTTPException(
                status_code=409,
                detail="Deflection report artifact is not available.",
            )
        if not _deflection_report_checkout_inside_retention_window(record.created_at):
            raise HTTPException(
                status_code=409,
                detail="Deflection report checkout window expired.",
            )
        return {
            "request_id": request_id,
            "status": "authorized",
            "checkout": checkout,
        }

    @router.post(
        "/deflection-reports/{request_id}/paid",
        dependencies=list(deflection_report_paid_dependencies or ()),
    )
    async def mark_deflection_report_paid(
        payload: DeflectionReportPaidModel = Body(...),
        request_id: str = PathParam(..., min_length=1, max_length=200),
    ) -> dict[str, Any]:
        store = await _resolve_deflection_report_store(deflection_report_store_provider)
        scope = await _resolve_scope(scope_provider)
        marked = await store.mark_paid(
            account_id=_required_scope_account_id(scope),
            request_id=request_id,
            payment_reference=_clean(getattr(payload, "payment_reference", None)),
        )
        if not marked:
            raise HTTPException(status_code=404, detail="Deflection report not found.")
        return {"request_id": request_id, "paid": True}

    @router.delete(
        "/deflection-reports/{request_id}",
        dependencies=public_deflection_dependencies,
        status_code=204,
    )
    async def delete_deflection_report(
        request_id: str = PathParam(..., min_length=1, max_length=200),
    ) -> None:
        store = await _resolve_deflection_report_store(deflection_report_store_provider)
        scope = await _resolve_scope(scope_provider)
        await store.delete_report(
            account_id=_required_scope_account_id(scope),
            request_id=request_id,
        )
        return None

    @router.post("/preview")
    async def preview_generation(
        payload: ContentOpsRequestModel = Body(...),
    ) -> dict[str, Any]:
        payload_mapping = await _payload_with_input_provider(
            _payload_to_mapping(payload, exclude_unset=input_provider is not None),
            input_provider=input_provider,
            scope_provider=scope_provider,
        )
        payload_mapping = await _payload_with_cache_policy_default(
            payload_mapping,
            cache_policy_default_provider=cache_policy_default_provider,
            scope_provider=scope_provider,
        )
        payload_mapping = await _payload_with_brand_voice_profile(
            payload_mapping,
            brand_voice_profile_provider=brand_voice_profile_provider,
            scope_provider=scope_provider,
        )
        budget_evaluation = await _evaluate_account_usage_budget(
            payload_mapping,
            usage_pool_provider=usage_pool_provider,
            scope_provider=scope_provider,
        )
        return _with_input_provider_diagnostics(
            _apply_usage_budget_to_preview(
                preview_from_mapping(payload_mapping),
                budget_evaluation,
            ),
            payload_mapping,
        )

    @router.post("/plan")
    async def plan_generation(
        payload: ContentOpsRequestModel = Body(...),
    ) -> dict[str, Any]:
        try:
            payload_mapping = await _payload_with_input_provider(
                _payload_to_mapping(payload, exclude_unset=input_provider is not None),
                input_provider=input_provider,
                scope_provider=scope_provider,
            )
            payload_mapping = await _payload_with_cache_policy_default(
                payload_mapping,
                cache_policy_default_provider=cache_policy_default_provider,
                scope_provider=scope_provider,
            )
            payload_mapping = await _payload_with_brand_voice_profile(
                payload_mapping,
                brand_voice_profile_provider=brand_voice_profile_provider,
                scope_provider=scope_provider,
            )
            budget_evaluation = await _evaluate_account_usage_budget(
                payload_mapping,
                usage_pool_provider=usage_pool_provider,
                scope_provider=scope_provider,
            )
            return _with_input_provider_diagnostics(
                _apply_usage_budget_to_plan(
                    build_generation_plan_from_mapping(payload_mapping),
                    budget_evaluation,
                ),
                payload_mapping,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/ingestion/inspect", deprecated=True)
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
        return _ingestion_diagnostics_response(
            report,
            include_source_material=_flag_enabled(data.get("include_source_material")),
        )

    @router.post("/ingestion/import", deprecated=True)
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
        diagnostics = _ingestion_diagnostics_response(
            report,
            include_source_material=_flag_enabled(data.get("include_source_material")),
        )
        if not report.ok:
            raise HTTPException(
                status_code=400,
                detail={
                    "reason": "ingestion_not_ready",
                    "diagnostics": diagnostics,
                },
            )
        dry_run = bool(data.get("dry_run"))
        result = await _import_ingestion_rows_with_admission(
            report.opportunities,
            import_gate=ingestion_import_gate,
            import_gate_provider=ingestion_import_admission_provider,
            pool_provider=opportunity_import_pool_provider,
            scope_provider=scope_provider,
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

    @router.post("/ingestion/files/inspect")
    async def inspect_ingestion_file_upload(
        file: UploadFile = File(...),
        source_rows: bool = Form(False),
        source: str | None = Form(None),
        target_mode: str | None = Form("vendor_retention"),
        file_format: str = Form("auto"),
        max_source_text_chars: int = Form(1200),
        sample_limit: int = Form(3),
        default_fields: str | None = Form(None),
        include_source_material: bool = Form(False),
    ) -> dict[str, Any]:
        report = await _inspect_uploaded_ingestion_file(
            file,
            source_rows=source_rows,
            source=source,
            target_mode=target_mode,
            file_format=file_format,
            max_source_text_chars=max_source_text_chars,
            sample_limit=sample_limit,
            default_fields=default_fields,
        )
        return _file_ingestion_response(
            report,
            source=source,
            include_source_material=_flag_enabled(include_source_material),
        )

    @router.post("/ingestion/files/import")
    async def import_ingestion_file_upload(
        file: UploadFile = File(...),
        source_rows: bool = Form(False),
        source: str | None = Form(None),
        target_mode: str | None = Form("vendor_retention"),
        file_format: str = Form("auto"),
        max_source_text_chars: int = Form(1200),
        sample_limit: int = Form(3),
        default_fields: str | None = Form(None),
        replace_existing: bool = Form(False),
        dry_run: bool = Form(False),
        include_source_material: bool = Form(False),
    ) -> dict[str, Any]:
        target_mode_value = _clean(target_mode) or "vendor_retention"
        report = await _inspect_uploaded_ingestion_file(
            file,
            source_rows=source_rows,
            source=source,
            target_mode=target_mode_value,
            file_format=file_format,
            max_source_text_chars=max_source_text_chars,
            sample_limit=sample_limit,
            default_fields=default_fields,
        )
        diagnostics = _file_ingestion_response(
            report,
            source=source,
            include_source_material=_flag_enabled(include_source_material),
        )
        if not report.ok:
            raise HTTPException(
                status_code=400,
                detail={
                    "reason": "ingestion_not_ready",
                    "diagnostics": diagnostics,
                },
            )
        result = await _import_ingestion_rows_with_admission(
            report.opportunities,
            import_gate=ingestion_import_gate,
            import_gate_provider=ingestion_import_admission_provider,
            pool_provider=opportunity_import_pool_provider,
            scope_provider=scope_provider,
            target_mode=target_mode_value,
            opportunity_table=resolved_config.ingestion_opportunity_table,
            replace_existing=bool(replace_existing),
            dry_run=bool(dry_run),
            source=_clean(source) or report.source,
        )
        return {
            "diagnostics": diagnostics,
            "import": result.as_dict(),
        }

    @router.post("/execute")
    async def execute_generation(
        payload: ContentOpsRequestModel = Body(...),
    ) -> dict[str, Any]:
        payload_mapping = _payload_to_mapping(payload, exclude_unset=input_provider is not None)
        internal_deflection_submit = (
            payload_mapping.pop(_DEFLECTION_SUBMIT_INTERNAL_TOKEN_KEY, None)
            is _DEFLECTION_SUBMIT_INTERNAL_TOKEN
        )
        if not await execute_gate.acquire():
            raise HTTPException(
                status_code=429,
                detail={
                    "reason": "content_ops_execute_at_capacity",
                    "max_concurrency": execute_gate.max_concurrency,
                },
            )
        try:
            scope = await _resolve_scope(scope_provider)
            payload_mapping = await _payload_with_input_provider(
                payload_mapping,
                input_provider=input_provider,
                scope=scope,
            )
            payload_mapping = await _payload_with_cache_policy_default(
                payload_mapping,
                cache_policy_default_provider=cache_policy_default_provider,
                scope=scope,
            )
            payload_mapping = await _payload_with_brand_voice_profile(
                payload_mapping,
                brand_voice_profile_provider=brand_voice_profile_provider,
                scope=scope,
            )
            if not internal_deflection_submit:
                _enforce_faq_execute_source_material_limit(
                    payload_mapping,
                    max_rows=resolved_config.faq_execute_max_source_material_rows,
                )
            budget_evaluation = await _evaluate_account_usage_budget(
                payload_mapping,
                usage_pool_provider=usage_pool_provider,
                scope_provider=scope_provider,
                scope=scope,
            )
            if budget_evaluation is not None and budget_evaluation.exceeded:
                raise HTTPException(
                    status_code=400,
                    detail=_budget_exceeded_error(budget_evaluation),
                )
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
            request_id = f"content-ops-{uuid4().hex}"
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
            try:
                result = await execute_content_ops_from_mapping(
                    payload_mapping,
                    services=services,
                    scope=scope,
                    trace_metadata={"request_id": request_id},
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            result = _sanitize_execution_result(result)
            result = await _gate_deflection_report_artifacts(
                result,
                store_provider=deflection_report_store_provider,
                scope=scope,
                request_id=request_id,
                top_n=resolved_config.deflection_snapshot_top_n,
                teaser_preview_count=(
                    resolved_config.deflection_snapshot_teaser_preview_count
                ),
                preview_summary_metadata=_deflection_resolution_preview_summary(
                    payload_mapping
                ),
                delivery_email=_delivery_email_from_payload(payload_mapping),
            )
            result = _with_input_provider_diagnostics(result, payload_mapping)
            result["request_id"] = request_id
            usage_summary = await _execute_usage_summary(
                usage_pool_provider=usage_pool_provider,
                scope=scope,
                request_id=request_id,
            )
            if usage_summary is not None:
                result["usage_summary"] = usage_summary
            if result["status"] == "blocked":
                raise HTTPException(status_code=400, detail=result)
            if result["status"] == "failed":
                raise HTTPException(status_code=502, detail=result)
            if result["status"] == "partial":
                raise HTTPException(status_code=207, detail=result)
            return result
        finally:
            await execute_gate.release()

    @router.post(
        "/deflection-reports/submit",
        dependencies=public_deflection_dependencies,
    )
    async def submit_deflection_report(
        request: Request,
    ) -> dict[str, Any]:
        (
            data,
            rows,
            byte_count,
            byte_count_key,
            csv_load_warnings,
            parser_source_row_count,
        ) = await _load_deflection_submit_rows_from_request(
            request,
            max_bytes=_MAX_DEFLECTION_SUBMIT_BLOB_BYTES,
        )
        if not rows:
            importer_mode = _clean(data.get("importer_mode")) or "csv"
            raise HTTPException(
                status_code=400,
                detail={
                    "reason": (
                        "deflection_submit_empty_full_thread"
                        if importer_mode == "full_thread"
                        else "deflection_submit_empty_csv"
                    ),
                    "message": (
                        "Submitted Zendesk full-thread JSON did not contain "
                        "support-ticket rows."
                        if importer_mode == "full_thread"
                        else "Submitted CSV did not contain support-ticket rows."
                    ),
                },
            )
        source_row_count = parser_source_row_count or len(rows)
        loaded_row_count = source_row_count
        loaded_included_row_count = len(rows)
        rows, language_filtered_row_count = _deflection_submit_english_rows(rows)
        if not rows:
            raise HTTPException(
                status_code=400,
                detail={
                    "reason": "deflection_submit_no_english_rows",
                    "message": "Submitted CSV did not contain English support-ticket rows.",
                    "source_row_count": loaded_row_count,
                    "language_filtered_row_count": language_filtered_row_count,
                },
            )
        max_rows = _deflection_submit_max_rows(data.get("limit"), source_row_count)
        submitted_rows = _deflection_submit_rows_with_defaults(data, rows)[:max_rows]
        truncated_row_count = _deflection_submit_truncated_row_count(
            source_row_count=source_row_count,
            loaded_included_row_count=loaded_included_row_count,
            eligible_row_count=len(rows),
            submitted_row_count=len(submitted_rows),
            parser_source_row_count=parser_source_row_count,
        )
        title = _deflection_submit_title(data)
        package = build_support_ticket_input_package(
            submitted_rows,
            provider="portfolio_deflection_submit",
            outputs=_DEFLECTION_SUBMIT_OUTPUTS,
            max_rows=max_rows,
            campaign_name=title,
        )
        included_row_count = int(package.metadata.get("included_row_count") or 0)
        if included_row_count <= 0:
            source_label = (
                "Zendesk full-thread JSON"
                if _clean(data.get("importer_mode")) == "full_thread"
                else "Blob CSV"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "reason": "deflection_submit_no_usable_rows",
                    "message": (
                        f"{source_label} did not include usable support-ticket wording."
                    ),
                    "source_row_count": source_row_count,
                    "max_source_material_rows": max_rows,
                },
            )

        execute_inputs = dict(package.inputs)
        execute_inputs.update({
            "deflection_report_title": title,
            "company_name": data["company_name"],
            "contact_email": data["contact_email"],
            "support_platform": data["support_platform"],
        })
        result = await execute_generation({
            _DEFLECTION_SUBMIT_INTERNAL_TOKEN_KEY: _DEFLECTION_SUBMIT_INTERNAL_TOKEN,
            "outputs": list(_DEFLECTION_SUBMIT_OUTPUTS),
            "limit": 1,
            "require_quality_gates": False,
            "inputs": execute_inputs,
        })
        return _with_deflection_submit_diagnostics(
            result,
            byte_count=byte_count,
            max_rows=max_rows,
            loaded_row_count=loaded_row_count,
            source_row_count=source_row_count,
            submitted_row_count=len(submitted_rows),
            truncated_row_count=truncated_row_count,
            language_filtered_row_count=language_filtered_row_count,
            csv_load_warnings=csv_load_warnings,
            package=package.as_dict(),
            support_platform=data["support_platform"],
            byte_count_key=byte_count_key,
            importer_mode=_clean(data.get("importer_mode")) or "csv",
        )

    return router


async def _load_deflection_submit_rows_from_request(
    request: Any,
    *,
    max_bytes: int,
) -> tuple[
    dict[str, Any],
    list[Any],
    int,
    str,
    tuple[dict[str, Any], ...],
    int | None,
]:
    if _is_deflection_submit_http_request(request):
        content_type = _request_content_type(request)
        if "multipart/form-data" in content_type:
            _reject_oversize_deflection_submit_multipart(request, max_bytes=max_bytes)
            try:
                form = await request.form(max_part_size=max_bytes)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Multipart deflection submit body could not be parsed.",
                ) from exc
            data = _deflection_submit_form_to_mapping(form)
            if data.get("importer_mode") == "full_thread":
                csv_file = form.get("csv_file") if hasattr(form, "get") else None
                if csv_file is not None:
                    raise HTTPException(
                        status_code=422,
                        detail="csv_file is not accepted with importer_mode=full_thread",
                    )
                json_file = form.get("json_file") if hasattr(form, "get") else None
                if json_file is None:
                    raise HTTPException(status_code=422, detail="json_file is required")
                rows, byte_count, load_warnings = await _load_deflection_submit_json_upload_rows(
                    json_file,
                    max_bytes=max_bytes,
                )
                return data, rows, byte_count, "uploaded_bytes", load_warnings, None
            json_file = form.get("json_file") if hasattr(form, "get") else None
            if json_file is not None:
                raise HTTPException(
                    status_code=422,
                    detail="json_file requires importer_mode=full_thread",
                )
            csv_file = form.get("csv_file") if hasattr(form, "get") else None
            if csv_file is None:
                raise HTTPException(status_code=422, detail="csv_file is required")
            loaded = await _load_deflection_submit_upload_rows(
                csv_file,
                max_bytes=max_bytes,
                max_rows=_deflection_submit_parse_max_rows(data.get("limit")),
            )
            return (
                data,
                loaded.rows,
                loaded.byte_count,
                "uploaded_bytes",
                loaded.warnings,
                loaded.source_row_count,
            )

        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="JSON deflection submit body could not be parsed.",
            ) from exc
        data = _deflection_submit_payload_to_mapping(payload)
        loaded = await _load_deflection_submit_blob_rows(
            data["blob_url"],
            max_bytes=max_bytes,
            importer_mode=data.get("importer_mode") or "csv",
            max_rows=_deflection_submit_parse_max_rows(data.get("limit")),
        )
        return (
            data,
            loaded.rows,
            loaded.byte_count,
            "blob_bytes",
            loaded.warnings,
            loaded.source_row_count,
        )

    data = _deflection_submit_payload_to_mapping(request)
    loaded = await _load_deflection_submit_blob_rows(
        data["blob_url"],
        max_bytes=max_bytes,
        importer_mode=data.get("importer_mode") or "csv",
        max_rows=_deflection_submit_parse_max_rows(data.get("limit")),
    )
    return (
        data,
        loaded.rows,
        loaded.byte_count,
        "blob_bytes",
        loaded.warnings,
        loaded.source_row_count,
    )


def _is_deflection_submit_http_request(value: Any) -> bool:
    return (
        hasattr(value, "headers")
        and (hasattr(value, "form") or hasattr(value, "json"))
    )


def _request_content_type(request: Any) -> str:
    headers = getattr(request, "headers", {}) or {}
    value = ""
    if hasattr(headers, "get"):
        value = headers.get("content-type") or headers.get("Content-Type") or ""
    return str(value).lower()


def _reject_oversize_deflection_submit_multipart(
    request: Any,
    *,
    max_bytes: int,
) -> None:
    headers = getattr(request, "headers", {}) or {}
    value = ""
    if hasattr(headers, "get"):
        value = headers.get("content-length") or headers.get("Content-Length") or ""
    if not _clean(value):
        return
    try:
        content_length = int(str(value))
    except ValueError:
        return
    if content_length > max_bytes + _MAX_DEFLECTION_SUBMIT_MULTIPART_OVERHEAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "reason": "deflection_submit_csv_too_large",
                "max_file_bytes": max_bytes,
            },
        )


def _deflection_submit_form_to_mapping(form: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "support_platform": form.get("support_platform") if hasattr(form, "get") else None,
        "company_name": form.get("company_name") if hasattr(form, "get") else None,
        "contact_email": form.get("contact_email") if hasattr(form, "get") else None,
    }
    limit = form.get("limit") if hasattr(form, "get") else None
    if _clean(limit):
        data["limit"] = limit
    importer_mode = form.get("importer_mode") if hasattr(form, "get") else None
    if _clean(importer_mode):
        data["importer_mode"] = importer_mode
    return _deflection_submit_fields_to_mapping(data)


def _deflection_submit_max_rows(limit: Any, raw_row_count: int) -> int:
    if raw_row_count <= 0:
        return 0
    if limit is None:
        return raw_row_count
    return min(int(limit), raw_row_count)


def _deflection_submit_truncated_row_count(
    *,
    source_row_count: int,
    loaded_included_row_count: int,
    eligible_row_count: int,
    submitted_row_count: int,
    parser_source_row_count: int | None,
) -> int:
    parser_truncated = (
        max(0, source_row_count - loaded_included_row_count)
        if parser_source_row_count is not None
        else 0
    )
    post_filter_truncated = max(0, eligible_row_count - submitted_row_count)
    return parser_truncated + post_filter_truncated


def _deflection_submit_parse_max_rows(limit: Any) -> int | None:
    if limit is None:
        return None
    return int(limit)


async def _inspect_uploaded_ingestion_file(
    file: UploadFile,
    *,
    source_rows: bool,
    source: str | None,
    target_mode: str | None,
    file_format: str,
    max_source_text_chars: int,
    sample_limit: int,
    default_fields: str | None,
):
    upload_bytes = await _read_bounded_upload(file)
    if not upload_bytes:
        raise HTTPException(status_code=400, detail="Uploaded ingestion file is empty.")
    resolved_format = _validate_upload_file_format(file_format)
    parsed_default_fields = _parse_default_fields_form(default_fields)
    suffix = _upload_suffix(file, resolved_format)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="content-ops-ingestion-",
            suffix=suffix,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(upload_bytes)
        try:
            report = inspect_ingestion_file(
                temp_path,
                source_rows=bool(source_rows),
                file_format=resolved_format,
                source_format=resolved_format,
                target_mode=_clean(target_mode) or "vendor_retention",
                max_source_text_chars=int(max_source_text_chars),
                sample_limit=int(sample_limit),
                default_fields=parsed_default_fields,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        row_count = len(report.opportunities)
        if row_count > _MAX_FILE_INGESTION_ROWS:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Uploaded ingestion file is too large after normalization; "
                    f"max {_MAX_FILE_INGESTION_ROWS} rows, got {row_count}."
                ),
            )
        return report
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Content Ops temporary ingestion upload cleanup failed",
                    extra={"source": _clean(source) or _clean(getattr(file, "filename", None))},
                )


async def _load_deflection_submit_blob_rows(
    blob_url: str,
    *,
    max_bytes: int,
    importer_mode: str = "csv",
    max_rows: int | None = None,
) -> _DeflectionSubmitRowsLoad:
    return await asyncio.to_thread(
        _load_deflection_submit_blob_rows_sync,
        blob_url,
        max_bytes,
        importer_mode,
        max_rows,
    )


async def _load_deflection_submit_upload_rows(
    csv_file: Any,
    *,
    max_bytes: int,
    max_rows: int | None = None,
) -> _DeflectionSubmitRowsLoad:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="content-ops-deflection-submit-",
            suffix=".csv",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            byte_count = await _copy_deflection_submit_upload_to_tempfile(
                csv_file,
                handle,
                max_bytes=max_bytes,
            )
        if byte_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty.",
            )
        return _parse_deflection_submit_csv_file(
            temp_path,
            byte_count=byte_count,
            parse_error_detail="Uploaded CSV could not be parsed.",
            max_rows=max_rows,
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Content Ops deflection submit temp cleanup failed")


async def _copy_deflection_submit_upload_to_tempfile(
    upload_file: Any,
    handle: Any,
    *,
    max_bytes: int,
    too_large_reason: str = "deflection_submit_csv_too_large",
    read_error_detail: str = "Uploaded CSV could not be read.",
    stage_error_detail: str = "Uploaded CSV could not be staged.",
) -> int:
    byte_count = 0
    while True:
        try:
            chunk = await upload_file.read(_DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES)
        except OSError as exc:
            raise HTTPException(
                status_code=400,
                detail=read_error_detail,
            ) from exc
        if not chunk:
            return byte_count
        next_byte_count = byte_count + len(chunk)
        if next_byte_count > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "reason": too_large_reason,
                    "max_file_bytes": max_bytes,
                },
            )
        try:
            handle.write(chunk)
        except OSError as exc:
            raise HTTPException(
                status_code=400,
                detail=stage_error_detail,
            ) from exc
        byte_count = next_byte_count


async def _load_deflection_submit_json_upload_rows(
    json_file: Any,
    *,
    max_bytes: int,
) -> tuple[list[Any], int, tuple[dict[str, Any], ...]]:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="content-ops-deflection-submit-",
            suffix=".json",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            byte_count = await _copy_deflection_submit_upload_to_tempfile(
                json_file,
                handle,
                max_bytes=max_bytes,
                too_large_reason="deflection_submit_full_thread_too_large",
                read_error_detail=(
                    "Uploaded Zendesk full-thread JSON could not be read."
                ),
                stage_error_detail=(
                    "Uploaded Zendesk full-thread JSON could not be staged."
                ),
            )
        if byte_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded Zendesk full-thread JSON is empty.",
            )
        return _parse_deflection_submit_zendesk_thread_file(
            temp_path,
            byte_count=byte_count,
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Content Ops deflection submit temp cleanup failed")


def _load_deflection_submit_blob_rows_sync(
    blob_url: str,
    max_bytes: int,
    importer_mode: str = "csv",
    max_rows: int | None = None,
) -> _DeflectionSubmitRowsLoad:
    if importer_mode == "full_thread":
        rows, byte_count, warnings = _load_deflection_submit_json_blob_rows_sync(
            blob_url,
            max_bytes=max_bytes,
        )
        return _DeflectionSubmitRowsLoad(rows, byte_count, warnings)
    return _load_deflection_submit_csv_blob_rows_sync(
        blob_url,
        max_bytes=max_bytes,
        max_rows=max_rows,
    )


def _load_deflection_submit_json_blob_rows_sync(
    blob_url: str,
    *,
    max_bytes: int,
) -> tuple[list[Any], int, tuple[dict[str, Any], ...]]:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="content-ops-deflection-submit-",
            suffix=".json",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            byte_count = _copy_bounded_https_blob_to_tempfile(
                blob_url,
                handle,
                max_bytes=max_bytes,
                stage_error_detail="Zendesk full-thread JSON could not be staged.",
                stage_log_message=(
                    "Content Ops deflection blob full-thread JSON staging failed"
                ),
            )
        if byte_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Zendesk full-thread JSON is empty.",
            )
        return _parse_deflection_submit_zendesk_thread_file(
            temp_path,
            byte_count=byte_count,
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Content Ops deflection submit temp cleanup failed")


def _load_deflection_submit_csv_blob_rows_sync(
    blob_url: str,
    *,
    max_bytes: int,
    max_rows: int | None = None,
) -> _DeflectionSubmitRowsLoad:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="content-ops-deflection-submit-",
            suffix=".csv",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            byte_count = _copy_bounded_https_blob_to_tempfile(
                blob_url,
                handle,
                max_bytes=max_bytes,
            )
        if byte_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Blob CSV is empty.",
            )
        return _parse_deflection_submit_csv_file(
            temp_path,
            byte_count=byte_count,
            parse_error_detail="Blob CSV could not be parsed.",
            max_rows=max_rows,
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Content Ops deflection submit temp cleanup failed")


def _copy_bounded_https_blob_to_tempfile(
    blob_url: str,
    handle: Any,
    *,
    max_bytes: int,
    too_large_reason: str = "deflection_submit_blob_too_large",
    stage_error_detail: str = "Blob CSV could not be staged.",
    stage_log_message: str = "Content Ops deflection blob CSV staging failed",
    fetch_log_message: str = "Content Ops deflection blob fetch failed",
) -> int:
    response: Any | None = None
    byte_count = 0
    try:
        response = _open_validated_https_blob_response(blob_url)
        while True:
            chunk = response.read(_DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES)
            if not chunk:
                return byte_count
            next_byte_count = byte_count + len(chunk)
            if next_byte_count > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "reason": too_large_reason,
                        "max_file_bytes": max_bytes,
                    },
                )
            try:
                handle.write(chunk)
            except OSError as exc:
                _log_deflection_blob_fetch_failure(
                    blob_url,
                    stage_log_message,
                )
                raise HTTPException(
                    status_code=400,
                    detail=stage_error_detail,
                ) from exc
            byte_count = next_byte_count
    except HTTPException:
        raise
    except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
        _log_deflection_blob_fetch_failure(
            blob_url,
            fetch_log_message,
        )
        raise HTTPException(
            status_code=400,
            detail="Blob URL could not be fetched.",
        ) from exc
    finally:
        if response is not None:
            _close_https_blob_response(response)


def _parse_deflection_submit_zendesk_thread_bytes(
    data: bytes,
) -> tuple[list[Any], int, tuple[dict[str, Any], ...]]:
    try:
        result = load_zendesk_full_thread_rows_from_json_bytes(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Zendesk full-thread JSON could not be parsed.",
        ) from exc
    return result.rows, len(data), result.warnings


def _parse_deflection_submit_zendesk_thread_file(
    temp_path: Path,
    *,
    byte_count: int,
) -> tuple[list[Any], int, tuple[dict[str, Any], ...]]:
    try:
        result = load_zendesk_full_thread_rows_from_json_file(temp_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Zendesk full-thread JSON could not be parsed.",
        ) from exc
    return result.rows, byte_count, result.warnings


def _parse_deflection_submit_csv_file(
    temp_path: Path,
    *,
    byte_count: int,
    parse_error_detail: str,
    max_rows: int | None = None,
) -> _DeflectionSubmitRowsLoad:
    try:
        result = load_csv_source_rows_result_from_file(
            temp_path,
            max_rows=max_rows,
        )
    except CsvCustomerDataParseError as exc:
        raise HTTPException(
            status_code=400,
            detail=_deflection_submit_csv_parse_error_detail(exc),
        ) from exc
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=parse_error_detail,
        ) from exc
    return _DeflectionSubmitRowsLoad(
        rows=result.rows,
        byte_count=byte_count,
        warnings=tuple(warning.as_dict() for warning in result.warnings),
        source_row_count=result.source_row_count,
    )


def _deflection_submit_csv_parse_error_detail(
    error: CsvCustomerDataParseError,
) -> dict[str, Any]:
    detail = error.as_dict()
    detail["reason"] = "deflection_submit_csv_parse_error"
    return detail


def _read_bounded_https_blob(blob_url: str, *, max_bytes: int) -> bytes:
    response: Any | None = None
    try:
        response = _open_validated_https_blob_response(blob_url)
        data = response.read(max_bytes + 1)
    except HTTPException:
        raise
    except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
        _log_deflection_blob_fetch_failure(
            blob_url,
            "Content Ops deflection blob fetch failed",
        )
        raise HTTPException(
            status_code=400,
            detail="Blob URL could not be fetched.",
        ) from exc
    finally:
        if response is not None:
            _close_https_blob_response(response)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "reason": "deflection_submit_blob_too_large",
                "max_file_bytes": max_bytes,
            },
        )
    return data


def _open_validated_https_blob_response(blob_url: str) -> Any:
    target = _validate_https_blob_fetch_target(blob_url)
    response: Any | None = None
    try:
        response = _open_https_blob_request(
            target,
            timeout=_DEFLECTION_SUBMIT_FETCH_TIMEOUT_SECONDS,
        )
        status = int(getattr(response, "status", 0) or 0)
        if 300 <= status < 400:
            raise HTTPException(
                status_code=400,
                detail="Blob URL redirects are not allowed.",
            )
        if status < 200 or status >= 300:
            raise HTTPException(
                status_code=400,
                detail="Blob URL could not be fetched.",
            )
        return response
    except HTTPException:
        if response is not None:
            _close_https_blob_response(response)
        raise
    except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
        if response is not None:
            _close_https_blob_response(response)
        _log_deflection_blob_fetch_failure(
            blob_url,
            "Content Ops deflection blob fetch failed",
        )
        raise HTTPException(
            status_code=400,
            detail="Blob URL could not be fetched.",
        ) from exc


@dataclass(frozen=True)
class _BlobFetchTarget:
    url: str
    host: str
    port: int
    path: str
    host_header: str
    connect_host: str


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        *,
        port: int,
        connect_host: str,
        timeout: int,
    ) -> None:
        super().__init__(
            host,
            port=port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._connect_host = connect_host

    def connect(self) -> None:
        self.sock = self._create_connection(
            (self._connect_host, self.port),
            self.timeout,
            self.source_address,
        )
        if self._tunnel_host:
            self._tunnel()
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


_PINNED_HTTPS_CONNECTION_CLASS = _PinnedHTTPSConnection


class _PinnedBlobResponse:
    def __init__(
        self,
        response: http.client.HTTPResponse,
        connection: http.client.HTTPSConnection,
    ) -> None:
        self._response = response
        self._connection = connection
        self.status = response.status

    def read(self, size: int) -> bytes:
        return self._response.read(size)

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()


def _open_https_blob_request(
    target: _BlobFetchTarget,
    *,
    timeout: int,
) -> Any:
    connection = _PINNED_HTTPS_CONNECTION_CLASS(
        target.host,
        port=target.port,
        connect_host=target.connect_host,
        timeout=timeout,
    )
    try:
        connection.request(
            "GET",
            target.path,
            headers={
                "Host": target.host_header,
                "User-Agent": "Atlas-Content-Ops/1.0",
            },
        )
        response = connection.getresponse()
        return _PinnedBlobResponse(response, connection)
    except Exception:
        connection.close()
        raise


def _close_https_blob_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _log_deflection_blob_fetch_failure(blob_url: str, message: str) -> None:
    parsed = urllib.parse.urlparse(blob_url)
    logger.warning(
        message,
        extra={
            "blob_host": parsed.hostname or "",
            "blob_path": parsed.path or "/",
        },
        exc_info=True,
    )


def _validate_https_blob_url(value: Any) -> str:
    text = _clean(value) or ""
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("blob_url must be an https URL")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("blob_url must be an https URL") from exc
    host = parsed.hostname or ""
    if not host or _is_blocked_blob_host(host):
        raise ValueError("blob_url host is not allowed")
    if parsed.username or parsed.password:
        raise ValueError("blob_url must not include credentials")
    return text


def _validate_https_blob_fetch_target(value: Any) -> _BlobFetchTarget:
    url = _validate_https_blob_url(value)
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or 443
    try:
        connect_host = _validate_blob_host_resolution(host, port)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = urllib.parse.urlunparse((
        "",
        "",
        parsed.path or "/",
        parsed.params,
        parsed.query,
        "",
    ))
    return _BlobFetchTarget(
        url=url,
        host=host,
        port=port,
        path=path,
        host_header=parsed.netloc,
        connect_host=connect_host,
    )


def _validate_blob_host_resolution(host: str, port: int) -> str:
    try:
        resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("blob_url host could not be resolved") from exc
    if not resolved:
        raise ValueError("blob_url host could not be resolved")
    connect_host: str | None = None
    for item in resolved:
        sockaddr = item[4]
        if not sockaddr:
            raise ValueError("blob_url host could not be resolved")
        address = str(sockaddr[0])
        if _is_blocked_blob_host(address):
            raise ValueError("blob_url host is not allowed")
        if connect_host is None:
            connect_host = address
    if connect_host is None:
        raise ValueError("blob_url host could not be resolved")
    return connect_host


def _is_blocked_blob_host(host: str) -> bool:
    lowered = host.strip().lower().rstrip(".")
    if lowered in {"localhost", "0.0.0.0"} or lowered.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        return False
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _deflection_submit_rows_with_defaults(
    data: Mapping[str, Any],
    rows: Sequence[Any],
) -> list[Any]:
    defaults = {
        "source_type": "support_ticket",
        "company_name": data["company_name"],
        "contact_email": data["contact_email"],
        "support_platform": data["support_platform"],
    }
    return [
        {**defaults, **dict(row)}
        if isinstance(row, Mapping)
        else row
        for row in rows
    ]


def _deflection_submit_english_rows(rows: Sequence[Any]) -> tuple[list[Any], int]:
    if not any(_deflection_submit_language(row) for row in rows):
        return list(rows), 0
    out: list[Any] = []
    filtered = 0
    for row in rows:
        language = _deflection_submit_language(row)
        if language and not _is_english_language(language):
            filtered += 1
            continue
        out.append(row)
    return out, filtered


def _deflection_submit_language(row: Any) -> str:
    if not isinstance(row, Mapping):
        return ""
    for key in ("language", "lang", "locale"):
        value = _clean(row.get(key))
        if value:
            return value
    return ""


def _is_english_language(value: str) -> bool:
    normalized = value.strip().lower().replace("_", "-")
    # Provider exports may use display forms such as "English (US)" or
    # "English (United Kingdom)"; the parenthetical region is not a
    # language signal, so strip it before matching.
    normalized = normalized.split("(", 1)[0].strip()
    return (
        normalized in {"en", "eng", "english"}
        or normalized.startswith("en-")
        or normalized.startswith("english")
    )


def _deflection_submit_title(data: Mapping[str, Any]) -> str:
    return f"{data['company_name']} Support Deflection Report"


def _deflection_checkout_terms(
    config: ContentOpsControlSurfaceApiConfig,
) -> dict[str, Any]:
    try:
        amount_cents = int(config.deflection_checkout_amount_cents)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Deflection checkout amount is not configured.",
        ) from exc
    currency = (_clean(config.deflection_checkout_currency) or "").lower()
    price_id = _clean(config.deflection_checkout_price_id)
    if amount_cents <= 0:
        raise HTTPException(
            status_code=503,
            detail="Deflection checkout amount is not configured.",
        )
    allowed_amounts = _deflection_checkout_allowed_amounts(config, amount_cents)
    if amount_cents not in allowed_amounts:
        raise HTTPException(
            status_code=503,
            detail="Deflection checkout amount is not accepted by the payment gate.",
        )
    if len(currency) != 3 or not currency.isalpha():
        raise HTTPException(
            status_code=503,
            detail="Deflection checkout currency is not configured.",
        )
    if not price_id:
        raise HTTPException(
            status_code=503,
            detail="Deflection checkout price is not configured.",
        )
    return {
        "amount_cents": amount_cents,
        "currency": currency,
        "price_id": price_id,
    }


def _deflection_checkout_allowed_amounts(
    config: ContentOpsControlSurfaceApiConfig,
    default_amount_cents: int,
) -> tuple[int, ...]:
    configured = _clean(config.deflection_checkout_allowed_amount_cents)
    if not configured:
        return (default_amount_cents,)
    amounts: list[int] = []
    for raw_part in configured.split(","):
        part = raw_part.strip()
        if not part:
            raise HTTPException(
                status_code=503,
                detail="Deflection checkout allowed amounts are not configured.",
            )
        try:
            amount_cents = int(part)
        except ValueError as exc:
            raise HTTPException(
                status_code=503,
                detail="Deflection checkout allowed amounts are not configured.",
            ) from exc
        if amount_cents <= 0:
            raise HTTPException(
                status_code=503,
                detail="Deflection checkout allowed amounts are not configured.",
            )
        amounts.append(amount_cents)
    return tuple(dict.fromkeys(amounts))


def _deflection_report_checkout_inside_retention_window(
    created_at: Any,
    *,
    now: datetime | None = None,
) -> bool:
    resolved_created_at = _coerce_aware_datetime(created_at)
    if resolved_created_at is None:
        return False
    checkout_deadline = (
        resolved_created_at
        + timedelta(days=_DEFLECTION_REPORT_RETENTION_DAYS)
        - _DEFLECTION_CHECKOUT_OPEN_SESSION_GRACE
    )
    resolved_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return resolved_now < checkout_deadline


def _coerce_aware_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        resolved = value
    elif isinstance(value, str) and value.strip():
        try:
            resolved = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        return None
    return resolved.astimezone(timezone.utc)


def _with_deflection_submit_diagnostics(
    response: Mapping[str, Any],
    *,
    byte_count: int,
    max_rows: int,
    loaded_row_count: int,
    source_row_count: int,
    submitted_row_count: int,
    truncated_row_count: int,
    language_filtered_row_count: int,
    package: Mapping[str, Any],
    support_platform: str,
    byte_count_key: str = "blob_bytes",
    csv_load_warnings: Sequence[Mapping[str, Any]] = (),
    importer_mode: str = "csv",
) -> dict[str, Any]:
    out = dict(response)
    existing = out.get("input_provider")
    diagnostics = dict(existing) if isinstance(existing, Mapping) else {}
    metadata = dict(diagnostics.get("metadata") or {})
    package_metadata = package.get("metadata") if isinstance(package, Mapping) else {}
    if isinstance(package_metadata, Mapping):
        metadata.update({
            key: package_metadata[key]
            for key in (
                "included_row_count",
                "skipped_row_count",
                "source_period",
                "top_ticket_clusters",
                "cluster_quality",
                "cluster_preview_skipped",
                "cluster_preview_token_set_row_count",
                "support_ticket_resolution_evidence_present",
                "support_ticket_resolution_evidence_count",
            )
            if key in package_metadata
        })
        if package_metadata.get("ticket_status_present"):
            metadata.update({
                key: package_metadata[key]
                for key in (
                    "ticket_status_present",
                    "ticket_status_present_count",
                    "ticket_status_summary",
                )
                if key in package_metadata
            })
        if package_metadata.get("csat_present"):
            metadata.update({
                key: package_metadata[key]
                for key in (
                    "csat_present",
                    "csat_present_count",
                    "csat_score_count",
                    "csat_score_average",
                )
                if key in package_metadata
            })
    metadata.update({
        "source": "portfolio_deflection_submit",
        "source_row_count": source_row_count,
        "submitted_row_count": submitted_row_count,
        "truncated_row_count": truncated_row_count,
        "max_source_material_rows": max_rows,
        byte_count_key: byte_count,
        "support_platform": support_platform,
    })
    if importer_mode != "csv":
        metadata["importer_mode"] = importer_mode
    if language_filtered_row_count:
        metadata["loaded_source_row_count"] = loaded_row_count
        metadata["language_filtered_row_count"] = language_filtered_row_count
    warnings = [
        dict(warning)
        for warning in diagnostics.get("warnings") or ()
        if isinstance(warning, Mapping)
    ]
    warnings.extend(
        dict(warning)
        for warning in csv_load_warnings
        if isinstance(warning, Mapping)
    )
    package_warnings = package.get("warnings") if isinstance(package, Mapping) else ()
    warnings.extend(
        dict(warning)
        for warning in package_warnings or ()
        if isinstance(warning, Mapping)
        and warning.get("code") == "cluster_preview_skipped_large_upload"
    )
    if language_filtered_row_count:
        warnings.append({
            "code": "deflection_submit_non_english_rows_filtered",
            "message": (
                f"Skipped {language_filtered_row_count} non-English support-ticket rows."
            ),
            "row_count": loaded_row_count,
            "filtered_row_count": language_filtered_row_count,
        })
    if truncated_row_count:
        warnings.append({
            "code": "deflection_submit_rows_truncated",
            "message": (
                f"Used first {submitted_row_count} support-ticket rows "
                f"out of {source_row_count}."
            ),
            "row_count": source_row_count,
            "max_rows": max_rows,
            "truncated_row_count": truncated_row_count,
        })
    out["input_provider"] = {
        "provider": "portfolio_deflection_submit",
        "metadata": metadata,
        "warnings": warnings,
    }
    return out


async def _read_bounded_upload(file: UploadFile) -> bytes:
    try:
        data = await file.read(_MAX_INGESTION_FILE_BYTES + 1)
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded ingestion file could not be read.",
        ) from exc
    if len(data) > _MAX_INGESTION_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "Uploaded ingestion file is too large; "
                f"max {_MAX_INGESTION_FILE_BYTES} bytes."
            ),
        )
    return data


def _validate_upload_file_format(value: str) -> str:
    text = _clean(value) or "auto"
    if text not in _UPLOAD_FILE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="file_format must be one of: auto, json, jsonl, csv.",
        )
    return text


def _parse_default_fields_form(value: str | None) -> dict[str, Any]:
    text = _clean(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="default_fields must be a JSON object.",
        ) from exc
    if not isinstance(parsed, Mapping):
        raise HTTPException(
            status_code=400,
            detail="default_fields must be a JSON object.",
        )
    parsed_dict = dict(parsed)
    if len(parsed_dict) > 50:
        raise HTTPException(
            status_code=400,
            detail="default_fields cannot contain more than 50 entries.",
        )
    try:
        _validate_input_shape(parsed_dict, depth=0)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return parsed_dict


def _upload_suffix(file: UploadFile, file_format: str) -> str:
    filename = _clean(getattr(file, "filename", None))
    suffix = Path(filename).suffix.lower() if filename else ""
    if suffix in {".json", ".jsonl", ".ndjson", ".csv"}:
        return ".jsonl" if suffix == ".ndjson" else suffix
    if file_format in {"json", "jsonl", "csv"}:
        return f".{file_format}"
    return ".json"


def _ingestion_diagnostics_response(
    report: Any,
    *,
    source: str | None = None,
    include_source_material: bool = False,
) -> dict[str, Any]:
    payload = report.as_dict()
    if source is not None:
        payload["source"] = _clean(source) or payload.get("source") or ""
    if include_source_material:
        payload["source_material"] = [
            dict(row)
            for row in getattr(report, "opportunities", ())
            if isinstance(row, Mapping)
        ]
    return payload


def _flag_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _file_ingestion_response(
    report: Any,
    *,
    source: str | None = None,
    include_source_material: bool = False,
) -> dict[str, Any]:
    payload = _ingestion_diagnostics_response(
        report,
        source=source,
        include_source_material=include_source_material,
    )
    payload["ingestion_path"] = "file_upload"
    payload["limits"] = {
        "max_file_bytes": _MAX_INGESTION_FILE_BYTES,
        "max_rows": _MAX_FILE_INGESTION_ROWS,
        "inline_rows_deprecated": True,
    }
    return payload


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


async def _payload_with_input_provider(
    payload: Mapping[str, Any],
    *,
    input_provider: InputProvider | None,
    scope_provider: ScopeProvider | None = None,
    scope: TenantScope | None = None,
) -> dict[str, Any]:
    if input_provider is None:
        return dict(payload)
    resolved_scope = scope
    if resolved_scope is None:
        resolved_scope = await _resolve_scope(scope_provider)
    try:
        package = input_provider.build_content_ops_input_package(
            scope=resolved_scope or TenantScope(),
            request=payload,
        )
        if hasattr(package, "__await__"):
            package = await package
        return merge_content_ops_input_package(payload, package)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            "Content Ops input provider failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Content Ops input provider is unavailable.",
        ) from exc


async def _payload_with_cache_policy_default(
    payload: Mapping[str, Any],
    *,
    cache_policy_default_provider: CachePolicyDefaultProvider | None,
    scope_provider: ScopeProvider | None = None,
    scope: TenantScope | None = None,
) -> dict[str, Any]:
    out = dict(payload)
    if cache_policy_default_provider is None:
        return out
    if _clean(out.get("content_ops_cache_policy")):
        return out
    resolved_scope = scope
    if resolved_scope is None:
        resolved_scope = await _resolve_scope(scope_provider)
    try:
        default_value = cache_policy_default_provider(resolved_scope or TenantScope())
        if hasattr(default_value, "__await__"):
            default_value = await default_value
        normalized = normalize_content_ops_cache_policy(default_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Content Ops cache policy default is invalid.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            "Content Ops cache policy default provider failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Content Ops cache policy default is unavailable.",
        ) from exc
    if normalized:
        out["content_ops_cache_policy"] = normalized
    return out


async def _payload_with_brand_voice_profile(
    payload: Mapping[str, Any],
    *,
    brand_voice_profile_provider: BrandVoiceProfileProvider | None,
    scope_provider: ScopeProvider | None = None,
    scope: TenantScope | None = None,
) -> dict[str, Any]:
    out = dict(payload)
    profile_id = _clean(out.get("brand_voice_profile_id"))
    if not profile_id:
        return out
    inputs = out.get("inputs")
    if isinstance(inputs, Mapping) and inputs.get("brand_voice") is not None:
        return out
    if brand_voice_profile_provider is None:
        raise HTTPException(
            status_code=503,
            detail="Content Ops brand voice profile lookup is not configured.",
        )
    resolved_scope = scope
    if resolved_scope is None:
        resolved_scope = await _resolve_scope(scope_provider)
    _required_scope_account_id(resolved_scope)
    try:
        value = brand_voice_profile_provider(resolved_scope or TenantScope(), profile_id)
        if hasattr(value, "__await__"):
            value = await value
        if value is None:
            raise HTTPException(
                status_code=404,
                detail="Brand voice profile not found.",
            )
        profile = brand_voice_profile_from_mapping(
            value,
            scope=resolved_scope,
            profile_id=profile_id,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning(
            "Content Ops brand voice profile provider failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Content Ops brand voice profile lookup is unavailable.",
        ) from exc
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail="Brand voice profile not found.",
        )
    out_inputs = dict(inputs) if isinstance(inputs, Mapping) else {}
    out_inputs["brand_voice"] = _brand_voice_profile_payload(profile)
    out["inputs"] = out_inputs
    return out


def _brand_voice_profile_payload(profile: BrandVoiceProfile) -> dict[str, Any]:
    return {
        "id": profile.id,
        "account_id": profile.account_id,
        "name": profile.name,
        "descriptors": list(profile.descriptors),
        "exemplars": list(profile.exemplars),
        "banned_terms": list(profile.banned_terms),
        "preferred_pov": profile.preferred_pov,
        "reading_level": profile.reading_level,
        "metadata": dict(profile.metadata or {}),
    }


def _with_input_provider_diagnostics(
    response: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostics = payload.get("input_provider")
    if not isinstance(diagnostics, Mapping):
        return dict(response)
    warnings = [
        dict(warning)
        for warning in diagnostics.get("warnings") or ()
        if isinstance(warning, Mapping)
    ]
    if _is_noop_input_provider_diagnostics(diagnostics, warnings):
        return dict(response)
    out = dict(response)
    out["input_provider"] = {
        "provider": _clean(diagnostics.get("provider")),
        "metadata": _input_provider_response_metadata(diagnostics.get("metadata")),
        "warnings": warnings,
    }
    return out


def _is_noop_input_provider_diagnostics(
    diagnostics: Mapping[str, Any],
    warnings: Sequence[Mapping[str, Any]],
) -> bool:
    metadata = diagnostics.get("metadata")
    return (
        not warnings
        and isinstance(metadata, Mapping)
        and _clean(metadata.get("mode")) == "noop"
    )


def _input_provider_response_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: value[key]
        for key in sorted(_INPUT_PROVIDER_RESPONSE_METADATA_KEYS)
        if key in value
    }


def _deflection_report_process_contract_payload(prefix: str) -> dict[str, Any]:
    base = _clean(prefix).rstrip("/")
    route_base = f"{base}/deflection-reports" if base else "/deflection-reports"
    return {
        "schema_version": DEFLECTION_PROCESS_CONTRACT_SCHEMA_VERSION,
        "service": "content_ops_deflection_reports",
        "contract": {
            "report_model_schema_version": DEFLECTION_REPORT_SCHEMA_VERSION,
            "report_model_contract": deflection_report_model_contract_shape(),
            "evidence_export_schema_version": DEFLECTION_EVIDENCE_EXPORT_SCHEMA_VERSION,
            "paid_artifact_requires": {
                "report_model": "object",
                "evidence_export": "object",
            },
        },
        "routes": {
            "process_contract": f"{route_base}/process-contract",
            "snapshot": f"{route_base}/{{request_id}}/snapshot",
            "artifact": f"{route_base}/{{request_id}}/artifact",
            "report_model": f"{route_base}/{{request_id}}/report-model",
            "delete": f"{route_base}/{{request_id}}",
        },
    }


def _enforce_faq_execute_source_material_limit(
    payload: Mapping[str, Any],
    *,
    max_rows: int,
) -> None:
    try:
        request = request_from_mapping(payload)
        outputs = resolve_outputs(request)
    except ValueError:
        return
    if not _FAQ_SOURCE_MATERIAL_LIMITED_OUTPUTS.intersection(outputs):
        return
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        return
    row_count = _source_material_row_count(inputs.get("source_material"))
    if row_count <= max_rows:
        return
    raise HTTPException(
        status_code=413,
        detail={
            "reason": "faq_source_material_too_large_for_sync_execute",
            "max_source_material_rows": max_rows,
            "source_material_rows": row_count,
            "large_upload_strategy": "background_or_offline",
        },
    )


def _source_material_row_count(source_material: Any) -> int:
    return len(source_material_to_source_rows(source_material))


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


def _required_scope_account_id(scope: TenantScope | None) -> str:
    account_id = _clean(getattr(scope, "account_id", None))
    if not account_id:
        raise HTTPException(status_code=400, detail="account_id is required")
    return account_id


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


async def _resolve_usage_pool(pool_provider: PoolProvider | None) -> Any:
    if pool_provider is None:
        raise HTTPException(
            status_code=503,
            detail="Content Ops usage database is unavailable.",
        )
    try:
        pool = await _resolve_provider(pool_provider)
    except Exception as exc:
        logger.warning(
            "Content Ops usage pool provider failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Content Ops usage database is unavailable.",
        ) from exc
    if pool is None or getattr(pool, "is_initialized", True) is False:
        raise HTTPException(
            status_code=503,
            detail="Content Ops usage database is unavailable.",
        )
    return pool


async def _resolve_deflection_report_store(
    provider: DeflectionReportStoreProvider | None,
) -> DeflectionReportArtifactStore:
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail="Content Ops deflection report store is not configured.",
        )
    try:
        store = await _resolve_provider(provider)
    except Exception as exc:
        logger.warning(
            "Content Ops deflection report store provider failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Content Ops deflection report store is unavailable.",
        ) from exc
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Content Ops deflection report store is unavailable.",
        )
    return store


async def _execute_usage_summary(
    *,
    usage_pool_provider: PoolProvider | None,
    scope: TenantScope,
    request_id: str,
) -> dict[str, Any] | None:
    if usage_pool_provider is None:
        return None
    account_id = _clean(getattr(scope, "account_id", None))
    if not account_id:
        return None
    try:
        pool = await _resolve_usage_pool(usage_pool_provider)
        return await summarize_content_ops_llm_usage(
            pool,
            days=1,
            account_id=account_id,
            request_id=request_id,
        )
    except Exception:
        logger.warning(
            "Content Ops execute usage summary unavailable",
            exc_info=True,
            extra={"request_id": request_id},
        )
        return None


async def _gate_deflection_report_artifacts(
    result: Mapping[str, Any],
    *,
    store_provider: DeflectionReportStoreProvider | None,
    scope: TenantScope | None,
    request_id: str,
    top_n: int,
    teaser_preview_count: int,
    preview_summary_metadata: Mapping[str, Any] | None = None,
    delivery_email: str | None = None,
) -> dict[str, Any]:
    gated = dict(result)
    steps = list(gated.get("steps", ()) or ())
    if not any(_is_completed_deflection_report_step(step) for step in steps):
        return gated
    store = await _resolve_deflection_report_store(store_provider)
    account_id = _required_scope_account_id(scope)
    gated_steps: list[dict[str, Any]] = []
    for step in steps:
        step_dict = dict(step)
        if not _is_completed_deflection_report_step(step_dict):
            gated_steps.append(step_dict)
            continue
        artifact = step_dict.get("result")
        if not isinstance(artifact, Mapping):
            raise HTTPException(
                status_code=502,
                detail="Deflection report artifact is malformed.",
            )
        try:
            scrubbed_artifact = scrub_deflection_report_payload(artifact)
            snapshot = build_deflection_snapshot(
                scrubbed_artifact,
                top_n=top_n,
                teaser_preview_count=teaser_preview_count,
            ).as_dict()
            if preview_summary_metadata:
                snapshot_summary = dict(snapshot.get("summary") or {})
                snapshot_summary.update(dict(preview_summary_metadata))
                snapshot["summary"] = snapshot_summary
            snapshot = scrub_deflection_report_payload(snapshot)
            await store.save_report(
                account_id=account_id,
                request_id=request_id,
                snapshot=snapshot,
                artifact=scrubbed_artifact,
                delivery_email=delivery_email,
            )
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            logger.warning(
                "Content Ops deflection report artifact storage failed",
                exc_info=True,
                extra={"request_id": request_id},
            )
            raise HTTPException(
                status_code=503,
                detail="Content Ops deflection report store is unavailable.",
            ) from exc
        step_dict["result"] = {
            "request_id": request_id,
            "snapshot": snapshot,
            "full_report": {
                "status": "locked",
                "reason": "payment_required",
            },
        }
        gated_steps.append(step_dict)
    gated["steps"] = gated_steps
    return gated


def _delivery_email_from_payload(payload: Mapping[str, Any]) -> str | None:
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    return _clean(inputs.get("contact_email")) or None


def _deflection_resolution_preview_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        return {}
    if (
        "support_ticket_resolution_evidence_present" not in inputs
        and "support_ticket_resolution_evidence_count" not in inputs
    ):
        return {}
    count = _nonnegative_int(inputs.get("support_ticket_resolution_evidence_count"))
    explicit_present = inputs.get("support_ticket_resolution_evidence_present")
    present = (
        explicit_present
        if isinstance(explicit_present, bool)
        else count > 0
    )
    return {
        "support_ticket_resolution_evidence_present": bool(present),
        "support_ticket_resolution_evidence_count": count,
    }


def _deflection_report_search_query(value: Any) -> str:
    query = _clean(value)
    if not query:
        raise HTTPException(status_code=400, detail="q is required")
    if len(query) > _DEFLECTION_REPORT_SEARCH_MAX_QUERY_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                "q must be "
                f"{_DEFLECTION_REPORT_SEARCH_MAX_QUERY_CHARS} characters or fewer"
            ),
        )
    return str(query)


def _deflection_report_search_limit(value: Any) -> int:
    if value is None:
        return _DEFLECTION_REPORT_SEARCH_DEFAULT_LIMIT
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="limit must be positive") from exc
    if parsed <= 0:
        raise HTTPException(status_code=400, detail="limit must be positive")
    return min(parsed, _DEFLECTION_REPORT_SEARCH_MAX_LIMIT)


def _search_deflection_report_artifact(
    artifact: Mapping[str, Any] | None,
    *,
    query: str,
    limit: int,
) -> dict[str, Any]:
    normalized_query = _deflection_report_search_query(query)
    query_tokens = _deflection_report_search_tokens(normalized_query)
    if not isinstance(artifact, Mapping) or not query_tokens or limit <= 0:
        return {"query": normalized_query, "count": 0, "results": []}

    matches: list[dict[str, Any]] = []
    for rank, item in enumerate(_deflection_report_artifact_items(artifact), start=1):
        full_item = _deflection_report_full_item(item)
        if full_item is None:
            continue
        score = _deflection_report_item_score(full_item, query_tokens)
        if score <= 0:
            continue
        matches.append({
            "item": full_item,
            "score": score,
            "rank": _nonnegative_int(full_item.get("rank")) or rank,
        })
    matches.sort(
        key=lambda result: (
            -int(result["score"]),
            int(result["rank"]),
            str(result["item"].get("question") or ""),
        )
    )
    selected = matches[:limit]
    return {
        "query": normalized_query,
        "count": len(selected),
        "results": selected,
    }


def _deflection_report_artifact_items(
    artifact: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    faq_result = artifact.get("faq_result")
    if not isinstance(faq_result, Mapping):
        return ()
    items = faq_result.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in items if isinstance(item, Mapping))


def _deflection_report_full_item(item: Mapping[str, Any]) -> dict[str, Any] | None:
    # Keep this renderable-item shape aligned with atlas-portfolio
    # `isRenderableItem` and `summarizeRenderableItem`.
    for field in _DEFLECTION_REPORT_SEARCH_TEXT_FIELDS:
        if not isinstance(item.get(field), str):
            return None
    for field in _DEFLECTION_REPORT_SEARCH_NUMBER_FIELDS:
        if not _deflection_report_renderable_number(item.get(field)):
            return None
    for field in _DEFLECTION_REPORT_SEARCH_STRING_ARRAY_FIELDS:
        if not _deflection_report_string_array(item.get(field)):
            return None
    if not _deflection_report_term_mappings(item.get("term_mappings")):
        return None
    return dict(item)


def _deflection_report_renderable_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _deflection_report_string_array(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _deflection_report_term_mappings(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    required_fields = ("customer_term", "documentation_term", "suggestion")
    return all(
        isinstance(mapping, Mapping)
        and all(isinstance(mapping.get(field), str) for field in required_fields)
        for mapping in value
    )


def _deflection_report_item_score(
    item: Mapping[str, Any],
    query_tokens: Sequence[str],
) -> int:
    question_tokens = set(_deflection_report_search_tokens(item.get("question")))
    topic_tokens = set(_deflection_report_search_tokens(item.get("topic")))
    text_tokens = _deflection_report_search_tokens(_deflection_report_item_search_text(item))
    score = 0
    for token in query_tokens:
        if token in question_tokens:
            score += 6
        if token in topic_tokens:
            score += 3
        score += min(text_tokens.count(token), 4)
    return score


def _deflection_report_item_search_text(item: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for field in ("topic", "question", "answer", "when_to_contact_support"):
        text = _clean(item.get(field))
        if text:
            parts.append(str(text))
    for field in ("steps", "action_items", "source_labels"):
        for value in _text_sequence(item.get(field)):
            parts.append(value)
    for mapping in item.get("term_mappings") or ():
        if not isinstance(mapping, Mapping):
            continue
        for field in ("customer_term", "documentation_term", "suggestion"):
            text = _clean(mapping.get(field))
            if text:
                parts.append(str(text))
    return " ".join(parts)


def _deflection_report_search_tokens(value: Any) -> list[str]:
    return _DEFLECTION_REPORT_SEARCH_TOKEN_RE.findall(str(value or "").lower())


def _text_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value if _clean(item))


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return 0


def _is_completed_deflection_report_step(step: Any) -> bool:
    if not isinstance(step, Mapping):
        return False
    return (
        step.get("output") == "faq_deflection_report"
        and step.get("status") == "completed"
    )


async def _evaluate_account_usage_budget(
    payload: Mapping[str, Any],
    *,
    usage_pool_provider: PoolProvider | None,
    scope_provider: ScopeProvider | None,
    scope: TenantScope | None = None,
) -> UsageBudgetEvaluation | None:
    request = request_from_mapping(payload)
    if request.account_usage_budget_usd is None:
        return None
    pool = await _resolve_usage_pool(usage_pool_provider)
    resolved_scope = scope if scope is not None else await _resolve_scope(scope_provider)
    usage = await summarize_content_ops_llm_usage(
        pool,
        days=request.account_usage_budget_days,
        account_id=_required_scope_account_id(resolved_scope),
    )
    current_cost = usage.get("summary", {}).get("total_cost_usd", 0.0)
    return evaluate_usage_budget(
        budget_usd=request.account_usage_budget_usd,
        period_days=request.account_usage_budget_days,
        current_cost_usd=float(current_cost or 0.0),
        estimated_cost_usd=preview_from_mapping(payload)["estimated_cost_usd"],
    )


def _budget_warning(evaluation: UsageBudgetEvaluation) -> str:
    return (
        "Projected account usage exceeds account_usage_budget_usd: "
        f"{evaluation.period_days}-day projected "
        f"{evaluation.projected_cost_usd:.2f} > {evaluation.budget_usd:.2f}"
    )


def _apply_usage_budget_to_preview(
    preview: dict[str, Any],
    evaluation: UsageBudgetEvaluation | None,
) -> dict[str, Any]:
    if evaluation is None:
        return preview
    preview = dict(preview)
    preview["usage_budget"] = evaluation.as_dict()
    if evaluation.exceeded:
        preview["can_run"] = False
        warnings = list(preview.get("warnings") or ())
        warnings.append(_budget_warning(evaluation))
        preview["warnings"] = warnings
    return preview


def _apply_usage_budget_to_plan(
    plan: dict[str, Any],
    evaluation: UsageBudgetEvaluation | None,
) -> dict[str, Any]:
    if evaluation is None:
        return plan
    plan = dict(plan)
    preview = _apply_usage_budget_to_preview(
        dict(plan.get("preview") or {}),
        evaluation,
    )
    plan["preview"] = preview
    if evaluation.exceeded:
        plan["can_execute"] = False
        blocked_steps: list[dict[str, Any]] = []
        for step in plan.get("steps") or ():
            next_step = dict(step)
            next_step["status"] = "blocked"
            next_step["reason"] = "account_usage_budget_exceeded"
            blocked_steps.append(next_step)
        plan["steps"] = blocked_steps
    return plan


def _budget_exceeded_error(evaluation: UsageBudgetEvaluation) -> dict[str, Any]:
    return {
        "reason": "account_usage_budget_exceeded",
        "message": _budget_warning(evaluation),
        "usage_budget": evaluation.as_dict(),
    }


async def _import_ingestion_rows_with_admission(
    rows: Sequence[Mapping[str, Any]],
    *,
    import_gate: _ExecuteConcurrencyGate,
    import_gate_provider: ImportAdmissionProvider | None,
    pool_provider: PoolProvider | None,
    scope_provider: ScopeProvider | None,
    target_mode: str,
    opportunity_table: str,
    replace_existing: bool,
    dry_run: bool,
    source: str | None,
) -> Any:
    if dry_run:
        scope = await _resolve_scope(scope_provider)
        return await _import_campaign_opportunities_for_route(
            object(),
            rows,
            scope=scope,
            target_mode=target_mode,
            opportunity_table=opportunity_table,
            replace_existing=replace_existing,
            dry_run=True,
            source=source,
        )

    active_gate = await _resolve_import_admission_gate(import_gate_provider, import_gate)
    if not await _acquire_import_admission(active_gate):
        raise HTTPException(
            status_code=429,
            detail=_import_admission_capacity_detail(active_gate),
        )
    try:
        pool = await _resolve_import_pool(pool_provider)
        scope = await _resolve_scope(scope_provider)
        return await _import_campaign_opportunities_for_route(
            pool,
            rows,
            scope=scope,
            target_mode=target_mode,
            opportunity_table=opportunity_table,
            replace_existing=replace_existing,
            dry_run=False,
            source=source,
        )
    finally:
        await _release_import_admission(active_gate)


async def _resolve_import_admission_gate(
    provider: ImportAdmissionProvider | None,
    default_gate: _ExecuteConcurrencyGate,
) -> Any:
    if provider is None:
        return default_gate
    try:
        gate = await _resolve_provider(provider)
    except Exception as exc:
        logger.warning(
            "Content Ops ingestion import admission provider failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Content Ops ingestion import admission is unavailable.",
        ) from exc
    if gate is None:
        raise HTTPException(
            status_code=503,
            detail="Content Ops ingestion import admission is unavailable.",
        )
    return gate


async def _acquire_import_admission(gate: Any) -> bool:
    acquire = getattr(gate, "acquire", None)
    if not callable(acquire):
        raise HTTPException(
            status_code=503,
            detail="Content Ops ingestion import admission is unavailable.",
        )
    try:
        result = acquire()
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        logger.warning(
            "Content Ops ingestion import admission acquire failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Content Ops ingestion import admission is unavailable.",
        ) from exc
    return bool(result)


async def _release_import_admission(gate: Any) -> None:
    release = getattr(gate, "release", None)
    if not callable(release):
        return
    try:
        result = release()
        if hasattr(result, "__await__"):
            await result
    except Exception as exc:
        logger.warning(
            "Content Ops ingestion import admission release failed",
            extra={"error_type": type(exc).__name__},
        )


def _import_admission_capacity_detail(gate: Any) -> dict[str, Any]:
    detail: dict[str, Any] = {"reason": "content_ops_ingestion_import_at_capacity"}
    max_concurrency = getattr(gate, "max_concurrency", None)
    if max_concurrency is not None:
        detail["max_concurrency"] = int(max_concurrency)
    return detail


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


def _payload_to_mapping(payload: Any, *, exclude_unset: bool = False) -> dict[str, Any]:
    if (
        isinstance(payload, Mapping)
        and payload.get(_DEFLECTION_SUBMIT_INTERNAL_TOKEN_KEY)
        is _DEFLECTION_SUBMIT_INTERNAL_TOKEN
    ):
        return _internal_deflection_submit_payload_to_mapping(payload)
    if BaseModel is not None and isinstance(payload, ContentOpsRequestModel):
        return payload.model_dump(exclude_unset=exclude_unset)
    try:
        return ContentOpsRequestModel.model_validate(payload).model_dump(
            exclude_unset=exclude_unset
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_detail(exc)) from exc


def _internal_deflection_submit_payload_to_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        request = request_from_mapping({
            "target_mode": payload.get("target_mode") or "vendor_retention",
            "outputs": payload.get("outputs") or _DEFLECTION_SUBMIT_OUTPUTS,
            "limit": payload.get("limit") or 1,
            "inputs": payload.get("inputs") if isinstance(payload.get("inputs"), Mapping) else {},
            "require_quality_gates": bool(payload.get("require_quality_gates", True)),
        })
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        _DEFLECTION_SUBMIT_INTERNAL_TOKEN_KEY: _DEFLECTION_SUBMIT_INTERNAL_TOKEN,
        "target_mode": request.target_mode,
        "outputs": list(request.outputs),
        "limit": request.limit,
        "inputs": dict(request.inputs),
        "require_quality_gates": request.require_quality_gates,
    }


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


def _deflection_submit_payload_to_mapping(payload: Any) -> dict[str, Any]:
    if BaseModel is not None and isinstance(payload, DeflectionReportSubmitModel):
        return payload.model_dump()
    try:
        return DeflectionReportSubmitModel.model_validate(payload).model_dump()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_detail(exc)) from exc


def _deflection_submit_fields_to_mapping(payload: Any) -> dict[str, Any]:
    if BaseModel is not None and isinstance(payload, DeflectionReportSubmitFieldsModel):
        return payload.model_dump()
    try:
        return DeflectionReportSubmitFieldsModel.model_validate(payload).model_dump()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_detail(exc)) from exc


def _validate_input_shape(
    value: Any,
    *,
    depth: int,
    path: tuple[str, ...] = (),
) -> None:
    if depth > _MAX_INPUT_DEPTH:
        raise ValueError("inputs are too deeply nested")
    if isinstance(value, Mapping):
        if len(value) > _MAX_INPUT_KEYS:
            raise ValueError("inputs has too many keys")
        for key, nested in value.items():
            if len(str(key)) > 100:
                raise ValueError("inputs keys must be 100 characters or fewer")
            _validate_input_shape(
                nested,
                depth=depth + 1,
                path=(*path, str(key)),
            )
    elif isinstance(value, (list, tuple)):
        max_items = _input_array_max_items(path)
        if len(value) > max_items:
            raise ValueError("inputs arrays are too large")
        for nested in value:
            _validate_input_shape(nested, depth=depth + 1, path=(*path, "[]"))
    elif isinstance(value, str) and len(value) > _MAX_INPUT_STRING_CHARS:
        raise ValueError("inputs strings are too large")


def _input_array_max_items(path: tuple[str, ...]) -> int:
    if path == ("source_material",):
        return _MAX_INGESTION_ROWS
    if (
        len(path) == 2
        and path[0] == "source_material"
        and path[1] in _SOURCE_MATERIAL_ROW_LIST_KEYS
    ):
        return _MAX_INGESTION_ROWS
    return _MAX_INPUT_KEYS


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
    "DeflectionReportSubmitFieldsModel",
    "DeflectionReportSubmitModel",
    "create_content_ops_control_surface_router",
]
