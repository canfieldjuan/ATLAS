"""Control-surface primitives for AI Content Ops generation.

This module is intentionally pure and deterministic. It does not call an LLM,
read a database, or know about HTTP. The UI/API layer should use it to turn
user-facing choices into a validated generation plan before any expensive work
runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .content_ops_cache_policy import normalize_content_ops_cache_policy
from .landing_page_repair_contract import (
    LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_DEFAULT,
    landing_page_quality_repair_attempts_from_inputs,
)
from .social_post_generation import normalize_social_post_channels

SOCIAL_POST_BRAND_VOICE_UNIT_COST_USD = 0.08
SOCIAL_POST_BRAND_VOICE_PARSE_RETRY_ATTEMPTS = 1


@dataclass(frozen=True)
class OutputDefinition:
    """Catalog entry for one generated content asset type."""

    id: str
    label: str
    description: str
    implemented: bool
    estimated_unit_cost_usd: float
    required_inputs: tuple[str, ...] = ()
    default_max_items: int = 1
    reasoning_requirement: str = "absent"
    # Must mirror each *GenerationConfig.parse_retry_attempts default used by
    # the generation services. If a service raises retries, update this field too.
    default_parse_retry_attempts: int = 1
    # Non-landing-page outputs have no quality-repair LLM loop.
    default_quality_repair_attempts: int = 0


@dataclass(frozen=True)
class ControlSurfacePreset:
    """Named output bundle for product-safe defaults."""

    id: str
    label: str
    outputs: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class ContentOpsRequest:
    """Normalized request shape produced by UI/API control surfaces."""

    target_mode: str = "vendor_retention"
    preset: str | None = None
    reasoning_preset: str | None = None
    outputs: tuple[str, ...] = ()
    limit: int = 1
    max_cost_usd: float | None = None
    account_usage_budget_usd: float | None = None
    account_usage_budget_days: int = 7
    content_ops_cache_policy: str | None = None
    brand_voice_profile_id: str | None = None
    inputs: Mapping[str, Any] = field(default_factory=dict)
    ingestion_profile: str = "domain_specific"
    require_quality_gates: bool = True
    allow_unimplemented_outputs: bool = False


@dataclass(frozen=True)
class ControlSurfacePreview:
    """Plan preview returned before generation starts."""

    can_run: bool
    outputs: tuple[str, ...]
    estimated_cost_usd: float
    missing_inputs: tuple[str, ...] = ()
    blocked_outputs: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    normalized_request: ContentOpsRequest | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "can_run": self.can_run,
            "outputs": list(self.outputs),
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "missing_inputs": list(self.missing_inputs),
            "blocked_outputs": list(self.blocked_outputs),
            "warnings": list(self.warnings),
            "normalized_request": None
            if self.normalized_request is None
            else {
                "target_mode": self.normalized_request.target_mode,
                "preset": self.normalized_request.preset,
                "reasoning_preset": self.normalized_request.reasoning_preset,
                "outputs": list(self.normalized_request.outputs),
                "limit": self.normalized_request.limit,
                "max_cost_usd": self.normalized_request.max_cost_usd,
                "account_usage_budget_usd": (
                    self.normalized_request.account_usage_budget_usd
                ),
                "account_usage_budget_days": (
                    self.normalized_request.account_usage_budget_days
                ),
                "content_ops_cache_policy": (
                    self.normalized_request.content_ops_cache_policy
                ),
                "brand_voice_profile_id": (
                    self.normalized_request.brand_voice_profile_id
                ),
                "ingestion_profile": self.normalized_request.ingestion_profile,
                "require_quality_gates": self.normalized_request.require_quality_gates,
                "allow_unimplemented_outputs": self.normalized_request.allow_unimplemented_outputs,
            },
        }


@dataclass(frozen=True)
class UsageBudgetEvaluation:
    """Account-period budget evaluation for a proposed Content Ops run."""

    budget_usd: float
    period_days: int
    current_cost_usd: float
    estimated_cost_usd: float
    projected_cost_usd: float
    exceeded: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget_usd": round(self.budget_usd, 6),
            "period_days": self.period_days,
            "current_cost_usd": round(self.current_cost_usd, 6),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "projected_cost_usd": round(self.projected_cost_usd, 6),
            "exceeded": self.exceeded,
        }


# PR-Audit-MINOR-Batch-1: ``OUTPUT_CATALOG`` and ``PRESETS`` are
# constants -- wrap them in ``MappingProxyType`` so accidental
# mutation raises ``TypeError`` rather than silently corrupting the
# catalog. Reads (``.get(...)``, iteration, membership) work
# unchanged.
OUTPUT_CATALOG: Mapping[str, OutputDefinition] = MappingProxyType({
    "email_campaign": OutputDefinition(
        id="email_campaign",
        label="Email Campaign",
        description="Cold email and follow-up campaign drafts.",
        implemented=True,
        estimated_unit_cost_usd=0.18,
        required_inputs=("target_account", "offer"),
        default_max_items=3,
        reasoning_requirement="optional_host_context",
    ),
    "blog_post": OutputDefinition(
        id="blog_post",
        label="Blog Post",
        description="Long-form blog draft from intelligence and evidence.",
        implemented=True,
        estimated_unit_cost_usd=0.45,
        required_inputs=("topic",),
        reasoning_requirement="optional_host_context",
    ),
    "report": OutputDefinition(
        id="report",
        label="Report",
        description="Structured intelligence report with references.",
        implemented=True,
        estimated_unit_cost_usd=0.55,
        required_inputs=("opportunity_id",),
        reasoning_requirement="optional_host_context",
    ),
    "landing_page": OutputDefinition(
        id="landing_page",
        label="Landing Page",
        description="Landing page sections for a specific offer and audience.",
        implemented=True,
        estimated_unit_cost_usd=0.65,
        required_inputs=("offer", "audience"),
        reasoning_requirement="optional_host_context",
        default_quality_repair_attempts=LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_DEFAULT,
    ),
    "sales_brief": OutputDefinition(
        id="sales_brief",
        label="Sales Brief",
        description="Sales enablement brief from account intelligence.",
        implemented=True,
        estimated_unit_cost_usd=0.35,
        required_inputs=("target_account",),
        reasoning_requirement="optional_host_context",
    ),
    "social_post": OutputDefinition(
        id="social_post",
        label="Social Posts",
        description="Short social post drafts from source evidence.",
        implemented=True,
        estimated_unit_cost_usd=0.0,
        required_inputs=("source_material",),
        default_max_items=3,
        reasoning_requirement="absent",
        default_parse_retry_attempts=0,
    ),
    "ad_copy": OutputDefinition(
        id="ad_copy",
        label="Ad Copy",
        description="Short paid-media copy drafts from source evidence.",
        implemented=True,
        estimated_unit_cost_usd=0.0,
        required_inputs=("source_material",),
        default_max_items=3,
        reasoning_requirement="absent",
        default_parse_retry_attempts=0,
    ),
    "quote_card": OutputDefinition(
        id="quote_card",
        label="Quote Cards",
        description="Short quote-card drafts from source evidence.",
        implemented=True,
        estimated_unit_cost_usd=0.0,
        required_inputs=("source_material",),
        default_max_items=3,
        reasoning_requirement="absent",
        default_parse_retry_attempts=0,
    ),
    "stat_card": OutputDefinition(
        id="stat_card",
        label="Stat Cards",
        description="Short stat-card drafts from source-backed numeric metrics.",
        implemented=True,
        estimated_unit_cost_usd=0.0,
        required_inputs=("source_material",),
        default_max_items=3,
        reasoning_requirement="absent",
        default_parse_retry_attempts=0,
    ),
    "signal_extraction": OutputDefinition(
        id="signal_extraction",
        label="Signal Extraction",
        description="Extracted opportunity or churn signals from source evidence.",
        implemented=True,
        estimated_unit_cost_usd=0.0,
        required_inputs=("source_material",),
        default_parse_retry_attempts=0,
    ),
    "faq_markdown": OutputDefinition(
        id="faq_markdown",
        label="FAQ Markdown",
        description="Grounded FAQ Markdown from support-ticket source evidence.",
        implemented=True,
        estimated_unit_cost_usd=0.0,
        required_inputs=("source_material",),
        default_max_items=8,
        reasoning_requirement="absent",
        default_parse_retry_attempts=0,
    ),
    "faq_deflection_report": OutputDefinition(
        id="faq_deflection_report",
        label="FAQ Deflection Report",
        description="Customer-facing deflection report from support-ticket evidence.",
        implemented=True,
        estimated_unit_cost_usd=0.0,
        required_inputs=("source_material",),
        default_max_items=20,
        reasoning_requirement="absent",
        default_parse_retry_attempts=0,
    ),
})


PRESETS: Mapping[str, ControlSurfacePreset] = MappingProxyType({
    "email_only": ControlSurfacePreset(
        id="email_only",
        label="Email Only",
        outputs=("email_campaign",),
        description="Lowest-cost outreach draft run.",
    ),
    "intelligence_report": ControlSurfacePreset(
        id="intelligence_report",
        label="Intelligence Report",
        outputs=("report",),
        description="Reference-backed report generation.",
    ),
    "content_marketing": ControlSurfacePreset(
        id="content_marketing",
        label="Content Marketing",
        outputs=("blog_post", "report"),
        description="Blog plus report using the same evidence base.",
    ),
    "marketer_evidence_bundle": ControlSurfacePreset(
        id="marketer_evidence_bundle",
        label="Marketer Evidence Bundle",
        outputs=("landing_page", "blog_post", "sales_brief"),
        description=(
            "Landing page, blog post, and sales brief from review or "
            "competitive evidence."
        ),
    ),
    "lead_gen_campaign": ControlSurfacePreset(
        id="lead_gen_campaign",
        label="Lead Gen Campaign",
        outputs=("email_campaign", "landing_page"),
        description="Outreach plus landing page.",
    ),
    "full_campaign": ControlSurfacePreset(
        id="full_campaign",
        label="Full Campaign",
        outputs=(
            "email_campaign",
            "blog_post",
            "report",
            "landing_page",
            "sales_brief",
        ),
        description="Full generated-content bundle.",
    ),
})


def normalize_outputs(raw_outputs: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize user/API output selection into stable unique ids."""

    if raw_outputs is None:
        return ()
    if isinstance(raw_outputs, str):
        candidates: Sequence[str] = raw_outputs.split(",")
    else:
        candidates = tuple(raw_outputs)

    normalized: list[str] = []
    for item in candidates:
        value = str(item or "").strip().lower().replace("-", "_")
        if value and value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def request_from_mapping(payload: Mapping[str, Any]) -> ContentOpsRequest:
    """Build a ContentOpsRequest from a plain dict payload."""

    limit = int(payload.get("limit") if payload.get("limit") is not None else 1)
    if limit < 1:
        raise ValueError(f"limit must be at least 1; got {limit}")

    max_cost_usd = (
        float(payload["max_cost_usd"])
        if payload.get("max_cost_usd") is not None
        else None
    )
    if max_cost_usd is not None and max_cost_usd <= 0:
        raise ValueError(f"max_cost_usd must be positive; got {max_cost_usd:g}")

    account_usage_budget_usd = (
        float(payload["account_usage_budget_usd"])
        if payload.get("account_usage_budget_usd") is not None
        else None
    )
    if account_usage_budget_usd is not None and account_usage_budget_usd <= 0:
        raise ValueError(
            "account_usage_budget_usd must be positive; "
            f"got {account_usage_budget_usd:g}"
        )

    account_usage_budget_days = int(
        payload.get("account_usage_budget_days")
        if payload.get("account_usage_budget_days") is not None
        else 7
    )
    if account_usage_budget_days < 1 or account_usage_budget_days > 90:
        raise ValueError(
            "account_usage_budget_days must be between 1 and 90; "
            f"got {account_usage_budget_days}"
        )

    raw_inputs = payload.get("inputs")
    if raw_inputs is not None and not isinstance(raw_inputs, Mapping):
        raise ValueError("inputs must be an object")

    return ContentOpsRequest(
        target_mode=str(payload.get("target_mode") or "vendor_retention").strip()
        or "vendor_retention",
        preset=(str(payload.get("preset")).strip() or None)
        if payload.get("preset") is not None
        else None,
        reasoning_preset=(str(payload.get("reasoning_preset")).strip() or None)
        if payload.get("reasoning_preset") is not None
        else None,
        outputs=normalize_outputs(payload.get("outputs")),
        limit=limit,
        max_cost_usd=max_cost_usd,
        account_usage_budget_usd=account_usage_budget_usd,
        account_usage_budget_days=account_usage_budget_days,
        content_ops_cache_policy=normalize_content_ops_cache_policy(
            payload.get("content_ops_cache_policy")
        ),
        brand_voice_profile_id=(
            str(payload.get("brand_voice_profile_id")).strip() or None
        )
        if payload.get("brand_voice_profile_id") is not None
        else None,
        inputs=raw_inputs if isinstance(raw_inputs, Mapping) else {},
        ingestion_profile=str(payload.get("ingestion_profile") or "domain_specific").strip()
        or "domain_specific",
        require_quality_gates=bool(payload.get("require_quality_gates", True)),
        allow_unimplemented_outputs=bool(payload.get("allow_unimplemented_outputs", False)),
    )


def resolve_outputs(request: ContentOpsRequest) -> tuple[str, ...]:
    """Resolve explicit outputs, falling back to a preset when provided."""

    if request.outputs:
        return normalize_outputs(request.outputs)
    if request.preset:
        preset = PRESETS.get(request.preset)
        if preset:
            return preset.outputs
        return ()
    return PRESETS["email_only"].outputs


def _quality_repair_attempts_for_output(
    output_id: str,
    definition: OutputDefinition,
    *,
    inputs: Mapping[str, Any],
    require_quality_gates: bool,
) -> int:
    """Return preview quality-repair attempts for one selected output."""

    repair_attempts = definition.default_quality_repair_attempts
    if output_id == "landing_page":
        override = landing_page_quality_repair_attempts_from_inputs(
            inputs,
        )
        if not require_quality_gates:
            return 0
        if override is not None:
            repair_attempts = override
    elif not require_quality_gates:
        return 0
    return max(0, int(repair_attempts))


def retry_adjusted_unit_cost_usd(
    definition: OutputDefinition,
    *,
    quality_repair_attempts: int | None = None,
) -> float:
    """Return the per-item preview budget including parse and repair attempts."""

    parse_attempts = max(1, int(definition.default_parse_retry_attempts) + 1)
    repair_attempts = (
        definition.default_quality_repair_attempts
        if quality_repair_attempts is None
        else quality_repair_attempts
    )
    repair_attempts = max(0, int(repair_attempts))
    return definition.estimated_unit_cost_usd * parse_attempts * (repair_attempts + 1)


def _brand_voice_requested(
    inputs: Mapping[str, Any],
    *,
    brand_voice_profile_id: str | None,
) -> bool:
    if str(brand_voice_profile_id or "").strip():
        return True
    value = inputs.get("brand_voice")
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(value)
    return True


def _cost_definition_for_output(
    output_id: str,
    definition: OutputDefinition,
    *,
    inputs: Mapping[str, Any],
    brand_voice_profile_id: str | None,
) -> OutputDefinition:
    """Return the dynamic preview-cost definition for one output."""

    if output_id == "social_post" and _brand_voice_requested(
        inputs,
        brand_voice_profile_id=brand_voice_profile_id,
    ):
        return replace(
            definition,
            estimated_unit_cost_usd=SOCIAL_POST_BRAND_VOICE_UNIT_COST_USD,
            default_parse_retry_attempts=SOCIAL_POST_BRAND_VOICE_PARSE_RETRY_ATTEMPTS,
        )
    return definition


def _item_multiplier_for_output(
    output_id: str,
    *,
    inputs: Mapping[str, Any],
) -> int:
    if output_id != "social_post":
        return 1
    channels = inputs.get("social_channels")
    if channels is None:
        channels = inputs.get("social_post_channels")
    return max(1, len(normalize_social_post_channels(channels)))


def estimate_cost_usd(
    outputs: Sequence[str],
    *,
    limit: int,
    inputs: Mapping[str, Any] | None = None,
    require_quality_gates: bool = True,
    brand_voice_profile_id: str | None = None,
) -> float:
    """Estimate cost from selected outputs and opportunity limit.

    The estimates are intentionally conservative placeholders. Generated
    assets default to one parse retry, and landing pages can add quality repair
    attempts. Preview budgets use the worst-case attempt count instead of the
    first-call cost.
    """

    total = 0.0
    provided_inputs = inputs or {}
    for output_id in outputs:
        definition = OUTPUT_CATALOG.get(output_id)
        if not definition:
            continue
        cost_definition = _cost_definition_for_output(
            output_id,
            definition,
            inputs=provided_inputs,
            brand_voice_profile_id=brand_voice_profile_id,
        )
        repair_attempts = _quality_repair_attempts_for_output(
            output_id,
            cost_definition,
            inputs=provided_inputs,
            require_quality_gates=require_quality_gates,
        )
        total += (
            retry_adjusted_unit_cost_usd(
                cost_definition,
                quality_repair_attempts=repair_attempts,
            )
            * max(1, limit)
            * _item_multiplier_for_output(output_id, inputs=provided_inputs)
        )
    return total


def evaluate_usage_budget(
    *,
    budget_usd: float,
    period_days: int,
    current_cost_usd: float,
    estimated_cost_usd: float,
) -> UsageBudgetEvaluation:
    """Return whether current account usage plus a proposed run exceeds budget."""

    if budget_usd <= 0:
        raise ValueError(f"budget_usd must be positive; got {budget_usd:g}")
    if period_days < 1 or period_days > 90:
        raise ValueError(f"period_days must be between 1 and 90; got {period_days}")
    current = max(0.0, float(current_cost_usd))
    estimated = max(0.0, float(estimated_cost_usd))
    projected = current + estimated
    return UsageBudgetEvaluation(
        budget_usd=float(budget_usd),
        period_days=int(period_days),
        current_cost_usd=current,
        estimated_cost_usd=estimated,
        projected_cost_usd=projected,
        exceeded=projected > float(budget_usd),
    )


def missing_required_inputs(
    outputs: Sequence[str],
    provided_inputs: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return required input names missing for selected outputs."""

    missing: list[str] = []
    for output_id in outputs:
        definition = OUTPUT_CATALOG.get(output_id)
        if not definition:
            continue
        for input_name in definition.required_inputs:
            value = provided_inputs.get(input_name)
            if value in (None, "", [], {}, (), set()):
                if input_name not in missing:
                    missing.append(input_name)
    return tuple(missing)


def preview_control_surface(request: ContentOpsRequest) -> ControlSurfacePreview:
    """Validate a generation request and return a preflight plan."""

    outputs = resolve_outputs(request)
    warnings: list[str] = []
    blocked: list[str] = []

    if request.outputs and request.preset:
        warnings.append(
            f"Preset ignored because explicit outputs were provided: {request.preset}"
        )

    if request.preset and not request.outputs and request.preset not in PRESETS:
        blocked.append(request.preset)
        warnings.append(f"Unknown preset: {request.preset}")

    unknown_outputs = tuple(output for output in outputs if output not in OUTPUT_CATALOG)
    for output_id in unknown_outputs:
        blocked.append(output_id)
        warnings.append(f"Unknown output type: {output_id}")

    for output_id in outputs:
        definition = OUTPUT_CATALOG.get(output_id)
        if not definition:
            continue
        if not definition.implemented and not request.allow_unimplemented_outputs:
            blocked.append(output_id)
            warnings.append(f"Output not implemented yet: {output_id}")

    selected_outputs = tuple(output for output in outputs if output not in blocked)
    missing = missing_required_inputs(selected_outputs, request.inputs)
    estimated_cost = estimate_cost_usd(
        selected_outputs,
        limit=request.limit,
        inputs=request.inputs,
        require_quality_gates=request.require_quality_gates,
        brand_voice_profile_id=request.brand_voice_profile_id,
    )

    if request.max_cost_usd is not None and estimated_cost > request.max_cost_usd:
        warnings.append(
            "Estimated cost exceeds max_cost_usd: "
            f"{estimated_cost:.2f} > {request.max_cost_usd:.2f}"
        )

    if request.ingestion_profile not in {"domain_specific", "manual", "existing_evidence"}:
        warnings.append(
            "Unknown ingestion_profile; expected one of "
            "domain_specific, manual, existing_evidence"
        )

    can_run = (
        bool(selected_outputs)
        and not missing
        and not blocked
        and (
            request.max_cost_usd is None
            or estimated_cost <= request.max_cost_usd
        )
    )

    normalized_request = ContentOpsRequest(
        target_mode=request.target_mode,
        preset=request.preset,
        reasoning_preset=request.reasoning_preset,
        outputs=selected_outputs,
        limit=request.limit,
        max_cost_usd=request.max_cost_usd,
        account_usage_budget_usd=request.account_usage_budget_usd,
        account_usage_budget_days=request.account_usage_budget_days,
        content_ops_cache_policy=request.content_ops_cache_policy,
        brand_voice_profile_id=request.brand_voice_profile_id,
        inputs=request.inputs,
        ingestion_profile=request.ingestion_profile,
        require_quality_gates=request.require_quality_gates,
        allow_unimplemented_outputs=request.allow_unimplemented_outputs,
    )

    return ControlSurfacePreview(
        can_run=can_run,
        outputs=selected_outputs,
        estimated_cost_usd=estimated_cost,
        missing_inputs=missing,
        blocked_outputs=tuple(dict.fromkeys(blocked)),
        warnings=tuple(warnings),
        normalized_request=normalized_request,
    )


def preview_from_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Dict-friendly wrapper for API routes and UI smoke tests."""

    return preview_control_surface(request_from_mapping(payload)).as_dict()
