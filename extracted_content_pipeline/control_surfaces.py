"""Control-surface primitives for AI Content Ops generation.

This module is intentionally pure and deterministic. It does not call an LLM,
read a database, or know about HTTP. The UI/API layer should use it to turn
user-facing choices into a validated generation plan before any expensive work
runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


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
    # Must mirror each *GenerationConfig.parse_retry_attempts default used in
    # generation_plan.py. If a service raises retries, update this field too.
    default_parse_retry_attempts: int = 1


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
                "ingestion_profile": self.normalized_request.ingestion_profile,
                "require_quality_gates": self.normalized_request.require_quality_gates,
                "allow_unimplemented_outputs": self.normalized_request.allow_unimplemented_outputs,
            },
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


def retry_adjusted_unit_cost_usd(definition: OutputDefinition) -> float:
    """Return the per-item preview budget including default parse retries."""

    attempts = max(1, int(definition.default_parse_retry_attempts) + 1)
    return definition.estimated_unit_cost_usd * attempts


def estimate_cost_usd(outputs: Sequence[str], *, limit: int) -> float:
    """Estimate cost from selected outputs and opportunity limit.

    The estimates are intentionally conservative placeholders. Generated
    assets default to one parse retry, so preview budgets use the worst-case
    attempt count instead of the first-call cost.
    """

    total = 0.0
    for output_id in outputs:
        definition = OUTPUT_CATALOG.get(output_id)
        if not definition:
            continue
        total += retry_adjusted_unit_cost_usd(definition) * max(1, limit)
    return total


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
    estimated_cost = estimate_cost_usd(selected_outputs, limit=request.limit)

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
