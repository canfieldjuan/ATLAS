"""Map AI Content Ops control-surface requests into generation plans.

This module is the bridge between preflight preview and actual execution. It
still does not run LLMs or touch storage. It converts a validated control
surface request into asset-specific execution steps that future API handlers can
call. Thrilling bureaucracy, but the useful kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .blog_generation import BlogPostGenerationConfig
from .campaign_generation import CampaignGenerationConfig
from .control_surfaces import (
    ContentOpsRequest,
    ControlSurfacePreview,
    preview_control_surface,
    request_from_mapping,
)
from .landing_page_generation import LandingPageGenerationConfig
from .reasoning_policy import (
    NOOP_REASONING_PRESETS,
    PACKAGED_REASONING_RUNTIME_OUTPUTS,
    packaged_reasoning_runtime_presets_for_output,
    resolve_reasoning_policy,
)
from .report_generation import ReportGenerationConfig
from .sales_brief_generation import SalesBriefGenerationConfig
from .signal_extraction import SignalExtractionConfig


@dataclass(frozen=True)
class GenerationPlanStep:
    """One runnable or planned asset generation step."""

    output: str
    runner: str
    status: str
    config: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "runner": self.runner,
            "status": self.status,
            "config": dict(self.config),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GenerationPlan:
    """Resolved generation plan for a normalized content-ops request."""

    can_execute: bool
    target_mode: str
    limit: int
    steps: tuple[GenerationPlanStep, ...]
    preview: ControlSurfacePreview

    def as_dict(self) -> dict[str, Any]:
        return {
            "can_execute": self.can_execute,
            "target_mode": self.target_mode,
            "limit": self.limit,
            "steps": [step.as_dict() for step in self.steps],
            "preview": self.preview.as_dict(),
        }


def _campaign_config_for_request(request: ContentOpsRequest) -> CampaignGenerationConfig:
    channels = request.inputs.get("channels")
    if channels is None:
        channels = request.inputs.get("campaign_channels")
    if isinstance(channels, str):
        channel_tuple = tuple(
            item.strip()
            for item in channels.split(",")
            if item.strip()
        )
    elif isinstance(channels, Sequence) and not isinstance(channels, (bytes, bytearray)):
        channel_tuple = tuple(
            str(item or "").strip()
            for item in channels
            if str(item or "").strip()
        )
    else:
        channel_tuple = ("email_cold", "email_followup")

    return CampaignGenerationConfig(
        channels=channel_tuple,
        limit=request.limit,
        quality_revalidation_enabled=request.require_quality_gates,
    )


def _report_config_for_request(request: ContentOpsRequest) -> ReportGenerationConfig:
    report_type = str(
        request.inputs.get("report_type")
        or "vendor_pressure"
    ).strip() or "vendor_pressure"
    return ReportGenerationConfig(
        default_report_type=report_type,
        limit=request.limit,
    )


def _reasoning_config_for_output(output: str, request: ContentOpsRequest) -> dict[str, Any]:
    # Informational plan metadata only. Runtime behavior is applied by
    # ContentOpsExecutionServices.with_reasoning_context at execute time.
    if request.reasoning_preset is None:
        return {}
    _policy, definition = resolve_reasoning_policy(output, request.reasoning_preset)
    if definition.id in NOOP_REASONING_PRESETS:
        return {}
    runtime_presets = packaged_reasoning_runtime_presets_for_output(output)
    if definition.id not in runtime_presets:
        raise ValueError(
            "Content Ops packaged reasoning currently supports "
            "multi_pass_structured for blog_post and "
            "multi_pass_structured or multi_pass_strict for report and sales_brief."
        )
    return {
        "reasoning_preset": definition.id,
        "reasoning_multi_pass": definition.multi_pass,
        "reasoning_narrative_planning": definition.narrative_planning,
        "reasoning_output_validation": definition.output_validation,
        "reasoning_blocking_validation": definition.blocking_validation,
        "reasoning_falsification": definition.falsification,
    }


def _validate_reasoning_runtime_request(
    outputs: tuple[str, ...],
    request: ContentOpsRequest,
) -> None:
    preset = request.reasoning_preset
    if preset is None:
        return
    selected = str(preset or "").strip()
    if not selected or selected in NOOP_REASONING_PRESETS:
        return
    runtime_outputs = tuple(
        output for output in outputs if output in PACKAGED_REASONING_RUNTIME_OUTPUTS
    )
    if not runtime_outputs:
        raise ValueError(
            "reasoning_preset currently applies only to blog_post, report, "
            "and sales_brief."
        )
    for output in runtime_outputs:
        _reasoning_config_for_output(output, request)


def _landing_page_config_for_request(request: ContentOpsRequest) -> LandingPageGenerationConfig:
    """Return defaults; the request is intentionally not consumed.

    Other ``_*_config_for_request`` helpers thread ``request.inputs``
    or ``request.limit`` into their config. Landing pages are
    per-campaign single-shot (one MarketingCampaign in, one draft
    out) so ``limit`` doesn't apply, and per-call inputs land on the
    ``MarketingCampaign`` payload built by
    ``content_ops_execution._marketing_campaign_from_inputs`` rather
    than on the config dataclass. Discarding the request here is
    deliberate -- documented to close the audit-trail gap on the
    helper-shape asymmetry (PR-Audit-MINOR-Batch-3).
    """
    del request  # intentional; see docstring
    return LandingPageGenerationConfig()


def _sales_brief_config_for_request(request: ContentOpsRequest) -> SalesBriefGenerationConfig:
    brief_type = str(
        request.inputs.get("brief_type")
        or "pre_call"
    ).strip() or "pre_call"
    return SalesBriefGenerationConfig(
        default_brief_type=brief_type,
        limit=request.limit,
    )


def _blog_post_config_for_request(request: ContentOpsRequest) -> BlogPostGenerationConfig:
    return BlogPostGenerationConfig(limit=request.limit)


def _signal_extraction_config_for_request(
    request: ContentOpsRequest,
) -> SignalExtractionConfig:
    config = SignalExtractionConfig(limit=request.limit)
    max_text_chars = _positive_int_input(request.inputs, "source_max_text_chars")
    if max_text_chars is None:
        return config
    return SignalExtractionConfig(
        limit=config.limit,
        max_text_chars=max_text_chars,
    )


def _positive_int_input(inputs: Mapping[str, Any], key: str) -> int | None:
    raw = inputs.get(key)
    if raw is None:
        return None
    if isinstance(raw, (bool, float)):
        raise ValueError(f"{key} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer") from None
    if value < 1:
        raise ValueError(f"{key} must be at least 1; got {value}")
    return value


def _step_for_output(output: str, request: ContentOpsRequest) -> GenerationPlanStep:
    if output == "email_campaign":
        config = _campaign_config_for_request(request)
        return GenerationPlanStep(
            output=output,
            runner="CampaignGenerationService.generate",
            status="runnable",
            config={
                "skill_name": config.skill_name,
                "channels": list(config.channels),
                "limit": config.limit,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "quality_revalidation_enabled": config.quality_revalidation_enabled,
                "quality_prompt_proof_term_limit": config.quality_prompt_proof_term_limit,
                "parse_retry_attempts": config.parse_retry_attempts,
                "parse_retry_response_excerpt_chars": config.parse_retry_response_excerpt_chars,
            },
        )
    if output == "report":
        config = _report_config_for_request(request)
        return GenerationPlanStep(
            output=output,
            runner="ReportGenerationService.generate",
            status="runnable",
            config={
                "skill_name": config.skill_name,
                "default_report_type": config.default_report_type,
                "limit": config.limit,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "quality_gates_enabled": request.require_quality_gates,
                "parse_retry_attempts": config.parse_retry_attempts,
                "parse_retry_response_excerpt_chars": config.parse_retry_response_excerpt_chars,
                **_reasoning_config_for_output(output, request),
            },
        )
    if output == "landing_page":
        config = _landing_page_config_for_request(request)
        return GenerationPlanStep(
            output=output,
            runner="LandingPageGenerationService.generate",
            status="runnable",
            config={
                "skill_name": config.skill_name,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "quality_gates_enabled": request.require_quality_gates,
                "parse_retry_attempts": config.parse_retry_attempts,
                "parse_retry_response_excerpt_chars": config.parse_retry_response_excerpt_chars,
            },
        )
    if output == "sales_brief":
        config = _sales_brief_config_for_request(request)
        return GenerationPlanStep(
            output=output,
            runner="SalesBriefGenerationService.generate",
            status="runnable",
            config={
                "skill_name": config.skill_name,
                "default_brief_type": config.default_brief_type,
                "limit": config.limit,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "quality_gates_enabled": request.require_quality_gates,
                "parse_retry_attempts": config.parse_retry_attempts,
                "parse_retry_response_excerpt_chars": config.parse_retry_response_excerpt_chars,
                **_reasoning_config_for_output(output, request),
            },
        )
    if output == "blog_post":
        config = _blog_post_config_for_request(request)
        return GenerationPlanStep(
            output=output,
            runner="BlogPostGenerationService.generate",
            status="runnable",
            config={
                "skill_name": config.skill_name,
                "limit": config.limit,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "quality_gates_enabled": request.require_quality_gates,
                "parse_retry_attempts": config.parse_retry_attempts,
                "parse_retry_response_excerpt_chars": config.parse_retry_response_excerpt_chars,
                "topic": request.inputs.get("topic"),
                **_reasoning_config_for_output(output, request),
            },
        )
    if output == "signal_extraction":
        config = _signal_extraction_config_for_request(request)
        return GenerationPlanStep(
            output=output,
            runner="SignalExtractionService.generate",
            status="runnable",
            config={
                "limit": config.limit,
                "max_text_chars": config.max_text_chars,
            },
        )
    return GenerationPlanStep(
        output=output,
        runner="",
        status="blocked",
        reason="No generation runner is mapped for this output.",
    )


def build_generation_plan(request: ContentOpsRequest) -> GenerationPlan:
    """Build an execution plan from a content-ops request.

    The plan only becomes executable when preview passes, no selected output was
    blocked, and every selected step is runnable. Planned steps are deliberately
    not executed by future callers until they get service-shaped adapters.
    """

    preview = preview_control_surface(request)
    normalized = preview.normalized_request or request
    _validate_reasoning_runtime_request(preview.outputs, normalized)
    steps = tuple(_step_for_output(output, normalized) for output in preview.outputs)
    can_execute = (
        preview.can_run
        and not preview.blocked_outputs
        and all(step.status == "runnable" for step in steps)
    )
    return GenerationPlan(
        can_execute=can_execute,
        target_mode=normalized.target_mode,
        limit=normalized.limit,
        steps=steps,
        preview=preview,
    )


def build_generation_plan_from_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Dict-friendly wrapper for API routes and UI smoke tests."""

    return build_generation_plan(request_from_mapping(payload)).as_dict()


__all__ = [
    "GenerationPlan",
    "GenerationPlanStep",
    "build_generation_plan",
    "build_generation_plan_from_mapping",
]
