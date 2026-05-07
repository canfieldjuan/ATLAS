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
from .report_generation import ReportGenerationConfig
from .sales_brief_generation import SalesBriefGenerationConfig


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


def _landing_page_config_for_request(request: ContentOpsRequest) -> LandingPageGenerationConfig:
    del request
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
                "parse_retry_attempts": config.parse_retry_attempts,
                "parse_retry_response_excerpt_chars": config.parse_retry_response_excerpt_chars,
                "topic": request.inputs.get("topic"),
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
