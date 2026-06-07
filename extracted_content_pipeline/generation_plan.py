"""Map AI Content Ops control-surface requests into generation plans.

This module is the bridge between preflight preview and actual execution. It
still does not run LLMs or touch storage. It converts a validated control
surface request into asset-specific execution steps that future API handlers can
call. Thrilling bureaucracy, but the useful kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Mapping, Sequence

from .ad_copy_generation import AdCopyGenerationConfig
from .blog_generation import BlogPostGenerationConfig
from .campaign_generation import CampaignGenerationConfig
from .control_surfaces import (
    ContentOpsRequest,
    ControlSurfacePreview,
    preview_control_surface,
    request_from_mapping,
)
from .output_variations import selected_variant_angles
from .landing_page_generation import LandingPageGenerationConfig
from .landing_page_repair_contract import (
    landing_page_quality_repair_attempts_from_inputs,
)
from .quote_card_generation import QuoteCardGenerationConfig
from .reasoning_policy import (
    NOOP_REASONING_PRESETS,
    PACKAGED_REASONING_RUNTIME_OUTPUTS,
    packaged_reasoning_runtime_presets_for_output,
    resolve_reasoning_policy,
)
from .report_generation import ReportGenerationConfig
from .sales_brief_generation import SalesBriefGenerationConfig
from .signal_extraction import SignalExtractionConfig
from .social_post_generation import (
    SocialPostGenerationConfig,
    normalize_social_post_channels,
)
from .stat_card_generation import StatCardGenerationConfig
from .ticket_faq_markdown import (
    DEFAULT_INTENT_RULES,
    TicketFAQMarkdownConfig,
    normalize_intent_rules,
    normalize_vocabulary_gap_rules,
)

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
            "multi_pass_structured for email_campaign, blog_post, and "
            "landing_page, and multi_pass_structured or multi_pass_strict "
            "for report and sales_brief."
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
            "reasoning_preset currently applies only to email_campaign, "
            "blog_post, report, landing_page, and sales_brief."
        )
    for output in runtime_outputs:
        _reasoning_config_for_output(output, request)


def _landing_page_config_for_request(request: ContentOpsRequest) -> LandingPageGenerationConfig:
    """Return landing-page generation config for a control-surface request.

    Landing pages are per-campaign single-shot (one MarketingCampaign in, one
    draft out) so ``limit`` doesn't apply, and per-call copy inputs land on the
    ``MarketingCampaign`` payload built by
    ``content_ops_execution._marketing_campaign_from_inputs`` rather
    than on the config dataclass.
    """
    defaults = LandingPageGenerationConfig()
    repair_attempts = landing_page_quality_repair_attempts_from_inputs(request.inputs)
    if repair_attempts is None:
        return defaults
    return replace(defaults, quality_repair_attempts=repair_attempts)


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


def _variant_config_for_request(request: ContentOpsRequest) -> dict[str, Any]:
    if request.variant_count <= 1:
        return {}
    angles = selected_variant_angles(request.variant_count)
    return {
        "variant_count": len(angles),
        "variant_angles": [angle.as_dict() for angle in angles],
    }


def _brand_voice_config_for_request(request: ContentOpsRequest) -> dict[str, Any]:
    profile_id = str(request.brand_voice_profile_id or "").strip()
    inline = request.inputs.get("brand_voice")
    if not profile_id and isinstance(inline, Mapping):
        profile_id = str(inline.get("id") or inline.get("profile_id") or "").strip()
    return {"brand_voice_profile_id": profile_id} if profile_id else {}


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


def _social_post_config_for_request(request: ContentOpsRequest) -> SocialPostGenerationConfig:
    channels = request.inputs.get("social_channels")
    if channels is None:
        channels = request.inputs.get("social_post_channels")
    config = SocialPostGenerationConfig(
        limit=request.limit,
        channels=normalize_social_post_channels(channels),
    )
    max_text_chars = _positive_int_input(request.inputs, "source_max_text_chars")
    if max_text_chars is None:
        return config
    return SocialPostGenerationConfig(
        limit=config.limit,
        channels=config.channels,
        max_text_chars=max_text_chars,
    )


def _ad_copy_config_for_request(request: ContentOpsRequest) -> AdCopyGenerationConfig:
    config = AdCopyGenerationConfig(limit=request.limit)
    max_text_chars = _positive_int_input(request.inputs, "source_max_text_chars")
    if max_text_chars is None:
        return config
    return AdCopyGenerationConfig(
        limit=config.limit,
        max_text_chars=max_text_chars,
    )


def _quote_card_config_for_request(request: ContentOpsRequest) -> QuoteCardGenerationConfig:
    config = QuoteCardGenerationConfig(limit=request.limit)
    max_text_chars = _positive_int_input(request.inputs, "source_max_text_chars")
    if max_text_chars is None:
        return config
    return QuoteCardGenerationConfig(
        limit=config.limit,
        max_text_chars=max_text_chars,
    )


def _stat_card_config_for_request(request: ContentOpsRequest) -> StatCardGenerationConfig:
    config = StatCardGenerationConfig(limit=request.limit)
    max_text_chars = _positive_int_input(request.inputs, "source_max_text_chars")
    if max_text_chars is None:
        return config
    return StatCardGenerationConfig(
        limit=config.limit,
        max_text_chars=max_text_chars,
    )


def _faq_markdown_config_for_request(request: ContentOpsRequest) -> TicketFAQMarkdownConfig:
    defaults = TicketFAQMarkdownConfig()
    window_days = _positive_int_input(request.inputs, "faq_window_days")
    as_of_date = _text_input(request.inputs, "faq_as_of_date")
    custom_intent_rules = _intent_rules_input(request.inputs, "faq_intent_rules")
    if as_of_date is not None and window_days is None:
        raise ValueError("faq_as_of_date requires faq_window_days")
    if as_of_date is not None:
        try:
            date.fromisoformat(as_of_date)
        except ValueError:
            raise ValueError("faq_as_of_date must use YYYY-MM-DD format") from None
    return TicketFAQMarkdownConfig(
        title=_text_input(request.inputs, "faq_title") or defaults.title,
        max_items=request.limit,
        max_evidence_per_item=(
            _positive_int_input(request.inputs, "faq_max_evidence_per_item")
            or defaults.max_evidence_per_item
        ),
        source_types=_text_sequence_input(request.inputs, "faq_source_types")
        or defaults.source_types,
        max_text_chars=(
            _positive_int_input(request.inputs, "source_max_text_chars")
            or defaults.max_text_chars
        ),
        window_days=window_days,
        as_of_date=as_of_date,
        support_contact=_text_input(request.inputs, "faq_support_contact")
        or defaults.support_contact,
        intent_rules=(
            (*custom_intent_rules, *DEFAULT_INTENT_RULES)
            if custom_intent_rules
            else defaults.intent_rules
        ),
        documentation_terms=(
            _text_sequence_input(request.inputs, "faq_documentation_terms")
            or defaults.documentation_terms
        ),
        vocabulary_gap_rules=(
            _nested_text_sequence_input(request.inputs, "faq_vocabulary_gap_rules")
            or defaults.vocabulary_gap_rules
        ),
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


def _text_input(inputs: Mapping[str, Any], key: str) -> str | None:
    value = str(inputs.get(key) or "").strip()
    return value or None


def _text_sequence_input(inputs: Mapping[str, Any], key: str) -> tuple[str, ...] | None:
    raw = inputs.get(key)
    if raw is None:
        return None
    if isinstance(raw, str):
        items = tuple(item.strip() for item in raw.split(",") if item.strip())
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        items = tuple(str(item).strip() for item in raw if str(item).strip())
    else:
        return None
    return items or None


def _nested_text_sequence_input(
    inputs: Mapping[str, Any],
    key: str,
) -> tuple[tuple[str, ...], ...] | None:
    raw = inputs.get(key)
    if raw is None:
        return None
    return normalize_vocabulary_gap_rules(raw, label=key) or None


def _intent_rules_input(
    inputs: Mapping[str, Any],
    key: str,
) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
    raw = inputs.get(key)
    if raw is None:
        return None
    return normalize_intent_rules(raw, label=key) or None


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
                **_brand_voice_config_for_request(request),
                **_reasoning_config_for_output(output, request),
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
                "quality_repair_attempts": config.quality_repair_attempts,
                "parse_retry_attempts": config.parse_retry_attempts,
                "parse_retry_response_excerpt_chars": config.parse_retry_response_excerpt_chars,
                **_variant_config_for_request(request),
                **_brand_voice_config_for_request(request),
                **_reasoning_config_for_output(output, request),
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
                **_brand_voice_config_for_request(request),
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
                "quality_repair_attempts": config.quality_repair_attempts,
                "parse_retry_attempts": config.parse_retry_attempts,
                "parse_retry_response_excerpt_chars": config.parse_retry_response_excerpt_chars,
                "topic": request.inputs.get("topic"),
                **_variant_config_for_request(request),
                **_brand_voice_config_for_request(request),
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
    if output == "social_post":
        config = _social_post_config_for_request(request)
        return GenerationPlanStep(
            output=output,
            runner="SocialPostGenerationService.generate",
            status="runnable",
            config={
                "skill_name": config.skill_name,
                "channels": list(config.channels),
                "limit": config.limit,
                "max_text_chars": config.max_text_chars,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "parse_retry_attempts": config.parse_retry_attempts,
                "parse_retry_response_excerpt_chars": config.parse_retry_response_excerpt_chars,
                **_brand_voice_config_for_request(request),
            },
        )
    if output == "ad_copy":
        config = _ad_copy_config_for_request(request)
        return GenerationPlanStep(
            output=output,
            runner="AdCopyGenerationService.generate",
            status="runnable",
            config={
                "limit": config.limit,
                "max_text_chars": config.max_text_chars,
            },
        )
    if output == "quote_card":
        config = _quote_card_config_for_request(request)
        return GenerationPlanStep(
            output=output,
            runner="QuoteCardGenerationService.generate",
            status="runnable",
            config={
                "limit": config.limit,
                "max_text_chars": config.max_text_chars,
            },
        )
    if output == "stat_card":
        config = _stat_card_config_for_request(request)
        return GenerationPlanStep(
            output=output,
            runner="StatCardGenerationService.generate",
            status="runnable",
            config={
                "limit": config.limit,
                "max_text_chars": config.max_text_chars,
            },
        )
    if output in {"faq_markdown", "faq_deflection_report"}:
        config = _faq_markdown_config_for_request(request)
        step_config: dict[str, Any] = {
            "title": config.title,
            "max_items": config.max_items,
            "max_evidence_per_item": config.max_evidence_per_item,
            "source_types": list(config.source_types),
            "max_text_chars": config.max_text_chars,
        }
        if config.window_days is not None:
            step_config["window_days"] = config.window_days
        if config.window_days is not None and config.as_of_date is not None:
            step_config["as_of_date"] = config.as_of_date
        if config.support_contact:
            step_config["support_contact"] = config.support_contact
        if config.intent_rules != DEFAULT_INTENT_RULES:
            step_config["intent_rules"] = [
                {"topic": topic, "keywords": list(keywords)}
                for topic, keywords in config.intent_rules
            ]
        if config.documentation_terms:
            step_config["documentation_terms"] = list(config.documentation_terms)
        if config.vocabulary_gap_rules:
            step_config["vocabulary_gap_rules"] = [
                list(rule) for rule in config.vocabulary_gap_rules
            ]
        if output == "faq_deflection_report":
            step_config["report_title"] = (
                _text_input(request.inputs, "deflection_report_title")
                or "Support Ticket Deflection Report"
            )
        return GenerationPlanStep(
            output=output,
            runner=(
                "FAQDeflectionReportService.generate"
                if output == "faq_deflection_report"
                else "TicketFAQMarkdownService.generate"
            ),
            status="runnable",
            config=step_config,
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
