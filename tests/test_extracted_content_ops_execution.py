from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.campaign_llm_client import (
    current_content_ops_llm_trace_context,
)
from extracted_content_pipeline.ad_copy_generation import AdCopyGenerationService
from extracted_content_pipeline.content_ops_cache_policy import (
    ContentOpsExactCachePolicy,
)
from extracted_content_pipeline.content_ops_execution import (
    ContentOpsExecutionServices,
    execute_content_ops_from_mapping,
)
from extracted_content_pipeline.output_variations import VARIANT_ANGLES
from extracted_content_pipeline.faq_deflection_report import FAQDeflectionReportService
from extracted_content_pipeline.quote_card_generation import QuoteCardGenerationService
from extracted_content_pipeline.signal_extraction import SignalExtractionService
from extracted_content_pipeline.social_post_generation import SocialPostGenerationService
from extracted_content_pipeline.stat_card_generation import StatCardGenerationService
from extracted_content_pipeline.ticket_faq_markdown import TicketFAQMarkdownService


@dataclass(frozen=True)
class _Result:
    generated: int = 1
    reasoning_contexts_used: int = 0
    consumed_reasoning_contexts: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        data = {
            "generated": self.generated,
            "reasoning_contexts_used": self.reasoning_contexts_used,
            "saved_ids": ["draft-1"],
        }
        if self.consumed_reasoning_contexts:
            data["consumed_reasoning_contexts"] = [
                dict(item) for item in self.consumed_reasoning_contexts
            ]
        return data


class _OpportunityService:
    """Records every kwarg the executor passes; tolerates new optional kwargs.

    Accepts the smoking-gun per-call overrides PR-OptionA-1 threads through
    (`channels`, `default_report_type`, `default_brief_type`) plus a generic
    `**extras` so future overrides don't require touching this fake.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        channels: Any | None = None,
        default_report_type: str | None = None,
        default_brief_type: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        parse_retry_attempts: int | None = None,
        quality_repair_attempts: int | None = None,
        quality_revalidation_enabled: bool | None = None,
        quality_prompt_proof_term_limit: int | None = None,
        parse_retry_response_excerpt_chars: int | None = None,
        quality_gates_enabled: bool | None = None,
        topic: str | None = None,
        brand_voice: Any | None = None,
        opportunity_defaults: Mapping[str, Any] | None = None,
        source_material: Any | None = None,
        **extras: Any,
    ) -> _Result:
        self.calls.append({
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "filters": dict(filters or {}),
            "channels": channels,
            "default_report_type": default_report_type,
            "default_brief_type": default_brief_type,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "parse_retry_attempts": parse_retry_attempts,
            "quality_repair_attempts": quality_repair_attempts,
            "quality_revalidation_enabled": quality_revalidation_enabled,
            "quality_prompt_proof_term_limit": quality_prompt_proof_term_limit,
            "parse_retry_response_excerpt_chars": parse_retry_response_excerpt_chars,
            "quality_gates_enabled": quality_gates_enabled,
            "topic": topic,
            "brand_voice": brand_voice,
            "opportunity_defaults": dict(opportunity_defaults or {}),
            "source_material": source_material,
            "extras": dict(extras),
        })
        return _Result()


class _VariantBlogService:
    def __init__(self, *, fail_on_angle: str = "") -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_on_angle = fail_on_angle

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        variant_angle: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append({
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "filters": dict(filters or {}),
            "variant_angle": variant_angle,
            "kwargs": dict(kwargs),
        })
        if self.fail_on_angle and self.fail_on_angle in str(variant_angle or ""):
            raise RuntimeError("variant failed")
        call_no = len(self.calls)
        return {
            "requested": int(limit or 1),
            "generated": 1,
            "skipped": 0,
            "reasoning_contexts_used": 0,
            "saved_ids": [f"draft-{call_no}"],
            "errors": [],
        }


class _VariantSalesBriefService:
    def __init__(self, *, fail_on_angle: str = "") -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_on_angle = fail_on_angle

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        variant_angle: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append({
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "filters": dict(filters or {}),
            "variant_angle": variant_angle,
            "kwargs": dict(kwargs),
        })
        if self.fail_on_angle and self.fail_on_angle in str(variant_angle or ""):
            raise RuntimeError("sales brief variant failed")
        call_no = len(self.calls)
        return {
            "requested": int(limit or 1),
            "generated": 1,
            "skipped": 0,
            "reasoning_contexts_used": 1,
            "consumed_reasoning_contexts": [{
                "target_id": f"sales-brief-{call_no}",
                "variant_angle": variant_angle,
            }],
            "saved_ids": [f"sales-brief-{call_no}"],
            "errors": [],
        }


class _ReasoningAwareOpportunityService(_OpportunityService):
    def __init__(self, reasoning_context: Any | None = None) -> None:
        super().__init__()
        self._reasoning_context = reasoning_context

    def with_reasoning_context(
        self,
        provider: Any | None,
    ) -> "_ReasoningAwareOpportunityService":
        return _ReasoningAwareOpportunityService(reasoning_context=provider)


class _StrictCampaignService:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        channels: Any | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        parse_retry_attempts: int | None = None,
        quality_revalidation_enabled: bool | None = None,
        quality_prompt_proof_term_limit: int | None = None,
        parse_retry_response_excerpt_chars: int | None = None,
        brand_voice: Any | None = None,
    ) -> _Result:
        del scope
        del target_mode
        del limit
        del filters
        del channels
        del temperature
        del max_tokens
        del parse_retry_attempts
        del quality_revalidation_enabled
        del quality_prompt_proof_term_limit
        del parse_retry_response_excerpt_chars
        del brand_voice
        self.calls += 1
        return _Result()


class _TraceContextCaptureService(_OpportunityService):
    async def generate(self, **kwargs: Any) -> _Result:
        await super().generate(**kwargs)
        self.calls[-1]["trace_context"] = current_content_ops_llm_trace_context()
        return _Result()


def test_services_with_reasoning_context_can_target_specific_outputs() -> None:
    provider = object()
    campaign = _ReasoningAwareOpportunityService()
    report = _ReasoningAwareOpportunityService()
    bundle = ContentOpsExecutionServices(campaign=campaign, report=report)

    derived = bundle.with_reasoning_context(provider, outputs=("report",))

    assert derived.campaign is campaign
    assert derived.report is not report
    assert derived.report._reasoning_context is provider
    assert derived.reasoning_provider_active_for("report") is True
    assert derived.reasoning_provider_active_for("email_campaign") is False


def test_services_with_reasoning_context_preserves_ad_copy_service() -> None:
    ad_copy = AdCopyGenerationService()
    derived = ContentOpsExecutionServices(ad_copy=ad_copy).with_reasoning_context(
        object()
    )

    assert derived.ad_copy is ad_copy
    assert derived.for_output("ad_copy") is ad_copy
    assert "ad_copy" in derived.configured_outputs()


def test_services_with_reasoning_context_preserves_stat_card_service() -> None:
    stat_card = StatCardGenerationService()
    derived = ContentOpsExecutionServices(stat_card=stat_card).with_reasoning_context(
        object()
    )

    assert derived.stat_card is stat_card
    assert derived.for_output("stat_card") is stat_card
    assert "stat_card" in derived.configured_outputs()


def test_services_with_reasoning_context_accumulates_targeted_outputs() -> None:
    campaign_provider = object()
    report_provider = object()
    campaign = _ReasoningAwareOpportunityService()
    report = _ReasoningAwareOpportunityService()
    bundle = ContentOpsExecutionServices(campaign=campaign, report=report)

    derived = bundle.with_reasoning_context(
        campaign_provider,
        outputs=("email_campaign",),
    ).with_reasoning_context(
        report_provider,
        outputs=("report",),
    )

    assert derived.reasoning_provider_active_for("email_campaign") is True
    assert derived.reasoning_provider_active_for("report") is True
    assert derived.campaign._reasoning_context is campaign_provider
    assert derived.report._reasoning_context is report_provider


def test_services_with_reasoning_context_none_clears_targeted_output() -> None:
    campaign_provider = object()
    report_provider = object()
    campaign = _ReasoningAwareOpportunityService()
    report = _ReasoningAwareOpportunityService()
    bundle = ContentOpsExecutionServices(campaign=campaign, report=report)

    derived = bundle.with_reasoning_context(
        campaign_provider,
        outputs=("email_campaign",),
    ).with_reasoning_context(
        report_provider,
        outputs=("report",),
    ).with_reasoning_context(
        None,
        outputs=("email_campaign",),
    )

    assert derived.reasoning_provider_active_for("email_campaign") is False
    assert derived.reasoning_provider_active_for("report") is True
    assert derived.campaign._reasoning_context is None
    assert derived.report._reasoning_context is report_provider


def test_services_with_reasoning_context_none_clears_from_global_provider() -> None:
    campaign = _ReasoningAwareOpportunityService(reasoning_context=object())
    report = _ReasoningAwareOpportunityService(reasoning_context=object())
    bundle = ContentOpsExecutionServices(
        campaign=campaign,
        report=report,
        reasoning_provider_configured=True,
    )

    derived = bundle.with_reasoning_context(None, outputs=("email_campaign",))

    assert derived.reasoning_provider_active_for("email_campaign") is False
    assert derived.reasoning_provider_active_for("report") is True
    assert derived.campaign._reasoning_context is None
    assert derived.report is report


def test_services_rejects_outputs_without_provider_flag() -> None:
    with pytest.raises(ValueError, match="requires reasoning_provider_configured"):
        ContentOpsExecutionServices(reasoning_provider_outputs=("report",))


def test_services_with_reasoning_context_honors_empty_output_selection() -> None:
    campaign = _ReasoningAwareOpportunityService()
    derived = ContentOpsExecutionServices(campaign=campaign).with_reasoning_context(
        object(),
        outputs=(),
    )

    assert derived.campaign is campaign
    assert derived.reasoning_provider_configured is False


def test_services_with_reasoning_context_rejects_string_outputs() -> None:
    campaign = _ReasoningAwareOpportunityService()
    with pytest.raises(TypeError, match="outputs must be a sequence"):
        ContentOpsExecutionServices(campaign=campaign).with_reasoning_context(
            object(),
            outputs="report",
        )


def test_services_with_reasoning_context_rejects_unknown_outputs() -> None:
    campaign = _ReasoningAwareOpportunityService()
    with pytest.raises(ValueError, match="unknown reasoning-aware outputs"):
        ContentOpsExecutionServices(campaign=campaign).with_reasoning_context(
            object(),
            outputs=("signal_extraction",),
        )


class _LandingPageService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        scope: TenantScope,
        campaign: Any,
        temperature: float | None = None,
        max_tokens: int | None = None,
        parse_retry_attempts: int | None = None,
        parse_retry_response_excerpt_chars: int | None = None,
        quality_gates_enabled: bool | None = None,
        quality_repair_attempts: int | None = None,
        brand_voice: Any | None = None,
        **extras: Any,
    ) -> _Result:
        self.calls.append({
            "scope": scope,
            "campaign": campaign,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "parse_retry_attempts": parse_retry_attempts,
            "parse_retry_response_excerpt_chars": parse_retry_response_excerpt_chars,
            "quality_gates_enabled": quality_gates_enabled,
            "quality_repair_attempts": quality_repair_attempts,
            "brand_voice": brand_voice,
            "extras": dict(extras),
        })
        return _Result()


class _VariantLandingPageService:
    def __init__(self, *, fail_on_angle: str = "") -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_on_angle = fail_on_angle

    async def generate(
        self,
        *,
        scope: TenantScope,
        campaign: Any,
        variant_angle: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append({
            "scope": scope,
            "campaign": campaign,
            "variant_angle": variant_angle,
            "kwargs": dict(kwargs),
        })
        if self.fail_on_angle and self.fail_on_angle in str(variant_angle or ""):
            raise RuntimeError("landing variant failed")
        call_no = len(self.calls)
        return {
            "requested": 1,
            "generated": 1,
            "skipped": 0,
            "reasoning_contexts_used": 0,
            "saved_ids": [f"landing-page-{call_no}"],
            "errors": [],
        }


class _BarrierOpportunityService:
    def __init__(
        self,
        *,
        name: str,
        starts: list[str],
        all_started: asyncio.Event,
    ) -> None:
        self.name = name
        self.starts = starts
        self.all_started = all_started

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        **extras: Any,
    ) -> _Result:
        del extras  # PR-OptionA-1 may pass channels/default_*type kwargs.
        self.starts.append(self.name)
        if len(self.starts) == 2:
            self.all_started.set()
        await self.all_started.wait()
        return _Result()


@pytest.mark.asyncio
async def test_execute_runs_email_and_report_services_with_scope_and_filters() -> None:
    campaign = _OpportunityService()
    report = _OpportunityService()
    scope = TenantScope(account_id="acct-1")

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign", "report"],
            "target_mode": "vendor_retention",
            "limit": 2,
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
                "opportunity_id": "opp-1",
                "filters": {"status": "ready"},
            },
        },
        services=ContentOpsExecutionServices(campaign=campaign, report=report),
        scope=scope,
    )

    assert result["status"] == "completed"
    assert [step["output"] for step in result["steps"]] == ["email_campaign", "report"]
    # Both services receive scope, target_mode, limit, filters; the executor
    # also threads each output's smoking-gun step.config field as a kwarg
    # (channels for email_campaign, default_report_type for report).
    # PR-OptionA-1: this is what makes plan-as-execution-contract real.
    assert len(campaign.calls) == 1
    campaign_call = campaign.calls[0]
    assert campaign_call["scope"] == scope
    assert campaign_call["target_mode"] == "vendor_retention"
    assert campaign_call["limit"] == 2
    assert campaign_call["filters"] == {"status": "ready"}
    # Plan default channels reach the service even without per-request override.
    assert campaign_call["channels"] == ("email_cold", "email_followup")
    assert campaign_call["opportunity_defaults"] == {}
    assert campaign_call["default_report_type"] is None
    # extras locks the kwarg surface: a typo'd kwarg in PR-OptionA-2/3 would
    # silently land here and never reach the right service field.
    assert campaign_call["extras"] == {}
    assert len(report.calls) == 1
    report_call = report.calls[0]
    assert report_call["scope"] == scope
    assert report_call["target_mode"] == "vendor_retention"
    assert report_call["limit"] == 2
    assert report_call["filters"] == {"status": "ready"}
    # Plan default report type reaches the service.
    assert report_call["default_report_type"] == "vendor_pressure"
    assert report_call["channels"] is None
    assert report_call["opportunity_defaults"] == {}
    assert report_call["extras"] == {}


@pytest.mark.asyncio
async def test_execute_sets_content_ops_llm_trace_context_from_scope() -> None:
    campaign = _TraceContextCaptureService()
    scope = TenantScope(account_id="acct-1", user_id="user-1")

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "target_mode": "vendor_retention",
            "limit": 1,
            "inputs": {"target_account": "Acme", "offer": "Churn audit"},
        },
        services=ContentOpsExecutionServices(campaign=campaign),
        scope=scope,
    )

    assert result["status"] == "completed"
    assert campaign.calls[0]["trace_context"] == {
        "account_id": "acct-1",
        "user_id": "user-1",
    }
    assert current_content_ops_llm_trace_context() == {}


@pytest.mark.asyncio
async def test_execute_marks_support_ticket_trace_context_for_cache_policy() -> None:
    blog = _TraceContextCaptureService()
    scope = TenantScope(account_id="acct-1", user_id="user-1")

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["blog_post"],
            "content_ops_cache_policy": "exact-cache",
            "inputs": {
                "topic": "Support-ticket questions customers keep asking",
                "filters": {"topic_type": "content_ops_support_ticket_faq"},
                "included_ticket_row_count": 25,
                "top_ticket_clusters": [{"intent": "billing", "count": 12}],
            },
        },
        services=ContentOpsExecutionServices(blog_post=blog),
        scope=scope,
    )

    assert result["status"] == "completed"
    trace_context = blog.calls[0]["trace_context"]
    assert trace_context == {
        "account_id": "acct-1",
        "user_id": "user-1",
        "content_ops_cache_policy": "exact",
        "source_type": "support_ticket",
        "input_provider": "support_ticket_provider",
    }
    decision = ContentOpsExactCachePolicy(exact_cache_enabled=True).decide({
        **trace_context,
        "target_mode": "vendor_retention",
        "blueprint_id": "support-ticket-blog",
        "skill_name": "digest/blog_post_generation",
        "asset_type": "blog_post",
        "attempt_no": 1,
    })
    assert decision.mode == "no_store"
    assert decision.reason == "customer_data_no_store"
    assert current_content_ops_llm_trace_context() == {}


@pytest.mark.asyncio
async def test_execute_runs_independent_steps_concurrently_preserving_plan_order() -> None:
    starts: list[str] = []
    all_started = asyncio.Event()
    campaign = _BarrierOpportunityService(
        name="email_campaign",
        starts=starts,
        all_started=all_started,
    )
    report = _BarrierOpportunityService(
        name="report",
        starts=starts,
        all_started=all_started,
    )

    result = await asyncio.wait_for(
        execute_content_ops_from_mapping(
            {
                "outputs": ["email_campaign", "report"],
                "target_mode": "vendor_retention",
                "inputs": {
                    "target_account": "Acme",
                    "offer": "Churn audit",
                    "opportunity_id": "opp-1",
                },
            },
            services=ContentOpsExecutionServices(campaign=campaign, report=report),
        ),
        timeout=1.0,
    )

    assert starts == ["email_campaign", "report"]
    assert result["status"] == "completed"
    assert [step["output"] for step in result["steps"]] == ["email_campaign", "report"]


@pytest.mark.asyncio
async def test_execute_runs_landing_page_with_marketing_campaign_input() -> None:
    service = _LandingPageService()
    scope = TenantScope(account_id="acct-1")

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {
                "campaign_name": "Q2 churn audit",
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
                "vendors": "HubSpot, Salesforce",
                "tags": ["retention", "pipeline"],
            },
        },
        services=ContentOpsExecutionServices(landing_page=service),
        scope=scope,
    )

    assert result["status"] == "completed"
    campaign = service.calls[0]["campaign"]
    assert campaign.name == "Q2 churn audit"
    assert campaign.value_prop == "Churn audit"
    assert campaign.persona == "B2B SaaS founders"
    assert campaign.vendors == ("HubSpot", "Salesforce")
    assert campaign.tags == ("retention", "pipeline")


@pytest.mark.asyncio
async def test_execute_threads_brand_voice_to_supported_copy_outputs() -> None:
    campaign = _OpportunityService()
    blog = _OpportunityService()
    sales_brief = _OpportunityService()
    landing_page = _LandingPageService()
    social_post = _OpportunityService()
    scope = TenantScope(account_id="acct-1")

    result = await execute_content_ops_from_mapping(
        {
            "outputs": [
                "email_campaign",
                "blog_post",
                "landing_page",
                "sales_brief",
                "social_post",
            ],
            "target_mode": "vendor_retention",
            "brand_voice_profile_id": "acme-main",
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
                "topic": "Churn pressure",
                "audience": "B2B SaaS founders",
                "source_material": [
                    {
                        "review_id": "review-1",
                        "review_text": "Pricing is a problem after renewal.",
                    }
                ],
                "brand_voice": {
                    "id": "acme-main",
                    "account_id": "acct-1",
                    "name": "Acme main voice",
                    "descriptors": ["plainspoken"],
                    "exemplars": ["You get the tradeoff in one paragraph."],
                },
            },
        },
        services=ContentOpsExecutionServices(
            campaign=campaign,
            blog_post=blog,
            landing_page=landing_page,
            sales_brief=sales_brief,
            social_post=social_post,
        ),
        scope=scope,
    )

    assert result["status"] == "completed"
    for call in (
        campaign.calls[0],
        blog.calls[0],
        sales_brief.calls[0],
        landing_page.calls[0],
        social_post.calls[0],
    ):
        assert call["brand_voice"].id == "acme-main"
        assert call["brand_voice"].account_id == "acct-1"


@pytest.mark.asyncio
async def test_execute_blocks_brand_voice_profile_id_without_inline_profile() -> None:
    campaign = _OpportunityService()

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "brand_voice_profile_id": "acme-main",
            "inputs": {"target_account": "Acme", "offer": "Churn audit"},
        },
        services=ContentOpsExecutionServices(campaign=campaign),
        scope=TenantScope(account_id="acct-1"),
    )

    assert result["status"] == "blocked"
    assert result["errors"] == [{"reason": "brand_voice_profile_id requires inputs.brand_voice"}]
    assert campaign.calls == []


@pytest.mark.asyncio
async def test_execute_blocks_cross_tenant_inline_brand_voice() -> None:
    campaign = _OpportunityService()

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
                "brand_voice": {
                    "account_id": "acct-2",
                    "name": "Wrong tenant voice",
                    "descriptors": ["warm"],
                },
            },
        },
        services=ContentOpsExecutionServices(campaign=campaign),
        scope=TenantScope(account_id="acct-1"),
    )

    assert result["status"] == "blocked"
    assert result["errors"] == [
        {"reason": "brand_voice.account_id does not match tenant scope"}
    ]
    assert campaign.calls == []


@pytest.mark.asyncio
async def test_execute_runs_blog_post_service_with_scope_and_filters() -> None:
    service = _OpportunityService()
    scope = TenantScope(account_id="acct-1")

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["blog_post"],
            "limit": 2,
            "inputs": {
                "topic": "Churn pressure",
                "filters": {"topic_type": "vendor_alternative"},
            },
        },
        services=ContentOpsExecutionServices(blog_post=service),
        scope=scope,
    )

    assert result["status"] == "completed"
    assert result["steps"][0]["output"] == "blog_post"
    # blog_post stays on the default dispatcher in PR-OptionA-1; per-call
    # config kwargs are deferred to PR-OptionA-2 (see plans/PR-OptionA-1.md).
    assert len(service.calls) == 1
    blog_call = service.calls[0]
    assert blog_call["scope"] == scope
    assert blog_call["target_mode"] == "vendor_retention"
    assert blog_call["limit"] == 2
    assert blog_call["filters"] == {"topic_type": "vendor_alternative"}
    assert blog_call["channels"] is None
    assert blog_call["default_report_type"] is None
    assert blog_call["default_brief_type"] is None
    assert blog_call["extras"] == {}


@pytest.mark.asyncio
async def test_execute_runs_signal_extraction_service_from_source_material() -> None:
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["signal_extraction"],
            "inputs": {
                "source_max_text_chars": 7,
                "source_material": [
                    {
                        "id": "review-1",
                        "company": "Acme",
                        "vendor": "HubSpot",
                        "review_text": "Pricing is a problem.",
                        "contact_email": "buyer@example.com",
                    }
                ],
            },
        },
        services=ContentOpsExecutionServices(
            signal_extraction=SignalExtractionService()
        ),
    )

    assert result["status"] == "completed"
    step = result["steps"][0]
    assert step["output"] == "signal_extraction"
    assert step["result"]["generated"] == 1
    opportunity = step["result"]["opportunities"][0]
    assert opportunity["target_id"] == "review-1"
    assert opportunity["evidence"][0]["text"] == "Pricing"
    assert step["result"]["warnings"] == []


@pytest.mark.asyncio
async def test_execute_runs_social_post_service_from_source_material() -> None:
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["social_post"],
            "limit": 1,
            "inputs": {
                "source_max_text_chars": 20,
                "source_material": [
                    {
                        "review_id": "review-1",
                        "company": "Acme",
                        "vendor": "HubSpot",
                        "review_text": "Pricing is a problem after renewal.",
                        "pain_category": "pricing pressure",
                    }
                ],
            },
        },
        services=ContentOpsExecutionServices(
            social_post=SocialPostGenerationService()
        ),
    )

    assert result["status"] == "completed"
    step = result["steps"][0]
    assert step["output"] == "social_post"
    assert step["runner"] == "SocialPostGenerationService.generate"
    assert step["result"]["generated"] == 1
    post = step["result"]["posts"][0]
    assert post["source_id"] == "review-1"
    assert post["vendor_name"] == "HubSpot"
    assert "Pricing is a problem" in post["text"]
    assert result["plan"]["steps"][0]["config"] == {
        "skill_name": "digest/social_post_generation",
        "channels": ["linkedin"],
        "limit": 1,
        "max_text_chars": 20,
        "max_tokens": 700,
        "temperature": 0.4,
        "parse_retry_attempts": 1,
        "parse_retry_response_excerpt_chars": 800,
    }


@pytest.mark.asyncio
async def test_execute_threads_social_post_channels_from_request() -> None:
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["social_post"],
            "limit": 1,
            "inputs": {
                "social_channels": ["linkedin", "twitter"],
                "source_material": [
                    {
                        "review_id": "review-1",
                        "company": "Acme",
                        "vendor": "HubSpot",
                        "review_text": "Pricing is a problem after renewal.",
                    }
                ],
            },
        },
        services=ContentOpsExecutionServices(
            social_post=SocialPostGenerationService()
        ),
    )

    assert result["status"] == "completed"
    step = result["steps"][0]
    assert step["result"]["generated"] == 2
    assert [post["channel"] for post in step["result"]["posts"]] == ["linkedin", "x"]
    assert result["plan"]["steps"][0]["config"]["channels"] == ["linkedin", "x"]


@pytest.mark.asyncio
async def test_execute_social_post_brand_voice_fails_when_service_lacks_llm() -> None:
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["social_post"],
            "brand_voice_profile_id": "voice-1",
            "inputs": {
                "brand_voice": {
                    "id": "voice-1",
                    "account_id": "acct-1",
                    "descriptors": ["plainspoken"],
                },
                "source_material": [
                    {
                        "review_id": "review-1",
                        "vendor": "HubSpot",
                        "review_text": "Pricing is a problem after renewal.",
                    }
                ],
            },
        },
        services=ContentOpsExecutionServices(
            social_post=SocialPostGenerationService()
        ),
        scope=TenantScope(account_id="acct-1"),
    )

    assert result["status"] == "failed"
    step = result["steps"][0]
    assert step["output"] == "social_post"
    assert step["status"] == "failed"
    assert "requires configured LLM and skill store" in step["error"]


@pytest.mark.asyncio
async def test_execute_runs_ad_copy_service_from_source_material() -> None:
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["ad_copy"],
            "limit": 1,
            "inputs": {
                "source_max_text_chars": 20,
                "source_material": [
                    {
                        "review_id": "review-1",
                        "company": "Acme",
                        "vendor": "HubSpot",
                        "review_text": "Pricing is a problem after renewal.",
                        "pain_category": "pricing pressure",
                    }
                ],
            },
        },
        services=ContentOpsExecutionServices(
            ad_copy=AdCopyGenerationService()
        ),
    )

    assert result["status"] == "completed"
    step = result["steps"][0]
    assert step["output"] == "ad_copy"
    assert step["runner"] == "AdCopyGenerationService.generate"
    assert step["result"]["generated"] == 1
    ad = step["result"]["ads"][0]
    assert ad["source_id"] == "review-1"
    assert ad["vendor_name"] == "HubSpot"
    assert "Pricing is a problem" in ad["primary_text"]
    assert result["plan"]["steps"][0]["config"] == {
        "limit": 1,
        "max_text_chars": 20,
    }


@pytest.mark.asyncio
async def test_execute_runs_quote_card_service_from_source_material() -> None:
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["quote_card"],
            "limit": 1,
            "inputs": {
                "source_max_text_chars": 20,
                "source_material": [
                    {
                        "review_id": "review-1",
                        "company": "Acme",
                        "vendor": "HubSpot",
                        "review_text": "Pricing is a problem after renewal.",
                        "pain_category": "pricing pressure",
                    }
                ],
            },
        },
        services=ContentOpsExecutionServices(
            quote_card=QuoteCardGenerationService()
        ),
    )

    assert result["status"] == "completed"
    step = result["steps"][0]
    assert step["output"] == "quote_card"
    assert step["runner"] == "QuoteCardGenerationService.generate"
    assert step["result"]["generated"] == 1
    card = step["result"]["cards"][0]
    assert card["source_id"] == "review-1"
    assert card["vendor_name"] == "HubSpot"
    assert "Pricing is a problem" in card["quote"]
    assert result["plan"]["steps"][0]["config"] == {
        "limit": 1,
        "max_text_chars": 20,
    }


@pytest.mark.asyncio
async def test_execute_runs_stat_card_service_from_source_material() -> None:
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["stat_card"],
            "limit": 1,
            "inputs": {
                "source_max_text_chars": 40,
                "source_material": [
                    {
                        "review_id": "review-1",
                        "company": "Acme",
                        "vendor": "HubSpot",
                        "review_text": "NPS score is 42 after renewal.",
                        "nps_score": 42,
                        "pain_category": "renewal risk",
                    }
                ],
            },
        },
        services=ContentOpsExecutionServices(
            stat_card=StatCardGenerationService()
        ),
    )

    assert result["status"] == "completed"
    step = result["steps"][0]
    assert step["output"] == "stat_card"
    assert step["runner"] == "StatCardGenerationService.generate"
    assert step["result"]["generated"] == 1
    stat = step["result"]["stats"][0]
    assert stat["source_id"] == "review-1"
    assert stat["vendor_name"] == "HubSpot"
    assert stat["claim"] == "NPS score: 42"
    assert result["plan"]["steps"][0]["config"] == {
        "limit": 1,
        "max_text_chars": 40,
    }


@pytest.mark.asyncio
async def test_execute_runs_faq_markdown_service_from_source_material() -> None:
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["faq_markdown"],
            "limit": 2,
            "inputs": {
                "faq_title": "Support FAQ",
                "source_material": [
                    {
                        "ticket_id": "ticket-1",
                        "source_type": "ticket",
                        "created_at": "2026-05-01",
                        "subject": "SSO setup",
                        "message": "How do I enable SSO for my team?",
                        "pain_category": "login",
                    },
                    {
                        "ticket_id": "ticket-old",
                        "source_type": "ticket",
                        "created_at": "2026-01-01",
                        "subject": "Billing export",
                        "message": "Billing export is confusing.",
                        "pain_category": "billing",
                    }
                ],
                "faq_window_days": 90,
                "faq_as_of_date": "2026-05-20",
                "faq_support_contact": "1-800-555-0100",
                "faq_documentation_terms": ["Single sign-on setup"],
                "faq_vocabulary_gap_rules": [["SSO", "single sign-on"]],
            },
        },
        services=ContentOpsExecutionServices(faq_markdown=TicketFAQMarkdownService()),
    )

    assert result["status"] == "completed"
    step = result["steps"][0]
    assert step["output"] == "faq_markdown"
    assert step["result"]["generated"] == 1
    assert step["result"]["markdown"].startswith("# Support FAQ")
    assert "How do I enable SSO for my team?" in step["result"]["markdown"]
    assert "contact support at 1-800-555-0100" in step["result"]["markdown"]
    assert result["plan"]["steps"][0]["config"]["documentation_terms"] == [
        "Single sign-on setup"
    ]
    assert result["plan"]["steps"][0]["config"]["vocabulary_gap_rules"] == [
        ["SSO", "single sign-on"]
    ]
    assert step["result"]["items"][0]["term_mappings"][0]["customer_term"] == "SSO"
    assert (
        step["result"]["items"][0]["term_mappings"][0]["documentation_term"]
        == "Single sign-on setup"
    )
    assert "Billing export is confusing." not in step["result"]["markdown"]


@pytest.mark.asyncio
async def test_execute_applies_hosted_faq_intent_rules() -> None:
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["faq_markdown"],
            "limit": 2,
            "inputs": {
                "faq_intent_rules": [
                    {"topic": "data freshness", "keywords": ["warehouse sync"]}
                ],
                "source_material": [
                    {
                        "ticket_id": "ticket-1",
                        "source_type": "ticket",
                        "subject": "Warehouse sync lag",
                        "message": "The warehouse sync is delayed again.",
                    },
                    {
                        "ticket_id": "ticket-2",
                        "source_type": "ticket",
                        "subject": "Warehouse sync stale",
                        "message": "Warehouse sync data is stale this morning.",
                    },
                ],
            },
        },
        services=ContentOpsExecutionServices(faq_markdown=TicketFAQMarkdownService()),
    )

    assert result["status"] == "completed"
    assert result["plan"]["steps"][0]["config"]["intent_rules"][0] == {
        "topic": "data freshness",
        "keywords": ["warehouse sync"],
    }
    item = result["steps"][0]["result"]["items"][0]
    assert item["topic"] == "data freshness"
    assert item["ticket_count"] == 2


@pytest.mark.asyncio
async def test_execute_applies_hosted_faq_intent_rules_to_deflection_report_without_inventing_answers() -> None:
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["faq_deflection_report"],
            "limit": 2,
            "inputs": {
                "faq_intent_rules": [
                    "data freshness=warehouse sync,connector lag"
                ],
                "source_material": [
                    {
                        "ticket_id": "ticket-1",
                        "source_type": "support_ticket",
                        "subject": "Warehouse sync lag",
                        "message": "The warehouse sync is delayed again.",
                    },
                    {
                        "ticket_id": "ticket-2",
                        "source_type": "support_ticket",
                        "subject": "Connector lag",
                        "message": "CRM connector lag repeats every morning.",
                    },
                ],
            },
        },
        services=ContentOpsExecutionServices(
            faq_deflection_report=FAQDeflectionReportService()
        ),
    )

    assert result["status"] == "completed"
    assert result["plan"]["steps"][0]["config"]["intent_rules"][0] == {
        "topic": "data freshness",
        "keywords": ["warehouse sync", "connector lag"],
    }

    step = result["steps"][0]
    assert step["output"] == "faq_deflection_report"
    assert step["status"] == "completed"
    assert step["result"]["summary"]["drafted_answer_count"] == 0
    assert step["result"]["summary"]["no_proven_answer_count"] == 1

    item = step["result"]["faq_result"]["items"][0]
    assert item["topic"] == "data freshness"
    assert item["ticket_count"] == 2
    assert item["answer_evidence_status"] == "draft_needs_review"
    assert "## Support Tax Confirmation" in step["result"]["markdown"]
    assert "## Publishable Help-Center Copy From Proven Resolutions" in step["result"]["markdown"]
    assert "## No Proven Answer Yet" in step["result"]["markdown"]
    assert "No verified support resolution was present" in step["result"]["markdown"]


@pytest.mark.asyncio
async def test_execute_runs_faq_deflection_report_from_source_material() -> None:
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["faq_deflection_report"],
            "limit": 3,
            "inputs": {
                "deflection_report_title": "Acme Support Deflection Report",
                "faq_title": "Acme Support FAQ",
                "source_material": [
                    {
                        "ticket_id": "ticket-1",
                        "source_type": "support_ticket",
                        "subject": "Export attribution",
                        "message": "How do I export attribution reports?",
                        "pain_category": "exports",
                        "resolution_text": (
                            "Open Analytics, choose Attribution, then select "
                            "Download report."
                        ),
                    },
                    {
                        "ticket_id": "ticket-2",
                        "source_type": "support_ticket",
                        "subject": "Renewal invoice",
                        "message": "How do I confirm my renewal invoice before payment?",
                        "pain_category": "billing",
                    },
                ],
            },
        },
        services=ContentOpsExecutionServices(
            faq_deflection_report=FAQDeflectionReportService()
        ),
    )

    assert result["status"] == "completed"
    step = result["steps"][0]
    assert step["output"] == "faq_deflection_report"
    assert step["runner"] == "FAQDeflectionReportService.generate"
    assert step["status"] == "completed"
    assert step["result"]["markdown"].startswith("# Acme Support Deflection Report")
    assert "## Support Tax Confirmation" in step["result"]["markdown"]
    assert "## Ranked Question Opportunities" in step["result"]["markdown"]
    assert "## Publishable Help-Center Copy From Proven Resolutions" in step["result"]["markdown"]
    assert "Open Analytics, choose Attribution" in step["result"]["markdown"]
    assert "## No Proven Answer Yet" in step["result"]["markdown"]
    assert step["result"]["summary"]["source_count"] == 2
    assert step["result"]["summary"]["drafted_answer_count"] == 1
    assert step["result"]["summary"]["no_proven_answer_count"] == 1
    assert step["result"]["faq_result"]["markdown"].startswith("# Acme Support FAQ")
    assert result["plan"]["steps"][0]["config"]["report_title"] == (
        "Acme Support Deflection Report"
    )


@pytest.mark.asyncio
async def test_execute_reports_missing_service_as_failed_when_all_steps_fail() -> None:
    """PR-Audit-MINOR-Batch-1: when every step fails, status is now
    'failed' (was 'partial' pre-fix -- misled dashboards). Error dict
    shape now matches ContentOpsStepExecution (output / runner / error)
    with ``reason`` kept as a backwards-compat alias."""
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["sales_brief"],
            "inputs": {"target_account": "Acme"},
        },
        services=ContentOpsExecutionServices(),
    )

    assert result["status"] == "failed"
    assert result["steps"][0]["status"] == "failed"
    assert result["steps"][0]["error"] == "service_not_configured"
    assert result["errors"] == [
        {
            "output": "sales_brief",
            "runner": "SalesBriefGenerationService.generate",
            "error": "service_not_configured",
            "reason": "service_not_configured",
        }
    ]


@pytest.mark.asyncio
async def test_execute_reports_service_without_generate_as_not_configured() -> None:
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
            },
        },
        services=ContentOpsExecutionServices(campaign=object()),
    )

    assert result["status"] == "failed"  # all (1) steps failed
    assert result["steps"][0]["status"] == "failed"
    assert result["steps"][0]["error"] == "service_not_configured"
    assert result["errors"] == [
        {
            "output": "email_campaign",
            "runner": "CampaignGenerationService.generate",
            "error": "service_not_configured",
            "reason": "service_not_configured",
        }
    ]


# -----------------------
# PR-OptionA-1: smoking-gun fields (channels, default_report_type,
# default_brief_type) flow from plan into the service call.
# -----------------------


@pytest.mark.asyncio
async def test_execute_threads_user_selected_channels_into_email_campaign_service() -> None:
    """The audit's specific failure mode: user picks channels=['email_cold'] in the
    control surface, plan records it in step.config, and the executor MUST pass
    it to the service. Before PR-OptionA-1, the service used its construction-
    time default and the user's selection was silently dropped."""

    campaign = _OpportunityService()
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
                "channels": ["email_cold"],
            },
        },
        services=ContentOpsExecutionServices(campaign=campaign),
    )

    assert result["status"] == "completed"
    assert campaign.calls[0]["channels"] == ("email_cold",)
    assert campaign.calls[0]["extras"] == {}


@pytest.mark.asyncio
async def test_execute_omits_opportunity_defaults_when_selling_context_absent() -> None:
    campaign = _StrictCampaignService()
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
            },
        },
        services=ContentOpsExecutionServices(campaign=campaign),
    )

    assert result["status"] == "completed"
    assert campaign.calls == 1


@pytest.mark.asyncio
async def test_execute_threads_selling_context_into_email_campaign_service() -> None:
    campaign = _OpportunityService()
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
                "selling": {
                    "booking_url": "https://customer.test/book",
                    "sender_name": "Juan",
                },
            },
        },
        services=ContentOpsExecutionServices(campaign=campaign),
    )

    assert result["status"] == "completed"
    assert campaign.calls[0]["opportunity_defaults"] == {
        "selling": {
            "booking_url": "https://customer.test/book",
            "sender_name": "Juan",
        }
    }
    assert campaign.calls[0]["extras"] == {}


@pytest.mark.asyncio
async def test_execute_promotes_flat_booking_url_into_selling_context() -> None:
    campaign = _OpportunityService()
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
                "booking_url": "https://customer.test/book",
            },
        },
        services=ContentOpsExecutionServices(campaign=campaign),
    )

    assert result["status"] == "completed"
    assert campaign.calls[0]["opportunity_defaults"] == {
        "selling": {"booking_url": "https://customer.test/book"}
    }
    assert campaign.calls[0]["extras"] == {}


@pytest.mark.asyncio
async def test_execute_falls_back_to_plan_default_channels_when_request_omits_them() -> None:
    """No request-level override -> plan emits the construction-time default
    channels and the executor still threads them through (so the service
    behavior is identical, but the plan is now the single source of truth)."""

    campaign = _OpportunityService()
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
            },
        },
        services=ContentOpsExecutionServices(campaign=campaign),
    )

    assert result["status"] == "completed"
    assert campaign.calls[0]["channels"] == ("email_cold", "email_followup")
    assert campaign.calls[0]["extras"] == {}


@pytest.mark.asyncio
async def test_execute_threads_user_selected_report_type_into_report_service() -> None:
    report = _OpportunityService()
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["report"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
                "opportunity_id": "opp-1",
                "report_type": "customer_health",
            },
        },
        services=ContentOpsExecutionServices(report=report),
    )

    assert result["status"] == "completed"
    assert report.calls[0]["default_report_type"] == "customer_health"
    assert report.calls[0]["extras"] == {}


@pytest.mark.asyncio
async def test_execute_threads_user_selected_brief_type_into_sales_brief_service() -> None:
    sales_brief = _OpportunityService()
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["sales_brief"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
                "opportunity_id": "opp-1",
                "brief_type": "renewal",
            },
        },
        services=ContentOpsExecutionServices(sales_brief=sales_brief),
    )

    assert result["status"] == "completed"
    assert sales_brief.calls[0]["default_brief_type"] == "renewal"
    assert sales_brief.calls[0]["extras"] == {}


@pytest.mark.asyncio
async def test_execute_threads_source_material_into_sales_brief_service() -> None:
    source_material = [{
        "target_id": "competitive-1",
        "source_id": "competitive-1",
        "source_type": "competitive",
        "vendor_name": "Slack",
        "competitor": "Teams",
        "text": "Slack buyers cite Teams as the replacement alternative.",
    }]
    sales_brief = _OpportunityService()

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["sales_brief"],
            "inputs": {
                "target_account": "Slack",
                "offer": "Competitive displacement audit",
                "opportunity_id": "competitive-1",
                "brief_type": "displacement",
                "source_material": source_material,
            },
        },
        services=ContentOpsExecutionServices(sales_brief=sales_brief),
    )

    assert result["status"] == "completed"
    call = sales_brief.calls[0]
    assert call["default_brief_type"] == "displacement"
    assert call["source_material"] == source_material
    assert call["extras"] == {}


@pytest.mark.asyncio
async def test_execute_landing_page_dispatcher_unchanged_no_per_call_overrides() -> None:
    """Landing page still gets the campaign-from-inputs dispatch (no per-call
    overrides in this PR; deferred to PR-OptionA-2)."""

    landing = _LandingPageService()
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {
                "campaign_name": "Q3 audit",
                "offer": "Audit",
                "audience": "VPs",
            },
        },
        services=ContentOpsExecutionServices(landing_page=landing),
    )

    assert result["status"] == "completed"
    assert len(landing.calls) == 1
    assert landing.calls[0]["campaign"].name == "Q3 audit"


# -----------------------
# PR-OptionA-2: LLM-tuning kwargs (temperature / max_tokens /
# parse_retry_attempts) flow from plan defaults into every service call.
# -----------------------


@pytest.mark.asyncio
async def test_execute_threads_llm_tuning_kwargs_into_email_campaign() -> None:
    """Plan-default temperature/max_tokens/parse_retry_attempts (from the
    CampaignGenerationConfig defaults the plan layer reads) reach the
    service via dispatch."""

    campaign = _OpportunityService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {"target_account": "Acme", "offer": "Audit"},
        },
        services=ContentOpsExecutionServices(campaign=campaign),
    )

    call = campaign.calls[0]
    # Plan emits the construction-time config defaults; they are now
    # threaded as named kwargs rather than relying on the service's
    # construction-time config alone.
    assert call["temperature"] == 0.4  # CampaignGenerationConfig default
    assert call["max_tokens"] == 1200
    assert call["parse_retry_attempts"] == 1
    assert call["extras"] == {}


@pytest.mark.asyncio
async def test_execute_threads_llm_tuning_kwargs_into_report() -> None:
    report = _OpportunityService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["report"],
            "inputs": {"target_account": "Acme", "offer": "Audit", "opportunity_id": "opp-1"},
        },
        services=ContentOpsExecutionServices(report=report),
    )

    call = report.calls[0]
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 4096
    assert call["parse_retry_attempts"] == 1
    assert call["extras"] == {}


@pytest.mark.asyncio
async def test_execute_threads_llm_tuning_kwargs_into_sales_brief() -> None:
    sales_brief = _OpportunityService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["sales_brief"],
            "inputs": {"target_account": "Acme", "offer": "Audit", "opportunity_id": "opp-1"},
        },
        services=ContentOpsExecutionServices(sales_brief=sales_brief),
    )

    call = sales_brief.calls[0]
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 4096
    assert call["parse_retry_attempts"] == 1
    assert call["extras"] == {}


@pytest.mark.asyncio
async def test_execute_threads_llm_tuning_kwargs_into_landing_page() -> None:
    landing = _LandingPageService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {"campaign_name": "Q3", "offer": "Audit", "audience": "VPs"},
        },
        services=ContentOpsExecutionServices(landing_page=landing),
    )

    call = landing.calls[0]
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 4096
    assert call["parse_retry_attempts"] == 1
    assert call["extras"] == {}


@pytest.mark.asyncio
async def test_execute_threads_llm_tuning_kwargs_into_blog_post() -> None:
    """Blog_post graduates to its own dispatcher in PR-OptionA-2 so the
    LLM-tuning kwargs reach the service. Previously it was on the default
    dispatcher and the kwargs were silently dropped."""

    blog = _OpportunityService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["blog_post"],
            "inputs": {"target_account": "Acme", "topic": "Churn"},
        },
        services=ContentOpsExecutionServices(blog_post=blog),
    )

    call = blog.calls[0]
    assert call["temperature"] == 0.3  # BlogPostGenerationConfig default
    assert call["max_tokens"] == 4096
    assert call["parse_retry_attempts"] == 1
    assert call["extras"] == {}


# -----------------------
# PR-OptionA-3: quality + retry-excerpt knobs
# (quality_revalidation_enabled / quality_prompt_proof_term_limit /
# parse_retry_response_excerpt_chars) flow from plan defaults into
# every service call.
# -----------------------


@pytest.mark.asyncio
async def test_execute_threads_quality_and_excerpt_kwargs_into_email_campaign() -> None:
    """Plan defaults for the campaign-only quality knobs reach the service.
    The bool flag, the int term-limit, and the int excerpt-chars all flow
    through the email_campaign dispatcher."""

    campaign = _OpportunityService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {"target_account": "Acme", "offer": "Audit"},
        },
        services=ContentOpsExecutionServices(campaign=campaign),
    )

    call = campaign.calls[0]
    # CampaignGenerationConfig defaults
    assert call["quality_revalidation_enabled"] is True
    assert call["quality_prompt_proof_term_limit"] == 5
    assert call["parse_retry_response_excerpt_chars"] == 800
    assert call["extras"] == {}


@pytest.mark.asyncio
async def test_execute_threads_excerpt_kwarg_into_report_sales_brief_blog_landing() -> None:
    """parse_retry_response_excerpt_chars reaches every service via its
    dispatcher; quality_revalidation_enabled / proof_term_limit are
    campaign-only and stay None for the others."""

    report = _OpportunityService()
    sales_brief = _OpportunityService()
    blog = _OpportunityService()
    landing = _LandingPageService()

    await execute_content_ops_from_mapping(
        {
            "outputs": ["report"],
            "inputs": {"target_account": "Acme", "offer": "Audit", "opportunity_id": "opp-1"},
        },
        services=ContentOpsExecutionServices(report=report),
    )
    await execute_content_ops_from_mapping(
        {
            "outputs": ["sales_brief"],
            "inputs": {"target_account": "Acme", "offer": "Audit", "opportunity_id": "opp-1"},
        },
        services=ContentOpsExecutionServices(sales_brief=sales_brief),
    )
    await execute_content_ops_from_mapping(
        {
            "outputs": ["blog_post"],
            "inputs": {"target_account": "Acme", "topic": "Churn"},
        },
        services=ContentOpsExecutionServices(blog_post=blog),
    )
    await execute_content_ops_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {"campaign_name": "Q3", "offer": "Audit", "audience": "VPs"},
        },
        services=ContentOpsExecutionServices(landing_page=landing),
    )

    for label, call in (
        ("report", report.calls[0]),
        ("sales_brief", sales_brief.calls[0]),
        ("blog_post", blog.calls[0]),
        ("landing_page", landing.calls[0]),
    ):
        assert call["parse_retry_response_excerpt_chars"] == 800, label
        assert call["extras"] == {}, label
        # Campaign-only kwargs are absent on the non-campaign services.
        if label != "landing_page":
            assert call.get("quality_revalidation_enabled") is None, label
            assert call.get("quality_prompt_proof_term_limit") is None, label


# -----------------------
# PR-OptionA-4: quality_gates_enabled + MarketingCampaign.context leak fix
# -----------------------


@pytest.mark.asyncio
async def test_execute_threads_quality_gates_enabled_into_sales_brief() -> None:
    """When the plan emits quality_gates_enabled in step.config, the executor
    threads it through. Default is True; this test only confirms threading."""

    sales_brief = _OpportunityService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["sales_brief"],
            "inputs": {"target_account": "Acme", "offer": "Audit", "opportunity_id": "opp-1"},
        },
        services=ContentOpsExecutionServices(sales_brief=sales_brief),
    )

    call = sales_brief.calls[0]
    # Plan default emits quality_gates_enabled=True (from request.require_quality_gates).
    assert call["quality_gates_enabled"] is True
    assert call["extras"] == {}


@pytest.mark.asyncio
async def test_execute_threads_quality_gates_enabled_into_landing_page() -> None:
    landing = _LandingPageService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {"campaign_name": "Q3", "offer": "Audit", "audience": "VPs"},
        },
        services=ContentOpsExecutionServices(landing_page=landing),
    )

    call = landing.calls[0]
    assert call["quality_gates_enabled"] is True
    assert call["extras"] == {}


@pytest.mark.asyncio
async def test_execute_threads_quality_repair_attempts_into_landing_page() -> None:
    landing = _LandingPageService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {"campaign_name": "Q3", "offer": "Audit", "audience": "VPs"},
        },
        services=ContentOpsExecutionServices(landing_page=landing),
    )

    call = landing.calls[0]
    assert call["quality_repair_attempts"] == 1
    assert call["extras"] == {}


@pytest.mark.asyncio
async def test_execute_threads_quality_repair_attempt_override_into_landing_page() -> None:
    landing = _LandingPageService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {
                "campaign_name": "Q3",
                "offer": "Audit",
                "audience": "VPs",
                "landing_page_quality_repair_attempts": 0,
            },
        },
        services=ContentOpsExecutionServices(landing_page=landing),
    )

    call = landing.calls[0]
    assert call["quality_repair_attempts"] == 0
    assert call["extras"] == {}


@pytest.mark.asyncio
async def test_marketing_campaign_context_does_not_leak_unrelated_inputs() -> None:
    """Audit MAJOR fix: prior shape dumped every non-{name, persona,
    value_prop, vendors, categories, tags} input field into
    campaign.context. Now context is an explicit allowlist; standard
    control-surface inputs (target_account, opportunity_id, channels,
    filters, etc.) stay out."""

    landing = _LandingPageService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {
                "campaign_name": "Q3",
                "offer": "Audit",
                "audience": "VPs",
                "target_account": "Acme",  # was leaking pre-fix
                "opportunity_id": "opp-1",  # was leaking pre-fix
                "filters": {"status": "ready"},  # was leaking pre-fix
                "channels": ["email_cold"],  # was leaking pre-fix
                # Allowlisted fields DO flow through:
                "industry": "fintech",
                "pain_points": ["churn", "renewal"],
                "target_keyword": "customer support FAQ",
                "secondary_keywords": [
                    "support ticket FAQ",
                    "reduce repeat support tickets",
                ],
                "faq_questions": [
                    "What do you need from us?",
                    "What happens after upload?",
                ],
                "cta_url": "/systems/ai-content-ops/intake",
                "source_row_count": 4,
                "included_ticket_row_count": 4,
                "skipped_ticket_row_count": 0,
                "truncated_ticket_row_count": 0,
                "question_like_ticket_count": 2,
                "has_dated_window": True,
                "top_ticket_clusters": [{"label": "reporting friction", "count": 2}],
                "customer_wording_examples": [{
                    "source_id": "ticket-1",
                    "source_title": "Export dashboard",
                    "pain_category": "reporting friction",
                    "text": "How do we export campaign attribution data before renewal?",
                }],
                "support_ticket_resolution_evidence_present": True,
                "support_ticket_resolution_evidence_count": 1,
                "support_ticket_resolution_examples": [{
                    "source_id": "ticket-1",
                    "source_title": "Export dashboard",
                    "text": "Open Reports, choose Export, then select CSV.",
                }],
                "has_measured_outcomes": True,
                "measured_outcome_count": 1,
                "measured_outcome_examples": [{
                    "source_id": "ticket-1",
                    "source_title": "Export dashboard",
                    "text": "Repeat reporting tickets fell from 9 to 4.",
                }],
                "support_ticket_source_summary": {"unsafe": "should not pass through"},
            },
        },
        services=ContentOpsExecutionServices(landing_page=landing),
    )

    campaign = landing.calls[0]["campaign"]
    context = dict(campaign.context)

    # Allowlisted fields land in context.
    assert context.get("industry") == "fintech"
    assert context.get("pain_points") == ["churn", "renewal"]
    assert context.get("target_keyword") == "customer support FAQ"
    assert context.get("secondary_keywords") == [
        "support ticket FAQ",
        "reduce repeat support tickets",
    ]
    assert context.get("faq_questions") == [
        "What do you need from us?",
        "What happens after upload?",
    ]
    assert context.get("cta_url") == "/systems/ai-content-ops/intake"
    assert context.get("source_row_count") == 4
    assert context.get("included_ticket_row_count") == 4
    assert context.get("skipped_ticket_row_count") == 0
    assert context.get("truncated_ticket_row_count") == 0
    assert context.get("question_like_ticket_count") == 2
    assert context.get("has_dated_window") is True
    assert context.get("top_ticket_clusters") == [
        {"label": "reporting friction", "count": 2}
    ]
    assert context.get("customer_wording_examples") == [{
        "source_id": "ticket-1",
        "source_title": "Export dashboard",
        "pain_category": "reporting friction",
        "text": "How do we export campaign attribution data before renewal?",
    }]
    assert context.get("support_ticket_resolution_evidence_present") is True
    assert context.get("support_ticket_resolution_evidence_count") == 1
    assert context.get("support_ticket_resolution_examples") == [{
        "source_id": "ticket-1",
        "source_title": "Export dashboard",
        "text": "Open Reports, choose Export, then select CSV.",
    }]
    assert context.get("has_measured_outcomes") is True
    assert context.get("measured_outcome_count") == 1
    assert context.get("measured_outcome_examples") == [{
        "source_id": "ticket-1",
        "source_title": "Export dashboard",
        "text": "Repeat reporting tickets fell from 9 to 4.",
    }]
    assert "support_ticket_source_summary" not in context

    # Pre-fix leakers stay out.
    assert "target_account" not in context
    assert "opportunity_id" not in context
    assert "filters" not in context
    assert "channels" not in context


# -----------------------
# PR-OptionA-5: quality_gates_enabled symmetry for report + blog_post
# -----------------------


@pytest.mark.asyncio
async def test_execute_threads_quality_gates_enabled_into_report() -> None:
    """Plan now emits quality_gates_enabled for report; executor threads it."""

    report = _OpportunityService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["report"],
            "inputs": {"target_account": "Acme", "offer": "Audit", "opportunity_id": "opp-1"},
        },
        services=ContentOpsExecutionServices(report=report),
    )

    call = report.calls[0]
    assert call["quality_gates_enabled"] is True
    assert call["extras"] == {}


@pytest.mark.asyncio
async def test_execute_threads_quality_gates_enabled_into_blog_post() -> None:
    blog = _OpportunityService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["blog_post"],
            "inputs": {"target_account": "Acme", "topic": "Churn"},
        },
        services=ContentOpsExecutionServices(blog_post=blog),
    )

    call = blog.calls[0]
    assert call["quality_gates_enabled"] is True
    assert call["extras"] == {}


# -----------------------
# PR-Audit-MINOR-Batch-1: status distinguishes partial from failed
# -----------------------


@pytest.mark.asyncio
async def test_execute_reports_partial_when_only_some_steps_fail() -> None:
    """When some steps succeed and others fail, status is 'partial'.
    Reserved for the genuine mixed case -- not 'failed' (which is now
    all-failed) and not 'completed' (no errors)."""

    campaign = _OpportunityService()
    # report has no service -> that step fails, but campaign succeeds
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign", "report"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Audit",
                "opportunity_id": "opp-1",
            },
        },
        services=ContentOpsExecutionServices(campaign=campaign),
    )

    assert result["status"] == "partial"
    assert {step["output"]: step["status"] for step in result["steps"]} == {
        "email_campaign": "completed",
        "report": "failed",
    }
    # Error payload uses the new aligned shape.
    assert len(result["errors"]) == 1
    err = result["errors"][0]
    assert err["output"] == "report"
    assert err["runner"] == "ReportGenerationService.generate"
    assert err["error"] == "service_not_configured"
    assert err["reason"] == "service_not_configured"  # backwards-compat alias


# -----------------------
# PR-Blog-Topic-Per-Call: dispatcher threads topic from step.config
# -----------------------


@pytest.mark.asyncio
async def test_execute_threads_topic_into_blog_post_dispatcher() -> None:
    """The plan emits ``step.config["topic"]`` from
    ``request.inputs.get("topic")``; the executor's blog dispatcher
    threads it through to ``BlogPostGenerationService.generate``."""

    blog = _OpportunityService()
    await execute_content_ops_from_mapping(
        {
            "outputs": ["blog_post"],
            "inputs": {
                "target_account": "Acme",
                "topic": "Renewal pricing pressure",
            },
        },
        services=ContentOpsExecutionServices(blog_post=blog),
    )

    call = blog.calls[0]
    assert call["topic"] == "Renewal pricing pressure"
    assert call["quality_repair_attempts"] == 2
    assert call["extras"] == {}


@pytest.mark.asyncio
async def test_execute_fans_out_blog_variants_by_angle() -> None:
    blog = _VariantBlogService()

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["blog_post"],
            "variant_count": 3,
            "inputs": {
                "target_account": "Acme",
                "topic": "Renewal pricing pressure",
            },
        },
        services=ContentOpsExecutionServices(blog_post=blog),
    )

    step_result = result["steps"][0]["result"]
    assert result["status"] == "completed"
    assert len(blog.calls) == 3
    assert [call["variant_angle"] for call in blog.calls] == [
        VARIANT_ANGLES[0].instruction,
        VARIANT_ANGLES[1].instruction,
        VARIANT_ANGLES[2].instruction,
    ]
    assert step_result["variant_count"] == 3
    assert step_result["requested"] == 3
    assert step_result["generated"] == 3
    assert step_result["skipped"] == 0
    assert step_result["saved_ids"] == ["draft-1", "draft-2", "draft-3"]
    assert [
        item["variant_angle"]["id"] for item in step_result["variant_results"]
    ] == ["pain_led", "outcome_led", "social_proof"]


@pytest.mark.asyncio
async def test_execute_blog_variant_failure_does_not_abort_batch() -> None:
    blog = _VariantBlogService(fail_on_angle="Outcome-led")

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["blog_post"],
            "variant_count": 3,
            "inputs": {
                "target_account": "Acme",
                "topic": "Renewal pricing pressure",
            },
        },
        services=ContentOpsExecutionServices(blog_post=blog),
    )

    step_result = result["steps"][0]["result"]
    assert result["status"] == "completed"
    assert result["errors"] == []
    assert len(blog.calls) == 3
    assert step_result["generated"] == 2
    assert step_result["skipped"] == 1
    assert step_result["saved_ids"] == ["draft-1", "draft-3"]
    assert step_result["errors"] == [{
        "variant_angle": "outcome_led",
        "variant_label": "Outcome-led",
        "reason": "variant failed",
        "error_type": "RuntimeError",
    }]


@pytest.mark.asyncio
async def test_execute_all_raising_blog_variants_fails_step_with_warning() -> None:
    blog = _VariantBlogService(fail_on_angle="led")

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["blog_post"],
            "variant_count": 3,
            "inputs": {
                "target_account": "Acme",
                "topic": "Renewal pricing pressure",
            },
        },
        services=ContentOpsExecutionServices(blog_post=blog),
    )

    step = result["steps"][0]
    assert result["status"] == "failed"
    assert result["errors"] == [{
        "output": "blog_post",
        "runner": "BlogPostGenerationService.generate",
        "error": "all_blog_variants_failed",
        "reason": "all_blog_variants_failed",
    }]
    assert step["status"] == "failed"
    assert step["error"] == "all_blog_variants_failed"
    assert step["result"]["generated"] == 0
    assert step["result"]["skipped"] == 3
    assert step["result"]["warnings"] == [
        "No blog variants generated; all requested variants were blocked or skipped."
    ]
    assert [
        item["variant_angle"]["id"] for item in step["result"]["variant_results"]
    ] == ["pain_led", "outcome_led", "social_proof"]
    assert [item["variant_angle"] for item in step["result"]["errors"]] == [
        "pain_led",
        "outcome_led",
        "social_proof",
    ]


@pytest.mark.asyncio
async def test_execute_fans_out_landing_page_variants_by_angle() -> None:
    landing = _VariantLandingPageService()

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["landing_page"],
            "variant_count": 3,
            "inputs": {
                "campaign_name": "Q3 retention page",
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
            },
        },
        services=ContentOpsExecutionServices(landing_page=landing),
    )

    step_result = result["steps"][0]["result"]
    assert result["status"] == "completed"
    assert len(landing.calls) == 3
    assert [call["variant_angle"] for call in landing.calls] == [
        VARIANT_ANGLES[0].instruction,
        VARIANT_ANGLES[1].instruction,
        VARIANT_ANGLES[2].instruction,
    ]
    assert [call["campaign"].name for call in landing.calls] == [
        "Q3 retention page",
        "Q3 retention page",
        "Q3 retention page",
    ]
    assert step_result["variant_count"] == 3
    assert step_result["requested"] == 3
    assert step_result["generated"] == 3
    assert step_result["skipped"] == 0
    assert step_result["saved_ids"] == [
        "landing-page-1",
        "landing-page-2",
        "landing-page-3",
    ]
    assert [
        item["variant_angle"]["id"] for item in step_result["variant_results"]
    ] == ["pain_led", "outcome_led", "social_proof"]


@pytest.mark.asyncio
async def test_execute_landing_page_variant_failure_does_not_abort_batch() -> None:
    landing = _VariantLandingPageService(fail_on_angle="Outcome-led")

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["landing_page"],
            "variant_count": 3,
            "inputs": {
                "campaign_name": "Q3 retention page",
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
            },
        },
        services=ContentOpsExecutionServices(landing_page=landing),
    )

    step_result = result["steps"][0]["result"]
    assert result["status"] == "completed"
    assert result["errors"] == []
    assert len(landing.calls) == 3
    assert step_result["generated"] == 2
    assert step_result["skipped"] == 1
    assert step_result["saved_ids"] == ["landing-page-1", "landing-page-3"]
    assert step_result["errors"] == [{
        "variant_angle": "outcome_led",
        "variant_label": "Outcome-led",
        "reason": "landing variant failed",
        "error_type": "RuntimeError",
    }]


@pytest.mark.asyncio
async def test_execute_all_raising_landing_page_variants_fails_step_with_warning() -> None:
    landing = _VariantLandingPageService(fail_on_angle="led")

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["landing_page"],
            "variant_count": 3,
            "inputs": {
                "campaign_name": "Q3 retention page",
                "offer": "Churn audit",
                "audience": "B2B SaaS founders",
            },
        },
        services=ContentOpsExecutionServices(landing_page=landing),
    )

    step = result["steps"][0]
    assert result["status"] == "failed"
    assert result["errors"] == [{
        "output": "landing_page",
        "runner": "LandingPageGenerationService.generate",
        "error": "all_landing_page_variants_failed",
        "reason": "all_landing_page_variants_failed",
    }]
    assert step["status"] == "failed"
    assert step["error"] == "all_landing_page_variants_failed"
    assert step["result"]["generated"] == 0
    assert step["result"]["skipped"] == 3
    assert step["result"]["warnings"] == [
        "No landing-page variants generated; all requested variants were blocked or skipped."
    ]
    assert [
        item["variant_angle"]["id"] for item in step["result"]["variant_results"]
    ] == ["pain_led", "outcome_led", "social_proof"]
    assert [item["variant_angle"] for item in step["result"]["errors"]] == [
        "pain_led",
        "outcome_led",
        "social_proof",
    ]


@pytest.mark.asyncio
async def test_execute_fans_out_sales_brief_variants_by_angle() -> None:
    sales_brief = _VariantSalesBriefService()

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["sales_brief"],
            "variant_count": 3,
            "inputs": {
                "target_account": "Acme",
                "brief_type": "renewal",
            },
        },
        services=ContentOpsExecutionServices(sales_brief=sales_brief),
    )

    step_result = result["steps"][0]["result"]
    assert result["status"] == "completed"
    assert len(sales_brief.calls) == 3
    assert [call["variant_angle"] for call in sales_brief.calls] == [
        VARIANT_ANGLES[0].instruction,
        VARIANT_ANGLES[1].instruction,
        VARIANT_ANGLES[2].instruction,
    ]
    assert [call["kwargs"]["default_brief_type"] for call in sales_brief.calls] == [
        "renewal",
        "renewal",
        "renewal",
    ]
    assert step_result["variant_count"] == 3
    assert step_result["requested"] == 3
    assert step_result["generated"] == 3
    assert step_result["skipped"] == 0
    assert step_result["reasoning_contexts_used"] == 3
    assert step_result["saved_ids"] == [
        "sales-brief-1",
        "sales-brief-2",
        "sales-brief-3",
    ]
    assert [
        item["variant_angle"]["id"] for item in step_result["variant_results"]
    ] == ["pain_led", "outcome_led", "social_proof"]
    assert [
        item["target_id"] for item in step_result["consumed_reasoning_contexts"]
    ] == ["sales-brief-1", "sales-brief-2", "sales-brief-3"]


@pytest.mark.asyncio
async def test_execute_sales_brief_variant_failure_does_not_abort_batch() -> None:
    sales_brief = _VariantSalesBriefService(fail_on_angle="Outcome-led")

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["sales_brief"],
            "variant_count": 3,
            "inputs": {
                "target_account": "Acme",
                "brief_type": "renewal",
            },
        },
        services=ContentOpsExecutionServices(sales_brief=sales_brief),
    )

    step_result = result["steps"][0]["result"]
    assert result["status"] == "completed"
    assert result["errors"] == []
    assert len(sales_brief.calls) == 3
    assert step_result["generated"] == 2
    assert step_result["skipped"] == 1
    assert step_result["saved_ids"] == ["sales-brief-1", "sales-brief-3"]
    assert step_result["errors"] == [{
        "variant_angle": "outcome_led",
        "variant_label": "Outcome-led",
        "reason": "sales brief variant failed",
        "error_type": "RuntimeError",
    }]


@pytest.mark.asyncio
async def test_execute_all_raising_sales_brief_variants_fails_step_with_warning() -> None:
    sales_brief = _VariantSalesBriefService(fail_on_angle="led")

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["sales_brief"],
            "variant_count": 3,
            "inputs": {
                "target_account": "Acme",
                "brief_type": "renewal",
            },
        },
        services=ContentOpsExecutionServices(sales_brief=sales_brief),
    )

    step = result["steps"][0]
    assert result["status"] == "failed"
    assert result["errors"] == [{
        "output": "sales_brief",
        "runner": "SalesBriefGenerationService.generate",
        "error": "all_sales_brief_variants_failed",
        "reason": "all_sales_brief_variants_failed",
    }]
    assert step["status"] == "failed"
    assert step["error"] == "all_sales_brief_variants_failed"
    assert step["result"]["generated"] == 0
    assert step["result"]["skipped"] == 3
    assert step["result"]["warnings"] == [
        "No sales-brief variants generated; all requested variants were blocked or skipped."
    ]
    assert [
        item["variant_angle"]["id"] for item in step["result"]["variant_results"]
    ] == ["pain_led", "outcome_led", "social_proof"]
    assert [item["variant_angle"] for item in step["result"]["errors"]] == [
        "pain_led",
        "outcome_led",
        "social_proof",
    ]


# -----------------------
# PR-ControlSurfaces-Reasoning-Provider: bundle-level helper
# -----------------------


@pytest.mark.asyncio
async def test_execute_step_reports_reasoning_audit_when_provider_attached():
    """Execution results expose a compact per-step reasoning audit without
    exposing the full prompt context payload."""

    provider = object()
    campaign = _ReasoningAwareOpportunityService()
    services = ContentOpsExecutionServices(campaign=campaign).with_reasoning_context(
        provider
    )

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {"target_account": "Acme", "offer": "Audit"},
        },
        services=services,
    )

    assert result["steps"][0]["reasoning"] == {
        "requirement": "optional_host_context",
        "service_supports_reasoning": True,
        "provider_configured": True,
        "contexts_used": 0,
    }


@pytest.mark.asyncio
async def test_execute_step_reports_reasoning_audit_when_provider_absent():
    campaign = _ReasoningAwareOpportunityService()

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {"target_account": "Acme", "offer": "Audit"},
        },
        services=ContentOpsExecutionServices(campaign=campaign),
    )

    assert result["steps"][0]["reasoning"] == {
        "requirement": "optional_host_context",
        "service_supports_reasoning": True,
        "provider_configured": False,
        "contexts_used": 0,
    }


@pytest.mark.asyncio
async def test_execute_step_reports_actual_reasoning_contexts_used():
    class _ReasoningUsageService(_ReasoningAwareOpportunityService):
        async def generate(self, **kwargs: Any) -> _Result:
            self.calls.append(dict(kwargs))
            return _Result(generated=2, reasoning_contexts_used=2)

    services = ContentOpsExecutionServices(
        campaign=_ReasoningUsageService(),
        reasoning_provider_configured=True,
    )

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {"target_account": "Acme", "offer": "Audit"},
        },
        services=services,
    )

    assert result["steps"][0]["reasoning"] == {
        "requirement": "optional_host_context",
        "service_supports_reasoning": True,
        "provider_configured": True,
        "contexts_used": 2,
    }


@pytest.mark.asyncio
async def test_execute_step_reports_consumed_reasoning_contexts_when_result_includes_them():
    consumed = {
        "wedge": "pricing_pressure",
        "confidence": 0.82,
        "proof_points": [{"label": "Renewal spike", "value": "34%"}],
    }

    class _ReasoningPayloadService(_ReasoningAwareOpportunityService):
        async def generate(self, **kwargs: Any) -> _Result:
            self.calls.append(dict(kwargs))
            return _Result(
                generated=1,
                reasoning_contexts_used=1,
                consumed_reasoning_contexts=(consumed,),
            )

    services = ContentOpsExecutionServices(
        campaign=_ReasoningPayloadService(),
        reasoning_provider_configured=True,
    )

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {"target_account": "Acme", "offer": "Audit"},
        },
        services=services,
    )

    assert result["steps"][0]["reasoning"] == {
        "requirement": "optional_host_context",
        "service_supports_reasoning": True,
        "provider_configured": True,
        "contexts_used": 1,
        "consumed_contexts": [consumed],
    }


@pytest.mark.asyncio
async def test_execute_step_reports_strict_validation_failures_from_result_errors():
    class _StrictBlockedService(_ReasoningAwareOpportunityService):
        async def generate(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(dict(kwargs))
            return {
                "generated": 0,
                "skipped": 1,
                "reasoning_contexts_used": 0,
                "errors": [
                    {
                        "target_id": "vendor-acme",
                        "reason": (
                            "reasoning_validation_blocked:"
                            "claim_missing_citations:0,no_claims"
                        ),
                    }
                ],
            }

    services = ContentOpsExecutionServices(
        report=_StrictBlockedService(),
        reasoning_provider_configured=True,
        reasoning_provider_outputs=("report",),
    )

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["report"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Audit",
                "opportunity_id": "opp-1",
            },
        },
        services=services,
    )

    assert result["steps"][0]["reasoning"] == {
        "requirement": "optional_host_context",
        "service_supports_reasoning": True,
        "provider_configured": True,
        "contexts_used": 0,
        "validation_blocked": True,
        "validation_failures": [
            {
                "reason": "reasoning_validation_blocked",
                "target_id": "vendor-acme",
                "blockers": ["claim_missing_citations:0", "no_claims"],
            }
        ],
    }


@pytest.mark.asyncio
async def test_execute_step_logs_strict_validation_failures(caplog: pytest.LogCaptureFixture):
    class _StrictBlockedService(_ReasoningAwareOpportunityService):
        async def generate(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(dict(kwargs))
            return {
                "generated": 0,
                "skipped": 1,
                "reasoning_contexts_used": 0,
                "errors": [
                    {
                        "target_id": "vendor-acme",
                        "reason": "reasoning_validation_blocked:no_claims",
                    }
                ],
            }

    caplog.set_level(
        logging.WARNING,
        logger="extracted_content_pipeline.content_ops_execution",
    )
    services = ContentOpsExecutionServices(
        report=_StrictBlockedService(),
        reasoning_provider_configured=True,
        reasoning_provider_outputs=("report",),
    )

    await execute_content_ops_from_mapping(
        {
            "outputs": ["report"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Audit",
                "opportunity_id": "opp-1",
            },
        },
        services=services,
    )

    records = [
        record
        for record in caplog.records
        if record.message == "content_ops_strict_validation_blocked"
    ]
    assert len(records) == 1
    assert records[0].output == "report"
    assert records[0].failure_count == 1
    assert records[0].truncated is False


@pytest.mark.asyncio
async def test_execute_step_caps_strict_validation_failures():
    class _StrictBlockedService(_ReasoningAwareOpportunityService):
        async def generate(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(dict(kwargs))
            return {
                "generated": 0,
                "skipped": 55,
                "reasoning_contexts_used": 0,
                "errors": [
                    {
                        "target_id": f"vendor-{index}",
                        "reason": "reasoning_validation_blocked:no_claims",
                    }
                    for index in range(55)
                ],
            }

    services = ContentOpsExecutionServices(
        report=_StrictBlockedService(),
        reasoning_provider_configured=True,
        reasoning_provider_outputs=("report",),
    )

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["report"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Audit",
                "opportunity_id": "opp-1",
            },
        },
        services=services,
    )

    reasoning = result["steps"][0]["reasoning"]
    assert reasoning["validation_blocked"] is True
    assert reasoning["validation_failures_truncated"] is True
    assert len(reasoning["validation_failures"]) == 50
    assert reasoning["validation_failures"][0]["target_id"] == "vendor-0"
    assert reasoning["validation_failures"][-1]["target_id"] == "vendor-49"


@pytest.mark.asyncio
async def test_execute_step_ignores_malformed_consumed_reasoning_contexts():
    class _MalformedReasoningPayloadService(_ReasoningAwareOpportunityService):
        async def generate(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(dict(kwargs))
            return {
                "generated": 1,
                "reasoning_contexts_used": 1,
                "consumed_reasoning_contexts": ["not-a-context", {}, 3],
            }

    services = ContentOpsExecutionServices(
        campaign=_MalformedReasoningPayloadService(),
        reasoning_provider_configured=True,
    )

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {"target_account": "Acme", "offer": "Audit"},
        },
        services=services,
    )

    assert result["steps"][0]["reasoning"] == {
        "requirement": "optional_host_context",
        "service_supports_reasoning": True,
        "provider_configured": True,
        "contexts_used": 1,
    }


@pytest.mark.asyncio
async def test_execute_step_omits_contexts_used_when_result_is_uninstrumented():
    class _UninstrumentedResult:
        def as_dict(self) -> dict[str, Any]:
            return {"generated": 1, "saved_ids": ["draft-1"]}

    class _UninstrumentedReasoningService(_ReasoningAwareOpportunityService):
        async def generate(self, **kwargs: Any) -> _UninstrumentedResult:
            self.calls.append(dict(kwargs))
            return _UninstrumentedResult()

    services = ContentOpsExecutionServices(
        campaign=_UninstrumentedReasoningService(),
        reasoning_provider_configured=True,
    )

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["email_campaign"],
            "inputs": {"target_account": "Acme", "offer": "Audit"},
        },
        services=services,
    )

    assert result["steps"][0]["reasoning"] == {
        "requirement": "optional_host_context",
        "service_supports_reasoning": True,
        "provider_configured": True,
    }


@pytest.mark.asyncio
async def test_execute_step_omits_reasoning_audit_for_absent_requirement():
    signal = SignalExtractionService()

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["signal_extraction"],
            "inputs": {"source_material": "Acme switched from HubSpot."},
        },
        services=ContentOpsExecutionServices(signal_extraction=signal),
    )

    assert "reasoning" not in result["steps"][0]


def test_services_with_reasoning_context_derives_new_bundle_with_provider_attached():
    """``ContentOpsExecutionServices.with_reasoning_context`` returns a
    derived bundle where each opt-in service has been rebound via its
    own ``with_reasoning_context``. Services that don't expose the
    helper (or are None) are passed through unchanged."""

    class _OptInService:
        def __init__(self, reasoning_context=None):
            self._reasoning_context = reasoning_context

        def with_reasoning_context(self, provider):
            return _OptInService(reasoning_context=provider)

    class _OpaqueService:
        # No with_reasoning_context method.
        async def generate(self, **kwargs):
            del kwargs
            return {}

    base_opt_in = _OptInService()
    base_opaque = _OpaqueService()

    base = ContentOpsExecutionServices(
        campaign=base_opt_in,
        signal_extraction=base_opaque,
        # blog_post / report / landing_page / sales_brief left None.
    )

    sentinel = object()
    derived = base.with_reasoning_context(sentinel)

    # Different bundle, original unchanged.
    assert derived is not base
    assert base.campaign is base_opt_in  # untouched
    assert base_opt_in._reasoning_context is None  # not mutated

    # The opt-in service got a new instance with the sentinel attached.
    assert derived.campaign is not base_opt_in
    assert derived.campaign._reasoning_context is sentinel

    # The opaque service was passed through (no helper, no rebind).
    assert derived.signal_extraction is base_opaque

    # None slots stay None.
    assert derived.blog_post is None
    assert derived.report is None
    assert derived.landing_page is None
    assert derived.sales_brief is None
