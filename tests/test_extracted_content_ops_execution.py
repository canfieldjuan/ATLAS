from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.content_ops_execution import (
    ContentOpsExecutionServices,
    execute_content_ops_from_mapping,
)
from extracted_content_pipeline.signal_extraction import SignalExtractionService


@dataclass(frozen=True)
class _Result:
    generated: int = 1
    reasoning_contexts_used: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated": self.generated,
            "reasoning_contexts_used": self.reasoning_contexts_used,
            "saved_ids": ["draft-1"],
        }


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
        quality_revalidation_enabled: bool | None = None,
        quality_prompt_proof_term_limit: int | None = None,
        parse_retry_response_excerpt_chars: int | None = None,
        quality_gates_enabled: bool | None = None,
        topic: str | None = None,
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
            "quality_revalidation_enabled": quality_revalidation_enabled,
            "quality_prompt_proof_term_limit": quality_prompt_proof_term_limit,
            "parse_retry_response_excerpt_chars": parse_retry_response_excerpt_chars,
            "quality_gates_enabled": quality_gates_enabled,
            "topic": topic,
            "extras": dict(extras),
        })
        return _Result()


class _ReasoningAwareOpportunityService(_OpportunityService):
    def __init__(self, reasoning_context: Any | None = None) -> None:
        super().__init__()
        self._reasoning_context = reasoning_context

    def with_reasoning_context(
        self,
        provider: Any | None,
    ) -> "_ReasoningAwareOpportunityService":
        return _ReasoningAwareOpportunityService(reasoning_context=provider)


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
            "extras": dict(extras),
        })
        return _Result()


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
    assert report_call["extras"] == {}


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
            },
        },
        services=ContentOpsExecutionServices(landing_page=landing),
    )

    campaign = landing.calls[0]["campaign"]
    context = dict(campaign.context)

    # Allowlisted fields land in context.
    assert context.get("industry") == "fintech"
    assert context.get("pain_points") == ["churn", "renewal"]

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
    assert call["extras"] == {}


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
