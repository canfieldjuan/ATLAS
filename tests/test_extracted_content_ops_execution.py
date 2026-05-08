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


@dataclass(frozen=True)
class _Result:
    generated: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {"generated": self.generated, "saved_ids": ["draft-1"]}


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
            "extras": dict(extras),
        })
        return _Result()


class _LandingPageService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(self, *, scope: TenantScope, campaign: Any) -> _Result:
        self.calls.append({"scope": scope, "campaign": campaign})
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
    assert len(report.calls) == 1
    report_call = report.calls[0]
    assert report_call["scope"] == scope
    assert report_call["target_mode"] == "vendor_retention"
    assert report_call["limit"] == 2
    assert report_call["filters"] == {"status": "ready"}
    # Plan default report type reaches the service.
    assert report_call["default_report_type"] == "vendor_pressure"
    assert report_call["channels"] is None


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


@pytest.mark.asyncio
async def test_execute_reports_missing_service_as_partial() -> None:
    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["sales_brief"],
            "inputs": {"target_account": "Acme"},
        },
        services=ContentOpsExecutionServices(),
    )

    assert result["status"] == "partial"
    assert result["steps"][0]["status"] == "failed"
    assert result["steps"][0]["error"] == "service_not_configured"
    assert result["errors"] == [
        {"output": "sales_brief", "reason": "service_not_configured"}
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

    assert result["status"] == "partial"
    assert result["steps"][0]["status"] == "failed"
    assert result["steps"][0]["error"] == "service_not_configured"
    assert result["errors"] == [
        {"output": "email_campaign", "reason": "service_not_configured"}
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
