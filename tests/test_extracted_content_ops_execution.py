from __future__ import annotations

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
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> _Result:
        self.calls.append({
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "filters": dict(filters or {}),
        })
        return _Result()


class _LandingPageService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(self, *, scope: TenantScope, campaign: Any) -> _Result:
        self.calls.append({"scope": scope, "campaign": campaign})
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
    assert campaign.calls == [{
        "scope": scope,
        "target_mode": "vendor_retention",
        "limit": 2,
        "filters": {"status": "ready"},
    }]
    assert report.calls == campaign.calls


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
async def test_execute_blocks_non_executable_plan_without_calling_services() -> None:
    service = _OpportunityService()

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["blog_post"],
            "inputs": {"topic": "Churn pressure"},
        },
        services=ContentOpsExecutionServices(campaign=service),
    )

    assert result["status"] == "blocked"
    assert result["errors"] == [{"reason": "plan_not_executable"}]
    assert result["steps"] == []
    assert service.calls == []


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
