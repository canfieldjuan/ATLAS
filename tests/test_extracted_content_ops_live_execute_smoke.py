from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pytest

import extracted_content_pipeline.api.control_surfaces as api_module
from extracted_content_pipeline.api.control_surfaces import (
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.blog_generation import BlogPostGenerationService
from extracted_content_pipeline.blog_ports import BlogPostDraft
from extracted_content_pipeline.campaign_generation import CampaignGenerationService
from extracted_content_pipeline.campaign_ports import (
    CampaignDraft,
    CampaignReasoningContext,
    LLMMessage,
    LLMResponse,
    TenantScope,
)
from extracted_content_pipeline.content_ops_execution import (
    ContentOpsExecutionServices,
)
from extracted_content_pipeline.landing_page_generation import (
    LandingPageGenerationService,
)
from extracted_content_pipeline.landing_page_ports import LandingPageDraft
from extracted_content_pipeline.report_generation import ReportGenerationService
from extracted_content_pipeline.report_ports import ReportDraft
from extracted_content_pipeline.sales_brief_generation import (
    SalesBriefGenerationService,
)
from extracted_content_pipeline.sales_brief_ports import SalesBriefDraft
from extracted_content_pipeline.signal_extraction import SignalExtractionService

pytestmark = pytest.mark.skipif(
    api_module.APIRouter is None,
    reason="fastapi is not installed in this test environment",
)


try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - covered by pytestmark above.
    FastAPI = None
    TestClient = None


class _MemoryIntelligence:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = [dict(row) for row in rows]
        self.reads: list[dict[str, Any]] = []

    async def read_campaign_opportunities(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        self.reads.append({
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "filters": dict(filters or {}),
        })
        return self.rows[:limit]

    async def read_vendor_targets(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        vendor_name: str | None = None,
    ) -> Sequence[dict[str, Any]]:
        del scope, target_mode, vendor_name
        return ()


class _MemoryDraftRepository:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.saved: list[tuple[TenantScope, Any]] = []

    async def save_drafts(
        self,
        drafts: Sequence[Any],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        ids: list[str] = []
        for draft in drafts:
            ids.append(f"{self.prefix}-{len(self.saved) + 1}")
            self.saved.append((scope, draft))
        return ids


class _MemoryBlueprintRepository:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = [dict(row) for row in rows]
        self.reads: list[dict[str, Any]] = []

    async def read_blog_blueprints(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        self.reads.append({
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "filters": dict(filters or {}),
        })
        return self.rows[:limit]


class _MemorySkillStore:
    _PROMPTS: Mapping[str, str] = {
        "digest/b2b_campaign_generation": "Campaign {target_mode} {channel} {opportunity_json}",
        "digest/blog_post_generation": "Blog topic={topic} blueprint={blueprint_json}",
        "digest/report_generation": "Report {target_mode} {opportunity_json}",
        "digest/landing_page_generation": "Landing {campaign_json}",
        "digest/sales_brief_generation": "Sales {target_mode} {opportunity_json}",
    }

    def get_prompt(self, name: str) -> str | None:
        return self._PROMPTS.get(name)


class _DeterministicContentLLM:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        del messages, max_tokens, temperature
        meta = dict(metadata or {})
        self.calls.append(meta)
        content = json.dumps(_response_for_metadata(meta), separators=(",", ":"))
        return LLMResponse(
            content=content,
            model="deterministic-content-smoke",
            usage={"input_tokens": 3, "output_tokens": 7},
        )


class _StaticReasoningProvider:
    async def read_campaign_reasoning_context(
        self,
        *,
        scope: TenantScope,
        target_id: str,
        target_mode: str,
        opportunity: Mapping[str, Any],
    ) -> CampaignReasoningContext:
        del scope, opportunity
        return CampaignReasoningContext(
            canonical_reasoning={
                "summary": f"{target_id} has a {target_mode} timing trigger",
            },
            proof_points=(
                {"label": "source_material", "value": "pricing pressure"},
            ),
        )


def _response_for_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    asset_type = str(metadata.get("asset_type") or "")
    skill_name = str(metadata.get("skill_name") or "")
    if skill_name == "digest/b2b_campaign_generation":
        return {"subject": "Acme audit", "body": "Acme has a timely reason to review vendors."}
    if asset_type == "blog_post":
        return {
            "slug": "acme-vendor-pressure",
            "title": "Acme Vendor Pressure",
            "content": "A concise blog draft grounded in the supplied blueprint.",
            "topic_type": "vendor_pressure",
        }
    if asset_type == "report":
        return {
            "title": "Acme Vendor Pressure Report",
            "summary": "Acme shows pricing pressure and switching intent.",
            "sections": [
                {"id": "overview", "title": "Overview", "body_markdown": "Evidence summary."}
            ],
            "reference_ids": ["ev-1"],
        }
    if asset_type == "landing_page":
        return {
            "title": "Lower Vendor Risk",
            "slug": "lower-vendor-risk",
            "hero": {"headline": "Lower vendor risk"},
            "sections": [
                {"id": "proof", "title": "Proof", "body_markdown": "Proof section."}
            ],
            "cta": {"label": "Book audit"},
            "meta": {"title_tag": "Lower Vendor Risk"},
        }
    if asset_type == "sales_brief":
        return {
            "title": "Acme Sales Brief",
            "headline": "Acme is ready for a vendor audit.",
            "sections": [
                {"id": "talking-points", "title": "Talking Points", "body_markdown": "Lead with pricing."}
            ],
            "reference_ids": ["ev-1"],
        }
    raise AssertionError(f"unexpected LLM metadata: {metadata!r}")


def test_execute_route_persists_all_content_ops_assets_with_reasoning() -> None:
    """Route smoke: POST /content-ops/execute runs every output through
    real generated-asset services and host persistence ports."""

    assert FastAPI is not None
    assert TestClient is not None

    intelligence = _MemoryIntelligence([
        {
            "target_id": "acct-acme",
            "company_name": "Acme",
            "vendor_name": "HubSpot",
            "contact_email": "buyer@example.com",
            "pain_points": ["pricing pressure"],
            "evidence": [{"id": "ev-1", "text": "Pricing pressure is rising."}],
        }
    ])
    blueprints = _MemoryBlueprintRepository([
        {
            "id": "bp-acme",
            "slug": "acme-vendor-pressure",
            "topic": "Acme vendor pressure",
            "topic_type": "vendor_pressure",
            "data_context": {"vendor": "HubSpot"},
        }
    ])
    llm = _DeterministicContentLLM()
    skills = _MemorySkillStore()
    campaigns = _MemoryDraftRepository("campaign")
    blog_posts = _MemoryDraftRepository("blog")
    reports = _MemoryDraftRepository("report")
    landing_pages = _MemoryDraftRepository("landing")
    sales_briefs = _MemoryDraftRepository("brief")

    services = ContentOpsExecutionServices(
        campaign=CampaignGenerationService(
            intelligence=intelligence,
            campaigns=campaigns,
            llm=llm,
            skills=skills,
        ),
        blog_post=BlogPostGenerationService(
            blueprints=blueprints,
            blog_posts=blog_posts,
            llm=llm,
            skills=skills,
        ),
        report=ReportGenerationService(
            intelligence=intelligence,
            reports=reports,
            llm=llm,
            skills=skills,
        ),
        landing_page=LandingPageGenerationService(
            landing_pages=landing_pages,
            llm=llm,
            skills=skills,
        ),
        sales_brief=SalesBriefGenerationService(
            intelligence=intelligence,
            sales_briefs=sales_briefs,
            llm=llm,
            skills=skills,
        ),
        signal_extraction=SignalExtractionService(),
    )

    app = FastAPI()
    app.include_router(
        create_content_ops_control_surface_router(
            execution_services_provider=lambda: services,
            reasoning_context_provider=lambda: _StaticReasoningProvider(),
            scope_provider=lambda: TenantScope(account_id="acct-live", user_id="user-live"),
        )
    )

    response = TestClient(app).post(
        "/content-ops/execute",
        json={
            "target_mode": "vendor_retention",
            "outputs": [
                "email_campaign",
                "blog_post",
                "report",
                "landing_page",
                "sales_brief",
                "signal_extraction",
            ],
            "limit": 1,
            "require_quality_gates": False,
            "inputs": {
                "target_account": "Acme",
                "offer": "Vendor risk audit",
                "topic": "Acme vendor pressure",
                "opportunity_id": "acct-acme",
                "audience": "RevOps leaders",
                "source_material": {
                    "reviews": [
                        {
                            "name": "Q1 review export",
                            "vendor": "HubSpot",
                            "text": "Acme reports pricing pressure and migration planning.",
                        }
                    ]
                },
            },
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"

    steps = {step["output"]: step for step in payload["steps"]}
    assert set(steps) == {
        "email_campaign",
        "blog_post",
        "report",
        "landing_page",
        "sales_brief",
        "signal_extraction",
    }
    for output in ("email_campaign", "blog_post", "report", "landing_page", "sales_brief"):
        assert steps[output]["status"] == "completed"
        assert steps[output]["result"]["generated"] >= 1
        assert steps[output]["result"]["saved_ids"]
        assert steps[output]["reasoning"]["provider_configured"] is True
        assert steps[output]["reasoning"]["contexts_used"] >= 1
        assert steps[output]["reasoning"]["consumed_contexts"]

    assert steps["signal_extraction"]["status"] == "completed"
    assert steps["signal_extraction"]["result"]["generated"] == 1
    assert steps["signal_extraction"]["reasoning"]["requirement"] == "absent"

    assert len(campaigns.saved) == 2
    assert isinstance(campaigns.saved[0][1], CampaignDraft)
    assert len(blog_posts.saved) == 1
    assert isinstance(blog_posts.saved[0][1], BlogPostDraft)
    assert len(reports.saved) == 1
    assert isinstance(reports.saved[0][1], ReportDraft)
    assert len(landing_pages.saved) == 1
    assert isinstance(landing_pages.saved[0][1], LandingPageDraft)
    assert len(sales_briefs.saved) == 1
    assert isinstance(sales_briefs.saved[0][1], SalesBriefDraft)

    for repository in (campaigns, blog_posts, reports, landing_pages, sales_briefs):
        for scope, _draft in repository.saved:
            assert scope.account_id == "acct-live"
            assert scope.user_id == "user-live"

    assert {call["skill_name"] for call in llm.calls} == {
        "digest/b2b_campaign_generation",
        "digest/blog_post_generation",
        "digest/report_generation",
        "digest/landing_page_generation",
        "digest/sales_brief_generation",
    }
