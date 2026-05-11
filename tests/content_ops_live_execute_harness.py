from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from extracted_content_pipeline.blog_generation import BlogPostGenerationService
from extracted_content_pipeline.campaign_generation import CampaignGenerationService
from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    LLMMessage,
    LLMResponse,
    TenantScope,
)
from extracted_content_pipeline.content_ops_execution import ContentOpsExecutionServices
from extracted_content_pipeline.landing_page_generation import (
    LandingPageGenerationService,
)
from extracted_content_pipeline.report_generation import ReportGenerationService
from extracted_content_pipeline.sales_brief_generation import SalesBriefGenerationService
from extracted_content_pipeline.signal_extraction import SignalExtractionService


class MemoryIntelligence:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = [dict(row) for row in rows]

    async def read_campaign_opportunities(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        del scope, target_mode, filters
        return self.rows[:limit]


class MemoryDraftRepository:
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


class MemoryBlueprintRepository:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = [dict(row) for row in rows]

    async def read_blog_blueprints(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        del scope, target_mode, filters
        return self.rows[:limit]


class MemorySkillStore:
    prompts: Mapping[str, str] = {
        "digest/b2b_campaign_generation": "Campaign {target_mode} {channel} {opportunity_json}",
        "digest/blog_post_generation": "Blog topic={topic} blueprint={blueprint_json}",
        "digest/report_generation": "Report {target_mode} {opportunity_json}",
        "digest/landing_page_generation": "Landing {campaign_json}",
        "digest/sales_brief_generation": "Sales {target_mode} {opportunity_json}",
    }

    def get_prompt(self, name: str) -> str | None:
        return self.prompts.get(name)


class DeterministicContentLLM:
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
        content = json.dumps(response_for_metadata(meta), separators=(",", ":"))
        return LLMResponse(
            content=content,
            model="deterministic-content-smoke",
            usage={"input_tokens": 3, "output_tokens": 7},
        )


class StaticReasoningProvider:
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


@dataclass(frozen=True)
class ContentOpsLiveExecuteHarness:
    services: ContentOpsExecutionServices
    scope: TenantScope
    intelligence: MemoryIntelligence
    blueprints: MemoryBlueprintRepository
    llm: DeterministicContentLLM
    campaigns: MemoryDraftRepository
    blog_posts: MemoryDraftRepository
    reports: MemoryDraftRepository
    landing_pages: MemoryDraftRepository
    sales_briefs: MemoryDraftRepository
    reasoning: StaticReasoningProvider


def build_content_ops_live_execute_harness() -> ContentOpsLiveExecuteHarness:
    intelligence = MemoryIntelligence([default_opportunity()])
    blueprints = MemoryBlueprintRepository([default_blueprint()])
    llm = DeterministicContentLLM()
    skills = MemorySkillStore()
    campaigns = MemoryDraftRepository("campaign")
    blog_posts = MemoryDraftRepository("blog")
    reports = MemoryDraftRepository("report")
    landing_pages = MemoryDraftRepository("landing")
    sales_briefs = MemoryDraftRepository("brief")
    reasoning = StaticReasoningProvider()
    services = ContentOpsExecutionServices(
        campaign=CampaignGenerationService(intelligence=intelligence, campaigns=campaigns, llm=llm, skills=skills),
        blog_post=BlogPostGenerationService(blueprints=blueprints, blog_posts=blog_posts, llm=llm, skills=skills),
        report=ReportGenerationService(intelligence=intelligence, reports=reports, llm=llm, skills=skills),
        landing_page=LandingPageGenerationService(landing_pages=landing_pages, llm=llm, skills=skills),
        sales_brief=SalesBriefGenerationService(intelligence=intelligence, sales_briefs=sales_briefs, llm=llm, skills=skills),
        signal_extraction=SignalExtractionService(),
    )
    return ContentOpsLiveExecuteHarness(
        services=services,
        scope=TenantScope(account_id="acct-live", user_id="user-live"),
        intelligence=intelligence,
        blueprints=blueprints,
        llm=llm,
        campaigns=campaigns,
        blog_posts=blog_posts,
        reports=reports,
        landing_pages=landing_pages,
        sales_briefs=sales_briefs,
        reasoning=reasoning,
    )


def default_content_ops_execute_payload() -> dict[str, Any]:
    return {
        "target_mode": "vendor_retention",
        "outputs": ["email_campaign", "blog_post", "report", "landing_page", "sales_brief", "signal_extraction"],
        "limit": 1,
        "require_quality_gates": False,
        "inputs": {
            "target_account": "Acme",
            "offer": "Vendor risk audit",
            "topic": "Acme vendor pressure",
            "opportunity_id": "acct-acme",
            "audience": "RevOps leaders",
            "source_material": {
                "reviews": [{
                    "name": "Q1 review export",
                    "vendor": "HubSpot",
                    "text": "Acme reports pricing pressure and migration planning.",
                }]
            },
        },
    }


def default_opportunity() -> dict[str, Any]:
    return {
        "target_id": "acct-acme",
        "company_name": "Acme",
        "vendor_name": "HubSpot",
        "contact_email": "buyer@example.com",
        "pain_points": ["pricing pressure"],
        "evidence": [{"id": "ev-1", "text": "Pricing pressure is rising."}],
    }


def default_blueprint() -> dict[str, Any]:
    return {
        "id": "bp-acme",
        "slug": "acme-vendor-pressure",
        "topic": "Acme vendor pressure",
        "topic_type": "vendor_pressure",
        "data_context": {"vendor": "HubSpot"},
    }


def response_for_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
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
            "sections": [{"id": "overview", "title": "Overview", "body_markdown": "Evidence summary."}],
            "reference_ids": ["ev-1"],
        }
    if asset_type == "landing_page":
        return {
            "title": "Lower Vendor Risk",
            "slug": "lower-vendor-risk",
            "hero": {"headline": "Lower vendor risk"},
            "sections": [{"id": "proof", "title": "Proof", "body_markdown": "Proof section."}],
            "cta": {"label": "Book audit"},
            "meta": {"title_tag": "Lower Vendor Risk"},
        }
    if asset_type == "sales_brief":
        return {
            "title": "Acme Sales Brief",
            "headline": "Acme is ready for a vendor audit.",
            "sections": [{"id": "talking-points", "title": "Talking Points", "body_markdown": "Lead with pricing."}],
            "reference_ids": ["ev-1"],
        }
    raise AssertionError(f"unexpected LLM metadata: {metadata!r}")
