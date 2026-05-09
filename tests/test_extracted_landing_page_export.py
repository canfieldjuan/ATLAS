from __future__ import annotations

import pytest

from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    LLMResponse,
    TenantScope,
)
from extracted_content_pipeline.landing_page_export import (
    LandingPageDraftExportResult,
    export_landing_page_drafts,
)
from extracted_content_pipeline.landing_page_generation import (
    LandingPageGenerationConfig,
    LandingPageGenerationService,
)
from extracted_content_pipeline.landing_page_ports import (
    LandingPageDraft,
    LandingPageSection,
    MarketingCampaign,
)


class _Repository:
    def __init__(self, drafts=None) -> None:
        self.drafts = tuple(drafts or ())
        self.list_calls: list[dict] = []

    async def save_drafts(self, drafts, *, scope):
        raise NotImplementedError

    async def list_drafts(
        self,
        *,
        scope,
        status=None,
        campaign_name=None,
        slug=None,
        limit=None,
    ):
        self.list_calls.append({
            "scope": scope,
            "status": status,
            "campaign_name": campaign_name,
            "slug": slug,
            "limit": limit,
        })
        return self.drafts

    async def update_status(self, landing_page_id, status, *, scope):
        raise NotImplementedError


class _SavingRepository(_Repository):
    async def save_drafts(self, drafts, *, scope):
        self.drafts = tuple(drafts)
        return tuple(f"landing-page-{index + 1}" for index, _ in enumerate(drafts))


class _LLM:
    async def complete(self, messages, *, max_tokens, temperature, metadata=None):
        return LLMResponse(
            content=(
                '{"title":"Acme launch","slug":"acme-launch",'
                '"hero":{"headline":"Stop renewal surprises"},'
                '"sections":[{"id":"problem","title":"Problem",'
                '"body_markdown":"Pricing pressure is rising"}],'
                '"cta":{"label":"Book a demo","url":"/demo"},'
                '"meta":{"title_tag":"Acme launch"},'
                '"reference_ids":["r1"]}'
            ),
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


class _Skills:
    def get_prompt(self, name):
        return "TEMPLATE {campaign_json}"


class _ReasoningProvider:
    async def read_campaign_reasoning_context(
        self,
        *,
        scope,
        target_id,
        target_mode,
        opportunity,
    ):
        return CampaignReasoningContext(
            canonical_reasoning={
                "wedge": "price_squeeze",
                "confidence": "high",
                "summary": "Pricing pressure creates displacement risk.",
            },
        )


def _draft(**overrides) -> LandingPageDraft:
    draft = LandingPageDraft(
        campaign_name="acme-q3-launch",
        persona="VP Engineering",
        value_prop="Cut renewal leakage",
        title="Acme Q3 launch",
        slug="acme-q3-launch",
        hero={"headline": "Stop renewal surprises"},
        sections=(
            LandingPageSection(
                id="problem",
                title="Problem",
                body_markdown="Renewal pricing is the main signal.",
                metadata={"order": 1},
            ),
        ),
        cta={"label": "Book a demo", "url": "/demo"},
        meta={"title_tag": "Acme Q3 launch"},
        reference_ids=("r1", "r2"),
        metadata={
            "generation_usage": {
                "input_tokens": 12,
                "output_tokens": 6,
                "total_tokens": 18,
            },
            "generation_parse_attempts": 2,
            "reasoning_context": {
                "wedge": "price_squeeze",
                "confidence": "high",
            },
        },
    )
    return LandingPageDraft(
        campaign_name=overrides.get("campaign_name", draft.campaign_name),
        persona=overrides.get("persona", draft.persona),
        value_prop=overrides.get("value_prop", draft.value_prop),
        title=overrides.get("title", draft.title),
        slug=overrides.get("slug", draft.slug),
        hero=overrides.get("hero", draft.hero),
        sections=overrides.get("sections", draft.sections),
        cta=overrides.get("cta", draft.cta),
        meta=overrides.get("meta", draft.meta),
        reference_ids=overrides.get("reference_ids", draft.reference_ids),
        metadata=overrides.get("metadata", draft.metadata),
    )


@pytest.mark.asyncio
async def test_export_landing_page_drafts_passes_filters_to_repository() -> None:
    repo = _Repository(drafts=[_draft()])

    result = await export_landing_page_drafts(
        repo,
        scope={"account_id": "acct_1", "user_id": "user_1"},
        status="approved",
        campaign_name="acme-q3-launch",
        slug="acme-q3-launch",
        limit=7,
    )

    call = repo.list_calls[0]
    assert isinstance(call["scope"], TenantScope)
    assert call["scope"].account_id == "acct_1"
    assert call["scope"].user_id == "user_1"
    assert call["status"] == "approved"
    assert call["campaign_name"] == "acme-q3-launch"
    assert call["slug"] == "acme-q3-launch"
    assert call["limit"] == 7
    assert result.limit == 7
    assert result.filters == {
        "status": "approved",
        "account_id": "acct_1",
        "campaign_name": "acme-q3-launch",
        "slug": "acme-q3-launch",
    }


@pytest.mark.asyncio
async def test_export_landing_page_drafts_derives_review_summary_fields() -> None:
    result = await export_landing_page_drafts(
        _Repository(drafts=[_draft()]),
        scope=TenantScope(account_id="acct_1"),
    )

    row = result.rows[0]
    assert row["campaign_name"] == "acme-q3-launch"
    assert row["section_count"] == 1
    assert row["reference_count"] == 2
    assert row["generation_input_tokens"] == 12
    assert row["generation_output_tokens"] == 6
    assert row["generation_total_tokens"] == 18
    assert row["generation_parse_attempts"] == 2
    assert row["reasoning_context_used"] is True
    assert row["reasoning_wedge"] == "price_squeeze"
    assert row["reasoning_confidence"] == "high"


@pytest.mark.asyncio
async def test_export_landing_page_drafts_defaults_summary_fields_without_metadata() -> None:
    result = await export_landing_page_drafts(
        _Repository(drafts=[_draft(metadata={})]),
        limit=1,
    )

    row = result.rows[0]
    assert row["generation_input_tokens"] is None
    assert row["generation_output_tokens"] is None
    assert row["generation_total_tokens"] is None
    assert row["generation_parse_attempts"] is None
    assert row["reasoning_context_used"] is False
    assert row["reasoning_wedge"] is None
    assert row["reasoning_confidence"] is None


@pytest.mark.asyncio
async def test_generated_landing_page_export_includes_reasoning_summary_fields() -> None:
    repo = _SavingRepository()
    service = LandingPageGenerationService(
        landing_pages=repo,
        llm=_LLM(),
        skills=_Skills(),
        reasoning_context=_ReasoningProvider(),
        config=LandingPageGenerationConfig(),
    )

    generated = await service.generate(
        scope=TenantScope(account_id="acct_1"),
        campaign=MarketingCampaign(
            name="acme-launch",
            persona="VP Engineering",
            value_prop="Catch pressure early",
        ),
    )
    exported = await export_landing_page_drafts(
        repo,
        scope=TenantScope(account_id="acct_1"),
    )

    assert generated.reasoning_contexts_used == 1
    row = exported.rows[0]
    assert row["reasoning_context_used"] is True
    assert row["reasoning_wedge"] == "price_squeeze"
    assert row["reasoning_confidence"] == "high"


@pytest.mark.asyncio
async def test_export_landing_page_drafts_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="limit must be non-negative"):
        await export_landing_page_drafts(_Repository(), limit=-1)


def test_landing_page_draft_export_result_renders_dict_and_csv() -> None:
    result = LandingPageDraftExportResult(
        rows=(
            {
                "campaign_name": "acme",
                "persona": "VP Engineering",
                "value_prop": "Catch pressure early",
                "title": "Acme page",
                "slug": "acme-page",
                "section_count": 1,
                "reference_count": 1,
                "generation_input_tokens": 12,
                "generation_output_tokens": 6,
                "generation_total_tokens": 18,
                "generation_parse_attempts": 1,
                "reasoning_context_used": True,
                "reasoning_wedge": "price_squeeze",
                "reasoning_confidence": "high",
                "hero": {"headline": "Stop surprises"},
                "sections": [{"id": "problem"}],
                "cta": {"label": "Book a demo"},
                "meta": {"title_tag": "Acme page"},
                "reference_ids": ["r1"],
                "metadata": {"scope": {"account_id": "acct_1"}},
            },
        ),
        limit=1,
        filters={"status": "draft"},
    )

    as_dict = result.as_dict()
    csv_text = result.as_csv()

    assert as_dict["count"] == 1
    assert as_dict["rows"][0]["reasoning_wedge"] == "price_squeeze"
    assert "campaign_name,persona,value_prop" in csv_text
    assert "generation_input_tokens,generation_output_tokens" in csv_text
    assert "reasoning_context_used,reasoning_wedge,reasoning_confidence" in csv_text
    assert "price_squeeze" in csv_text
