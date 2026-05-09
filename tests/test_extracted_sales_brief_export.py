from __future__ import annotations

import pytest

from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    LLMResponse,
    TenantScope,
)
from extracted_content_pipeline.sales_brief_export import (
    SalesBriefDraftExportResult,
    export_sales_brief_drafts,
)
from extracted_content_pipeline.sales_brief_generation import (
    SalesBriefGenerationConfig,
    SalesBriefGenerationService,
)
from extracted_content_pipeline.sales_brief_ports import (
    SalesBriefDraft,
    SalesBriefSection,
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
        target_mode=None,
        brief_type=None,
        limit=None,
    ):
        self.list_calls.append({
            "scope": scope,
            "status": status,
            "target_mode": target_mode,
            "brief_type": brief_type,
            "limit": limit,
        })
        return self.drafts

    async def update_status(self, brief_id, status, *, scope):
        raise NotImplementedError


class _SavingRepository(_Repository):
    async def save_drafts(self, drafts, *, scope):
        self.drafts = tuple(drafts)
        return tuple(f"sales-brief-{index + 1}" for index, _ in enumerate(drafts))


class _Intelligence:
    async def read_campaign_opportunities(self, *, scope, target_mode, limit, filters=None):
        return ({
            "target_id": "vendor-acme",
            "vendor_name": "Acme",
            "company_name": "Acme",
            "evidence": [{"id": "r1", "quote": "Pricing pressure is rising"}],
        },)

    async def read_vendor_targets(self, *, scope, target_mode, vendor_name=None):  # pragma: no cover
        raise AssertionError("not used")


class _LLM:
    async def complete(self, messages, *, max_tokens, temperature, metadata=None):
        return LLMResponse(
            content=(
                '{"title":"Acme pre-call brief","headline":"Renewal pressure '
                'opens this week","brief_type":"pre_call","sections":[{'
                '"id":"context","title":"Context","body_markdown":"Pricing '
                'pressure is rising","evidence_ids":["r1"]}],'
                '"reference_ids":["r1"],"confidence":0.82}'
            ),
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


class _Skills:
    def get_prompt(self, name):
        return "TEMPLATE {target_mode}: {opportunity_json}"


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


def _draft(**overrides) -> SalesBriefDraft:
    draft = SalesBriefDraft(
        target_id="vendor-acme",
        target_mode="vendor_retention",
        brief_type="pre_call",
        title="Acme pre-call brief",
        headline="Renewal pressure opens this week",
        sections=(
            SalesBriefSection(
                id="context",
                title="Context",
                body_markdown="Renewal pricing is the main signal.",
                claim_ids=("c1",),
                evidence_ids=("r1",),
                metadata={"order": 1},
            ),
        ),
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
    return SalesBriefDraft(
        target_id=overrides.get("target_id", draft.target_id),
        target_mode=overrides.get("target_mode", draft.target_mode),
        brief_type=overrides.get("brief_type", draft.brief_type),
        title=overrides.get("title", draft.title),
        headline=overrides.get("headline", draft.headline),
        sections=overrides.get("sections", draft.sections),
        reference_ids=overrides.get("reference_ids", draft.reference_ids),
        metadata=overrides.get("metadata", draft.metadata),
    )


@pytest.mark.asyncio
async def test_export_sales_brief_drafts_passes_filters_to_repository() -> None:
    repo = _Repository(drafts=[_draft()])

    result = await export_sales_brief_drafts(
        repo,
        scope={"account_id": "acct_1", "user_id": "user_1"},
        status="approved",
        target_mode="vendor_retention",
        brief_type="pre_call",
        limit=7,
    )

    call = repo.list_calls[0]
    assert isinstance(call["scope"], TenantScope)
    assert call["scope"].account_id == "acct_1"
    assert call["scope"].user_id == "user_1"
    assert call["status"] == "approved"
    assert call["target_mode"] == "vendor_retention"
    assert call["brief_type"] == "pre_call"
    assert call["limit"] == 7
    assert result.limit == 7
    assert result.filters == {
        "status": "approved",
        "account_id": "acct_1",
        "target_mode": "vendor_retention",
        "brief_type": "pre_call",
    }


@pytest.mark.asyncio
async def test_export_sales_brief_drafts_derives_review_summary_fields() -> None:
    result = await export_sales_brief_drafts(
        _Repository(drafts=[_draft()]),
        scope=TenantScope(account_id="acct_1"),
    )

    row = result.rows[0]
    assert row["target_id"] == "vendor-acme"
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
async def test_export_sales_brief_drafts_defaults_summary_fields_without_metadata() -> None:
    result = await export_sales_brief_drafts(
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
async def test_generated_sales_brief_export_includes_reasoning_summary_fields() -> None:
    repo = _SavingRepository()
    service = SalesBriefGenerationService(
        intelligence=_Intelligence(),
        sales_briefs=repo,
        llm=_LLM(),
        skills=_Skills(),
        reasoning_context=_ReasoningProvider(),
        config=SalesBriefGenerationConfig(),
    )

    generated = await service.generate(
        scope=TenantScope(account_id="acct_1"),
        target_mode="vendor_retention",
    )
    exported = await export_sales_brief_drafts(
        repo,
        scope=TenantScope(account_id="acct_1"),
    )

    assert generated.reasoning_contexts_used == 1
    row = exported.rows[0]
    assert row["reasoning_context_used"] is True
    assert row["reasoning_wedge"] == "price_squeeze"
    assert row["reasoning_confidence"] == "high"


@pytest.mark.asyncio
async def test_export_sales_brief_drafts_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="limit must be non-negative"):
        await export_sales_brief_drafts(_Repository(), limit=-1)


def test_sales_brief_draft_export_result_renders_dict_and_csv() -> None:
    result = SalesBriefDraftExportResult(
        rows=(
            {
                "target_id": "vendor-acme",
                "target_mode": "vendor_retention",
                "brief_type": "pre_call",
                "title": "Acme brief",
                "headline": "Renewal pressure opens this week",
                "section_count": 1,
                "reference_count": 1,
                "generation_input_tokens": 12,
                "generation_output_tokens": 6,
                "generation_total_tokens": 18,
                "generation_parse_attempts": 1,
                "reasoning_context_used": True,
                "reasoning_wedge": "price_squeeze",
                "reasoning_confidence": "high",
                "sections": [{"id": "context"}],
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
    assert "target_id,target_mode,brief_type" in csv_text
    assert "generation_input_tokens,generation_output_tokens" in csv_text
    assert "reasoning_context_used,reasoning_wedge,reasoning_confidence" in csv_text
    assert "price_squeeze" in csv_text
