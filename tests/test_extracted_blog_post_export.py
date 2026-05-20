from __future__ import annotations

import pytest

from extracted_content_pipeline.blog_ports import BlogPostDraft
from extracted_content_pipeline.blog_post_export import (
    BlogPostDraftExportResult,
    export_blog_post_drafts,
)
from extracted_content_pipeline.campaign_ports import TenantScope


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
        topic_type=None,
        limit=None,
    ):
        self.list_calls.append({
            "scope": scope,
            "status": status,
            "topic_type": topic_type,
            "limit": limit,
        })
        return self.drafts

    async def update_status(self, blog_post_id, status, *, scope):
        raise NotImplementedError


def _draft(**overrides) -> BlogPostDraft:
    draft = BlogPostDraft(
        slug="acme-pricing-pressure",
        title="Acme Pricing Pressure",
        description="Pricing pressure dominates.",
        topic_type="vendor_alternative",
        tags=("pricing", "retention"),
        content=(
            "## Why is Acme pricing pressure showing up?\n\n"
            "Acme pricing pressure is visible in the last 90 days of review "
            "patterns, especially across 214 reviews where buyers describe "
            "renewal friction, budget concerns, and comparison shopping. The "
            "answer is that Acme buyers are not only comparing features; they "
            "are checking whether the contract still fits the budget before "
            "another renewal cycle starts.\n\n"
            "## How should teams read the Acme pricing evidence?\n\n"
            "Acme pricing evidence should be read as a renewal-risk signal, not "
            "as proof that every buyer has the same problem. The useful pattern "
            "is that customers keep using similar wording about budget pressure, "
            "contract terms, and alternative comparisons when they explain why "
            "pricing has become harder to justify."
        ),
        charts=({"id": "chart-1"},),
        data_context={"vendor": "Acme", "review_period": "last 90 days"},
        metadata={
            "seo_title": "Acme Pricing Pressure 2026",
            "seo_description": "Acme pricing pressure from recent review data.",
            "target_keyword": "acme pricing pressure",
            "secondary_keywords": ["acme pricing", "acme alternatives"],
            "faq": [
                {
                    "question": "Why is Acme pricing a concern?",
                    "answer": "Pricing pressure appears in review data.",
                },
                {
                    "question": "Who should compare Acme alternatives?",
                    "answer": "Teams with budget pressure should compare options.",
                },
                {
                    "question": "What should buyers check?",
                    "answer": "Buyers should check contract and support terms.",
                },
            ],
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
    return BlogPostDraft(
        slug=overrides.get("slug", draft.slug),
        title=overrides.get("title", draft.title),
        description=overrides.get("description", draft.description),
        topic_type=overrides.get("topic_type", draft.topic_type),
        tags=overrides.get("tags", draft.tags),
        content=overrides.get("content", draft.content),
        charts=overrides.get("charts", draft.charts),
        data_context=overrides.get("data_context", draft.data_context),
        metadata=overrides.get("metadata", draft.metadata),
    )


@pytest.mark.asyncio
async def test_export_blog_post_drafts_passes_filters_to_repository() -> None:
    repo = _Repository(drafts=[_draft()])

    result = await export_blog_post_drafts(
        repo,
        scope={"account_id": "acct_1", "user_id": "user_1"},
        status="approved",
        topic_type="vendor_alternative",
        limit=7,
    )

    call = repo.list_calls[0]
    assert isinstance(call["scope"], TenantScope)
    assert call["scope"].account_id == "acct_1"
    assert call["scope"].user_id == "user_1"
    assert call["status"] == "approved"
    assert call["topic_type"] == "vendor_alternative"
    assert call["limit"] == 7
    assert result.filters == {
        "status": "approved",
        "account_id": "acct_1",
        "topic_type": "vendor_alternative",
    }


@pytest.mark.asyncio
async def test_export_blog_post_drafts_derives_review_summary_fields() -> None:
    result = await export_blog_post_drafts(
        _Repository(drafts=[_draft()]),
        scope=TenantScope(account_id="acct_1"),
    )

    row = result.rows[0]
    assert row["slug"] == "acme-pricing-pressure"
    assert row["tag_count"] == 2
    assert row["chart_count"] == 1
    assert row["generation_input_tokens"] == 12
    assert row["generation_output_tokens"] == 6
    assert row["generation_total_tokens"] == 18
    assert row["generation_parse_attempts"] == 2
    assert row["reasoning_context_used"] is True
    assert row["reasoning_wedge"] == "price_squeeze"
    assert row["reasoning_confidence"] == "high"
    assert row["passed_output_checks"] == 6
    assert row["output_checks"] == {
        "seo_title_ready": True,
        "seo_description_ready": True,
        "target_keyword_present": True,
        "secondary_keywords_present": True,
        "faq_ready": True,
        "aeo_structure_detected": True,
    }
    assert row["seo_aeo_readiness"] == {
        "status": "ready",
        "passed": 6,
        "total": 6,
        "missing": [],
        "checks": row["output_checks"],
    }
    assert row["geo_readiness"] == {
        "status": "ready",
        "passed": 7,
        "total": 7,
        "missing": [],
        "checks": {
            "entity_clarity": True,
            "answer_first_sections": True,
            "citable_section_structure": True,
            "evidence_specificity": True,
            "freshness_context": True,
            "faq_coverage": True,
            "citation_safety": True,
        },
    }


@pytest.mark.asyncio
async def test_export_blog_post_drafts_defaults_summary_fields_without_metadata() -> None:
    result = await export_blog_post_drafts(
        _Repository(drafts=[_draft(content="body", metadata={})]),
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
    assert row["passed_output_checks"] == 0
    assert row["seo_aeo_readiness"]["status"] == "needs_review"
    assert row["seo_aeo_readiness"]["missing"] == [
        "seo_title_ready",
        "seo_description_ready",
        "target_keyword_present",
        "secondary_keywords_present",
        "faq_ready",
        "aeo_structure_detected",
    ]
    assert row["geo_readiness"]["status"] == "needs_review"
    assert row["geo_readiness"]["checks"]["citation_safety"] is True


def test_blog_post_export_result_renders_csv() -> None:
    result = BlogPostDraftExportResult(
        rows=(
            {
                "slug": "acme-pricing-pressure",
                "title": "Acme Pricing Pressure",
                "topic_type": "vendor_alternative",
                "tags": ["pricing"],
                "metadata": {"generation_parse_attempts": 2},
            },
        ),
        limit=20,
        filters={"status": "draft"},
    )

    csv_text = result.as_csv()

    assert csv_text.startswith("slug,title,description,topic_type")
    assert "passed_output_checks,output_checks,seo_aeo_readiness,geo_readiness" in csv_text
    assert "acme-pricing-pressure" in csv_text
    assert "{\"\"generation_parse_attempts\"\":2}" in csv_text


@pytest.mark.asyncio
async def test_export_blog_post_drafts_surfaces_incomplete_geo_readiness() -> None:
    result = await export_blog_post_drafts(
        _Repository(drafts=[
            _draft(
                title="",
                content="## Overview\n\nNo clear answer yet. {{todo}}",
                data_context={},
                metadata={},
            )
        ]),
        scope=TenantScope(account_id="acct_1"),
    )

    readiness = result.rows[0]["geo_readiness"]
    assert readiness["status"] == "needs_review"
    assert readiness["passed"] == 0
    assert readiness["missing"] == [
        "entity_clarity",
        "answer_first_sections",
        "citable_section_structure",
        "evidence_specificity",
        "freshness_context",
        "faq_coverage",
        "citation_safety",
    ]


@pytest.mark.asyncio
async def test_export_blog_post_drafts_detects_answer_first_sections() -> None:
    content = (
        "## Acme Pricing Pattern\n\n"
        "Acme pricing pressure appears in review data when buyers describe "
        "renewal friction, budget pressure, and increased comparison shopping. "
        "The point is not that every buyer has the same problem, but that the "
        "article opens the section with a direct answer before detail."
    )

    result = await export_blog_post_drafts(
        _Repository(drafts=[_draft(content=content)]),
        scope=TenantScope(account_id="acct_1"),
    )

    assert result.rows[0]["output_checks"]["aeo_structure_detected"] is True


@pytest.mark.asyncio
async def test_export_blog_post_drafts_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="limit must be non-negative"):
        await export_blog_post_drafts(_Repository(), limit=-1)
