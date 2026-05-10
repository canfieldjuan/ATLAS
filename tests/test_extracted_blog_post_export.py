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
        content="body",
        charts=({"id": "chart-1"},),
        data_context={"vendor": "Acme"},
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


@pytest.mark.asyncio
async def test_export_blog_post_drafts_defaults_summary_fields_without_metadata() -> None:
    result = await export_blog_post_drafts(
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
    assert "acme-pricing-pressure" in csv_text
    assert "{\"\"generation_parse_attempts\"\":2}" in csv_text


@pytest.mark.asyncio
async def test_export_blog_post_drafts_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="limit must be non-negative"):
        await export_blog_post_drafts(_Repository(), limit=-1)
