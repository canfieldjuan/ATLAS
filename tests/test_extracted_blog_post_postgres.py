from __future__ import annotations

import json
import logging
from datetime import date

import pytest

from extracted_content_pipeline.blog_ports import BlogPostDraft
from extracted_content_pipeline.blog_post_postgres import PostgresBlogPostRepository
from extracted_content_pipeline.campaign_ports import TenantScope


class _Pool:
    def __init__(self):
        self.fetchval_results: list[object] = []
        self.fetch_rows: list[dict] = []
        self.fetchval_calls: list[dict] = []
        self.fetch_calls: list[dict] = []
        self.execute_calls: list[dict] = []
        self.execute_result: object = "UPDATE 1"

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": query, "args": args})
        return self.fetchval_results.pop(0)

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": query, "args": args})
        return self.fetch_rows

    async def execute(self, query, *args):
        self.execute_calls.append({"query": query, "args": args})
        return self.execute_result


def _draft() -> BlogPostDraft:
    return BlogPostDraft(
        slug="acme-pricing-pressure",
        title="Acme Pricing Pressure",
        description="Pricing pressure dominates the latest review sample.",
        topic_type="vendor_alternative",
        tags=("pricing", "retention"),
        content="Pricing pressure is rising.",
        charts=({"id": "pricing_mentions", "type": "bar"},),
        data_context={"source_report_date": "2026-05-09", "vendor": "Acme"},
        metadata={
            "generation_model": "fake-llm",
            "generation_usage": {"input_tokens": 12, "output_tokens": 6},
            "seo_title": "Acme Pricing Pressure 2026",
            "seo_description": "Acme pricing pressure from recent review data.",
            "target_keyword": "acme pricing pressure",
            "secondary_keywords": ["acme pricing", "acme alternatives"],
            "faq": [{
                "question": "Why are Acme customers frustrated by pricing?",
                "answer": "Recent review evidence points to pricing pressure.",
            }],
        },
    )


@pytest.mark.asyncio
async def test_save_drafts_persists_scope_metadata_and_returns_ids() -> None:
    pool = _Pool()
    pool.fetchval_results = ["blog-post-uuid-1"]
    repo = PostgresBlogPostRepository(pool)

    saved = await repo.save_drafts([_draft()], scope=TenantScope(account_id="acct-1"))

    assert saved == ("blog-post-uuid-1",)
    args = pool.fetchval_calls[0]["args"]
    assert args[0] == "acct-1"
    assert args[1] == "acme-pricing-pressure"
    assert json.loads(args[5]) == ["pricing", "retention"]
    assert json.loads(args[7]) == [{"id": "pricing_mentions", "type": "bar"}]
    data_context = json.loads(args[8])
    assert data_context["scope"]["account_id"] == "acct-1"
    assert data_context["_metadata"]["generation_model"] == "fake-llm"
    assert data_context["_metadata"]["target_keyword"] == "acme pricing pressure"
    assert args[9] == "fake-llm"
    assert args[10] == date(2026, 5, 9)
    assert args[11] == "Acme Pricing Pressure 2026"
    assert args[12] == "Acme pricing pressure from recent review data."
    assert args[13] == "acme pricing pressure"
    assert json.loads(args[14]) == ["acme pricing", "acme alternatives"]
    assert json.loads(args[15]) == [{
        "question": "Why are Acme customers frustrated by pricing?",
        "answer": "Recent review evidence points to pricing pressure.",
    }]
    assert args[16] is True
    assert args[17] is True
    sql = pool.fetchval_calls[0]["query"]
    assert "seo_title, seo_description, target_keyword" in sql
    assert "secondary_keywords, faq" in sql


@pytest.mark.asyncio
async def test_save_drafts_retries_cross_tenant_slug_conflict_with_scope_slug() -> None:
    pool = _Pool()
    pool.fetchval_results = [None, "blog-post-uuid-2"]
    repo = PostgresBlogPostRepository(pool)

    saved = await repo.save_drafts([_draft()], scope=TenantScope(account_id="acct-b"))

    assert saved == ("blog-post-uuid-2",)
    assert len(pool.fetchval_calls) == 2
    assert pool.fetchval_calls[0]["args"][1] == "acme-pricing-pressure"
    assert pool.fetchval_calls[1]["args"][1] == "acme-pricing-pressure-acct-b"
    sql = pool.fetchval_calls[0]["query"]
    assert "blog_posts.account_id = EXCLUDED.account_id" in sql


@pytest.mark.asyncio
async def test_save_drafts_uses_numbered_scope_slug_when_first_fallback_conflicts() -> None:
    pool = _Pool()
    pool.fetchval_results = [None, None, "blog-post-uuid-3"]
    repo = PostgresBlogPostRepository(pool)

    saved = await repo.save_drafts([_draft()], scope=TenantScope(account_id="acct-b"))

    assert saved == ("blog-post-uuid-3",)
    assert pool.fetchval_calls[1]["args"][1] == "acme-pricing-pressure-acct-b"
    assert pool.fetchval_calls[2]["args"][1] == "acme-pricing-pressure-acct-b-2"


@pytest.mark.asyncio
async def test_save_drafts_warns_when_slug_retries_are_exhausted(caplog) -> None:
    pool = _Pool()
    pool.fetchval_results = [None, None, None, None, None]
    repo = PostgresBlogPostRepository(pool)

    with caplog.at_level(
        logging.WARNING,
        logger="extracted_content_pipeline.blog_post_postgres",
    ):
        saved = await repo.save_drafts([_draft()], scope=TenantScope(account_id="acct-b"))

    assert saved == ()
    assert [call["args"][1] for call in pool.fetchval_calls] == [
        "acme-pricing-pressure",
        "acme-pricing-pressure-acct-b",
        "acme-pricing-pressure-acct-b-2",
        "acme-pricing-pressure-acct-b-3",
        "acme-pricing-pressure-acct-b-4",
    ]
    warning = caplog.records[0]
    assert warning.message == "blog_post_draft_save_exhausted_slug_attempts"
    assert warning.account_id == "acct-b"
    assert warning.slug == "acme-pricing-pressure"
    assert warning.attempts == 5


@pytest.mark.asyncio
async def test_save_drafts_preserves_existing_seo_on_conflict_when_metadata_omits_it() -> None:
    pool = _Pool()
    pool.fetchval_results = ["blog-post-uuid-1"]
    repo = PostgresBlogPostRepository(pool)
    base = _draft()
    draft = BlogPostDraft(
        slug=base.slug,
        title=base.title,
        description=base.description,
        topic_type=base.topic_type,
        tags=base.tags,
        content=base.content,
        charts=base.charts,
        data_context=base.data_context,
        metadata={"generation_model": "fake-llm"},
    )

    await repo.save_drafts([draft], scope=TenantScope(account_id="acct-1"))

    args = pool.fetchval_calls[0]["args"]
    assert args[11] is None
    assert args[12] is None
    assert args[13] is None
    assert json.loads(args[14]) == []
    assert json.loads(args[15]) == []
    assert args[16] is False
    assert args[17] is False
    sql = pool.fetchval_calls[0]["query"]
    assert "seo_title = COALESCE(EXCLUDED.seo_title, blog_posts.seo_title)" in sql
    assert "WHEN $17 THEN EXCLUDED.secondary_keywords" in sql
    assert "WHEN $18 THEN EXCLUDED.faq" in sql


@pytest.mark.asyncio
async def test_list_drafts_filters_by_status_topic_and_scope() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "slug": "acme-pricing-pressure",
            "title": "Acme Pricing Pressure",
            "description": "Pricing pressure dominates.",
            "topic_type": "vendor_alternative",
            "tags": json.dumps(["pricing"]),
            "content": "body",
            "charts": json.dumps([{"id": "chart-1"}]),
            "data_context": json.dumps({
                "vendor": "Acme",
                "_metadata": {"generation_model": "fake-llm"},
            }),
            "llm_model": "fake-llm",
            "seo_title": "Acme Pricing Pressure 2026",
            "seo_description": "Acme pricing pressure from recent review data.",
            "target_keyword": "acme pricing pressure",
            "secondary_keywords": json.dumps(["acme pricing", "acme alternatives"]),
            "faq": json.dumps([{
                "question": "Why are Acme customers frustrated by pricing?",
                "answer": "Recent review evidence points to pricing pressure.",
            }]),
        }
    ]
    repo = PostgresBlogPostRepository(pool)

    drafts = await repo.list_drafts(
        scope=TenantScope(account_id="acct-1"),
        status="draft",
        topic_type="vendor_alternative",
        limit=10,
    )

    assert len(drafts) == 1
    assert drafts[0].slug == "acme-pricing-pressure"
    assert drafts[0].tags == ("pricing",)
    assert drafts[0].charts == ({"id": "chart-1"},)
    assert drafts[0].data_context == {"vendor": "Acme"}
    assert drafts[0].metadata["generation_model"] == "fake-llm"
    assert drafts[0].metadata["seo_title"] == "Acme Pricing Pressure 2026"
    assert drafts[0].metadata["seo_description"] == (
        "Acme pricing pressure from recent review data."
    )
    assert drafts[0].metadata["target_keyword"] == "acme pricing pressure"
    assert drafts[0].metadata["secondary_keywords"] == [
        "acme pricing",
        "acme alternatives",
    ]
    assert drafts[0].metadata["faq"] == [{
        "question": "Why are Acme customers frustrated by pricing?",
        "answer": "Recent review evidence points to pricing pressure.",
    }]
    sql = pool.fetch_calls[0]["query"]
    args = pool.fetch_calls[0]["args"]
    assert "seo_title, seo_description, target_keyword" in sql
    assert "secondary_keywords, faq" in sql
    assert "account_id = $1" in sql
    assert "status = $2" in sql
    assert "topic_type = $3" in sql
    assert "LIMIT $4" in sql
    assert args == ("acct-1", "draft", "vendor_alternative", 10)


@pytest.mark.asyncio
async def test_list_drafts_preserves_metadata_seo_arrays_when_columns_are_empty() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "slug": "acme-pricing-pressure",
            "title": "Acme Pricing Pressure",
            "description": "Pricing pressure dominates.",
            "topic_type": "vendor_alternative",
            "tags": json.dumps([]),
            "content": "body",
            "charts": json.dumps([]),
            "data_context": json.dumps({
                "_metadata": {
                    "secondary_keywords": ["legacy keyword"],
                    "faq": [{"question": "Legacy?", "answer": "Yes."}],
                },
            }),
            "llm_model": None,
            "seo_title": None,
            "seo_description": None,
            "target_keyword": None,
            "secondary_keywords": json.dumps([]),
            "faq": json.dumps([]),
        }
    ]
    repo = PostgresBlogPostRepository(pool)

    drafts = await repo.list_drafts(scope=TenantScope(account_id="acct-1"))

    assert drafts[0].metadata["secondary_keywords"] == ["legacy keyword"]
    assert drafts[0].metadata["faq"] == [{"question": "Legacy?", "answer": "Yes."}]


@pytest.mark.asyncio
async def test_update_status_returns_true_on_hit_and_runs_scoped_update() -> None:
    pool = _Pool()
    repo = PostgresBlogPostRepository(pool)

    hit = await repo.update_status(
        "blog-post-uuid-1",
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert hit is True
    query = pool.execute_calls[0]["query"]
    args = pool.execute_calls[0]["args"]
    assert "UPDATE blog_posts" in query
    assert "account_id = $3" in query
    assert args == ("blog-post-uuid-1", "approved", "acct-1")


@pytest.mark.asyncio
async def test_update_status_returns_false_on_miss() -> None:
    pool = _Pool()
    pool.execute_result = "UPDATE 0"
    repo = PostgresBlogPostRepository(pool)

    hit = await repo.update_status("missing-id", "approved", scope=TenantScope())

    assert hit is False
