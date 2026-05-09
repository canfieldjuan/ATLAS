from __future__ import annotations

import json
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
    assert args[9] == "fake-llm"
    assert args[10] == date(2026, 5, 9)


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
    sql = pool.fetch_calls[0]["query"]
    args = pool.fetch_calls[0]["args"]
    assert "account_id = $1" in sql
    assert "status = $2" in sql
    assert "topic_type = $3" in sql
    assert "LIMIT $4" in sql
    assert args == ("acct-1", "draft", "vendor_alternative", 10)


@pytest.mark.asyncio
async def test_list_drafts_handles_pre_decoded_jsonb_columns() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "slug": "decoded",
            "title": "Decoded",
            "description": "",
            "topic_type": "market_landscape",
            "tags": ["market"],
            "content": "body",
            "charts": [{"id": "chart"}],
            "data_context": {"_metadata": {"generation_parse_attempts": 2}},
            "llm_model": None,
        }
    ]
    repo = PostgresBlogPostRepository(pool)

    drafts = await repo.list_drafts(scope=TenantScope())

    assert drafts[0].tags == ("market",)
    assert drafts[0].charts == ({"id": "chart"},)
    assert drafts[0].metadata["generation_parse_attempts"] == 2


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


@pytest.mark.asyncio
async def test_save_drafts_returns_empty_tuple_for_empty_input() -> None:
    pool = _Pool()
    repo = PostgresBlogPostRepository(pool)

    saved = await repo.save_drafts([], scope=TenantScope(account_id="acct-1"))

    assert saved == ()
    assert pool.fetchval_calls == []
