from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.stat_card_ports import StatCardDraft
from extracted_content_pipeline.stat_card_postgres import (
    PostgresStatCardRepository,
)


class _Pool:
    def __init__(self) -> None:
        self.fetchval_results: list[object] = []
        self.fetch_rows: list[dict[str, object]] = []
        self.fetchval_calls: list[dict[str, object]] = []
        self.fetch_calls: list[dict[str, object]] = []
        self.execute_calls: list[dict[str, object]] = []
        self.execute_result: object = "UPDATE 1"

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": str(query), "args": args})
        return self.fetchval_results.pop(0)

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": str(query), "args": args})
        return self.fetch_rows

    async def execute(self, query, *args):
        self.execute_calls.append({"query": str(query), "args": args})
        return self.execute_result


def _compact_sql(query: object) -> str:
    return " ".join(str(query).split())


def _draft() -> StatCardDraft:
    return StatCardDraft(
        target_id="review-1",
        target_mode="vendor_retention",
        theme="customer_metric",
        metric_label="NPS score",
        metric_value=42,
        metric_display="42",
        claim="NPS score: 42",
        headline="Customer metric for Zendesk",
        supporting_text="Use this stat to frame renewal dissatisfaction.",
        evidence="NPS score dropped to 42 after renewal.",
        source_id="review-1",
        source_type="review",
        company_name="Acme Logistics",
        vendor_name="Zendesk",
        pain_points=("renewal dissatisfaction",),
        metadata={"confidence": 0.88},
    )


@pytest.mark.asyncio
async def test_save_drafts_persists_stat_cards_and_returns_ids() -> None:
    pool = _Pool()
    pool.fetchval_results = ["stat-card-uuid-1"]
    repo = PostgresStatCardRepository(pool)

    saved = await repo.save_drafts(
        [_draft()],
        scope=TenantScope(account_id="acct-1", user_id="user-1"),
    )

    assert saved == ("stat-card-uuid-1",)
    call = pool.fetchval_calls[0]
    assert "INSERT INTO stat_card_drafts" in call["query"]
    args = call["args"]
    assert args[:15] == (
        "acct-1",
        "review-1",
        "vendor_retention",
        "customer_metric",
        "NPS score",
        "42",
        "42",
        "NPS score: 42",
        "Customer metric for Zendesk",
        "Use this stat to frame renewal dissatisfaction.",
        "NPS score dropped to 42 after renewal.",
        "review-1",
        "review",
        "Acme Logistics",
        "Zendesk",
    )
    assert json.loads(args[15]) == ["renewal dissatisfaction"]
    metadata = json.loads(args[16])
    assert metadata["target_id"] == "review-1"
    assert metadata["target_mode"] == "vendor_retention"
    assert metadata["scope"] == {"account_id": "acct-1", "user_id": "user-1"}
    assert metadata["confidence"] == 0.88


@pytest.mark.asyncio
async def test_list_drafts_filters_by_scope_status_target_mode_and_theme() -> None:
    pool = _Pool()
    pool.fetch_rows = [{
        "id": "stat-card-uuid-1",
        "target_id": "review-1",
        "target_mode": "vendor_retention",
        "theme": "customer_metric",
        "metric_label": "NPS score",
        "metric_value": 42,
        "metric_display": "42",
        "claim": "NPS score: 42",
        "headline": "Customer metric for Zendesk",
        "supporting_text": "Use this stat to frame renewal dissatisfaction.",
        "evidence": "NPS score dropped to 42 after renewal.",
        "source_id": "review-1",
        "source_type": "review",
        "company_name": "Acme Logistics",
        "vendor_name": "Zendesk",
        "pain_points": json.dumps(["renewal dissatisfaction"]),
        "metadata": json.dumps({"confidence": 0.88}),
        "status": "draft",
    }]
    repo = PostgresStatCardRepository(pool)

    drafts = await repo.list_drafts(
        scope=TenantScope(account_id="acct-1"),
        status="draft",
        target_mode="vendor_retention",
        theme="customer_metric",
        limit=10,
    )

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.id == "stat-card-uuid-1"
    assert draft.theme == "customer_metric"
    assert draft.metric_label == "NPS score"
    assert draft.metric_value == 42
    assert draft.pain_points == ("renewal dissatisfaction",)
    assert draft.metadata == {"confidence": 0.88}
    call = pool.fetch_calls[0]
    assert "FROM stat_card_drafts" in call["query"]
    assert "account_id = $1" in call["query"]
    assert call["args"] == (
        "acct-1",
        "draft",
        "vendor_retention",
        "customer_metric",
        10,
    )


@pytest.mark.asyncio
async def test_update_status_is_tenant_scoped_and_reports_misses() -> None:
    pool = _Pool()
    pool.execute_result = "UPDATE 0"
    repo = PostgresStatCardRepository(pool)

    updated = await repo.update_status(
        "stat-card-uuid-1",
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated is False
    call = pool.execute_calls[0]
    sql = _compact_sql(call["query"])
    assert "UPDATE stat_card_drafts" in sql
    assert "WHERE id = $1::uuid AND account_id = $3" in sql
    assert call["args"] == ("stat-card-uuid-1", "approved", "acct-1")


@pytest.mark.asyncio
async def test_update_statuses_returns_scoped_matches() -> None:
    pool = _Pool()
    pool.fetch_rows = [{"id": "stat-card-uuid-1"}]
    repo = PostgresStatCardRepository(pool)

    updated = await repo.update_statuses(
        ["stat-card-uuid-1", ""],
        "rejected",
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated == ("stat-card-uuid-1",)
    call = pool.fetch_calls[0]
    sql = _compact_sql(call["query"])
    assert "UPDATE stat_card_drafts" in sql
    assert "WHERE id = ANY($1::uuid[]) AND account_id = $3" in sql
    assert call["args"] == (["stat-card-uuid-1"], "rejected", "acct-1")
