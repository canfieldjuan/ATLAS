from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.campaign_postgres_review import (
    CampaignDraftReviewResult,
    review_campaign_drafts,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/review_extracted_campaign_drafts.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "review_extracted_campaign_drafts",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pool:
    def __init__(self, rows=None) -> None:
        self.rows = list(rows or [])
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        return self.rows

    async def close(self):
        self.closed = True


def _row(**overrides):
    row = {
        "id": "00000000-0000-0000-0000-000000000001",
        "previous_status": "draft",
        "status": "approved",
        "company_name": "Acme",
        "vendor_name": "LegacyCRM",
        "channel": "email_cold",
        "recipient_email": "buyer@example.com",
        "from_email": None,
        "metadata": {"scope": {"account_id": "acct_1"}},
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_review_campaign_drafts_approves_scoped_draft_rows() -> None:
    pool = _Pool(rows=[_row()])

    result = await review_campaign_drafts(
        pool,
        campaign_ids=["00000000-0000-0000-0000-000000000001"],
        status="approved",
        scope=TenantScope(account_id="acct_1"),
        reason="approved by customer",
        reviewed_by="ops@example.com",
    )

    query, args = pool.fetch_calls[0]
    assert "UPDATE \"b2b_campaigns\" AS campaign" in query
    assert "id = ANY($1::uuid[])" in query
    assert "status = ANY($2::text[])" in query
    assert "metadata -> 'scope' ->> 'account_id' = $3" in query
    assert "campaign.status = matched.previous_status" in query
    assert "campaign.metadata -> 'scope' ->> 'account_id' = $3" in query
    assert "status = $4" in query
    assert args[0] == ["00000000-0000-0000-0000-000000000001"]
    assert args[1] == ["draft"]
    assert args[2] == "acct_1"
    assert args[3] == "approved"
    assert json.loads(args[4]) == {
        "review_status": "approved",
        "review_reason": "approved by customer",
        "reviewed_by": "ops@example.com",
    }
    assert args[5] is None
    assert result.updated == 1
    assert result.rows[0]["status"] == "approved"
    assert result.filters["account_id"] == "acct_1"


@pytest.mark.asyncio
async def test_review_campaign_drafts_queues_with_from_email() -> None:
    pool = _Pool(rows=[_row(status="queued", from_email="sales@example.com")])

    await review_campaign_drafts(
        pool,
        campaign_ids=["00000000-0000-0000-0000-000000000001"],
        status="queued",
        from_statuses=("draft", "approved"),
        from_email="sales@example.com",
    )

    query, args = pool.fetch_calls[0]
    assert "approved_at = CASE" in query
    assert "from_email = COALESCE($5::text, campaign.from_email)" in query
    assert args == (
        ["00000000-0000-0000-0000-000000000001"],
        ["draft", "approved"],
        "queued",
        "{\"review_status\":\"queued\"}",
        "sales@example.com",
    )


@pytest.mark.asyncio
async def test_review_campaign_drafts_only_stamps_from_email_when_queueing() -> None:
    pool = _Pool(rows=[_row(status="approved", from_email="existing@example.com")])

    await review_campaign_drafts(
        pool,
        campaign_ids=["00000000-0000-0000-0000-000000000001"],
        status="approved",
        from_email="sales@example.com",
    )

    query, args = pool.fetch_calls[0]
    assert "from_email = COALESCE($5::text, campaign.from_email)" in query
    assert args == (
        ["00000000-0000-0000-0000-000000000001"],
        ["draft"],
        "approved",
        "{\"review_status\":\"approved\"}",
        None,
    )


@pytest.mark.asyncio
async def test_review_campaign_drafts_dry_run_selects_without_update() -> None:
    pool = _Pool(rows=[_row(status="draft")])

    result = await review_campaign_drafts(
        pool,
        campaign_ids=["00000000-0000-0000-0000-000000000001"],
        status="queued",
        dry_run=True,
    )

    query, args = pool.fetch_calls[0]
    assert "SELECT" in query
    assert "UPDATE" not in query
    assert args == (["00000000-0000-0000-0000-000000000001"], ["draft"])
    assert result.dry_run is True
    assert result.status == "queued"


@pytest.mark.asyncio
async def test_review_campaign_drafts_can_disable_source_status_guard() -> None:
    pool = _Pool(rows=[_row(status="expired")])

    await review_campaign_drafts(
        pool,
        campaign_ids=["00000000-0000-0000-0000-000000000001"],
        status="expired",
        from_statuses=(),
    )

    query, args = pool.fetch_calls[0]
    assert "status = ANY" not in query
    assert args[0] == ["00000000-0000-0000-0000-000000000001"]
    assert args[1] == "expired"


@pytest.mark.asyncio
async def test_review_campaign_drafts_rejects_empty_ids() -> None:
    with pytest.raises(ValueError, match="at least one campaign id"):
        await review_campaign_drafts(_Pool(), campaign_ids=[])


@pytest.mark.asyncio
async def test_review_campaign_drafts_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="unsupported review status"):
        await review_campaign_drafts(
            _Pool(),
            campaign_ids=["00000000-0000-0000-0000-000000000001"],
            status="sent",
        )


@pytest.mark.asyncio
async def test_review_campaign_drafts_rejects_unsafe_table_name() -> None:
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        await review_campaign_drafts(
            _Pool(),
            campaign_ids=["00000000-0000-0000-0000-000000000001"],
            campaign_table="bad-table",
        )


def test_campaign_draft_review_result_as_dict() -> None:
    result = CampaignDraftReviewResult(
        rows=(_row(status="queued"),),
        requested_ids=("00000000-0000-0000-0000-000000000001",),
        status="queued",
        dry_run=False,
        filters={"from_statuses": ("draft",)},
    )

    data = result.as_dict()

    assert data["updated"] == 1
    assert data["requested"] == 1
    assert data["rows"][0]["status"] == "queued"


@pytest.mark.asyncio
async def test_review_campaign_drafts_cli_outputs_json(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool(rows=[_row(status="queued")])
    created_urls: list[str] = []

    async def create_pool(database_url):
        created_urls.append(database_url)
        return pool

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "review",
            "00000000-0000-0000-0000-000000000001,00000000-0000-0000-0000-000000000002",
            "--database-url",
            "postgres://example",
            "--account-id",
            "acct_1",
            "--status",
            "queued",
            "--from-status",
            "draft,approved",
            "--from-email",
            "sales@example.com",
            "--json",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert created_urls == ["postgres://example"]
    assert pool.closed is True
    assert output["status"] == "queued"
    assert output["requested"] == 2


@pytest.mark.asyncio
async def test_review_campaign_drafts_cli_requires_database_url(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["review", "00000000-0000-0000-0000-000000000001"],
    )

    with pytest.raises(SystemExit, match="Missing --database-url"):
        await cli._main()
