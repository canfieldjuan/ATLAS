from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.campaign_postgres_export import (
    CampaignDraftExportResult,
    list_campaign_drafts,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/export_extracted_campaign_drafts.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "export_extracted_campaign_drafts",
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
        "id": "campaign_1",
        "company_name": "Acme",
        "vendor_name": "LegacyCRM",
        "target_mode": "vendor_retention",
        "channel": "email_cold",
        "status": "draft",
        "recipient_email": "buyer@example.com",
        "subject": "Acme renewal plan",
        "body": "Body",
        "cta": "Review plan",
        "llm_model": "offline",
        "created_at": datetime(2026, 5, 4, tzinfo=timezone.utc),
        "metadata": {"scope": {"account_id": "acct_1"}},
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_list_campaign_drafts_filters_review_rows() -> None:
    pool = _Pool(rows=[_row()])

    result = await list_campaign_drafts(
        pool,
        scope=TenantScope(account_id="acct_1"),
        statuses=("draft", "approved"),
        target_mode="vendor_retention",
        channel="email_cold",
        vendor_name="LegacyCRM",
        company_name="Acme",
        limit=7,
    )

    query, args = pool.fetch_calls[0]
    assert "FROM \"b2b_campaigns\"" in query
    assert "status = ANY($1::text[])" in query
    assert "metadata -> 'scope' ->> 'account_id' = $2" in query
    assert "target_mode = $3" in query
    assert "channel = $4" in query
    assert "LOWER(vendor_name) = LOWER($5)" in query
    assert "LOWER(company_name) = LOWER($6)" in query
    assert args == (
        ["draft", "approved"],
        "acct_1",
        "vendor_retention",
        "email_cold",
        "LegacyCRM",
        "Acme",
        7,
    )
    assert result.rows[0]["created_at"] == "2026-05-04 00:00:00+00:00"
    assert result.filters["account_id"] == "acct_1"


@pytest.mark.asyncio
async def test_list_campaign_drafts_all_statuses_when_statuses_empty() -> None:
    pool = _Pool(rows=[_row()])

    await list_campaign_drafts(pool, statuses=(), limit=2)

    query, args = pool.fetch_calls[0]
    assert "status = ANY" not in query
    assert args == (2,)


@pytest.mark.asyncio
async def test_list_campaign_drafts_rejects_unsafe_table_name() -> None:
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        await list_campaign_drafts(_Pool(), campaign_table="bad-table")


def test_campaign_draft_export_result_renders_csv() -> None:
    result = CampaignDraftExportResult(
        rows=(_row(metadata={"scope": {"account_id": "acct_1"}, "tags": ["renewal"]}),),
        limit=1,
        filters={"statuses": ("draft",)},
    )

    csv_text = result.as_csv()

    assert "company_name,vendor_name" in csv_text
    assert "Acme" in csv_text
    assert "{\"\"scope\"\":{\"\"account_id\"\":\"\"acct_1\"\"}" in csv_text


@pytest.mark.asyncio
async def test_campaign_draft_export_cli_outputs_json(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool(rows=[_row()])
    created_urls: list[str] = []

    async def create_pool(database_url):
        created_urls.append(database_url)
        return pool

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "export",
            "--database-url",
            "postgres://example",
            "--account-id",
            "acct_1",
            "--status",
            "draft,approved",
            "--target-mode",
            "vendor_retention",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert created_urls == ["postgres://example"]
    assert pool.closed is True
    assert output["count"] == 1
    assert output["rows"][0]["company_name"] == "Acme"


@pytest.mark.asyncio
async def test_campaign_draft_export_cli_writes_csv(monkeypatch, tmp_path) -> None:
    cli = _load_cli_module()
    pool = _Pool(rows=[_row()])
    output_path = tmp_path / "drafts.csv"

    async def create_pool(database_url):
        return pool

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "export",
            "--database-url",
            "postgres://example",
            "--format",
            "csv",
            "--output",
            str(output_path),
        ],
    )

    exit_code = await cli._main()

    assert exit_code == 0
    assert "Acme" in output_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_campaign_draft_export_cli_requires_database_url(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(cli.sys, "argv", ["export"])

    with pytest.raises(SystemExit, match="Missing --database-url"):
        await cli._main()
