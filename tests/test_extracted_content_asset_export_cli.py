from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/export_extracted_content_assets.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "export_extracted_content_assets",
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


def _report_row():
    return {
        "target_id": "vendor-acme",
        "target_mode": "vendor_retention",
        "report_type": "vendor_pressure",
        "title": "Acme report",
        "summary": "Pricing pressure dominates.",
        "sections": [{"id": "summary", "title": "Summary", "body_markdown": "Body"}],
        "reference_ids": ["r1"],
        "metadata": {
            "generation_usage": {"input_tokens": 10, "output_tokens": 5},
            "reasoning_context": {"wedge": "price_squeeze", "confidence": "high"},
        },
    }


def _blog_post_row():
    return {
        "slug": "acme-pricing-pressure",
        "title": "Acme Pricing Pressure",
        "description": "Pricing pressure dominates.",
        "topic_type": "vendor_alternative",
        "tags": ["pricing"],
        "content": "body",
        "charts": [],
        "data_context": {
            "_metadata": {
                "generation_usage": {"input_tokens": 9, "output_tokens": 4},
                "reasoning_context": {"wedge": "price_squeeze", "confidence": "high"},
            }
        },
        "llm_model": "fake-llm",
    }


def _landing_page_row():
    return {
        "campaign_name": "acme-launch",
        "persona": "VP Engineering",
        "value_prop": "Catch pressure early",
        "title": "Acme landing page",
        "slug": "acme-launch",
        "hero": {"headline": "Stop surprises"},
        "sections": [{"id": "problem", "title": "Problem", "body_markdown": "Body"}],
        "cta": {"label": "Book a demo"},
        "meta": {"title_tag": "Acme landing page"},
        "reference_ids": ["r1"],
        "metadata": {},
    }


def _sales_brief_row():
    return {
        "target_id": "vendor-acme",
        "target_mode": "vendor_retention",
        "brief_type": "pre_call",
        "title": "Acme brief",
        "headline": "Renewal pressure opens this week",
        "sections": [{"id": "context", "title": "Context", "body_markdown": "Body"}],
        "reference_ids": ["r1"],
        "metadata": {
            "generation_usage": {"input_tokens": 8, "output_tokens": 4},
            "reasoning_context": {"wedge": "support_erosion", "confidence": "medium"},
        },
    }


def _ticket_faq_row():
    return {
        "id": "faq-uuid-1",
        "status": "draft",
        "target_id": "acct_1",
        "target_mode": "support_account",
        "title": "Support FAQ",
        "markdown": "# Support FAQ\n\n## How do I reset login?",
        "items": [{"question": "How do I reset login?", "answer": "Use the reset link."}],
        "source_count": 3,
        "ticket_source_count": 2,
        "output_checks": {"uses_user_vocabulary": True, "has_action_items": True},
        "warnings": [],
        "metadata": {},
    }


@pytest.mark.asyncio
async def test_asset_export_cli_outputs_report_json(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool(rows=[_report_row()])
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
            "--asset",
            "report",
            "--account-id",
            "acct_1",
            "--target-mode",
            "vendor_retention",
            "--report-type",
            "vendor_pressure",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    query, args = pool.fetch_calls[0]
    assert exit_code == 0
    assert created_urls == ["postgres://example"]
    assert pool.closed is True
    assert "FROM reports" in query
    assert args == ("acct_1", "draft", "vendor_retention", "vendor_pressure", 20)
    assert output["count"] == 1
    assert output["rows"][0]["target_id"] == "vendor-acme"
    assert output["rows"][0]["reasoning_context_used"] is True


@pytest.mark.asyncio
async def test_asset_export_cli_outputs_blog_post_json(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool(rows=[_blog_post_row()])

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
            "--asset",
            "blog_post",
            "--account-id",
            "acct_1",
            "--topic-type",
            "vendor_alternative",
            "--limit",
            "4",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    query, args = pool.fetch_calls[0]
    assert exit_code == 0
    assert "FROM blog_posts" in query
    assert args == ("acct_1", "draft", "vendor_alternative", 4)
    assert output["rows"][0]["slug"] == "acme-pricing-pressure"
    assert output["rows"][0]["reasoning_wedge"] == "price_squeeze"


@pytest.mark.asyncio
async def test_asset_export_cli_writes_landing_page_csv(monkeypatch, tmp_path) -> None:
    cli = _load_cli_module()
    pool = _Pool(rows=[_landing_page_row()])
    output_path = tmp_path / "landing-pages.csv"

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
            "--asset",
            "landing_page",
            "--campaign-name",
            "acme-launch",
            "--slug",
            "acme-launch",
            "--format",
            "csv",
            "--output",
            str(output_path),
        ],
    )

    exit_code = await cli._main()

    assert exit_code == 0
    query, args = pool.fetch_calls[0]
    assert "FROM landing_pages" in query
    assert args == ("", "draft", "acme-launch", "acme-launch", 20)
    assert "Acme landing page" in output_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_asset_export_cli_outputs_sales_brief_json(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool(rows=[_sales_brief_row()])

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
            "--asset",
            "sales_brief",
            "--status",
            "",
            "--target-mode",
            "vendor_retention",
            "--brief-type",
            "pre_call",
            "--limit",
            "3",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    query, args = pool.fetch_calls[0]
    assert exit_code == 0
    assert "FROM sales_briefs" in query
    assert "status = " not in query
    assert args == ("", "vendor_retention", "pre_call", 3)
    assert output["rows"][0]["brief_type"] == "pre_call"
    assert output["rows"][0]["reasoning_wedge"] == "support_erosion"


@pytest.mark.asyncio
async def test_asset_export_cli_outputs_ticket_faq_json(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool(rows=[_ticket_faq_row()])

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
            "--asset",
            "faq_markdown",
            "--account-id",
            "acct_1",
            "--target-mode",
            "support_account",
            "--limit",
            "2",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    query, args = pool.fetch_calls[0]
    assert exit_code == 0
    assert "FROM ticket_faq_markdown" in query
    assert args == ("acct_1", "draft", "support_account", 2)
    assert output["rows"][0]["title"] == "Support FAQ"
    assert output["rows"][0]["passed_output_checks"] == 2


@pytest.mark.asyncio
async def test_asset_export_cli_requires_database_url(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(cli.sys, "argv", ["export", "--asset", "report"])

    with pytest.raises(SystemExit, match="Missing --database-url"):
        await cli._main()
