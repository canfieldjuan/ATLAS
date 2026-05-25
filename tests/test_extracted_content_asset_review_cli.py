from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/review_extracted_content_assets.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "review_extracted_content_assets",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pool:
    def __init__(self, result: str = "UPDATE 1") -> None:
        self.result = result
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        if "UPDATE ticket_faq_markdown" not in str(query):
            return []
        if self.result == "UPDATE 0":
            return []
        return [{
            "id": args[0],
            "target_id": "support-account-1",
            "target_mode": "support_account",
            "title": "Support FAQ",
            "markdown": "# Support FAQ",
            "items": json.dumps([{
                "question": "How do I reset login?",
                "answer": "Use the reset link.",
                "source_ids": ["ticket-1"],
                "ticket_count": 1,
            }]),
            "source_count": 1,
            "ticket_source_count": 1,
            "output_checks": "{}",
            "warnings": "[]",
            "metadata": json.dumps({"corpus_id": "support-account-1"}),
            "status": args[1],
        }]

    async def execute(self, query, *args):
        self.execute_calls.append((str(query), args))
        return self.result

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_asset_review_cli_updates_report_status(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool()
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
            "--database-url",
            "postgres://example",
            "--asset",
            "report",
            "--id",
            "report-uuid-1",
            "--status",
            "approved",
            "--account-id",
            "acct_1",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    query, args = pool.execute_calls[0]
    assert exit_code == 0
    assert created_urls == ["postgres://example"]
    assert pool.closed is True
    assert "UPDATE reports" in query
    assert args == ("report-uuid-1", "approved", "acct_1")
    assert output == {
        "account_id": "acct_1",
        "asset": "report",
        "id": "report-uuid-1",
        "status": "approved",
        "updated": True,
    }


@pytest.mark.asyncio
async def test_asset_review_cli_updates_blog_post_status(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool()

    async def create_pool(database_url):
        return pool

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "review",
            "--database-url",
            "postgres://example",
            "--asset",
            "blog_post",
            "--id",
            "blog-post-uuid-1",
            "--status",
            "approved",
            "--account-id",
            "acct_1",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    query, args = pool.execute_calls[0]
    assert exit_code == 0
    assert "UPDATE blog_posts" in query
    assert args == ("blog-post-uuid-1", "approved", "acct_1")
    assert output == {
        "account_id": "acct_1",
        "asset": "blog_post",
        "id": "blog-post-uuid-1",
        "status": "approved",
        "updated": True,
    }


@pytest.mark.asyncio
async def test_asset_review_cli_returns_nonzero_on_landing_page_miss(
    monkeypatch,
    capsys,
) -> None:
    cli = _load_cli_module()
    pool = _Pool(result="UPDATE 0")

    async def create_pool(database_url):
        return pool

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "review",
            "--database-url",
            "postgres://example",
            "--asset",
            "landing_page",
            "--id",
            "landing-page-uuid-1",
            "--status",
            "rejected",
            "--account-id",
            "acct_1",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    query, args = pool.execute_calls[0]
    assert exit_code == 1
    assert pool.closed is True
    assert "UPDATE landing_pages" in query
    assert args == ("landing-page-uuid-1", "rejected", "acct_1")
    assert output["updated"] is False


@pytest.mark.asyncio
async def test_asset_review_cli_updates_sales_brief_status(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool()

    async def create_pool(database_url):
        return pool

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "review",
            "--database-url",
            "postgres://example",
            "--asset",
            "sales_brief",
            "--id",
            "sales-brief-uuid-1",
            "--status",
            "queued",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    query, args = pool.execute_calls[0]
    assert exit_code == 0
    assert "UPDATE sales_briefs" in query
    assert args == ("sales-brief-uuid-1", "queued", "")
    assert output["account_id"] is None
    assert output["updated"] is True


@pytest.mark.asyncio
async def test_asset_review_cli_updates_ticket_faq_status(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool()

    async def create_pool(database_url):
        return pool

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "review",
            "--database-url",
            "postgres://example",
            "--asset",
            "faq_markdown",
            "--id",
            "faq-uuid-1",
            "--status",
            "approved",
            "--account-id",
            "acct_1",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    query, args = pool.fetch_calls[0]
    assert exit_code == 0
    assert "UPDATE ticket_faq_markdown" in query
    assert args == ("faq-uuid-1", "approved", "acct_1")
    assert "INSERT INTO ticket_faq_search_documents" in pool.execute_calls[1][0]
    assert output == {
        "account_id": "acct_1",
        "asset": "faq_markdown",
        "id": "faq-uuid-1",
        "status": "approved",
        "updated": True,
    }


@pytest.mark.asyncio
async def test_asset_review_cli_allows_host_defined_status(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool()

    async def create_pool(database_url):
        return pool

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "review",
            "--database-url",
            "postgres://example",
            "--asset",
            "report",
            "--id",
            "report-uuid-1",
            "--status",
            "published",
            "--account-id",
            "acct_1",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    query, args = pool.execute_calls[0]
    assert exit_code == 0
    assert "UPDATE reports" in query
    assert args == ("report-uuid-1", "published", "acct_1")
    assert output["status"] == "published"


@pytest.mark.asyncio
async def test_asset_review_cli_rejects_empty_status(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "review",
            "--database-url",
            "postgres://example",
            "--asset",
            "report",
            "--id",
            "report-uuid-1",
            "--status",
            "",
        ],
    )

    with pytest.raises(SystemExit, match="--status must be a non-empty string"):
        await cli._main()


@pytest.mark.asyncio
async def test_asset_review_cli_requires_database_url(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "review",
            "--asset",
            "report",
            "--id",
            "report-uuid-1",
            "--status",
            "approved",
        ],
    )

    with pytest.raises(SystemExit, match="Missing --database-url"):
        await cli._main()
