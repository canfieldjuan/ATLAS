from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_analytics import CampaignAnalyticsRefreshResult
from extracted_content_pipeline.campaign_postgres_analytics import (
    refresh_campaign_analytics_from_postgres,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/refresh_extracted_campaign_analytics.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "refresh_extracted_campaign_analytics",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pool:
    def __init__(self, *, refresh_error: Exception | None = None) -> None:
        self.refresh_error = refresh_error
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def execute(self, query, *args):
        self.execute_calls.append((str(query), args))
        if "REFRESH MATERIALIZED VIEW" in str(query) and self.refresh_error:
            raise self.refresh_error
        return "OK"

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_postgres_analytics_refresh_runner_refreshes_and_audits_success() -> None:
    pool = _Pool()

    result = await refresh_campaign_analytics_from_postgres(pool)

    assert result.as_dict() == {"refreshed": True, "error": None}
    assert "REFRESH MATERIALIZED VIEW CONCURRENTLY campaign_funnel_stats" in (
        pool.execute_calls[0][0]
    )
    audit_query, audit_args = pool.execute_calls[1]
    assert "INSERT INTO campaign_audit_log" in audit_query
    assert audit_args[2] == "analytics_refreshed"


@pytest.mark.asyncio
async def test_postgres_analytics_refresh_runner_audits_failure() -> None:
    pool = _Pool(refresh_error=RuntimeError("view locked"))

    result = await refresh_campaign_analytics_from_postgres(pool)

    assert result.as_dict() == {"refreshed": False, "error": "view locked"}
    assert "REFRESH MATERIALIZED VIEW" in pool.execute_calls[0][0]
    audit_query, audit_args = pool.execute_calls[1]
    assert "INSERT INTO campaign_audit_log" in audit_query
    assert audit_args[2] == "analytics_refresh_failed"


def test_analytics_cli_parses_env_database_url(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.setenv("EXTRACTED_DATABASE_URL", "postgres://example")

    args = cli._parse_args([])

    assert args.database_url == "postgres://example"
    assert args.json is False


@pytest.mark.asyncio
async def test_analytics_cli_requires_database_url(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    parse_args = cli._parse_args
    monkeypatch.setattr(cli, "_parse_args", lambda: parse_args([]))

    with pytest.raises(SystemExit, match="Missing --database-url"):
        await cli._main()


@pytest.mark.asyncio
async def test_analytics_cli_closes_pool_and_prints_json(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool()

    async def fake_create_pool(database_url):
        assert database_url == "postgres://example"
        return pool

    async def fake_refresh(received_pool):
        assert received_pool is pool
        return CampaignAnalyticsRefreshResult(refreshed=True)

    parse_args = cli._parse_args
    monkeypatch.setattr(cli, "_parse_args", lambda: parse_args([
        "--database-url",
        "postgres://example",
        "--json",
    ]))
    monkeypatch.setattr(cli, "_create_pool", fake_create_pool)
    monkeypatch.setattr(cli, "refresh_campaign_analytics_from_postgres", fake_refresh)

    exit_code = await cli._main()

    assert exit_code == 0
    assert pool.closed is True
    assert '"refreshed": true' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_analytics_cli_returns_nonzero_on_refresh_error(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool()

    async def fake_create_pool(database_url):
        return pool

    async def fake_refresh(received_pool):
        return CampaignAnalyticsRefreshResult(refreshed=False, error="view locked")

    parse_args = cli._parse_args
    monkeypatch.setattr(cli, "_parse_args", lambda: parse_args([
        "--database-url",
        "postgres://example",
    ]))
    monkeypatch.setattr(cli, "_create_pool", fake_create_pool)
    monkeypatch.setattr(cli, "refresh_campaign_analytics_from_postgres", fake_refresh)

    exit_code = await cli._main()

    assert exit_code == 1
    assert pool.closed is True
    assert "refreshed=False error=view locked" in capsys.readouterr().out
