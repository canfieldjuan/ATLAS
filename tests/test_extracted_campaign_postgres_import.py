from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.campaign_postgres_import import (
    import_campaign_opportunities,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/load_extracted_campaign_opportunities.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "load_extracted_campaign_opportunities",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pool:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def execute(self, query, *args):
        self.executed.append((str(query), args))
        return "EXECUTE"

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_import_campaign_opportunities_inserts_normalized_rows() -> None:
    pool = _Pool()

    result = await import_campaign_opportunities(
        pool,
        [
            {
                "company": "Acme",
                "vendor": "LegacyCRM",
                "email": "buyer@example.com",
                "opportunity_score": "88",
                "custom_segment": "enterprise",
            }
        ],
        scope=TenantScope(account_id="acct_1"),
        target_mode="vendor_retention",
    )

    assert result.inserted == 1
    assert result.skipped == 0
    assert result.target_ids == ("buyer@example.com",)
    assert len(pool.executed) == 1
    query, args = pool.executed[0]
    assert "INSERT INTO \"campaign_opportunities\"" in query
    assert args[0] == "acct_1"
    assert args[1] == "buyer@example.com"
    assert args[3] == "Acme"
    assert args[4] == "LegacyCRM"
    assert json.loads(args[13])["custom_segment"] == "enterprise"


@pytest.mark.asyncio
async def test_import_campaign_opportunities_skips_rows_without_target_id() -> None:
    pool = _Pool()

    result = await import_campaign_opportunities(
        pool,
        [{"opportunity_score": 77}],
        target_mode="vendor_retention",
    )

    assert result.inserted == 0
    assert result.skipped == 1
    assert pool.executed == []
    assert any(warning.code == "missing_target_id" for warning in result.warnings)


@pytest.mark.asyncio
async def test_import_campaign_opportunities_dry_run_does_not_touch_database() -> None:
    pool = _Pool()

    result = await import_campaign_opportunities(
        pool,
        [{"company": "Acme", "vendor": "LegacyCRM"}],
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.inserted == 1
    assert pool.executed == []


@pytest.mark.asyncio
async def test_import_campaign_opportunities_replace_existing_deletes_matching_targets() -> None:
    pool = _Pool()

    result = await import_campaign_opportunities(
        pool,
        [{"company": "Acme", "vendor": "LegacyCRM"}],
        scope=TenantScope(account_id="acct_1"),
        target_mode="vendor_retention",
        replace_existing=True,
    )

    assert result.inserted == 1
    assert len(pool.executed) == 2
    delete_query, delete_args = pool.executed[0]
    assert "DELETE FROM \"campaign_opportunities\"" in delete_query
    assert delete_args == ("acct_1", "vendor_retention", ["Acme"])


@pytest.mark.asyncio
async def test_import_campaign_opportunities_rejects_unsafe_table_name() -> None:
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        await import_campaign_opportunities(
            _Pool(),
            [{"company": "Acme"}],
            opportunity_table="campaign-opportunities",
        )


@pytest.mark.asyncio
async def test_opportunity_import_cli_dry_run_outputs_json(monkeypatch, capsys, tmp_path) -> None:
    cli = _load_cli_module()
    data_path = tmp_path / "opportunities.json"
    data_path.write_text(
        json.dumps({"opportunities": [{"company": "Acme", "vendor": "LegacyCRM"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["load", str(data_path), "--dry-run", "--json"],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["dry_run"] is True
    assert output["inserted"] == 1
    assert output["target_ids"] == ["Acme"]


@pytest.mark.asyncio
async def test_opportunity_import_cli_requires_database_url(monkeypatch, tmp_path) -> None:
    cli = _load_cli_module()
    data_path = tmp_path / "opportunities.json"
    data_path.write_text(json.dumps([{"company": "Acme"}]), encoding="utf-8")
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(cli.sys, "argv", ["load", str(data_path)])

    with pytest.raises(SystemExit, match="Missing --database-url"):
        await cli._main()


@pytest.mark.asyncio
async def test_opportunity_import_cli_wires_pool_and_closes(monkeypatch, capsys, tmp_path) -> None:
    cli = _load_cli_module()
    data_path = tmp_path / "opportunities.csv"
    data_path.write_text("company,vendor\nAcme,LegacyCRM\n", encoding="utf-8")
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
            "load",
            str(data_path),
            "--database-url",
            "postgres://example",
            "--format",
            "csv",
            "--account-id",
            "acct_1",
            "--replace-existing",
            "--json",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert created_urls == ["postgres://example"]
    assert pool.closed is True
    assert output["inserted"] == 1
    assert output["replace_existing"] is True
