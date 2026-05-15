"""Tests for the campaign reasoning context list/export CLI."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.campaign_reasoning_postgres import (
    CampaignReasoningContextListResult,
    PostgresCampaignReasoningContextRepository,
)


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "list_extracted_campaign_reasoning_contexts.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "list_extracted_campaign_reasoning_contexts",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
list_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(list_cli)


class _Pool:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.fetch_calls: list[dict[str, Any]] = []
        self.closed = False

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append({"query": query, "args": args})
        return self.rows

    async def close(self) -> None:
        self.closed = True


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "id": "ctx-1",
        "account_id": "acct-1",
        "target_mode": "vendor_retention",
        "selectors": ["opp-1", "Acme"],
        "selector_key": "abc123",
        "payload": {"top_theses": [{"summary": "Renewal pressure"}]},
        "updated_at": datetime(2026, 5, 15, tzinfo=timezone.utc),
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_repository_list_contexts_filters_and_serializes_rows() -> None:
    """Operators can inventory rows by tenant, mode, and selector."""

    pool = _Pool(rows=[_row(payload=json.dumps({"confidence": "high"}))])
    repository = PostgresCampaignReasoningContextRepository(pool=pool)

    result = await repository.list_contexts(
        scope=TenantScope(account_id="acct-1"),
        target_mode="Vendor_Retention",
        selectors=("Acme",),
        limit=5,
    )

    query = pool.fetch_calls[0]["query"]
    args = pool.fetch_calls[0]["args"]
    assert 'FROM "campaign_reasoning_contexts"' in query
    assert "account_id = $1" in query
    assert "target_mode = $2" in query
    assert "selectors && $3::text[]" in query
    assert args == ("acct-1", "vendor_retention", ["Acme", "acme"], 5)
    assert result.filters == {
        "account_id": "acct-1",
        "target_mode": "vendor_retention",
        "selectors": ("Acme", "acme"),
    }
    assert result.rows[0]["payload"] == {"confidence": "high"}
    assert result.rows[0]["updated_at"] == "2026-05-15 00:00:00+00:00"


@pytest.mark.asyncio
async def test_repository_list_contexts_preserves_non_object_payloads() -> None:
    """Inventory/export should surface unexpected stored payload shapes."""

    pool = _Pool(rows=[_row(payload='["unexpected", "shape"]')])
    repository = PostgresCampaignReasoningContextRepository(pool=pool)

    result = await repository.list_contexts(limit=1)

    assert result.rows[0]["payload"] == ["unexpected", "shape"]


@pytest.mark.asyncio
async def test_repository_list_contexts_allows_unfiltered_inventory() -> None:
    """No filter args means list the newest rows across accounts/modes."""

    pool = _Pool(rows=[])
    repository = PostgresCampaignReasoningContextRepository(pool=pool)

    result = await repository.list_contexts(limit=0)

    assert pool.fetch_calls[0]["args"] == (None, None, None, 0)
    assert result.as_dict()["rows"] == []
    assert result.limit == 0


@pytest.mark.asyncio
async def test_repository_list_contexts_rejects_negative_limit() -> None:
    repository = PostgresCampaignReasoningContextRepository(pool=_Pool())

    with pytest.raises(ValueError, match="limit must be non-negative"):
        await repository.list_contexts(limit=-1)


@pytest.mark.asyncio
async def test_repository_list_contexts_rejects_unsafe_table_name() -> None:
    repository = PostgresCampaignReasoningContextRepository(
        pool=_Pool(),
        table="bad-table",
    )

    with pytest.raises(ValueError, match="invalid SQL identifier"):
        await repository.list_contexts()


def test_context_list_result_renders_csv() -> None:
    result = CampaignReasoningContextListResult(
        rows=(_row(),),
        limit=1,
        filters={"account_id": "acct-1"},
    )

    csv_text = result.as_csv()

    assert "id,account_id,target_mode,selectors,selector_key,updated_at,payload" in csv_text
    assert "ctx-1" in csv_text
    assert "{\"\"top_theses\"\":[{\"\"summary\"\":\"\"Renewal pressure\"\"}]}" in csv_text


@pytest.mark.asyncio
async def test_cli_outputs_json_and_closes_pool(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    pool = _Pool(rows=[_row()])
    created_urls: list[str] = []

    async def create_pool(database_url: str) -> _Pool:
        created_urls.append(database_url)
        return pool

    monkeypatch.setattr(list_cli, "_create_pool", create_pool)
    exit_code = await list_cli._main_from_args([
        "--database-url",
        "postgres://example",
        "--account-id",
        "acct-1",
        "--selector",
        "Acme",
        "--limit",
        "1",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert created_urls == ["postgres://example"]
    assert pool.closed is True
    assert payload["count"] == 1
    assert payload["rows"][0]["id"] == "ctx-1"


@pytest.mark.asyncio
async def test_cli_outputs_csv(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    pool = _Pool(rows=[_row()])

    async def create_pool(database_url: str) -> _Pool:
        return pool

    monkeypatch.setattr(list_cli, "_create_pool", create_pool)
    exit_code = await list_cli._main_from_args([
        "--database-url",
        "postgres://example",
        "--format",
        "csv",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "id,account_id,target_mode,selectors,selector_key,updated_at,payload" in captured.out
    assert "ctx-1" in captured.out
    assert pool.closed is True
