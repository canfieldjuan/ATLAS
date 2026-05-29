from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback import MacroWritebackMapping


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "reconcile_content_ops_faq_macro_writebacks.py"
SPEC = importlib.util.spec_from_file_location(
    "reconcile_content_ops_faq_macro_writebacks",
    SCRIPT_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
SPEC.loader.exec_module(cli)


class _Pool:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.fetch_calls: list[dict[str, Any]] = []
        self.closed = False

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append({"query": query, "args": args})
        return self.rows

    async def close(self) -> None:
        self.closed = True


class _Result:
    def __init__(self, mapping: MacroWritebackMapping, status: str, error: str = "") -> None:
        self.mapping = mapping
        self.status = status
        self.error = error

    def as_dict(self) -> dict[str, Any]:
        return {
            "mapping": self.mapping.as_dict(),
            "status": self.status,
            "external_id": self.mapping.external_id,
            "error": self.error,
        }


class _Provider:
    def __init__(self, status: str = "reconciled", error: str = "") -> None:
        self.status = status
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def reconcile_pending_mapping(
        self,
        mapping: MacroWritebackMapping,
        *,
        scope: TenantScope,
    ) -> _Result:
        self.calls.append({"mapping": mapping, "scope": scope})
        return _Result(mapping, self.status, self.error)


def _args(**overrides: Any) -> argparse.Namespace:
    values = {
        "account_id": "acct-1",
        "limit": 10,
        "execute": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "platform": "zendesk",
        "faq_draft_id": "11111111-1111-1111-1111-111111111111",
        "faq_item_id": "faq-draft-1:item-1",
        "external_id": "",
        "external_url": "",
        "publish_status": "pending",
        "metadata": json.dumps({"title": "Why was I charged twice?"}),
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_reconcile_cli_dry_run_lists_pending_rows_without_provider_call() -> None:
    pool = _Pool([_row()])
    provider = _Provider()

    payload = await cli.reconcile_pending_macro_writebacks(
        _args(execute=False),
        pool,
        provider=provider,
    )

    assert payload["execute"] is False
    assert payload["pending_count"] == 1
    assert payload["result_counts"] == {"dry_run": 1}
    assert payload["results"][0]["status"] == "dry_run"
    assert payload["results"][0]["mapping"]["metadata"] == {
        "title": "Why was I charged twice?"
    }
    assert provider.calls == []
    assert pool.fetch_calls[0]["args"] == ("acct-1", "zendesk", 10)


@pytest.mark.asyncio
async def test_reconcile_cli_execute_surfaces_ambiguous_pending_rows() -> None:
    pool = _Pool([_row(), _row(faq_item_id="faq-draft-1:item-2")])
    provider = _Provider(
        status="pending",
        error="zendesk_macro_mapping_ambiguous_reconcile",
    )

    payload = await cli.reconcile_pending_macro_writebacks(
        _args(execute=True, limit=2),
        pool,
        provider=provider,
    )

    assert payload["execute"] is True
    assert payload["pending_count"] == 2
    assert payload["result_counts"] == {"pending": 2}
    assert [call["scope"].account_id for call in provider.calls] == ["acct-1", "acct-1"]
    assert {
        result["error"] for result in payload["results"]
    } == {"zendesk_macro_mapping_ambiguous_reconcile"}


@pytest.mark.asyncio
async def test_reconcile_cli_main_writes_output_and_closes_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pool = _Pool([_row()])
    output = tmp_path / "reconcile.json"

    async def create_pool(database_url: str) -> _Pool:
        assert database_url == "postgres://example"
        return pool

    monkeypatch.setattr(cli, "_create_pool", create_pool)

    code = await cli._main([
        "--database-url",
        "postgres://example",
        "--account-id",
        "acct-1",
        "--output",
        str(output),
    ])

    assert code == 0
    assert pool.closed is True
    payload = json.loads(output.read_text())
    assert payload["result_counts"] == {"dry_run": 1}
