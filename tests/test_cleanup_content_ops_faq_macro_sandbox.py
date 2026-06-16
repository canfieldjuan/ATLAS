from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback_zendesk import (
    ZENDESK_PLATFORM,
    ZendeskMacroCredentials,
    ZendeskMacroTransportError,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "cleanup_content_ops_faq_macro_sandbox.py"
SPEC = importlib.util.spec_from_file_location(
    "cleanup_content_ops_faq_macro_sandbox",
    SCRIPT_PATH,
)
cleanup = importlib.util.module_from_spec(SPEC)
assert SPEC is not None
assert SPEC.loader is not None
sys.modules[SPEC.name] = cleanup
SPEC.loader.exec_module(cleanup)


ACCOUNT_ID = "acct-1"
FAQ_ID = "faq-1"
EXPECTED_BASE_URL = "https://sandbox.zendesk.com"


class _Pool:
    def __init__(
        self,
        *,
        fetchrow_rows: list[Any] | None = None,
        fetch_rows: list[list[Any]] | None = None,
        fetchval_rows: list[Any] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.fetchrow_rows = list(fetchrow_rows or [])
        self.fetch_rows = list(fetch_rows or [])
        self.fetchval_rows = list(fetchval_rows or [])
        self.fetchrow_calls: list[dict[str, Any]] = []
        self.fetch_calls: list[dict[str, Any]] = []
        self.fetchval_calls: list[dict[str, Any]] = []
        self.events = events
        self.closed = False

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.fetchrow_calls.append({"query": query, "args": args})
        return self.fetchrow_rows.pop(0) if self.fetchrow_rows else None

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.fetch_calls.append({"query": query, "args": args})
        if "DELETE FROM ticket_faq_markdown" in query and self.events is not None:
            self.events.append("local_delete")
        rows = self.fetch_rows.pop(0) if self.fetch_rows else []
        if "platform = $3" in query:
            return [row for row in rows if row.get("platform") == args[2]]
        return rows

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchval_calls.append({"query": query, "args": args})
        return self.fetchval_rows.pop(0) if self.fetchval_rows else 0

    async def close(self) -> None:
        self.closed = True


class _CredentialsProvider:
    def __init__(self, credentials: ZendeskMacroCredentials | None) -> None:
        self.credentials = credentials
        self.calls: list[TenantScope] = []

    async def credentials_for_scope(
        self,
        scope: TenantScope,
    ) -> ZendeskMacroCredentials | None:
        self.calls.append(scope)
        return self.credentials


class _Transport:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.error = error
        self.events = events
        self.calls: list[dict[str, Any]] = []

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append({
            "method": method,
            "url": url,
            "headers": dict(headers),
            "json": dict(json),
        })
        if self.events is not None:
            self.events.append("external_delete")
        if self.error is not None:
            raise self.error
        return {}


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "database_url": "postgres://example",
        "account_id": ACCOUNT_ID,
        "faq_id": FAQ_ID,
        "expected_zendesk_base_url": "",
        "execute": False,
        "confirm_live_zendesk_delete": False,
        "output": None,
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _credentials(*, subdomain: str = "sandbox") -> ZendeskMacroCredentials:
    return ZendeskMacroCredentials(
        subdomain=subdomain,
        email="agent@example.com",
        api_token="secret-token",
    )


def _pool_with_snapshot(
    *,
    events: list[str] | None = None,
    mappings: list[dict[str, Any]] | None = None,
    external_id: str = "macro-1",
    external_url: str | None = None,
    publish_status: str = "published",
) -> _Pool:
    if mappings is None:
        mappings = [
            {
                "platform": ZENDESK_PLATFORM,
                "faq_item_id": "item-1",
                "external_id": external_id,
                "external_url": (
                    external_url
                    if external_url is not None
                    else f"{EXPECTED_BASE_URL}/api/v2/macros/{external_id}"
                ),
                "publish_status": publish_status,
            }
        ]
    return _Pool(
        fetchrow_rows=[{"id": FAQ_ID, "status": "draft"}],
        fetch_rows=[mappings],
        fetchval_rows=[2, 3],
        events=events,
    )


@pytest.mark.asyncio
async def test_preview_reports_scoped_counts_without_deletes() -> None:
    pool = _pool_with_snapshot()
    transport = _Transport()

    code, payload = await cleanup.cleanup_sandbox_macro_writeback(
        _args(),
        pool,
        credentials_provider=_CredentialsProvider(_credentials()),
        transport=transport,
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["execute"] is False
    assert payload["stage"] == "preview"
    assert payload["account_id"] == ACCOUNT_ID
    assert payload["faq_id"] == FAQ_ID
    assert payload["snapshot"]["found"] is True
    assert payload["snapshot"]["external_ids"] == ["macro-1"]
    assert payload["snapshot"]["missing_external_id_count"] == 0
    assert payload["snapshot"]["publish_attempt_count"] == 2
    assert payload["snapshot"]["search_document_count"] == 3
    mapping_query = pool.fetch_calls[0]
    assert "platform = $3" in mapping_query["query"]
    assert mapping_query["args"] == (ACCOUNT_ID, FAQ_ID, ZENDESK_PLATFORM)
    assert transport.calls == []
    assert all("DELETE FROM ticket_faq_markdown" not in call["query"] for call in pool.fetch_calls)


@pytest.mark.asyncio
async def test_execute_missing_confirm_skips_before_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _raise_create_pool(database_url: str) -> Any:
        raise AssertionError(f"pool opened unexpectedly: {database_url}")

    monkeypatch.setattr(cleanup, "_create_pool", _raise_create_pool)

    code = await cleanup._main([
        "--database-url",
        "postgres://example",
        "--account-id",
        ACCOUNT_ID,
        "--faq-id",
        FAQ_ID,
        "--execute",
        "--expected-zendesk-base-url",
        EXPECTED_BASE_URL,
    ])

    assert code == cleanup.SKIPPED_EXIT


@pytest.mark.asyncio
async def test_execute_missing_expected_url_skips_before_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise_create_pool(database_url: str) -> Any:
        raise AssertionError(f"pool opened unexpectedly: {database_url}")

    monkeypatch.setattr(cleanup, "_create_pool", _raise_create_pool)

    code = await cleanup._main([
        "--database-url",
        "postgres://example",
        "--account-id",
        ACCOUNT_ID,
        "--faq-id",
        FAQ_ID,
        "--execute",
        "--confirm-live-zendesk-delete",
    ])

    assert code == cleanup.SKIPPED_EXIT


@pytest.mark.asyncio
async def test_execute_rejects_unexpected_zendesk_base_url_before_deletes() -> None:
    pool = _pool_with_snapshot()
    transport = _Transport()

    code, payload = await cleanup.cleanup_sandbox_macro_writeback(
        _args(
            execute=True,
            confirm_live_zendesk_delete=True,
            expected_zendesk_base_url="https://other.zendesk.com",
        ),
        pool,
        credentials_provider=_CredentialsProvider(_credentials()),
        transport=transport,
    )

    assert code == cleanup.SKIPPED_EXIT
    assert payload["not_run_reason"] == "unexpected_zendesk_base_url"
    assert payload["observed_zendesk_base_url"] == EXPECTED_BASE_URL
    assert transport.calls == []
    assert all("DELETE FROM ticket_faq_markdown" not in call["query"] for call in pool.fetch_calls)


@pytest.mark.asyncio
async def test_execute_missing_mapping_external_id_leaves_local_rows() -> None:
    pool = _pool_with_snapshot(external_id="", publish_status="pending")
    credentials_provider = _CredentialsProvider(_credentials())
    transport = _Transport()

    code, payload = await cleanup.cleanup_sandbox_macro_writeback(
        _args(
            execute=True,
            confirm_live_zendesk_delete=True,
            expected_zendesk_base_url=EXPECTED_BASE_URL,
        ),
        pool,
        credentials_provider=credentials_provider,
        transport=transport,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["stage"] == "mapping_preflight"
    assert payload["errors"] == ["macro_mapping_missing_external_id"]
    assert payload["snapshot"]["macro_mapping_count"] == 1
    assert payload["snapshot"]["external_ids"] == []
    assert payload["snapshot"]["missing_external_id_count"] == 1
    assert credentials_provider.calls == []
    assert transport.calls == []
    assert all("DELETE FROM ticket_faq_markdown" not in call["query"] for call in pool.fetch_calls)


@pytest.mark.asyncio
async def test_snapshot_ignores_non_zendesk_mappings() -> None:
    pool = _pool_with_snapshot(
        mappings=[
            {
                "platform": "freshdesk",
                "faq_item_id": "item-1",
                "external_id": "",
                "external_url": "https://freshdesk.example/macros/1",
                "publish_status": "pending",
            },
            {
                "platform": ZENDESK_PLATFORM,
                "faq_item_id": "item-1",
                "external_id": "macro-1",
                "external_url": f"{EXPECTED_BASE_URL}/api/v2/macros/macro-1",
                "publish_status": "published",
            },
        ]
    )

    code, payload = await cleanup.cleanup_sandbox_macro_writeback(
        _args(),
        pool,
        credentials_provider=_CredentialsProvider(_credentials()),
        transport=_Transport(),
    )

    assert code == 0
    assert payload["snapshot"]["macro_mapping_count"] == 1
    assert payload["snapshot"]["external_ids"] == ["macro-1"]
    assert payload["snapshot"]["missing_external_id_count"] == 0


@pytest.mark.asyncio
async def test_execute_rejects_mapping_url_from_unexpected_zendesk_base() -> None:
    pool = _pool_with_snapshot(
        external_url="https://other.zendesk.com/api/v2/macros/macro-1",
    )
    transport = _Transport()

    code, payload = await cleanup.cleanup_sandbox_macro_writeback(
        _args(
            execute=True,
            confirm_live_zendesk_delete=True,
            expected_zendesk_base_url=EXPECTED_BASE_URL,
        ),
        pool,
        credentials_provider=_CredentialsProvider(_credentials()),
        transport=transport,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["stage"] == "mapping_preflight"
    assert payload["errors"] == ["zendesk_macro_url_base_mismatch"]
    assert payload["unexpected_external_urls"] == [
        {
            "external_id": "macro-1",
            "external_url": "https://other.zendesk.com/api/v2/macros/macro-1",
        }
    ]
    assert transport.calls == []
    assert all("DELETE FROM ticket_faq_markdown" not in call["query"] for call in pool.fetch_calls)


@pytest.mark.asyncio
async def test_execute_deletes_external_macro_before_local_faq_row() -> None:
    events: list[str] = []
    pool = _pool_with_snapshot(events=events)
    pool.fetch_rows.append([{"id": FAQ_ID}])
    transport = _Transport(events=events)

    code, payload = await cleanup.cleanup_sandbox_macro_writeback(
        _args(
            execute=True,
            confirm_live_zendesk_delete=True,
            expected_zendesk_base_url=EXPECTED_BASE_URL,
        ),
        pool,
        credentials_provider=_CredentialsProvider(_credentials()),
        transport=transport,
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["stage"] == "complete"
    assert payload["deleted_faq_count"] == 1
    assert payload["external_deletes"] == [
        {"external_id": "macro-1", "ok": True, "error": ""}
    ]
    assert events == ["external_delete", "local_delete"]
    assert transport.calls[0]["method"] == "DELETE"
    assert transport.calls[0]["url"] == f"{EXPECTED_BASE_URL}/api/v2/macros/macro-1"
    local_delete = pool.fetch_calls[-1]
    assert "DELETE FROM ticket_faq_markdown" in local_delete["query"]
    assert "account_id = $1" in local_delete["query"]
    assert "id = $2" in local_delete["query"]
    assert local_delete["args"] == (ACCOUNT_ID, FAQ_ID)


@pytest.mark.asyncio
async def test_execute_treats_not_found_macro_delete_as_idempotent_success() -> None:
    events: list[str] = []
    pool = _pool_with_snapshot(events=events)
    pool.fetch_rows.append([{"id": FAQ_ID}])
    transport = _Transport(
        error=ZendeskMacroTransportError(status_code=404),
        events=events,
    )

    code, payload = await cleanup.cleanup_sandbox_macro_writeback(
        _args(
            execute=True,
            confirm_live_zendesk_delete=True,
            expected_zendesk_base_url=EXPECTED_BASE_URL,
        ),
        pool,
        credentials_provider=_CredentialsProvider(_credentials()),
        transport=transport,
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["stage"] == "complete"
    assert payload["external_deletes"] == [
        {
            "external_id": "macro-1",
            "ok": True,
            "error": "zendesk_macro_not_found",
            "already_deleted": True,
        }
    ]
    assert events == ["external_delete", "local_delete"]


@pytest.mark.asyncio
async def test_external_delete_failure_leaves_local_rows() -> None:
    events: list[str] = []
    pool = _pool_with_snapshot(events=events)
    transport = _Transport(error=RuntimeError("delete failed"), events=events)

    code, payload = await cleanup.cleanup_sandbox_macro_writeback(
        _args(
            execute=True,
            confirm_live_zendesk_delete=True,
            expected_zendesk_base_url=EXPECTED_BASE_URL,
        ),
        pool,
        credentials_provider=_CredentialsProvider(_credentials()),
        transport=transport,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["stage"] == "external_delete"
    assert payload["errors"] == ["zendesk_macro_delete_failed"]
    assert events == ["external_delete"]
    assert all("DELETE FROM ticket_faq_markdown" not in call["query"] for call in pool.fetch_calls)


@pytest.mark.asyncio
async def test_execute_missing_draft_skips_without_credentials_or_deletes() -> None:
    pool = _Pool(fetchrow_rows=[None])
    credentials_provider = _CredentialsProvider(_credentials())
    transport = _Transport()

    code, payload = await cleanup.cleanup_sandbox_macro_writeback(
        _args(
            execute=True,
            confirm_live_zendesk_delete=True,
            expected_zendesk_base_url=EXPECTED_BASE_URL,
        ),
        pool,
        credentials_provider=credentials_provider,
        transport=transport,
    )

    assert code == cleanup.SKIPPED_EXIT
    assert payload["not_run_reason"] == "faq_draft_not_found"
    assert credentials_provider.calls == []
    assert transport.calls == []
    assert all("DELETE FROM ticket_faq_markdown" not in call["query"] for call in pool.fetch_calls)
