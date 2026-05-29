from __future__ import annotations

import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback import MacroWritebackMapping
from extracted_content_pipeline.faq_macro_writeback_postgres import (
    PostgresFAQMacroWritebackMappingRepository,
)


MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "extracted_content_pipeline"
    / "storage"
    / "migrations"
    / "328_ticket_faq_macro_writebacks.sql"
)


class _Pool:
    def __init__(self) -> None:
        self.fetch_rows: list[dict] = []
        self.fetchrow_rows: list[dict] = []
        self.fetch_calls: list[dict] = []
        self.fetchrow_calls: list[dict] = []

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": query, "args": args})
        return self.fetch_rows

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append({"query": query, "args": args})
        return self.fetchrow_rows.pop(0)


def _row(**overrides) -> dict:
    row = {
        "platform": "zendesk",
        "faq_draft_id": "11111111-1111-1111-1111-111111111111",
        "faq_item_id": "faq-draft-1:item-1",
        "external_id": "macro-123",
        "external_url": "https://example.zendesk.com/macros/123",
        "metadata": json.dumps({"source": "dry_run"}),
    }
    row.update(overrides)
    return row


def test_ticket_faq_macro_writeback_migration_adds_scoped_unique_mapping() -> None:
    sql = MIGRATION.read_text()

    assert "CREATE TABLE IF NOT EXISTS ticket_faq_macro_writebacks" in sql
    assert (
        "faq_draft_id UUID NOT NULL REFERENCES ticket_faq_markdown(id) "
        "ON DELETE CASCADE"
    ) in sql
    assert "UNIQUE (account_id, platform, faq_draft_id, faq_item_id)" in sql
    assert "UNIQUE (account_id, platform, external_id)" in sql
    assert "chk_ticket_faq_macro_writebacks_account_id" in sql
    assert "idx_ticket_faq_macro_writebacks_platform" in sql


@pytest.mark.asyncio
async def test_get_mapping_filters_by_tenant_platform_and_faq_item() -> None:
    pool = _Pool()
    pool.fetch_rows = [_row()]
    repo = PostgresFAQMacroWritebackMappingRepository(pool)

    mapping = await repo.get_mapping(
        platform=" zendesk ",
        faq_draft_id="11111111-1111-1111-1111-111111111111",
        faq_item_id=" faq-draft-1:item-1 ",
        scope=TenantScope(account_id="acct-1"),
    )

    assert mapping == MacroWritebackMapping(
        platform="zendesk",
        faq_draft_id="11111111-1111-1111-1111-111111111111",
        faq_item_id="faq-draft-1:item-1",
        external_id="macro-123",
        external_url="https://example.zendesk.com/macros/123",
        metadata={"source": "dry_run"},
    )
    call = pool.fetch_calls[0]
    assert "FROM ticket_faq_macro_writebacks" in call["query"]
    assert "account_id = $1" in call["query"]
    assert "platform = $2" in call["query"]
    assert "faq_draft_id = $3" in call["query"]
    assert "faq_item_id = $4" in call["query"]
    assert call["args"] == (
        "acct-1",
        "zendesk",
        "11111111-1111-1111-1111-111111111111",
        "faq-draft-1:item-1",
    )


@pytest.mark.asyncio
async def test_get_mapping_returns_none_on_miss() -> None:
    pool = _Pool()
    repo = PostgresFAQMacroWritebackMappingRepository(pool)

    mapping = await repo.get_mapping(
        platform="zendesk",
        faq_draft_id="11111111-1111-1111-1111-111111111111",
        faq_item_id="faq-draft-1:item-1",
        scope=TenantScope(account_id="acct-1"),
    )

    assert mapping is None


@pytest.mark.asyncio
async def test_upsert_mapping_uses_faq_item_idempotency_key_and_metadata_jsonb() -> None:
    pool = _Pool()
    pool.fetchrow_rows = [_row(external_id="macro-456", metadata={"updated": True})]
    repo = PostgresFAQMacroWritebackMappingRepository(pool)

    mapping = await repo.upsert_mapping(
        MacroWritebackMapping(
            platform=" zendesk ",
            faq_draft_id="11111111-1111-1111-1111-111111111111",
            faq_item_id=" faq-draft-1:item-1 ",
            external_id=" macro-456 ",
            external_url=" https://example.zendesk.com/macros/456 ",
            metadata={"updated": True},
        ),
        scope=TenantScope(account_id="acct-1"),
    )

    call = pool.fetchrow_calls[0]
    assert "INSERT INTO ticket_faq_macro_writebacks" in call["query"]
    assert "ON CONFLICT (account_id, platform, faq_draft_id, faq_item_id)" in call["query"]
    assert "DO UPDATE SET" in call["query"]
    assert "updated_at = NOW()" in call["query"]
    assert call["args"] == (
        "acct-1",
        "zendesk",
        "11111111-1111-1111-1111-111111111111",
        "faq-draft-1:item-1",
        "macro-456",
        "https://example.zendesk.com/macros/456",
        '{"updated":true}',
    )
    assert mapping.external_id == "macro-456"
    assert mapping.metadata == {"updated": True}


@pytest.mark.asyncio
async def test_upsert_mapping_keeps_account_scope_outside_client_payload() -> None:
    pool = _Pool()
    pool.fetchrow_rows = [_row()]
    repo = PostgresFAQMacroWritebackMappingRepository(pool)

    await repo.upsert_mapping(
        MacroWritebackMapping(
            platform="zendesk",
            faq_draft_id="11111111-1111-1111-1111-111111111111",
            faq_item_id="faq-draft-1:item-1",
            external_id="macro-123",
        ),
        scope=TenantScope(account_id="acct-tenant-a"),
    )

    assert pool.fetchrow_calls[0]["args"][0] == "acct-tenant-a"
    assert "account_id" not in pool.fetchrow_calls[0]["args"][-1]
