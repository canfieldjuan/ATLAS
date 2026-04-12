import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from atlas_brain.api import blog_admin as mod


@pytest.mark.asyncio
async def test_list_drafts_normalizes_blank_status_and_query_default(monkeypatch):
    class Pool:
        is_initialized = True

        async def fetch(self, query, *args):
            assert "WHERE status = $1" not in query
            assert args == (50,)
            return []

    monkeypatch.setattr(mod, "get_db_pool", lambda: Pool())

    result = await mod.list_drafts(status="   ")

    assert result == []


@pytest.mark.asyncio
async def test_list_drafts_trims_active_status(monkeypatch):
    class Pool:
        is_initialized = True

        async def fetch(self, query, *args):
            assert "WHERE status = $1" in query
            assert args == ("draft", 25)
            return []

    monkeypatch.setattr(mod, "get_db_pool", lambda: Pool())

    result = await mod.list_drafts(status="  draft  ", limit=25)

    assert result == []


@pytest.mark.asyncio
async def test_get_draft_evidence_normalizes_query_default_limit(monkeypatch):
    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(return_value={
            "data_context": json.dumps({"vendor_name": "Salesforce"}),
            "source_report_date": None,
        }),
        fetch=AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    result = await mod.get_draft_evidence(uuid4())

    assert pool.fetch.await_args.args[1:] == ("Salesforce", 20)
    assert result == {"basis": mod._REVIEW_BASIS_CANONICAL, "reviews": [], "count": 0}
