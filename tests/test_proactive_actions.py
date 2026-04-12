import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)


@pytest.mark.asyncio
async def test_list_actions_rejects_invalid_status_before_db_touch(monkeypatch):
    from atlas_brain.api import proactive_actions as mod

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr("atlas_brain.storage.database.get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.list_actions(status="invalid", limit=50)

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid status"


@pytest.mark.asyncio
async def test_list_actions_normalizes_blank_status_to_pending(monkeypatch):
    from atlas_brain.api import proactive_actions as mod

    row = {
        "id": uuid4(),
        "action_text": "Follow up",
        "action_type": "email",
        "status": "pending",
        "source_time": None,
        "session_id": None,
        "created_at": datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc),
        "resolved_at": None,
    }
    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock(return_value=[row]))
    monkeypatch.setattr("atlas_brain.storage.database.get_db_pool", lambda: pool)

    result = await mod.list_actions(status="   ", limit=25)

    query, status_value, limit_value = pool.fetch.await_args.args
    assert "WHERE status = $1" in query
    assert status_value == "pending"
    assert limit_value == 25
    assert result["count"] == 1
    assert result["actions"][0]["status"] == "pending"


@pytest.mark.asyncio
async def test_list_actions_trims_and_normalizes_all_status(monkeypatch):
    from atlas_brain.api import proactive_actions as mod

    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock(return_value=[]))
    monkeypatch.setattr("atlas_brain.storage.database.get_db_pool", lambda: pool)

    result = await mod.list_actions(status="  ALL  ", limit=10)

    query, limit_value = pool.fetch.await_args.args
    assert "WHERE status = $1" not in query
    assert limit_value == 10
    assert result == {"count": 0, "actions": []}


@pytest.mark.asyncio
async def test_list_actions_trims_and_normalizes_specific_status(monkeypatch):
    from atlas_brain.api import proactive_actions as mod

    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock(return_value=[]))
    monkeypatch.setattr("atlas_brain.storage.database.get_db_pool", lambda: pool)

    await mod.list_actions(status="  Done  ", limit=5)

    _, status_value, limit_value = pool.fetch.await_args.args
    assert status_value == "done"
    assert limit_value == 5
