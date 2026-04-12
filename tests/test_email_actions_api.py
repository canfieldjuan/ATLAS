from unittest.mock import AsyncMock

import pytest

from atlas_brain.api import email_actions as mod


@pytest.mark.asyncio
async def test_load_email_rejects_blank_message_id_before_db_touch(monkeypatch):
    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr("atlas_brain.storage.database.get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod._load_email("   ")

    assert exc.value.status_code == 422
    assert exc.value.detail == "gmail_message_id is required"


@pytest.mark.asyncio
async def test_load_email_trims_message_id_before_query(monkeypatch):
    class Pool:
        is_initialized = True
        fetchrow = AsyncMock(return_value={"gmail_message_id": "uid-1"})

    monkeypatch.setattr("atlas_brain.storage.database.get_db_pool", lambda: Pool())

    result = await mod._load_email("  uid-1  ")

    assert Pool.fetchrow.await_args.args[1] == "uid-1"
    assert result == {"gmail_message_id": "uid-1"}


@pytest.mark.asyncio
async def test_generate_quote_rejects_blank_message_id_before_load(monkeypatch):
    monkeypatch.setattr(mod, "_load_email", AsyncMock(side_effect=AssertionError("load touched")))

    with pytest.raises(mod.HTTPException) as exc:
        await mod.generate_quote("   ")

    assert exc.value.status_code == 422
    assert exc.value.detail == "gmail_message_id is required"


@pytest.mark.asyncio
async def test_generate_quote_trims_message_id_for_downstream_calls(monkeypatch):
    monkeypatch.setattr(mod, "_load_email", AsyncMock(return_value={"contact_id": "c-1"}))

    import atlas_brain.api.email_drafts as draft_mod
    monkeypatch.setattr(draft_mod, "generate_draft", AsyncMock(return_value={"draft_id": "d-1"}))

    result = await mod.generate_quote("  uid-42  ")

    mod._load_email.assert_awaited_once_with("uid-42")
    draft_mod.generate_draft.assert_awaited_once_with("uid-42")
    assert result["gmail_message_id"] == "uid-42"
    assert result["draft"] == {"draft_id": "d-1"}
