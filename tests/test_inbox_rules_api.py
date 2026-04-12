from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from atlas_brain.api import inbox_rules as mod


@pytest.mark.asyncio
async def test_create_rule_rejects_blank_name_before_db_touch(monkeypatch):
    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(ValidationError):
        mod.InboxRuleCreate(name="   ")


@pytest.mark.asyncio
async def test_create_rule_trims_optional_fields_before_insert(monkeypatch):
    row = {"id": uuid4(), "created_at": datetime.now(timezone.utc), "name": "Rule"}
    pool = SimpleNamespace(is_initialized=True, fetchrow=AsyncMock(return_value=row))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    await mod.create_rule(
        mod.InboxRuleCreate(
            name="  Rule  ",
            sender_domain="  example.com  ",
            sender_contains="   ",
            subject_contains="  renewal  ",
            category="  sales  ",
            priority="  high  ",
            set_priority="  urgent  ",
            set_category="   ",
            label="  nurture  ",
        )
    )

    args = pool.fetchrow.await_args.args
    assert args[1:] == (
        "Rule", True, 0,
        "example.com", None, "renewal",
        "sales", None, "high", None, None,
        "urgent", None, None, "nurture",
        False, False, False,
    )


@pytest.mark.asyncio
async def test_update_rule_trims_and_clears_text_fields(monkeypatch):
    rule_id = uuid4()
    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(return_value={"id": rule_id, "updated_at": datetime.now(timezone.utc)}),
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    await mod.update_rule(
        rule_id,
        mod.InboxRuleUpdate(name="  VIP Rule  ", sender_domain="   ", label="  Prospect  "),
    )

    sql, *params = pool.fetchrow.await_args.args
    assert "name = $1" in sql
    assert "sender_domain = $2" in sql
    assert "label = $3" in sql
    assert params[0] == "VIP Rule"
    assert params[1] is None
    assert params[2] == "Prospect"
    assert params[-1] == rule_id


@pytest.mark.asyncio
async def test_reorder_rules_rejects_invalid_id_before_db_touch(monkeypatch):
    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.reorder_rules(mod.ReorderBody(rules=[mod.ReorderItem(id="not-a-uuid", position=1)]))

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid id"


@pytest.mark.asyncio
async def test_test_rules_rejects_blank_sender_before_db_touch(monkeypatch):
    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(ValidationError):
        mod.InboxRuleTest(sender="   ", subject="  ok  ")
