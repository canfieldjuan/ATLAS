from unittest.mock import AsyncMock

import pytest

from atlas_brain.reasoning.graph import _execute_single_action, _valid_uuid


def test_valid_uuid_accepts_uuid_string():
    value = _valid_uuid("123e4567-e89b-12d3-a456-426614174000")
    assert value == "123e4567-e89b-12d3-a456-426614174000"


def test_valid_uuid_rejects_company_name():
    assert _valid_uuid("Home & Away Realty") is None


@pytest.mark.asyncio
async def test_log_interaction_skips_when_entity_is_company(monkeypatch):
    crm = AsyncMock()
    crm.log_interaction = AsyncMock()
    monkeypatch.setattr(
        "atlas_brain.services.crm_provider.get_crm_provider",
        lambda: crm,
    )

    result = await _execute_single_action(
        "log_interaction",
        {"summary": "Reasoning agent action"},
        {"entity_type": "company", "entity_id": "Home & Away Realty"},
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "no valid contact UUID available"
    crm.log_interaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_log_interaction_uses_contact_uuid(monkeypatch):
    crm = AsyncMock()
    crm.log_interaction = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(
        "atlas_brain.services.crm_provider.get_crm_provider",
        lambda: crm,
    )

    result = await _execute_single_action(
        "log_interaction",
        {"summary": "Reasoning agent action"},
        {"entity_type": "contact", "entity_id": "123e4567-e89b-12d3-a456-426614174000"},
    )

    assert result == {"ok": True}
    crm.log_interaction.assert_awaited_once()
