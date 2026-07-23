"""Tests for issue #2151 read-path scoping on the CRM MCP server.

Semantics under test (plans/PR-EOM-Read-Scoping.md):
- With ATLAS_MCP_CRM_DEFAULT_BUSINESS_CONTEXT configured, read tools that
  receive no explicit business_context_id operate on the default tenant's
  page first, then the NULL-context legacy page (the claimable population
  from PR #2153).
- An explicit argument always wins; no default configured = legacy
  unscoped behavior (backward compatible).
- Id-addressed tools treat foreign-tenant contacts as nonexistent
  (fail-closed, no cross-tenant existence leak).
- The MCP create tool default-stamps new contacts so the MCP surface stops
  minting NULL-context rows.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

import atlas_brain.mcp.crm_server as crm_srv  # noqa: E402

EOM = "effingham_maids"
UUID = "12345678-1234-5678-1234-567812345678"


@pytest.fixture
def default_ctx(monkeypatch):
    from atlas_brain.config import settings

    monkeypatch.setattr(settings.mcp, "crm_default_business_context", EOM)


@pytest.fixture
def no_default(monkeypatch):
    from atlas_brain.config import settings

    monkeypatch.setattr(settings.mcp, "crm_default_business_context", None)


@pytest.fixture(autouse=True)
def _clear_provider_override():
    yield
    crm_srv.set_provider_override(None)


def _provider_mock(monkeypatch, **overrides):
    provider = MagicMock()
    provider.search_contacts = AsyncMock(return_value=overrides.get("search", []))
    provider.get_contact = AsyncMock(return_value=overrides.get("get"))
    provider.update_contact = AsyncMock(return_value={"id": UUID})
    provider.delete_contact = AsyncMock(return_value=True)
    provider.list_contacts = AsyncMock(return_value=[])
    provider.log_interaction = AsyncMock(return_value={"id": "i-1"})
    provider.get_interactions = AsyncMock(return_value=[])
    provider.get_contact_appointments = AsyncMock(return_value=[])
    crm_srv.set_provider_override(lambda: provider)
    return provider


# ---------------------------------------------------------------------------
# Scoped search semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_applies_default_then_null_fallback(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch)
    await crm_srv.search_contacts(query="jane")
    calls = provider.search_contacts.await_args_list
    assert calls[0].kwargs["business_context_id"] == EOM
    assert calls[1].kwargs["business_context_id_is_null"] is True


@pytest.mark.asyncio
async def test_search_scoped_hit_skips_null_fallback(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch, search=[{"id": "c1", "business_context_id": EOM}])
    out = json.loads(await crm_srv.search_contacts(query="jane"))
    assert out["found"] is True
    assert len(provider.search_contacts.await_args_list) == 1


@pytest.mark.asyncio
async def test_search_explicit_context_wins_over_default(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch, search=[{"id": "b1"}])
    await crm_srv.search_contacts(query="acme", business_context_id="churnsignals")
    calls = provider.search_contacts.await_args_list
    assert len(calls) == 1
    assert calls[0].kwargs["business_context_id"] == "churnsignals"


@pytest.mark.asyncio
async def test_search_unscoped_without_default(no_default, monkeypatch):
    provider = _provider_mock(monkeypatch, search=[{"id": "c1"}])
    await crm_srv.search_contacts(query="jane")
    calls = provider.search_contacts.await_args_list
    assert len(calls) == 1
    assert "business_context_id" not in calls[0].kwargs


@pytest.mark.asyncio
async def test_default_alone_satisfies_search_requires_one_of(default_ctx, monkeypatch):
    """A bare list-my-tenant search is valid once a default exists."""
    _provider_mock(monkeypatch)
    out = json.loads(await crm_srv.search_contacts())
    assert "error" not in out or "required" not in out.get("error", "")


# ---------------------------------------------------------------------------
# Id-addressed tenant guard (fail-closed)
# ---------------------------------------------------------------------------


FOREIGN = {"id": UUID, "business_context_id": "churnsignals"}
LEGACY_NULL = {"id": UUID, "business_context_id": None}
SAME = {"id": UUID, "business_context_id": EOM}


@pytest.mark.asyncio
async def test_get_contact_hides_foreign_tenant(default_ctx, monkeypatch):
    _provider_mock(monkeypatch, get=FOREIGN)
    out = json.loads(await crm_srv.get_contact(UUID))
    assert out == {"found": False, "contact": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("row", [LEGACY_NULL, SAME])
async def test_get_contact_shows_default_and_null(default_ctx, monkeypatch, row):
    _provider_mock(monkeypatch, get=row)
    out = json.loads(await crm_srv.get_contact(UUID))
    assert out["found"] is True


@pytest.mark.asyncio
async def test_update_and_delete_refuse_foreign_tenant(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch, get=FOREIGN)
    up = json.loads(await crm_srv.update_contact(UUID, full_name="X"))
    assert up["success"] is False
    provider.update_contact.assert_not_awaited()
    de = json.loads(await crm_srv.delete_contact(UUID))
    assert de["success"] is False
    provider.delete_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_log_interaction_refuses_foreign_tenant(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch, get=FOREIGN)
    out = json.loads(await crm_srv.log_interaction(UUID, interaction_type="note", summary="x"))
    assert out["success"] is False
    provider.log_interaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_guard_disabled_without_default(no_default, monkeypatch):
    provider = _provider_mock(monkeypatch, get=FOREIGN)
    out = json.loads(await crm_srv.get_contact(UUID))
    assert out["found"] is True  # legacy behavior preserved
    # and mutations pass through
    await crm_srv.update_contact(UUID, full_name="X")
    provider.update_contact.assert_awaited()


# ---------------------------------------------------------------------------
# Create default-stamp + list default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_contact_default_stamps(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch)
    provider.create_contact = AsyncMock(return_value={"id": "new-1"})
    await crm_srv.create_contact(full_name="Jane")
    data = provider.create_contact.await_args.args[0]
    assert data["business_context_id"] == EOM


@pytest.mark.asyncio
async def test_create_contact_explicit_context_wins(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch)
    provider.create_contact = AsyncMock(return_value={"id": "new-1"})
    await crm_srv.create_contact(full_name="Acme", business_context_id="churnsignals")
    assert provider.create_contact.await_args.args[0]["business_context_id"] == "churnsignals"


@pytest.mark.asyncio
async def test_list_contacts_uses_default(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch)
    await crm_srv.list_contacts()
    assert provider.list_contacts.await_args.kwargs["business_context_id"] == EOM
