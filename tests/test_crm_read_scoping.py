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

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO = Path(__file__).resolve().parent.parent

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
    provider.claim_contact = AsyncMock(
        return_value={"id": UUID, "business_context_id": EOM})
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
async def test_search_merges_tenant_and_legacy_pages(default_ctx, monkeypatch):
    """A tenant hit must NOT hide claimable legacy rows: both pages are
    queried and merged, tenant rows first (#2157 post-merge review)."""
    provider = _provider_mock(monkeypatch)
    provider.search_contacts = AsyncMock(side_effect=[
        [{"id": "t1", "business_context_id": EOM}],
        [{"id": "n1", "business_context_id": None}],
    ])
    out = json.loads(await crm_srv.search_contacts(query="jane"))
    assert out["found"] is True
    assert [c["id"] for c in out["contacts"]] == ["t1", "n1"]
    assert len(provider.search_contacts.await_args_list) == 2


@pytest.mark.asyncio
async def test_search_merge_truncates_to_limit(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch)
    provider.search_contacts = AsyncMock(side_effect=[
        [{"id": "t1", "business_context_id": EOM}],
        [{"id": "n1", "business_context_id": None}],
    ])
    out = json.loads(await crm_srv.search_contacts(query="jane", limit=1))
    assert [c["id"] for c in out["contacts"]] == ["t1"]


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
async def test_list_contacts_merges_null_page_under_default(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch)
    tenant_page = [{"id": "t1", "business_context_id": EOM}]
    legacy_page = [{"id": "n1", "business_context_id": None}]
    provider.list_contacts = AsyncMock(side_effect=[tenant_page, legacy_page])
    out = json.loads(await crm_srv.list_contacts())
    calls = provider.list_contacts.await_args_list
    assert calls[0].kwargs["business_context_id"] == EOM
    assert calls[1].kwargs["business_context_id_is_null"] is True
    assert [c["id"] for c in out["contacts"]] == ["t1", "n1"]


@pytest.mark.asyncio
async def test_update_lead_pipeline_is_scoped_and_requires_lead(
    default_ctx, monkeypatch
):
    provider = _provider_mock(
        monkeypatch,
        get={**SAME, "contact_type": "lead"},
    )
    out = json.loads(await crm_srv.update_contact(
        UUID,
        lead_stage=" qualified ",
        lead_owner=" Juan ",
        next_follow_up_at="2026-07-24T10:00:00-05:00",
    ))

    assert out["success"] is True
    args, kwargs = provider.update_contact.await_args
    assert args[0] == UUID
    assert args[1] == {
        "lead_stage": "qualified",
        "lead_owner": "Juan",
        "next_follow_up_at": datetime(2026, 7, 24, 15, tzinfo=timezone.utc),
    }
    assert kwargs == {"require_contact_type": "lead"}


@pytest.mark.asyncio
async def test_update_lead_pipeline_rejects_non_lead_and_foreign(
    default_ctx, monkeypatch
):
    provider = _provider_mock(
        monkeypatch,
        get={**SAME, "contact_type": "customer"},
    )
    not_lead = json.loads(await crm_srv.update_contact(UUID, lead_stage="new"))
    assert not_lead["success"] is False
    assert "lead contact" in not_lead["error"]
    provider.update_contact.assert_not_awaited()

    provider.get_contact.return_value = {**FOREIGN, "contact_type": "lead"}
    foreign = json.loads(await crm_srv.update_contact(UUID, lead_stage="new"))
    assert foreign == {"success": False, "error": "Contact not found"}
    provider.update_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_lead_pipeline_clear_and_validation(default_ctx, monkeypatch):
    provider = _provider_mock(
        monkeypatch,
        get={**SAME, "contact_type": "lead"},
    )
    cleared = json.loads(await crm_srv.update_contact(
        UUID, clear_lead_owner=True, clear_next_follow_up=True
    ))
    assert cleared["success"] is True
    assert provider.update_contact.await_args.args[1] == {
        "lead_owner": None,
        "next_follow_up_at": None,
    }

    provider.update_contact.reset_mock()
    invalid = json.loads(await crm_srv.update_contact(
        UUID, next_follow_up_at="2026-07-24T10:00:00"
    ))
    assert invalid["success"] is False
    assert "UTC offset" in invalid["error"]
    provider.update_contact.assert_not_awaited()


def test_pipeline_input_boundaries():
    assert crm_srv._pipeline_text("x" * 64, "lead_stage", 64) == "x" * 64
    with pytest.raises(ValueError, match="at most 64"):
        crm_srv._pipeline_text("x" * 65, "lead_stage", 64)
    assert crm_srv._pipeline_timestamp(
        "2026-07-24T22:00:00Z", "next_follow_up_at"
    ) == datetime(2026, 7, 24, 22, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="at most 64"):
        crm_srv._pipeline_timestamp("2" * 65, "next_follow_up_at")


@pytest.mark.asyncio
async def test_due_lead_list_uses_one_sql_scoped_population(
    default_ctx, monkeypatch
):
    provider = _provider_mock(monkeypatch)
    provider.list_contacts.return_value = [{"id": "due-1"}]

    out = json.loads(await crm_srv.list_contacts(
        lead_stage="qualified",
        next_follow_up_before="2026-07-24T17:00:00-05:00",
        limit=10,
    ))

    assert [row["id"] for row in out["contacts"]] == ["due-1"]
    provider.list_contacts.assert_awaited_once()
    kwargs = provider.list_contacts.await_args.kwargs
    assert kwargs["business_context_id"] == EOM
    assert kwargs["include_unclaimed_legacy"] is True
    assert kwargs["contact_type"] == "lead"
    assert kwargs["lead_stage"] == "qualified"
    assert kwargs["next_follow_up_before"] == datetime(
        2026, 7, 24, 22, tzinfo=timezone.utc
    )


@pytest.mark.asyncio
async def test_provider_rejects_pipeline_fields_on_new_non_lead():
    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    provider = DatabaseCRMProvider()
    with pytest.raises(ValueError, match="contact_type='lead'"):
        await provider.create_contact({
            "full_name": "Customer",
            "contact_type": "customer",
            "lead_stage": "new",
        })
    with pytest.raises(ValueError, match="contact_type='lead'"):
        await provider.update_contact(
            UUID,
            {"contact_type": "customer", "lead_stage": "qualified"},
        )


@pytest.mark.asyncio
async def test_list_contacts_explicit_context_single_page(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch)
    await crm_srv.list_contacts(business_context_id="churnsignals")
    calls = provider.list_contacts.await_args_list
    assert len(calls) == 1
    assert calls[0].kwargs["business_context_id"] == "churnsignals"


@pytest.mark.asyncio
async def test_get_contact_single_fetch_validates_returned_row(default_ctx, monkeypatch):
    """TOCTOU regression (#2157 post-merge review): the row validated must
    be the row returned. With guard-then-refetch, a NULL row claimed by
    another tenant between the awaits serialized as found=true foreign."""
    provider = _provider_mock(monkeypatch)
    provider.get_contact = AsyncMock(side_effect=[LEGACY_NULL, FOREIGN])
    out = json.loads(await crm_srv.get_contact(UUID))
    assert out["found"] is True
    assert out["contact"]["business_context_id"] is None
    assert provider.get_contact.await_count == 1


def test_row_visible_semantics(default_ctx):
    assert crm_srv._row_visible(None, None) is False
    assert crm_srv._row_visible(SAME, None) is True
    assert crm_srv._row_visible(LEGACY_NULL, None) is True
    assert crm_srv._row_visible(FOREIGN, None) is False
    assert crm_srv._row_visible(FOREIGN, "churnsignals") is True
    assert crm_srv._row_visible(LEGACY_NULL, EOM) is False  # explicit = exact


class _StubCtx:
    """Minimal CustomerContext stand-in for behavioral TOCTOU tests."""

    def __init__(self, contact):
        self.is_empty = False
        self.contact = contact
        self.interactions = []
        self.appointments = []
        self.call_transcripts = []
        self.sent_emails = []
        self.inbox_emails = []
        self.b2b_churn_signals = []


@pytest.mark.asyncio
async def test_customer_context_refuses_row_claimed_mid_gather(default_ctx, monkeypatch):
    """Behavioral TOCTOU regression (#2165 review): the pre-guard sees a
    claimable NULL row, but the service's own fetch returns the row already
    claimed by the other tenant -- the response must refuse, not serialize
    the foreign contact."""
    provider = _provider_mock(monkeypatch)
    provider.get_contact = AsyncMock(side_effect=[LEGACY_NULL, FOREIGN])
    from atlas_brain.services import customer_context as ccx

    class _Svc:
        async def get_context(self, contact_id, **kwargs):
            return _StubCtx(dict(LEGACY_NULL))

    monkeypatch.setattr(ccx, "get_customer_context_service", lambda: _Svc())
    out = json.loads(await crm_srv.get_customer_context(contact_id=UUID))
    assert out == {"found": False, "context": None}
    assert provider.get_contact.await_count == 2


@pytest.mark.asyncio
async def test_explicit_context_refuses_row_cleared_mid_gather(
    default_ctx, monkeypatch
):
    provider = _provider_mock(monkeypatch)
    provider.get_contact = AsyncMock(side_effect=[SAME, LEGACY_NULL])
    from atlas_brain.services import customer_context as ccx

    class _Svc:
        async def get_context(self, contact_id, **kwargs):
            return _StubCtx(dict(SAME))

    monkeypatch.setattr(ccx, "get_customer_context_service", lambda: _Svc())
    out = json.loads(
        await crm_srv.get_customer_context(
            contact_id=UUID,
            business_context_id=EOM,
        )
    )
    assert out == {"found": False, "context": None}


@pytest.mark.asyncio
async def test_default_context_retains_null_legacy_visibility_mid_gather(
    default_ctx, monkeypatch
):
    provider = _provider_mock(monkeypatch)
    provider.get_contact = AsyncMock(side_effect=[SAME, LEGACY_NULL])
    from atlas_brain.services import customer_context as ccx

    class _Svc:
        async def get_context(self, contact_id, **kwargs):
            return _StubCtx(dict(SAME))

    monkeypatch.setattr(ccx, "get_customer_context_service", lambda: _Svc())
    out = json.loads(await crm_srv.get_customer_context(contact_id=UUID))
    assert out["found"] is True
    assert out["contact"]["business_context_id"] is None


@pytest.mark.asyncio
async def test_customer_context_serializes_still_visible_row(default_ctx, monkeypatch):
    """The omission assertions only prove something if the stub SEEDS each
    omitted source non-empty -- an empty-in/empty-out check passes even
    when the serializer regresses (Codex, #2172)."""
    _provider_mock(monkeypatch, get=SAME)
    from atlas_brain.services import customer_context as ccx

    class _Svc:
        async def get_context(self, contact_id, **kwargs):
            ctx = _StubCtx(dict(SAME))
            ctx.sent_emails = [{"subject": "old estimate thread"}]
            ctx.inbox_emails = [{"subject": "customer reply"}]
            ctx.b2b_churn_signals = [{"vendor": "acme-saas"}]
            return ctx

    monkeypatch.setattr(ccx, "get_customer_context_service", lambda: _Svc())
    out = json.loads(await crm_srv.get_customer_context(contact_id=UUID))
    assert out["found"] is True
    assert out["contact"]["business_context_id"] == EOM
    assert out["emails_omitted_under_scope"] is True
    assert out["email_sources_omitted_under_scope"] == ["inbox_emails"]
    assert out["b2b_enrichment_omitted_under_scope"] is True
    assert out["sent_emails"] == [{"subject": "old estimate thread"}]
    assert out["inbox_emails"] == []
    assert out["b2b_churn_signals"] == []


@pytest.mark.asyncio
async def test_provider_interactions_scoped_query_executes_atomically(monkeypatch):
    """Behavioral (#2165 review): the tenant predicate is bound in the SAME
    statement as the page read -- join + bound params -- not applied after
    the query; unscoped callers keep the legacy statement."""
    from atlas_brain.services.crm_provider import DatabaseCRMProvider
    from atlas_brain.storage import database

    captured = {}

    class _Pool:
        async def fetch(self, sql, *params):
            captured["sql"] = " ".join(sql.split())
            captured["params"] = params
            return [{"id": "i-1"}]

    monkeypatch.setattr(database, "get_db_pool", lambda: _Pool())
    provider = DatabaseCRMProvider.__new__(DatabaseCRMProvider)

    rows = await provider.get_interactions(UUID, limit=5, business_context_id=EOM)
    assert rows == [{"id": "i-1"}]
    assert "JOIN contacts c ON c.id = ci.contact_id" in captured["sql"]
    assert "(c.business_context_id = $2 OR c.business_context_id IS NULL)" in captured["sql"]
    assert captured["params"] == (UUID, EOM, 5)

    await provider.get_interactions(UUID, limit=5)
    assert "JOIN" not in captured["sql"]
    assert captured["params"] == (UUID, 5)


def test_customer_context_validates_fetched_row_source():
    """get_customer_context must validate ctx.contact (the service's own
    fetch), not only the pre-guard read."""
    src = (REPO / "atlas_brain/mcp/crm_server.py").read_text(encoding="utf-8")
    validation = src.split("if ctx.is_empty:", 1)[1][:600]
    assert "_row_visible(" in validation and "ctx.contact" in validation


def test_provider_interactions_scope_is_atomic():
    src = (REPO / "atlas_brain/services/crm_provider.py").read_text(encoding="utf-8")
    block = src.split("async def get_interactions", 1)[1][:1400]
    assert "JOIN contacts c ON c.id = ci.contact_id" in block
    assert "c.business_context_id = $2 OR c.business_context_id IS NULL" in block


@pytest.mark.asyncio
async def test_get_interactions_passes_scope(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch, get=SAME)
    await crm_srv.get_interactions(UUID)
    call = provider.get_interactions.await_args
    assert call.kwargs["business_context_id"] == EOM


# ---------------------------------------------------------------------------
# Explicit tenant override on id-addressed tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_contact_explicit_context_reaches_foreign_tenant(default_ctx, monkeypatch):
    _provider_mock(monkeypatch, get=FOREIGN)
    out = json.loads(await crm_srv.get_contact(UUID, business_context_id="churnsignals"))
    assert out["found"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("row", [LEGACY_NULL, FOREIGN])
async def test_explicit_context_is_exact_match(default_ctx, monkeypatch, row):
    """Explicit tenant addressing mirrors explicit search: exact page only,
    no NULL-context fallback (NULL rows stay reachable via the default)."""
    _provider_mock(monkeypatch, get=row)
    out = json.loads(await crm_srv.get_contact(UUID, business_context_id=EOM))
    assert out["found"] is False


@pytest.mark.asyncio
async def test_update_contact_explicit_override_reaches_foreign_tenant(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch, get=FOREIGN)
    out = json.loads(await crm_srv.update_contact(
        UUID, full_name="X", business_context_id="churnsignals"))
    assert out["success"] is True
    provider.update_contact.assert_awaited()


@pytest.mark.asyncio
async def test_update_claims_null_contact_under_default(default_ctx, monkeypatch):
    """Claim-on-write: updating a NULL-context legacy row from a scoped
    session claims it for the default tenant via compare-and-set."""
    provider = _provider_mock(monkeypatch, get=LEGACY_NULL)
    out = json.loads(await crm_srv.update_contact(UUID, phone="217-555-0000"))
    assert out["success"] is True
    provider.claim_contact.assert_awaited_once_with(UUID, EOM)
    data = provider.update_contact.await_args.args[1]
    assert "business_context_id" not in data


@pytest.mark.asyncio
async def test_update_fails_closed_when_claim_lost(default_ctx, monkeypatch):
    """A concurrent claim by another tenant between guard and write must
    abort the mutation, not overwrite the other tenant's claim."""
    provider = _provider_mock(monkeypatch, get=LEGACY_NULL)
    provider.claim_contact = AsyncMock(return_value=None)
    out = json.loads(await crm_srv.update_contact(UUID, phone="217-555-0000"))
    assert out["success"] is False
    provider.update_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_same_tenant_row_not_reclaimed(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch, get=SAME)
    await crm_srv.update_contact(UUID, phone="217-555-0000")
    provider.claim_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_without_default_never_claims(no_default, monkeypatch):
    provider = _provider_mock(monkeypatch, get=LEGACY_NULL)
    await crm_srv.update_contact(UUID, phone="217-555-0000")
    provider.claim_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_claims_null_contact_under_default(default_ctx, monkeypatch):
    """Archiving is a legacy mutation: the row is claimed first so one
    tenant cannot hide shared legacy data from the other's default view."""
    provider = _provider_mock(monkeypatch, get=LEGACY_NULL)
    out = json.loads(await crm_srv.delete_contact(UUID))
    assert out["success"] is True
    provider.claim_contact.assert_awaited_once_with(UUID, EOM)


@pytest.mark.asyncio
async def test_delete_fails_closed_when_claim_lost(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch, get=LEGACY_NULL)
    provider.claim_contact = AsyncMock(return_value=None)
    out = json.loads(await crm_srv.delete_contact(UUID))
    assert out["success"] is False
    provider.delete_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_contact_appointments_filtered_to_scope(default_ctx, monkeypatch):
    """A NULL-context legacy contact must not expose foreign-tenant
    appointment history through the linked-appointments tool; the scope is
    also pushed into the provider query (before its LIMIT)."""
    provider = _provider_mock(monkeypatch, get=LEGACY_NULL)
    provider.get_contact_appointments = AsyncMock(return_value=[
        {"id": "a1", "business_context_id": EOM},
        {"id": "a2", "business_context_id": "churnsignals"},
    ])
    out = json.loads(await crm_srv.get_contact_appointments(UUID))
    assert [a["id"] for a in out["appointments"]] == ["a1"]
    assert out["count"] == 1
    call = provider.get_contact_appointments.await_args
    assert call.kwargs["business_context_id"] == EOM


def test_provider_claim_contact_is_compare_and_set():
    """The claim must be conditional in SQL, not a blind stamp."""
    src = (REPO / "atlas_brain/services/crm_provider.py").read_text(encoding="utf-8")
    block = src.split("async def claim_contact", 1)[1][:900]
    assert "business_context_id IS NULL OR business_context_id = $2" in block


def test_call_repo_scopes_before_limit():
    src = (REPO / "atlas_brain/storage/repositories/call_transcript.py").read_text(encoding="utf-8")
    block = src.split("async def get_by_contact_id", 1)[1].split(
        "async def get_recent", 1
    )[0]
    legacy_block = block.split("elif business_context_id:", 1)[1]
    assert "business_context_id = $2" in legacy_block
    assert "business_context_id IS NULL" in legacy_block
    assert legacy_block.index("business_context_id IS NULL") < legacy_block.index(
        "LIMIT $3"
    )


def test_context_service_threads_scope_to_child_queries():
    src = (REPO / "atlas_brain/services/customer_context.py").read_text(encoding="utf-8")
    gather = src.split("async def _gather", 1)[1]
    assert gather.count("business_context_id=business_context_id") >= 3


@pytest.mark.asyncio
async def test_create_contact_claims_legacy_match_with_cas():
    """The stamped-create legacy merge claims by compare-and-set, never by
    blind update (round-5 MAJOR)."""
    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    class _Provider(DatabaseCRMProvider):
        def __init__(self, claim_result):
            self.claim_calls = []
            self.updates = []
            self._claim_result = claim_result

        async def search_contacts(self, **kwargs):
            if kwargs.get("business_context_id_is_null"):
                return [{"id": UUID, "business_context_id": None,
                         "phone": "2175550000"}]
            return []

        async def claim_contact(self, contact_id, business_context_id):
            self.claim_calls.append((contact_id, business_context_id))
            return self._claim_result

        async def update_contact(self, contact_id, data):
            self.updates.append((contact_id, data))
            return {"id": contact_id, **data}

    p = _Provider({"id": UUID, "business_context_id": EOM})
    result = await p.create_contact({
        "phone": "2175550000", "full_name": "Jane",
        "business_context_id": EOM,
    })
    assert p.claim_calls == [(UUID, EOM)]
    assert result["_was_created"] is False


@pytest.mark.asyncio
async def test_create_contact_lost_claim_never_blind_merges():
    """When another tenant wins the claim race, the NULL match is not ours:
    no merge lands on the stolen row (create falls through to insert)."""
    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    class _Provider(DatabaseCRMProvider):
        def __init__(self):
            self.claim_calls = []
            self.updates = []

        async def search_contacts(self, **kwargs):
            if kwargs.get("business_context_id_is_null"):
                return [{"id": UUID, "business_context_id": None,
                         "phone": "2175550000"}]
            return []

        async def claim_contact(self, contact_id, business_context_id):
            self.claim_calls.append((contact_id, business_context_id))
            return None

        async def update_contact(self, contact_id, data):
            self.updates.append((contact_id, data))
            return {"id": contact_id, **data}

    p = _Provider()
    try:
        await p.create_contact({
            "phone": "2175550000", "full_name": "Jane",
            "business_context_id": EOM,
        })
    except Exception:
        pass  # the fresh-insert path needs a live pool; irrelevant here
    assert p.claim_calls == [(UUID, EOM)]
    assert p.updates == []


@pytest.mark.asyncio
async def test_create_contact_non_merging_mode_returns_same_tenant_without_writes():
    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    class _Provider(DatabaseCRMProvider):
        def __init__(self):
            self.searches = []
            self.claim_calls = []
            self.updates = []

        async def search_contacts(self, **kwargs):
            self.searches.append(kwargs)
            if kwargs.get("business_context_id") == EOM:
                return [{
                    "id": UUID,
                    "business_context_id": EOM,
                    "email": "jane@example.com",
                }]
            raise AssertionError("non-merging mode must not query NULL fallback")

        async def claim_contact(self, contact_id, business_context_id):
            self.claim_calls.append((contact_id, business_context_id))
            raise AssertionError("non-merging mode must not claim")

        async def update_contact(self, contact_id, data):
            self.updates.append((contact_id, data))
            raise AssertionError("non-merging mode must not update")

    p = _Provider()
    result = await p.create_contact(
        {
            "phone": "2175550000",
            "email": "jane@example.com",
            "full_name": "Changed Name",
            "business_context_id": EOM,
        },
        merge_existing=False,
    )
    assert result["id"] == UUID
    assert result["_was_created"] is False
    assert p.searches == [{
        "business_context_id": EOM,
        "email": "jane@example.com",
    }]
    assert p.claim_calls == [] and p.updates == []


@pytest.mark.asyncio
async def test_create_contact_non_merging_mode_skips_null_fallback_on_miss():
    from atlas_brain.services.crm_provider import DatabaseCRMProvider
    import atlas_brain.storage.database as database_mod

    class _Provider(DatabaseCRMProvider):
        def __init__(self):
            self.searches = []

        async def search_contacts(self, **kwargs):
            self.searches.append(kwargs)
            if kwargs.get("business_context_id_is_null"):
                raise AssertionError("non-merging mode must not query NULL fallback")
            return []

    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value={"id": "new-eom"})
    previous_pool = database_mod._db_pool
    database_mod._db_pool = pool
    try:
        p = _Provider()
        result = await p.create_contact(
            {
                "phone": "2175550000",
                "email": "new@example.com",
                "full_name": "New",
                "business_context_id": EOM,
            },
            merge_existing=False,
        )
    finally:
        database_mod._db_pool = previous_pool
    assert result["id"] == "new-eom"
    assert result["_was_created"] is True
    assert p.searches == [{
        "business_context_id": EOM,
        "email": "new@example.com",
    }]
    insert_args = pool.fetchrow.await_args.args
    assert insert_args[6] == "2175550000"


@pytest.mark.asyncio
async def test_create_contact_default_still_claims_and_merges_legacy_match():
    """Default compatibility is explicit: omitting the keyword keeps the old
    NULL-tenant claim and merge path."""
    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    class _Provider(DatabaseCRMProvider):
        def __init__(self):
            self.claim_calls = []
            self.updates = []

        async def search_contacts(self, **kwargs):
            if kwargs.get("business_context_id_is_null"):
                return [{
                    "id": UUID,
                    "business_context_id": None,
                    "email": "jane@example.com",
                }]
            return []

        async def claim_contact(self, contact_id, business_context_id):
            self.claim_calls.append((contact_id, business_context_id))
            return {"id": contact_id, "business_context_id": business_context_id}

        async def update_contact(self, contact_id, data):
            self.updates.append((contact_id, data))
            return {"id": contact_id, **data}

    p = _Provider()
    result = await p.create_contact({
        "email": "jane@example.com",
        "full_name": "Jane",
        "business_context_id": EOM,
    })
    assert result["_was_created"] is False
    assert p.claim_calls == [(UUID, EOM)]
    assert p.updates


# ---------------------------------------------------------------------------
# Appointment-fallback scope filter (pure)
# ---------------------------------------------------------------------------


def test_appointments_in_scope_filters_foreign(default_ctx):
    rows = [
        {"id": 1, "business_context_id": EOM},
        {"id": 2, "business_context_id": "churnsignals"},
    ]
    assert [a["id"] for a in crm_srv._appointments_in_scope(rows, None)] == [1]
    assert [a["id"] for a in crm_srv._appointments_in_scope(rows, "churnsignals")] == [2]


def test_appointments_in_scope_unfiltered_without_default(no_default):
    rows = [{"id": 1, "business_context_id": "churnsignals"}]
    assert crm_srv._appointments_in_scope(rows, None) == rows


def test_calls_in_scope_keeps_null_and_same_tenant(default_ctx):
    rows = [
        {"id": 1, "business_context_id": EOM},
        {"id": 2, "business_context_id": None},
        {"id": 3, "business_context_id": "churnsignals"},
    ]
    assert [r["id"] for r in crm_srv._calls_in_scope(rows, None)] == [1, 2]


def test_calls_in_scope_unfiltered_without_default(no_default):
    rows = [{"id": 1, "business_context_id": "churnsignals"}]
    assert crm_srv._calls_in_scope(rows, None) == rows


def test_linked_appointment_query_selects_tenant_column():
    """The scoped filter reads business_context_id off provider rows; the
    SELECT must return it or every scoped result empties (round-3 P2)."""
    src = (REPO / "atlas_brain/services/crm_provider.py").read_text(encoding="utf-8")
    block = src.split("async def get_contact_appointments", 1)[1][:700]
    assert "business_context_id" in block


def test_appointment_repo_accepts_scope_param():
    """Fallback scoping happens in SQL, before LIMIT (round-3 P2)."""
    src = (REPO / "atlas_brain/storage/repositories/appointment.py").read_text(encoding="utf-8")
    for fn in ("async def get_by_phone", "async def search_by_name"):
        sig = src.split(fn, 1)[1][:300]
        assert "business_context_id" in sig, fn


def test_fallback_repo_calls_pass_scope():
    tree = ast.parse((REPO / "atlas_brain/mcp/crm_server.py").read_text(encoding="utf-8"))
    wanted = {"get_by_phone": False, "search_by_name": False}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in wanted
        ):
            if any(k.arg == "business_context_id" for k in node.keywords):
                wanted[node.func.attr] = True
    assert all(wanted.values()), wanted


@pytest.mark.asyncio
async def test_log_interaction_claims_null_contact_under_default(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch, get=LEGACY_NULL)
    out = json.loads(await crm_srv.log_interaction(UUID, interaction_type="note", summary="x"))
    assert out["success"] is True
    provider.claim_contact.assert_awaited_once_with(UUID, EOM)
    provider.log_interaction.assert_awaited()


@pytest.mark.asyncio
async def test_log_interaction_fails_closed_when_claim_lost(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch, get=LEGACY_NULL)
    provider.claim_contact = AsyncMock(return_value=None)
    out = json.loads(await crm_srv.log_interaction(UUID, interaction_type="note", summary="x"))
    assert out["success"] is False
    provider.log_interaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_log_interaction_same_tenant_no_claim(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch, get=SAME)
    await crm_srv.log_interaction(UUID, interaction_type="note", summary="x")
    provider.claim_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_log_interaction_without_default_no_claim(no_default, monkeypatch):
    provider = _provider_mock(monkeypatch, get=LEGACY_NULL)
    await crm_srv.log_interaction(UUID, interaction_type="note", summary="x")
    provider.claim_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_customer_context_scoped_lookup_is_phone_first(default_ctx, monkeypatch):
    """Scoped resolution keeps legacy phone-first semantics: a stale email
    must not veto a valid phone match (round-3 MAJOR)."""
    provider = _provider_mock(monkeypatch)
    await crm_srv.get_customer_context(phone="217-555-0000", email="stale@example.com")
    first = provider.search_contacts.await_args_list[0]
    assert first.kwargs.get("phone") == "217-555-0000"
    assert "email" not in first.kwargs


# ---------------------------------------------------------------------------
# get_customer_context scoping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_customer_context_hides_foreign_uuid(default_ctx, monkeypatch):
    _provider_mock(monkeypatch, get=FOREIGN)
    out = json.loads(await crm_srv.get_customer_context(contact_id=UUID))
    assert out == {"found": False, "context": None}


@pytest.mark.asyncio
async def test_customer_context_name_lookup_is_scoped(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch)
    out = json.loads(await crm_srv.get_customer_context(name="jane"))
    assert out["found"] is False
    calls = provider.search_contacts.await_args_list
    assert calls[0].kwargs["business_context_id"] == EOM
    assert calls[1].kwargs["business_context_id_is_null"] is True


@pytest.mark.asyncio
async def test_customer_context_phone_lookup_is_scoped(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch)
    out = json.loads(await crm_srv.get_customer_context(phone="217-555-0000"))
    assert out == {"found": False, "context": None}
    calls = provider.search_contacts.await_args_list
    assert calls[0].kwargs["business_context_id"] == EOM


@pytest.mark.asyncio
async def test_list_contacts_uses_default(default_ctx, monkeypatch):
    provider = _provider_mock(monkeypatch)
    await crm_srv.list_contacts()
    first = provider.list_contacts.await_args_list[0]
    assert first.kwargs["business_context_id"] == EOM
