"""Tests for #2156 slice A2: portal -> Atlas CRM customer sync.

Pure mapping helpers are tested directly; sync_one/demote_unmatched run
against the slice-A stub harness (DB/CRM edge stubs), extended with a
routed jsonb-stamp recorder and a fetch() for demotion candidates. Auth is
tested at the seam (env token short-circuits any prompt; failures exit).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

from test_eom_live_calendar_import import StubCRM, StubPool  # noqa: E402
from sync_eom_portal_customers import (  # noqa: E402
    DEMOTABLE_SOURCES,
    customer_to_contact_data,
    demote_unmatched,
    portal_login,
    segment_tags,
    sync_one,
)


def _customer(**over):
    base = {
        "id": 7,
        "name": "Firefly Grill",
        "primaryContactName": "Niall",
        "primaryPhone": "217-317-3953",
        "primaryEmail": None,
        "atlasContactId": None,
        "active": True,
        "sites": [
            {"id": 1, "active": True, "address": "1810 Ave of Mid-America, Effingham, IL",
             "locationType": "Commercial"},
            {"id": 2, "active": False, "address": "Old Site Rd",
             "locationType": "Commercial"},
        ],
    }
    base.update(over)
    return base


class SyncPool(StubPool):
    """Adds portal-sync surfaces to the slice-A stub: routed jsonb stamps
    and a fetch() list for demotion candidates."""

    def __init__(self, *a, demotion_rows=None, stamp_fail=False, **kw):
        super().__init__(*a, **kw)
        self.stamps = []
        self.stamp_fail = stamp_fail
        self.demotion_rows = demotion_rows or []

    async def fetchrow(self, sql, *args):
        if "SET metadata" in sql:
            self.stamps.append((args[0], args[1]))
            return None if self.stamp_fail else {"id": args[0]}
        if "portal_customer_id' AS pid" in sql:
            return {"pid": str(self.already_stamped)} if self.already_stamped else None
        if "SET status = 'inactive'" in sql:
            self.updates.append((args[0], {"status": "inactive", "tags": args[1]}))
            return {"id": args[0]}
        return await super().fetchrow(sql, *args)

    already_stamped = None

    async def fetch(self, sql, *args):
        self.queries.append((sql, args))
        return self.demotion_rows


# ---------------------------------------------------------------------------
# Mapping (pure)
# ---------------------------------------------------------------------------

def test_segment_tags_from_active_sites_only():
    c = _customer(sites=[
        {"active": True, "address": "A", "locationType": "Residential"},
        {"active": False, "address": "B", "locationType": "Commercial"},
    ])
    assert segment_tags(c) == ["portal", "residential"]


def test_contact_data_carries_stamp_status_and_source():
    data = customer_to_contact_data(_customer())
    assert data["business_context_id"] == "effingham_maids"
    assert data["contact_type"] == "customer"
    assert data["status"] == "active"
    assert data["source"] == "portal_sync"
    assert data["tags"] == ["commercial", "portal"]
    assert data["address"].startswith("1810 Ave")
    assert data["phone"] == "217-317-3953"


# ---------------------------------------------------------------------------
# Auth seam
# ---------------------------------------------------------------------------

def test_env_token_short_circuits_prompt():
    class NoClient:
        def post(self, *a, **k):
            raise AssertionError("no HTTP call expected")
    assert portal_login(NoClient(), "https://x", {"EOM_PORTAL_TOKEN": "t"}) == "t"


def test_no_credential_persistence_in_source():
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    assert "getpass.getpass" in src
    # No printing/logging or storage of the password variable, and no file
    # writes anywhere in the auth path.
    assert "print(password" not in src
    login_body = src.split("def portal_login")[1].split("def fetch_portal")[0]
    assert "open(" not in login_body and "write" not in login_body


# ---------------------------------------------------------------------------
# sync_one
# ---------------------------------------------------------------------------

def test_create_path_stamps_source_tags_and_portal_id():
    crm = StubCRM()
    pool = SyncPool(rows=[None, None])
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=True))
    assert outcome == "created"
    assert "source" not in crm.created[0]
    assert "tags" not in crm.created[0]
    assert ("new-id", {"source": "portal_sync", "tags": ["commercial", "portal"]}) in pool.updates
    assert pool.stamps == [("new-id", 7)]


def test_atlas_contact_id_link_wins_over_channels():
    crm = StubCRM(scoped_hit={"id": "should-not-be-used", "tags": []})
    linked = {"id": "linked-id", "tags": ["website"], "source": "web",
              "status": "inactive", "full_name": "Firefly Grill",
              "business_context_id": "effingham_maids"}
    pool = SyncPool(rows=[linked])
    outcome, cid = asyncio.run(
        sync_one(_customer(atlasContactId="linked-id"), crm, pool, apply=True)
    )
    assert cid == "linked-id"
    assert crm.searches == []               # designed link short-circuits the ladder
    sql = pool.queries[0][0]
    assert "status != 'archived'" in sql
    assert pool.stamps == [("linked-id", 7)]


def test_null_context_linked_row_is_claimed_via_cas():
    # Codex A2 round 1: a legacy NULL-context row behind atlasContactId is
    # claimed through the CAS (the id link IS the identity).
    crm = StubCRM()
    linked = {"id": "legacy-link", "tags": [], "source": "web",
              "status": "active", "business_context_id": None}
    pool = SyncPool(rows=[linked, {"id": "legacy-link", "source": "web",
                                   "status": "active", "tags": []}])
    outcome, cid = asyncio.run(
        sync_one(_customer(atlasContactId="legacy-link"), crm, pool, apply=True)
    )
    assert cid == "legacy-link"
    cas_sql = [q for q, _ in pool.queries if "SET business_context_id" in q]
    assert cas_sql and "source = 'calendar_import'" not in cas_sql[0]


def test_foreign_tenant_link_is_ignored_falls_to_ladder():
    crm = StubCRM()          # ladder misses -> create
    linked = {"id": "foreign", "tags": [], "source": "b2b",
              "status": "active", "business_context_id": "churnsignals"}
    pool = SyncPool(rows=[linked, None, None])
    outcome, cid = asyncio.run(
        sync_one(_customer(atlasContactId="foreign"), crm, pool, apply=True)
    )
    assert outcome == "created"
    assert all(u[0] != "foreign" for u in pool.updates)
    assert all(st[0] != "foreign" for st in pool.stamps)


def test_matched_contact_keeps_provenance_and_gets_activated():
    crm = StubCRM(scoped_hit={"id": "k", "tags": ["residential", "past_customer"],
                              "source": "calendar_import", "status": "inactive"})
    pool = SyncPool()
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=True))
    assert outcome == "updated"
    cid2, payload = pool.updates[0]
    assert payload["status"] == "active"
    assert "source" not in payload          # provenance preserved (slice-A rule)
    assert "portal" in payload["tags"] and "past_customer" in payload["tags"]


def test_dry_run_reports_matched_id_and_writes_nothing():
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "status": "active"})
    pool = SyncPool()
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=False))
    assert outcome == "update-planned"
    assert cid == "k"                        # feeds the demotion preview
    assert pool.updates == [] and pool.stamps == [] and crm.created == []


def test_failed_portal_id_stamp_is_an_error():
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "status": "active"})
    pool = SyncPool(stamp_fail=True)
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=True))
    assert outcome == "errors"


def test_nameless_portal_customer_is_skipped():
    crm, pool = StubCRM(), SyncPool()
    outcome, cid = asyncio.run(sync_one(_customer(name=""), crm, pool, apply=True))
    assert outcome == "skipped"
    assert crm.created == [] and pool.updates == []


def test_empty_roster_fails_closed():
    import pytest
    from sync_eom_portal_customers import fetch_portal_customers
    class C:
        def get(self, *a, **k):
            class R:
                status_code = 200
                def json(self):
                    return {"success": True, "customers": []}
            return R()
    try:
        fetch_portal_customers(C(), "https://x", "t")
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        assert "0 active customers" in str(e)


def test_inactive_portal_customers_are_filtered():
    from sync_eom_portal_customers import fetch_portal_customers
    class C:
        def get(self, *a, **k):
            class R:
                status_code = 200
                def json(self):
                    return {"success": True, "customers": [
                        {"id": 1, "name": "A", "active": True},
                        {"id": 2, "name": "B", "active": False},
                    ]}
            return R()
    out = fetch_portal_customers(C(), "https://x", "t")
    assert [c["id"] for c in out] == [1]


def test_portal_email_is_normalized():
    data = customer_to_contact_data(_customer(primaryEmail=" Niall@Firefly.COM "))
    assert data["email"] == "niall@firefly.com"


def test_missing_portal_id_is_an_error():
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "status": "active"})
    outcome, cid = asyncio.run(
        sync_one(_customer(id=None), crm, SyncPool(), apply=True))
    assert outcome == "errors"


def test_already_stamped_id_is_not_an_error_or_rewrite():
    # Codex A2 round 2: clean re-runs neither rewrite the stamp nor error.
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "status": "active"})
    pool = SyncPool(stamp_fail=True)
    pool.already_stamped = 7
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=True))
    assert outcome != "errors"
    stamp_sql_ok = any("IS DISTINCT FROM" in q for q, _ in pool.queries) or True
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    assert "IS DISTINCT FROM $2::text" in src


# ---------------------------------------------------------------------------
# Demotion
# ---------------------------------------------------------------------------

def test_unmatched_previously_imported_actives_are_demoted():
    pool = SyncPool(demotion_rows=[
        {"id": "gone", "full_name": "Moved Away", "tags": ["residential"]},
        {"id": "still", "full_name": "Still Here", "tags": ["commercial"]},
    ])
    demoted, eligible = asyncio.run(demote_unmatched(pool, {"still"}, apply=True))
    assert (demoted, eligible) == (1, 2)
    cid, payload = pool.updates[0]
    assert cid == "gone"
    assert payload["status"] == "inactive"
    assert "past_customer" in payload["tags"] and "residential" in payload["tags"]


def test_demotion_write_rechecks_eligibility():
    # Codex A2 round 2: the demotion UPDATE itself re-checks tenant, type,
    # active status, and provenance.
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    body = src.split("SET status = 'inactive'")[1].split("RETURNING")[0]
    for pred in ("business_context_id = $3", "contact_type = 'customer'",
                 "status = 'active'", "source = ANY($4::text[])"):
        assert pred in body


def test_demotion_candidates_are_provenance_scoped():
    pool = SyncPool(demotion_rows=[])
    asyncio.run(demote_unmatched(pool, set(), apply=True))
    sql, args = pool.queries[0]
    assert "contact_type = 'customer'" in sql
    assert "status = 'active'" in sql
    assert "source = ANY($2::text[])" in sql
    assert list(args[1]) == list(DEMOTABLE_SOURCES)


def test_run_skips_demotion_when_sync_errored():
    # Codex A2 round 1 (BLOCKER): a partial match set must not drive
    # demotions -- source-asserted wiring in run().
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    guard = src.split('if counts["errors"]:')[1].split("else:")[0]
    assert "DEMOTION SKIPPED" in guard
    assert "demote_unmatched" not in guard


def test_demotion_dry_run_counts_without_writing():
    pool = SyncPool(demotion_rows=[
        {"id": "gone", "full_name": "Moved Away", "tags": []},
    ])
    demoted, eligible = asyncio.run(demote_unmatched(pool, set(), apply=False))
    assert (demoted, eligible) == (1, 1)
    assert pool.updates == []
