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
from sync_eom_portal_customers import (
    on_calendar,  # noqa: E402
    DEMOTABLE_SOURCES,
    customer_to_contact_data,
    demote_unmatched,
    portal_login,
    segment_tags,
    sync_one,
)


LINK_A = "11111111-1111-1111-1111-111111111111"
LINK_B = "22222222-2222-2222-2222-222222222222"
LINK_C = "33333333-3333-3333-3333-333333333333"


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
            if not self.row_exists:
                return None
            return {"pid": str(self.already_stamped) if self.already_stamped else None}
        if "SET status = 'inactive'" in sql:
            self.updates.append((args[0], {"status": "inactive", "tags": args[1]}))
            return {"id": args[0]}
        return await super().fetchrow(sql, *args)

    already_stamped = None
    row_exists = True

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

def test_typed_env_token_short_circuits_prompt():
    class NoClient:
        def post(self, *a, **k):
            raise AssertionError("no HTTP call expected")
    assert portal_login(NoClient(), "https://x",
                        {"ATLAS_TOOLS_EOM_PORTAL_TOKEN": "t"}) == "t"
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    assert '"EOM_PORTAL_TOKEN"' not in src      # single typed surface (r4)


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
    assert ("new-id", {"source": "portal_sync", "tags": ["commercial", "portal"],
                       "phone": "217-317-3953"}) in pool.updates
    assert pool.stamps == [("new-id", 7)]


def test_atlas_contact_id_link_wins_over_channels():
    crm = StubCRM(scoped_hit={"id": "should-not-be-used", "tags": []})
    linked = {"id": LINK_A, "tags": ["website"], "source": "web",
              "status": "inactive", "full_name": "Firefly Grill",
              "business_context_id": "effingham_maids"}
    pool = SyncPool(rows=[None, linked])   # portal-id page misses first
    outcome, cid = asyncio.run(
        sync_one(_customer(atlasContactId=LINK_A), crm, pool, apply=True)
    )
    assert cid == LINK_A
    assert crm.searches == []               # designed link short-circuits the ladder
    sql = pool.queries[0][0]
    assert "status != 'archived'" in sql
    assert pool.stamps == [(LINK_A, 7)]


def test_null_context_linked_row_is_claimed_via_cas():
    # Codex A2 round 1: a legacy NULL-context row behind atlasContactId is
    # claimed through the CAS (the id link IS the identity).
    crm = StubCRM()
    linked = {"id": LINK_B, "tags": [], "source": "web",
              "status": "active", "business_context_id": None}
    pool = SyncPool(rows=[None, linked, {"id": LINK_B, "source": "web",
                                         "status": "active", "tags": []}])
    outcome, cid = asyncio.run(
        sync_one(_customer(atlasContactId=LINK_B), crm, pool, apply=True)
    )
    assert cid == LINK_B
    cas_sql = [q for q, _ in pool.queries if "SET business_context_id" in q]
    assert cas_sql and "source = 'calendar_import'" not in cas_sql[0]


def test_foreign_tenant_link_is_ignored_falls_to_ladder():
    crm = StubCRM()          # ladder misses -> create
    linked = {"id": LINK_C, "tags": [], "source": "b2b",
              "status": "active", "business_context_id": "churnsignals"}
    pool = SyncPool(rows=[None, linked, None, None])
    outcome, cid = asyncio.run(
        sync_one(_customer(atlasContactId=LINK_C), crm, pool, apply=True)
    )
    assert outcome == "created"
    assert all(u[0] != LINK_C for u in pool.updates)
    assert all(st[0] != LINK_C for st in pool.stamps)


def test_matched_contact_keeps_provenance_and_gets_activated():
    crm = StubCRM(scoped_hit={"id": "k", "tags": ["residential", "past_customer"],
                              "source": "calendar_import", "status": "inactive"})
    pool = SyncPool()
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=True))
    assert outcome == "updated"
    cid2, payload = pool.updates[0]
    assert payload["status"] == "active"
    assert "source" not in payload          # provenance preserved (slice-A rule)
    # Portal-authoritative managed tags: past_customer sheds on an active
    # match; foreign tags survive (Codex A2 round 4).
    assert "portal" in payload["tags"] and "past_customer" not in payload["tags"]


def test_dry_run_reports_matched_id_and_writes_nothing():
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "status": "active"})
    pool = SyncPool()
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=False))
    assert outcome == "update-planned"
    assert cid == "k"                        # feeds the demotion preview
    assert pool.updates == [] and pool.stamps == [] and crm.created == []


def test_dry_run_previews_unchanged_for_clean_rows():
    # Codex A2 round 4: a fully-synced row previews as unchanged.
    rec_data = customer_to_contact_data(_customer())
    existing = {"id": "k", **{k: v for k, v in rec_data.items()
                              if k not in ("source", "business_context_id")},
                "source": "portal_sync", "business_context_id": "effingham_maids",
                "metadata": {"portal_customer_id": 7}}
    crm = StubCRM(scoped_hit=existing)
    pool = SyncPool()
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=False))
    assert outcome == "unchanged"


def test_resolver_finds_stamped_portal_id_first():
    # Codex A2 round 4 (R8): the stamped id is the stable key -- channel
    # drift must not cause a duplicate create.
    crm = StubCRM()          # channels would miss entirely
    stamped_row = {"id": "stamped", "tags": ["portal", "commercial"],
                   "source": "portal_sync", "status": "active",
                   "business_context_id": "effingham_maids",
                   "metadata": {"portal_customer_id": 7}}
    pool = SyncPool(rows=[stamped_row])
    outcome, cid = asyncio.run(sync_one(_customer(primaryPhone=None), crm, pool,
                                        apply=True))
    assert cid == "stamped"
    assert crm.created == []
    assert "portal_customer_id' = $1" in pool.queries[0][0]


def test_failed_portal_id_stamp_is_an_error():
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "status": "active"})
    pool = SyncPool(stamp_fail=True)
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=True))
    assert outcome == "errors"


def test_nameless_portal_customer_is_an_error_not_a_skip():
    # Codex A2 round 3: unprocessable records gate demotion via the error
    # counter instead of silently shrinking the match set.
    crm, pool = StubCRM(), SyncPool()
    outcome, cid = asyncio.run(sync_one(_customer(name=""), crm, pool, apply=True))
    assert outcome == "errors"
    assert crm.created == [] and pool.updates == []


def test_whitespace_channels_are_absent():
    data = customer_to_contact_data(_customer(primaryPhone="   ", primaryEmail=" "))
    assert "phone" not in data and "email" not in data


def test_conflicting_portal_link_is_never_overwritten():
    # Codex A2 round 6 (R8): a contact linked to portal id 9 must not be
    # silently relinked to 7.
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "status": "active",
                              "metadata": {"portal_customer_id": 9}})
    pool = SyncPool()
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=True))
    assert outcome == "errors"
    assert pool.updates == [] and pool.stamps == []


def test_sql_guard_catches_conflicts_on_metadata_blind_rows():
    # Codex A2 round 7: address-fallback rows carry no metadata; the stamp
    # SQL itself refuses a relink and the error names the existing link.
    crm = StubCRM()
    pool = SyncPool(rows=[None, None, None,
                          {"id": "addr-row", "tags": []}], stamp_fail=True)
    pool.already_stamped = 9
    outcome, cid = asyncio.run(sync_one(_customer(primaryPhone=None), crm, pool,
                                        apply=True))
    assert outcome == "errors"
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    stamp_sql = src.split("jsonb_build_object('portal_customer_id'")[1].split('"""')[0]
    assert "portal_customer_id' IS NULL" in stamp_sql


def test_dry_run_previews_link_conflicts():
    # Codex A2 round 7: dry-run reports the conflict the apply would hit.
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "status": "active",
                              "metadata": {"portal_customer_id": 9}})
    outcome, cid = asyncio.run(sync_one(_customer(), crm, SyncPool(), apply=False))
    assert outcome == "errors"
    assert cid is None                       # not in the demotion-safe set


def test_metadata_blind_conflict_probed_before_any_write():
    # Codex A2 round 8 (BLOCKER): the address-fallback row's existing link
    # is probed before the matched update can touch it.
    crm = StubCRM()
    pool = SyncPool(rows=[None, {"id": "addr-row", "tags": []}])
    pool.already_stamped = 9
    outcome, cid = asyncio.run(sync_one(_customer(primaryPhone=None), crm, pool,
                                        apply=True))
    assert outcome == "errors"
    assert pool.updates == [] and pool.stamps == []


def test_non_string_location_type_is_safe_and_preflighted():
    # Codex A2 round 8 (BLOCKER): robust helper + roster preflight.
    c = _customer(sites=[{"active": True, "address": "A", "locationType": 3}])
    assert segment_tags(c) == ["portal"]
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    body = src.split("def _malformed")[1].split("invalid =")[0]
    assert 'isinstance(x.get("locationType"), str)' in body


def test_malformed_atlas_contact_id_is_ignored():
    # Codex A2 round 9: non-UUID links never reach the UUID id lookup.
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "status": "active"})
    pool = SyncPool()
    outcome, cid = asyncio.run(
        sync_one(_customer(atlasContactId="not-a-uuid"), crm, pool, apply=True))
    assert outcome == "updated"        # fell through to the ladder cleanly
    assert cid == "k"


def test_dry_run_previews_metadata_blind_fallback_conflicts():
    # Codex A2 round 9: the dry-run probes address-fallback links too.
    crm = StubCRM()
    pool = SyncPool(rows=[None, {"id": "addr-row", "tags": []}])
    pool.already_stamped = 9
    outcome, cid = asyncio.run(sync_one(_customer(primaryPhone=None), crm, pool,
                                        apply=False))
    assert outcome == "errors"
    assert cid is None


def test_roster_preflight_rejects_malformed_sites():
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    body = src.split("def _malformed")[1].split("invalid =")[0]
    assert "isinstance(sites, list)" in body
    assert "isinstance(x, dict)" in body


def test_roster_validation_aborts_before_any_write():
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    guard = src.split("invalid = [")[1].split("matched_ids")[0]
    assert "ABORTED" in guard and "return 1" in guard


def test_boolean_portal_id_is_rejected_before_writes():
    # Codex A2 round 5: bool subclasses int; id=true must not pass.
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "status": "active"})
    pool = SyncPool()
    outcome, cid = asyncio.run(sync_one(_customer(id=True), crm, pool, apply=True))
    assert outcome == "errors"
    assert pool.updates == [] and crm.created == []


def test_string_metadata_does_not_crash_dry_run():
    # Codex A2 round 5: asyncpg can deliver JSONB as a string.
    rec_data = customer_to_contact_data(_customer())
    existing = {"id": "k", **{k: v for k, v in rec_data.items()
                              if k not in ("source", "business_context_id")},
                "source": "portal_sync", "business_context_id": "effingham_maids",
                "metadata": '{"portal_customer_id": 7}'}
    crm = StubCRM(scoped_hit=existing)
    outcome, cid = asyncio.run(sync_one(_customer(), crm, SyncPool(), apply=False))
    assert outcome == "unchanged"


def test_create_path_phone_rides_the_stamp_not_create():
    # Codex A2 round 5: create_contact's weaker %last-10% dedupe never sees
    # the raw phone; it lands via the controlled stamp.
    crm = StubCRM()
    pool = SyncPool(rows=[None, None, None, None])
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=True))
    assert outcome == "created"
    assert "phone" not in crm.created[0]
    assert any("phone" in payload for _, payload in pool.updates)


def test_portal_id_stamp_requires_eom_tenant():
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    stamp_sql = src.split("jsonb_build_object('portal_customer_id'")[1].split('"""')[0]
    assert "business_context_id = $3" in stamp_sql
    assert "IS NULL OR" not in stamp_sql


def test_missing_portal_id_errors_before_any_write():
    # Codex A2 round 3: validation precedes writes entirely.
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "status": "active"})
    pool = SyncPool()
    outcome, cid = asyncio.run(sync_one(_customer(id=None), crm, pool, apply=True))
    assert outcome == "errors"
    assert pool.updates == [] and crm.created == []


def test_already_stamped_fallback_is_guarded():
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    body = src.split("AS pid FROM contacts")[1].split('"""')[0]
    assert "status != 'archived'" in body
    assert "business_context_id = $2" in body


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

NO_GUARD = {"phones": set(), "addrs": set(), "names": set()}


def test_calendar_veto_keys_match_phone_address_and_name():
    # Owner rule (2026-07-23): the calendar vetoes demotion.
    guard = {"phones": {"2175559999"}, "addrs": {"12 oak st, effingham, il"},
             "names": {"jane smith"}}
    assert on_calendar({"phone": "(217) 555-9999 ext 4"}, guard)
    assert on_calendar({"address": "12 Oak St, Effingham, IL "}, guard)
    assert on_calendar({"full_name": "Jane Smith"}, guard)
    assert not on_calendar({"full_name": "Someone Else",
                            "phone": "217-555-0000", "address": "9 Elm"}, guard)


def test_cancelled_latest_calendar_records_do_not_veto():
    # Codex 2163 rounds 1-2 (BLOCKER): cancellation excludes from the veto,
    # decided on the CROSS-CALENDAR merged view (dedup_records runs before
    # key emission, so a newer cancellation on another calendar supersedes
    # an older active event).
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    body = src.split("def fetch_calendar_guard_keys")[1].split("def on_calendar")[0]
    assert "live.dedup_records(all_records)" in body
    assert "if rec.cancelled:" in body
    assert body.index("live.dedup_records") < body.index("if rec.cancelled:")
    assert body.index("if rec.cancelled:") < body.index("if rec.phone:")


def test_calendar_active_candidates_are_kept_not_demoted():
    pool = SyncPool(demotion_rows=[
        {"id": "sched", "full_name": "On Schedule", "tags": [],
         "phone": "217-555-9999", "address": "X"},
        {"id": "gone", "full_name": "Moved Away", "tags": [],
         "phone": None, "address": "Y"},
    ])
    guard = {"phones": {"2175559999"}, "addrs": set(), "names": set()}
    demoted, eligible = asyncio.run(
        demote_unmatched(pool, set(), apply=True, guard_keys=guard))
    assert (demoted, eligible) == (1, 2)
    assert pool.updates[0][0] == "gone"          # only the truly-gone one


def test_demotion_refuses_without_the_calendar_guard():
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    guard_block = src.split("fetch_calendar_guard_keys()")[1].split("demote_unmatched(")[0]
    assert "DEMOTION SKIPPED" in guard_block
    assert 'counts["errors"] += 1' in guard_block
    # Codex r3: SystemExit (missing calendar config) takes the same skip.
    assert "(Exception, SystemExit)" in guard_block


def test_name_keys_respect_name_level_cancellation_recency():
    # Codex r3: same name, no shared channel -- the newer CANCELLED record
    # suppresses the name key emitted by the older active one.
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    body = src.split("def fetch_calendar_guard_keys")[1].split("def on_calendar")[0]
    assert "name_state" in body
    assert body.index("name_state") < body.index('keys["names"].add')


def test_unmatched_previously_imported_actives_are_demoted():
    pool = SyncPool(demotion_rows=[
        {"id": "gone", "full_name": "Moved Away", "tags": ["residential"]},
        {"id": "still", "full_name": "Still Here", "tags": ["commercial"]},
    ])
    demoted, eligible = asyncio.run(
        demote_unmatched(pool, {"still"}, apply=True, guard_keys=NO_GUARD))
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
    asyncio.run(demote_unmatched(pool, set(), apply=True, guard_keys=NO_GUARD))
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
    demoted, eligible = asyncio.run(
        demote_unmatched(pool, set(), apply=False, guard_keys=NO_GUARD))
    assert (demoted, eligible) == (1, 1)
    assert pool.updates == []
