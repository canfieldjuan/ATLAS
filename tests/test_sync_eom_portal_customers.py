"""Tests for #2156 slice A2: portal -> Atlas CRM customer sync.

Pure mapping helpers are tested directly; sync_one/demote_unmatched run
against the slice-A stub harness (DB/CRM edge stubs), extended with a
routed jsonb-stamp recorder and a fetch() for demotion candidates. Auth is
tested at the seam (env token short-circuits any prompt; failures exit).
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

from test_eom_live_calendar_import import StubCRM, StubPool  # noqa: E402
from sync_eom_portal_customers import (  # noqa: E402
    DEMOTABLE_SOURCES,
    MANAGED_TAGS,
    customer_to_contact_data,
    demote_unmatched,
    fetch_calendar_guard_keys,
    on_calendar,
    preflight_roster,
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

    def __init__(
        self,
        *a,
        demotion_rows=None,
        stamp_fail=False,
        atomic_changed=True,
        **kw,
    ):
        super().__init__(*a, **kw)
        self.stamps = []
        self.atomic_writes = []
        self.stamp_fail = stamp_fail
        self.atomic_changed = atomic_changed
        self.demotion_rows = demotion_rows or []

    async def fetchrow(self, sql, *args):
        if "jsonb_build_object('portal_customer_id'" in sql:
            self.stamps.append((args[0], args[1]))
            self.atomic_writes.append((sql, args))
            if self.stamp_fail:
                return None
            import re as _re
            payload = {}
            set_clause = sql.split("UPDATE contacts", 1)[1].split("FROM live", 1)[0]
            for column, param in _re.findall(
                r"(\w+) = \$(\d+)", set_clause
            ):
                if column not in {"updated_at", "business_context_id"}:
                    payload[column] = args[int(param) - 1]
            tags_param = _re.search(
                r"COALESCE\(contacts\.tags.*?\|\| \$(\d+)::text\[\]",
                set_clause,
                _re.DOTALL,
            )
            if tags_param:
                payload["tags"] = args[int(tags_param.group(1)) - 1]
            if payload and self.atomic_changed:
                self.updates.append((args[0], payload))
            return {"id": args[0], "changed": self.atomic_changed}
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
    assert crm.created[0]["source"] == "portal_sync"
    assert crm.created[0]["tags"] == ["commercial", "portal"]
    assert crm.created[0]["phone"] == "217-317-3953"
    assert crm.created[0]["metadata"] == {"portal_customer_id": 7}
    assert pool.updates == [] and pool.stamps == []


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
    assert len(pool.atomic_writes) == 1


def test_null_context_linked_row_is_claimed_in_atomic_reconciliation():
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
    assert len(pool.atomic_writes) == 1
    atomic_sql = pool.atomic_writes[0][0]
    assert "(business_context_id IS NULL OR business_context_id = $3)" in atomic_sql
    assert "source = 'calendar_import'" not in atomic_sql


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


def test_rejected_atomic_reconciliation_is_an_error():
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "status": "active"})
    pool = SyncPool(stamp_fail=True)
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=True))
    assert outcome == "errors"
    assert pool.updates == []
    assert len(pool.atomic_writes) == 1


def test_provider_create_race_uses_non_merging_mode_and_rejection_writes_nothing():
    class RaceCRM(StubCRM):
        async def create_contact(self, data, *, merge_existing=True):
            # merge_existing=False is a deliberate, load-bearing workaround, not
            # an oversight -- do NOT "fix" this assertion by flipping it. It is
            # the portal-reconciliation RACE SEAM (crm_provider.create_contact
            # docstring): resolve_contact already owns identity resolution across
            # its full ladder BEFORE this call, so create_contact must be a
            # race-safe clean insert, not a second resolver. merge_existing=True
            # would re-run resolution and MERGE fields in the window between
            # resolve and create, risking a cross-link to a concurrently-created
            # row and overwriting fields the clean path leaves alone.
            #
            # NOT a matcher-strength workaround: the portal resolver's phone rung
            # uses the SAME crm.search_contacts as the create path (via
            # live._search_channel), so 0B's matcher work (ATLAS #2313) is
            # orthogonal and does not obsolete this seam. Website #127 (D4),
            # corrected after Codex R1/R14 on 2026-08-07.
            assert merge_existing is False
            self.created.append(data)
            return {
                "id": "race-row",
                "_was_created": False,
                "business_context_id": "effingham_maids",
                "email": data["email"],
                "tags": [],
                "status": "active",
            }

    customer = _customer(primaryPhone=None, primaryEmail="race@example.com", sites=[])
    pool = SyncPool(rows=[None], stamp_fail=True)
    outcome, cid = asyncio.run(sync_one(customer, RaceCRM(), pool, apply=True))
    assert (outcome, cid) == ("errors", None)
    assert pool.updates == []
    assert len(pool.atomic_writes) == 1
    sql, args = pool.atomic_writes[0]
    assert "LOWER(email) = LOWER" in sql
    assert args[1] == 7


def test_atomic_reconciliation_identity_drift_has_zero_writes():
    existing = {
        "id": "email-row",
        "business_context_id": "effingham_maids",
        "email": "race@example.com",
        "tags": [],
        "status": "active",
    }
    crm = StubCRM(scoped_hit=existing)
    pool = SyncPool(stamp_fail=True)
    outcome, cid = asyncio.run(sync_one(
        _customer(primaryPhone=None, primaryEmail="race@example.com", sites=[]),
        crm,
        pool,
        apply=True,
    ))
    assert (outcome, cid) == ("errors", None)
    assert pool.updates == []
    assert "LOWER(email) = LOWER" in pool.atomic_writes[0][0]


def test_atomic_reconciliation_preserves_live_foreign_tags_and_source():
    existing = {
        "id": "tag-row",
        "business_context_id": "effingham_maids",
        "phone": "217-317-3953",
        "tags": ["past_customer", "vip"],
        "source": "calendar_import",
        "status": "inactive",
    }
    pool = SyncPool()
    outcome, _ = asyncio.run(sync_one(
        _customer(),
        StubCRM(scoped_hit=existing),
        pool,
        apply=True,
    ))
    assert outcome == "updated"
    sql, args = pool.atomic_writes[0]
    set_clause = sql.split("UPDATE contacts", 1)[1].split("FROM live", 1)[0]
    assert "COALESCE(contacts.tags" in set_clause
    assert "source =" not in set_clause
    assert ["commercial", "portal"] in args
    assert sorted(MANAGED_TAGS) in args


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
    # Address-fallback rows carry no metadata in the resolver result; the
    # first and only mutation still carries the portal-link predicate.
    crm = StubCRM()
    pool = SyncPool(
        rows=[None, {"id": "addr-row", "tags": []}],
        stamp_fail=True,
    )
    pool.already_stamped = 9
    outcome, cid = asyncio.run(sync_one(_customer(primaryPhone=None), crm, pool,
                                        apply=True))
    assert outcome == "errors"
    atomic_sql = pool.atomic_writes[0][0]
    assert "portal_customer_id' IS NULL" in atomic_sql
    assert pool.updates == []


def test_dry_run_previews_link_conflicts():
    # Codex A2 round 7: dry-run reports the conflict the apply would hit.
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "status": "active",
                              "metadata": {"portal_customer_id": 9}})
    outcome, cid = asyncio.run(sync_one(_customer(), crm, SyncPool(), apply=False))
    assert outcome == "errors"
    assert cid is None                       # not in the demotion-safe set


def test_metadata_blind_conflict_rejection_has_zero_writes():
    # The address-fallback result is metadata-blind, so emulate the conditional
    # UPDATE rejecting its conflicting live link. No field mutation lands.
    crm = StubCRM()
    pool = SyncPool(
        rows=[None, {"id": "addr-row", "tags": []}],
        stamp_fail=True,
    )
    outcome, cid = asyncio.run(sync_one(_customer(primaryPhone=None), crm, pool,
                                        apply=True))
    assert outcome == "errors"
    assert pool.updates == []
    assert len(pool.atomic_writes) == 1


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


def test_roster_preflight_rejects_duplicate_normalized_portal_identity():
    customers = [
        _customer(id=7, name="A", primaryPhone="(217) 555-1234", sites=[]),
        _customer(id=8, name="B", primaryPhone="217-555-1234 ext 9", sites=[]),
    ]
    resolutions, errors = asyncio.run(
        preflight_roster(customers, StubCRM(), SyncPool(rows=[None, None]))
    )
    assert set(resolutions) == {7, 8}
    assert any("share normalized phone identity" in error for error in errors)


def test_roster_preflight_rejects_two_portal_rows_resolving_to_one_contact():
    shared = {
        "id": "same-contact",
        "business_context_id": "effingham_maids",
        "tags": [],
        "status": "active",
    }
    customers = [
        _customer(id=7, name="A", primaryPhone="217-555-1000", sites=[]),
        _customer(id=8, name="B", primaryPhone="217-555-2000", sites=[]),
    ]
    _, errors = asyncio.run(
        preflight_roster(customers, StubCRM(scoped_hit=shared), SyncPool())
    )
    assert any("resolve to CRM contact same-contact" in error for error in errors)


def test_apply_entrypoint_aborts_roster_collision_before_sync(monkeypatch):
    import httpx
    import atlas_brain.services.crm_provider as crm_provider_mod
    import atlas_brain.storage.database as database_mod
    import sync_eom_portal_customers as sync_mod

    customers = [
        _customer(id=7, name="A", primaryPhone="217-555-1000", sites=[]),
        _customer(id=8, name="B", primaryPhone="217-555-1000", sites=[]),
    ]

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Pool(SyncPool):
        async def initialize(self):
            return None

    pool = Pool(rows=[None, None])
    class Receipt:
        def __init__(self):
            self.counts = []

        def record_outcome_counts(self, counts):
            self.counts.append(dict(counts))

    receipt = Receipt()
    monkeypatch.setattr(httpx, "Client", Client)
    monkeypatch.setattr(sync_mod, "portal_login", lambda *_args: "token")
    monkeypatch.setattr(
        sync_mod,
        "fetch_portal_customers",
        lambda *_args: customers,
    )
    monkeypatch.setattr(crm_provider_mod, "get_crm_provider", StubCRM)
    previous_pool = database_mod._db_pool
    database_mod._db_pool = pool

    sync_calls = []

    async def forbidden_sync(*args, **kwargs):
        sync_calls.append((args, kwargs))
        raise AssertionError("collision preflight must abort before sync")

    monkeypatch.setattr(sync_mod, "sync_one", forbidden_sync)
    try:
        result = asyncio.run(sync_mod.run(SimpleNamespace(
            apply=True,
            base_url="https://example.invalid",
        ), receipt=receipt))
    finally:
        database_mod._db_pool = previous_pool
    assert result == 1
    assert sync_calls == []
    assert pool.updates == [] and pool.atomic_writes == []
    assert receipt.counts[-1]["errors"] == 1


def test_apply_run_persists_calendar_guard_error_counts(monkeypatch):
    import httpx
    import atlas_brain.services.crm_provider as crm_provider_mod
    import atlas_brain.storage.database as database_mod
    import sync_eom_portal_customers as sync_mod

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Pool(SyncPool):
        async def initialize(self):
            return None

    class Receipt:
        def __init__(self):
            self.counts = []

        def record_outcome_counts(self, counts):
            self.counts.append(dict(counts))

    async def unavailable_guard():
        raise RuntimeError("calendar offline")

    pool = Pool(rows=[])
    receipt = Receipt()
    monkeypatch.setattr(httpx, "Client", Client)
    monkeypatch.setattr(sync_mod, "portal_login", lambda *_args: "token")
    monkeypatch.setattr(sync_mod, "fetch_portal_customers", lambda *_args: [])
    monkeypatch.setattr(sync_mod, "fetch_calendar_guard_keys", unavailable_guard)
    monkeypatch.setattr(crm_provider_mod, "get_crm_provider", StubCRM)
    previous_pool = database_mod._db_pool
    database_mod._db_pool = pool
    try:
        result = asyncio.run(sync_mod.run(
            SimpleNamespace(apply=True, base_url="https://example.invalid"),
            receipt=receipt,
        ))
    finally:
        database_mod._db_pool = previous_pool

    assert result == 1
    assert receipt.counts[-1]["errors"] == 1


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


def test_create_path_inserts_phone_without_provider_phone_dedup():
    # Non-merging mode skips provider phone dedup but preserves phone in the
    # initial INSERT payload.
    crm = StubCRM()
    pool = SyncPool(rows=[None, None, None, None])
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=True))
    assert outcome == "created"
    assert crm.created[0]["phone"] == "217-317-3953"
    assert pool.updates == []


def test_atomic_reconciliation_requires_or_claims_eom_tenant():
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    body = src.split("async def reconcile_portal_contact")[1].split(
        "async def portal_id_current"
    )[0]
    assert 'else "business_context_id = $3"' in body
    assert "(business_context_id IS NULL OR business_context_id = $3)" in body


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
    crm = StubCRM(scoped_hit={
        "id": "k",
        "tags": ["commercial", "portal"],
        "status": "active",
        "full_name": "Firefly Grill",
        "address": "1810 Ave of Mid-America, Effingham, IL",
        "contact_type": "customer",
        "phone": "217-317-3953",
        "notes": "Contact: Niall",
        "business_context_id": "effingham_maids",
        "metadata": {"portal_customer_id": 7},
    })
    pool = SyncPool(atomic_changed=False)
    outcome, cid = asyncio.run(sync_one(_customer(), crm, pool, apply=True))
    assert outcome == "unchanged"
    assert len(pool.atomic_writes) == 1
    assert pool.updates == []


def test_clean_preflight_snapshot_still_reconciles_against_locked_live_row():
    data = customer_to_contact_data(_customer())
    existing = {
        "id": "live-drift",
        **{key: value for key, value in data.items() if key != "source"},
        "source": "calendar_import",
        "business_context_id": "effingham_maids",
        "metadata": {"portal_customer_id": 7},
    }
    pool = SyncPool()

    outcome, cid = asyncio.run(
        sync_one(_customer(), StubCRM(scoped_hit=existing), pool, apply=True)
    )

    assert (outcome, cid) == ("updated", "live-drift")
    sql, args = pool.atomic_writes[0]
    assert "WITH live AS MATERIALIZED" in sql
    assert "FOR UPDATE" in sql
    assert "contacts.full_name IS DISTINCT FROM" in sql
    assert data["full_name"] in args


# ---------------------------------------------------------------------------
# Demotion
# ---------------------------------------------------------------------------

NO_GUARD = {
    "phones": set(), "emails": set(), "addrs": set(), "names": set(),
}


def test_calendar_veto_keys_match_phone_email_address_and_name():
    # Owner rule (2026-07-23): the calendar vetoes demotion.
    guard = {"phones": {"2175559999"}, "addrs": {"12 oak st, effingham, il"},
             "emails": {"jane@example.com"}, "names": {"jane smith"}}
    assert on_calendar({"phone": "(217) 555-9999 ext 4"}, guard)
    assert on_calendar({"email": " Jane@Example.COM "}, guard)
    assert on_calendar({"address": "12 Oak St, Effingham, IL "}, guard)
    assert on_calendar({"full_name": "Jane Smith"}, guard)
    assert not on_calendar({"full_name": "Someone Else",
                            "phone": "217-555-0000", "address": "9 Elm",
                            "email": "other@example.com"}, guard)


def test_calendar_guard_uses_real_producer_for_email_and_cancellation(
        monkeypatch):
    from atlas_brain.services import calendar_provider

    now = datetime.now(timezone.utc)
    active = calendar_provider.CalendarEvent(
        uid="active",
        summary="Active Customer",
        location="1 Active St, Effingham, IL",
        description=" ACTIVE@Example.COM ",
        start=now,
        end=now + timedelta(hours=1),
    )
    older = calendar_provider.CalendarEvent(
        uid="older",
        summary="Cancelled Customer",
        location="2 Cancelled St, Effingham, IL",
        description="stale@example.com",
        start=now - timedelta(days=2),
        end=now - timedelta(days=2) + timedelta(hours=1),
    )
    cancellation = calendar_provider.CalendarEvent(
        uid="cancelled",
        summary="Cancelled Customer - CANCELLED",
        location="2 Cancelled St, Effingham, IL",
        description="",
        start=now - timedelta(days=1),
        end=now - timedelta(days=1) + timedelta(hours=1),
    )

    class FakeGoogleCalendarProvider:
        calls = []
        closed = False

        async def list_events(self, *, start, end, calendar_id):
            self.calls.append(calendar_id)
            return {
                "commercial@test": [active],
                "residential@test": [older],
                "one-time@test": [cancellation],
            }[calendar_id]

        async def aclose(self):
            type(self).closed = True

    monkeypatch.setattr(
        calendar_provider, "GoogleCalendarProvider", FakeGoogleCalendarProvider
    )
    monkeypatch.setenv("EOM_CALENDAR_COMMERCIAL", "commercial@test")
    monkeypatch.setenv("EOM_CALENDAR_RESIDENTIAL", "residential@test")
    monkeypatch.setenv("EOM_CALENDAR_ONE_TIME", "one-time@test")

    keys = asyncio.run(fetch_calendar_guard_keys())

    assert keys["emails"] == {"active@example.com"}
    assert FakeGoogleCalendarProvider.calls == [
        "commercial@test", "residential@test", "one-time@test",
    ]
    assert FakeGoogleCalendarProvider.closed is True
    pool = SyncPool(demotion_rows=[
        {"id": "active", "full_name": "Different CRM Name", "tags": [],
         "phone": None, "email": " ACTIVE@Example.COM ", "address": "X"},
        {"id": "cancelled", "full_name": "Former CRM Name", "tags": [],
         "phone": None, "email": "stale@example.com", "address": "Y"},
        {"id": "gone", "full_name": "Moved Away", "tags": [],
         "phone": None, "email": "other@example.com", "address": "Z"},
    ])
    demoted, eligible = asyncio.run(
        demote_unmatched(pool, set(), apply=True, guard_keys=keys))

    assert (demoted, eligible) == (2, 3)
    assert [contact_id for contact_id, _ in pool.updates] == [
        "cancelled", "gone",
    ]
    assert "email" in pool.queries[0][0]


def test_demotion_refuses_without_the_calendar_guard():
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    guard_block = src.split("fetch_calendar_guard_keys()")[1].split("demote_unmatched(")[0]
    assert "DEMOTION SKIPPED" in guard_block
    assert 'counts["errors"] += 1' in guard_block
    # Codex r3: SystemExit (missing calendar config) takes the same skip.
    assert "(Exception, SystemExit)" in guard_block


def test_equal_time_name_cancellation_tie_resolves_to_cancelled():
    # Codex 2163 r4: same name, same timestamp, no shared channel -- the
    # cancelled state wins in either visit order (source-asserted).
    src = (REPO / "scripts" / "sync_eom_portal_customers.py").read_text()
    body = src.split("name_state.get(nm)")[1].split("for rec in merged")[0]
    assert "dt == cur[0]" in body
    assert "rec.cancelled and not cur[1]" in body


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


def test_demotion_failure_persists_partial_receipt_before_reraising(monkeypatch):
    import httpx
    import atlas_brain.services.crm_provider as crm_provider_mod
    import atlas_brain.storage.database as database_mod
    import sync_eom_portal_customers as sync_mod

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Pool(SyncPool):
        async def initialize(self):
            return None

        async def fetchrow(self, sql, *args):
            if "SET status = 'inactive'" in sql and self.updates:
                raise RuntimeError("second demotion failed")
            return await super().fetchrow(sql, *args)

    class Receipt:
        def __init__(self):
            self.changed = []
            self.counts = []
            self.demotions = []

        @contextlib.asynccontextmanager
        async def mutation_boundary(self):
            yield

        def record_changed_contact_id(self, contact_id):
            self.changed.append(str(contact_id))

        def record_outcome_counts(self, counts):
            self.counts.append(dict(counts))

        def record_demotions(self, **totals):
            self.demotions.append(dict(totals))

    async def guard_keys():
        return NO_GUARD

    pool = Pool(demotion_rows=[
        {"id": "gone", "full_name": "Moved Away", "tags": []},
        {"id": "also-gone", "full_name": "Also Gone", "tags": []},
    ])
    receipt = Receipt()
    monkeypatch.setattr(httpx, "Client", Client)
    monkeypatch.setattr(sync_mod, "portal_login", lambda *_args: "token")
    monkeypatch.setattr(sync_mod, "fetch_portal_customers", lambda *_args: [])
    monkeypatch.setattr(sync_mod, "fetch_calendar_guard_keys", guard_keys)
    monkeypatch.setattr(crm_provider_mod, "get_crm_provider", StubCRM)
    previous_pool = database_mod._db_pool
    database_mod._db_pool = pool
    try:
        with pytest.raises(RuntimeError, match="second demotion failed"):
            asyncio.run(sync_mod.run(
                SimpleNamespace(apply=True, base_url="https://example.invalid"),
                receipt=receipt,
            ))
    finally:
        database_mod._db_pool = previous_pool

    assert receipt.changed == ["gone"]
    assert receipt.demotions[-1] == {"demoted": 1, "eligible": 2, "kept": 0}
    assert receipt.counts[-1]["errors"] == 1


def test_demotable_sources_are_pinned_to_calendar_and_portal_only():
    """Widening this set silently archives live customers (website #127).

    ``demote_unmatched`` sets status='inactive' for EOM customers whose source
    is in DEMOTABLE_SOURCES and who no longer appear in the portal roster. Only
    This pins the set to make WIDENING it a visible, reviewed change (adding
    'manual'/'web' would mass-archive live customers); it does not certify each
    member unconditionally safe. Members are roster-authoritative provenance --
    the portal roster is the authority on current membership, so absence means
    churn: portal_sync (from the roster) and calendar_import (a booking is meant
    to veto demotion, though that veto has a known 4-vs-12-month horizon gap,
    website #138). NOT merely 'system-managed' -- email_backfill is
    system-created (D2, ATLAS #2314) yet excluded, because a backfilled email is
    not a roster-membership claim. Adding any value here -- 'manual', 'web',
    'email_backfill' -- turns a routine sync into a mass archive, so this is
    pinned exact. Changing it is a deliberate, separately-reviewed decision, not
    a casual edit.
    """
    assert DEMOTABLE_SOURCES == ("calendar_import", "portal_sync")
