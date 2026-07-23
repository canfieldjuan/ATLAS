"""Tests for #2156 slice A: live-calendar EOM customer import.

parse_events and record_to_contact_data are pure and tested behaviorally with
duck-typed events (the provider's CalendarEvent surface: status/summary/
location/description/start). import_one is tested against stub crm/pool
objects at the DB edge -- the address-only pre-resolution is the idempotency
guarantee create_contact cannot provide, so both its branches are covered.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from import_eom_customers_live import (  # noqa: E402
    BOOKING_CALENDARS,
    EOM_CONTEXT_ID,
    _phone_digits,
    dedup_records,
    effective_calendar_env,
    exit_code_for,
    import_one,
    parse_events,
    record_to_contact_data,
    resolve_calendar_ids,
)


def _event(summary, location, description="", start=None, status="confirmed"):
    return SimpleNamespace(
        summary=summary,
        location=location,
        description=description,
        start=start or datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc),
        status=status,
    )


# ---------------------------------------------------------------------------
# Calendar configuration
# ---------------------------------------------------------------------------

def test_booking_calendars_exclude_estimates():
    keys = {c["key"] for c in BOOKING_CALENDARS}
    assert keys == {"commercial", "residential", "one_time"}
    source = (REPO / "scripts" / "import_eom_customers_live.py").read_text()
    assert "EOM_CALENDAR_ESTIMATE" not in source


def test_resolve_calendar_ids_fails_fast_on_missing():
    try:
        resolve_calendar_ids({"EOM_CALENDAR_COMMERCIAL": "c@x"})
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        msg = str(e)
        assert "EOM_CALENDAR_RESIDENTIAL" in msg and "EOM_CALENDAR_ONE_TIME" in msg


def test_resolve_calendar_ids_maps_all_three():
    env = {
        "EOM_CALENDAR_COMMERCIAL": "c@x",
        "EOM_CALENDAR_RESIDENTIAL": "r@x",
        "EOM_CALENDAR_ONE_TIME": "o@x",
    }
    assert resolve_calendar_ids(env) == {
        "commercial": "c@x", "residential": "r@x", "one_time": "o@x",
    }


def test_single_calendar_mode_requires_only_its_own_id():
    # Codex R1: a targeted rerun must not demand unrelated secrets.
    env = {"EOM_CALENDAR_RESIDENTIAL": "r@x"}
    assert resolve_calendar_ids(env, "residential") == {"residential": "r@x"}
    try:
        resolve_calendar_ids(env, "commercial")
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        assert "EOM_CALENDAR_COMMERCIAL" in str(e)
        assert "EOM_CALENDAR_ONE_TIME" not in str(e)


# ---------------------------------------------------------------------------
# parse_events (pure)
# ---------------------------------------------------------------------------

def test_parse_events_dedupes_by_address_and_keeps_latest_date():
    events = [
        _event("Jane Smith", "12 Oak St, Effingham, IL",
               start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc)),
        _event("Jane Smith - Deep Clean", "12 Oak St, Effingham, IL",
               description="Phone: (217) 555-1234",
               start=datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)),
    ]
    records = parse_events(events, ["residential"], "customer", False, "Residential")
    assert len(records) == 1
    rec = records[0]
    assert rec.event_count == 2
    assert rec.last_event_date == date(2026, 7, 1)
    assert rec.phone and "555" in rec.phone
    assert rec.tags == ["residential"]
    assert rec.contact_type == "customer"


def test_parse_events_skips_cancelled_status_and_blank_fields():
    events = [
        _event("Someone", "99 Elm St, Effingham, IL", status="cancelled"),
        _event("", "99 Elm St, Effingham, IL"),
        _event("No Location", ""),
    ]
    assert parse_events(events, ["one_time"], "customer", False, "One-Time") == []


def test_latest_cancelled_event_wins_over_older_active():
    # Codex round 3 (R1): a newer CANCELLED booking re-flags the record even
    # when an older active event exists -- in either input order.
    older_active = _event("Jane Smith", "12 Oak St, Effingham, IL",
                          start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc))
    newer_cancelled = _event("Jane Smith - CANCELLED", "12 Oak St, Effingham, IL",
                             start=datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc))
    for events in ([older_active, newer_cancelled], [newer_cancelled, older_active]):
        rec = parse_events(events, ["residential"], "customer", False, "Residential")[0]
        assert rec.cancelled is True
        assert record_to_contact_data(rec)["status"] == "inactive"


def test_latest_active_event_wins_over_older_cancelled():
    older_cancelled = _event("Jane Smith - CANCELLED", "12 Oak St, Effingham, IL",
                             start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc))
    newer_active = _event("Jane Smith", "12 Oak St, Effingham, IL",
                          start=datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc))
    for events in ([older_cancelled, newer_active], [newer_active, older_cancelled]):
        rec = parse_events(events, ["residential"], "customer", False, "Residential")[0]
        assert rec.cancelled is False
        assert record_to_contact_data(rec)["status"] == "active"


def test_parse_events_extracts_email_from_description():
    events = [
        _event("Bob Jones", "5 Pine Rd, Effingham, IL",
               description="bob.jones@example.com\nGate code 4321"),
    ]
    rec = parse_events(events, ["one_time"], "customer", False, "One-Time")[0]
    assert rec.email == "bob.jones@example.com"
    assert "Gate code 4321" in rec.notes


# ---------------------------------------------------------------------------
# record_to_contact_data (pure) -- the tenant stamp lives here
# ---------------------------------------------------------------------------

def test_contact_data_carries_tenant_stamp_and_source():
    events = [_event("Jane Smith", "12 Oak St, Effingham, IL")]
    rec = parse_events(events, ["residential"], "customer", False, "Residential")[0]
    data = record_to_contact_data(rec)
    assert data["business_context_id"] == EOM_CONTEXT_ID == "effingham_maids"
    assert data["source"] == "calendar_import"
    assert data["contact_type"] == "customer"
    assert data["status"] == "active"
    assert "residential" in data["tags"]


def test_contact_data_marks_cancelled_records_inactive():
    events = [_event("Jane Smith - CANCELLED", "12 Oak St, Effingham, IL")]
    rec = parse_events(events, ["residential"], "customer", False, "Residential")[0]
    assert record_to_contact_data(rec)["status"] == "inactive"


# ---------------------------------------------------------------------------
# import_one -- address-only idempotency at the DB edge
# ---------------------------------------------------------------------------

class StubCRM:
    def __init__(self, was_created=True, claim_result="row",
                 scoped_hit=None, legacy_hit=None):
        self.was_created = was_created
        self.claim_result = claim_result   # "row" -> claim succeeds; None -> foreign claim
        self.scoped_hit = scoped_hit       # returned for tenant-scoped channel search
        self.legacy_hit = legacy_hit       # returned for NULL-context channel search
        self.created = []
        self.updated = []
        self.claims = []
        self.searches = []
        self.interactions = []

    async def search_contacts(self, business_context_id=None,
                              business_context_id_is_null=False, **channel):
        self.searches.append((business_context_id, business_context_id_is_null, channel))
        if business_context_id and self.scoped_hit:
            return [self.scoped_hit]
        if business_context_id_is_null and self.legacy_hit:
            return [self.legacy_hit]
        return []

    async def claim_contact(self, contact_id, business_context_id):
        self.claims.append((contact_id, business_context_id))
        if self.claim_result == "row":
            return {"id": contact_id, "business_context_id": business_context_id}
        return None

    async def create_contact(self, data):
        self.created.append(data)
        return {
            "id": "new-id",
            "_was_created": self.was_created,
            "status": data["status"] if self.was_created else self.existing_status,
        }

    async def update_contact(self, contact_id, data):
        self.updated.append((contact_id, data))
        return {"id": contact_id}

    async def log_interaction(self, **kwargs):
        self.interactions.append(kwargs)
        return {"id": "i"}


class StubPool:
    """rows are returned in fetchrow call order (tenant page, then NULL page)."""

    def __init__(self, row=None, rows=None):
        self.rows = list(rows) if rows is not None else [row]
        self.queries = []

    async def fetchrow(self, sql, *args):
        self.queries.append((sql, args))
        return self.rows.pop(0) if self.rows else None


def _record(phone=None, email=None):
    events = [_event("Jane Smith", "12 Oak St, Effingham, IL",
                     description="\n".join(filter(None, [phone, email])))]
    return parse_events(events, ["residential"], "customer", False, "Residential")[0]


def test_address_only_record_updates_existing_row():
    rec = _record()
    assert not rec.phone and not rec.email
    crm, pool = StubCRM(), StubPool(rows=[{"id": "existing-id", "tags": None}])
    outcome = asyncio.run(import_one(rec, crm, pool))
    assert outcome == "updated"
    assert crm.created == []
    assert crm.claims == []               # same-tenant hit needs no claim
    assert crm.updated and crm.updated[0][0] == "existing-id"


def test_address_only_record_creates_when_unmatched():
    rec = _record()
    crm, pool = StubCRM(), StubPool(rows=[None, None])
    outcome = asyncio.run(import_one(rec, crm, pool))
    assert outcome == "created"
    assert len(crm.created) == 1
    assert crm.created[0]["business_context_id"] == "effingham_maids"


def test_phone_digits_strips_trailing_extension():
    # Codex round 5 (R1/R8): last-10 matching must use the base number.
    assert _phone_digits("217-555-9999 ext 123") == "2175559999"
    assert _phone_digits("217-555-9999 x42") == "2175559999"
    assert _phone_digits("(217) 555-9999") == "2175559999"
    assert _phone_digits("1-217-555-9999")[-10:] == "2175559999"


def test_same_day_later_cancellation_wins():
    # Codex round 5 (R1): recency compares full timestamps, so a later
    # same-day CANCELLED marker re-flags the record -- and vice versa.
    morning_active = _event("Jane Smith", "12 Oak St, Effingham, IL",
                            start=datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc))
    afternoon_cancel = _event("Jane Smith - CANCELLED", "12 Oak St, Effingham, IL",
                              start=datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc))
    for events in ([morning_active, afternoon_cancel], [afternoon_cancel, morning_active]):
        rec = parse_events(events, ["residential"], "customer", False, "Residential")[0]
        assert rec.cancelled is True
    morning_cancel = _event("Jane Smith - CANCELLED", "12 Oak St, Effingham, IL",
                            start=datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc))
    afternoon_active = _event("Jane Smith", "12 Oak St, Effingham, IL",
                              start=datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc))
    rec = parse_events([morning_cancel, afternoon_active],
                       ["residential"], "customer", False, "Residential")[0]
    assert rec.cancelled is False


def test_transitive_phone_address_dedupe():
    # Codex round 5 (R1/R8): phone links addr A to addr B; a later
    # address-only event at B must land on the same merged customer.
    rec_a = parse_events(
        [_event("Jane Smith", "12 Oak St, Effingham, IL", description="217-555-8888",
                start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc))],
        ["commercial"], "customer", True, "Commercial")
    rec_b = parse_events(
        [_event("Jane Smith", "77 Elm St, Effingham, IL", description="217-555-8888",
                start=datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc))],
        ["residential"], "customer", False, "Residential")
    rec_b_only = parse_events(
        [_event("Jane Smith", "77 Elm St, Effingham, IL",
                start=datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc))],
        ["one_time"], "customer", False, "One-Time")
    merged = dedup_records(rec_a + rec_b + rec_b_only)
    assert len(merged) == 1
    assert set(merged[0].tags) == {"commercial", "residential", "one_time"}
    assert merged[0].event_count == 3


def test_unchanged_contact_is_not_rewritten():
    # Codex round 5 (P1): a repeat import of an unchanged calendar performs
    # zero writes on matched contacts.
    rec = _record(phone="(217) 555-9999")
    stored = record_to_contact_data(rec)
    stored.pop("source")
    existing = {"id": "x", **stored}
    crm = StubCRM(scoped_hit=existing)
    outcome = asyncio.run(import_one(rec, crm, StubPool()))
    assert outcome == "unchanged"
    assert crm.updated == []
    assert crm.created == []


def test_matched_update_never_overwrites_source():
    # Codex round 5 (R1): a returning website lead keeps source='web'.
    rec = _record(phone="(217) 555-9999")
    crm = StubCRM(scoped_hit={"id": "k", "tags": [], "source": "web", "status": "inactive"})
    asyncio.run(import_one(rec, crm, StubPool()))
    assert "source" not in crm.updated[0][1]


def test_cross_calendar_dedupe_preserves_cancellation_recency():
    # Codex round 4 (R1): the latest-dated record decides `cancelled` across
    # calendars too -- the inherited merger's any-active-clears is corrected.
    older_active = parse_events(
        [_event("Jane Smith", "12 Oak St, Effingham, IL",
                description="217-555-8888",
                start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc))],
        ["one_time"], "customer", False, "One-Time")
    newer_cancelled = parse_events(
        [_event("Jane Smith - CANCELLED", "12 Oak St, Effingham, IL",
                description="217-555-8888",
                start=datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc))],
        ["residential"], "customer", False, "Residential")
    for records in (older_active + newer_cancelled, newer_cancelled + older_active):
        merged = dedup_records(list(records))
        assert len(merged) == 1
        assert merged[0].cancelled is True
    # And the reverse: a newer active booking reactivates across calendars.
    newer_active = parse_events(
        [_event("Jane Smith", "12 Oak St, Effingham, IL",
                description="217-555-8888",
                start=datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc))],
        ["residential"], "customer", False, "Residential")
    merged = dedup_records(list(newer_cancelled + newer_active))
    assert merged[0].cancelled is False


def test_legacy_null_context_row_is_claimed_not_duplicated():
    # Codex round 2 (R4/R8): a pre-#2155 NULL-context address-only contact is
    # claimed through the provider CAS and updated, never duplicated.
    rec = _record()
    crm = StubCRM(claim_result="row")
    pool = StubPool(rows=[None, {"id": "legacy-id", "tags": []}])
    outcome = asyncio.run(import_one(rec, crm, pool))
    assert outcome == "updated"
    assert crm.claims == [("legacy-id", "effingham_maids")]
    assert crm.created == []
    assert crm.updated and crm.updated[0][0] == "legacy-id"


def test_concurrently_claimed_legacy_row_falls_through_to_create():
    # CAS returns None when another tenant claimed the row first: fail closed,
    # create a fresh EOM row instead of overwriting the foreign claim.
    rec = _record()
    crm = StubCRM(claim_result=None)
    pool = StubPool(rows=[None, {"id": "legacy-id", "tags": []}])
    outcome = asyncio.run(import_one(rec, crm, pool))
    assert outcome == "created"
    assert crm.claims == [("legacy-id", "effingham_maids")]
    assert crm.updated == []
    assert len(crm.created) == 1


def test_settings_provide_ids_and_process_env_wins():
    # Codex round 2 (R11): ids recorded in .env reach the script through typed
    # settings; a raw env var still overrides for ad-hoc runs.
    tools = SimpleNamespace(
        eom_calendar_commercial="c@settings",
        eom_calendar_residential="r@settings",
        eom_calendar_one_time=None,
    )
    env = {"EOM_CALENDAR_RESIDENTIAL": "r@env", "EOM_CALENDAR_ONE_TIME": "o@env"}
    merged = effective_calendar_env(env, tools=tools)
    assert merged == {
        "EOM_CALENDAR_COMMERCIAL": "c@settings",
        "EOM_CALENDAR_RESIDENTIAL": "r@env",   # env beats settings
        "EOM_CALENDAR_ONE_TIME": "o@env",      # settings gap filled by env
    }


def test_phone_match_updates_without_address_lookup_and_unions_tags():
    # Codex round 4 (R1): the imported segment tag joins, never replaces,
    # tags the CRM already recorded (e.g. website intake provenance).
    rec = _record(phone="(217) 555-9999")
    assert rec.phone
    crm = StubCRM(scoped_hit={"id": "known-id", "tags": ["website", "estimate_request"]})
    pool = StubPool(row={"id": "should-not-be-used"})
    outcome = asyncio.run(import_one(rec, crm, pool))
    assert outcome == "updated"
    assert pool.queries == []            # address net not consulted on a channel hit
    assert crm.created == []
    cid, data = crm.updated[0]
    assert cid == "known-id"
    assert data["tags"] == ["estimate_request", "residential", "website"]
    assert data["status"] == "active"    # calendar-computed status persisted


def test_channel_miss_falls_back_to_address_before_creating():
    # Codex round 4 (R8): a calendar edit that adds a phone to a previously
    # address-only customer must enrich the existing row, not duplicate it.
    rec = _record(phone="(217) 555-9999")
    crm = StubCRM()                                    # both channel pages miss
    pool = StubPool(rows=[{"id": "addr-row", "tags": []}])
    outcome = asyncio.run(import_one(rec, crm, pool))
    assert outcome == "updated"
    assert crm.created == []
    assert crm.updated[0][0] == "addr-row"
    assert len(pool.queries) == 1        # tenant address page hit


def test_matched_contact_gets_calendar_computed_status():
    # Codex R1/R8: the provider merge allowlist excludes `status`; the update
    # path always carries the calendar-computed one.
    rec = _record(phone="(217) 555-9999")
    crm = StubCRM(scoped_hit={"id": "known-id", "tags": [], "status": "inactive"})
    outcome = asyncio.run(import_one(rec, crm, StubPool()))
    assert outcome == "updated"
    assert crm.updated[0][1]["status"] == "active"


def test_address_resolver_excludes_archived_rows():
    # Codex R4/R8: mirror the provider's own search guard so an import can
    # never resurrect an archived contact -- on BOTH pages (tenant + legacy).
    rec = _record()
    crm, pool = StubCRM(), StubPool(rows=[None, None])
    asyncio.run(import_one(rec, crm, pool))
    assert len(pool.queries) == 2
    for sql, _ in pool.queries:
        assert "status != 'archived'" in sql
    assert "business_context_id IS NULL" in pool.queries[1][0]
    # Codex round 3 (R4/R8): only rows with provable EOM provenance are
    # claimable; unknown-source legacy rows at a shared address stay put.
    assert "source = 'calendar_import'" in pool.queries[1][0]


def test_exit_code_reflects_errors():
    # Codex R6: a partial import must not exit 0.
    assert exit_code_for({"created": 5, "updated": 2, "errors": 0}) == 0
    assert exit_code_for({"created": 5, "updated": 2, "errors": 1}) == 1


def test_interaction_carries_stable_dedupe_anchor():
    rec = _record()
    crm, pool = StubCRM(), StubPool(row=None)
    asyncio.run(import_one(rec, crm, pool))
    assert crm.interactions, "interaction should be logged for dated records"
    meta = crm.interactions[0]["metadata"]
    assert meta["source_ref"].startswith("eom_live_calendar:")
    assert crm.interactions[0]["interaction_type"] == "appointment"
