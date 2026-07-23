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
    def __init__(self, was_created=True):
        self.was_created = was_created
        self.created = []
        self.updated = []
        self.interactions = []

    async def create_contact(self, data):
        self.created.append(data)
        return {"id": "new-id", "_was_created": self.was_created}

    async def update_contact(self, contact_id, data):
        self.updated.append((contact_id, data))
        return {"id": contact_id}

    async def log_interaction(self, **kwargs):
        self.interactions.append(kwargs)
        return {"id": "i"}


class StubPool:
    def __init__(self, row=None):
        self.row = row
        self.queries = []

    async def fetchrow(self, sql, *args):
        self.queries.append((sql, args))
        return self.row


def _record(phone=None, email=None):
    events = [_event("Jane Smith", "12 Oak St, Effingham, IL",
                     description="\n".join(filter(None, [phone, email])))]
    return parse_events(events, ["residential"], "customer", False, "Residential")[0]


def test_address_only_record_updates_existing_row():
    rec = _record()
    assert not rec.phone and not rec.email
    crm, pool = StubCRM(), StubPool(row={"id": "existing-id"})
    outcome = asyncio.run(import_one(rec, crm, pool))
    assert outcome == "updated"
    assert crm.created == []
    assert crm.updated and crm.updated[0][0] == "existing-id"


def test_address_only_record_creates_when_unmatched():
    rec = _record()
    crm, pool = StubCRM(), StubPool(row=None)
    outcome = asyncio.run(import_one(rec, crm, pool))
    assert outcome == "created"
    assert len(crm.created) == 1
    assert crm.created[0]["business_context_id"] == "effingham_maids"


def test_phone_record_uses_provider_dedupe_not_address_lookup():
    rec = _record(phone="(217) 555-9999")
    assert rec.phone
    crm, pool = StubCRM(was_created=False), StubPool(row={"id": "should-not-be-used"})
    outcome = asyncio.run(import_one(rec, crm, pool))
    assert outcome == "updated"          # _was_created False -> merged into existing
    assert pool.queries == []            # address net not consulted
    assert len(crm.created) == 1         # create_contact called (dedupes internally)


def test_interaction_carries_stable_dedupe_anchor():
    rec = _record()
    crm, pool = StubCRM(), StubPool(row=None)
    asyncio.run(import_one(rec, crm, pool))
    assert crm.interactions, "interaction should be logged for dated records"
    meta = crm.interactions[0]["metadata"]
    assert meta["source_ref"].startswith("eom_live_calendar:")
    assert crm.interactions[0]["interaction_type"] == "appointment"
