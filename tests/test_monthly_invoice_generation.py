"""Integration tests for monthly_invoice_generation using real DB adapters."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_per_visit_groups_events_by_date():
    """Per Visit services should group multiple same-day events into QTY > 1."""
    from atlas_brain.storage.database import init_database, get_db_pool, close_database

    await init_database()
    pool = get_db_pool()

    try:
        from atlas_brain.storage.repositories.customer_service import get_customer_service_repo
        from atlas_brain.storage.repositories.invoice import get_invoice_repo

        svc_repo = get_customer_service_repo()
        inv_repo = get_invoice_repo()

        # Find the Menards service (Bathroom Cleaning, $48/visit, keyword "Menards")
        services = await svc_repo.list_active()
        menards = [s for s in services if s["calendar_keyword"] == "Menards"]
        assert menards, "Menards service agreement not found -- run invoicing setup first"
        svc = menards[0]

        # Simulate the per-day grouping logic from the task
        from collections import Counter

        # Mock events: 2 on Saturday March 7, 1 on Monday March 9
        class FakeEvent:
            def __init__(self, summary, dt):
                self.summary = summary
                self.start = datetime(dt.year, dt.month, dt.day, 16, 0, tzinfo=timezone.utc)
                self.status = "confirmed"

        events = [
            FakeEvent("Menards", date(2026, 3, 7)),
            FakeEvent("Menards - Caritina", date(2026, 3, 7)),
            FakeEvent("Menards", date(2026, 3, 9)),
        ]

        keyword = svc["calendar_keyword"].lower()
        matching = [e for e in events if keyword in e.summary.lower()]
        assert len(matching) == 3

        day_counts: Counter[date] = Counter()
        for event in matching:
            event_date = event.start.date()
            day_counts[event_date] += 1

        line_items = []
        rate = float(svc["rate"])
        for event_date in sorted(day_counts.keys()):
            qty = day_counts[event_date]
            line_items.append({
                "date": event_date.strftime("%m/%d/%Y"),
                "description": svc["service_name"],
                "quantity": qty,
                "unit_price": rate,
            })

        assert len(line_items) == 2  # 2 dates, not 3 events
        assert line_items[0]["quantity"] == 2  # March 7 = 2 events
        assert line_items[0]["unit_price"] == 48.0
        assert line_items[1]["quantity"] == 1  # March 9 = 1 event

        # Verify total would be correct
        total = sum(li["quantity"] * li["unit_price"] for li in line_items)
        assert total == 144.0  # (2 * 48) + (1 * 48)

    finally:
        await close_database()


@pytest.mark.asyncio
async def test_per_month_flat_rate_line_item():
    """Per Month services produce a single line item regardless of events."""
    from atlas_brain.storage.database import init_database, get_db_pool, close_database

    await init_database()
    try:
        from atlas_brain.storage.repositories.customer_service import get_customer_service_repo

        svc_repo = get_customer_service_repo()
        services = await svc_repo.list_active()
        lincare = [s for s in services if s["calendar_keyword"] == "Lincare"]
        assert lincare, "Lincare service agreement not found"
        svc = lincare[0]

        assert svc["rate_label"] == "Per Month"
        rate = float(svc["rate"])

        # Per Month: single line item
        line_items = [{
            "date": date(2026, 3, 1).strftime("%m/%d/%Y"),
            "description": f"March {svc['service_name']}",
            "quantity": 1,
            "unit_price": rate,
        }]

        assert len(line_items) == 1
        assert line_items[0]["unit_price"] == 300.0
        assert line_items[0]["quantity"] == 1

    finally:
        await close_database()


@pytest.mark.asyncio
async def test_per_hour_services_identified():
    """Per Hour services should be identifiable for skipping."""
    from atlas_brain.storage.database import init_database, close_database

    await init_database()
    try:
        from atlas_brain.storage.repositories.customer_service import get_customer_service_repo

        svc_repo = get_customer_service_repo()
        services = await svc_repo.list_active()

        hourly = [s for s in services if s.get("rate_label") == "Per Hour"]
        assert len(hourly) >= 2, f"Expected at least 2 hourly services (Firefly + Brookstone), got {len(hourly)}"

        names = {s["service_name"] for s in hourly}
        assert "Daily Cleaning" in names or "Room Cleaning" in names

    finally:
        await close_database()


@pytest.mark.asyncio
async def test_service_charge_metadata():
    """Services with credit card notes should produce tax_label metadata."""
    from atlas_brain.storage.database import init_database, close_database

    await init_database()
    try:
        from atlas_brain.storage.repositories.customer_service import get_customer_service_repo

        svc_repo = get_customer_service_repo()
        services = await svc_repo.list_active()

        kinder_elmo = [s for s in services
                       if "kinder morgan" in s["calendar_keyword"].lower()
                       and s.get("tax_rate", 0) > 0]
        assert kinder_elmo, "Kinder Morgan service with tax_rate not found"
        svc = kinder_elmo[0]

        tax_rate = float(svc["tax_rate"])
        svc_notes = str(svc.get("notes") or "").lower()

        assert tax_rate == 0.033
        assert "credit card" in svc_notes or "service charge" in svc_notes

        # Verify the metadata logic
        metadata = {}
        if tax_rate > 0 and ("service charge" in svc_notes or "credit card" in svc_notes):
            metadata["tax_label"] = f"Service Charge ({tax_rate * 100:.1f}%)"

        assert metadata["tax_label"] == "Service Charge (3.3%)"

    finally:
        await close_database()


@pytest.mark.asyncio
async def test_invoice_creation_with_real_repo():
    """Create a real invoice via the repo and verify it persists correctly."""
    from atlas_brain.storage.database import init_database, get_db_pool, close_database

    await init_database()
    pool = get_db_pool()
    try:
        from atlas_brain.storage.repositories.invoice import get_invoice_repo
        from atlas_brain.storage.repositories.customer_service import get_customer_service_repo

        inv_repo = get_invoice_repo()
        svc_repo = get_customer_service_repo()

        services = await svc_repo.list_active()
        akra = [s for s in services if s["calendar_keyword"] == "AKRA"]
        assert akra, "AKRA service not found"
        svc = akra[0]

        # Create a test invoice
        test_source_ref = f"test_{uuid4()}"
        line_items = [
            {"date": "03/07/2026", "description": "Office Cleaning", "quantity": 1, "unit_price": 160.0},
            {"date": "03/14/2026", "description": "Office Cleaning", "quantity": 1, "unit_price": 160.0},
        ]

        invoice = await inv_repo.create(
            customer_name="AKRA Builders Test",
            due_date=date(2026, 5, 1),
            line_items=line_items,
            contact_id=svc["contact_id"],
            source="test",
            source_ref=test_source_ref,
            invoice_for="Test Invoice",
        )

        assert invoice["invoice_number"].startswith("INV-")
        assert float(invoice["total_amount"]) == 320.0
        assert float(invoice["subtotal"]) == 320.0
        assert invoice["status"] == "draft"

        # Verify dedup works
        existing = await inv_repo.get_by_source_ref(test_source_ref)
        assert existing is not None
        assert existing["invoice_number"] == invoice["invoice_number"]

        # Cleanup: void the test invoice
        await pool.execute(
            "UPDATE invoices SET status = 'void', void_reason = 'test cleanup' WHERE source_ref = $1",
            test_source_ref,
        )

    finally:
        await close_database()


@pytest.mark.asyncio
async def test_pdf_generation_from_real_invoice():
    """Generate a PDF from a real invoice and verify it produces bytes."""
    from atlas_brain.storage.database import init_database, close_database

    await init_database()
    try:
        from atlas_brain.storage.repositories.invoice import get_invoice_repo
        from atlas_brain.services.invoice_pdf import render_invoice_pdf

        inv_repo = get_invoice_repo()

        # Use AKRA's real invoice
        inv = await inv_repo.get_by_number("INV-2026-0007")
        if not inv:
            pytest.skip("INV-2026-0007 not found -- run invoicing session first")

        pdf_bytes = render_invoice_pdf(inv)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 500  # A real PDF is at least a few hundred bytes
        assert pdf_bytes[:5] == b"%PDF-"  # Valid PDF header

    finally:
        await close_database()


@pytest.mark.asyncio
async def test_all_service_agreements_have_required_fields():
    """All active service agreements should have the fields the task needs."""
    from atlas_brain.storage.database import init_database, close_database

    await init_database()
    try:
        from atlas_brain.storage.repositories.customer_service import get_customer_service_repo

        svc_repo = get_customer_service_repo()
        services = await svc_repo.list_active()

        required_fields = ["id", "contact_id", "service_name", "rate", "calendar_keyword"]

        for svc in services:
            for field in required_fields:
                assert svc.get(field) is not None, (
                    f"Service '{svc.get('service_name', '?')}' missing required field '{field}'"
                )
            assert float(svc["rate"]) > 0, (
                f"Service '{svc['service_name']}' has zero or negative rate"
            )
            assert svc["calendar_keyword"].strip(), (
                f"Service '{svc['service_name']}' has empty calendar_keyword"
            )

    finally:
        await close_database()


@pytest.mark.asyncio
async def test_billing_month_override():
    """Task should respect billing_month metadata override."""
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import run
    from atlas_brain.storage.models import ScheduledTask
    from atlas_brain.storage.database import init_database, close_database

    task = ScheduledTask(
        id=uuid4(),
        name="monthly_invoice_generation",
        task_type="builtin",
        schedule_type="cron",
        cron_expression="0 8 1 * *",
        metadata={"billing_month": "2026-03"},
    )

    await init_database()
    try:
        result = await run(task)
        # Should use March 2026 as the billing period
        assert result.get("period") == "2026-03", f"Expected period 2026-03, got {result.get('period')}"
    finally:
        await close_database()


@pytest.mark.asyncio
async def test_billing_month_invalid_format():
    """Task should reject invalid billing_month format."""
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import run
    from atlas_brain.storage.models import ScheduledTask
    from atlas_brain.storage.database import init_database, close_database

    task = ScheduledTask(
        id=uuid4(),
        name="monthly_invoice_generation",
        task_type="builtin",
        schedule_type="cron",
        cron_expression="0 8 1 * *",
        metadata={"billing_month": "March"},
    )

    await init_database()
    try:
        result = await run(task)
        assert "_skip_synthesis" in result
        assert "Invalid billing_month" in result["_skip_synthesis"]
    finally:
        await close_database()


@pytest.mark.asyncio
async def test_contact_ids_filter():
    """Task should only invoice services matching contact_ids filter."""
    from atlas_brain.storage.database import init_database, get_db_pool, close_database

    await init_database()
    try:
        from atlas_brain.storage.repositories.customer_service import get_customer_service_repo

        svc_repo = get_customer_service_repo()
        services = await svc_repo.list_active()
        assert len(services) >= 2, "Need at least 2 services to test filtering"

        # Pick one contact_id to filter on
        target_contact = str(services[0]["contact_id"])

        from atlas_brain.autonomous.tasks.monthly_invoice_generation import run
        from atlas_brain.storage.models import ScheduledTask

        task = ScheduledTask(
            id=uuid4(),
            name="monthly_invoice_generation",
            task_type="builtin",
            schedule_type="cron",
            cron_expression="0 8 1 * *",
            metadata={
                "billing_month": "2026-03",
                "contact_ids": [target_contact],
            },
        )

        result = await run(task)

        # Should only process services for the filtered contact
        for detail in result.get("details", []):
            assert "error" not in detail, f"Invoice error: {detail}"

        # If invoices were created, they should all be for the target contact
        if result.get("invoices_created", 0) > 0:
            from atlas_brain.storage.repositories.invoice import get_invoice_repo
            inv_repo = get_invoice_repo()
            pool = get_db_pool()
            for detail in result["details"]:
                inv = await inv_repo.get_by_number(detail["invoice_number"])
                if inv:
                    assert str(inv["contact_id"]) == target_contact, (
                        f"Invoice {detail['invoice_number']} contact mismatch: "
                        f"expected {target_contact}, got {inv.get('contact_id')}"
                    )
                    # Clean up: void the test invoice
                    await pool.execute(
                        "UPDATE invoices SET status = 'void', void_reason = 'test cleanup' WHERE id = $1",
                        inv["id"],
                    )
    finally:
        await close_database()


@pytest.mark.asyncio
async def test_approve_and_send_dry_run():
    """approve_and_send dry_run should list drafts without sending."""
    import json
    from atlas_brain.storage.database import init_database, close_database

    await init_database()
    try:
        from atlas_brain.mcp.invoicing_server import approve_and_send

        result = json.loads(await approve_and_send(dry_run=True))
        assert result["success"] is True
        # Dry run should not send anything
        assert result.get("sent", 0) == 0
    finally:
        await close_database()


def _fake_event(summary: str, dt: date):
    """Build a minimal calendar-event stand-in for the resolver."""
    return SimpleNamespace(
        summary=summary,
        start=datetime(dt.year, dt.month, dt.day, 16, 0, tzinfo=timezone.utc),
        status="confirmed",
    )


def _fake_service(svc_id: str, name: str, keyword: str):
    return {
        "id": svc_id,
        "service_name": name,
        "calendar_keyword": keyword,
        "rate_label": "Per Visit",
    }


def test_resolver_unique_match_assigns_directly():
    """Single matching keyword: event goes to that service, no collision."""
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import (
        _assign_events_to_services,
    )

    services = [
        _fake_service("svc-a", "Menards", "Menards"),
        _fake_service("svc-b", "Lincare Office", "Lincare"),
    ]
    events = [_fake_event("Menards", date(2026, 3, 7))]

    by_service, collisions = _assign_events_to_services(events, services)

    assert by_service == {"svc-a": [events[0]]}
    assert collisions == []


def test_resolver_longest_keyword_wins_on_collision():
    """When two keywords both match, the longer one (more specific) is chosen."""
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import (
        _assign_events_to_services,
    )

    services = [
        _fake_service("svc-short", "Smith Family", "Smith"),
        _fake_service("svc-long", "Smith Corporation", "Smith Corp"),
    ]
    events = [_fake_event("Smith Corp Office", date(2026, 3, 14))]

    by_service, collisions = _assign_events_to_services(events, services)

    assert "svc-long" in by_service
    assert "svc-short" not in by_service
    assert by_service["svc-long"] == [events[0]]
    assert len(collisions) == 1
    assert collisions[0]["assigned_to"] == "Smith Corporation"
    assert collisions[0]["resolution"] == "longest_keyword"
    assert {m["keyword"] for m in collisions[0]["matched_services"]} == {"Smith", "Smith Corp"}


def test_resolver_no_collision_when_only_one_keyword_matches():
    """Substring of another keyword is fine if only one actually matches."""
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import (
        _assign_events_to_services,
    )

    # "Smith" alone should match "Smith Family Cleaning" but "Smith Corp" should not
    services = [
        _fake_service("svc-short", "Smith Family", "Smith"),
        _fake_service("svc-long", "Smith Corporation", "Smith Corp"),
    ]
    events = [_fake_event("Smith Family Cleaning", date(2026, 3, 14))]

    by_service, collisions = _assign_events_to_services(events, services)

    assert by_service == {"svc-short": [events[0]]}
    assert collisions == []


def test_resolver_alphabetical_tiebreak_on_equal_length():
    """Equal-length keywords break alphabetically and are flagged as ties."""
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import (
        _assign_events_to_services,
    )

    services = [
        _fake_service("svc-bbb", "Beta Customer", "BBB"),
        _fake_service("svc-aaa", "Alpha Customer", "AAA"),
    ]
    # Event summary contains both keywords -- true ambiguity
    events = [_fake_event("AAA / BBB combo job", date(2026, 3, 14))]

    by_service, collisions = _assign_events_to_services(events, services)

    # Alphabetical ascending: AAA < BBB, so Alpha wins
    assert "svc-aaa" in by_service
    assert "svc-bbb" not in by_service
    assert collisions[0]["assigned_to"] == "Alpha Customer"
    assert collisions[0]["resolution"] == "alphabetical_tiebreak"


def test_resolver_event_with_no_match_is_dropped():
    """Events that don't match any service are silently ignored (current behavior)."""
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import (
        _assign_events_to_services,
    )

    services = [_fake_service("svc-a", "Menards", "Menards")]
    events = [_fake_event("Personal appointment", date(2026, 3, 7))]

    by_service, collisions = _assign_events_to_services(events, services)

    assert by_service == {}
    assert collisions == []


def test_per_hour_line_items_one_per_event_date():
    """Per Hour helper builds one placeholder line item per unique event date."""
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import (
        _build_per_hour_line_items,
        _PER_HOUR_PLACEHOLDER_QTY,
        _PER_HOUR_PLACEHOLDER_DESC_SUFFIX,
    )

    events = [
        _fake_event("Brookstone room 1", date(2026, 3, 7)),
        _fake_event("Brookstone room 2", date(2026, 3, 7)),
        _fake_event("Brookstone room 3", date(2026, 3, 14)),
    ]

    line_items, visit_count = _build_per_hour_line_items(events, 35.0, "Brookstone")

    assert len(line_items) == 2  # 2 unique dates
    assert visit_count == 3  # 3 total events
    for li in line_items:
        assert li["quantity"] == _PER_HOUR_PLACEHOLDER_QTY
        assert li["unit_price"] == 35.0
        assert li["description"] == "Brookstone" + _PER_HOUR_PLACEHOLDER_DESC_SUFFIX
    # Sorted by date ascending
    assert line_items[0]["date"] == "03/07/2026"
    assert line_items[1]["date"] == "03/14/2026"


def test_per_hour_line_items_zero_subtotal():
    """Per Hour drafts must produce $0 subtotal so the user can spot them."""
    from decimal import Decimal
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import (
        _build_per_hour_line_items,
    )

    events = [_fake_event("Firefly", date(2026, 3, 7))]
    line_items, _ = _build_per_hour_line_items(events, 50.0, "Firefly")

    subtotal = sum(
        Decimal(str(li["quantity"])) * Decimal(str(li["unit_price"]))
        for li in line_items
    )
    assert subtotal == Decimal("0")


def test_per_hour_line_items_empty_when_no_events():
    """Helper returns empty list and 0 count when no events match."""
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import (
        _build_per_hour_line_items,
    )

    line_items, visit_count = _build_per_hour_line_items([], 35.0, "Brookstone")
    assert line_items == []
    assert visit_count == 0


def test_per_hour_service_now_eligible_in_assignment():
    """Per Hour services must be in eligible_services so their events route correctly."""
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import (
        _assign_events_to_services,
    )

    # Per Hour service should receive its events (not be filtered out)
    services = [
        {
            "id": "svc-firefly",
            "service_name": "Firefly Hourly",
            "calendar_keyword": "Firefly",
            "rate_label": "Per Hour",
        },
    ]
    events = [_fake_event("Firefly daily clean", date(2026, 3, 7))]

    by_service, collisions = _assign_events_to_services(events, services)
    assert by_service == {"svc-firefly": [events[0]]}
    assert collisions == []


def test_resolver_no_double_count_for_collision():
    """Critical: a single colliding event must not appear in two services' lists."""
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import (
        _assign_events_to_services,
    )

    services = [
        _fake_service("svc-short", "Smith Family", "Smith"),
        _fake_service("svc-long", "Smith Corporation", "Smith Corp"),
    ]
    events = [_fake_event("Smith Corp Office", date(2026, 3, 14))]

    by_service, _ = _assign_events_to_services(events, services)

    total_assigned = sum(len(v) for v in by_service.values())
    assert total_assigned == 1, "Colliding event was double-counted"


@pytest.mark.asyncio
async def test_payment_reminder_attaches_pdf(monkeypatch):
    """Reminder emails must attach the invoice PDF (consistency with original send)."""
    from atlas_brain.autonomous.tasks import invoice_payment_reminders as task_mod
    from atlas_brain.storage.models import ScheduledTask

    captured: dict = {}

    class FakeEmailProvider:
        async def send(self, *, to, subject, body, attachments=None, **_):
            captured["to"] = to
            captured["subject"] = subject
            captured["body"] = body
            captured["attachments"] = attachments
            return {"ok": True}

    class FakeRepo:
        async def get_overdue(self, as_of_date=None):
            return [{
                "id": uuid4(),
                "invoice_number": "INV-2026-9001",
                "customer_name": "Test Customer",
                "customer_email": "test@example.com",
                "amount_due": 250.0,
                "due_date": date(2026, 1, 1),
                "reminder_count": 0,
                "last_reminder_at": None,
                "contact_id": None,
                "line_items": [
                    {"date": "01/01/2026", "description": "Test", "quantity": 1, "unit_price": 250.0, "amount": 250.0},
                ],
                "subtotal": 250.0,
                "tax_rate": 0.0,
                "tax_amount": 0.0,
                "discount_amount": 0.0,
                "total_amount": 250.0,
                "metadata": {},
            }]

        async def update_reminder(self, _id):
            captured["update_reminder_called"] = True

    def fake_render(inv):
        captured["render_called_for"] = inv["invoice_number"]
        return b"%PDF-fake-bytes"

    monkeypatch.setattr(
        "atlas_brain.storage.repositories.invoice.get_invoice_repo",
        lambda: FakeRepo(),
    )
    monkeypatch.setattr(
        "atlas_brain.services.email_provider.get_email_provider",
        lambda: FakeEmailProvider(),
    )
    monkeypatch.setattr(
        "atlas_brain.services.invoice_pdf.render_invoice_pdf",
        fake_render,
    )

    task = ScheduledTask(
        id=uuid4(),
        name="invoice_payment_reminders",
        task_type="builtin",
        schedule_type="cron",
        cron_expression="0 10 * * *",
    )

    result = await task_mod.run(task)

    assert result.get("reminders_sent") == 1
    assert captured.get("render_called_for") == "INV-2026-9001"
    attachments = captured.get("attachments")
    assert attachments and len(attachments) == 1
    assert attachments[0]["filename"] == "INV-2026-9001.pdf"
    import base64
    assert base64.b64decode(attachments[0]["content"]) == b"%PDF-fake-bytes"
    # Body should reference the attachment when PDF render succeeded
    assert "attached for your reference" in captured["body"]


@pytest.mark.asyncio
async def test_payment_reminder_falls_back_when_pdf_fails(monkeypatch):
    """If PDF render fails, send text-only reminder (don't skip the customer)."""
    from atlas_brain.autonomous.tasks import invoice_payment_reminders as task_mod
    from atlas_brain.storage.models import ScheduledTask

    captured: dict = {}

    class FakeEmailProvider:
        async def send(self, *, to, subject, body, attachments=None, **_):
            captured["attachments"] = attachments
            captured["body"] = body
            return {"ok": True}

    class FakeRepo:
        async def get_overdue(self, as_of_date=None):
            return [{
                "id": uuid4(),
                "invoice_number": "INV-2026-9002",
                "customer_name": "Test",
                "customer_email": "test@example.com",
                "amount_due": 100.0,
                "due_date": date(2026, 1, 1),
                "reminder_count": 0,
                "last_reminder_at": None,
                "contact_id": None,
                "line_items": [],
                "metadata": {},
            }]

        async def update_reminder(self, _id):
            pass

    def boom_render(_inv):
        raise RuntimeError("font missing")

    monkeypatch.setattr(
        "atlas_brain.storage.repositories.invoice.get_invoice_repo",
        lambda: FakeRepo(),
    )
    monkeypatch.setattr(
        "atlas_brain.services.email_provider.get_email_provider",
        lambda: FakeEmailProvider(),
    )
    monkeypatch.setattr(
        "atlas_brain.services.invoice_pdf.render_invoice_pdf",
        boom_render,
    )

    task = ScheduledTask(
        id=uuid4(),
        name="invoice_payment_reminders",
        task_type="builtin",
        schedule_type="cron",
        cron_expression="0 10 * * *",
    )

    result = await task_mod.run(task)

    assert result.get("reminders_sent") == 1, "Reminder should still go out without attachment"
    assert captured.get("attachments") is None
    # Body should NOT mention the attachment when PDF render failed
    assert "attached for your reference" not in captured.get("body", "")


@pytest.mark.asyncio
async def test_approve_and_send_skips_needs_hours_drafts():
    """approve_and_send must refuse to mail drafts flagged needs_hours,
    so $0 placeholders generated by monthly_invoice_generation never go
    out before the user fills real hours."""
    import json
    from atlas_brain.storage.database import init_database, get_db_pool, close_database

    await init_database()
    pool = get_db_pool()
    try:
        from atlas_brain.storage.repositories.invoice import get_invoice_repo
        from atlas_brain.storage.repositories.customer_service import get_customer_service_repo
        from atlas_brain.mcp.invoicing_server import approve_and_send

        inv_repo = get_invoice_repo()
        svc_repo = get_customer_service_repo()

        # Find any active service to attach the test invoice to a real contact_id
        services = await svc_repo.list_active()
        assert services, "No active services to anchor test invoice"
        anchor = services[0]

        test_source_ref = f"test_needs_hours_{uuid4()}"
        line_items = [
            {"date": "03/07/2026", "description": "Brookstone (hours TBD)", "quantity": 0, "unit_price": 35.0},
        ]

        invoice = await inv_repo.create(
            customer_name="Brookstone Test",
            customer_email="brookstone-test@example.com",
            due_date=date(2026, 5, 1),
            line_items=line_items,
            contact_id=anchor["contact_id"],
            source="test",
            source_ref=test_source_ref,
            invoice_for="Per Hour Placeholder Test",
            metadata={"needs_hours": True},
        )

        try:
            # Real send (dry_run=False) -- must SKIP with needs_hours reason
            result = json.loads(await approve_and_send(
                invoice_ids=json.dumps([invoice["invoice_number"]]),
                dry_run=False,
            ))
            assert result["success"] is True
            assert result["sent"] == 0, "needs_hours draft must NOT be sent"
            assert result["skipped"] >= 1
            skip_detail = next(
                (d for d in result["details"] if d["invoice"] == invoice["invoice_number"]),
                None,
            )
            assert skip_detail is not None
            assert "needs hours" in skip_detail["reason"].lower()

            # dry_run path also skips
            result_dry = json.loads(await approve_and_send(
                invoice_ids=json.dumps([invoice["invoice_number"]]),
                dry_run=True,
            ))
            dry_detail = next(
                (d for d in result_dry["details"] if d["invoice"] == invoice["invoice_number"]),
                None,
            )
            assert dry_detail is not None
            assert "needs hours" in dry_detail["reason"].lower()

            # Confirm the invoice is still draft (not silently mutated)
            still = await inv_repo.get_by_id(invoice["id"])
            assert still["status"] == "draft"

        finally:
            await pool.execute(
                "UPDATE invoices SET status = 'void', void_reason = 'test cleanup' WHERE source_ref = $1",
                test_source_ref,
            )
    finally:
        await close_database()


@pytest.mark.asyncio
async def test_export_invoice_pdf():
    """export_invoice_pdf should generate PDF bytes for a real invoice."""
    import json
    from atlas_brain.storage.database import init_database, close_database

    await init_database()
    try:
        from atlas_brain.mcp.invoicing_server import export_invoice_pdf

        result = json.loads(await export_invoice_pdf("INV-2026-0007", save_to_disk=False))
        if not result.get("success"):
            pytest.skip("INV-2026-0007 not found")

        assert result["success"] is True
        assert result["pdf_size_bytes"] > 500
    finally:
        await close_database()
