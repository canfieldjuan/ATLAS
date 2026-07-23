#!/usr/bin/env python3
"""Live Google Calendar -> Atlas CRM customer import (EOM booking calendars).

#2151 Phase 3 / #2156 slice A. Additive sibling of import_calendar_contacts.py
(the ICS-snapshot importer, which stays unchanged): this script reads the three
EOM *booking* calendars live through GoogleCalendarProvider and imports
tenant-stamped customer contacts. The Estimates calendar is deliberately
excluded -- estimate appointments are leads, and leads arrive in real time via
the intake endpoint (#2153).

Calendar IDs are provided via environment so they never enter this public
repo:

  EOM_CALENDAR_COMMERCIAL   Google calendar id of "Commercial Customers"
  EOM_CALENDAR_RESIDENTIAL  Google calendar id of "Effingham - Residential Customers"
  EOM_CALENDAR_ONE_TIME     Google calendar id of "Effingham - One Time Cleanings"

Run from a directory where data/google_tokens.json resolves (the token store
is cwd-relative; the deploy runbook covers the runtime worktree symlink).

Usage:
  python scripts/import_eom_customers_live.py --dry-run
  python scripts/import_eom_customers_live.py
  python scripts/import_eom_customers_live.py --calendar residential --months-back 12
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))          # sibling script import
sys.path.insert(0, str(Path(__file__).parent.parent))   # atlas_brain import

import import_calendar_contacts as ics  # noqa: E402  (reused extraction core)

EOM_CONTEXT_ID = "effingham_maids"

# The three booking calendars. Estimates is intentionally absent: it holds
# leads, not customers, and the real-time intake endpoint owns leads now.
BOOKING_CALENDARS = [
    {
        "key": "commercial",
        "env": "EOM_CALENDAR_COMMERCIAL",
        "tags": ["commercial"],
        "label": "Commercial Customers",
    },
    {
        "key": "residential",
        "env": "EOM_CALENDAR_RESIDENTIAL",
        "tags": ["residential"],
        "label": "Residential Customers",
    },
    {
        "key": "one_time",
        "env": "EOM_CALENDAR_ONE_TIME",
        "tags": ["one_time"],
        "label": "One-Time Cleanings",
    },
]


def resolve_calendar_ids(env: dict, selected: str = "all") -> dict:
    """Map calendar key -> calendar id from the environment, failing fast on
    any missing variable so a partial import can't silently pass as a full one.
    Only the selected calendar's variable is required in single-calendar mode
    (Codex R1: a targeted rerun must not demand unrelated secrets)."""
    wanted = [c for c in BOOKING_CALENDARS if selected in ("all", c["key"])]
    missing = [c["env"] for c in wanted if not env.get(c["env"])]
    if missing:
        raise SystemExit(
            "Missing calendar id environment variable(s): " + ", ".join(missing)
        )
    return {c["key"]: env[c["env"]] for c in wanted}


def parse_events(events, tags, contact_type, is_commercial, label):
    """Mirror of ics.parse_ics's per-event pipeline, fed from provider
    CalendarEvent objects instead of ICS VEVENTs. Dedupe within one calendar
    is by normalized address; richest contact info wins."""
    by_address = {}

    for ev in events:
        if (ev.status or "").lower() == "cancelled":
            continue

        raw_summary = (ev.summary or "").strip()
        raw_location = (ev.location or "").strip()
        raw_description = ics._strip_html(ev.description or "")
        if not raw_summary or not raw_location:
            continue

        name, cancelled = ics._clean_summary(raw_summary, is_commercial)
        if not name:
            continue

        address = ics._normalize_address(raw_location)
        if len(address) < 5:
            continue

        phone = ics._extract_phone(raw_description)
        email = ics._extract_email(raw_description)
        contact_name = ics._extract_contact_name(raw_description)
        event_date = ev.start.date() if ev.start else None

        notes_lines = []
        for line in raw_description.splitlines():
            line = line.strip()
            if not line:
                continue
            if ics._PHONE_RE.fullmatch(line) or ics._EMAIL_RE.fullmatch(line):
                continue
            if contact_name and line.lower().startswith(contact_name.lower()[:8]):
                continue
            notes_lines.append(line)
        notes = " | ".join(notes_lines[:3])

        addr_key = address.lower()
        if addr_key not in by_address:
            by_address[addr_key] = ics.CustomerRecord(
                name=name,
                address=address,
                phone=phone,
                email=email,
                contact_name=contact_name,
                notes=notes,
                tags=list(tags),
                contact_type=contact_type,
                source_calendar=label[:40],
                last_event_date=event_date,
                event_count=1,
                cancelled=cancelled,
            )
        else:
            rec = by_address[addr_key]
            rec.event_count += 1
            if len(name) < len(rec.name):
                rec.name = name
            if not rec.phone and phone:
                rec.phone = phone
            if not rec.email and email:
                rec.email = email
            if not rec.contact_name and contact_name:
                rec.contact_name = contact_name
            if not rec.notes and notes:
                rec.notes = notes
            if event_date and (rec.last_event_date is None or event_date > rec.last_event_date):
                rec.last_event_date = event_date
            if not cancelled:
                rec.cancelled = False

    return list(by_address.values())


def record_to_contact_data(rec) -> dict:
    """Contact payload for one record. Tenant stamp and source are
    non-negotiable here; tests assert both."""
    data = {
        "full_name": rec.name,
        "address": rec.address,
        "contact_type": rec.contact_type,
        "source": "calendar_import",
        "business_context_id": EOM_CONTEXT_ID,
        "tags": rec.tags,
        "status": "inactive" if rec.cancelled else "active",
    }
    if rec.phone:
        data["phone"] = rec.phone
    if rec.email:
        data["email"] = rec.email
    if rec.contact_name:
        data["notes"] = f"Contact: {rec.contact_name}"
        if rec.notes:
            data["notes"] += f" | {rec.notes}"
    elif rec.notes:
        data["notes"] = rec.notes
    return data


async def resolve_by_address(pool, address: str):
    """Pre-resolution for records with no phone and no email: create_contact
    only dedupes on those two channels, so without this net an address-only
    customer would insert a duplicate row on every re-run. Archived rows are
    excluded exactly like the provider's own search path (crm_provider
    search_contacts), so an import can never resurrect an archived contact
    (Codex R4/R8)."""
    return await pool.fetchrow(
        """
        SELECT id FROM contacts
        WHERE business_context_id = $1
          AND status != 'archived'
          AND address IS NOT NULL AND LOWER(address) = LOWER($2)
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        EOM_CONTEXT_ID,
        address,
    )


async def import_one(rec, crm, pool) -> str:
    """Import a single record; returns 'created' or 'updated'."""
    data = record_to_contact_data(rec)

    if rec.phone or rec.email:
        result = await crm.create_contact(data)
        contact_id = str(result.get("id", ""))
        if result.get("_was_created"):
            outcome = "created"
        else:
            outcome = "updated"
            # The provider's merge allowlist excludes `status`, so a matched
            # contact would silently keep its old status (Codex R1/R8):
            # persist the calendar-computed one explicitly.
            if contact_id and result.get("status") != data["status"]:
                await crm.update_contact(contact_id, {"status": data["status"]})
    else:
        row = await resolve_by_address(pool, rec.address)
        if row:
            contact_id = str(row["id"])
            await crm.update_contact(contact_id, data)
            outcome = "updated"
        else:
            result = await crm.create_contact(data)
            contact_id = str(result.get("id", ""))
            outcome = "created"

    if contact_id and rec.last_event_date:
        # source_ref inside metadata is a dedupe anchor (migration 256): the
        # same address never accumulates duplicate import interactions across
        # re-runs.
        await crm.log_interaction(
            contact_id=contact_id,
            interaction_type="appointment",
            summary=(
                f"Live calendar import: {rec.event_count} booking event(s). "
                f"Most recent: {rec.last_event_date.isoformat()}. "
                f"Source: {rec.source_calendar}"
            ),
            occurred_at=datetime.combine(
                rec.last_event_date, datetime.min.time()
            ).replace(tzinfo=timezone.utc).isoformat(),
            metadata={"source_ref": f"eom_live_calendar:{rec.address.lower()}"},
        )
    return outcome


def exit_code_for(counts: dict) -> int:
    """Non-zero when any record failed: a partially populated customer master
    must not read as success to a shell or operator runbook (Codex R6)."""
    return 1 if counts.get("errors") else 0


async def run_import(records, dry_run: bool) -> dict:
    counts = {"created": 0, "updated": 0, "errors": 0}
    crm = None
    pool = None
    if not dry_run:
        from atlas_brain.services.crm_provider import get_crm_provider
        from atlas_brain.storage.database import get_db_pool

        crm = get_crm_provider()
        pool = get_db_pool()

    for rec in sorted(records, key=lambda r: r.name.lower()):
        marker = "[CANCELLED]" if rec.cancelled else ""
        print(
            f"  {marker or '           '} {rec.name:<45} {(rec.phone or 'no phone'):<18} "
            f"{(rec.email or ''):<35} {rec.address[:50]:<52} "
            f"[{','.join(rec.tags)}] events={rec.event_count}"
        )
        if dry_run:
            continue
        try:
            counts[await import_one(rec, crm, pool)] += 1
        except Exception as e:  # noqa: BLE001 -- operator script: report and continue
            print(f"    ERROR: {rec.name} -- {e}")
            counts["errors"] += 1
    return counts


async def main():
    parser = argparse.ArgumentParser(
        description="Import EOM customers from the live booking calendars"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only -- no DB writes")
    parser.add_argument(
        "--calendar",
        choices=["commercial", "residential", "one_time", "all"],
        default="all",
    )
    parser.add_argument("--months-back", type=int, default=24)
    parser.add_argument("--months-forward", type=int, default=12)
    args = parser.parse_args()

    calendar_ids = resolve_calendar_ids(os.environ, args.calendar)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.months_back * 30)
    end = now + timedelta(days=args.months_forward * 30)

    mode = "DRY RUN" if args.dry_run else "LIVE IMPORT"
    print(f"\n{'=' * 70}")
    print(f"  Atlas CRM -- EOM live calendar customer import [{mode}]")
    print(f"  Window: {start.date()} .. {end.date()}")
    print(f"{'=' * 70}\n")

    from atlas_brain.services.calendar_provider import GoogleCalendarProvider

    provider = GoogleCalendarProvider()
    all_records = []
    try:
        for cal in BOOKING_CALENDARS:
            if args.calendar != "all" and cal["key"] != args.calendar:
                continue
            print(f"Fetching: {cal['label']} ...")
            events = await provider.list_events(
                start=start, end=end, calendar_id=calendar_ids[cal["key"]]
            )
            records = parse_events(
                events,
                tags=cal["tags"],
                contact_type="customer",
                is_commercial=(cal["key"] == "commercial"),
                label=cal["label"],
            )
            print(f"  {len(events)} events -> {len(records)} unique locations\n")
            all_records.extend(records)
    finally:
        await provider.aclose()

    print(f"Total before cross-calendar dedup: {len(all_records)}")
    records = ics.dedup_across_calendars(all_records)
    print(f"Total after dedup:                 {len(records)}")

    if not args.dry_run:
        from atlas_brain.storage.database import get_db_pool

        pool = get_db_pool()
        await pool.initialize()
        print("Database pool initialized.\n")

    counts = await run_import(records, args.dry_run)

    print(f"\n{'=' * 70}")
    if args.dry_run:
        print(f"  DRY RUN COMPLETE -- would import {len(records)} customers")
        print("  Run without --dry-run to write to the CRM.")
    else:
        print(
            f"  Created: {counts['created']}   Updated: {counts['updated']}   "
            f"Errors: {counts['errors']}"
        )
    print(f"{'=' * 70}\n")
    return exit_code_for(counts)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
