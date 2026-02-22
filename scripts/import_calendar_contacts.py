"""
Import customer contacts from Google Calendar ICS exports.

Parses all 4 Effingham Office Maids calendars, deduplicates by address
(within calendar) and phone/email (across calendars), then creates contacts
in the Atlas CRM via find_or_create_contact().

Usage:
    # Preview what would be imported (no DB writes)
    python scripts/import_calendar_contacts.py --dry-run

    # Live import
    python scripts/import_calendar_contacts.py

    # Single calendar only
    python scripts/import_calendar_contacts.py --dry-run --calendar commercial

Output:
    Console summary + data/calendar_import.log (one line per contact)
"""

import argparse
import asyncio
import html
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ICS file definitions
# ---------------------------------------------------------------------------

_PDF_DIR = Path("/home/juan-canfield/PDF")

ICS_FILES = [
    {
        "key": "commercial",
        "path": _PDF_DIR / "Commercial Customers - Effingham Office Maids_689cr4i2b0cii1k4c645mc57d8@group.calendar.google.com.ics",
        "tags": ["commercial"],
        "contact_type": "customer",
        "label": "Commercial Customers",
    },
    {
        "key": "residential",
        "path": _PDF_DIR / "Effingham - Residential Customers_pj1f79l6onh69nbcjud6gcl2f0@group.calendar.google.com.ics",
        "tags": ["residential"],
        "contact_type": "customer",
        "label": "Residential Customers",
    },
    {
        "key": "one_time",
        "path": _PDF_DIR / "Effingham - One Time Cleanings_jtej0t5tmnjs84gqev9o7jvmhk@group.calendar.google.com.ics",
        "tags": ["one_time"],
        "contact_type": "customer",
        "label": "One-Time Cleanings",
    },
    {
        "key": "estimates",
        "path": _PDF_DIR / "Effingham - Estimate  Appointments_t1e8rsnnpm9u43hdsq3993sma8@group.calendar.google.com.ics",
        "tags": ["estimate"],
        "contact_type": "lead",
        "label": "Estimate Appointments",
    },
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CustomerRecord:
    name: str
    address: str
    phone: Optional[str] = None
    email: Optional[str] = None
    contact_name: Optional[str] = None   # person's name at the business
    notes: str = ""
    tags: list = field(default_factory=list)
    contact_type: str = "customer"
    source_calendar: str = ""
    last_event_date: Optional[date] = None
    event_count: int = 0
    cancelled: bool = False

# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

# Operational suffixes to strip from SUMMARY (case-insensitive)
_STRIP_RE = re.compile(
    r"\s*[-]\s*("
    r"skip(\s+on\s+vacation)?|"
    r"deep\s*clean(ing)?|"
    r"carpet\s+cleaning.*|"
    r"cancelled|"
    r"free\s+estimate|"
    r"skip\s*$"
    r")\s*$",
    re.IGNORECASE,
)

# Crew-name suffix for commercial events: "- FirstName" or "- First y Last"
# Only strip if the suffix is 1-3 short words (crew member name, not a department)
_CREW_RE = re.compile(r"\s+-\s+([A-Za-z]+(?: y [A-Za-z]+)?)\s*$")

# Phone: 10 digits with common separators; optional ext
_PHONE_RE = re.compile(
    r"(?:phone(?:\s+number)?[\s:.-]*)?"
    r"\(?\d{3}\)?[\s.-]{0,3}\d{3}[\s.-]{0,3}\d{4}"
    r"(?:\s*(?:ext\.?|x)\s*\d+)?",
    re.IGNORECASE,
)

# Email in plain text or inside href="mailto:..."
_EMAIL_RE = re.compile(
    r'(?:mailto:)?([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})',
    re.IGNORECASE,
)

# Detect "CANCELLED" anywhere in summary
_CANCELLED_RE = re.compile(r"\bcancel", re.IGNORECASE)


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    # ICS escape sequences
    text = text.replace("\\n", "\n").replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\")
    # Collapse whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _clean_summary(raw: str, is_commercial: bool) -> tuple[str, bool]:
    """
    Return (cleaned_name, is_cancelled).

    For commercial: strip crew name suffix (short word after last dash).
    For both: strip operational keywords (skip, cancelled, deep clean, etc.).
    """
    cancelled = bool(_CANCELLED_RE.search(raw))
    name = raw.strip()

    # Strip operational suffixes first
    name = _STRIP_RE.sub("", name).strip()

    # For commercial events, also strip crew name suffix
    if is_commercial:
        m = _CREW_RE.search(name)
        if m:
            suffix = m.group(1)
            # Only strip if it looks like a person name (no digits, <= 4 words)
            words = suffix.split()
            if len(words) <= 4 and not any(c.isdigit() for c in suffix):
                name = name[: m.start()].strip()

    return name.strip(" -"), cancelled


def _extract_phone(text: str) -> Optional[str]:
    """Extract and normalize the first phone number found."""
    m = _PHONE_RE.search(text)
    if not m:
        return None
    raw = m.group(0)
    # Normalize: keep digits + ext
    digits = re.sub(r"[^\d]", "", raw.split("ext")[0].split("x")[0])
    if len(digits) < 10:
        return None
    ext_m = re.search(r"(?:ext\.?|x)\s*(\d+)", raw, re.IGNORECASE)
    normalized = f"{digits[-10:]}"  # take last 10 in case of leading 1
    # Format as XXX-XXX-XXXX
    normalized = f"{normalized[:3]}-{normalized[3:6]}-{normalized[6:]}"
    if ext_m:
        normalized += f" ext {ext_m.group(1)}"
    return normalized


def _extract_email(text: str) -> Optional[str]:
    """Extract first email address (handles plain text and mailto: hrefs)."""
    m = _EMAIL_RE.search(text)
    return m.group(1).lower() if m else None


def _extract_contact_name(text: str) -> Optional[str]:
    """
    Heuristic: first line of description often has 'Name - Phone' or
    'Name Phone' or 'contact Name' or 'Att: Name'.
    Returns a plausible contact name or None.
    """
    if not text:
        return None
    first_line = text.strip().splitlines()[0].strip()

    # "Att: Name" or "Att Name"
    m = re.match(r"Att:?\s+(.+)", first_line, re.IGNORECASE)
    if m:
        return _strip_phone_email(m.group(1))

    # "contact Name"
    m = re.match(r"contacts?\s+[-–]?\s*(.+)", first_line, re.IGNORECASE)
    if m:
        return _strip_phone_email(m.group(1))

    # "Name - Phone" pattern: name is the part before the dash/phone
    m = re.match(r"^([A-Za-z][A-Za-z\s]{2,40}?)\s*[-–]\s*[\d(]", first_line)
    if m:
        candidate = m.group(1).strip()
        if 2 <= len(candidate.split()) <= 5:
            return candidate

    # "Name Phone" where phone starts right after the name
    m = re.match(r"^([A-Za-z][A-Za-z\s]{2,30}?)\s+\d", first_line)
    if m:
        candidate = m.group(1).strip()
        if 1 <= len(candidate.split()) <= 4:
            return candidate

    return None


def _strip_phone_email(text: str) -> Optional[str]:
    """Remove phone/email from a string; return None if nothing useful remains."""
    text = re.sub(r"[\d()\-.\s/ext]+$", "", text, flags=re.IGNORECASE).strip(" -,")
    text = _EMAIL_RE.sub("", text).strip(" -,")
    return text if len(text) >= 2 else None


def _normalize_address(raw: str) -> str:
    """
    Strip leading 'Venue Name, ' prefix and normalize whitespace.

    e.g. 'The New York Blower Company, 1304 W Jaycee Ave, Effingham, IL'
    -> '1304 W Jaycee Ave, Effingham, IL'
    """
    # ICS uses \, for literal comma
    addr = raw.replace("\\,", ",").strip()
    # If first token looks like a venue name (no digits), strip it
    parts = [p.strip() for p in addr.split(",")]
    if parts and not re.search(r"\d", parts[0]):
        parts = parts[1:]
    return ", ".join(parts).strip()


def _event_date(dtstart) -> Optional[date]:
    """Extract a plain date from a DTSTART value (datetime or date)."""
    if dtstart is None:
        return None
    val = dtstart.dt if hasattr(dtstart, "dt") else dtstart
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    return None

# ---------------------------------------------------------------------------
# ICS parser
# ---------------------------------------------------------------------------

def parse_ics(path: Path, tags: list, contact_type: str, is_commercial: bool) -> list[CustomerRecord]:
    """
    Parse one ICS file and return deduplicated CustomerRecords.

    Deduplication key within this file: normalized address.
    For each address we keep the event with the richest description
    (most contact info) and track the latest event date.
    """
    try:
        from icalendar import Calendar
    except ImportError:
        print("ERROR: icalendar not installed. Run: pip install icalendar")
        sys.exit(1)

    with open(path, "rb") as f:
        cal = Calendar.from_ical(f.read())

    # address -> best CustomerRecord so far
    by_address: dict[str, CustomerRecord] = {}

    for component in cal.walk():
        if component.name != "VEVENT":
            continue

        raw_summary = str(component.get("SUMMARY", "")).strip()
        raw_location = str(component.get("LOCATION", "")).strip()
        raw_description = _strip_html(str(component.get("DESCRIPTION", "")))
        dtstart = component.get("DTSTART")
        status = str(component.get("STATUS", "")).upper()

        if not raw_summary or not raw_location:
            continue

        # Skip explicitly cancelled events from the calendar (STATUS=CANCELLED)
        # — SUMMARY "CANCELLED" keeps them as inactive leads, handled below
        if status == "CANCELLED":
            continue

        name, cancelled = _clean_summary(raw_summary, is_commercial)
        if not name:
            continue

        address = _normalize_address(raw_location)
        if len(address) < 5:
            continue

        phone = _extract_phone(raw_description)
        email = _extract_email(raw_description)
        contact_name = _extract_contact_name(raw_description)
        event_date = _event_date(dtstart)

        # Build notes from description, stripping phone/email lines
        notes_lines = []
        for line in raw_description.splitlines():
            line = line.strip()
            if not line:
                continue
            # Skip lines that are just phone numbers or emails
            if _PHONE_RE.fullmatch(line) or _EMAIL_RE.fullmatch(line):
                continue
            # Skip lines that only contain the contact name we already extracted
            if contact_name and line.lower().startswith(contact_name.lower()[:8]):
                continue
            notes_lines.append(line)
        notes = " | ".join(notes_lines[:3])  # cap at 3 lines

        addr_key = address.lower()

        if addr_key not in by_address:
            by_address[addr_key] = CustomerRecord(
                name=name,
                address=address,
                phone=phone,
                email=email,
                contact_name=contact_name,
                notes=notes,
                tags=list(tags),
                contact_type=contact_type,
                source_calendar=path.stem.split("_")[0][:40],
                last_event_date=event_date,
                event_count=1,
                cancelled=cancelled,
            )
        else:
            rec = by_address[addr_key]
            rec.event_count += 1
            # Keep the shortest/cleanest name (strips crew suffixes better)
            if len(name) < len(rec.name):
                rec.name = name
            # Prefer records with more contact info
            if not rec.phone and phone:
                rec.phone = phone
            if not rec.email and email:
                rec.email = email
            if not rec.contact_name and contact_name:
                rec.contact_name = contact_name
            if not rec.notes and notes:
                rec.notes = notes
            # Track latest event date
            if event_date and (rec.last_event_date is None or event_date > rec.last_event_date):
                rec.last_event_date = event_date
            # If ANY event is not cancelled, mark as not cancelled
            if not cancelled:
                rec.cancelled = False

    return list(by_address.values())


# ---------------------------------------------------------------------------
# Cross-calendar deduplication
# ---------------------------------------------------------------------------

def _phone_key(phone: Optional[str]) -> Optional[str]:
    """Normalize to last 10 digits for comparison."""
    if not phone:
        return None
    digits = re.sub(r"\D", "", phone)
    return digits[-10:] if len(digits) >= 10 else None


def dedup_across_calendars(records: list[CustomerRecord]) -> list[CustomerRecord]:
    """
    Merge records that refer to the same customer across calendars.

    Priority: customer > lead, commercial > one_time > estimates > residential
    (so a commercial customer that also has an estimate gets contact_type=customer)
    """
    _TYPE_PRIORITY = {"customer": 0, "lead": 1}
    _CAL_PRIORITY = {"commercial": 0, "residential": 1, "one_time": 2, "estimates": 3}

    # Index by phone and by address
    by_phone: dict[str, CustomerRecord] = {}
    by_address: dict[str, CustomerRecord] = {}
    merged: list[CustomerRecord] = []
    seen_ids: set[int] = set()

    for rec in records:
        pk = _phone_key(rec.phone)
        ak = rec.address.lower()

        existing = by_phone.get(pk) if pk else None
        if existing is None:
            existing = by_address.get(ak)

        if existing is None:
            # New record
            merged.append(rec)
            if pk:
                by_phone[pk] = rec
            by_address[ak] = rec
        else:
            # Merge into existing — keep best fields
            if not existing.email and rec.email:
                existing.email = rec.email
            if not existing.phone and rec.phone:
                existing.phone = rec.phone
                if _phone_key(rec.phone):
                    by_phone[_phone_key(rec.phone)] = existing
            if not existing.contact_name and rec.contact_name:
                existing.contact_name = rec.contact_name
            if not existing.notes and rec.notes:
                existing.notes = rec.notes

            # Merge tags (deduplicated)
            for tag in rec.tags:
                if tag not in existing.tags:
                    existing.tags.append(tag)

            # Prefer higher-priority contact_type
            if _TYPE_PRIORITY.get(rec.contact_type, 99) < _TYPE_PRIORITY.get(existing.contact_type, 99):
                existing.contact_type = rec.contact_type

            # Latest event date
            if rec.last_event_date and (
                existing.last_event_date is None or rec.last_event_date > existing.last_event_date
            ):
                existing.last_event_date = rec.last_event_date

            existing.event_count += rec.event_count

            # If merged record is a confirmed customer, uncancel
            if not rec.cancelled:
                existing.cancelled = False

    return merged


# ---------------------------------------------------------------------------
# CRM import
# ---------------------------------------------------------------------------

async def import_records(records: list[CustomerRecord], dry_run: bool) -> dict:
    """Create/update contacts in the Atlas CRM."""

    created = 0
    updated = 0
    skipped = 0
    errors = 0

    log_lines = []

    for rec in sorted(records, key=lambda r: r.name.lower()):
        status_marker = "[LEAD/CANCELLED]" if rec.cancelled else ""
        phone_str = rec.phone or "no phone"
        email_str = rec.email or ""
        contact_str = f" ({rec.contact_name})" if rec.contact_name else ""
        tags_str = ",".join(rec.tags)

        line = (
            f"  {status_marker or '          '} {rec.name:<45} "
            f"{phone_str:<18} {email_str:<35} "
            f"{rec.address[:50]:<52} [{tags_str}] events={rec.event_count}"
        )
        print(line)
        log_lines.append(line)

        if dry_run:
            continue

        try:
            from atlas_brain.services.crm_provider import get_crm_provider

            crm = get_crm_provider()
            contact_data: dict = {
                "full_name": rec.name,
                "address": rec.address,
                "contact_type": rec.contact_type,
                "source": "calendar_import",
                "tags": rec.tags,
                "status": "inactive" if rec.cancelled else "active",
            }
            if rec.phone:
                contact_data["phone"] = rec.phone
            if rec.email:
                contact_data["email"] = rec.email
            if rec.contact_name:
                contact_data["notes"] = f"Contact: {rec.contact_name}"
                if rec.notes:
                    contact_data["notes"] += f" | {rec.notes}"
            elif rec.notes:
                contact_data["notes"] = rec.notes

            result = await crm.create_contact(contact_data)
            contact_id = str(result.get("id", ""))

            if contact_id:
                # Log one interaction: most recent cleaning date
                if rec.last_event_date:
                    interaction_summary = (
                        f"Calendar import: {rec.event_count} cleaning event(s). "
                        f"Most recent: {rec.last_event_date.isoformat()}. "
                        f"Source: {rec.source_calendar}"
                    )
                    await crm.log_interaction(
                        contact_id=contact_id,
                        interaction_type="appointment",
                        summary=interaction_summary,
                        occurred_at=datetime.combine(
                            rec.last_event_date, datetime.min.time()
                        ).replace(tzinfo=timezone.utc).isoformat(),
                    )

                # Check if this was truly new or an existing update
                if result.get("created_at") == result.get("updated_at"):
                    created += 1
                else:
                    updated += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"    ERROR: {rec.name} — {e}")
            errors += 1

    return {"created": created, "updated": updated, "skipped": skipped, "errors": errors,
            "log_lines": log_lines}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Import calendar contacts into Atlas CRM")
    parser.add_argument("--dry-run", action="store_true", help="Preview only — no DB writes")
    parser.add_argument(
        "--calendar",
        choices=["commercial", "residential", "one_time", "estimates", "all"],
        default="all",
        help="Which calendar to import (default: all)",
    )
    args = parser.parse_args()

    dry_run = args.dry_run
    mode = "DRY RUN" if dry_run else "LIVE IMPORT"
    print(f"\n{'='*70}")
    print(f"  Atlas CRM — Calendar Contact Import [{mode}]")
    print(f"{'='*70}\n")

    all_records: list[CustomerRecord] = []

    for ics in ICS_FILES:
        if args.calendar != "all" and ics["key"] != args.calendar:
            continue

        path = ics["path"]
        if not path.exists():
            print(f"  SKIP (file not found): {path.name}")
            continue

        is_commercial = ics["key"] == "commercial"
        print(f"Parsing: {ics['label']} ...")
        records = parse_ics(path, ics["tags"], ics["contact_type"], is_commercial)
        print(f"  Found {len(records)} unique locations\n")
        all_records.extend(records)

    print(f"Total before cross-calendar dedup: {len(all_records)}")
    records = dedup_across_calendars(all_records)
    print(f"Total after dedup:                 {len(records)}")

    customers = [r for r in records if r.contact_type == "customer" and not r.cancelled]
    leads     = [r for r in records if r.contact_type == "lead" or r.cancelled]
    print(f"  Customers (active):  {len(customers)}")
    print(f"  Leads/cancelled:     {len(leads)}")

    if not dry_run:
        # Initialize the DB pool explicitly (FastAPI lifespan doesn't run in script context)
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from atlas_brain.storage.database import get_db_pool
        pool = get_db_pool()
        await pool.initialize()
        print("Database pool initialized.\n")

    print(f"\n{'─'*70}")
    print("CUSTOMERS")
    print(f"{'─'*70}")
    result = await import_records(customers, dry_run)

    print(f"\n{'─'*70}")
    print("LEADS / CANCELLED")
    print(f"{'─'*70}")
    result2 = await import_records(leads, dry_run)

    # Summary
    print(f"\n{'='*70}")
    if dry_run:
        print(f"  DRY RUN COMPLETE — no changes made")
        print(f"  Would import {len(customers)} customers + {len(leads)} leads")
        print(f"\n  Run without --dry-run to write to the CRM.")
    else:
        total_created = result["created"] + result2["created"]
        total_updated = result["updated"] + result2["updated"]
        total_errors  = result["errors"]  + result2["errors"]
        print(f"  Created: {total_created}   Updated: {total_updated}   Errors: {total_errors}")

        # Write log
        log_path = Path("data/calendar_import.log")
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"Atlas CRM Calendar Import — {datetime.now().isoformat()}\n")
            f.write(f"Created: {total_created}  Updated: {total_updated}  Errors: {total_errors}\n\n")
            for line in result["log_lines"] + result2["log_lines"]:
                f.write(line + "\n")
        print(f"  Log written to {log_path}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
