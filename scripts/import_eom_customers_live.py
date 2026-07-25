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
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))          # sibling script import
sys.path.insert(0, str(Path(__file__).parent.parent))   # atlas_brain import

import import_calendar_contacts as ics  # noqa: E402  (reused extraction core)
from eom_execution_receipt import (  # noqa: E402
    EomExecutionReceipt,
    run_receipted,
)

EOM_CONTEXT_ID = "effingham_maids"

# The three booking calendars. Estimates is intentionally absent: it holds
# leads, not customers, and the real-time intake endpoint owns leads now.
BOOKING_CALENDARS = [
    {
        "key": "commercial",
        "env": "EOM_CALENDAR_COMMERCIAL",
        "settings_attr": "eom_calendar_commercial",
        "tags": ["commercial"],
        "label": "Commercial Customers",
    },
    {
        "key": "residential",
        "env": "EOM_CALENDAR_RESIDENTIAL",
        "settings_attr": "eom_calendar_residential",
        "tags": ["residential"],
        "label": "Residential Customers",
    },
    {
        "key": "one_time",
        "env": "EOM_CALENDAR_ONE_TIME",
        "settings_attr": "eom_calendar_one_time",
        "tags": ["one_time"],
        "label": "One-Time Cleanings",
    },
]


def effective_calendar_env(env: dict, tools=None) -> dict:
    """Merge typed Atlas settings with raw environment variables.

    The Atlas config loads .env/.env.local through pydantic-settings, so ids
    recorded there (ATLAS_TOOLS_EOM_CALENDAR_*) never appear in os.environ
    (Codex R11). Settings provide the base; the process environment wins so
    an ad-hoc operator run can still override.
    """
    if tools is None:
        try:
            from atlas_brain.config import settings as _settings

            tools = _settings.tools
        except Exception as exc:  # noqa: BLE001 -- minimal envs run env-only
            print(
                f"  note: Atlas settings unavailable "
                f"({exc.__class__.__name__}); using process env only"
            )
            tools = None
    merged: dict = {}
    if tools is not None:
        for cal in BOOKING_CALENDARS:
            val = getattr(tools, cal["settings_attr"], None)
            if val:
                merged[cal["env"]] = val
    for cal in BOOKING_CALENDARS:
        if env.get(cal["env"]):
            merged[cal["env"]] = env[cal["env"]]
    return merged


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
    is by normalized address; richest contact info wins. Cancellation recency
    is tracked at full event-timestamp granularity so a same-day later
    CANCELLED marker still wins (Codex round 5, R1)."""
    by_address = {}
    latest_dt = {}   # addr_key -> (event datetime, cancelled)

    for ev in events:
        if (ev.status or "").lower() == "cancelled":
            continue

        raw_summary = (ev.summary or "").strip()
        raw_location = (ev.location or "").strip()
        raw_description = ics._strip_html(ev.description or "")
        # GoogleCalendarProvider maps a missing summary to "Untitled" --
        # treat it as blank or nameless placeholder events import as
        # customers (Codex round 15).
        if not raw_summary or raw_summary.lower() == "untitled" or not raw_location:
            continue

        name, cancelled = ics._clean_summary(raw_summary, is_commercial)
        if not name:
            continue

        address = ics._normalize_address(raw_location)
        if len(address) < 5:
            continue

        phone = _extract_phone_live(raw_description)
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
        if ev.start is not None:
            cur = latest_dt.get(addr_key)
            # Equal-start ties resolve to cancelled deterministically,
            # matching the cross-calendar rule (Codex rounds 17-18).
            if cur is None or ev.start > cur[0] or (
                    ev.start == cur[0] and cancelled and not cur[1]):
                latest_dt[addr_key] = (ev.start, cancelled)
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
            if ev.start:
                if phone:
                    by_address[addr_key]._phone_dt = ev.start
                if email:
                    by_address[addr_key]._email_dt = ev.start
        else:
            rec = by_address[addr_key]
            rec.event_count += 1
            if len(name) < len(rec.name):
                rec.name = name
            # The latest event's channel wins, tracked PER FIELD so a newer
            # email cannot make an older phone look fresh (Codex rounds
            # 13+15, R1).
            if phone and (not rec.phone or (ev.start and (
                    getattr(rec, "_phone_dt", None) is None or ev.start > rec._phone_dt))):
                rec.phone = phone
                if ev.start:
                    rec._phone_dt = ev.start
            if email and (not rec.email or (ev.start and (
                    getattr(rec, "_email_dt", None) is None or ev.start > rec._email_dt))):
                rec.email = email
                if ev.start:
                    rec._email_dt = ev.start
            if not rec.contact_name and contact_name:
                rec.contact_name = contact_name
            if not rec.notes and notes:
                rec.notes = notes
            if event_date and (rec.last_event_date is None or event_date > rec.last_event_date):
                rec.last_event_date = event_date

    # The latest event (full timestamp) decides the cancellation state
    # (Codex rounds 3+5, R1): a newer CANCELLED booking re-flags the record
    # even on the same calendar day as an active one. The timestamp rides
    # along for cross-calendar same-day ties (Codex round 13).
    for addr_key, (dt, cancelled) in latest_dt.items():
        by_address[addr_key].cancelled = cancelled
        by_address[addr_key].latest_event_dt = dt

    return list(by_address.values())


def _phone_digits(phone: str) -> str:
    """Digits of the base number with any trailing extension stripped:
    last-10 matching on '217-555-9999 ext 123' would otherwise search for
    '5559999123' and miss the tenant contact (Codex round 5, R1/R8)."""
    base = re.split(r"(?:ext\.?|x)\s*\d+\s*$", phone, flags=re.IGNORECASE)[0]
    return "".join(c for c in base if c.isdigit())


def _extract_phone_live(text: str):
    """ics._extract_phone splits extensions case-sensitively, so 'X42'/'EXT 42'
    leak extension digits into the base number ('755-599-9942 ext 42').
    Lowercase the ext/x tokens ahead of extraction (Codex round 8)."""
    normalized = re.sub(
        r"(?i)\b(ext\.?|x)(?=\s*\d)", lambda m: m.group(0).lower(), text
    )
    return ics._extract_phone(normalized)


def dedup_records(records):
    """Cross-calendar merge with TRANSITIVE phone/address identity.

    Replaces the inherited ics.dedup_across_calendars for the live path
    (Codex rounds 4-5, R1/R8): that merger is not transitive (a
    phone-merged record keeps only its first address in the index, so a
    later address-only event at the second address duplicates the customer)
    and it clears cancellation whenever any merged record is active. Here:
    union-find over phone/address keys, the inherited field-merge rules per
    group, and recency-based cancellation — the latest event date decides,
    with cancelled winning a same-date cross-calendar tie (conservative)."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    key_owner = {}
    for i, rec in enumerate(records):
        keys = []
        # Extension-stripped key: ics._phone_key folds 'ext 123' digits into
        # the last-10 window and would split one customer in two (round 6).
        digits = _phone_digits(rec.phone) if rec.phone else ""
        pk = digits[-10:] if len(digits) >= 10 else None
        if pk:
            keys.append(("phone", pk))
        if rec.email:
            keys.append(("email", rec.email.lower()))
        keys.append(("addr", rec.address.lower()))
        for k in keys:
            if k in key_owner:
                ri, rj = find(key_owner[k]), find(i)
                if ri != rj:
                    parent[rj] = ri
            else:
                key_owner[k] = i

    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(records[i])

    _TYPE = {"customer": 0, "lead": 1}
    merged = []
    for group in groups.values():
        base = group[0]

        def _dt(r):
            dt = getattr(r, "latest_event_dt", None)
            if dt is None and r.last_event_date:
                from datetime import datetime as _dtm, time as _t, timezone as _tz
                dt = _dtm.combine(r.last_event_date, _t.min).replace(tzinfo=_tz.utc)
            return dt

        def _field_key(attr):
            def key(r):
                from datetime import datetime as _dtm, timezone as _tz
                return (getattr(r, attr, None) or _dt(r)
                        or _dtm(1970, 1, 1, tzinfo=_tz.utc))
            return key

        # Channels merge by PER-FIELD event recency, matching the
        # in-calendar rule (Codex rounds 14-15, P1): the latest record with
        # a value for THAT field wins.
        phones = [r for r in group if r.phone]
        if phones:
            base.phone = max(phones, key=_field_key("_phone_dt")).phone
        emails = [r for r in group if r.email]
        if emails:
            base.email = max(emails, key=_field_key("_email_dt")).email

        for rec in group[1:]:
            if len(rec.name) < len(base.name):
                base.name = rec.name
            if not base.contact_name and rec.contact_name:
                base.contact_name = rec.contact_name
            if not base.notes and rec.notes:
                base.notes = rec.notes
            for tag in rec.tags:
                if tag not in base.tags:
                    base.tags.append(tag)
            if _TYPE.get(rec.contact_type, 99) < _TYPE.get(base.contact_type, 99):
                base.contact_type = rec.contact_type
            base.event_count += rec.event_count
        base.all_addresses = []
        for r in group:
            if r.address.lower() not in [a.lower() for a in base.all_addresses]:
                base.all_addresses.append(r.address)
        dated = [(_dt(r), r.cancelled, r.address, r.last_event_date)
                 for r in group if _dt(r) is not None]
        if dated:
            latest = max(d for d, _, _, _ in dated)
            base.latest_event_dt = latest
            base.last_event_date = max(ld for _, _, _, ld in dated)
            # full-timestamp recency decides cross-calendar too; an
            # equal-timestamp tie resolves to cancelled deterministically,
            # independent of input order (rounds 13+17)
            base.cancelled = any(c for d, c, _, _ in dated if d == latest)
            # The customer's CURRENT address is the latest-dated one; keep
            # every group address for fallback lookup so an existing
            # address-only contact at any of them is enriched, never
            # duplicated (Codex round 9, P1).
            base.address = next(a for d, _, a, _ in dated if d == latest)
        merged.append(base)
    return merged


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
    (Codex R4/R8).

    Returns (row, needs_claim): the same-tenant page is checked first; a
    NULL-context legacy row (pre-#2155 history that the provenance backfill
    could not classify) is returned with needs_claim=True so the caller can
    claim it through the provider's CAS instead of duplicating it
    (Codex round 2, R4/R8)."""
    row = await pool.fetchrow(
        """
        SELECT id, full_name, address, contact_type, business_context_id,
               tags, status, phone, email, notes, source
        FROM contacts
        WHERE business_context_id = $1
          AND status != 'archived'
          AND address IS NOT NULL AND LOWER(address) = LOWER($2)
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        EOM_CONTEXT_ID,
        address,
    )
    if row:
        return row, False
    legacy = await pool.fetchrow(
        """
        SELECT id, full_name, address, contact_type, business_context_id,
               tags, status, phone, email, notes, source
        FROM contacts
        WHERE business_context_id IS NULL
          AND status != 'archived'
          AND source = 'calendar_import'
          AND address IS NOT NULL AND LOWER(address) = LOWER($1)
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        address,
    )
    return legacy, legacy is not None


def _diff_updates(existing: dict, data: dict) -> dict:
    """Subset of `data` that actually differs from the stored row."""
    changed = {}
    for k, v in data.items():
        if k == "tags":
            if sorted(existing.get("tags") or []) != sorted(v or []):
                changed[k] = v
        elif existing.get(k) != v:
            changed[k] = v
    return changed


async def claim_legacy_row(pool, contact_id: str, require_import_source: bool,
                           identity: tuple = None):
    """CAS claim of a NULL-context legacy row with the FULL guard inside the
    UPDATE itself: unarchived always, calendar_import provenance additionally
    for weak (address-only) identity, and the MATCHED IDENTITY itself
    (address / phone last-10 / email) so a row corrected between SELECT and
    claim no longer matches the predicate and is never tenant-stamped
    (Codex rounds 8 + 12)."""
    src_clause = "AND source = 'calendar_import'" if require_import_source else ""
    id_clause, id_val = "", None
    if identity:
        kind, id_val = identity
        if kind == "address":
            id_clause = "AND LOWER(address) = LOWER($3)"
        elif kind == "phone":
            id_clause = ("AND regexp_replace(COALESCE(phone,''), '[^0-9]', '', 'g')"
                         " LIKE '%' || $3 || '%'")
        elif kind == "email":
            id_clause = "AND LOWER(email) = LOWER($3)"
    args = [contact_id, EOM_CONTEXT_ID] + ([id_val] if id_clause else [])
    return await pool.fetchrow(
        f"""
        UPDATE contacts
           SET business_context_id = $2, updated_at = NOW()
         WHERE id = $1
           AND (business_context_id IS NULL OR business_context_id = $2)
           AND status != 'archived'
           {src_clause}
           {id_clause}
        RETURNING id, full_name, address, contact_type, business_context_id,
                  tags, status, phone, email, notes, source
        """,
        *args,
    )


_GUARDED_COLUMNS = (
    "full_name", "address", "contact_type", "business_context_id",
    "tags", "status", "phone", "email", "notes", "source",
)


async def _guarded_update(pool, contact_id: str, updates: dict):
    """UPDATE with the archived guard INSIDE the statement: an operator
    archiving the contact between resolution and this write must not be
    resurrected by the import (Codex round 9, R4/R8). Returns the row or
    None when the guard rejected the write."""
    cols = [k for k in updates if k in _GUARDED_COLUMNS]
    sets = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(cols))
    vals = [updates[k].lower() if k == "email" and updates[k] else updates[k] for k in cols]
    ctx_param = len(cols) + 2
    return await pool.fetchrow(
        f"""
        UPDATE contacts
           SET {sets}, updated_at = NOW()
         WHERE id = $1
           AND status != 'archived'
           AND (business_context_id IS NULL OR business_context_id = ${ctx_param})
        RETURNING id
        """,
        contact_id,
        *vals,
        EOM_CONTEXT_ID,
    )


async def _update_matched(pool, existing: dict, data: dict):
    """Reconcile an imported record onto an existing contact: tags UNION
    (Codex round 4), a field diff so unchanged rows get zero writes
    (Codex round 5, P1), NEVER `source` -- schema-default 'manual' rows are
    legitimate recorded provenance, so only the create path stamps
    calendar_import; the failed-post-create-stamp window is an accepted
    provenance-label trade-off, reported as the run error (Codex rounds 7
    and 9) -- all through the archive-guarded UPDATE."""
    contact_id = str(existing["id"])
    payload = dict(data)
    prior = existing.get("tags") or []
    payload["tags"] = sorted(set(prior) | set(payload["tags"]))
    payload.pop("source", None)
    # Resolution already established tenancy; re-sending the stamp could
    # steal a concurrently reassigned row back across tenants (round 11).
    payload.pop("business_context_id", None)
    updates = _diff_updates(existing, payload)
    if updates:
        row = await _guarded_update(pool, contact_id, updates)
        if row is None:
            print(f"    note: contact {contact_id} archived mid-run; write skipped")
            # Distinct outcome: the caller must not touch the archived
            # contact's timeline either (Codex round 10, R4/R8).
            return contact_id, "skipped"
        return contact_id, "updated"
    return contact_id, "unchanged"


async def _search_channel(crm, **channel):
    """Provider-order channel resolution: same-tenant page first, then the
    NULL-context claimable page (mirrors create_contact's own _resolve)."""
    scoped = await crm.search_contacts(business_context_id=EOM_CONTEXT_ID, **channel)
    if scoped:
        return scoped[0], False
    legacy = await crm.search_contacts(business_context_id_is_null=True, **channel)
    return (legacy[0], True) if legacy else (None, False)


async def import_one(rec, crm, pool, receipt=None) -> str:
    """Import a single record; returns 'created' or 'updated'.

    Resolution order: tenant-scoped phone, then email (provider priority),
    then the address net -- BEFORE any create. Without the address step a
    previously address-only row would be duplicated the moment a calendar
    edit adds a phone/email, because create_contact only dedupes on those
    two channels (Codex round 4, R8)."""
    data = record_to_contact_data(rec)

    existing, needs_claim = None, False
    claim_by_address = False
    matched_identity = None
    if rec.phone:
        digits = _phone_digits(rec.phone)
        if len(digits) >= 10:
            existing, needs_claim = await _search_channel(crm, phone=digits)
            if existing is not None:
                matched_identity = ("phone", digits[-10:])
    if existing is None and rec.email:
        existing, needs_claim = await _search_channel(crm, email=rec.email)
        if existing is not None:
            matched_identity = ("email", rec.email)
    if existing is None:
        candidates = [rec.address] + [
            a for a in (getattr(rec, "all_addresses", None) or [])
            if a.lower() != rec.address.lower()
        ]
        for addr in candidates:
            existing, needs_claim = await resolve_by_address(pool, addr)
            if existing is not None:
                matched_identity = ("address", addr)
                break
        claim_by_address = needs_claim

    if existing is not None and needs_claim:
        # Script-side CAS with the full guard IN the claim itself: the
        # provider's claim_contact guards context only, which would leave a
        # raced (archived / source-corrected) row tenant-stamped before any
        # post-hoc re-check could reject it (Codex round 8, P1). Identity
        # from a phone/email match is strong, so those claims require only
        # unarchived; the weaker address-only match additionally requires
        # calendar_import provenance (rounds 3+7 semantics, race-proof).
        existing = await claim_legacy_row(
            pool, str(existing["id"]),
            require_import_source=claim_by_address,
            identity=matched_identity,
        )
        if existing is not None and receipt is not None:
            receipt.record_changed_contact_id(existing["id"])

    if existing is not None:
        contact_id, outcome = await _update_matched(pool, existing, data)
        if outcome == "updated" and receipt is not None:
            receipt.record_changed_contact_id(contact_id)
    else:
        create_data = dict(data)
        # create_contact can still race-merge into a contact created
        # concurrently (e.g. the real-time intake); its merge would apply
        # `source` and `tags` wholesale, clobbering recorded provenance --
        # so both ride the controlled paths only (Codex rounds 6-7, R1/R8).
        create_data.pop("source", None)
        create_data.pop("tags", None)
        result = await crm.create_contact(create_data)
        if result.get("_was_created"):
            contact_id = str(result.get("id", ""))
            outcome = "created"
            if contact_id and receipt is not None:
                receipt.record_changed_contact_id(contact_id)
            if contact_id:
                stamp = await _guarded_update(
                    pool, contact_id,
                    {"source": "calendar_import", "tags": data["tags"]},
                )
                if stamp is None:
                    # Archived/reassigned within the create window: the row
                    # exists but is unstamped -- report it as a run FAILURE
                    # so exit_code_for fails the run (Codex rounds 15+17,
                    # R6/R8).
                    print(f"    ERROR: contact {contact_id} changed before "
                          "the provenance stamp; left unstamped")
                    outcome = "errors"
        else:
            # Race-merged: reconcile exactly like any matched contact
            # (Codex round 7, R1/R8).
            contact_id = str(result.get("id", ""))
            if contact_id and receipt is not None:
                # create_contact's default existing-match path always calls
                # update_contact for this non-empty payload before returning.
                receipt.record_changed_contact_id(contact_id)
            contact_id, outcome = await _update_matched(pool, result, data)
            if outcome == "updated" and receipt is not None:
                receipt.record_changed_contact_id(contact_id)

    if contact_id and rec.last_event_date and outcome != "skipped":
        # The archive can land after the contact write but before this log:
        # re-check so an archived contact's timeline is never touched
        # (Codex round 11, R4/R8).
        row = await pool.fetchrow(
            "SELECT status, business_context_id FROM contacts WHERE id = $1",
            contact_id,
        )
        if (
            row is None
            or row.get("status") == "archived"
            or row.get("business_context_id") != EOM_CONTEXT_ID
        ):
            return outcome
        # source_ref inside metadata is a dedupe anchor (migration 256): the
        # same address never accumulates duplicate import interactions across
        # re-runs.
        interaction = await crm.log_interaction(
            contact_id=contact_id,
            interaction_type="appointment",
            summary=(
                f"Live calendar import: {rec.event_count} booking event(s). "
                f"Most recent: {rec.last_event_date.isoformat()}. "
                f"Source: {rec.source_calendar}"
            ),
            occurred_at=(
                getattr(rec, "latest_event_dt", None)
                or datetime.combine(rec.last_event_date, datetime.min.time())
                .replace(tzinfo=timezone.utc)
            ).isoformat(),
            # Date-scoped anchor: re-runs with the same latest booking dedupe,
            # while each NEW latest booking advances the timeline instead of
            # colliding with the old anchored row forever (Codex round 11,
            # R1/R6).
            metadata={
                # Timestamp-scoped: a same-day newer booking advances the
                # timeline too (Codex rounds 11+15, R1/R6).
                "source_ref": (
                    f"eom_live_calendar:{rec.address.lower()}:"
                    + (getattr(rec, "latest_event_dt", None)
                       or rec.last_event_date).isoformat()
                )
            },
        )
        if interaction.get("inserted") is True and receipt is not None:
            receipt.record_changed_contact_id(contact_id)
    return outcome


def exit_code_for(counts: dict) -> int:
    """Non-zero when any record failed: a partially populated customer master
    must not read as success to a shell or operator runbook (Codex R6)."""
    return 1 if counts.get("errors") else 0


async def run_import(records, dry_run: bool, receipt=None) -> dict:
    counts = {
        "created": 0,
        "updated": 0,
        "unchanged": 0,
        "skipped": 0,
        "errors": 0,
        "import-planned": len(records) if dry_run else 0,
    }
    if receipt is not None:
        receipt.set_outcome_counts(counts)
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
            counts[await import_one(rec, crm, pool, receipt=receipt)] += 1
        except Exception as e:  # noqa: BLE001 -- operator script: report and continue
            print(f"    ERROR: {rec.name} -- {e}")
            counts["errors"] += 1
        if receipt is not None:
            receipt.set_outcome_counts(counts)
    return counts


def _parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--receipt-dir",
        help="Private execution-receipt directory; required for live writes",
    )
    return parser


async def run(args, receipt=None):

    calendar_ids = resolve_calendar_ids(effective_calendar_env(os.environ), args.calendar)

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
    records = dedup_records(all_records)
    print(f"Total after dedup:                 {len(records)}")

    if not args.dry_run:
        from atlas_brain.storage.database import get_db_pool

        pool = get_db_pool()
        await pool.initialize()
        print("Database pool initialized.\n")

    counts = await run_import(records, args.dry_run, receipt=receipt)

    print(f"\n{'=' * 70}")
    if args.dry_run:
        print(f"  DRY RUN COMPLETE -- would import {len(records)} customers")
        print("  Run without --dry-run to write to the CRM.")
    else:
        print(
            f"  Created: {counts['created']}   Updated: {counts['updated']}   "
            f"Unchanged: {counts['unchanged']}   Skipped: {counts['skipped']}   "
            f"Errors: {counts['errors']}"
        )
    print(f"{'=' * 70}\n")
    return exit_code_for(counts)


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.dry_run and not args.receipt_dir:
        parser.error("live writes require --receipt-dir")
    receipt = None
    if args.receipt_dir:
        receipt = EomExecutionReceipt(
            receipt_dir=args.receipt_dir,
            tool="import_eom_customers_live",
            mode="dry-run" if args.dry_run else "write",
            script_path=Path(__file__),
        )
    return run_receipted(receipt, lambda: asyncio.run(run(args, receipt=receipt)))


if __name__ == "__main__":
    sys.exit(main())
