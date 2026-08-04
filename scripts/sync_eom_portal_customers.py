#!/usr/bin/env python3
"""Portal -> Atlas CRM customer sync (EOM current-customer master).

#2156 slice A2. The owner's canonical current roster is the portal's
server-owned Customer aggregate (wiw backend, `GET /api/admin/customers`);
calendar history (slice A) is enrichment, not the active set. This script:

  1. authenticates to the backend at RUNTIME (getpass; nothing stored),
  2. fetches the active customers + sites,
  3. resolves each to a CRM contact (atlasContactId first, then the slice-A
     identity ladder: phone -> email -> site addresses),
  4. writes through the slice-A guarded machinery (diffed, archive- and
     tenant-guarded), stamping `metadata.portal_customer_id`,
  5. demotes previously-imported "active customers" that no portal customer
     matched this run to status='inactive' + a `past_customer` tag.

Dry-run by default; pass --apply to write (demotion-bearing script, #2155
convention). Non-interactive runs may supply a pre-obtained token via the
ATLAS_TOOLS_EOM_PORTAL_TOKEN setting/env. Credentials are never written to
disk, argv, or logs.

Usage:
  python scripts/sync_eom_portal_customers.py            # dry-run
  python scripts/sync_eom_portal_customers.py --apply --receipt-dir ~/.local/state/atlas/eom-receipts
  python scripts/sync_eom_portal_customers.py --base-url http://localhost:8000
"""

import argparse
import asyncio
import json
import getpass
import os
import sys
import uuid as _uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))          # sibling script import
sys.path.insert(0, str(Path(__file__).parent.parent))   # atlas_brain import

import import_eom_customers_live as live  # noqa: E402  (slice-A machinery)
from eom_execution_receipt import EomExecutionReceipt, run_receipted  # noqa: E402

EOM_CONTEXT_ID = live.EOM_CONTEXT_ID
DEFAULT_BASE_URL = "https://eom-timetracker.onrender.com"
DEMOTABLE_SOURCES = ("calendar_import", "portal_sync")
# Tags this pipeline manages: the portal replaces them on a match; every
# other (foreign) tag is preserved (Codex A2 round 4).
MANAGED_TAGS = {"portal", "residential", "commercial", "one_time", "past_customer"}


async def _await_receipted_mutation(receipt, awaitable, *, changed_contact_id_from=None):
    if receipt is None:
        return await awaitable
    async with receipt.mutation_boundary():
        result = await awaitable
        if changed_contact_id_from is not None:
            _record_receipt_contact(receipt, changed_contact_id_from(result))
        return result


def _record_receipt_contact(receipt, contact_id) -> None:
    if receipt is not None and contact_id:
        receipt.record_changed_contact_id(contact_id)


def settings_default(attr: str):
    """Typed Atlas settings (pydantic reads .env/.env.local that never reach
    os.environ -- slice-A R11 rule). Never raises in minimal envs."""
    try:
        from atlas_brain.config import settings as _settings
        return getattr(_settings.tools, attr, None)
    except Exception:  # noqa: BLE001 -- settings absent: env/prompt only
        return None


def portal_login(client, base_url: str, env: dict):
    """Runtime auth: env or typed-settings token when provided
    (non-interactive), else a getpass prompt. The password variable never
    leaves this function and is never printed or persisted."""
    token = env.get("ATLAS_TOOLS_EOM_PORTAL_TOKEN") or settings_default("eom_portal_token")
    if token:
        return token
    name = input("Portal admin name: ")
    password = getpass.getpass("Portal admin password (not stored): ")
    resp = client.post(
        f"{base_url}/api/auth/login", json={"name": name, "password": password}
    )
    if resp.status_code != 200:
        raise SystemExit(f"Portal login failed (HTTP {resp.status_code})")
    token = (resp.json() or {}).get("token")
    if not token:
        raise SystemExit("Portal login returned no token")
    return token


def fetch_portal_customers(client, base_url: str, token: str) -> list:
    resp = client.get(
        f"{base_url}/api/admin/customers",
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code != 200:
        raise SystemExit(f"Customer fetch failed (HTTP {resp.status_code})")
    payload = resp.json() or {}
    if not payload.get("success"):
        raise SystemExit("Customer fetch returned success=false")
    customers = [c for c in (payload.get("customers") or [])
                 if c.get("active", True)]
    if not customers:
        # An empty roster would demote the entire base -- refuse to
        # proceed rather than trust it (Codex A2 round 2).
        raise SystemExit("Portal returned 0 active customers; refusing to sync")
    return customers


def active_sites(customer: dict) -> list:
    return [s for s in (customer.get("sites") or []) if s.get("active")]


def segment_tags(customer: dict) -> list:
    tags = {"portal"}
    for site in active_sites(customer):
        lt = str(site.get("locationType") or "").strip().lower()
        if lt in ("residential", "commercial"):
            tags.add(lt)
    return sorted(tags)


def customer_to_contact_data(customer: dict) -> dict:
    """Contact payload for one portal customer. The tenant stamp is
    non-negotiable. New rows receive the complete payload in their initial
    insert; existing rows preserve recorded provenance and replace only the
    portal-managed tags through the atomic reconciliation."""
    data = {
        "full_name": str(customer.get("name") or "").strip(),
        "contact_type": "customer",
        "business_context_id": EOM_CONTEXT_ID,
        "status": "active",
        "tags": segment_tags(customer),
        "source": "portal_sync",
    }
    phone = str(customer.get("primaryPhone") or "").strip()
    if phone:
        data["phone"] = phone
    email = str(customer.get("primaryEmail") or "").strip().lower()
    if email:
        data["email"] = email
    sites = active_sites(customer)
    if sites and sites[0].get("address"):
        data["address"] = str(sites[0]["address"])
    if customer.get("primaryContactName"):
        data["notes"] = f"Contact: {customer['primaryContactName']}"
    return data


async def resolve_contact(customer: dict, data: dict, crm, pool):
    """atlasContactId first (the designed cross-system link), then the
    slice-A identity ladder. Returns (existing_row_or_None, needs_claim,
    identity_tuple_or_None)."""
    pid = customer.get("id")
    if isinstance(pid, int) and not isinstance(pid, bool):
        row = await pool.fetchrow(
            """
            SELECT id, full_name, address, contact_type, business_context_id,
                   tags, status, phone, email, notes, source, metadata
            FROM contacts
            WHERE metadata->>'portal_customer_id' = $1
              AND business_context_id = $2
              AND status != 'archived'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            str(pid),
            EOM_CONTEXT_ID,
        )
        if row:
            return row, False, ("portal_id", str(pid))

    atlas_id = customer.get("atlasContactId")
    if atlas_id is not None:
        try:
            _uuid.UUID(str(atlas_id))
        except (ValueError, AttributeError, TypeError):
            print(f"    note: malformed atlasContactId {atlas_id!r}; "
                  "ignoring the link")
            atlas_id = None
    if atlas_id:
        row = await pool.fetchrow(
            """
            SELECT id, full_name, address, contact_type, business_context_id,
                   tags, status, phone, email, notes, source, metadata
            FROM contacts
            WHERE id = $1 AND status != 'archived'
            """,
            str(atlas_id),
        )
        if row:
            ctx = row.get("business_context_id")
            if ctx == EOM_CONTEXT_ID:
                return row, False, ("id", str(atlas_id))
            if ctx is None:
                # Designed link to a legacy NULL-context row: claim it via
                # the CAS (the id link IS the identity) -- Codex A2 round 1.
                return row, True, ("id", str(atlas_id))
            # Linked to a FOREIGN tenant: fail closed, fall to the ladder
            # and report rather than writing across tenants.
            print(f"    note: atlasContactId {atlas_id} belongs to tenant "
                  f"{ctx!r}; ignoring the link")

    if data.get("phone"):
        digits = live._phone_digits(data["phone"])
        if len(digits) >= 10:
            existing, needs_claim = await live._search_channel(crm, phone=digits)
            if existing is not None:
                return existing, needs_claim, ("phone", digits[-10:])
    if data.get("email"):
        existing, needs_claim = await live._search_channel(crm, email=data["email"])
        if existing is not None:
            return existing, needs_claim, ("email", data["email"])
    for site in active_sites(customer):
        addr = site.get("address")
        if not addr:
            continue
        existing, needs_claim = await live.resolve_by_address(pool, addr)
        if existing is not None:
            return existing, needs_claim, ("address", addr)
    return None, False, None


def portal_identity_keys(customer: dict, data: dict) -> set[tuple[str, str]]:
    """Normalized roster identities used only by the write preflight."""
    keys: set[tuple[str, str]] = set()
    if data.get("phone"):
        digits = live._phone_digits(data["phone"])
        if len(digits) >= 10:
            keys.add(("phone", digits[-10:]))
    if data.get("email"):
        keys.add(("email", str(data["email"]).strip().lower()))
    for site in active_sites(customer):
        address = str(site.get("address") or "").strip().lower()
        if address:
            keys.add(("address", address))
    return keys


async def preflight_roster(customers: list, crm, pool):
    """Resolve the whole roster read-only and report cross-roster collisions.

    Apply consumes these cached resolutions, so a later duplicate cannot be
    discovered only after an earlier row has already been written.
    """
    identity_owners: dict[tuple[str, str], int] = {}
    contact_owners: dict[str, int] = {}
    resolutions = {}
    errors: list[str] = []
    for customer in customers:
        portal_id = int(customer["id"])
        data = customer_to_contact_data(customer)
        for key in portal_identity_keys(customer, data):
            prior = identity_owners.get(key)
            if prior is not None and prior != portal_id:
                errors.append(
                    f"portal customers {prior} and {portal_id} share normalized "
                    f"{key[0]} identity"
                )
            else:
                identity_owners[key] = portal_id
        resolved = await resolve_contact(customer, data, crm, pool)
        resolutions[portal_id] = resolved
        existing = resolved[0]
        if existing is None:
            continue
        contact_id = str(existing["id"])
        prior = contact_owners.get(contact_id)
        if prior is not None and prior != portal_id:
            errors.append(
                f"portal customers {prior} and {portal_id} resolve to CRM "
                f"contact {contact_id}"
            )
        else:
            contact_owners[contact_id] = portal_id
    return resolutions, errors


def parse_meta(existing: dict):
    """asyncpg can deliver JSONB as str; absent metadata (narrow SELECTs)
    reads as unknown ({})."""
    meta = existing.get("metadata") if "metadata" in existing else None
    if isinstance(meta, str):
        try:
            meta = json.loads(meta or "{}")
        except ValueError:
            meta = {}
    return meta or {}


def portal_final_tags(existing: dict, data: dict) -> list:
    prior = set(existing.get("tags") or [])
    return sorted((prior - MANAGED_TAGS) | set(data["tags"]))


async def reconcile_portal_contact(
    pool,
    existing: dict,
    data: dict,
    portal_id: int,
    *,
    needs_claim: bool,
    identity,
):
    """Atomically claim/link/reconcile one existing portal contact.

    The portal-link, tenant/archive, and resolution-identity predicates guard
    the first and only mutation. A rejected statement therefore writes
    nothing, including on provider create races.
    """
    contact_id = str(existing["id"])
    payload = dict(data)
    payload.pop("source", None)
    payload.pop("business_context_id", None)
    payload.pop("tags", None)

    args = [contact_id, portal_id, EOM_CONTEXT_ID]
    sets = [
        "business_context_id = $3",
        (
            "metadata = COALESCE(metadata, '{}'::jsonb) "
            "|| jsonb_build_object('portal_customer_id', $2::int)"
        ),
    ]
    # Compute managed tags from the locked live row. A stale preflight snapshot
    # must never erase a foreign tag added between resolution and this statement.
    args.append(data["tags"])
    desired_tags_param = len(args)
    args.append(sorted(MANAGED_TAGS))
    managed_tags_param = len(args)
    desired_tags_sql = (
        "ARRAY("
        "SELECT DISTINCT tag FROM unnest("
        "COALESCE(contacts.tags, '{}'::text[]) "
        f"|| ${desired_tags_param}::text[]) AS tag "
        f"WHERE NOT (tag = ANY(${managed_tags_param}::text[])) "
        f"OR tag = ANY(${desired_tags_param}::text[]) "
        "ORDER BY tag)"
    )
    sets.append(
        f"tags = {desired_tags_sql}"
    )
    change_conditions = [
        "contacts.business_context_id IS DISTINCT FROM $3",
        "contacts.metadata->>'portal_customer_id' IS DISTINCT FROM $2::text",
        f"contacts.tags IS DISTINCT FROM {desired_tags_sql}",
    ]
    for column, value in payload.items():
        args.append(value.lower() if column == "email" and value else value)
        sets.append(f"{column} = ${len(args)}")
        change_conditions.append(
            f"contacts.{column} IS DISTINCT FROM ${len(args)}"
        )

    identity_clause = ""
    source_clause = ""
    if identity:
        kind, value = identity
        if kind == "address":
            args.append(value)
            identity_clause = f"AND LOWER(address) = LOWER(${len(args)})"
            if needs_claim:
                source_clause = "AND source = 'calendar_import'"
        elif kind == "phone":
            args.append(value)
            identity_clause = (
                "AND regexp_replace(COALESCE(phone,''), '[^0-9]', '', 'g') "
                f"LIKE '%' || ${len(args)} || '%'"
            )
        elif kind == "email":
            args.append(value)
            identity_clause = f"AND LOWER(email) = LOWER(${len(args)})"
        elif kind == "portal_id":
            identity_clause = (
                "AND metadata->>'portal_customer_id' = $2::text"
            )
        # kind == "id" is already guarded by the primary-key predicate.

    tenant_clause = (
        "(business_context_id IS NULL OR business_context_id = $3)"
        if needs_claim
        else "business_context_id = $3"
    )
    row = await pool.fetchrow(
        f"""
        WITH live AS MATERIALIZED (
            SELECT id
            FROM contacts
            WHERE id = $1
              AND status != 'archived'
              AND {tenant_clause}
              AND (metadata->>'portal_customer_id' IS NULL
                   OR metadata->>'portal_customer_id' = $2::text)
              {source_clause}
              {identity_clause}
            FOR UPDATE
        ),
        updated AS (
            UPDATE contacts
               SET {", ".join(sets)}, updated_at = NOW()
              FROM live
             WHERE contacts.id = live.id
               AND ({" OR ".join(change_conditions)})
            RETURNING contacts.id
        )
        SELECT live.id, EXISTS (SELECT 1 FROM updated) AS changed
        FROM live
        """,
        *args,
    )
    if row is None:
        print(
            f"    ERROR: contact {contact_id} changed identity, tenant, archive "
            "state, or portal link before atomic reconciliation; nothing written"
        )
        return contact_id, "rejected"
    return contact_id, "updated" if row["changed"] else "unchanged"


async def portal_id_current(pool, contact_id: str):
    """(row_found, pid) under the same guards as the stamp itself."""
    row = await pool.fetchrow(
        """
        SELECT metadata->>'portal_customer_id' AS pid FROM contacts
        WHERE id = $1
          AND status != 'archived'
          AND business_context_id = $2
        """,
        contact_id,
        EOM_CONTEXT_ID,
    )
    return (row is not None), (row.get("pid") if row else None)


async def sync_one(
    customer: dict, crm, pool, apply: bool, *, resolved=None, receipt=None
) -> tuple:
    """Sync a single portal customer; returns (outcome, contact_id)."""
    data = customer_to_contact_data(customer)
    if not data["full_name"]:
        # A nameless ACTIVE portal record is unprocessable: it may shadow a
        # real matched customer, and a match set missing it must not drive
        # demotions -- error (which also gates demotion) rather than skip
        # (Codex A2 round 3).
        print("    ERROR: active portal customer with blank name; run failed closed")
        return "errors", None
    portal_id = customer.get("id")
    if not isinstance(portal_id, int) or isinstance(portal_id, bool):
        # Validate BEFORE any write: --apply must not activate a contact and
        # only then discover it cannot stamp the predicate (Codex A2 r3).
        print(f"    ERROR: portal customer {customer.get('name')!r} has no "
              "usable id; nothing written")
        return "errors", None

    if resolved is None:
        existing, needs_claim, identity = await resolve_contact(
            customer, data, crm, pool
        )
    else:
        existing, needs_claim, identity = resolved
    if not apply:
        # Dry-run reports the matched id (demotion preview accuracy) AND
        # computes the real diff/stamp need, so a clean re-run previews
        # zero updates (Codex A2 round 4).
        if existing is None:
            return "create-planned", None
        if "metadata" in existing:
            prior_pid = parse_meta(existing).get("portal_customer_id")
            if prior_pid is not None and str(prior_pid) != str(portal_id):
                print(f"    ERROR (previewed): contact {existing['id']} "
                      f"already linked to portal customer {prior_pid}")
                return "errors", None
        else:
            found, pid = await portal_id_current(pool, str(existing["id"]))
            if found and pid not in (None, "") and pid != str(portal_id):
                print(f"    ERROR (previewed): contact {existing['id']} "
                      f"already linked to portal customer {pid}")
                return "errors", None
        payload = dict(data)
        payload["tags"] = portal_final_tags(existing, data)
        payload.pop("source", None)
        payload.pop("business_context_id", None)
        updates = live._diff_updates(existing, payload)
        stamped = ("metadata" in existing and
                   str(parse_meta(existing).get("portal_customer_id", ""))
                   == str(customer.get("id")))
        if updates or not stamped:
            return "update-planned", str(existing["id"])
        return "unchanged", str(existing["id"])

    if existing is not None:
        prior_pid = parse_meta(existing).get("portal_customer_id")
        if prior_pid is not None and str(prior_pid) != str(portal_id):
            # Never bounce a stable portal link between ids: two portal
            # customers sharing a channel/address must be fixed in the
            # portal, not by overwriting the CRM link (Codex A2 round 6).
            print(f"    ERROR: contact {existing['id']} already linked to "
                  f"portal customer {prior_pid}; refusing to relink to "
                  f"{portal_id}")
            return "errors", None

    if existing is not None:
        contact_id, outcome = await _await_receipted_mutation(
            receipt,
            reconcile_portal_contact(
                pool,
                existing,
                data,
                portal_id,
                needs_claim=needs_claim,
                identity=identity,
            ),
            changed_contact_id_from=(
                lambda result: result[0] if result[1] == "updated" else None
            ),
        )
        if outcome == "rejected":
            return "errors", None
    else:
        create_data = dict(data)
        create_data["metadata"] = {"portal_customer_id": portal_id}
        result = await _await_receipted_mutation(
            receipt,
            crm.create_contact(create_data, merge_existing=False),
            changed_contact_id_from=(
                lambda row: row.get("id") if row.get("_was_created") else None
            ),
        )
        contact_id = str(result.get("id", ""))
        if result.get("_was_created"):
            outcome = "created"
        else:
            race_identity = ("email", data["email"]) if data.get("email") else None
            contact_id, outcome = await _await_receipted_mutation(
                receipt,
                reconcile_portal_contact(
                    pool,
                    result,
                    data,
                    portal_id,
                    needs_claim=False,
                    identity=race_identity,
                ),
                changed_contact_id_from=(
                    lambda reconcile_result: (
                        reconcile_result[0]
                        if reconcile_result[1] == "updated"
                        else None
                    )
                ),
            )
            if outcome == "rejected":
                return "errors", None
    return outcome, contact_id


async def fetch_calendar_guard_keys(months_back: int = 1, months_forward: int = 4):
    """Owner rule (2026-07-23): the CALENDAR VETOES DEMOTION. Live booking
    events (recent past through upcoming) yield phone/email/address/name keys;
    any demotion candidate matching one is a CURRENT customer regardless of
    portal state and is kept (reported for portal reconciliation)."""
    from atlas_brain.services.calendar_provider import GoogleCalendarProvider

    ids = live.resolve_calendar_ids(live.effective_calendar_env(os.environ), "all")
    provider = GoogleCalendarProvider()
    keys = {"phones": set(), "emails": set(), "addrs": set(), "names": set()}
    try:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=months_back * 30)
        end = now + timedelta(days=months_forward * 30)
        all_records = []
        for cal in live.BOOKING_CALENDARS:
            events = await provider.list_events(
                start=start, end=end, calendar_id=ids[cal["key"]]
            )
            all_records.extend(live.parse_events(
                events, cal["tags"], "customer",
                cal["key"] == "commercial", cal["label"],
            ))
        # Cross-calendar recency first (Codex 2163 r2): an older active event
        # on one calendar must not veto when a NEWER cancellation on another
        # calendar supersedes it -- the merged record's state decides.
        merged = live.dedup_records(all_records)
        # The merger unions by phone/email/address, so same-NAME records
        # without a shared identity channel stay separate: name keys get
        # their own latest-record-wins pass (Codex 2163 r3).
        name_state = {}
        for rec in merged:
            nm = rec.name.strip().lower()
            dt = getattr(rec, "latest_event_dt", None)
            cur = name_state.get(nm)
            if cur is None or (dt and (cur[0] is None or dt > cur[0])) or (
                    dt and cur[0] and dt == cur[0]
                    and rec.cancelled and not cur[1]):
                # Equal-time ties resolve to cancelled, mirroring the
                # slice-A determinism rule (Codex 2163 r4).
                name_state[nm] = (dt, rec.cancelled)
        for rec in merged:
            if rec.cancelled:
                # A latest-event cancellation is evidence of ENDING, not
                # currency -- it must not veto demotion (Codex 2163 r1-r2).
                continue
            if rec.phone:
                digits = live._phone_digits(rec.phone)
                if len(digits) >= 10:
                    keys["phones"].add(digits[-10:])
            if rec.email:
                keys["emails"].add(rec.email.strip().lower())
            for addr in getattr(rec, "all_addresses", None) or [rec.address]:
                keys["addrs"].add(addr.lower())
        for nm, (_, cancelled) in name_state.items():
            if not cancelled:
                keys["names"].add(nm)
    finally:
        await provider.aclose()
    return keys


def on_calendar(row: dict, guard_keys: dict) -> bool:
    phone = str(row.get("phone") or "")
    if phone:
        digits = live._phone_digits(phone)
        if len(digits) >= 10 and digits[-10:] in guard_keys["phones"]:
            return True
    email = str(row.get("email") or "").strip().lower()
    if email and email in guard_keys["emails"]:
        return True
    addr = str(row.get("address") or "").strip().lower()
    if addr and addr in guard_keys["addrs"]:
        return True
    name = str(row.get("full_name") or "").strip().lower()
    return bool(name) and name in guard_keys["names"]


async def demote_unmatched(pool, matched_ids: set, apply: bool,
                           guard_keys: dict, receipt=None) -> tuple:
    """Previously-imported active 'customers' that matched no portal customer
    this run become past customers -- UNLESS they appear on the live booking
    calendars (owner veto rule). Only rows this pipeline claimed as active
    (calendar_import / portal_sync provenance) are eligible; leads, manual,
    web, and every other source are never touched."""
    rows = []
    demoted = 0
    kept = 0
    try:
        rows = await pool.fetch(
            """
            SELECT id, full_name, tags, phone, email, address FROM contacts
            WHERE business_context_id = $1
              AND contact_type = 'customer'
              AND status = 'active'
              AND source = ANY($2::text[])
            ORDER BY lower(full_name)
            """,
            EOM_CONTEXT_ID,
            list(DEMOTABLE_SOURCES),
        )
        for row in rows:
            if str(row["id"]) in matched_ids:
                continue
            if on_calendar(row, guard_keys):
                kept += 1
                print(f"  KEPT (on your calendar, missing from portal): "
                      f"{row['full_name']} -- add them to the portal")
                continue
            print(f"  DEMOTE (past customer): {row['full_name']}")
            if not apply:
                demoted += 1
                continue
            tags = sorted(set(row["tags"] or []) | {"past_customer"})
            result = await _await_receipted_mutation(
                receipt,
                pool.fetchrow(
                    """
                    UPDATE contacts
                       SET status = 'inactive', tags = $2, updated_at = NOW()
                     WHERE id = $1
                       AND business_context_id = $3
                       AND contact_type = 'customer'
                       AND status = 'active'
                       AND source = ANY($4::text[])
                    RETURNING id
                    """,
                    str(row["id"]),
                    tags,
                    EOM_CONTEXT_ID,
                    list(DEMOTABLE_SOURCES),
                ),
                changed_contact_id_from=lambda row: row and row["id"],
            )
            if result is not None:
                demoted += 1
    except Exception:
        if receipt is not None:
            receipt.record_demotions(demoted=demoted, eligible=len(rows), kept=kept)
        raise
    if kept:
        print(f"\n  {kept} calendar-active customer(s) kept; reconcile them "
              "in the portal.")
    if receipt is not None:
        receipt.record_demotions(demoted=demoted, eligible=len(rows), kept=kept)
    return demoted, len(rows)


async def run(args, receipt=None) -> int:
    import httpx

    counts = {"created": 0, "updated": 0, "unchanged": 0, "skipped": 0,
              "errors": 0, "create-planned": 0, "update-planned": 0}
    if receipt is not None:
        receipt.record_outcome_counts(counts)
    with httpx.Client(timeout=30.0) as client:
        token = portal_login(client, args.base_url, os.environ)
        customers = fetch_portal_customers(client, args.base_url, token)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"\n{'=' * 70}")
    print(f"  Atlas CRM -- EOM portal customer sync [{mode}]")
    print(f"  Portal customers (active): {len(customers)}")
    print(f"{'=' * 70}\n")

    from atlas_brain.services.crm_provider import get_crm_provider
    from atlas_brain.storage.database import get_db_pool

    pool = get_db_pool()
    await pool.initialize()
    crm = get_crm_provider()

    def _malformed(c):
        if not str(c.get("name") or "").strip():
            return True
        if not isinstance(c.get("id"), int) or isinstance(c.get("id"), bool):
            return True
        sites = c.get("sites")
        if sites is None:
            return False
        if not isinstance(sites, list) or any(
                not isinstance(x, dict) for x in sites):
            return True
        return any(
            x.get("locationType") is not None
            and not isinstance(x.get("locationType"), str)
            for x in sites
        )

    invalid = [c for c in customers if _malformed(c)]
    if invalid:
        # Validate the WHOLE roster before any write: a malformed record
        # later in sort order must not leave earlier writes applied with
        # demotion skipped (Codex A2 round 6).
        for c in invalid:
            print(f"  INVALID portal record: id={c.get('id')!r} "
                  f"name={c.get('name')!r}")
        print(f"\n  ABORTED: {len(invalid)} malformed portal record(s); "
              "nothing written.")
        counts["errors"] += len(invalid)
        if receipt is not None:
            receipt.record_outcome_counts(counts)
        return 1

    resolutions = {}
    if args.apply:
        try:
            resolutions, collision_errors = await preflight_roster(
                customers, crm, pool
            )
        except Exception as e:  # noqa: BLE001 -- no writes have started
            print(f"\n  ABORTED: roster preflight failed ({e}); nothing written.")
            counts["errors"] += 1
            if receipt is not None:
                receipt.record_outcome_counts(counts)
            return 1
        if collision_errors:
            for error in collision_errors:
                print(f"  COLLISION: {error}")
            print(
                f"\n  ABORTED: {len(collision_errors)} roster collision(s); "
                "nothing written."
            )
            counts["errors"] += len(collision_errors)
            if receipt is not None:
                receipt.record_outcome_counts(counts)
            return 1

    matched_ids: set = set()
    for customer in sorted(customers, key=lambda c: str(c.get("name") or "").lower()):
        try:
            print(f"  {str(customer.get('name') or '(unnamed)'):<45} "
                  f"sites={len(active_sites(customer))} "
                  f"[{','.join(segment_tags(customer))}]")
            resolved = resolutions.get(customer["id"]) if args.apply else None
            outcome, contact_id = await sync_one(
                customer, crm, pool, args.apply, resolved=resolved, receipt=receipt
            )
            counts[outcome] += 1
            if contact_id:
                matched_ids.add(contact_id)
        except Exception as e:  # noqa: BLE001 -- operator script: report, continue
            print(f"    ERROR: {customer.get('name')} -- {e}")
            counts["errors"] += 1
        if receipt is not None:
            receipt.record_outcome_counts(counts)

    print(f"\n{'-' * 70}")
    if counts["errors"]:
        # An errored customer never reached matched_ids; demoting on a
        # partial sync could retire real customers (Codex A2 round 1,
        # BLOCKER). Fail the run and leave demotion for a clean pass.
        print(f"  DEMOTION SKIPPED: {counts['errors']} sync error(s) -- "
              "a partial match set must not drive demotions.")
        demoted, eligible = 0, 0
    else:
        try:
            guard_keys = await fetch_calendar_guard_keys()
        except (Exception, SystemExit) as e:  # noqa: BLE001 -- fail closed;
            # SystemExit included: missing calendar config must surface as
            # the skip, not kill an --apply run mid-flight (Codex r3).
            print(f"  DEMOTION SKIPPED: calendar guard unavailable ({e}) -- "
                  "refusing to demote without the calendar veto check.")
            counts["errors"] += 1
            guard_keys = None
        if guard_keys is None:
            demoted, eligible = 0, 0
        else:
            try:
                demoted, eligible = await demote_unmatched(
                    pool, matched_ids, args.apply, guard_keys, receipt=receipt
                )
            except Exception:
                counts["errors"] += 1
                if receipt is not None:
                    receipt.record_outcome_counts(counts)
                raise
    if receipt is not None:
        receipt.record_outcome_counts(counts)

    print(f"\n{'=' * 70}")
    if args.apply:
        print(f"  Created: {counts['created']}   Updated: {counts['updated']}   "
              f"Unchanged: {counts['unchanged']}   Demoted: {demoted}   "
              f"Skipped: {counts['skipped']}   Errors: {counts['errors']}")
    else:
        print(f"  DRY RUN -- would create {counts['create-planned']}, "
              f"update {counts['update-planned']}, demote {demoted} "
              f"(of {eligible} eligible). Run with --apply to write.")
    print(f"{'=' * 70}\n")
    return 1 if counts["errors"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sync the portal Customer aggregate into the Atlas CRM"
    )
    parser.add_argument("--apply", action="store_true",
                        help="Write changes (default is dry-run)")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ATLAS_TOOLS_EOM_PORTAL_BASE_URL")
        or settings_default("eom_portal_base_url")
        or DEFAULT_BASE_URL,
    )
    parser.add_argument(
        "--receipt-dir",
        help="Private execution-receipt directory; required with --apply",
    )
    args = parser.parse_args(argv)
    if args.apply and not args.receipt_dir:
        parser.error("--apply requires --receipt-dir")
    receipt = None
    if args.receipt_dir:
        receipt = EomExecutionReceipt(
            receipt_dir=args.receipt_dir,
            tool="sync_eom_portal_customers",
            mode="apply" if args.apply else "dry-run",
            script_path=Path(__file__),
        )
    return run_receipted(receipt, lambda: asyncio.run(run(args, receipt=receipt)))


if __name__ == "__main__":
    sys.exit(main())
