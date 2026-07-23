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
EOM_PORTAL_TOKEN environment variable. Credentials are never written to
disk, argv, or logs.

Usage:
  python scripts/sync_eom_portal_customers.py            # dry-run
  python scripts/sync_eom_portal_customers.py --apply
  python scripts/sync_eom_portal_customers.py --base-url http://localhost:8000
"""

import argparse
import asyncio
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))          # sibling script import
sys.path.insert(0, str(Path(__file__).parent.parent))   # atlas_brain import

import import_eom_customers_live as live  # noqa: E402  (slice-A machinery)

EOM_CONTEXT_ID = live.EOM_CONTEXT_ID
DEFAULT_BASE_URL = "https://eom-timetracker.onrender.com"
DEMOTABLE_SOURCES = ("calendar_import", "portal_sync")


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
    token = env.get("EOM_PORTAL_TOKEN") or settings_default("eom_portal_token")
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
        lt = (site.get("locationType") or "").strip().lower()
        if lt in ("residential", "commercial"):
            tags.add(lt)
    return sorted(tags)


def customer_to_contact_data(customer: dict) -> dict:
    """Contact payload for one portal customer. The tenant stamp is
    non-negotiable; `source`/`tags` ride the slice-A controlled write paths,
    not the create call (race rules, slice A rounds 6-7)."""
    data = {
        "full_name": str(customer.get("name") or "").strip(),
        "contact_type": "customer",
        "business_context_id": EOM_CONTEXT_ID,
        "status": "active",
        "tags": segment_tags(customer),
        "source": "portal_sync",
    }
    if customer.get("primaryPhone"):
        data["phone"] = customer["primaryPhone"]
    if customer.get("primaryEmail"):
        data["email"] = str(customer["primaryEmail"]).strip().lower()
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
    atlas_id = customer.get("atlasContactId")
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
                return row, False, None
            if ctx is None:
                # Designed link to a legacy NULL-context row: claim it via
                # the CAS (the id link IS the identity) -- Codex A2 round 1.
                return row, True, None
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


async def stamp_portal_id(pool, contact_id: str, portal_id: int):
    """Guarded jsonb merge of metadata.portal_customer_id -- the slice-B
    watcher predicate. Same archive/tenant guards as every other write."""
    return await pool.fetchrow(
        """
        UPDATE contacts
           SET metadata = COALESCE(metadata, '{}'::jsonb)
                          || jsonb_build_object('portal_customer_id', $2::int),
               updated_at = NOW()
         WHERE id = $1
           AND status != 'archived'
           AND (business_context_id IS NULL OR business_context_id = $3)
           AND COALESCE(metadata->>'portal_customer_id', '')
               IS DISTINCT FROM $2::text
        RETURNING id
        """,
        contact_id,
        portal_id,
        EOM_CONTEXT_ID,
    )


async def portal_id_already_stamped(pool, contact_id: str, portal_id: int) -> bool:
    row = await pool.fetchrow(
        "SELECT metadata->>'portal_customer_id' AS pid FROM contacts WHERE id = $1",
        contact_id,
    )
    return bool(row) and row.get("pid") == str(portal_id)


async def sync_one(customer: dict, crm, pool, apply: bool) -> tuple:
    """Sync a single portal customer; returns (outcome, contact_id)."""
    data = customer_to_contact_data(customer)
    if not data["full_name"]:
        return "skipped", None

    existing, needs_claim, identity = await resolve_contact(customer, data, crm, pool)
    if not apply:
        # Dry-run still reports the matched id so the demotion preview is
        # accurate (a matched contact is NOT a demotion candidate).
        if existing is not None:
            return "update-planned", str(existing["id"])
        return "create-planned", None

    if existing is not None and needs_claim:
        existing = await live.claim_legacy_row(
            pool, str(existing["id"]),
            require_import_source=(identity is not None and identity[0] == "address"),
            identity=identity,
        )

    if existing is not None:
        contact_id, outcome = await live._update_matched(pool, existing, data)
        if outcome == "skipped":
            return "skipped", contact_id
    else:
        create_data = dict(data)
        create_data.pop("source", None)
        create_data.pop("tags", None)
        result = await crm.create_contact(create_data)
        contact_id = str(result.get("id", ""))
        if result.get("_was_created"):
            outcome = "created"
            if contact_id:
                stamp = await live._guarded_update(
                    pool, contact_id,
                    {"source": "portal_sync", "tags": data["tags"]},
                )
                if stamp is None:
                    print(f"    ERROR: contact {contact_id} changed before the "
                          "provenance stamp; left unstamped")
                    return "errors", contact_id
        else:
            contact_id, outcome = await live._update_matched(pool, result, data)
            if outcome == "skipped":
                return "skipped", contact_id

    portal_id = customer.get("id")
    if not isinstance(portal_id, int):
        # A contact synced without the watcher predicate is a failure, not
        # a silent skip (Codex A2 round 2).
        print(f"    ERROR: portal customer {customer.get('name')!r} has no "
              "usable id; predicate not stamped")
        return "errors", contact_id
    if contact_id:
        stamped = await stamp_portal_id(pool, contact_id, portal_id)
        if stamped is None and not await portal_id_already_stamped(
                pool, contact_id, portal_id):
            print(f"    ERROR: portal-id stamp rejected for contact {contact_id}")
            return "errors", contact_id
    return outcome, contact_id


async def demote_unmatched(pool, matched_ids: set, apply: bool) -> tuple:
    """Previously-imported active 'customers' that matched no portal customer
    this run become past customers. Only rows this pipeline claimed as
    active (calendar_import / portal_sync provenance) are eligible; leads,
    manual, web, and every other source are never touched."""
    rows = await pool.fetch(
        """
        SELECT id, full_name, tags FROM contacts
        WHERE business_context_id = $1
          AND contact_type = 'customer'
          AND status = 'active'
          AND source = ANY($2::text[])
        ORDER BY lower(full_name)
        """,
        EOM_CONTEXT_ID,
        list(DEMOTABLE_SOURCES),
    )
    demoted = 0
    for row in rows:
        if str(row["id"]) in matched_ids:
            continue
        print(f"  DEMOTE (past customer): {row['full_name']}")
        if not apply:
            demoted += 1
            continue
        tags = sorted(set(row["tags"] or []) | {"past_customer"})
        result = await pool.fetchrow(
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
        )
        if result is not None:
            demoted += 1
    return demoted, len(rows)


async def run(args) -> int:
    import httpx

    counts = {"created": 0, "updated": 0, "unchanged": 0, "skipped": 0,
              "errors": 0, "create-planned": 0, "update-planned": 0}
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

    matched_ids: set = set()
    for customer in sorted(customers, key=lambda c: str(c.get("name") or "").lower()):
        print(f"  {customer.get('name'):<45} sites={len(active_sites(customer))} "
              f"[{','.join(segment_tags(customer))}]")
        try:
            outcome, contact_id = await sync_one(customer, crm, pool, args.apply)
            counts[outcome] += 1
            if contact_id:
                matched_ids.add(contact_id)
        except Exception as e:  # noqa: BLE001 -- operator script: report, continue
            print(f"    ERROR: {customer.get('name')} -- {e}")
            counts["errors"] += 1

    print(f"\n{'-' * 70}")
    if counts["errors"]:
        # An errored customer never reached matched_ids; demoting on a
        # partial sync could retire real customers (Codex A2 round 1,
        # BLOCKER). Fail the run and leave demotion for a clean pass.
        print(f"  DEMOTION SKIPPED: {counts['errors']} sync error(s) -- "
              "a partial match set must not drive demotions.")
        demoted, eligible = 0, 0
    else:
        demoted, eligible = await demote_unmatched(pool, matched_ids, args.apply)

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


def main():
    parser = argparse.ArgumentParser(
        description="Sync the portal Customer aggregate into the Atlas CRM"
    )
    parser.add_argument("--apply", action="store_true",
                        help="Write changes (default is dry-run)")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EOM_PORTAL_BASE_URL")
        or settings_default("eom_portal_base_url")
        or DEFAULT_BASE_URL,
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
