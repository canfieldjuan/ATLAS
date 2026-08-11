"""Backfill EOM contacts.customer_type from tracker site evidence (Slice 1 / Req A).

Migration 366 adds the column defaulting every row to ``unknown``. The evidence
that resolves it already exists, but it lives in a DIFFERENT database: the
tracker's ``locations.location_type``, human-entered per site, and the tracker's
``customers.atlas_contact_id`` linking each account to its Atlas contact.

**This script deliberately does not reach into the tracker.** It takes the
mapping as input and applies it to Atlas. Two reasons: an Atlas maintenance
script should not carry Render credentials for a second datastore, and a script
whose input is a file can be tested end-to-end in CI, which one that dials out
to production cannot.

Produce the mapping from the tracker (read-only), then feed it here:

    render psql <tracker-db> --confirm -c "\\copy (
        SELECT c.atlas_contact_id,
               CASE
                 WHEN bool_and(l.location_type = 'Commercial')  THEN 'commercial'
                 WHEN bool_and(l.location_type = 'Residential') THEN 'residential'
               END AS customer_type,
               'tracker.customer:' || c.id || ' sites=' || count(l.id) AS evidence
          FROM customers c
          JOIN locations l ON l.customer_id = c.id AND l.archived_at IS NULL
         WHERE c.atlas_contact_id IS NOT NULL AND c.active
         GROUP BY c.id, c.atlas_contact_id
        HAVING count(*) FILTER (WHERE l.location_type IS NULL) = 0
           AND (bool_and(l.location_type = 'Commercial')
             OR bool_and(l.location_type = 'Residential'))
    ) TO STDOUT WITH CSV HEADER" > customer_type_mapping.csv

    python scripts/backfill_eom_customer_type.py customer_type_mapping.csv
    python scripts/backfill_eom_customer_type.py customer_type_mapping.csv --apply

The HAVING is the point: a customer whose sites disagree produces no row at all
and therefore stays ``unknown``. Mixed-type accounts are a real possibility and
guessing one would put a residential customer into commercial billing, or hide
billing fields from a commercial one. Silence is the correct output for them.

The explicit NULL count is load-bearing and not defensive noise. ``bool_and``
IGNORES NULL inputs, so ``bool_and(location_type = 'Commercial')`` returns TRUE
for a customer with one Commercial site and one untyped site -- a confident
classification drawn from partial evidence, which is exactly the outcome this
script exists to avoid. Verified: ``bool_and`` over ``('Commercial', NULL)``
evaluates to ``true``. Live data currently has zero untyped sites on active
customers, so nothing is misclassified today; the guard is there for the first
site somebody adds without a type.

Safety properties, each enforced below rather than assumed:

* **Dry run by default.** ``--apply`` writes; nothing else does.
* **Tenant-scoped.** Only ``effingham_maids`` contacts are ever updated. A
  mapping row naming a contact in another business context is refused, not
  skipped quietly.
* **Never overwrites a decision.** Only rows still ``unknown`` are touched, so
  re-running is a no-op and an operator's later correction in the CRM survives a
  second run of a stale mapping file.
* **Every row accounted for.** Applied, already-set, unknown-contact,
  wrong-tenant, conflict and rejected-value are each counted and printed. A row
  that does nothing still appears in the report.
* **Conflicting duplicate links are refused up front.** Two tracker customers
  may share one Atlas contact, so the mapping can name the same contact twice.
  Identical values are harmless; disagreeing ones would let row order decide a
  billing-driving classification, so the whole mapping is rejected before any
  write.
"""

import argparse
import asyncio
import csv
import sys
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EOM_CONTEXT = "effingham_maids"
# Bound to EOM_CUSTOMER_TYPES / chk_contacts_customer_type. 'unknown' is
# deliberately NOT accepted from a mapping file: writing it changes nothing and
# would only mask a mapping that failed to classify.
APPLICABLE_TYPES = ("residential", "commercial")

SQL_TARGET = """
    SELECT id, business_context_id, customer_type
      FROM contacts
     WHERE id = $1::uuid
"""

SQL_APPLY = """
    UPDATE contacts
       SET customer_type = $2, updated_at = NOW()
     WHERE id = $1::uuid
       AND business_context_id = $3
       AND customer_type = 'unknown'
 RETURNING id
"""


def read_mapping(path: Path) -> list[dict[str, str]]:
    """Parse the mapping CSV, refusing malformed rows rather than skipping them."""
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"atlas_contact_id", "customer_type"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"mapping is missing column(s): {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            contact_id = (row.get("atlas_contact_id") or "").strip()
            customer_type = (row.get("customer_type") or "").strip().lower()
            if not contact_id:
                raise SystemExit(f"line {line_number}: atlas_contact_id is empty")
            # Parsed here, before any write happens. The apply loop commits per
            # row, so a malformed id discovered mid-run would raise on the
            # `$1::uuid` cast after earlier rows had already been committed --
            # ending the run with a partial backfill and no summary, which is
            # the one outcome an operator cannot act on.
            try:
                # Store the CANONICAL form, not the caller's spelling. Python
                # accepts spellings PostgreSQL does not -- `urn:uuid:<id>` and
                # brace/dash variants all parse here and then fail on the
                # `$1::uuid` cast mid-run, which is the very partial-backfill
                # failure this validation was added to prevent.
                contact_id = str(uuid.UUID(contact_id))
            except ValueError:
                raise SystemExit(
                    f"line {line_number}: atlas_contact_id is not a UUID: {contact_id!r}"
                ) from None
            rows.append(
                {
                    "contact_id": contact_id,
                    "customer_type": customer_type,
                    "evidence": (row.get("evidence") or "").strip(),
                    "line": str(line_number),
                }
            )

    # Two TRACKER customers can share one Atlas contact -- the query groups by
    # tracker customer, and no unique index forbids the duplicate link yet.
    # ("Mid Illinois Concrete" and "Mid Illinois Concrete - Pike & Raney" share
    # one contact today; both are commercial, so they agree, but agreement is
    # not guaranteed.) If they disagreed, the apply loop would commit whichever
    # CSV row came first and report the other as already-set: an arbitrary
    # billing-driving value decided by row order. Refuse the whole mapping
    # instead, before anything is written.
    by_contact: dict[str, dict[str, str]] = {}
    for row in rows:
        # Only rows that could actually be applied. A pair of unusable values
        # is not a conflict worth a confusing error -- each is rejected on its
        # own merits by the apply loop.
        if row["customer_type"] not in APPLICABLE_TYPES:
            continue
        seen = by_contact.get(row["contact_id"])
        if seen is None:
            by_contact[row["contact_id"]] = row
        elif seen["customer_type"] != row["customer_type"]:
            raise SystemExit(
                f"line {row['line']}: {row['contact_id']} is mapped to "
                f"{row['customer_type']!r} but line {seen['line']} maps it to "
                f"{seen['customer_type']!r}; resolve the duplicate tracker link "
                f"before backfilling"
            )
    return rows


async def run(mapping_path: Path, apply: bool, pool: object | None = None) -> int:
    """Apply the mapping. ``pool`` is injectable so a test can supply its own.

    Patching ``atlas_brain.storage.database.get_db_pool`` would work too, and
    would couple every test here to a private module attribute -- the same
    reason the CRM provider takes a pool rather than reaching for the global.
    """
    rows = read_mapping(mapping_path)
    if pool is None:
        from atlas_brain.storage.database import get_db_pool

        pool = get_db_pool()
    await pool.initialize()

    tally: Counter[str] = Counter()
    problems: list[str] = []

    print(f"mode: {'APPLY' if apply else 'DRY RUN'}   mapping: {mapping_path}")
    print(f"rows: {len(rows)}\n")

    for row in rows:
        contact_id = row["contact_id"]
        customer_type = row["customer_type"]
        label = f"line {row['line']} {contact_id}"

        if customer_type not in APPLICABLE_TYPES:
            tally["rejected-value"] += 1
            problems.append(f"{label}: refusing customer_type={customer_type!r}")
            continue

        target = await pool.fetchrow(SQL_TARGET, contact_id)
        if target is None:
            tally["unknown-contact"] += 1
            problems.append(f"{label}: no such contact")
            continue
        if target["business_context_id"] != EOM_CONTEXT:
            tally["wrong-tenant"] += 1
            problems.append(
                f"{label}: business_context_id={target['business_context_id']!r}"
            )
            continue
        if target["customer_type"] != "unknown":
            tally["already-set"] += 1
            if target["customer_type"] != customer_type:
                problems.append(
                    f"{label}: already {target['customer_type']!r}, "
                    f"mapping says {customer_type!r} -- left alone"
                )
            continue

        if apply:
            # RETURNING, not execute(): the row was read a moment ago and the
            # UPDATE is guarded, so another writer classifying or re-tenanting
            # it in between yields zero rows. Counting that as "applied" would
            # report a change the database refused -- the guards would still
            # protect the data, but the report would be a lie.
            changed = await pool.fetchrow(SQL_APPLY, contact_id, customer_type, EOM_CONTEXT)
            if changed is None:
                tally["conflict"] += 1
                problems.append(
                    f"{label}: nothing updated -- the row changed after it was read"
                )
                continue
            tally["applied"] += 1
        else:
            tally["would-apply"] += 1
        print(f"  {customer_type:<12} {contact_id}  {row['evidence']}")

    print("\nsummary:")
    for key in (
        "applied",
        "would-apply",
        "already-set",
        "conflict",
        "unknown-contact",
        "wrong-tenant",
        "rejected-value",
    ):
        if tally[key]:
            print(f"  {key}: {tally[key]}")

    if problems:
        print("\nneeds attention:")
        for problem in problems:
            print(f"  {problem}")

    totals = await pool.fetch(
        """
        SELECT customer_type, COUNT(*) AS n
          FROM contacts
         WHERE business_context_id = $1
         GROUP BY customer_type ORDER BY n DESC
        """,
        EOM_CONTEXT,
    )
    print("\nEOM contacts by customer_type:")
    for row in totals:
        print(f"  {row['customer_type']}: {row['n']}")

    # A mapping row that resolved to nothing is a mapping worth looking at, so
    # it is a non-zero exit even though nothing is broken in the database.
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", type=Path, help="CSV from the tracker query above")
    parser.add_argument(
        "--apply", action="store_true", help="write changes (default: dry run)"
    )
    args = parser.parse_args()
    return asyncio.run(run(mapping_path=args.mapping, apply=args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
