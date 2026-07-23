"""One-time backfill of NULL business_context_id contacts (issue #2151 Phase 2).

Two evidence tiers:

1. **Appointment linkage (default, schema-trustworthy):** a contact linked
   from an appointment stamped ``effingham_maids`` is claimed for EOM —
   ``appointments.business_context_id`` is NOT NULL by schema, so this
   provenance cannot lie.
2. **Source maps (opt-in via ``--classify-by-source``):** ``contacts.source``
   is a free VARCHAR also settable through the MCP crm tool without a
   context, so source-only classification is operator-attested, not
   automatic. Without the flag those rows are only REPORTED as proposals.

Everything unclassified stays NULL and is reported, never guessed.
Dry-run by default; ``--apply`` writes. Idempotent: only rows still NULL are
ever touched.

Usage:
    python scripts/backfill_business_context.py                          # report
    python scripts/backfill_business_context.py --apply                  # tier 1 only
    python scripts/backfill_business_context.py --apply --classify-by-source
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EOM_CONTEXT = "effingham_maids"
B2B_CONTEXT = "churnsignals"

EOM_SOURCES = ("booking", "web", "email_backfill", "calendar_import")
B2B_SOURCES = ("briefing_gate", "campaign_reply")


def classify_source(source: str | None) -> str | None:
    """Pure source-map classification (tier 2; operator-attested)."""
    if source in EOM_SOURCES:
        return EOM_CONTEXT
    if source in B2B_SOURCES:
        return B2B_CONTEXT
    return None


# Count/update pairs share their WHERE clause verbatim; parameters match
# positionally in BOTH statements (no skipped placeholders).
SQL_COUNT_BY_SOURCE = """
    SELECT COUNT(*) FROM contacts
     WHERE business_context_id IS NULL
       AND source = ANY($1::text[])
"""
SQL_UPDATE_BY_SOURCE = """
    UPDATE contacts
       SET business_context_id = $2, updated_at = NOW()
     WHERE business_context_id IS NULL
       AND source = ANY($1::text[])
"""
SQL_COUNT_BY_APPOINTMENT = """
    SELECT COUNT(*) FROM contacts c
     WHERE c.business_context_id IS NULL
       AND EXISTS (
           SELECT 1 FROM appointments a
            WHERE a.contact_id = c.id
              AND a.business_context_id = $1
       )
"""
SQL_UPDATE_BY_APPOINTMENT = """
    UPDATE contacts c
       SET business_context_id = $1, updated_at = NOW()
     WHERE c.business_context_id IS NULL
       AND EXISTS (
           SELECT 1 FROM appointments a
            WHERE a.contact_id = c.id
              AND a.business_context_id = $1
       )
"""
SQL_COUNT_NULL_BY_SOURCE = """
    SELECT COALESCE(source, '(none)') AS source, COUNT(*) AS n
      FROM contacts
     WHERE business_context_id IS NULL
     GROUP BY 1 ORDER BY n DESC
"""


async def run(apply: bool, classify_by_source: bool) -> int:
    from atlas_brain.storage.database import get_db_pool

    pool = get_db_pool()
    await pool.initialize()

    # (label, enabled, count_sql, count_params, update_sql, update_params)
    steps = [
        ("tier 1: EOM by appointment linkage", True,
         SQL_COUNT_BY_APPOINTMENT, (EOM_CONTEXT,),
         SQL_UPDATE_BY_APPOINTMENT, (EOM_CONTEXT,)),
        ("tier 2: EOM by source (attested)", classify_by_source,
         SQL_COUNT_BY_SOURCE, (list(EOM_SOURCES),),
         SQL_UPDATE_BY_SOURCE, (list(EOM_SOURCES), EOM_CONTEXT)),
        ("tier 2: B2B by source (attested)", classify_by_source,
         SQL_COUNT_BY_SOURCE, (list(B2B_SOURCES),),
         SQL_UPDATE_BY_SOURCE, (list(B2B_SOURCES), B2B_CONTEXT)),
    ]

    print(f"mode: {'APPLY' if apply else 'DRY RUN'}"
          f"{' + classify-by-source' if classify_by_source else ''}")
    for label, enabled, count_sql, count_params, update_sql, update_params in steps:
        n = await pool.fetchval(count_sql, *count_params) or 0
        if not enabled:
            print(f"{label}: PROPOSED {n} (enable with --classify-by-source)")
            continue
        if apply and n:
            result = await pool.execute(update_sql, *update_params)
            print(f"{label}: {result} (matched {n})")
        else:
            print(f"{label}: would update {n}")

    print("\nremaining NULL-context contacts by source:")
    for row in await pool.fetch(SQL_COUNT_NULL_BY_SOURCE):
        print(f"  {row['source']}: {row['n']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write changes (default: dry run)")
    parser.add_argument("--classify-by-source", action="store_true",
                        help="also apply the operator-attested source maps (tier 2)")
    args = parser.parse_args()
    return asyncio.run(run(apply=args.apply, classify_by_source=args.classify_by_source))


if __name__ == "__main__":
    raise SystemExit(main())
