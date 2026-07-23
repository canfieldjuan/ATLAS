"""One-time backfill of NULL business_context_id contacts (issue #2151 Phase 2).

Classifies existing NULL-context contact rows by their write provenance:

- ``effingham_maids``: sources written only by EOM flows (booking, web,
  email_backfill, calendar_import), plus any contact linked from a
  tenant-stamped EOM appointment (appointments.business_context_id is
  NOT NULL by schema, so that linkage is trustworthy provenance).
- ``churnsignals``: sources written only by the B2B flows
  (briefing_gate, campaign_reply).
- Everything else stays NULL and is reported, never guessed.

Dry-run by default; pass ``--apply`` to write. Idempotent: only rows that
are still NULL are ever touched, so re-running after the writers were
stamped (same slice) converges to zero work.

Usage:
    python scripts/backfill_business_context.py            # report only
    python scripts/backfill_business_context.py --apply    # write
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EOM_CONTEXT = "effingham_maids"
B2B_CONTEXT = "churnsignals"

# Provenance maps: a source value appears in exactly one tenant's writers.
EOM_SOURCES = ("booking", "web", "email_backfill", "calendar_import")
B2B_SOURCES = ("briefing_gate", "campaign_reply")


def classify_source(source: str | None) -> str | None:
    """Pure classification used by both the SQL below and the tests."""
    if source in EOM_SOURCES:
        return EOM_CONTEXT
    if source in B2B_SOURCES:
        return B2B_CONTEXT
    return None


SQL_BACKFILL_EOM_BY_SOURCE = """
    UPDATE contacts
       SET business_context_id = $1, updated_at = NOW()
     WHERE business_context_id IS NULL
       AND source = ANY($2::text[])
"""

SQL_BACKFILL_EOM_BY_APPOINTMENT = """
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


async def run(apply: bool) -> int:
    from atlas_brain.storage.database import get_db_pool

    pool = get_db_pool()
    await pool.initialize()

    async def _count(sql: str, *params) -> int:
        # Reuse the UPDATE predicates as SELECT COUNTs for the dry run.
        counting = sql.replace(
            "UPDATE contacts c\n       SET business_context_id = $1, updated_at = NOW()",
            "SELECT COUNT(*) FROM contacts c",
        ).replace(
            "UPDATE contacts\n       SET business_context_id = $1, updated_at = NOW()",
            "SELECT COUNT(*) FROM contacts",
        )
        return await pool.fetchval(counting, *params) or 0

    plan = [
        ("EOM by source", SQL_BACKFILL_EOM_BY_SOURCE, (EOM_CONTEXT, list(EOM_SOURCES))),
        ("B2B by source", SQL_BACKFILL_EOM_BY_SOURCE, (B2B_CONTEXT, list(B2B_SOURCES))),
        ("EOM by appointment linkage", SQL_BACKFILL_EOM_BY_APPOINTMENT, (EOM_CONTEXT,)),
    ]

    print(f"mode: {'APPLY' if apply else 'DRY RUN'}")
    for label, sql, params in plan:
        n = await _count(sql, *params)
        if apply and n:
            result = await pool.execute(sql, *params)
            print(f"{label}: updated {result} (matched {n})")
        else:
            print(f"{label}: would update {n}")

    print("\nremaining NULL-context contacts by source:")
    for row in await pool.fetch(SQL_COUNT_NULL_BY_SOURCE):
        print(f"  {row['source']}: {row['n']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()
    return asyncio.run(run(apply=args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
