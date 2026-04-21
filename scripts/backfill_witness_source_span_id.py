"""Backfill source_span_id on b2b_vendor_witnesses from enrichment evidence_spans.

One-time script. Safe to re-run -- only updates rows where source_span_id IS NULL.

The witness excerpt_text is taken from span["text"][:320] during witness assembly.
This script matches each witness back to its source span via that text equality,
then writes the span_id.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def main():
    import asyncpg

    dsn = os.environ.get(
        "ATLAS_DB_URL",
        "postgresql://atlas:atlas@localhost:5433/atlas",
    )
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4)

    # Count rows needing backfill
    total = await pool.fetchval(
        "SELECT COUNT(*) FROM b2b_vendor_witnesses WHERE source_span_id IS NULL"
    )
    print(f"Witnesses needing backfill: {total}")
    if total == 0:
        print("Nothing to do.")
        await pool.close()
        return

    # Batch update: match witness excerpt_text against enrichment evidence_spans[].text
    updated = await pool.execute(
        """
        UPDATE b2b_vendor_witnesses w
        SET source_span_id = matched.span_id
        FROM (
            SELECT DISTINCT ON (w2.vendor_name, w2.as_of_date, w2.analysis_window_days,
                                w2.schema_version, w2.witness_id)
                   w2.vendor_name, w2.as_of_date, w2.analysis_window_days,
                   w2.schema_version, w2.witness_id,
                   span->>'span_id' AS span_id
            FROM b2b_vendor_witnesses w2
            JOIN b2b_reviews r ON r.id = w2.review_id::uuid
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE WHEN jsonb_typeof(r.enrichment->'evidence_spans') = 'array'
                     THEN r.enrichment->'evidence_spans'
                     ELSE '[]'::jsonb
                END
            ) AS span
            WHERE w2.source_span_id IS NULL
              AND span->>'span_id' IS NOT NULL
              AND lower(trim(left(span->>'text', 320))) = lower(trim(w2.excerpt_text))
            ORDER BY w2.vendor_name, w2.as_of_date, w2.analysis_window_days,
                     w2.schema_version, w2.witness_id
        ) matched
        WHERE w.vendor_name = matched.vendor_name
          AND w.as_of_date = matched.as_of_date
          AND w.analysis_window_days = matched.analysis_window_days
          AND w.schema_version = matched.schema_version
          AND w.witness_id = matched.witness_id
        """
    )

    count = int(updated.split()[-1]) if updated else 0
    print(f"Updated: {count}/{total}")

    remaining = await pool.fetchval(
        "SELECT COUNT(*) FROM b2b_vendor_witnesses WHERE source_span_id IS NULL"
    )
    if remaining:
        print(f"Remaining unmatched: {remaining} (these had no matching span text)")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
