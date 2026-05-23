# Content Ops File Route Postgres 10,000-Row Run - 2026-05-23

## Summary

The uploaded-file Content Ops route passed a non-dry-run write against the
local Atlas Postgres database with 10,000 CFPB-derived support-ticket source
rows. This exercises the production-shaped file ingestion path at the current
route cap:

1. Read the uploaded JSONL source-row file.
2. Parse it through `/content-ops/ingestion/files/import`.
3. Apply public-dataset default fields for customer-style metadata.
4. Resolve a real Postgres import pool provider.
5. Persist normalized campaign opportunities under a scoped account.
6. Rerun with `--replace-existing` and confirm the scoped row count stays
   stable.

The command exited `0`, inserted 10,000 opportunities, reported zero warnings,
and wrote a compact result artifact.

## Source Data

- Source artifact: `tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl`
- Source rows: `10,000`
- Source format: `jsonl`
- File size: `21M`
- Route limit exercised: `max_rows=10000`
- Source type after normalization: `support_ticket=10000`

## Database

The run used Atlas local database settings derived from
`atlas_brain.storage.config.db_settings`:

- host: `localhost`
- port: `5433`
- database: `atlas`
- user: `atlas`
- password: not set

The DSN was derived at runtime and was not printed or checked in.

## Commands

Environment check:

```bash
python - <<'PY'
from atlas_brain.storage.config import db_settings
print("derived_dsn_present=", bool(db_settings.dsn))
print("host=", db_settings.host)
print("port=", db_settings.port)
print("database=", db_settings.database)
print("user=", db_settings.user)
print("password_set=", bool(db_settings.password))
PY
```

Source file check:

```bash
wc -l tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl
ls -lh tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl
```

Route write smoke:

```bash
/usr/bin/time -v python scripts/smoke_content_ops_ingestion_file_route.py \
  tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl \
  --source-format jsonl \
  --source cfpb-route-postgres-10000 \
  --min-source-rows 10000 \
  --default-field company_name=CFPB \
  --default-field vendor_name=CFPB \
  --default-field contact_email=cfpb-public-archive@example.invalid \
  --write \
  --account-id acct-file-route-postgres-10000-20260523 \
  --user-id user-file-route-postgres \
  --replace-existing \
  --output-result tmp/content_ops_file_route_postgres_smoke_20260523_10000.json \
  --json
```

Saved artifact syntax check:

```bash
python -m json.tool tmp/content_ops_file_route_postgres_smoke_20260523_10000.json
```

Scoped database count and sample check:

```bash
python - <<'PY'
import asyncio, json
from atlas_brain.storage.config import db_settings

ACCOUNT = "acct-file-route-postgres-10000-20260523"

async def main():
    import asyncpg
    pool = await asyncpg.create_pool(dsn=db_settings.dsn, min_size=1, max_size=1)
    try:
        count = await pool.fetchval(
            """
            SELECT count(*)
              FROM campaign_opportunities
             WHERE account_id = $1
               AND target_mode = $2
            """,
            ACCOUNT,
            "vendor_retention",
        )
        sample = await pool.fetchrow(
            """
            SELECT target_id,
                   company_name,
                   vendor_name,
                   contact_email,
                   raw_payload->>'source_type' AS source_type,
                   jsonb_array_length(evidence) AS evidence_count
              FROM campaign_opportunities
             WHERE account_id = $1
               AND target_mode = $2
             ORDER BY target_id
             LIMIT 1
            """,
            ACCOUNT,
            "vendor_retention",
        )
        print(json.dumps({"count": count, "sample": dict(sample)}, sort_keys=True))
    finally:
        await pool.close()

asyncio.run(main())
PY
```

Idempotency rerun:

```bash
/usr/bin/time -v python scripts/smoke_content_ops_ingestion_file_route.py \
  tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl \
  --source-format jsonl \
  --source cfpb-route-postgres-10000 \
  --min-source-rows 10000 \
  --default-field company_name=CFPB \
  --default-field vendor_name=CFPB \
  --default-field contact_email=cfpb-public-archive@example.invalid \
  --write \
  --account-id acct-file-route-postgres-10000-20260523 \
  --user-id user-file-route-postgres \
  --replace-existing \
  --output-result tmp/content_ops_file_route_postgres_smoke_20260523_10000_rerun.json \
  --json
```

## Result

The first route write smoke wrote:

```text
tmp/content_ops_file_route_postgres_smoke_20260523_10000.json
```

Compact route result:

```json
{
  "account_id": "acct-file-route-postgres-10000-20260523",
  "dry_run": false,
  "errors": [],
  "inserted": 10000,
  "missing_field_counts": {},
  "ok": true,
  "opportunity_count": 10000,
  "replace_existing": true,
  "source_type_counts": {
    "support_ticket": 10000
  },
  "status_code": 200,
  "target_id_count": 10000,
  "warning_count": 0
}
```

Timing:

```text
Elapsed wall time: 0:04.83
Maximum resident set size: 154556 KB
Exit status: 0
```

Database confirmation:

```json
{
  "count": 10000,
  "sample": {
    "company_name": "CFPB",
    "contact_email": "cfpb-public-archive@example.invalid",
    "evidence_count": 1,
    "source_type": "support_ticket",
    "target_id": "cfpb:3134827",
    "vendor_name": "EQUIFAX, INC."
  }
}
```

The sample `vendor_name` came from the CFPB source row, which correctly
overrode the fallback default. Public-dataset `company_name` and
`contact_email` defaults filled the campaign-required metadata fields.

Idempotency rerun:

```json
{
  "count_after_rerun": 10000
}
```

The rerun exited `0`, inserted 10,000 rows after the scoped
`--replace-existing` delete, and left the scoped row count at 10,000 rather
than accumulating duplicates.

## Issues Surfaced

No product issue surfaced.

One validation-query mistake happened during manual verification: the first
sample query selected a non-existent `source` column from
`campaign_opportunities`. The corrected query reads `source_type` from
`raw_payload`. This did not affect the route smoke, persisted rows, or result
artifact, so no `HARDENING.md` entry was added.

## Conclusion

The uploaded-file route write path is now validated at the current 10,000-row
cap against local Postgres. The route accepted the file, normalized all rows,
persisted all rows under the scoped account, wrote a compact artifact, and
reran idempotently with `--replace-existing`.
