# Content Ops FAQ DB Lifecycle 1,000-Row Run - 2026-05-23

## Summary

The FAQ lifecycle smoke passed against the local Atlas Postgres database with
1,000 CFPB-derived support-ticket source rows. The run exercised the full
database-backed lifecycle:

1. Load JSONL source rows.
2. Generate FAQ Markdown.
3. Persist the generated FAQ draft.
4. Export the draft.
5. Update the draft to `published`.
6. Export the reviewed row.

The command exited `0`, saved one FAQ row, exported one draft row, exported one
published row, and reported all output checks passing.

## Source Data

- Source artifact: `tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl`
- Source rows: `1,000`
- Source format: `jsonl`
- Input profile:

```json
{
  "missing_source_text_count": 0,
  "raw_row_count": 1000,
  "raw_row_count_source": "jsonl_lines",
  "skipped_row_count": 0,
  "status": "ok",
  "usable_source_count": 1000,
  "usable_source_ratio": 1.0,
  "warning_count": 0,
  "warning_sample": [],
  "warnings_by_code": {}
}
```

## Database

The repo did not contain a literal `DATABASE_URL` or `EXTRACTED_DATABASE_URL` in
`.env`, `.env.local`, or `.env.backup`. The run used the Atlas local database
settings derived from `atlas_brain.storage.config.db_settings`:

- host: `localhost`
- port: `5433`
- database: `atlas`
- user: `atlas`
- password: not set

The DSN was passed through `EXTRACTED_DATABASE_URL` at command runtime and was
not printed or checked in.

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

Migration dry-run:

```bash
EXTRACTED_DATABASE_URL="$(python - <<'PY'
from atlas_brain.storage.config import db_settings
print(db_settings.dsn)
PY
)" python scripts/run_extracted_content_pipeline_migrations.py --dry-run --json
```

Migration apply:

```bash
EXTRACTED_DATABASE_URL="$(python - <<'PY'
from atlas_brain.storage.config import db_settings
print(db_settings.dsn)
PY
)" python scripts/run_extracted_content_pipeline_migrations.py --json
```

Lifecycle smoke:

```bash
EXTRACTED_DATABASE_URL="$(python - <<'PY'
from atlas_brain.storage.config import db_settings
print(db_settings.dsn)
PY
)" /usr/bin/time -v python scripts/smoke_content_ops_faq_lifecycle.py \
  tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl \
  --source-format jsonl \
  --account-id acct-faq-db-scale-20260523 \
  --user-id user-faq-db-scale \
  --title 'CFPB 1,000 Row FAQ DB Lifecycle Smoke 2026-05-23' \
  --min-source-rows 1000 \
  --min-saved-faqs 1 \
  --export-limit 5 \
  --max-text-chars 1200 \
  --default-field company_name=CFPB \
  --default-field contact_email=cfpb-public-archive@example.invalid \
  --default-field vendor_name=CFPB \
  --output-result tmp/faq_lifecycle_smoke_20260523_scale1000.json \
  --json
```

Saved artifact syntax check:

```bash
python -m json.tool tmp/faq_lifecycle_smoke_20260523_scale1000.json
```

## Result

The lifecycle smoke wrote:

```text
tmp/faq_lifecycle_smoke_20260523_scale1000.json
```

Compact lifecycle summary:

```json
{
  "draft_export_count": 1,
  "error_count": 0,
  "errors": [],
  "generated_item_count": 8,
  "output_checks": {
    "condensed": true,
    "has_action_items": true,
    "uses_user_vocabulary": true
  },
  "review_status": "published",
  "reviewed_export_count": 1,
  "saved_faq_count": 1,
  "source_count": 1000,
  "source_format": "jsonl",
  "source_rows": 1000,
  "status": "ok",
  "ticket_source_count": 1000
}
```

Additional observed values:

- Saved FAQ IDs: one row, `34d2f7aa-119b-4b9c-9ae4-4464db1064aa`
- Generated FAQ items: `8`
- Ticket counts by item: `[633, 166, 111, 57, 8, 6, 6, 13]`
- Question sources: `source_policy=8`
- Draft export rows: `1`
- Reviewed export rows: `1`
- Errors: `[]`

Timing:

```text
Elapsed wall time: 0:01.23
Maximum resident set size: 40988 KB
Exit status: 0
```

## Issues Surfaced

### FAQLIFE1000-1 - Migration dry-run reports already-applied migrations as pending

After the local extracted migrations were already applied, the dry-run command
still reported:

```text
dry_run=true
applied_count=28
skipped_count=0
first_status=dry_run
```

The actual non-dry migration run reported:

```text
applied_count=0
skipped_count=28
```

Impact: this did not block the FAQ lifecycle run, but it can mislead operators
into thinking migrations are pending when the migration table already marks them
applied.

Resolution: parked in `HARDENING.md` because the lifecycle flow succeeded and
the fix is not required for this slice to function.

## Conclusion

The real local database lifecycle proof passed at 1,000 source rows. The file
scale smoke, hosted execute smoke, fake-pool lifecycle test, and local Postgres
lifecycle path now all cover the 1,000-row confidence threshold.
