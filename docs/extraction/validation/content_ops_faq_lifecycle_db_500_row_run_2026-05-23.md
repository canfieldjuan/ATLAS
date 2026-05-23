# Content Ops FAQ DB Lifecycle 500-Row Run - 2026-05-23

## Summary

The FAQ lifecycle smoke passed against the local Atlas Postgres database with
500 CFPB-derived support-ticket source rows. The run exercised the full
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

- Source seed artifact: `tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl`
- 500-row source artifact: `tmp/faq_lifecycle_500_validation_20260523/cfpb_500_source_rows.jsonl`
- Source rows: `500`
- Source format: `jsonl`
- Source fixture command:

```bash
head -n 500 tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl \
  > tmp/faq_lifecycle_500_validation_20260523/cfpb_500_source_rows.jsonl
```

## Database

The run used the Atlas local database settings derived from
`atlas_brain.storage.config.db_settings`. The DSN was passed through
`EXTRACTED_DATABASE_URL` at command runtime and was not printed or checked in.

## Command

```bash
EXTRACTED_DATABASE_URL="$(python - <<'PY'
from atlas_brain.storage.config import db_settings
print(db_settings.dsn)
PY
)" /usr/bin/time -v -o tmp/faq_lifecycle_500_validation_20260523/time.txt \
  python scripts/smoke_content_ops_faq_lifecycle.py \
  tmp/faq_lifecycle_500_validation_20260523/cfpb_500_source_rows.jsonl \
  --source-format jsonl \
  --account-id acct-faq-db-500-validation-20260523 \
  --user-id user-faq-db-500-validation \
  --title 'CFPB 500 Row FAQ DB Lifecycle Validation 2026-05-23' \
  --min-source-rows 500 \
  --min-saved-faqs 1 \
  --export-limit 5 \
  --max-text-chars 1200 \
  --default-field company_name=CFPB \
  --default-field contact_email=cfpb-public-archive@example.invalid \
  --default-field vendor_name=CFPB \
  --output-result tmp/faq_lifecycle_500_validation_20260523/result.json \
  --summary-json > tmp/faq_lifecycle_500_validation_20260523/stdout_summary.json
```

Artifact syntax checks:

```bash
python -m json.tool tmp/faq_lifecycle_500_validation_20260523/stdout_summary.json
python -m json.tool tmp/faq_lifecycle_500_validation_20260523/result.json
```

## Result

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
  "source_count": 500,
  "source_format": "jsonl",
  "source_rows": 500,
  "status": "ok",
  "ticket_source_count": 500
}
```

Additional observed values:

- `stdout_summary.json` exactly matched `result.json.lifecycle_summary`.
- Full result artifact retained `reviewed_export`.
- Normalization warning count: `0`.
- Question sources: `source_policy=8`.
- Ticket counts by item: `[322, 87, 50, 26, 4, 2, 2, 7]`.
- Result artifact size: `182K`.
- Summary stdout artifact size: `818B`.

Timing:

```text
Elapsed wall time: 0:00.64
Maximum resident set size: 36144 KB
Exit status: 0
```

## Issues Surfaced

No new issues surfaced in the 500-row lifecycle run. The compact stdout path
kept operator output small while preserving the full lifecycle payload for
audit.

## Conclusion

The real local database lifecycle proof passed at 500 source rows. Together with
the 1,000-row lifecycle proof, this covers the common upload-size confidence
range requested for the FAQ product wedge.
