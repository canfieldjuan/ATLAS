# Content Ops File Route Concurrency Validation - 2026-05-23

## Summary

The uploaded-file route write path was pressure-tested against local Atlas
Postgres with concurrent CFPB-derived source-row uploads. Correctness held
through:

- 5 concurrent 10,000-row route writes;
- 20 concurrent 1,000-row route writes;
- 100 concurrent 1,000-row route writes.

At 150 concurrent 1,000-row route writes, the route surfaced the expected
database connection ceiling: 141 runs succeeded and 9 failed during pool
creation with `asyncpg.exceptions.TooManyConnectionsError`.

This is a resource/backpressure finding, not a parsing or persistence
correctness failure. Successful runs persisted the expected row counts under
their scoped accounts with zero ingestion warnings.

## Source Data

10,000-row cap fixture:

- `tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl`
- Rows: `10,000`

1,000-row fixture:

- `tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl`
- Rows: `1,000`

Both fixtures use source-row mode with these fallback fields:

```text
company_name=CFPB
vendor_name=CFPB
contact_email=cfpb-public-archive@example.invalid
```

## Database

The run used Atlas local database settings derived from
`atlas_brain.storage.config.db_settings`:

- host: `localhost`
- port: `5433`
- database: `atlas`
- user: `atlas`
- password: not set

Observed local Postgres connection settings:

```json
{
  "max_connections": "100",
  "superuser_reserved_connections": "3"
}
```

## Commands

Each process ran `scripts/smoke_content_ops_ingestion_file_route.py` with
`--write`, `--replace-existing`, a unique `account_id`, and an
`--output-result` path under:

```text
tmp/content_ops_file_route_concurrency_20260523/
```

Example shape:

```bash
python scripts/smoke_content_ops_ingestion_file_route.py \
  tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl \
  --source-format jsonl \
  --source cfpb-route-concurrency-100x1000 \
  --min-source-rows 1000 \
  --default-field company_name=CFPB \
  --default-field vendor_name=CFPB \
  --default-field contact_email=cfpb-public-archive@example.invalid \
  --write \
  --account-id acct-file-route-concurrency-100x1000-20260523-1 \
  --user-id user-file-route-concurrency \
  --replace-existing \
  --output-result tmp/content_ops_file_route_concurrency_20260523/100x1000/result_1.json \
  --json
```

## Results

| Probe | Source rows per run | Concurrent processes | Successes | Failures | Inserted rows | Warnings | Elapsed | Max RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5x10000 | 10,000 | 5 | 5 | 0 | 50,000 | 0 | 0:05.93 | 154,760 KB |
| 20x1000 | 1,000 | 20 | 20 | 0 | 20,000 | 0 | 0:01.69 | 65,868 KB |
| 100x1000 | 1,000 | 100 | 100 | 0 | 100,000 | 0 | 0:07.85 | 66,028 KB |
| 150x1000 | 1,000 | 150 | 141 | 9 | 141,000 | 0 | 0:12.49 | 66,280 KB |

Database row counts matched successful run counts:

```json
{"accounts": 5, "batch": "5x10000", "rows": 50000}
{"accounts": 20, "batch": "20x1000", "rows": 20000}
{"accounts": 100, "batch": "100x1000", "rows": 100000}
{"accounts": 141, "batch": "150x1000", "rows": 141000}
```

## Failure Shape

The 150-way probe produced 9 compact failure artifacts. Example:

```json
{
  "account_id": "acct-file-route-concurrency-150x1000-20260523-22",
  "detail": null,
  "diagnostics": null,
  "dry_run": false,
  "errors": [
    "TooManyConnectionsError: None"
  ],
  "import": null,
  "ok": false,
  "status_code": 500
}
```

The failure happens before route diagnostics/import work because the smoke must
create a database pool before wiring `opportunity_import_pool_provider`.
Successful runs in the same batch wrote all expected rows and warning counts
remained zero.

## Issues Surfaced

### FILECONCURRENCY-1 - Uploaded-file write pressure needs bounded DB concurrency

At 150 concurrent 1,000-row route-write processes, 9 processes failed during
pool creation with `TooManyConnectionsError`. The local database reports
`max_connections=100` and `superuser_reserved_connections=3`, so hosted
uploaded-file writes need admission control before this route is exposed to
unbounded concurrent customer uploads.

Resolution: parked in `HARDENING.md`. The single-run and moderate concurrent
flows work, and fixing hosted queue/backpressure is larger than a validation
record.

## Conclusion

The uploaded-file route write path is correct through the current 10,000-row
file cap and through 100 simultaneous 1,000-row writes on local Postgres. The
first observed ceiling is connection pressure at 150 simultaneous write
processes. That ceiling is now logged as hardening work for bounded hosted
write concurrency or background job admission control.
