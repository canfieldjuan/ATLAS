# PR-Content-Ops-File-Route-Postgres-10k-Validation

## Why this slice exists

PR #873 added an explicit `--write` mode to the uploaded-file route smoke and
proved a 1,000-row local Postgres write. The file route's production cap is
10,000 normalized rows, so the next confidence slice is to run the same real
route write at the cap and record the evidence.

This slice is validation only. It should not change product code unless the
10,000-row route write exposes a blocker that prevents the flow from working.

## Scope (this PR)

Ownership lane: content-ops/backend-file-ingestion-validation

1. Run the uploaded-file route smoke in `--write` mode against local Postgres
   with 10,000 CFPB-derived source rows.
2. Confirm the route response, saved result artifact, and database count agree.
3. Document the command, result, timing, and any surfaced issues.
4. Park non-blocking findings in `HARDENING.md` only if the run exposes them.

### Files touched

- `docs/extraction/validation/content_ops_file_route_postgres_10000_row_run_2026-05-23.md`
- `plans/PR-Content-Ops-File-Route-Postgres-10k-Validation.md`

## Mechanism

Use `scripts/smoke_content_ops_ingestion_file_route.py` with:

- `tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl`;
- `--write`;
- `--replace-existing`;
- a unique account id for this validation slice;
- mocked public-dataset defaults for `company_name`, `vendor_name`, and
  `contact_email`.

After the route smoke returns, query `campaign_opportunities` for the scoped
account and `vendor_retention` target mode. The doc records the compact result
summary and the database count without checking in temporary output artifacts
or database credentials.

## Intentional

- No code changes planned. This is a real-flow validation slice.
- The run uses a unique account id and `--replace-existing` so reruns are
  idempotent for the scoped validation rows.
- The local DSN is derived from Atlas settings at runtime and is not printed or
  checked in.

## Deferred

- Concurrent uploaded-file write pressure is deferred until the single-run
  10,000-row cap path is recorded.
- Background jobs, durable upload storage, and cross-process admission control
  remain separate hardening work if pressure testing shows they are needed.
- No `HARDENING.md` entry was added because no product issue surfaced.

## Verification

- Environment check: local Atlas DB settings derived from
  `atlas_brain.storage.config.db_settings`; host `localhost`, port `5433`,
  database `atlas`, user `atlas`, password not set.
- Source file check:
  - `10000 tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl`
  - File size: `21M`
- Uploaded-file route write smoke:
  - `python scripts/smoke_content_ops_ingestion_file_route.py tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl --source-format jsonl --source cfpb-route-postgres-10000 --min-source-rows 10000 --default-field company_name=CFPB --default-field vendor_name=CFPB --default-field contact_email=cfpb-public-archive@example.invalid --write --account-id acct-file-route-postgres-10000-20260523 --user-id user-file-route-postgres --replace-existing --output-result tmp/content_ops_file_route_postgres_smoke_20260523_10000.json --json`
  - Exit `0`; `dry_run=false`, `opportunity_count=10000`,
    `inserted=10000`, `warning_count=0`, `missing_field_counts={}`.
  - Elapsed `0:04.83`; maximum RSS `154556 KB`.
- Saved artifact syntax check passed for
  `tmp/content_ops_file_route_postgres_smoke_20260523_10000.json`.
- Database count/sample check:
  - Scoped `campaign_opportunities` count: `10000`.
  - Sample row retained `target_id`, default `company_name` and
    `contact_email`, source `vendor_name`, `source_type=support_ticket`, and
    one evidence item.
- Idempotency rerun:
  - Exit `0`; scoped count after rerun stayed `10000`.
- `bash scripts/local_pr_review.sh --allow-dirty`
  - Pending.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~91 |
| Validation record | ~242 |
| **Total** | ~333 |
