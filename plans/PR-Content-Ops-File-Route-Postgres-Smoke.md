# PR-Content-Ops-File-Route-Postgres-Smoke

## Why this slice exists

PR #866 added a dry-run uploaded-file route smoke, and PR #871 made route tests
execute in extracted CI. The remaining backend proof is the production-shaped
write path: uploaded source rows should flow through
`/content-ops/ingestion/files/import`, resolve a real import pool provider, and
persist normalized opportunities instead of only exercising `dry_run=True`.

This slice extends the existing route smoke with an explicit write mode so
reviewers can run the same command against local Postgres and get a compact
artifact on success or failure.

## Scope (this PR)

Ownership lane: content-ops/backend-file-ingestion-validation

1. Add opt-in non-dry-run support to the uploaded-file route smoke.
2. Require account scope and database URL only when write mode is selected.
3. Preserve the current dry-run default and compact result output.
4. Add focused tests for write-mode validation and pool-provider wiring.

### Files touched

- `scripts/smoke_content_ops_ingestion_file_route.py`
- `tests/test_smoke_content_ops_ingestion_file_route.py`
- `plans/PR-Content-Ops-File-Route-Postgres-Smoke.md`

## Mechanism

The script keeps `dry_run=True` by default. A new `--write` flag flips the route
call to `dry_run=False`, creates an asyncpg pool from `--database-url`,
`EXTRACTED_DATABASE_URL`, `DATABASE_URL`, or Atlas local DB settings, wires that
pool into `create_content_ops_control_surface_router` through
`opportunity_import_pool_provider`, and supplies a `TenantScope` from
`--account-id` / `--user-id`.

The response payload now includes `dry_run`, `account_id`, `user_id`,
`replace_existing`, and `opportunity_table` so the artifact tells reviewers
which execution mode and scope were used. Pool creation or route failures still
write the same compact failure payload when `--output-result` is provided.

## Intentional

- Write mode is explicit. The default remains side-effect-free so the smoke is
  safe in CI and for quick local checks.
- No DB migrations or table creation are added. The smoke validates the route
  against the existing `campaign_opportunities` import path.
- No UI code is touched. The UI upload lane is separate.

## Deferred

- A checked-in live Postgres validation record is deferred until this smoke is
  reviewed; this PR ships the reusable command and deterministic tests first.
- Background jobs, durable upload storage, and cross-process admission control
  remain parked in `HARDENING.md` under the existing FAQ scale/concurrency
  entries.

## Verification

- Focused route smoke/API tests:
  - `python -m pytest tests/test_smoke_content_ops_ingestion_file_route.py tests/test_extracted_content_control_surface_api.py -q`
  - `92 passed`
- Compile check passed for `scripts/smoke_content_ops_ingestion_file_route.py`.
- Live local Postgres uploaded-file route smoke:
  - `python scripts/smoke_content_ops_ingestion_file_route.py tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl --source-format jsonl --source cfpb-route-postgres-1000 --min-source-rows 1000 --default-field company_name=CFPB --default-field vendor_name=CFPB --default-field contact_email=cfpb-public-archive@example.invalid --write --account-id acct-file-route-postgres-20260523 --user-id user-file-route-postgres --replace-existing --output-result tmp/content_ops_file_route_postgres_smoke_20260523_1000.json --json`
  - Exit `0`; `dry_run=false`, `opportunity_count=1000`,
    `inserted=1000`, `warning_count=0`, `missing_field_counts={}`.
- Database count check:
  - `campaign_opportunities` rows for
    `acct-file-route-postgres-20260523` / `vendor_retention`: `1000`.
- Extracted pipeline runner passed for `scripts/run_extracted_pipeline_checks.sh`:
  - `1869 passed, 1 skipped, 1 warning`
- `bash scripts/local_pr_review.sh --allow-dirty`
  - Pending.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~85 |
| Smoke script | ~100 |
| Smoke tests | ~132 |
| **Total** | ~317 |
