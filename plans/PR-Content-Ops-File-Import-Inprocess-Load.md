# PR-Content-Ops-File-Import-Inprocess-Load

## Why this slice exists

PR-Content-Ops-File-Import-Admission-Gate added an in-process admission gate for
non-dry-run ingestion imports. The next proof should exercise the real route in
the hosted shape: many uploaded-file import calls inside one app process, using
one shared pool provider, where excess requests receive deterministic 429
responses instead of creating more database pools.

This closes the validation gap left by the older 150-process smoke. That smoke
is still useful for pressure discovery, but each process creates its own asyncpg
pool, so it cannot prove shared-pool admission behavior.

This slice is over the 400 LOC target because the runner needs to be reusable,
reviewable, and safe to run against a real database: it includes CLI validation,
shared-pool setup, async fan-out, compact failure artifacts, and deterministic
fake-pool tests. Splitting the tests from the runner would ship an unverified
load tool, which is not useful for this hardening lane.

## Scope (this PR)

Ownership lane: content-ops/backend-file-ingestion-validation

1. Add a same-process uploaded-file import load runner script.
2. Share one import pool across all route calls in the script.
3. Allow callers to configure total concurrency and route import admission
   concurrency.
4. Emit a compact JSON result that counts successes, 429 admission responses,
   unexpected failures, and inserted rows.
5. Add deterministic tests with a fake slow pool to prove the runner observes
   gate-full 429 responses and writes its artifact.

### Files touched

- `scripts/smoke_content_ops_ingestion_file_route_inprocess_load.py`
- `tests/test_smoke_content_ops_ingestion_file_route_inprocess_load.py`
- `plans/PR-Content-Ops-File-Import-Inprocess-Load.md`

## Mechanism

The new script builds the Content Ops control-surface router once with
`ContentOpsControlSurfaceApiConfig.ingestion_import_max_concurrency` set from
the CLI. It creates one asyncpg pool from the existing database URL resolution
helper and passes that same pool through `opportunity_import_pool_provider` for
every route call.

It then launches `N` async tasks against `/ops/ingestion/files/import` with the
same uploaded file bytes and `dry_run=False`. Per-call results are compacted to
status code, elapsed time, inserted count, and error reason. The overall result
is successful only when:

1. at least the configured minimum number of imports succeed,
2. at least the configured minimum number of calls return the expected 429
   admission response, and
3. no unexpected non-429 failures occur.

## Intentional

- This is a validation runner, not product code. It proves the route contract
  introduced by the previous slice.
- The runner does not add a queue or retry behavior. Queueing remains product
  hardening, not required to prove bounded admission.
- The default expected 429 count is conservative and caller-configurable because
  row count, database speed, and host scheduling affect how many concurrent
  calls collide while the gate is held.

## Deferred

- Cross-process or multi-worker admission remains a later production-hardening
  slice.
- Cleanup tooling for rows inserted during load validation is left out; callers
  should use a unique account id per run.
- Durable background import jobs remain deferred until after synchronous route
  behavior is fully characterized.

## Verification

- Focused runner tests:
  - `python -m pytest tests/test_smoke_content_ops_ingestion_file_route_inprocess_load.py -q`
  - `3 passed`
- Live shared-pool route load:
  - `python scripts/smoke_content_ops_ingestion_file_route_inprocess_load.py tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl --source-format jsonl --source cfpb-route-inprocess-load-20260523 --min-source-rows 1000 --default-field company_name=CFPB --default-field vendor_name=CFPB --default-field contact_email=cfpb-public-archive@example.invalid --account-id acct-file-route-inprocess-load-20260523 --user-id user-file-route-inprocess-load --concurrency 8 --import-max-concurrency 2 --min-successes 1 --expect-at-capacity-min 1 --output-result tmp/content_ops_file_route_inprocess_load_20260523.json --json`
  - Exit `0`; `2` successes, `6` admission 429 responses, `0`
    unexpected failures, `2000` rows inserted.
  - DB verification for `acct-file-route-inprocess-load-20260523` /
    `vendor_retention`: `2000` rows, `1000` distinct target IDs.
- Extracted route smoke/API tests:
  - `python -m pytest tests/test_smoke_content_ops_ingestion_file_route.py tests/test_smoke_content_ops_ingestion_file_route_inprocess_load.py tests/test_extracted_content_control_surface_api.py -q`
  - `97 passed`
- Local PR review:
  - `bash scripts/local_pr_review.sh --allow-dirty`
  - Pending.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 105 |
| Load runner script | 370 |
| Runner tests | 235 |
| Total | 710 |
