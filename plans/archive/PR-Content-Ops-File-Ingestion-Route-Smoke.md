# PR-Content-Ops-File-Ingestion-Route-Smoke

## Why this slice exists

The server-side file ingestion routes landed so customer support-ticket exports
can bypass the old 1,000-row inline JSON cap. We still need a cheap,
repeatable proof that the production-shaped route contract accepts real
CFPB-style source-row files, applies mocked customer metadata through
`default_fields`, writes a machine-readable result, and fails closed when the
route cap is exceeded.

This slice adds that route-level smoke. It does not add a new product feature;
it gives builders and reviewers a direct command to exercise the real uploaded
file route before we keep layering UI and execution work on top.

This is slightly above the 400 LOC target because the slice needs a reusable
route smoke, deterministic success/failure tests, and the plan contract in one
PR. Splitting the tests from the smoke would leave the new validation command
itself unreviewed.

## Scope (this PR)

Ownership lane: content-ops/backend-file-ingestion-validation

1. Add a smoke script for `/content-ops/ingestion/files/import` that uses the
   FastAPI route endpoint directly with an upload-file shim.
2. Capture success and failure as compact JSON so large-upload validation has
   an artifact even when the route rejects an upload.
3. Cover CFPB-style source rows with missing campaign fields by requiring
   `default_fields` in the smoke path.
4. Add focused tests for success, failure artifact writing, and argument
   validation.
5. Enroll the smoke script and test in extracted pipeline CI so the validation
   command cannot rot outside the runner/path filters.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Content-Ops-File-Ingestion-Route-Smoke.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/smoke_content_ops_ingestion_file_route.py`
- `tests/test_smoke_content_ops_ingestion_file_route.py`

## Mechanism

The smoke script creates `create_content_ops_control_surface_router(prefix="/ops")`,
finds the `/ops/ingestion/files/import` route, wraps the local file bytes in a
minimal async upload object, and calls the route endpoint with
`source_rows=True`, `dry_run=True`, a requested `source_format`, and JSON form
`default_fields`.

The script writes a bounded result payload with:

- `ok`, `status_code`, and `errors`;
- the route, source path, source format, minimum row requirement, and elapsed
  time;
- diagnostics counts and import counts when the route succeeds;
- the route's HTTP detail when it fails.

## Intentional

- The smoke uses `dry_run=True`; it validates the uploaded-file route,
  source-row parsing, default-field form handling, row caps, and import result
  shape without writing duplicate campaign opportunities to the live DB.
- No UI code is touched. PR #865 owns the Intel UI file ingestion lane.
- No new parser abstraction is added. The route already centralizes parsing in
  `inspect_ingestion_file`; this smoke exercises that seam instead of copying
  parser logic.

## Deferred

- A non-dry-run API route smoke against local Postgres remains deferred until
  the UI upload path is stable; DB-backed FAQ lifecycle coverage already exists
  in the lifecycle smokes.
- Background jobs, durable upload storage, and cross-process admission control
  remain parked in `HARDENING.md` under the existing FAQ scale/concurrency
  entries.

## Verification

- `python -m pytest tests/test_smoke_content_ops_ingestion_file_route.py -q`
  - `3 passed in 2.00s`
- CFPB 10,000-row uploaded-file route smoke:
  - `python scripts/smoke_content_ops_ingestion_file_route.py tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl --source-format jsonl --source cfpb-route-smoke-10000 --min-source-rows 10000 --default-field company_name=CFPB --default-field vendor_name=CFPB --default-field contact_email=cfpb-public-archive@example.invalid --output-result tmp/content_ops_file_route_smoke_20260523_10000.json --json`
  - Exit `0`; `opportunity_count=10000`, `inserted=10000`,
    `warning_count=0`, `missing_field_counts={}`.
- `python -m pytest tests/test_smoke_content_ops_ingestion_file_route.py tests/test_extracted_content_control_surface_api.py -q`
  - `89 passed in 4.70s`
- Compile check passed for `scripts/smoke_content_ops_ingestion_file_route.py`.
- Extracted pipeline runner passed for `scripts/run_extracted_pipeline_checks.sh`.
  - `1864 passed, 1 skipped, 1 warning`
- `bash scripts/local_pr_review.sh --allow-dirty`
  - Pending rerun after CI enrollment fix.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~70 |
| CI enrollment | ~7 |
| Route smoke script | ~250 |
| Smoke tests | ~120 |
| **Total** | ~477 |
