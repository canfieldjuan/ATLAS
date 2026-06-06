# PR-Extracted-Fastapi-Route-CI

## Why this slice exists

PR #866 enrolled Content Ops uploaded-file route tests in extracted pipeline CI,
but reviewer follow-up noted a broader gap: the dependency-light extracted CI
lane collects FastAPI route tests and skips them because the workflow does not
install FastAPI. That means route assertions can pass locally while the main
extracted gate only proves collection, not behavior.

This slice makes the extracted pipeline gate actually execute the route tests
that already live in the runner.

## Scope (this PR)

Ownership lane: extracted-pipeline/route-ci

1. Add the missing FastAPI multipart dependencies to the extracted pipeline
   workflow install step.
2. Add a small contract test so route-test CI dependencies and runner
   enrollment cannot drift apart silently.
3. Enroll that contract test in the extracted pipeline runner/path filters.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_pipeline_route_ci_contract.py`
- `plans/PR-Extracted-Fastapi-Route-CI.md`

## Mechanism

The workflow install command gains `fastapi` and `python-multipart`. FastAPI is
required by the route modules, and `python-multipart` is required for route
registration that uses `File(...)` and `Form(...)`.

The new contract test reads the workflow and extracted runner as text. It
asserts:

- the extracted workflow install step includes `fastapi`;
- the same install step includes `python-multipart`;
- the extracted runner includes the uploaded-file route smoke test;
- the extracted runner includes the control-surface route API test.

## Intentional

- No route code changes. The behavior already exists; the gap is CI
  dependency coverage.
- No broad `tests/**` workflow path glob. This repo intentionally enrolls
  extracted files explicitly, and this slice follows that pattern.
- The contract test is text-based rather than importing CI YAML machinery; the
  check is about a concrete workflow command and runner list.

## Deferred

- A broader workflow normalization/refactor remains out of scope. If more
  dependency lanes diverge, that should be a separate CI maintenance slice.
- Non-dry-run uploaded-file import against local Postgres remains the next
  backend validation slice after this CI gate is real.

## Verification

- Focused CI contract + route tests:
  - `python -m pytest tests/test_extracted_pipeline_route_ci_contract.py tests/test_smoke_content_ops_ingestion_file_route.py tests/test_extracted_content_control_surface_api.py -q`
  - `91 passed`
- Extracted pipeline runner passed for `scripts/run_extracted_pipeline_checks.sh`:
  - `1866 passed, 1 skipped, 1 warning`
- `bash scripts/local_pr_review.sh --allow-dirty`
  - Passed before commit; rerun after commit before push.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Workflow dependency/path enrollment | ~6 |
| Runner enrollment | ~1 |
| Contract test | ~35 |
| **Total** | ~122 |
