# PR-Content-Ops-FAQ-Search-Setup-Failure-Result

## Why this slice exists

PR-Content-Ops-FAQ-Search-Latency-Gates made the FAQ search concurrency smoke
usable as a latency gate, but local validation surfaced a diagnostics gap: if
the database host is unreachable, pool creation raises before the script writes
the requested `--output-result` JSON file.

The smoke is now part of the robust-testing path for FAQ search survivability.
Failed setup should produce a structured artifact just like failed search
isolation or latency checks, so test runners can archive and compare the failure
without scraping a traceback.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: robust testing.

1. Convert database pool-creation failure into a deterministic JSON summary.
2. Preserve the existing non-zero exit behavior for setup failures.
3. Keep successful search-run output unchanged except for the additive setup
   status field.
4. Add focused unit coverage for the setup-failure result path.
5. Enroll the focused FAQ search smoke tests in extracted pipeline CI.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Setup-Failure-Result.md`
- `.github/workflows/extracted_pipeline_checks.yml`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/smoke_content_ops_faq_search_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_concurrency.py`

## Mechanism

`run_smoke` already builds the run id and search cases before opening the
Postgres pool. This slice catches exceptions from `_create_pool`, builds the
same top-level summary shape with zero requests, zero latency, no isolation
items, and a `setup` object containing the failure type/message, then returns
exit code `1`.

The main function remains responsible for writing `--output-result`, so both
successful runs and setup failures use the same artifact path.

The focused test is also added to `scripts/run_extracted_pipeline_checks.sh`,
and the workflow path filter now includes the FAQ search smoke script and test
for both pull requests and pushes.

## Intentional

- This does not catch argument validation failures; invalid CLI usage should
  still fail fast before a run artifact exists.
- This only handles pool-creation setup failures. Migration, seed, cleanup, and
  search-phase diagnostics stay out of this slice unless they block the pool
  failure artifact from working.
- The error message is stored as text for diagnostics, but no database URL or
  environment values are copied into the result.

## Deferred

- Structured artifacts for migration, seed, cleanup, or unexpected top-level
  failures can follow if those phases become noisy in live smoke runs.
- Hosted HTTP-route concurrency remains deferred until the deployed URL/token is
  available.

## Verification

- `pytest tests/test_smoke_content_ops_faq_search_concurrency.py -q` passed with
  8 tests.
- Python compile check for the smoke script and focused test module passed.
- Pool-failure CLI proof with `postgresql://atlas:atlas@127.0.0.1:1/atlas`
  exited `1` as expected and wrote a runtime JSON result under `tmp/` with `ok=false`,
  `requests.total=0`, `setup.phase=pool_create`, and
  `setup.error.type=ConnectionRefusedError`.
- CI enrollment check: `scripts/run_extracted_pipeline_checks.sh` includes the
  focused FAQ search smoke test, and
  `.github/workflows/extracted_pipeline_checks.yml` includes the FAQ search smoke
  script and test in the `pull_request` and `push` path filters.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 90 |
| CI enrollment | 5 |
| Smoke script | 95 |
| Tests | 48 |
| **Total** | **238** |
