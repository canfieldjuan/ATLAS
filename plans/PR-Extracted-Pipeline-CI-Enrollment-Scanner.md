# PR-Extracted-Pipeline-CI-Enrollment-Scanner

## Why this slice exists

`atlas-recurring-build-defects.md` identifies CI enrollment as the most common
first-push defect in the Content Ops batch: new extracted/content-ops test files
can be created without being added to both the explicit extracted runner and
the workflow path filters. That makes a PR look green while its new test never
runs in the extracted CI lane.

## Scope (this PR)

Ownership lane: workflow/extracted-pipeline-ci

Slice phase: Workflow/process

1. Expand the existing extracted pipeline route CI contract test into a
   scanner for curated extracted/content-ops test-file patterns.
2. Assert every matching test file is enrolled in
   `scripts/run_extracted_pipeline_checks.sh`.
3. Assert every matching test file is covered by
   `.github/workflows/extracted_pipeline_checks.yml` pull_request and push path
   filters.
4. Fail loudly if the scanner matches zero files.
5. Add the currently missing workflow path-filter entries found by the scanner.
6. Enroll the deterministic extracted/content-ops tests the scanner exposed as
   untracked.
7. Fix the surfaced `faq_markdown` reasoning-policy drift instead of excluding
   its test.
8. Keep the enrolled Atlas import-admission test dependency-light by skipping it
   only when `asyncpg` is unavailable.
9. Apply the same explicit `asyncpg` guard to the enrolled Atlas generated-asset
   API mount test, which imports the host API package and reaches Atlas storage.

### Files touched

- `plans/PR-Extracted-Pipeline-CI-Enrollment-Scanner.md`
- `.github/workflows/extracted_pipeline_checks.yml`
- `scripts/run_extracted_pipeline_checks.sh`
- `extracted_content_pipeline/reasoning_policy.py`
- `tests/test_extracted_pipeline_route_ci_contract.py`
- `tests/test_extracted_content_reasoning_policy.py`
- `tests/test_atlas_content_ops_import_admission.py`
- `tests/test_atlas_content_ops_generated_assets_api.py`

## Mechanism

The test uses Python `Path.glob` and `fnmatch` against a curated pattern list
instead of changing the runner to blanket-glob tests. The explicit runner list
stays curated, but every matched file must be enrolled. There is no exclusion
escape hatch in this slice.

Workflow parsing stays local to the test: read the YAML as text, collect quoted
path-filter entries, and assert matching files are covered by both
`pull_request` and `push` filters. Runner parsing checks for the literal test
path in the shell script.

The scanner exposed an existing deterministic failure:
`faq_markdown` existed in `OUTPUT_CATALOG` without an
`OUTPUT_REASONING_POLICIES` entry. This slice fixes that as an explicit
`none`-only policy, matching the catalog's `reasoning_requirement="absent"`.

## Intentional

- No production code changes.
- No switch from explicit runner entries to glob execution.
- No PyYAML dependency; text parsing is enough for this workflow's simple path
  list shape and keeps the guard in the dependency-light lane.
- The surfaced `faq_markdown` reasoning drift is fixed inline because the
  scanner would otherwise formalize hiding a real existing failure.
- The Atlas import-admission test stays enrolled. It has an explicit
  `asyncpg` guard because its import chain reaches Atlas storage, and
  dependency-light CI does not install that database driver.
- The Atlas generated-assets API mount test also stays enrolled with an
  explicit `asyncpg` guard. A no-`asyncpg` simulation showed the sibling
  `atlas_content_ops_reasoning` and `atlas_content_ops_scope` tests still run
  without the database driver, so they intentionally remain unguarded to keep
  their dependency-light coverage.

## Deferred

- Future PR: extend the same scanner pattern to other workflow-maintained test
  lanes if they show the same recurring enrollment drift.
- Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_pipeline_route_ci_contract.py -q`
  - passed, 3 tests.
- `python -m pytest tests/test_extracted_pipeline_route_ci_contract.py tests/test_extracted_content_reasoning_policy.py -q`
  - passed, 21 tests.
- `python -m pytest tests/test_atlas_content_ops_import_admission.py -q`
  - passed, 7 tests.
- `python -m pytest tests/test_extracted_pipeline_route_ci_contract.py tests/test_atlas_content_ops_import_admission.py -q`
  - passed, 10 tests.
- `python -m pytest tests/test_atlas_content_ops_generated_assets_api.py -q`
  - passed, 4 tests.
- Simulated dependency-light lane with `asyncpg` blocked for the four enrolled
  `atlas_content_ops_*` host test files.
  - passed, 36 tests; skipped, 2 guarded asyncpg-bound files.
- Newly enrolled focused batch:
  `tests/test_extracted_ticket_faq_search.py`,
  `tests/test_extracted_ticket_faq_search_api.py`,
  `tests/test_extracted_campaign_port_taxonomy.py`,
  `tests/test_atlas_content_ops_generated_assets_api.py`,
  `tests/test_atlas_content_ops_import_admission.py`,
  `tests/test_atlas_content_ops_reasoning.py`,
  `tests/test_atlas_content_ops_scope.py`,
  `tests/test_extracted_blog_blueprint_postgres.py`,
  `tests/test_extracted_campaign_reasoning_context_api.py`,
  `tests/test_extracted_ticket_faq_postgres.py`,
  `tests/test_extracted_ticket_faq_search_postgres.py`,
  `tests/test_smoke_content_ops_faq_lifecycle_run.py`,
  `tests/test_smoke_content_ops_review_source_generation.py`
  - passed, 102 tests, 1 skipped, 1 existing environment warning from
    `torch`/`pynvml`.
- Python compile check for `tests/test_extracted_pipeline_route_ci_contract.py`
  passed.
- Extracted pipeline runner `scripts/run_extracted_pipeline_checks.sh` passed,
  including 2133 pytest cases, 2 skipped, and 1 existing environment warning
  from `torch`/`pynvml`.
- `scripts/local_pr_review.sh --allow-dirty`
  - passed.
- Pending clean run after commit: `scripts/local_pr_review.sh`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~65 |
| Workflow path filters | ~60 |
| Runner enrollment | ~20 |
| Reasoning policy drift fix | ~10 |
| Asyncpg-bound host test guards | ~10 |
| CI contract scanner | ~90 |
| **Total** | **~255** |
