# Content Ops Ingestion Diagnostics

## Why this slice exists

AI Content Ops can load ready-made opportunity files and source-row exports,
but hosts currently need to run conversion, import, or generation before they
get a structured answer about whether an export is usable. The next ingestion
slice should make customer data readiness inspectable before database writes or
LLM calls.

This intentionally ships as one slice even though the diff is over the soft
budget: the diagnostics module, CLI, manifest coverage, tests, and docs are one
host-facing seam. Splitting the CLI from the module would leave no usable host
entry point; splitting docs or runner wiring would make the new command harder
to discover or easy to miss in extracted checks.

## Scope (this PR)

1. Add a reusable ingestion diagnostics module over the existing opportunity
   and source-row loaders.
2. Add a thin CLI that reports readiness for opportunity files or source-row
   files.
3. Add tests for report shape, source-type counts, warning grouping, sample
   limits, and CLI JSON output.
4. Wire the new test file into the extracted pipeline check runner.
5. Document the inspection command before conversion/import examples.
6. Remove merged #577 from coordination and claim this slice.

### Files touched

- `extracted_content_pipeline/ingestion_diagnostics.py`
- `scripts/inspect_extracted_content_ingestion.py`
- `tests/test_extracted_content_ingestion_diagnostics.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/manifest.json`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Ingestion-Diagnostics.md`

## Mechanism

`inspect_ingestion_file(...)` reuses existing loaders instead of creating a
parallel ingestion path. It builds a report with opportunity counts, warning
counts, missing required prompt fields, inferred source-type counts, and
bounded normalized samples.

The CLI prints JSON for automation or a concise text summary for operators.

## Intentional

- No new source aliases or source-type inference rules.
- No database writes.
- No LLM calls.
- No changes to opportunity normalization behavior.

## Deferred

- New adapters remain blocked on a real customer export fixture.
- DB-backed ingestion APIs are separate from this offline inspection slice.
- Rich UI rendering of ingestion diagnostics is a later operator-console task.

## Verification

- Command: python -m pytest -q tests/test_extracted_content_ingestion_diagnostics.py; result: 6 passed.
- Command: python -m py_compile extracted_content_pipeline/ingestion_diagnostics.py scripts/inspect_extracted_content_ingestion.py tests/test_extracted_content_ingestion_diagnostics.py; result: passed.
- Command: bash scripts/local_pr_review.sh; result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Diagnostics module | ~170 |
| CLI | ~95 |
| Tests | ~170 |
| Docs and runner | ~35 |
| Coordination and plan | ~90 |
| **Total** | ~560 |
