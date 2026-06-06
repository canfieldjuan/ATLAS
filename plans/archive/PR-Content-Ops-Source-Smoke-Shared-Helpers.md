# PR-Content-Ops-Source-Smoke-Shared-Helpers

## Why this slice exists

The review-source and CFPB Postgres smokes now exercise the same host-facing
path: source export, source-row ingestion, Postgres import, deterministic draft
generation, persisted draft fetch, and target-id verification. Their Postgres
readiness and persisted-draft checks were copied independently. That is the
kind of smoke-test drift that will slow the next source family, such as support
tickets from CRM or help desk exports.

This slice extracts the shared Postgres smoke logic before adding more source
adapters.

## Scope (this PR)

1. Add a shared helper module for source-row Postgres smoke checks.
2. Refactor the review-source and CFPB Postgres smoke scripts to call the
   shared helper while preserving source-specific export/import flow.
3. Add focused tests for the helper behavior.
4. Ensure the full extracted pipeline check runs both Postgres source smokes
   and the new helper tests.
5. Remove the merged support-ticket prompt-policy coordination row while
   claiming this slice.

### Files touched

- `scripts/content_ops_source_postgres_smoke_helpers.py`
- `scripts/smoke_content_ops_review_source_postgres.py`
- `scripts/smoke_content_ops_cfpb_source_postgres.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_source_postgres_smoke_helpers.py`
- `tests/test_smoke_content_ops_review_source_postgres.py`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Source-Smoke-Shared-Helpers.md`

## Mechanism

The helper module owns only source-agnostic Postgres smoke behavior:

```python
await generate_imported_target_drafts(...)
await schema_readiness_errors(...)
await fetch_saved_drafts(...)
generation_errors(...)
draft_errors(...)
saved_draft_target_errors(...)
```

Each smoke script still owns source-specific argument parsing, source export,
ingestion inspection, import, and final payload shape. The scripts pass plain
values into the helper instead of passing their full `argparse.Namespace`.

## Intentional

- This is a refactor plus test coverage update, not a runtime behavior change.
- The helper stays under `scripts/` because it supports host smoke commands,
  not the packaged product runtime.
- CFPB keeps its local dotenv/database-url bootstrap. Review-source continues
  to reuse the review-generation smoke bootstrap.
- The full extracted check gains the review-source Postgres smoke test because
  it was present but not included in the current check list.

## Deferred

- No new source family is added here. Support-ticket/CRM/call-transcript source
  adapters remain follow-up slices.
- No live Postgres smoke is added to CI; these tests keep using fake pools and
  deterministic generation seams.

## Verification

- `pytest tests/test_content_ops_source_postgres_smoke_helpers.py tests/test_smoke_content_ops_review_source_postgres.py tests/test_smoke_content_ops_cfpb_source_postgres.py -q` -> 20 passed.
- `python -m py_compile scripts/content_ops_source_postgres_smoke_helpers.py scripts/smoke_content_ops_review_source_postgres.py scripts/smoke_content_ops_cfpb_source_postgres.py tests/test_content_ops_source_postgres_smoke_helpers.py tests/test_smoke_content_ops_review_source_postgres.py tests/test_smoke_content_ops_cfpb_source_postgres.py` -> passed.
- `grep -nP '[^\x00-\x7F]' scripts/content_ops_source_postgres_smoke_helpers.py scripts/smoke_content_ops_review_source_postgres.py scripts/smoke_content_ops_cfpb_source_postgres.py tests/test_content_ops_source_postgres_smoke_helpers.py tests/test_smoke_content_ops_review_source_postgres.py tests/test_smoke_content_ops_cfpb_source_postgres.py` -> no matches.
- `rg -n "campaign_opportunities|b2b_campaigns|EXTRACTED_DATABASE_URL|appears to be weighing|TODO|FIXME|pass$" scripts/content_ops_source_postgres_smoke_helpers.py scripts/smoke_content_ops_review_source_postgres.py scripts/smoke_content_ops_cfpb_source_postgres.py tests/test_content_ops_source_postgres_smoke_helpers.py tests/test_smoke_content_ops_review_source_postgres.py tests/test_smoke_content_ops_cfpb_source_postgres.py` -> only existing table/env/forbidden-phrase fixtures and no TODO/FIXME/pass placeholders.
- `git diff --check` -> passed.
- `bash scripts/local_pr_review.sh` -> passed.
- `bash scripts/run_extracted_pipeline_checks.sh` -> 1433 passed, 1 existing torch/pynvml warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Shared helper | ~190 |
| Smoke script refactors | ~410 |
| Helper tests | ~160 |
| Runner + coordination + plan | ~110 |
| **Total** | **~870** |

The diff exceeds the soft 400 LOC budget because the refactor removes two
existing duplicate helper blocks while adding one shared helper and direct
helper regression coverage. Net production code shrinks, and splitting the
tests from the extraction would make review harder without reducing product
risk.
