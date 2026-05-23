# PR-Content-Ops-Migration-Dry-Run-Skip-Summary

## Why this slice exists

The 1,000-row FAQ lifecycle database validation surfaced a misleading migration
dry-run result: after the extracted Content Ops migrations were already applied,
`run_extracted_content_pipeline_migrations.py --dry-run --json` still reported
28 `dry_run` entries under `applied` and 0 skipped entries. The actual
non-dry run correctly reported 0 applied and 28 skipped.

That is a source-level correctness issue in the migration runner. Dry-run should
not mutate the database, but it should still inspect the migration table when it
exists so operators can distinguish pending migrations from already-applied
ones.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-io-tests

1. Make dry-run read applied migration versions when the migration table exists.
2. Keep first-run dry-run behavior safe when the migration table does not exist:
   report all packaged migrations as pending without creating the table.
3. Add regression tests for dry-run skipped/applied reporting.
4. Remove the drained hardening item from `HARDENING.md`.

### Files touched

- `plans/PR-Content-Ops-Migration-Dry-Run-Skip-Summary.md`
- `extracted_content_pipeline/storage/migration_runner.py`
- `tests/test_extracted_content_pipeline_migration_runner.py`
- `HARDENING.md`

## Mechanism

`apply_content_pipeline_migrations(..., dry_run=True)` will continue to avoid
`_ensure_migration_table(...)`, migration SQL, and migration-record writes.
Instead of returning an empty applied-version set, it will call
`_read_applied_versions(..., missing_table_ok=True)`.

If the migration table exists, dry-run can populate `skipped` for already
applied versions and `applied` with `status="dry_run"` for pending versions. If
the migration table does not exist yet, an undefined-table error is treated as
an empty applied-version set so first-run dry-runs still report all migrations as
pending.

## Intentional

- Dry-run still does not create tables, run migration SQL, or insert migration
  records.
- The missing-table fallback is narrow: only undefined-table errors are treated
  as "no applied migrations yet"; other database errors still surface.
- No CLI output schema change is needed. Existing `applied`, `skipped`,
  `applied_count`, and `skipped_count` fields become accurate in dry-run mode.

## Deferred

- No new hardening is parked for this slice. The previous dry-run hardening item
  is drained by this PR.
- A containerized integration test against real Postgres remains deferred; the
  fixture tests lock the runner contract without requiring a local DB in CI.

## Verification

- Passed: `pytest tests/test_extracted_content_pipeline_migration_runner.py -q` (11 passed)
- Passed: local DB dry-run check after fix reported `applied_count=0`, `skipped_count=28`.
- Passed: `scripts/run_extracted_pipeline_checks.sh` (1836 passed, 1 skipped)
- Passed: `bash scripts/local_pr_review.sh --allow-dirty`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| Migration runner | ~25 |
| Tests | ~65 |
| Hardening cleanup | ~10 |
| **Total** | **~175** |
