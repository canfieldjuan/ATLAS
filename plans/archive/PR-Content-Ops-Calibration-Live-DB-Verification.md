# PR-Content-Ops-Calibration-Live-DB-Verification

## Why this slice exists

Issue #1497 says the Content Ops calibration library shipped through slices
#1494/#1495/#1496 with SQL-text assertions, fake asyncpg pools, TestClient
router fakes, and GitHub CI, but no local or CI path applied migration 335 to a
real Postgres database and exercised the constraints and repository behavior
that only Postgres can prove.

Root cause: the calibration-library persistence tests stopped at parser/fake
pool boundaries. That verified SQL intent and router mapping, but never
verified the live database contract: foreign keys, CHECK constraints, partial
unique indexes, cascade delete, asyncpg exception types, and repo round-trips.

This fixes the root for the migration/repository half of #1497 by adding a
real-Postgres verification test and enrolling it in CI with a disposable
Postgres service. It also runs locally against a disposable Docker Postgres, so
the issue's "not just GitHub CI" gap is closed for this slice.

## Scope (this PR)

Ownership lane: content-ops/calibration-live-verification
Slice phase: Robust testing

1. Add an env-gated real-Postgres test for the calibration-library chain:
   minimal `saas_accounts` dependency, migration 334, then migration 335.
2. Prove migration 335's live constraints:
   - invalid `label` is rejected;
   - blank `excerpt` and `reasoning` are rejected;
   - duplicate active `example_id` per account raises asyncpg
     `UniqueViolationError`;
   - archiving the active row allows the same `example_id` to be reused;
   - deleting the tenant account cascades calibration rows.
3. Prove repository round-trip behavior against the live database:
   `create_calibration_example`, `list_calibration_examples`,
   `update_calibration_example`, `archive_calibration_example`, and
   `list_calibration_example_records`.
4. Enroll the live DB test in the Content Ops review workflow with a Postgres
   service and in the extracted pipeline local runner.
5. Keep the existing fake-pool tests; this adds coverage for database behavior,
   not a replacement for lightweight unit tests.

### Review Contract

Acceptance criteria:
- The new test skips cleanly when `ATLAS_MIGRATION_TEST_DATABASE_URL` is unset.
- Against a real Postgres, 334 then 335 apply in order after the minimal
  `saas_accounts` dependency exists.
- The test exercises the named CHECK, partial unique, archive/reuse, and cascade
  behaviors using real SQL errors from Postgres.
- The repo functions operate against asyncpg directly and surface the expected
  active rows after create/update/archive.
- The CI workflow provides `ATLAS_MIGRATION_TEST_DATABASE_URL` through a
  disposable Postgres service, so the live test runs in CI instead of skipping.
- Existing fake-pool unit tests remain enrolled and unchanged.

Affected surfaces:
- Content Ops calibration-library persistence verification.
- Content Ops review workflow CI enrollment.
- Extracted pipeline local runner enrollment.

Risk areas:
- Accidentally depending on a full fresh Atlas migration chain, which is known
  to have unrelated fresh-apply debt.
- Test data leaking across runs in the shared CI service database.
- Over-broad CI path triggers that slow unrelated PRs.

- Reviewer rules triggered: R2, R10, R12, R14.

### Files touched

- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`
- `plans/PR-Content-Ops-Calibration-Live-DB-Verification.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_calibration_library_live_db.py`

## Mechanism

The new test mirrors the existing deflection migration apply pattern: it is
guarded by `ATLAS_MIGRATION_TEST_DATABASE_URL`, uses asyncpg directly, creates a
throwaway schema per test, applies only the dependency chain needed for this
feature, and drops the schema in `finally`.

The test creates a minimal `saas_accounts` table in that throwaway schema
instead of applying the full migration set, because the full fresh chain has
unrelated historical dependencies. It then applies migrations 334 and 335 in
order and probes the database contract with real inserts/updates/deletes plus
the repository's async functions.

The Content Ops review workflow gets a Postgres service and runs this test in
the same job as the existing calibration fake-pool and router tests. The local
extracted pipeline runner includes the new test so a full local runner with
`ATLAS_MIGRATION_TEST_DATABASE_URL` exercises the live DB path.

## Intentional

- This slice does not attempt a full repo migration-chain fresh apply. That is
  explicitly outside #1497's calibration-library dependency chain and remains
  blocked by unrelated historical migration debt.
- The live test creates only the minimal `saas_accounts` table needed to prove
  the 334 -> 335 chain and cascade behavior.
- Router reachability against a running app is deferred because the immediate
  unverified database contract can be closed with a smaller, deterministic DB
  slice first.

## Deferred

- Running-app calibration admin router smoke: auth + admin gate + tenant
  scoping against a live app remains a separate #1497 follow-up.
- Full `bash scripts/run_extracted_pipeline_checks.sh` with all runtime deps is
  run in this slice if the local environment permits; any unrelated preexisting
  dependency/runtime failure will be reported rather than fixed here.
- Full migration-chain fresh apply remains separate migration debt.

Parked hardening: none.

## Verification

- Py compile for the new live DB test - pass.
- Live DB test with no DSN configured - pass, skipped cleanly.
- Local Docker-backed Postgres run of the new live calibration DB test - pass,
  4 tests.
- Content Ops review workflow pytest subset with the same local Postgres DSN -
  pass, 189 tests.
- Workflow YAML parse for the edited Content Ops review workflow - pass.
- Full extracted pipeline runner with the local Postgres DSN - pass, 4704
  passed, 10 skipped, 1 torch/pynvml warning.
- Pending before push: local PR review bundle through `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_review_workflow_checks.yml` | 19 |
| `plans/PR-Content-Ops-Calibration-Live-DB-Verification.md` | 139 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_calibration_library_live_db.py` | 227 |
| **Total** | **386** |
