# PR-Deflection-Delta-Drain-DB-Scope-Test

## Why this slice exists

#1869 tracks the DB-backed integration coverage deferred from #1868. The unit
and fake-pool tests prove that the pending deflection delta delivery SQL carries
`account_id` and `current_request_id` predicates, but they do not prove the live
Postgres claim/update path leaves non-target queued rows untouched.

Root cause: the delivery CI lane did not provide a Postgres service, so the
first attempt at live drain coverage could only be skipped locally or fail in CI
at DB setup. This PR fixes the root by giving the delivery gate a DB service and
adding one `@pytest.mark.integration` test that executes the real delivery drain
against migrated tables.

## Scope (this PR)

Ownership lane: deflection/report-deltas
Slice phase: Robust testing

1. Add a Postgres service and migration-test DSN to the deflection delivery CI
   workflow so DB-backed delivery tests run instead of skipping.
2. Add one integration test proving `send_pending_deflection_delta_deliveries`
   with both `account_id` and `current_request_id` sends only the targeted
   pending delta row.
3. Assert the same-account/different-current row and other-account/same-current
   row remain pending and unsent.
4. Enroll the migration files the integration test applies in the delivery
   workflow path filters so schema drift reruns this coverage.

### Review Contract

Acceptance criteria:
- The delivery workflow must provision Postgres and expose
  `ATLAS_MIGRATION_TEST_DATABASE_URL`.
- The delivery workflow path filters must include the deflection report, report
  email, delta, and delta delivery migrations consumed by the integration test.
- The new integration test must execute against real Postgres tables, not a fake
  pool or string-only SQL assertion.
- The test must insert at least three pending rows: target account/current,
  same account/different current, and other account/same current.
- Running the scoped drain must send exactly one email and leave the two
  non-target rows pending with no sent timestamp/message id.
- Existing non-DB delivery tests remain enrolled in the same workflow.

Affected surfaces:
- Deflection delivery CI workflow.
- Deflection delta delivery integration coverage in
  `tests/test_atlas_content_ops_deflection_delivery.py`.

Risk areas:
- Pulling Postgres-only dependencies into the wrong extracted collection lane.
- Accidentally weakening the existing fast delivery workflow while adding the DB
  service.
- Test fixture rows drifting away from the real migration schema.

Reviewer rules triggered: R1, R2, R6, R8, R10, R14.
- R1 Requirements match.
- R2 Test evidence.
- R6 CI/workflow behavior.
- R8 Persistence/data lifecycle.
- R10 Gate/predicate behavior.
- R14 Codebase verification.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml`
- `plans/PR-Deflection-Delta-Drain-DB-Scope-Test.md`
- `tests/test_atlas_content_ops_deflection_delivery.py`

## Mechanism

- Mirror the repo's existing Postgres service pattern from the review/migration
  workflows in `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml`.
- Add the migrations used by the live fixture to both `pull_request` and `push`
  path filters for the delivery workflow.
- Add the integration test to the existing delivery test file so workflow
  enrollment stays local to the delivery gate.
- The test applies the deflection report/delta/delta-delivery migrations to the
  disposable DB, inserts paid current/baseline reports, persisted deltas, and
  three pending delivery rows, then calls the real
  `send_pending_deflection_delta_deliveries` function with account/current
  scope.
- The sender is an in-memory fake at the email boundary only; the DB read,
  claim, success update, and untouched-row assertions use real Postgres.

## Intentional

- No production delivery code changes are planned. #1870 already threads the
  scopes into the live drain; this slice proves the live Postgres behavior.
- The email provider stays fake. The risk in #1869 is queue selection and row
  state, not Resend transport.
- The test lives in the delivery workflow rather than the extracted umbrella so
  asyncpg/Postgres requirements stay in a DB-provisioned Atlas lane.

## Deferred

- None.

Parked hardening: none.

## Verification

- Passed locally:
  - python -m py_compile tests/test_atlas_content_ops_deflection_delivery.py
  - pytest tests/test_atlas_content_ops_deflection_delivery.py -k "delta_delivery_scope" -q
    - 1 skipped, 34 deselected when `ATLAS_MIGRATION_TEST_DATABASE_URL` was unset.
  - ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas pytest tests/test_atlas_content_ops_deflection_delivery.py -k "delta_delivery_scope" -q
    - 1 passed, 34 deselected.
  - ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas pytest tests/test_atlas_content_ops_deflection_delivery.py -q
    - 35 passed.
  - python scripts/sync_pr_plan.py plans/PR-Deflection-Delta-Drain-DB-Scope-Test.md --check

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml` | 26 |
| `plans/PR-Deflection-Delta-Drain-DB-Scope-Test.md` | 121 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 191 |
| **Total** | **338** |
