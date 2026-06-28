# PR-Real-Adapters-Deflection-Report-Fixtures

## Why this slice exists

Issue #1878 follows the real-adapters rule added in #1881. The root cause is
that several deflection report-access tests instantiate fake `_Pool` objects and
assert SQL strings or call tuples while claiming to verify
`PostgresDeflectionReportArtifactStore` behavior. That tests the fake transport,
not the adapter, and it is the same class of miss that let account/payment
scope defects hide behind over-mocked tests.

This PR fixes the root for the highest-risk report-access tests by replacing
fake-pool SQL assertions with a disposable Postgres database and observable
store state. It does not add the global `INTERNAL_MOCK` ratchet; #1879 owns
that follow-up.

This slice is over the 400 LOC target because the old fake-pool tests are being
removed and replaced with real Postgres setup, cleanup, and persisted-state
assertions in the same PR that enrolls the CI service. Splitting the workflow
enrollment away from the tests would leave the new live-adapter proof
self-skipping in CI.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Workflow/process

1. Migrate the fake-pool deflection report-access tests that assert SQL/call
   shape into live Postgres tests that assert persisted behavior.
2. Enroll the live tests in the existing deflection report CI workflow by
   provisioning the same disposable Postgres service pattern used by the
   migration and delivery checks.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_report_checks.yml`
- `plans/PR-Real-Adapters-Deflection-Report-Fixtures.md`
- `tests/test_content_ops_deflection_report.py`

### Review Contract

Acceptance criteria:
- `PostgresDeflectionReportArtifactStore` paid-gate, list, delete, and
  retention behavior is exercised against a real Postgres connection, not a
  fake pool.
- The CI job that names those tests provisions Postgres and sets
  `ATLAS_MIGRATION_TEST_DATABASE_URL`, so the live-adapter proof is
  load-bearing in CI.
- Remaining fake or monkeypatched tests in this slice are limited to true
  external seams or deliberately malformed transport responses that cannot be
  produced by Postgres.

Affected surfaces:
- `tests/test_content_ops_deflection_report.py`
- `.github/workflows/atlas_content_ops_deflection_report_checks.yml`

Risk areas:
- CI runtime from adding the Postgres service to the report-check workflow.
- Cross-test database isolation, because the disposable database is shared by
  tests in the workflow.

Reviewer rules: R1, R2, R9, R13, R14.

## Mechanism

- Add test helpers that connect to `ATLAS_MIGRATION_TEST_DATABASE_URL`, apply
  the narrow deflection report table migrations, and clean only per-test
  account IDs before/after each live test.
- Replace fake `_Pool` assertions for paid/unpaid round-trip, list filtering,
  delete scoping, and retention purge with calls through the real
  `PostgresDeflectionReportArtifactStore`.
- Add the Postgres service/env/`asyncpg` install to
  `.github/workflows/atlas_content_ops_deflection_report_checks.yml` so the
  named tests do not silently skip in the job that advertises report coverage.

## Intentional

- This PR does not migrate every fake pool in the repo. It starts with
  `tests/test_content_ops_deflection_report.py`, which #1878 names and which
  has the highest local blast radius for deflection report access behavior.
- The synthetic unparseable-delete-count test may keep a tiny fake transport
  because it exercises a defensive adapter branch that a real Postgres
  connection should not produce. It is not used to prove SQL scope or persisted
  state.

## Deferred

- #1879: add the maturity-sweep `INTERNAL_MOCK` ratchet so new first-party
  mocks fail blocking CI unless intentionally baseline-accepted.
- Remaining fake-pool tests in billing/delivery/delta modules should migrate
  when their lane is touched or when #1879 starts burning down the baseline.

Parked hardening: none.

## Verification

- Py-compiled `tests/test_content_ops_deflection_report.py` - passed.
- Targeted live-adapter pytest nodes in `tests/test_content_ops_deflection_report.py`
  without `ATLAS_MIGRATION_TEST_DATABASE_URL` - 4 skipped as DB-gated.
- Targeted live-adapter pytest nodes in `tests/test_content_ops_deflection_report.py`
  with `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas` - 4 passed.
- Full `tests/test_content_ops_deflection_report.py` with the same local DB URL -
  174 passed.
- `scripts/local_pr_review.sh --allow-dirty` - passed.
- Pending before push: blocking local review via `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_report_checks.yml` | 23 |
| `plans/PR-Real-Adapters-Deflection-Report-Fixtures.md` | 114 |
| `tests/test_content_ops_deflection_report.py` | 580 |
| **Total** | **717** |
