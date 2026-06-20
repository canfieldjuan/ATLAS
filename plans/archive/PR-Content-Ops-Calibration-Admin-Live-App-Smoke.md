# PR-Content-Ops-Calibration-Admin-Live-App-Smoke

## Why this slice exists

Issue #1497 says the calibration-library persistence and admin router shipped
without proof against a real local machine/app surface. PR #1735 fixed the root
for the migration/repository half by applying migrations 334 -> 335 against
Postgres and round-tripping the repository. The remaining root gap is that the
admin router has still only been exercised as a direct router with fake auth and
fake pools; nothing proves the production-mounted Content Ops API route reaches
the calibration router with the real JWT/B2B plan gate, admin role check, tenant
scope, and Postgres pool wired together.

This change fixes that root for the running-app/router half by adding an
env-gated live ASGI smoke that imports the production aggregate API router,
mounts it under the same `/api/v1` prefix as `atlas_brain.main`, seeds real
SaaS account/user rows, signs real JWTs, and drives the calibration endpoints
over HTTP against the same Postgres service used by the live migration test.

## Scope (this PR)

Ownership lane: content-ops/calibration-live-verification
Slice phase: Functional validation

1. Add one live-app calibration admin router smoke test for issue #1497.
2. Enroll that test in the Content Ops review workflow and extracted pipeline
   check list so it remains continuous when a Postgres DSN is available.
3. Add the missing `saas_users.is_admin` migration required by the real auth
   dependency used by this route.

### Review Contract

Acceptance criteria: the smoke mounts production `atlas_brain.api.router` at
`/api/v1`, uses real JWT auth through `require_b2b_plan_or_api_key`, skips
without `ATLAS_MIGRATION_TEST_DATABASE_URL`, and proves 401 unauthenticated,
403 below-plan, 403 non-admin write, 201 admin create, 409 duplicate, tenant
isolation, and DB-backed delete/archive. Fresh migrations must include
`saas_users.is_admin`, and workflow filters must cover aggregate router/auth
paths that can break this proof.

Affected surfaces: Content Ops review workflow enrollment, extracted pipeline
aggregate enrollment, aggregate API/auth path filters, SaaS auth schema, and
calibration-library live verification tests only.

Risk areas: restore mutated global DB/auth settings, avoid fake auth/pools, and
clean seeded tenant rows even when assertions fail.

Reviewer rules triggered: R1, R2, R4, R9, R14.

### Files touched

- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`
- `atlas_brain/storage/migrations/338_saas_users_is_admin.sql`
- `plans/PR-Content-Ops-Calibration-Admin-Live-App-Smoke.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_calibration_library_live_app.py`

## Mechanism

The test points Atlas' DB settings at `ATLAS_MIGRATION_TEST_DATABASE_URL`,
initializes the real `DatabasePool`, applies migrations 076, 079, 338, 334, and
335, then seeds passing owner/member tenants, a second tenant, and a below-plan
tenant. It signs JWTs with `create_access_token` and sends HTTP requests through
`httpx.ASGITransport` to a FastAPI app containing the production aggregate API
router. Migration 338 fixes the auth-schema root cause surfaced by this smoke:
`require_auth` reads `saas_users.is_admin` on every authenticated request.

## Intentional

- The smoke mounts the production aggregate API router rather than importing
  `atlas_brain.main.app`. The main app lifespan starts unrelated services
  (ASR, discovery, reminders, MCP clients, voice, LLM warmups) that would make
  this verification slow and flaky. The router mount still exercises the real
  production route/dependency/DB wiring that #1497 asks about.
- The test uses the same workflow Postgres service as the migration live test
  and cleans up seeded account rows by ID. A separate temporary schema is not
  used because the production `DatabasePool` does not set `search_path` per
  pooled connection.

## Deferred

- Full `atlas_brain.main` lifespan smoke remains out of scope unless we add a
  dedicated startup profile that disables unrelated background services.

Parked hardening: none.

## Verification

- Passed: python -m py_compile tests/test_content_ops_calibration_library_live_app.py.
- Passed: no-asyncpg collection probe for tests/test_content_ops_calibration_library_live_app.py -- module import succeeds when `asyncpg` is blocked.
- Passed: env -u ATLAS_MIGRATION_TEST_DATABASE_URL python -m pytest tests/test_content_ops_calibration_library_live_app.py -q -- 1 skipped.
- Passed: python -m pytest tests/test_migrations_runner.py -q -- 4 passed.
- Passed: ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_content_ops_calibration_library_live_app.py -q -- 1 passed, 1 warning.
- Passed: Content Ops review workflow subset with the same `ATLAS_MIGRATION_TEST_DATABASE_URL` -- 190 passed, 1 warning.
- Passed: ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas bash scripts/run_extracted_pipeline_checks.sh -- 4707 passed, 10 skipped, 1 warning.
- Passed: bash scripts/push_pr.sh tmp/pr-body-content-ops-calibration-admin-live-app-smoke.md --force-with-lease origin HEAD:claude/pr-content-ops-calibration-admin-live-app-smoke -- local PR review passed and branch pushed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_review_workflow_checks.yml` | 9 |
| `atlas_brain/storage/migrations/338_saas_users_is_admin.sql` | 9 |
| `plans/PR-Content-Ops-Calibration-Admin-Live-App-Smoke.md` | 107 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_calibration_library_live_app.py` | 258 |
| **Total** | **384** |
