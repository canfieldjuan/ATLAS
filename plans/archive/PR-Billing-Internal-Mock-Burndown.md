# PR-Billing-Internal-Mock-Burndown

## Why this slice exists

Issue #1879 made first-party mocks visible through the maturity sweep. The
first high-severity debt burn-down target is `atlas_brain/api/billing.py`
because it is the money path and currently carries `INTERNAL_MOCK x62` in
`tests/maturity_sweep/baseline_atlas_brain_api.json`.

Root cause: billing route/webhook coverage leaned on hand-written pool fakes
that record SQL calls and return canned command tags. The earlier first pass
only moved those fakes from `monkeypatch(get_db_pool)` into direct function
arguments, which lowered the detector count without removing the fake. This
revision fixes the root for one representative checkout-paid path: it replaces
that fake with a live asyncpg/Postgres adapter and asserts persisted report,
delivery, and billing-event state. Existing fake tests remain detector-visible
until they are replaced by real adapter coverage in later slices.

This slice is slightly over the 400 LOC soft cap because the fix must include
the live Postgres test harness and workflow service enrollment in the same PR;
shipping only the DI seam or only the test would recreate the detector-evasion
gap the review flagged.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Add an explicit billing DB-pool dependency seam used by checkout, portal,
   status, and Stripe webhook routes.
2. Replace one fake-pool checkout-paid webhook test with a live asyncpg-backed
   integration test that asserts durable state, not SQL call arguments.
3. Add the pinned Postgres service/env to the billing Stripe-paid workflow so
   the live test runs in CI.
4. Refresh the atlas-brain API maturity baseline to the earned lower
   `billing.py` `INTERNAL_MOCK` floor.
5. Max files: 7.

### Review Contract

- Acceptance criteria:
  - Billing routes still call the same production `get_db_pool()` path through
    the new dependency.
  - Existing fake-pool tests stay visible to the ratchet; they are not
    relocated into an uncounted argument shape.
  - The new checkout-paid webhook integration test uses a live asyncpg pool and
    asserts persisted report/delivery/billing-event rows.
  - Stripe remains an external seam in tests; Checkout Session and webhook
    behavior is unchanged.
  - The maturity baseline for `atlas_brain/api/billing.py` decreases only by
    the fake test replaced with live coverage.
- Affected surfaces:
  - `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
  - `atlas_brain/api/billing.py`
  - focused billing tests
  - `tests/maturity_sweep/baseline_atlas_brain_api.json`
- Risk areas:
  - Stripe Checkout metadata/idempotency behavior.
  - Stripe webhook signature verification and idempotency.
  - FastAPI direct-call tests accidentally receiving a `Depends` sentinel
    instead of a pool.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R13, R14.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
- `atlas_brain/api/billing.py`
- `plans/PR-Billing-Internal-Mock-Burndown.md`
- `tests/maturity_sweep/baseline_atlas_brain_api.json`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

- Add a `_billing_db_pool()` dependency wrapper that returns the existing
  `get_db_pool()` result. Production still uses the same database adapter.
- Thread `pool=Depends(_billing_db_pool)` into the billing endpoints that read
  the pool at route time.
- Add `_resolve_billing_db_pool()` so direct function tests that omit the
  dependency parameter still resolve the same production getter. That keeps
  their existing fake-pool monkeypatches visible to the `INTERNAL_MOCK`
  detector until those tests are actually replaced.
- Replace the fake checkout-paid webhook route test with a live Postgres test
  that applies the required migrations, saves a report through the
  `PostgresDeflectionReportArtifactStore`, runs the signed webhook path, and
  asserts the report is paid, a delivery row exists, and the Stripe event is
  idempotently recorded.
- Add a pinned `postgres:16` service to the billing Stripe-paid workflow.
- Refresh only the relevant maturity baseline after the real test replacement.

## Intentional

- Stripe SDK remains faked in unit tests because it is the external transport
  seam; this slice is about first-party billing DB adapter fakes.
- Existing settings overrides and remaining fake-pool tests are left for later
  slices. The remaining fakes are intentionally still visible to the detector.
- No production SQL or Stripe behavior changes are intended.
- The Stripe planner MCP was checked but failed with a missing `mcp_session_id`;
  this plan uses the local Stripe billing/security references instead.

## Deferred

- Continue replacing the remaining fake-pool webhook tests with live
  asyncpg-backed state assertions in follow-up slices.
- Burn down billing settings/config `INTERNAL_MOCK` entries after the DB-pool
  fake class is drained.

Parked hardening: none.

## Verification

- Command: Python compile check for `atlas_brain/api/billing.py`,
  `tests/test_atlas_billing_stripe_hardening.py`,
  `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`, and
  `tests/test_atlas_billing_content_ops_deflection_paid_flow.py` - passed.
- Command: `python -m pytest tests/test_atlas_billing_stripe_hardening.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - 68 passed, 1 skipped, 1 existing torch/pynvml warning.
- Command: `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py::test_stripe_webhook_live_postgres_marks_deflection_report_paid_and_idempotent -q` - 1 passed, 1 existing torch/pynvml warning.
- Command: `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas_dev_password@localhost:5433/atlas python -m pytest tests/test_atlas_billing_stripe_hardening.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` - 69 passed, 1 existing torch/pynvml warning.
- Command: `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --top 80` - passed; `billing.py` is now `INTERNAL_MOCK x59`, score 262.
- Pending before push: `bash scripts/push_pr.sh <pr-body-file>`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml` | 28 |
| `atlas_brain/api/billing.py` | 41 |
| `plans/PR-Billing-Internal-Mock-Burndown.md` | 130 |
| `tests/maturity_sweep/baseline_atlas_brain_api.json` | 4 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 234 |
| **Total** | **437** |
