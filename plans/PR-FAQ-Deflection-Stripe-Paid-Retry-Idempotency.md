# PR-FAQ-Deflection-Stripe-Paid-Retry-Idempotency

## Why this slice exists

PR-FAQ-Deflection-Stripe-Event-Log-Best-Effort made the final
`billing_events` insert best-effort after the Stripe paid-unlock side effect.
That is the right secondary-write behavior, but the regression coverage only
proved the first webhook response stays `ok` when the audit insert fails.

This slice locks down the retry/idempotency contract around that behavior: a
successful audit insert must make duplicate Stripe events short-circuit, while a
retry after an audit-insert failure may re-run the paid update but must converge
by inserting the idempotency row on the successful retry.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Robust testing

1. Make the deflection Stripe paid webhook test double model processed
   `billing_events` ids instead of always returning no match.
2. Add duplicate-event coverage proving an already-recorded Stripe event returns
   `{"status": "already_processed"}` before any paid update.
3. Add retry-after-audit-failure coverage proving the paid update is safe to
   repeat once and the next successful audit insert restores the idempotency
   guard.

### Files touched

- `plans/PR-FAQ-Deflection-Stripe-Paid-Retry-Idempotency.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

The existing `_Pool` test double grows a `processed_event_ids` set. Its
`fetchval` method returns a sentinel id for `billing_events` lookups when the
event id is already recorded, and its `execute` method records the event id
when the `INSERT INTO billing_events` succeeds.

The new tests exercise `stripe_webhook` through the same fake Stripe module and
request shape as the existing paid-unlock tests, so the assertions cover the
real idempotency branch instead of a helper-level shortcut.

## Intentional

- No production code changes land here. The current webhook behavior already
  does the right thing: report unlock is idempotent, and the audit row is the
  duplicate-event guard when available.
- This slice stays in the host API billing test suite. It must not be enrolled
  in extracted-checks because the test imports `atlas_brain.api.billing`.
- A retry after an audit insert failure is allowed to perform the paid update a
  second time. The update is keyed by `(account_id, request_id)` and writes the
  same Stripe session reference, so the important contract is convergence after
  the successful retry.

## Deferred

- Parked hardening: none.
- A separate reconciliation job for audit rows missing after a completed unlock
  remains outside this slice. The Stripe 2xx response means Stripe normally
  will not retry automatically after the best-effort audit failure.

## Verification

- Command: python -m py_compile tests/test_atlas_billing_content_ops_deflection_stripe_paid.py
  - Result: passed.
- Command: python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q
  - Result: 18 passed, 1 warning.
- Command: python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py -q
  - Result: 19 passed, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 78 |
| Tests | 119 |
| **Total** | **197** |
