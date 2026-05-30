# PR-FAQ-Deflection-Stripe-Event-Log-Best-Effort

## Why this slice exists

The deflection/Stripe paid-unlock path now has a hosted smoke. The webhook marks
the deflection report paid before writing the generic `billing_events` audit row.
That audit insert is a secondary write after the side-effectful unlock; if it
fails, Stripe currently sees an error even though the report was already marked
paid.

This slice applies the repo's secondary-write rule to the deflection paid-unlock
path: audit logging failures are logged and do not fail an already-successful
unlock.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Production hardening

1. Wrap the post-webhook `billing_events` insert in best-effort error handling.
2. Preserve fail-closed behavior for the actual Stripe signature, event
   validation, report lookup, and paid update.
3. Add a regression proving a deflection checkout still returns `{"status":
   "ok"}` when the audit insert fails after the paid update succeeds.

### Files touched

- `plans/PR-FAQ-Deflection-Stripe-Event-Log-Best-Effort.md`
- `atlas_brain/api/billing.py`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

`stripe_webhook` keeps the existing idempotency check and event-specific
handlers unchanged. Only the final generic `billing_events` insert moves behind
a `try`/`except Exception` block with `logger.exception(...)`.

## Intentional

- This does not make paid updates best-effort. If the deflection report row is
  missing, the webhook still returns a non-2xx response for Stripe retry.
- This applies to the shared webhook audit insert after every successfully
  handled event, not just deflection events, because the insert is always
  secondary to the event-specific side effect.

## Deferred

- Parked hardening: none.
- Dedicated reconciliation for missing audit rows remains outside this slice;
  this slice only prevents secondary audit failures from reversing a completed
  webhook side effect.

## Verification

- py_compile for `atlas_brain/api/billing.py` and the deflection Stripe paid test file - passed.
- Focused pytest for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` - 16 passed, 1 warning.
- Path-scoped workflow pytest pair for deflection Stripe paid checks - 17 passed, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 66 |
| Billing route | 28 |
| Tests | 53 |
| **Total** | **148** |
