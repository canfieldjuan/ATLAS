# PR-Billing-Checkout-Completion-Leftovers-Live-Adapter

## Why this slice exists

The real-adapters/test-quality lane is burning down billing tests that still
prove money-path behavior with hand-written DB-pool fakes. #1902-#1908 covered
the main Checkout completion success, delivery, missing-report, reconciliation,
race-window, and lower-authorized-amount paths. The remaining Checkout
completion fake-pool tests are the endpoint for this sub-arc: invalid session
fail-closed, amount terms mismatch, missing-report incident emission, and wrong
authorized variant.

Root cause: these tests still rely on `_Pool` call history, `_Pool.add_report()`,
or fake row dictionaries. That proves the fake's internal bookkeeping, but it
does not prove persisted state for money-path failures and it weakens the
pre-adapter fail-closed boundary by using a fake DB object rather than proving
the DB is not touched.

This change fixes the root for the Checkout completion leftovers by:

1. using an explicit no-DB sentinel for invalid sessions, because those cases
   must return before any persistence adapter is touched;
2. using live asyncpg/Postgres state for terms mismatch, missing-report incident,
   and wrong-authorized-variant paths.

Diff budget note: this slice is over the 400 LOC soft cap because it deliberately
closes the Checkout-completion fake-test endpoint before the delta/subscription
section. Splitting it would leave the same handler boundary half-fake and add
another review/merge cycle for tightly coupled edge cases that share the same
helper setup and verification command.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Production hardening

1. Convert only the remaining Checkout-completion fake tests before the
   delta/subscription section:
   - `test_deflection_checkout_completion_fails_closed_for_invalid_sessions`
   - `test_deflection_checkout_completion_emits_incident_when_report_missing`
   - `test_deflection_checkout_completion_emits_incident_for_terms_mismatch`
   - `test_deflection_checkout_completion_rejects_wrong_authorized_variant`
2. Prove invalid sessions do not touch persistence at all.
3. Prove amount terms mismatch leaves the live report unpaid and queues no
   delivery.
4. Prove missing-report incident emission on the live missing-report path.
5. Prove wrong authorized variant leaves the live report unpaid, queues no
   delivery, and emits expected authorized-terms mismatch evidence.

### Review Contract

- Acceptance criteria:
  - Invalid-session coverage no longer uses `_Pool`; it uses a sentinel whose
    DB methods fail if touched.
  - Terms mismatch uses `_connect_live_billing_pool()` and asserts persisted
    unpaid/no-delivery state plus the incident payload.
  - Missing-report incident uses `_connect_live_billing_pool()` and asserts the
    409, incident payload, no report row, and no delivery row.
  - Wrong authorized variant uses live `record_checkout_authorization()` metadata
    and asserts persisted unpaid/no-delivery state plus the expected mismatch
    payload fields.
- Affected surfaces:
  - `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- Risk areas:
  - Billing/payment fail-closed behavior.
  - Checkout amount/currency gates.
  - Authorized checkout terms.
  - Paid-but-missing incident visibility.
  - Live test cleanup isolation.
- Reviewer rules:
  - R1 Requirements match.
  - R2 Test evidence.
  - R4 Data integrity / persistence.
  - R8 Error / edge cases.
  - R11 Money path.
  - R14 Codebase verification.

### Files touched

- `plans/PR-Billing-Checkout-Completion-Leftovers-Live-Adapter.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

The invalid-session test gets a tiny sentinel object with `is_initialized =
True` and DB methods that raise if called. That proves invalid mode/status,
missing/invalid IDs, bad amount, and bad currency return before the persistence
boundary.

The other three tests use the live billing pool, apply the same migrations as
the surrounding live Checkout tests, clean unique account/request rows, and
assert observable state:

- terms mismatch: seeded report remains unpaid, no delivery row, incident
  payload emitted;
- missing report: handler raises 409, no report/delivery rows, incident payload
  emitted;
- wrong authorized variant: seeded report has standard authorization metadata,
  lower paid session is globally allowed but does not match the report; report
  remains unpaid, no delivery row, expected mismatch payload emitted.

## Intentional

- Invalid sessions intentionally use a no-DB sentinel rather than a live pool:
  the correct behavior is to stop before the DB adapter, so a live adapter would
  be weaker than a fail-if-touched boundary probe.
- This groups the remaining Checkout-completion fake tests because they form a
  single endpoint before the delta/subscription section; the modest soft-cap
  overage is called out in Why this slice exists.
- This does not modify production billing code. The slice replaces fake proof
  with persisted-state or explicit no-DB proof for existing behavior.

## Deferred

- Delta subscription/entitlement fake-pool tests remain for the next sub-arc.
- Won-dispute/restore and remaining miss-incident fake-pool tests remain for
  follow-up slices.
- The temporary `_resolve_billing_db_pool` direct-call shim stays until the
  remaining fake-pool tests are drained.

Parked hardening: none.

## Verification

- py_compile for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
  - Pass.
- Focused live pytest for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
  with `-k 'invalid_sessions or terms_mismatch or report_missing or wrong_authorized_variant'`
  - Pass: 14 passed, 45 deselected.
- Full live pytest for `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
  - Pass: 59 passed.
- Maturity sweep: `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --top 10`
  - Pass/advisory: `billing.py` remains score 178 with `INTERNAL_MOCK x38`.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Billing-Checkout-Completion-Leftovers-Live-Adapter.md` | 141 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 319 |
| **Total** | **460** |
