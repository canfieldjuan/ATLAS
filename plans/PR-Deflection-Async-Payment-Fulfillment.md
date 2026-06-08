# PR-Deflection-Async-Payment-Fulfillment

## Why this slice exists

#1386 tracks a launch-gating paid-funnel gap: Stripe Checkout can complete for
delayed payment methods before funds are available, while ATLAS currently routes
only `checkout.session.completed`. For delayed methods, the completed event can
arrive with `payment_status != paid`, ATLAS skips unlock, and the later
`checkout.session.async_payment_succeeded` event is ignored. That leaves a buyer
who eventually paid with a locked report.

PR #1393 made ATLAS the checkout authorization source of truth, and
atlas-portfolio PR #298 made the portfolio call it before charge. This slice
closes the next money-path blocker: fulfillment must happen when Stripe says the
one-time checkout funds are available, including async success.

## Scope (this PR)

Ownership lane: deflection/go-live
Slice phase: Production hardening

1. Route `checkout.session.async_payment_succeeded` for
   `content_ops_deflection_report` sessions through the same paid-report
   validation, `mark_paid`, and delivery-queue path as a paid completed event.
2. Keep `checkout.session.completed` with `payment_status=paid` behavior
   unchanged.
3. Treat `checkout.session.completed` with non-paid status as pending, not a
   successful paid unlock event.
4. Add explicit observability for
   `checkout.session.async_payment_failed` deflection sessions without marking
   the report paid or queueing delivery.
5. Extend the deflection Stripe webhook tests to prove async success unlocks,
   completed-unpaid does not mark processed as a successful unlock, and async
   failure does not unlock.

### Review Contract

Acceptance criteria:

- Async success: a signed
  `checkout.session.async_payment_succeeded` webhook with
  `source=content_ops_deflection_report`, valid amount/currency, and paid
  `payment_status` updates `content_ops_deflection_reports`, queues delivery
  when an email exists, and records the async event type.
- Pending completed: a `checkout.session.completed` deflection webhook whose
  session is still unpaid returns 200 after inserting an observed-but-unfulfilled
  `billing_events` row, and does not update the report or queue delivery.
- Async failure: `checkout.session.async_payment_failed` logs the failed
  deflection payment context and does not update the report or queue delivery.
- Existing idempotency remains intact for fulfilled events: already processed
  event IDs skip before paid-update side effects.

Affected surfaces:

- Stripe webhook routing in `atlas_brain/api/billing.py`.
- Deflection paid-funnel webhook tests in
  `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`.

Risk areas:

- Do not change subscription/vendor billing behavior.
- Do not weaken amount/currency/account/request validation.
- Do not log Stripe secrets or customer ticket contents.

Reviewer rules triggered: R1 Requirements match; R2 Test evidence; R3 Security/auth; R5 Backward compatibility; R6 Error handling/fail-closed behavior; R8 Idempotency/payment state; R12 Deployment safety.

### Files touched

- `atlas_brain/api/billing.py`
- `plans/PR-Deflection-Async-Payment-Fulfillment.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

The webhook already verifies Stripe signatures, checks event idempotency before
side effects, and has a deflection-specific helper that validates session mode,
paid status, metadata, amount, and currency before `mark_paid`.

This slice keeps that helper as the single fulfillment path, but routes both
`checkout.session.completed` and `checkout.session.async_payment_succeeded` into
it for deflection sessions. Those routes remain valid only when Stripe's session
payload says `payment_status=paid`; otherwise they skip unlock, continue to the
normal `billing_events` audit insert, and return 200. A redelivered unpaid event
then dedupes by its own event ID, while a later async-success event has a
distinct ID and can still unlock.

`checkout.session.async_payment_failed` does not unlock anything. It logs the
session/account/request context at warning level and records the failed event in
`billing_events` so operators can distinguish a payment failure from an ignored
unrelated Stripe event.

## Intentional

- This PR does not add `payment_method_types=card`; the product should keep
  Stripe dynamic payment methods and fulfill when funds are available.
- This PR does not add delivery scheduler/URL reconciliation; it only ensures
  the delivery row is queued when async funds become available.
- This PR keeps the existing event-level idempotency model: paid side effects
  are followed by best-effort `billing_events` insertion, and duplicate
  processed event IDs skip before side effects.
- A non-paid async-success payload is treated like a non-paid completed payload:
  no unlock and no delivery, but the event is acknowledged and audited.

## Deferred

- #1386 delivery URL/scheduler reconciliation: prove paid report emails are
  sent automatically and link to the live result route.
- #1386 paid-funnel incident observability: promote amount/currency/report
  mismatch and async failure logs into the production alert sink.
- #1386 variant-aware authorization: re-enable partner checkout once ATLAS can
  authorize per-variant terms.

Parked hardening: none.

## Verification

- pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py
  - passed: 30 tests.
- pytest tests/test_atlas_billing_content_ops_deflection_paid_flow.py
  - passed: 1 test.
- bash scripts/local_pr_review.sh
  - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/billing.py` | 56 |
| `plans/PR-Deflection-Async-Payment-Fulfillment.md` | 131 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 223 |
| **Total** | **410** |
