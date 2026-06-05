## Why this slice exists

The deflection funnel now persists a server-side `delivery_email` and has
privacy regressions proving it does not leak through customer payloads. The
next safe post-purchase step is not sending email yet; it is recording a
durable delivery request only after the signed Stripe webhook verifies payment
and ATLAS marks the report paid.

Without this queue, a future email worker would have to infer work from paid
rows after the fact. This slice creates the explicit handoff point while
preserving the webhook trust boundary.

## Scope (this PR)

Ownership lane: ai-content-ops/faq-deflection-paid-unlock

Slice phase: Production hardening

1. Add a deflection report delivery queue table keyed by account/request.
2. After a valid `checkout.session.completed` webhook marks the report paid,
   enqueue a pending delivery only when the report row has `delivery_email`.
3. Keep the queue row free of the email address; workers can read the address
   from the report row later.
4. Preserve checkout, paid unlock, browser return, and report artifact behavior.
5. Add focused webhook tests for queue creation, missing-email skip, and
   idempotent duplicate handling.

### Files touched

- `atlas_brain/api/billing.py`
- `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
- `atlas_brain/storage/migrations/332_content_ops_deflection_report_deliveries.sql`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- `tests/test_atlas_billing_content_ops_deflection_paid_flow.py`
- `plans/PR-Deflection-Paid-Delivery-Queue.md`

## Mechanism

The verified webhook path already calls
`PostgresDeflectionReportArtifactStore.mark_paid(...)` after validating
Checkout mode, paid status, metadata, amount, and currency. This slice keeps
that order and then looks up the report record through the same store. If
`delivery_email` is present, the handler inserts or refreshes a pending row in
`content_ops_deflection_report_deliveries`:

```text
signed Stripe webhook -> mark report paid -> enqueue pending delivery
```

The queue records `account_id`, `request_id`, `payment_reference`, status, and
timestamps. It deliberately does not duplicate `delivery_email`; the email
remains on the paid-gated report row added by the previous slice.

Duplicate Stripe retries remain idempotent because the webhook-level
`billing_events` check still exits before side effects for already processed
events, and the queue upsert is keyed by `(account_id, request_id)`.

The dedicated `atlas_content_ops_deflection_stripe_paid_checks.yml` workflow
already owns the Stripe-paid webhook tests. This slice enrolls the new
migration path in both PR and push filters, and the existing pytest run-step
already includes the changed Stripe-paid and paid-flow regression files.

## Intentional

- No email is sent in this slice. This is the durable handoff before transport.
- No abandoned/cancel/non-buyer nurture flow is added; that still needs consent
  and opt-out policy.
- Delivery queue insertion is downstream of signature-verified Stripe webhook
  handling, not the browser Checkout return.
- The queue does not store the email address to avoid duplicating buyer PII.
- If the queue insert fails after `mark_paid`, the webhook fails so Stripe can
  retry the delivery handoff; the paid report row remains unlocked.

## Deferred

- Future slice: delivery worker/template that sends the report link from
  pending queue rows.
- Future slice: delivery attempt status transitions and provider message IDs.
- Future slice: abandoned/checkout-cancel follow-up capture policy and opt-out
  rules before any non-buyer nurture email is sent.
- Parked hardening: none.

## Verification

- `python -m py_compile atlas_brain/api/billing.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py` -- passed.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` -- 19 passed, 1 warning.
- `python -m pytest tests/test_atlas_billing_stripe_hardening.py tests/test_b2b_vendor_briefing.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py -q` -- 67 passed, 1 warning.
- `bash scripts/local_pr_review.sh --allow-dirty --current-pr-body-file .git/pr-deflection-paid-delivery-queue-body.md` -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Billing webhook queue hook | ~35 |
| Migration | ~25 |
| Workflow enrollment | ~2 |
| Tests | ~90 |
| Plan doc | ~90 |
| **Total** | **~242** |

Under the 400 LOC soft cap.
