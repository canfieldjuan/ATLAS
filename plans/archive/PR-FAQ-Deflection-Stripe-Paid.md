# PR-FAQ-Deflection-Stripe-Paid

## Why this slice exists

PR-FAQ-Deflection-Paid-Gate shipped the ATLAS-owned free snapshot and paid full
artifact gate, but the paid flag still requires a privileged operator call. The
portfolio results page can start Stripe Checkout, but ATLAS needs to consume the
verified Stripe completion event and flip the paid flag itself so the full
report release does not depend on a manual unlock.

This slice builds the thinnest production payment seam: a verified
`checkout.session.completed` webhook with deflection-report metadata marks the
matching ATLAS report paid. It keeps checkout creation portfolio-owned and does
not add UI, subscriptions, or new report persistence.

Review found three payment-delivery gaps while the PR was open: the Atlas test
was enrolled in the extracted standalone CI lane, the deflection account
namespace could collide with the `billing_events.account_id` FK, and a valid
payment with no matching report row could be acknowledged permanently. Those
are part of this production seam, so this slice fixes them before merge.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating

Slice phase: Production hardening

1. Teach the existing Stripe webhook to recognize
   `metadata.source=content_ops_deflection_report`.
2. Require the signed Stripe session to include `account_id` and `request_id`
   metadata and a completed one-time paid Checkout session.
3. Validate the session amount/currency against the configured deflection
   report one-time price.
4. Mark the matching `content_ops_deflection_reports` row paid with the Stripe
   checkout session id as the payment reference.
5. Keep deflection Stripe events out of the `billing_events.account_id`
   `saas_accounts` FK field because deflection report accounts are stored in a
   separate text namespace.
6. Fail closed when a valid paid session cannot find a report row, so Stripe
   retries instead of permanently stranding a paid-but-locked report.
7. Add focused tests proving the happy path and detector failure branches.
8. Move the Atlas webhook test into its own path-scoped CI workflow instead of
   the extracted standalone lane.

### Files touched

| File | Purpose |
|---|---|
| `atlas_brain/config.py` | Adds the configured one-time deflection report amount/currency used by the Stripe webhook guard. |
| `atlas_brain/api/billing.py` | Adds deflection-report Stripe completion handling to the existing webhook path. |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | Proves the webhook helper marks paid only for valid deflection report payment sessions. |
| `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml` | Runs the Atlas webhook test in a full-dependency path-scoped CI lane. |
| `plans/PR-FAQ-Deflection-Stripe-Paid.md` | Documents the slice contract and verification. |

## Mechanism

The existing `/webhooks/stripe` handler already verifies the Stripe signature,
deduplicates events through `billing_events`, and branches on
`checkout.session.completed`. This slice adds a deflection-report branch before
the existing subscription checkout handler:

```python
if meta.get("source") == "content_ops_deflection_report":
    account_id = await _handle_content_ops_deflection_report_checkout_completed(...)
```

The helper fail-closes unless the session is a completed one-time payment with:

- `mode == "payment"`
- `payment_status == "paid"`
- `amount_total >= ATLAS_SAAS_STRIPE_CONTENT_OPS_DEFLECTION_REPORT_AMOUNT_CENTS`
- `currency == ATLAS_SAAS_STRIPE_CONTENT_OPS_DEFLECTION_REPORT_CURRENCY`
- non-empty `metadata.account_id`
- non-empty `metadata.request_id`

When valid, it updates `content_ops_deflection_reports` with
`paid=true`, `paid_at`, and `payment_reference=session.id` for the exact
`(account_id, request_id)` pair. Deflection events still write the existing
`billing_events` idempotency/audit row, but with `account_id=NULL`; the
deflection report account namespace is intentionally not asserted to be a
`saas_accounts(id)` foreign key.

If the signed paid session is valid but no report row matches, the helper logs
an error and raises a non-2xx webhook response before the idempotency row is
written. That keeps the report locked and lets Stripe retry/reconcile instead
of acknowledging a paid event that did not unlock anything.

## Intentional

- No checkout creation endpoint lands here. The portfolio owns Stripe Checkout
  initiation and must set the metadata above.
- No subscription handling lands here. The $500/mo ongoing product remains a
  later slice with a different lifecycle than the one-time $1,500 report.
- Missing or invalid metadata does not 4xx the Stripe webhook after signature
  verification; it logs and leaves the report locked because retrying bad
  metadata will not make it valid.
- A valid paid session with no matching report row does return non-2xx. That
  is different from bad metadata: it can represent a report-save/checkout race,
  and acknowledging it would strand a paid customer with a locked report.
- Deflection checkout events log to `billing_events` with `account_id=NULL`.
  The event id remains the idempotency key, while the paid report state remains
  keyed by `content_ops_deflection_reports(account_id, request_id)`.

## Deferred

- Future slice: portfolio/checkout contract docs with the exact required Stripe
  metadata and success/cancel URL expectations.
- Future slice: subscription/writeback payment flow for the $500/mo ongoing
  offer.
- Parked hardening considered: prior `mark_paid` non-asyncpg command-tag NIT is
  not required for this Stripe webhook slice because production uses asyncpg and
  this slice tests the SQL call at the helper boundary.

## Verification

- Command: python -m py_compile atlas_brain/api/billing.py atlas_brain/config.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py
  - Result: passed.
- Command: pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_content_ops_deflection_report.py::test_postgres_deflection_report_store_round_trips_paid_gate -q
  - Result: 16 passed, 1 warning.
- Command: pytest tests/test_llm_gateway_plan_tier.py tests/test_atlas_content_ops_generated_assets_api.py::test_content_ops_deflection_paid_route_uses_operator_gate tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q
  - Result: 34 passed, 1 warning.
- Command: pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q
  - Result: 15 passed, 1 warning.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Stripe-Paid.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Stripe-Paid.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-stripe-paid.md
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Config + billing webhook helper | 122 |
| CI workflow | 41 |
| Tests | 260 |
| Plan doc | 146 |
| **Total** | **569** |

The slice exceeds the soft 400 LOC budget because the review fix has to move
the Atlas webhook test into a dedicated workflow and add negative fixtures for
each payment-failure branch. Splitting that would leave either red CI or an
unguarded paid-unlock seam.
