# PR-FAQ-Deflection-Paid-Flow-Validation

## Why this slice exists

The FAQ deflection funnel now has its runtime pieces merged: execute-time
snapshot gating, paid artifact retrieval, Stripe webhook paid marking, and the
frontend Checkout contract. What is still missing is one functional validation
that proves those pieces work together as the user-facing flow we designed.

This slice adds that proof without expanding the product surface: run the
deflection report through the Content Ops execute route, confirm the free page
only receives the snapshot, confirm the paid artifact is locked, process the
Stripe webhook paid event, then confirm the paid artifact route releases the
full report.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating

Slice phase: Functional validation

1. Add one focused in-process flow test for the free snapshot -> Stripe paid
   webhook -> full artifact release path.
2. Keep the test on the existing route/webhook/storage seams; do not add new
   endpoint behavior.
3. Prove the negative boundary that the locked response does not expose the paid
   Markdown artifact before payment.
4. Enroll the paid-flow test in the existing full-dependency billing CI lane.

### Files touched

| File | Purpose |
|---|---|
| `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml` | Runs the paid-flow validation in the existing Atlas billing CI lane. |
| `tests/test_atlas_billing_content_ops_deflection_paid_flow.py` | Validates the integrated free snapshot, Stripe webhook, and paid artifact release flow. |
| `plans/PR-FAQ-Deflection-Paid-Flow-Validation.md` | Documents the slice contract and verification. |

## Mechanism

The test wires the real Content Ops router with:

- `FAQDeflectionReportService`
- `InMemoryDeflectionReportArtifactStore`
- a fixed tenant scope

It calls the route endpoint directly for `/execute`, `/snapshot`, and
`/artifact`, then sends `atlas_brain.api.billing.stripe_webhook` a fake Stripe
module whose `Webhook.construct_event` returns a signed-event-equivalent
Checkout session. The fake billing pool maps the webhook's
`UPDATE content_ops_deflection_reports` call back to the same in-memory store
used by the route, so the test proves the route and webhook agree on
`(account_id, request_id)` and paid-state semantics.

## Intentional

- This is functional validation, not a new runtime feature.
- The test uses an in-memory store adapter instead of live Postgres. The target
  invariant is cross-seam control flow and contract shape; Postgres persistence
  is already covered by the store round-trip test and can be re-run in a later
  robust/live DB slice.
- The Stripe event is fake but enters through `stripe_webhook`, so signature
  verification placement, event routing, idempotency lookup, paid marking, and
  audit insert ordering are still exercised.

## Deferred

- Future slice: live/ephemeral Postgres validation for the same paid flow.
- Future slice: browser/UI validation once the portfolio results page consumes
  the merged Checkout contract.
- Parked hardening considered: none in `HARDENING.md` touch this functional
  validation slice.

## Verification

- Command: pytest tests/test_atlas_billing_content_ops_deflection_paid_flow.py -q
  - Result: 1 passed, 1 warning.
- Command: pytest tests/test_atlas_billing_content_ops_deflection_paid_flow.py tests/test_content_ops_deflection_report.py::test_postgres_deflection_report_store_round_trips_paid_gate tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q
  - Result: 17 passed, 1 warning.
- Command: pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py -q
  - Result: 16 passed, 1 warning.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Paid-Flow-Validation.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Paid-Flow-Validation.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-paid-flow-validation.md
  - Result: passed after enrolling the paid-flow test in the billing CI lane.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| CI workflow enrollment | 8 |
| Flow validation test | 224 |
| Plan doc | 99 |
| **Total** | **331** |

Actual diff is `+328 / -1`; the LOC budget counts total changed lines.
