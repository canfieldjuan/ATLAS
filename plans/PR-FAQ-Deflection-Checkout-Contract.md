# PR-FAQ-Deflection-Checkout-Contract

## Why this slice exists

The free deflection snapshot, paid artifact route, and Stripe webhook paid flag
now exist, but the portfolio/results-page session still needs a stable handoff
for the Checkout seam. Without a written contract, the frontend can easily
guess at metadata, paid-state, or trust boundaries and accidentally recreate the
portfolio-owned mark-paid path we intentionally avoided.

This slice closes the handoff requested in issue #1149: document the exact
snapshot/artifact endpoints, required Stripe Checkout metadata, ATLAS-owned
paid flag behavior, and paid-state interpretation for the frontend builder.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating

Slice phase: Product polish

1. Add a frontend contract doc for the one-time FAQ deflection Checkout flow.
2. Link that contract from the existing FAQ report frontend contract.
3. Add a doc test that pins the route names, metadata keys, Stripe source, and
   trust-boundary language.

### Files touched

| File | Purpose |
|---|---|
| `docs/frontend/content_ops_faq_deflection_checkout_contract.md` | Documents the portfolio-to-ATLAS Checkout and paid-release contract. |
| `docs/frontend/content_ops_faq_report_contract.md` | Links the Checkout contract from the existing report/snapshot contract. |
| `tests/test_content_ops_faq_report_contract_docs.py` | Pins the contract link and required Checkout handoff terms. |
| `plans/PR-FAQ-Deflection-Checkout-Contract.md` | Documents the slice contract and verification. |

## Mechanism

The new doc makes the paid funnel explicit:

1. Portfolio renders the free `DeflectionSnapshot`.
2. Portfolio creates Stripe Checkout with
   `metadata.source=content_ops_deflection_report`, `metadata.account_id`, and
   `metadata.request_id`.
3. Stripe calls ATLAS' signed webhook.
4. ATLAS verifies the event, validates amount/currency, and flips
   `content_ops_deflection_reports.paid`.
5. Portfolio hydrates the full report from
   `GET /content-ops/deflection-reports/{request_id}/artifact` only after the
   artifact route stops returning 403.

The test is intentionally string-level. This is a frontend handoff contract, so
the drift risk is missing or renamed contract terms rather than runtime logic.

## Intentional

- No runtime endpoint changes land here. The paid gate and Stripe webhook
  already shipped; this slice documents how the portfolio should use them.
- The contract tells the portfolio to treat the artifact route as the paid-state
  read (`200` unlocked, `403` locked). A separate status endpoint is not required
  for the current free snapshot -> paid report flow.
- The operator `POST /content-ops/deflection-reports/{request_id}/paid` route is
  named only as "not for portfolio use" because Stripe webhook is the source of
  truth for customer payment.

## Deferred

- Future slice: add a dedicated paid-state endpoint only if the frontend needs a
  lighter read than probing the artifact route.
- Future slice: subscription/writeback contract for the $500/mo ongoing offer.
- Parked hardening considered: none in `HARDENING.md` touch this docs-only
  handoff slice.

## Verification

- Command: pytest tests/test_content_ops_faq_report_contract_docs.py -q
  - Result: 5 passed.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Checkout-Contract.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Checkout-Contract.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-checkout-contract.md
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Checkout contract doc | 124 |
| Existing contract link | 5 |
| Doc test | 25 |
| Plan doc | 93 |
| **Total** | **247** |
