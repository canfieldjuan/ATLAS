# PR-Deflection-Checkout-Concurrency-Guard

## Why this slice exists

The operator explicitly deferred true multi-tenant work until after customer
traction and asked to harden the current paid funnel for multiple simultaneous
users/actions instead. The concrete concurrent-use gap already tracked in
#1386 is duplicate checkout creation: two clicks, two browser tabs, or a retry
can create more than one Stripe Checkout Session for the same deflection
report.

This is narrower than multi-tenant. It keeps the current configured-account
model, but makes the checkout boundary tolerate concurrent attempts against
the same report.

The estimated diff is over the 400 LOC soft cap because the slice needs both
the endpoint implementation and behavior-level portfolio tests for authorized,
duplicate, and unauthorized checkout paths. Splitting those would leave a
payment/idempotency change without its regression proof.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Before creating a Stripe Checkout Session, the portfolio checkout endpoint
   asks Atlas for deflection checkout authorization for the requested report.
   The backend authorization route already fails closed when the report is
   missing, already paid, lacks an artifact, or has payment terms that do not
   match the backend gate.
2. The Stripe Checkout create request carries a stable idempotency key derived
   from the checkout source, configured account, and request id, so duplicate
   concurrent checkout attempts converge at Stripe instead of creating
   independent sessions.
3. Fallback-key deployments validate the exact Stripe Price id chosen for the
   session after Atlas authorization, so Atlas/portfolio price drift fails
   before payment rather than after the webhook.
4. Inline amount-mode checkout validates the selected amount and currency
   before Stripe session creation, so Atlas-authorized inline terms cannot
   charge below the backend webhook floor or outside USD.
5. Portfolio tests cover authorized checkout, already-paid/missing-report
   rejection, Atlas authorization failure, the Stripe idempotency header, and
   the Atlas-authorized-price and inline-amount validation failures.
6. Update #1386 after the PR opens/merges so the issue records that this is
   concurrent-checkout hardening, not multi-tenant work.

### Files touched

- `plans/PR-Deflection-Checkout-Concurrency-Guard.md`
- `portfolio-ui/api/content-ops/deflection/checkout.js`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`

### Review Contract

- Acceptance criteria:
  - [ ] Checkout creation calls Atlas checkout authorization before Stripe.
  - [ ] Missing, already-paid, or otherwise unauthorized reports return a
        non-2xx response and do not call Stripe.
  - [ ] Stripe Checkout creation uses a stable idempotency key for the same
        configured account and request id.
  - [ ] The Stripe Price preflight validates the same price id that the handler
        will send to Stripe.
  - [ ] Inline Checkout amount-mode rejects non-USD terms and terms below
        150000 cents before creating a Stripe session.
  - [ ] Existing metadata, success/cancel URL, restricted-key, and no-privileged
        paid-route behavior stay intact.
- Affected surfaces: portfolio API, Atlas proxy boundary, Stripe Checkout,
  tests, #1386 tracker.
- Risk areas: payment correctness, concurrency/idempotency, authorization,
  public API behavior, CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R5, R8, R9, R12, R13, R14.

## Mechanism

The portfolio checkout handler already validates the request id and configured
account, builds return URLs, and creates a Stripe Checkout Session. This slice
adds a pre-Stripe authorization call to the Atlas route
`/api/v1/content-ops/deflection-reports/{request_id}/checkout-authorization`
using the same service token and configured account used by the result-page
proxy. The handler accepts only the backend's authorized response and fails
closed for missing, already-paid, unavailable-artifact, or proxy/config errors.

After authorization succeeds, the Stripe create request includes an
`Idempotency-Key` header built from a fixed namespace plus the configured
account and request id. That key is stable for duplicate attempts on the same
report and different for different reports/accounts. Stripe then owns
deduplicating concurrent create attempts at the payment boundary.

For fallback-key deployments that can read Stripe Price metadata, the handler
validates the same price id it will send to Stripe. Atlas-authorized price ids
win over portfolio env fallback; if that selected Price is inactive, non-USD,
or below the backend webhook floor, checkout fails before payment.

When no Stripe Price id resolves and the handler would create inline
`price_data`, it validates the selected inline terms first. Atlas inline terms
win over env fallback, and the selected amount/currency must be USD and at
least 150000 cents before any Stripe Checkout create call is attempted.

## Intentional

- Do not add multi-tenant account selection or per-company configuration in this
  slice. The operator explicitly deferred that until after customer traction.
- Do not call the privileged paid-release route from the portfolio checkout
  endpoint. Checkout authorization is read-only/payability proof; Stripe
  webhook fulfillment remains the only unlock path.
- Do not add captcha or website-edge IP throttling here. #1583 already hardened
  the Atlas backend route limiter; this slice targets duplicate paid-session
  creation.
- Keep restricted-key deployments on the existing no-Price-read path; restricted
  keys may be scoped to Checkout creation only. Atlas authorization remains the
  fail-closed terms source there, and inline amount-mode terms still pass the
  local USD/floor guard before Checkout creation.

## Deferred

- Real resolution-bearing export report-quality proof remains the next
  launch-gating product slice (#1419).
- Broad public-traffic edge controls, such as captcha or per-IP website
  throttles, remain operational hardening before a fully unattended launch.
- True multi-tenant auth/config remains future work after customer traction.

Parked hardening: none.

## Verification

- Local verification:
  - `npm run test:deflection-result` from `portfolio-ui/`
  - `npm run test:deflection-upload-shell` from `portfolio-ui/`
  - `npm run test:deflection-atlas-proxy` from `portfolio-ui/`
  - Local PR review bundle.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Checkout-Concurrency-Guard.md` | 139 |
| `portfolio-ui/api/content-ops/deflection/checkout.js` | 187 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 466 |
| **Total** | **792** |
