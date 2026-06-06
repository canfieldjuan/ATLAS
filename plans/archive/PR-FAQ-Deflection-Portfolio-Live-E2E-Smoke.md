# PR-FAQ-Deflection-Portfolio-Live-E2E-Smoke

## Why this slice exists

With the ATLAS backend paid-flow contract already in place, the next handoff
risk is the portfolio result page: it must expose the exact request and account
metadata that Checkout uses, so the hosted smoke can verify the customer-facing
page before any paid artifact is released.

The existing smoke already checks those page hooks, but the portfolio result
page only rendered the Checkout metadata in text. The unlock CTA itself did not
carry the `data-checkout-*` attributes that the smoke validates, leaving the
hosted result-page handoff false-red even when the underlying Checkout endpoint
was correctly building Stripe metadata.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection
Slice phase: Functional validation

1. Bind the result page unlock CTA to the Checkout metadata source,
   `request_id`, and `account_id` as machine-readable attributes.
2. Keep the existing fail-closed paid boundary: the page still calls portfolio
   Checkout only, never the privileged ATLAS `/paid` route.
3. Tighten the portfolio result-page test so the hosted HTML contract cannot
   regress back to text-only metadata.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Live-E2E-Smoke.md`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/src/pages/FaqDeflectionResult.tsx`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`

## Mechanism

`renderResultPage` already receives the canonical `requestId`, `accountId`, and
`CHECKOUT_SOURCE`. This slice threads those values onto the
`data-atlas-deflection-unlock` button:

```html
<button
  data-atlas-deflection-unlock
  data-checkout-source="content_ops_deflection_report"
  data-checkout-request_id="..."
  data-checkout-account_id="..."
>
```

The React fallback route gets the same attributes so both the Vercel rewrite
HTML and SPA page keep the same instrumentation contract.

## Intentional

- This does not create a new live smoke script; the existing
  `scripts/smoke_content_ops_deflection_portfolio_result_page.py` is already
  the contract runner.
- This does not change Stripe session creation, amount validation, webhook
  handling, or artifact fetching.
- The page continues to render only snapshot fields before payment; paid report
  rendering remains gated on the ATLAS artifact route returning `200`.

## Deferred

- Parked hardening: none. The existing `HARDENING.md` FAQ deflection note is
  for the Intel report UI lane, not the portfolio result page.
- Running the hosted smoke against the production portfolio URL remains the
  deploy verification step after this PR lands; this slice fixes the local
  contract gap that would block that smoke.

## Verification

- `npm run test:deflection-result` from `portfolio-ui/` - 12 checks passed.
- `npm run test:deflection-atlas-proxy` from `portfolio-ui/` - 14 checks passed.
- `npm run test:deflection-upload-shell` from `portfolio-ui/` - 18 checks passed.
- `python -m pytest tests/test_smoke_content_ops_deflection_portfolio_result_page.py -q` - 8 passed.
- `npm run build` from `portfolio-ui/` - passed; Vite emitted the existing large chunk advisory and skipped sitemap generation because `PORTFOLIO_SITE_URL` / `VITE_SITE_URL` was not set.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 70 |
| Result page attributes | 8 |
| React fallback attributes | 4 |
| Test tightening | 30 |
| **Total** | **112** |

Actual diff is 5 files, +130 / -4.
