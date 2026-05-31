# PR-FAQ-Deflection-Result-Configured-Account

## Why this slice exists

PR-FAQ-Deflection-Submit-Configured-Account removed the buyer-facing account id
field from the FAQ deflection upload form, but intentionally deferred the
follow-up because the existing result page, report proxy, Checkout route, and
smokes still carried `account_id` in the public result URL and browser Checkout
payload.

Now that the submit path is bound to the configured server account and the
route-handler smoke is merged, the customer-facing result URL can carry only
the Content Ops `request_id`. The portfolio server should derive the ATLAS
account id from `ATLAS_ACCOUNT_ID` for snapshot/artifact fetches and Stripe
metadata.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Product polish

1. Remove `account_id` from generated FAQ deflection result URLs and Checkout
   return URLs.
2. Let result-page and report proxy requests derive the ATLAS account binding
   from configured server env while still rejecting mismatched legacy query
   values.
3. Let the Checkout route derive Stripe metadata `account_id` from configured
   server env while accepting only matching legacy payloads.
4. Remove the result page's browser-side account id data attributes and
   Checkout request body field.
5. Update focused portfolio result/upload tests to lock the configured-account
   behavior and the no-account public URL shape.

### Files touched

- `plans/PR-FAQ-Deflection-Result-Configured-Account.md`
- `portfolio-ui/api/content-ops/deflection/atlas-report.js`
- `portfolio-ui/api/content-ops/deflection/checkout.js`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/src/pages/FaqDeflectionResult.tsx`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

`resultPath()` stops appending `account_id` and preserves only the optional
Checkout return token:

```text
/services/faq-deflection/results/{request_id}
/services/faq-deflection/results/{request_id}?checkout=success
```

The report proxy keeps the same ATLAS calls, but `loadDeflectionReport()` uses
`ATLAS_ACCOUNT_ID` when the browser omits `account_id`. If a legacy URL or API
caller sends an account id, it must still be a UUID and match the configured
account before ATLAS is called.

The Checkout route similarly reads `ATLAS_ACCOUNT_ID` server-side for Stripe
metadata. The browser posts only `{ request_id }`, and legacy `account_id`
payloads fail closed on mismatch.

## Intentional

- Stripe metadata still includes `account_id`; the change is where that value
  comes from, not the ATLAS webhook contract.
- The public submit response can continue to include `account_id` for
  compatibility, but its `result_path` no longer carries it.
- Legacy account-id query/body inputs are rejected on mismatch instead of
  silently ignored.
- This does not change ATLAS submit, private Blob upload, Stripe key selection,
  or artifact rendering semantics.

## Deferred

- Running the full hosted upload -> submit -> result -> Checkout loop remains
  an operator/live-environment step.
- Parked hardening: none.

## Verification

- `node --check portfolio-ui/api/content-ops/deflection/checkout.js && node --check portfolio-ui/api/content-ops/deflection/atlas-report.js && node --check portfolio-ui/api/content-ops/deflection/result-page.js && node --check portfolio-ui/api/content-ops/deflection/report.js && node --check portfolio-ui/scripts/faq-deflection-result-page.test.mjs && node --check portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs && node --check portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`
- `npm run test:deflection-result --prefix portfolio-ui` (16 checks)
- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` (15 checks)
- `npm run test:deflection-upload-shell --prefix portfolio-ui` (21 checks)
- `npm run build --prefix portfolio-ui` (passed after `npm install --prefix portfolio-ui`; Vite emitted the existing large-chunk advisory and skipped sitemap generation because `PORTFOLIO_SITE_URL` / `VITE_SITE_URL` is not set)
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-result-configured-account.md`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 100 |
| Result/report/checkout route changes | 49 |
| React result fallback page | 21 |
| Focused tests | 164 |
| **Total** | **334** |

Under the 400 LOC soft cap.
