# PR-FAQ-Deflection-Submit-Configured-Account

## Why this slice exists

Issue #1161's remaining public funnel is the portfolio upload -> private Blob
-> ATLAS multipart submit -> hosted results loop. The private Blob submit path
is now implemented and live-smokeable, but the public upload form still asks the
buyer to paste the ATLAS service `account_id`. That is an internal handoff
value already configured server-side for the B2B-Growth service JWT and Stripe
metadata, not a buyer input.

This slice removes that manual account step from the customer intake path and
binds the private-Blob submit mode to the configured server account before it
returns the canonical result URL.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Product polish

1. Remove the ATLAS account-id input from the FAQ deflection upload form,
   private Blob token request, and browser submit request.
2. Bind private Blob upload tokens and JSON/private-Blob submits to the
   configured `ATLAS_ACCOUNT_ID` on the portfolio server while preserving the
   existing multipart rollback guard.
3. Keep the public response shape unchanged: `{ ok, request_id, account_id,
   result_path }`.
4. Update focused upload-shell tests so the browser cannot regress to asking
   for account IDs or sending account-binding headers.

### Files touched

- `plans/PR-FAQ-Deflection-Submit-Configured-Account.md`
- `portfolio-ui/api/content-ops/deflection/upload.js`
- `portfolio-ui/api/content-ops/deflection/submit.js`
- `portfolio-ui/src/pages/FaqDeflectionUpload.tsx`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

The upload page keeps collecting only buyer-owned fields:

```text
company_name, contact_email, support_platform, csv_file
```

The private Blob token route no longer requires `clientPayload.account_id`; it
validates only the configured server account and deflection CSV pathname, then
stores the configured account in the Blob token payload. After upload completes,
the browser posts the `blob_pathname` and form metadata to the portfolio submit
endpoint without `X-Atlas-Account-Id`. The server already loads
`ATLAS_ACCOUNT_ID` through `configFromEnv`; for JSON private-Blob submit mode it
uses that configured account as the binding value. If a legacy client sends an
account header on the JSON path, it must match the configured account before
ATLAS is called.

The raw multipart rollback path keeps requiring `X-Atlas-Account-Id` so local
tests and older direct-submit clients retain the explicit mismatch guard.

## Intentional

- `account_id` still appears in the result URL and Checkout metadata because
  the existing result-page, report-proxy, and Stripe contract use it to bind the
  public route to the configured ATLAS service account.
- This does not change ATLAS submit, Stripe Checkout, artifact fetch, or result
  page rendering.
- The raw multipart portfolio endpoint remains available for rollback/local
  tests and keeps the existing account mismatch rejection.
- The JSON private-Blob path remains fail-closed when `ATLAS_ACCOUNT_ID` is not
  configured.

## Deferred

- Reworking the result-page URL to avoid carrying `account_id` is deferred; it
  crosses result-page, Checkout, smoke, and Stripe metadata contracts and is not
  required to remove the buyer-facing account field.
- Running the live submit smoke against deployed environment values remains an
  operator step.
- Parked hardening: none.

## Verification

- `npm run test:deflection-upload-shell --prefix portfolio-ui` -- 20 checks passed.
- `npm run test:deflection-result --prefix portfolio-ui` -- 12 checks passed.
- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` -- 14 checks passed.
- `npm run build --prefix portfolio-ui` -- passed after hydrating local
  `portfolio-ui/node_modules` with `npm install --prefix portfolio-ui`; Vite
  emitted the existing large chunk advisory and skipped sitemap generation
  because `PORTFOLIO_SITE_URL` / `VITE_SITE_URL` is not set.
- Pending: `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-submit-configured-account.md`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 103 |
| Upload token account binding | 2 |
| Submit endpoint account binding | 10 |
| Upload form simplification | 23 |
| Focused tests | 55 |
| **Total** | **193** |

Actual diff is 5 files, +170 / -23. Under the 400 LOC soft cap.
