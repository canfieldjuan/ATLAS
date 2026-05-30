# Content Ops FAQ Deflection Submit Handoff Runbook

Use this runbook to prove the deployed ATLAS API can consume the portfolio's
signed support-ticket CSV blob and return the gated deflection report response.

## Required Inputs

- `ATLAS_API_BASE_URL`: deployed Atlas API host. Localhost is rejected because
  this is a hosted handoff proof.
- `ATLAS_B2B_JWT` or `ATLAS_TOKEN`: bearer token for the shared B2B-Growth
  service account.
- `ATLAS_ACCOUNT_ID` or `ATLAS_FAQ_SEARCH_ACCOUNT_ID`: the account id that
  maps to the bearer token. The submit route derives account scope from auth;
  the portfolio needs this value for Stripe Checkout metadata.
- `ATLAS_DEFLECTION_SUBMIT_BLOB_URL`: HTTPS support-ticket CSV blob URL. A
  private signed URL is preferred; the URL must live long enough for the sync
  submit call.
- `ATLAS_DEFLECTION_COMPANY_NAME`: company name to include in the report title.
- `ATLAS_DEFLECTION_CONTACT_EMAIL`: buyer/contact email.
- `ATLAS_DEFLECTION_SUPPORT_PLATFORM`: `zendesk`, `intercom`, `help_scout`, or
  `other`. Defaults to `zendesk` when omitted.

## Prepare Auth Env

If you have a B2B-Growth ATLAS login but do not have the JWT/account id yet,
prepare the local `.env` without printing the token:

```bash
export ATLAS_LOGIN_PASSWORD='<password>'
python scripts/prepare_content_ops_deflection_env.py \
  --base-url https://<deployed-atlas-api-host> \
  --email <b2b-growth-user-email>
```

Prefer `ATLAS_LOGIN_PASSWORD` or the interactive password prompt over
`--password`; command-line arguments can be visible in shell history and process
list output.

The helper logs into `/api/v1/auth/login`, verifies `/api/v1/auth/me` returns a
B2B account on `b2b_growth` or higher, then writes:

```dotenv
ATLAS_API_BASE_URL=https://...
ATLAS_B2B_JWT=<redacted bearer token>
ATLAS_ACCOUNT_ID=<account id from /auth/me>
```

to `.env`. Existing ATLAS keys are not replaced unless `--force` is passed.

## Preflight

```bash
python scripts/smoke_content_ops_deflection_submit_handoff.py \
  --preflight-only \
  --output-result /tmp/faq-deflection-submit-preflight.json \
  --json
```

Preflight exits `0` only when all required inputs are present and shaped for a
hosted proof. It does not fetch the blob or call ATLAS.

## Hosted Smoke

```bash
python scripts/smoke_content_ops_deflection_submit_handoff.py \
  --limit 1000 \
  --output-result /tmp/faq-deflection-submit-handoff-result.json \
  --json
```

The smoke sends:

```json
{
  "blob_url": "https://...",
  "support_platform": "zendesk",
  "company_name": "Acme Co.",
  "contact_email": "lead@acme.co",
  "limit": 1000
}
```

to `/api/v1/content-ops/deflection-reports/submit`, then verifies:

- Submit returns `200` with `status: "completed"`.
- The top-level `request_id` is non-empty.
- The `faq_deflection_report` step result has a matching nested `request_id`.
- The free snapshot has `summary` and `top_questions` and does not contain
  answer, evidence, source id, Markdown, or full FAQ result fields.
- `full_report` is `{ "status": "locked", "reason": "payment_required" }`.
- Submit diagnostics identify `portfolio_deflection_submit`.
- `GET /api/v1/content-ops/deflection-reports/{request_id}/snapshot` returns
  `200` with the same snapshot.
- `GET /api/v1/content-ops/deflection-reports/{request_id}/artifact` returns
  `403` before payment.

## Portfolio Result Page Smoke

After the hosted submit smoke returns a `request_id`, validate the portfolio
result page using the same ATLAS auth values plus:

- `ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL`: hosted portfolio result page for
  the generated report.
- `ATLAS_DEFLECTION_REQUEST_ID`: `request_id` returned by the submit smoke.

```bash
python scripts/smoke_content_ops_deflection_portfolio_result_page.py \
  --output-result /tmp/faq-deflection-portfolio-result-page.json \
  --json
```

The smoke verifies the hosted page renders the same `request_id` and
`account_id`, exposes stable result/unlock hooks, preserves the Checkout
metadata keys (`content_ops_deflection_report`, `request_id`, `account_id`),
keeps the free snapshot limited to summary/top-question data, and confirms the
artifact endpoint still returns `403` before payment.

## Stripe Paid-Unlock Smoke

After the hosted result page smoke passes and before using the same request for
manual recovery, validate the signed Stripe webhook trust path with:

- `ATLAS_SAAS_STRIPE_WEBHOOK_SECRET` or `STRIPE_WEBHOOK_SECRET`: deployed ATLAS
  Stripe webhook signing secret.
- `ATLAS_DEFLECTION_REQUEST_ID`: `request_id` returned by the submit smoke.

```bash
python scripts/smoke_content_ops_deflection_stripe_paid_unlock.py \
  --output-result /tmp/faq-deflection-stripe-paid-unlock.json \
  --json
```

The smoke confirms the artifact route is locked before the webhook, posts a
Stripe-compatible signed `checkout.session.completed` event with the required
deflection metadata, and then confirms the artifact route returns the full paid
report. The event id and Checkout session id are generated when omitted, so a
fresh run does not reuse Stripe webhook idempotency keys.

To also prove the deployed duplicate-event guard, add `--replay-webhook`. That
posts the same signed event a second time and requires the webhook response to
return `{"status": "already_processed"}` before the final paid artifact fetch.
Use this only after the first paid-unlock smoke path is expected to succeed.

## Interpreting Results

The result artifact records HTTP statuses, the returned `request_id`, compact
submit diagnostics, and deterministic errors. It intentionally redacts the
bearer token and signed blob query string; only the blob host is recorded.

Exit codes:

- `0`: the hosted submit/snapshot/unpaid-artifact handoff passed.
- `1`: ATLAS returned a response that violated the handoff contract.
- `2`: required inputs were missing or unsafe for hosted proof.

Run the paid-unlock smoke only after the pre-payment submit and result-page
smokes pass. It intentionally consumes the request by marking the report paid.
