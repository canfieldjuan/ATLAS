# Content Ops FAQ Deflection Submit Handoff Runbook

Use this runbook to prove the deployed ATLAS API can consume the portfolio's
support-ticket CSV over the authenticated submit handoff and return the gated
deflection report response.

## Required Inputs

- `ATLAS_API_BASE_URL`: deployed Atlas API host. Localhost is rejected because
  this is a hosted handoff proof.
- `ATLAS_B2B_JWT` or `ATLAS_TOKEN`: bearer token for the shared B2B-Growth
  service account.
- `ATLAS_ACCOUNT_ID` or `ATLAS_FAQ_SEARCH_ACCOUNT_ID`: the account id that
  maps to the bearer token. The submit route derives account scope from auth;
  the portfolio needs this value for Stripe Checkout metadata.
- `ATLAS_DEFLECTION_SUBMIT_CSV_FILE`: optional local support-ticket CSV for the
  preferred multipart hosted smoke. When omitted and no blob URL is provided,
  the smoke uses
  `docs/extraction/validation/fixtures/faq_deflection_live_upload_sample.csv`.
  For customer data, point this to a representative private-Blob export
  downloaded by the operator or portfolio server-side code path.
- `ATLAS_DEFLECTION_SUBMIT_BLOB_URL`: optional legacy HTTPS support-ticket CSV
  blob URL. This fallback remains available for rollback coverage but is not
  the preferred production PII posture.
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

## Checked Fixture

The repo includes a non-sensitive CSV fixture for live upload and snapshot
generation:

```text
docs/extraction/validation/fixtures/faq_deflection_live_upload_sample.csv
```

It has 12 synthetic closed support tickets with repeated export, billing,
security, and team/admin themes so the generated free snapshot has meaningful
clusters without storing customer data.

The hosted ATLAS submit smoke uses this fixture by default when neither
`ATLAS_DEFLECTION_SUBMIT_CSV_FILE` nor `ATLAS_DEFLECTION_SUBMIT_BLOB_URL` is
set. To override the default explicitly:

```bash
export ATLAS_DEFLECTION_SUBMIT_CSV_FILE=docs/extraction/validation/fixtures/faq_deflection_live_upload_sample.csv
export ATLAS_DEFLECTION_COMPANY_NAME="Atlas Fixture Co."
export ATLAS_DEFLECTION_CONTACT_EMAIL="ops@example.com"
export ATLAS_DEFLECTION_SUPPORT_PLATFORM="zendesk"
```

To test the public portfolio upload page, select the same CSV file in the
browser. The page uploads it to private Vercel Blob, posts only the private
Blob pathname and buyer fields to the portfolio submit route, then redirects to
an account-less result URL:

```text
/services/faq-deflection/results/{request_id}
```

## Hosted Smoke

```bash
python scripts/smoke_content_ops_deflection_submit_handoff.py \
  --limit 1000 \
  --output-result /tmp/faq-deflection-submit-handoff-result.json \
  --json
```

With `ATLAS_DEFLECTION_SUBMIT_CSV_FILE` or `--csv-file`, the smoke sends:

```http
POST /api/v1/content-ops/deflection-reports/submit
Content-Type: multipart/form-data
```

Form fields:

```text
csv_file=@tickets.csv
support_platform=zendesk
company_name=Acme Co.
contact_email=lead@example.com
limit=1000
```

This matches the production handoff: the portfolio reads its private Blob
server-side, then posts raw CSV bytes to ATLAS with the existing bearer token.
Do not expose raw support-ticket CSVs through a public signed proxy.

If no CSV file is provided, the smoke falls back to the legacy JSON contract:

```json
{
  "blob_url": "https://...",
  "support_platform": "zendesk",
  "company_name": "Acme Co.",
  "contact_email": "lead@acme.co",
  "limit": 1000
}
```

Both submit modes then verify:

- Submit returns `200` with `status: "completed"`.
- The top-level `request_id` is non-empty.
- The `faq_deflection_report` step result has a matching nested `request_id`.
- The free snapshot has `summary` and `top_questions` and does not contain
  answer, evidence, source id, Markdown, or full FAQ result fields.
- `full_report` is `{ "status": "locked", "reason": "payment_required" }`.
- Submit diagnostics identify `portfolio_deflection_submit` and include the
  expected byte counter (`uploaded_bytes` for multipart, `blob_bytes` for the
  legacy JSON path).
- `GET /api/v1/content-ops/deflection-reports/{request_id}/snapshot` returns
  `200` with the same snapshot.
- `GET /api/v1/content-ops/deflection-reports/{request_id}/artifact` returns
  `403` before payment.

### Stale Deploy Diagnostic

If the multipart hosted smoke fails with:

```text
deployed submit route rejected multipart as a JSON body
```

then the host accepted auth but did not serve the current multipart submit
contract. The exact FastAPI shape is a `422` with
`detail[0].type == "model_attributes_type"` at `loc == ["body"]`, which is the
old JSON-body route behavior when it receives `multipart/form-data`.

Treat this as deployment/runtime drift, not bad portfolio input. Rebuild and
redeploy ATLAS from a commit that includes the multipart
`submit_deflection_report(request: Request)` route, the extracted package copy
in the Docker image, and the `python-multipart` dependency. Re-run the smoke
after deploy; do not continue to Stripe paid-unlock validation until submit
returns a real `request_id`.

### Full-Volume Gate Profile

For #1440-style full-volume CFPB proof runs, use the calibrated profile instead
of hand-entering each minimum:

```bash
python scripts/smoke_content_ops_deflection_submit_handoff.py \
  --csv-file /path/to/cfpb_real_upload_under_50mb.csv \
  --company-name "CFPB Public Archive" \
  --contact-email canfieldjuan24@gmail.com \
  --support-platform other \
  --volume-gate-profile full-volume-cfpb \
  --timeout 360 \
  --output-result /tmp/faq-deflection-submit-handoff.json \
  --json
```

The profile requires at least 50,000,000 uploaded bytes, 30,000 source rows,
30,000 submitted rows, 30 generated questions, 25,000 repeat tickets, and 5
visible top questions. The repeat-ticket threshold is calibrated below the
first committed live full-volume proof result of 27,384 repeat tickets while
still rejecting tiny fixture reports. If a stricter proof is intentional, pass a
nonzero explicit `--min-*` flag; explicit minimums override profile defaults.

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
exposes stable result/unlock hooks, preserves the Checkout source and
`request_id` metadata, rejects account ids in the public result URL and HTML,
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

If the webhook result records status `400` with `error_detail` equal to
`Invalid signature`, treat that as deployed Stripe signing-secret drift. Do not
bypass verification or retry with guessed secrets. Align the deployed
`ATLAS_SAAS_STRIPE_WEBHOOK_SECRET` with the secret used by the smoke, or rotate
the deployed value to include the new secret, then rerun this smoke.

To also prove the deployed duplicate-event guard, add `--replay-webhook`. That
posts the same signed event a second time and requires the webhook response to
return `{"status": "already_processed"}` before the final paid artifact fetch.
Use this only after the first paid-unlock smoke path is expected to succeed.

## Interpreting Results

The result artifact records HTTP statuses, submit mode, CSV file size when
multipart is used, the returned `request_id`, compact submit diagnostics, and
deterministic errors. It intentionally redacts the bearer token, signed blob
query string, and CSV contents; only the blob host is recorded for the legacy
JSON path.

Exit codes:

- `0`: the hosted submit/snapshot/unpaid-artifact handoff passed.
- `1`: ATLAS returned a response that violated the handoff contract.
- `2`: required inputs were missing or unsafe for hosted proof.

Run the paid-unlock smoke only after the pre-payment submit and result-page
smokes pass. It intentionally consumes the request by marking the report paid.
