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

## Interpreting Results

The result artifact records HTTP statuses, the returned `request_id`, compact
submit diagnostics, and deterministic errors. It intentionally redacts the
bearer token and signed blob query string; only the blob host is recorded.

Exit codes:

- `0`: the hosted submit/snapshot/unpaid-artifact handoff passed.
- `1`: ATLAS returned a response that violated the handoff contract.
- `2`: required inputs were missing or unsafe for hosted proof.

Stripe webhook paid-unlock validation is a separate follow-up. This smoke
expects the artifact route to stay locked before payment.
