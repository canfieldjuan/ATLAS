# Content Ops FAQ Deflection Portfolio Hosted Smoke - 2026-05-30

## Summary

The hosted ATLAS submit path passed and produced a fresh unpaid FAQ deflection
report, but the production portfolio result-page handoff did not pass.

ATLAS returned the expected report states:

- submit: `200`
- snapshot: `200`
- unpaid artifact: `403`

The production portfolio URL returned `404`, so this run does not prove the
production customer-facing result page. No bearer token, Stripe secret, or CSV
contents are recorded here.

Security update (2026-06-15): this historical request ID and result URL are now
redacted. The history sweep found the historical artifact unlocked under
service-token auth (`200`), relocked it through the signed Stripe revocation
path, and verified the artifact endpoint returned `403` afterward. Hash label:
`3a0db3e41b8f`.

## Inputs

| Input | Value |
|---|---|
| ATLAS API base URL | `https://atlas-brain.tailc7bd29.ts.net` |
| Portfolio production host | `https://juancanfield.com` |
| Submit CSV fixture | `extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv` |
| Submit mode | multipart |
| Request id | `content-ops-[redacted:3a0db3e41b8f]` |
| Account id | `<redacted-account-id>` |

## Commands

The submit smoke used the local root `.env` / `.env.local` credentials without
printing the bearer token:

```bash
python scripts/smoke_content_ops_deflection_submit_handoff.py \
  --csv-file extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv \
  --company-name "Atlas SaaS Demo" \
  --contact-email ops@example.com \
  --support-platform zendesk \
  --limit 1000 \
  --output-result tmp/faq_deflection_portfolio_hosted_smoke_20260530/submit-result.json \
  --json
```

The result-page smoke targeted the production portfolio URL:

```bash
python scripts/smoke_content_ops_deflection_portfolio_result_page.py \
  --output-result tmp/faq_deflection_portfolio_hosted_smoke_20260530/result-page-result.json \
  --json
```

with:

```text
ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL=<redacted-result-url:3a0db3e41b8f>
ATLAS_DEFLECTION_REQUEST_ID=content-ops-[redacted:3a0db3e41b8f]
```

## Result

Status: not accepted as a production portfolio proof.

| Check | Result | Notes |
|---|---|---|
| Hosted ATLAS submit | Passed | Multipart submit returned `200` with `portfolio_deflection_submit`. |
| Hosted ATLAS snapshot | Passed | Snapshot endpoint returned `200`. |
| Hosted ATLAS unpaid artifact | Passed | Artifact endpoint returned `403` before payment. |
| Production portfolio result page | Failed | Result URL returned `404`; required page hooks were absent. |

The result-page smoke reported:

```json
{
  "ok": false,
  "page": {"status": 404},
  "snapshot": {"status": 200},
  "artifact": {"status": 403},
  "errors": [
    "portfolio result page returned status 404",
    "portfolio result page missing marker: data-atlas-deflection-result",
    "portfolio result page missing marker: data-atlas-deflection-request-id",
    "portfolio result page missing marker: data-atlas-deflection-unlock",
    "portfolio result page missing marker: content_ops_deflection_report",
    "portfolio result page missing marker: request_id",
    "portfolio result page missing marker: account_id",
    "portfolio result page missing unlock CTA element",
    "portfolio result page missing request_id value",
    "portfolio result page missing account_id value"
  ]
}
```

## Diagnosis

`https://juancanfield.com` is the Vercel production deployment for
`canfieldjuan/atlas-portfolio`, project `atlas-portfolio`, root directory
`web`. That production Next app is separate from this ATLAS repository's
`portfolio-ui` Vite app.

The production host currently returns `404` for both:

```text
https://juancanfield.com/services/faq-deflection
https://juancanfield.com/services/faq-deflection/results/{request_id}?account_id={account_id}
```

The related `canfieldjuan/atlas-portfolio` PR #160 preview is deployed, but its
preview URL returned `401` to this smoke, so it is not usable as an unauthenticated
hosted proof from the ATLAS validation harness.

## Interpretation

The ATLAS backend handoff is healthy for this request: the report exists, the
snapshot is readable, and the paid artifact is locked before payment. The
customer-facing production proof is blocked because the production portfolio
deployment does not serve the expected result route.

The next product action is in `canfieldjuan/atlas-portfolio`: land or expose the
result route on the production portfolio host, then rerun this same smoke using
the production result URL.

## Artifacts

Local artifacts from this run:

| Artifact | Path |
|---|---|
| Submit result | `tmp/faq_deflection_portfolio_hosted_smoke_20260530/submit-result.json` |
| Production result-page smoke | `tmp/faq_deflection_portfolio_hosted_smoke_20260530/result-page-result.json` |
| PR #160 preview result-page smoke | `tmp/faq_deflection_portfolio_hosted_smoke_20260530/result-page-preview-result.json` |

## Verification

- Submit smoke: passed with `ok: true`, `submit.status: 200`,
  `snapshot.status: 200`, and `artifact.status: 403`.
- Production result-page smoke: failed with `page.status: 404` while ATLAS
  snapshot/artifact checks still returned `200`/`403`.
- Vercel inspection: `juancanfield.com` resolves to project `atlas-portfolio`,
  root directory `web`.
