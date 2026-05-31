# PR-FAQ-Deflection-Portfolio-Hosted-Smoke

## Why this slice exists

The portfolio result page now exposes the machine-readable Checkout metadata
that `scripts/smoke_content_ops_deflection_portfolio_result_page.py` validates.
The remaining handoff question is whether the production portfolio URL can read
a fresh unpaid ATLAS report and keep the artifact locked before payment.

This slice records that hosted proof, or the exact hosted blocker, using the
existing submit and result-page smokes. It closes the deploy-verification
deferred item from `PR-FAQ-Deflection-Portfolio-Live-E2E-Smoke`.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection
Slice phase: Functional validation

1. Generate a fresh unpaid deflection report through the hosted ATLAS submit
   smoke using the checked SaaS demo CSV fixture.
2. Validate the production portfolio result URL for that `request_id` and
   account id with the existing result-page smoke.
3. Add a redacted validation artifact that names the command, statuses,
   request id, and any blocker without recording bearer tokens or CSV contents.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Hosted-Smoke.md`
- `docs/extraction/validation/content_ops_faq_deflection_portfolio_hosted_smoke_2026-05-30.md`

## Mechanism

The run uses the existing hosted smoke chain:

```bash
python scripts/smoke_content_ops_deflection_submit_handoff.py ...
python scripts/smoke_content_ops_deflection_portfolio_result_page.py ...
```

The result page URL is built from the production portfolio host:

```text
https://juancanfield.com/services/faq-deflection/results/{request_id}?account_id={account_id}
```

The result-page smoke then checks the hosted HTML hooks, Checkout metadata
values, ATLAS snapshot `200`, and ATLAS artifact `403` before payment.

## Intentional

- This does not mark the report paid; the slice is the pre-payment portfolio
  result-page proof.
- This does not add another smoke script; the existing submit and result-page
  smokes are the source of truth.
- The validation document redacts credentials and records only non-secret run
  metadata.

## Deferred

- Parked hardening: none. The same-lane `HARDENING.md` entry targets the Intel
  report UI lane, not the production portfolio result-page proof.
- Stripe paid-unlock rendering through the production portfolio page remains a
  follow-up after this pre-payment hosted smoke is recorded.

## Verification

- `python scripts/smoke_content_ops_deflection_submit_handoff.py --csv-file extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --company-name "Atlas SaaS Demo" --contact-email ops@example.com --support-platform zendesk --limit 1000 --output-result tmp/faq_deflection_portfolio_hosted_smoke_20260530/submit-result.json --json` - passed; submit `200`, snapshot `200`, unpaid artifact `403`.
- `python scripts/smoke_content_ops_deflection_portfolio_result_page.py --output-result tmp/faq_deflection_portfolio_hosted_smoke_20260530/result-page-result.json --json` - failed as expected for this blocker artifact; production portfolio page returned `404` while ATLAS snapshot/artifact stayed `200`/`403`.
- `vercel inspect https://juancanfield.com` from `atlas-portfolio/web` - confirmed production host maps to Vercel project `atlas-portfolio`, root directory `web`.
- `python scripts/smoke_content_ops_deflection_portfolio_result_page.py --output-result tmp/faq_deflection_portfolio_hosted_smoke_20260530/result-page-preview-result.json --json` against the `atlas-portfolio` PR #160 preview - failed with preview `401`, so the preview is not usable as an unauthenticated hosted proof.
- `python -m pytest tests/test_smoke_content_ops_deflection_portfolio_result_page.py -q` - 8 passed.
- `python -m json.tool` over the three local smoke artifacts - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/faq-deflection-portfolio-hosted-smoke-pr-body.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 83 |
| Validation artifact | 140 |
| **Total** | **223** |

Actual diff is 2 files, +223 / -0.
