# PR-FAQ-Deflection-Hosted-Submit-Handoff

## Why this slice exists

#1166 merged the ATLAS-side portfolio blob-submit seam, and #1161 confirmed
the portfolio will build against that execute-envelope response. The remaining
handoff gap is a repeatable hosted proof that the deployed ATLAS API can accept
a signed/private CSV blob URL, return a persisted `request_id`, hydrate the
free snapshot immediately, and keep the paid artifact locked before Stripe
webhook release.

This slice does not claim that hosted proof locally because the required
deployment inputs are not present in this checkout (`ATLAS_API_BASE_URL`,
`ATLAS_B2B_JWT`/`ATLAS_TOKEN`, matching `account_id`, and a signed blob URL).
Instead it ships the fail-closed smoke command and runbook that will produce
that proof as soon as the inputs are provisioned. The diff exceeds the 400 LOC
target because this is a checker/smoke surface: the validation branches, result
envelope parsing, response-shape guards, redaction behavior, and failure
fixtures need to ship together so the tool cannot false-green a broken handoff
or leak signed URL / token material into artifacts.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection
Slice phase: Functional validation

1. Add a hosted `deflection-reports/submit` handoff smoke that posts the
   portfolio body to the deployed API and writes a machine-readable result
   artifact.
2. Validate the execute envelope returned by submit: top-level `request_id`,
   completed `faq_deflection_report` step, matching nested `request_id`,
   snapshot shape, locked `full_report`, submit diagnostics, and no leaked
   answer/evidence/markdown fields in the free snapshot.
3. Hydrate `GET /deflection-reports/{request_id}/snapshot` and verify it
   returns the same snapshot shape immediately.
4. Probe `GET /deflection-reports/{request_id}/artifact` and require `403`
   before payment, proving the paid trust boundary stays locked.
5. Document the hosted handoff command and add it to the extracted pipeline
   local/CI check inventory.

### Files touched

- `plans/PR-FAQ-Deflection-Hosted-Submit-Handoff.md`
- `.github/workflows/extracted_pipeline_checks.yml`
- `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`

## Mechanism

The smoke is an operator-facing Python script that loads `.env`/`.env.local`
when `python-dotenv` is installed, accepts CLI overrides, and refuses to run
   without a deployed API URL, bearer token, matching account id, HTTPS blob
   URL, company name, contact email, and support-platform value.

On a real run it sends:

```json
{
  "blob_url": "https://...",
  "support_platform": "zendesk",
  "company_name": "Acme Co.",
  "contact_email": "lead@acme.co",
  "limit": 1000
}
```

to `/api/v1/content-ops/deflection-reports/submit`, then validates the returned
execute envelope before making the snapshot and unpaid-artifact probes. The
result artifact records HTTP statuses, request id, compact submit diagnostics,
and deterministic errors without writing bearer tokens.

## Intentional

- The smoke rejects local API hosts. This is the deployed handoff proof for the
  portfolio funnel, not another localhost route check.
- The smoke does not upload a CSV itself. Portfolio owns blob creation and this
  slice verifies the ATLAS contract that consumes the signed/private blob URL.
- The smoke validates `account_id` presence for the portfolio Stripe metadata
  handoff but does not send it to submit/snapshot/artifact routes; ATLAS derives
  account scope from the bearer token.
- The smoke treats `/artifact` returning `403` as the expected unpaid state.
  Stripe webhook unlock remains a separate trust-boundary validation slice.
- Secrets are not printed or written to the result artifact; only presence and
  non-sensitive request/response metadata are recorded.

## Deferred

- Parked hardening: none. `HARDENING.md` has no active entries touching this
  endpoint or script lane.
- Actual hosted run artifact remains deferred until the deployment/config
  inputs exist: `ATLAS_API_BASE_URL`, B2B service JWT/token, matching
  `account_id`, and a signed support-ticket CSV blob URL.
- Stripe webhook paid-unlock E2E remains the next validation slice after the
  submit/snapshot/unpaid-artifact handoff is proven against the deployed host.

## Verification

- `python -m py_compile scripts/smoke_content_ops_deflection_submit_handoff.py tests/test_smoke_content_ops_deflection_submit_handoff.py` - passed.
- `python -m pytest tests/test_smoke_content_ops_deflection_submit_handoff.py -q` - 11 passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py` - passed, 137 matching tests enrolled.
- `bash scripts/check_ascii_python.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python -m pytest tests/test_extracted_content_deflection_submit.py tests/test_smoke_content_ops_deflection_submit_handoff.py -q` - 20 passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - passed, 2820 passed, 10 skipped, 1 warning.
- Live hosted submit run: not run. This checkout has no
  `ATLAS_API_BASE_URL`, `ATLAS_B2B_JWT`/`ATLAS_TOKEN`, matching `account_id`,
  or signed `ATLAS_DEFLECTION_SUBMIT_BLOB_URL`; the new smoke exits before
  network calls until those inputs are provisioned.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 123 |
| Smoke script | 531 |
| Smoke tests | 377 |
| Runbook | 83 |
| CI/check enrollment | 3 |
| **Total** | **1117** |
