# PR-FAQ-Deflection-Portfolio-Result-Page-Validation

## Why this slice exists

The deflection/Stripe lane now has the ATLAS submit seam, the checkout
contract, the hosted submit smoke, the env helper, and the blob-fetch hardening.
The remaining handoff gap is the portfolio result page: operators need one
repeatable hosted check that proves the page points at the same ATLAS
`request_id`, renders the free snapshot state, leaves the artifact locked before
payment, and preserves the Stripe metadata handoff contract.

This slice ships the validation harness without claiming a live run, because
the hosted portfolio URL, ATLAS bearer token, and generated `request_id` are
operator-provided values and should not be committed or pasted into chat.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Functional validation

1. Add a hosted portfolio result-page smoke that validates the page and ATLAS
   API contract from operator-provided URLs/token/request id.
2. Fail closed on missing or unsafe hosted inputs before any network call.
3. Require stable portfolio page validation hooks for request id, unlock CTA,
   and checkout metadata fields.
4. Enroll the smoke and tests in the extracted pipeline checks.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Result-Page-Validation.md`
- `.github/workflows/extracted_pipeline_checks.yml`
- `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/smoke_content_ops_deflection_portfolio_result_page.py`
- `tests/test_smoke_content_ops_deflection_portfolio_result_page.py`

## Mechanism

The smoke loads `.env`/`.env.local` when `python-dotenv` is installed and
accepts CLI overrides. On a real run it fetches the hosted portfolio result
page, then calls ATLAS snapshot and artifact endpoints for the same
`request_id`.

The page check is string-level by design so it can run in lightweight CI and
against static hosted output: it requires the request id, unlock CTA hook,
and checkout metadata keys/values to appear in the rendered HTML. The API check
requires snapshot `200` with no paid-report fields and artifact `403` before
payment.

## Intentional

- This is a validation harness, not a portfolio UI implementation.
- The smoke rejects localhost and non-HTTPS URLs because it proves hosted
  portfolio/ATLAS handoff behavior.
- The smoke does not create Stripe Checkout or mark reports paid; it validates
  the pre-payment result-page state and metadata contract only.
- Live run remains operator-driven because the required URL/token/request id
  values are deployment secrets or generated runtime values.

## Deferred

- Parked hardening: none.
- Live hosted portfolio result-page run remains deferred until the operator
  provides `ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL`, `ATLAS_API_BASE_URL`,
  bearer token, and `ATLAS_DEFLECTION_REQUEST_ID`.
- Stripe webhook paid-unlock hosted E2E remains the next validation step after
  the portfolio result page passes in pre-payment state.

## Verification

- py_compile for the new smoke script and test file - passed.
- Focused pytest for `tests/test_smoke_content_ops_deflection_portfolio_result_page.py` - 7 passed.
- Extracted pipeline CI-enrollment audit - `OK: 138 matching tests are enrolled.`
- Full extracted pipeline check wrapper - passed; `extracted_reasoning_core` 295 passed, and `extracted_content_pipeline` 2845 passed, 10 skipped, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 88 |
| Smoke script | 323 |
| Tests | 266 |
| Runbook/check enrollment | 26 |
| **Total** | **703** |

The diff exceeds the 400 LOC target because this is a validation/checker
surface. The fail-closed preflight branches, redaction/result artifact behavior,
page-contract checks, and API-envelope checks need to ship together.
