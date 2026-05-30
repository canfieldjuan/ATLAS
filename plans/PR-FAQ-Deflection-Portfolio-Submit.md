# PR-FAQ-Deflection-Portfolio-Submit

## Why this slice exists

Issue #1161 closed the portfolio integration questions and made one product
decision explicit: the portfolio should hand ATLAS a support-ticket CSV blob
reference, and ATLAS should fetch, normalize, execute, persist, and return the
locked deflection report response. Without this seam the portfolio must either
duplicate ATLAS support-ticket CSV normalization or cannot route a buyer to a
real `{request_id}` results page.

This slice exceeds the 400 LOC target because the endpoint is not useful
without the contract doc, bounded server-side blob fetch, SSRF guard fixtures,
URL/size/input failure fixtures, and extracted-runner enrollment. Splitting
those out would leave a public integration endpoint without its safety and
contract proof.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection
Slice phase: Vertical slice

1. Add `POST /content-ops/deflection-reports/submit` to the existing Content
   Ops router.
2. Accept the portfolio handoff body from #1161: `blob_url`,
   `support_platform`, `company_name`, and `contact_email`, with optional
   `limit`.
3. Fetch only HTTPS blob URLs through a bounded server-side read that rejects
   private DNS targets and redirects, parse CSV rows through the existing
   source-row loader, and run the existing synchronous `faq_deflection_report`
   execute path.
4. Return the same gated execute shape as `/execute`: `request_id`, snapshot,
   and locked `full_report`; no paid artifact leaks.
5. Document the endpoint contract and caps for the portfolio.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Submit.md`
- `docs/frontend/content_ops_faq_deflection_checkout_contract.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_deflection_submit.py`
- `scripts/run_extracted_pipeline_checks.sh`

## Mechanism

The new route lives in `create_content_ops_control_surface_router`, so it
inherits the same host auth dependency, tenant scope, execution-services
provider, deflection store, usage summary, and paid-gate behavior as `/execute`.
The route validates the URL shape, rejects localhost/private/link-local DNS
targets before fetch, blocks redirects instead of following them, downloads the
referenced CSV into a temporary file with a byte cap, loads rows with
`load_source_rows_from_file(file_format="csv")`, rejects empty or wholly
un-normalizable inputs, then builds an internal execute payload:

```json
{
  "outputs": ["faq_deflection_report"],
  "limit": 1000,
  "require_quality_gates": false,
  "inputs": {
    "source_material": [...],
    "company_name": "...",
    "contact_email": "...",
    "support_platform": "zendesk"
  }
}
```

The route delegates execution to the same internal flow that already persists
deflection artifacts and replaces the full artifact with the locked paid-gate
result.

## Intentional

- This is sync, matching the currently shipped execute seam. A `200` means the
  report was persisted and the returned `request_id` can hydrate `/snapshot`
  immediately for the same authenticated account.
- The endpoint accepts HTTPS blob URLs only. Public Vercel Blob works now;
  signed/private URLs can use the same field without changing the ATLAS API.
- DNS is preflighted before `urllib` connects and redirects are blocked. A
  full DNS-rebinding defense would require pinning the validated IP during
  connection construction; that is beyond this vertical slice.
- The route caps parsed rows at the same FAQ sync execute cap. It does not add
  a background ingestion path in this vertical slice.
- The route does not create Stripe Checkout or mark reports paid. The existing
  webhook remains the only customer payment trust boundary.

## Deferred

- Parked hardening: none.
- Async/background blob execution for exports beyond the sync row cap remains a
  future robust-testing / production-hardening slice.
- Private Vercel Blob signing is a portfolio-side follow-up from issue #1161;
  this endpoint can consume those URLs once the portfolio sends them.

## Verification

- `python -m pytest tests/test_extracted_content_deflection_submit.py -q` - 8 passed.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_paid_flow.py tests/test_extracted_content_deflection_submit.py -q` - 9 passed, 1 warning.
- `python -m pytest tests/test_extracted_content_control_surface_api.py -q` - 123 passed, 1 skipped.
- `python -m py_compile extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_deflection_submit.py` - passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/faq-deflection-portfolio-submit-pr-body.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~117 |
| Contract doc | ~37 |
| Router implementation | ~375 |
| Tests | ~343 |
| CI runner enrollment | ~1 |
| **Total** | **~873** |
