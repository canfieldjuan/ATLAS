# PR-FAQ-Deflection-Result-Page-CTA-Metadata-Binding

## Why this slice exists

PR #1176 shipped the hosted portfolio result-page smoke, but review caught a
false-green gap before the follow-up fix could be pushed: the page check proved
the expected `request_id` and `account_id` appeared somewhere in the HTML, not
that the unlock CTA itself carried those values in its Stripe Checkout metadata.

This slice closes that exact review finding on top of the merged result-page
smoke so stale or missing unlock CTA metadata cannot pass the hosted validation.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Functional validation

1. Parse the hosted result-page HTML with a stdlib HTML parser to find the
   `data-atlas-deflection-unlock` element.
2. Require that element to carry the exact expected Checkout metadata values:
   `data-checkout-source`, `data-checkout-request_id`, and
   `data-checkout-account_id`.
3. Add a regression where the correct values exist elsewhere on the page but
   stale values live on the unlock CTA.

### Files touched

- `plans/PR-FAQ-Deflection-Result-Page-CTA-Metadata-Binding.md`
- `scripts/smoke_content_ops_deflection_portfolio_result_page.py`
- `tests/test_smoke_content_ops_deflection_portfolio_result_page.py`

## Mechanism

`UnlockCtaParser` subclasses `html.parser.HTMLParser` and captures the first
start tag with `data-atlas-deflection-unlock`. `_page_errors` keeps the existing
page-level marker checks, then validates the Checkout metadata directly on that
captured element.

## Intentional

- The parser remains stdlib-only so the smoke stays lightweight and does not
  add a runtime dependency to operator machines.
- This PR does not change the runbook or CI enrollment; #1176 already enrolled
  the same smoke and test file.

## Deferred

- Parked hardening: none.
- Stripe webhook paid-unlock hosted E2E remains the next validation step after
  the pre-payment result-page smoke passes against the hosted portfolio page.

## Verification

- py_compile for the touched smoke script and test file - passed.
- Focused pytest for `tests/test_smoke_content_ops_deflection_portfolio_result_page.py` - 8 passed.
- Full extracted pipeline check wrapper - passed; `extracted_reasoning_core` 295 passed, and `extracted_content_pipeline` 2846 passed, 10 skipped, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 65 |
| Smoke script | 30 |
| Tests | 25 |
| **Total** | **120** |
