# PR-FAQ-Deflection-Portfolio-Smoke-Configured-Account

## Why this slice exists

PR-FAQ-Deflection-Result-Configured-Account removed `account_id` from the public
FAQ deflection result URL and browser Checkout payload. The hosted portfolio
result-page smoke still validates the older contract by requiring the account id
marker, the `data-checkout-account_id` attribute, and the raw account id value in
customer-facing HTML.

That makes the smoke stale exactly where it should be the live guard for the
configured-account handoff. This slice updates the smoke to prove the result
page is account-less in the browser while still using the operator-provided
account id for server-side ATLAS snapshot/artifact verification.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Functional validation

1. Remove the hosted portfolio result-page smoke's requirement that customer
   HTML contains `account_id`.
2. Keep the smoke's `--account-id` input for ATLAS API validation and result
   artifact traceability.
3. Fail the smoke when the hosted page exposes the configured account id through
   the result URL, unlock CTA, or page body.
4. Update focused smoke tests so fixtures match the merged configured-account
   producer output.

### Files touched

- `plans/PR-FAQ-Deflection-Portfolio-Smoke-Configured-Account.md`
- `scripts/smoke_content_ops_deflection_portfolio_result_page.py`
- `tests/test_smoke_content_ops_deflection_portfolio_result_page.py`

## Mechanism

`REQUIRED_PAGE_MARKERS` stops listing `account_id`, and `_page_errors()` keeps
validating the unlock CTA's source and `request_id` binding. It then fails
closed if the page carries either the legacy `data-checkout-account_id`
attribute or the configured account id value anywhere in the HTML.

Argument validation also rejects legacy public result URLs whose path or query
string carries `account_id` or the configured account id value, so a smoke run
cannot fetch an account-less page body from an account-bearing browser URL and
return a false green.

The CLI contract still requires `--account-id` because the smoke directly checks
the ATLAS snapshot and locked artifact endpoints for that tenant. The difference
is that the tenant binding is no longer expected to be visible to the browser.

## Intentional

- This does not remove the `--account-id` argument; the hosted smoke still needs
  it to validate ATLAS API state for the tenant.
- This does not change the portfolio implementation. PR #1208 already changed
  the producer; this slice aligns the hosted validation guard with that
  contract.
- The result payload can still record `inputs.account_id`; it is a local
  operator artifact, not customer-facing page output.
- Local review's cross-layer hints are same-name `_validate_args` helpers in
  unrelated scripts, not callers of this smoke module.

## Deferred

- A live hosted rerun with current production credentials remains an operator
  validation step after this smoke contract lands.
- Parked hardening: none.

## Verification

- Python compile check for the smoke script and focused smoke test - passed.
- Focused pytest for `tests/test_smoke_content_ops_deflection_portfolio_result_page.py` - 10 passed.
- Local PR review with the prepared PR body file - passed after the review fix.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 85 |
| Smoke contract update | 35 |
| Focused tests | 55 |
| **Total** | **175** |

Under the 400 LOC soft cap.
