# PR-FAQ-Deflection-Stripe-Paid-Unlock-Smoke

## Why this slice exists

The deflection/Stripe lane now proves the hosted submit handoff, the portfolio
result page, and the unlock CTA metadata binding. The remaining validation gap
is the Stripe trust path itself: after Checkout completes, ATLAS must accept a
signed `checkout.session.completed` webhook, mark the matching deflection
report paid, and release the full artifact through the existing artifact route.

This slice ships an operator-driven hosted smoke for that paid-unlock path
without committing webhook secrets or requiring live Stripe CLI state in CI.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Functional validation

1. Add a stdlib hosted smoke that signs a Stripe-compatible
   `checkout.session.completed` payload with the operator-provided webhook
   secret.
2. Fail closed before network calls when the hosted API URL, bearer token,
   webhook secret, account id, or request id is missing or unsafe.
3. Prove the paid-unlock sequence: artifact `403` before webhook, webhook `200`
   with `status: "ok"`, artifact `200` with full paid report fields after.
4. Enroll the smoke and test file in the extracted pipeline checks.

### Files touched

- `plans/PR-FAQ-Deflection-Stripe-Paid-Unlock-Smoke.md`
- `.github/workflows/extracted_pipeline_checks.yml`
- `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/smoke_content_ops_deflection_stripe_paid_unlock.py`
- `tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py`

## Mechanism

The smoke loads `.env`/`.env.local` when `python-dotenv` is installed, validates
hosted inputs, fetches the artifact route before the webhook to confirm it is
still locked, posts a locally signed Stripe event to `/webhooks/stripe`, then
fetches the artifact route again to confirm the full report is unlocked.

The payload uses the Checkout metadata contract from
`docs/frontend/content_ops_faq_deflection_checkout_contract.md`:
`source=content_ops_deflection_report`, `account_id`, and `request_id`.

## Intentional

- This is a hosted validation harness, not a Checkout-session creator.
- The script signs the event locally with the webhook secret rather than
  depending on Stripe CLI availability in CI or operator machines.
- The smoke requires the artifact to be locked before it posts the webhook so a
  previously paid report cannot produce a false-green unlock proof.
- Live execution remains operator-driven because the bearer token, webhook
  secret, account id, and generated request id are deployment/runtime values.

## Deferred

- Parked hardening: none.
- A real Stripe Checkout browser flow remains outside this slice; this proves
  the signed webhook trust path and artifact unlock contract.

## Verification

- py_compile for the new smoke script and test file - passed.
- Focused pytest for `tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py` - 7 passed.
- Extracted pipeline CI-enrollment audit - `OK: 139 matching tests are enrolled.`
- Full extracted pipeline check wrapper - passed; `extracted_reasoning_core` 295 passed, and `extracted_content_pipeline` 2853 passed, 10 skipped, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 84 |
| Smoke script | 389 |
| Tests | 255 |
| Runbook/check enrollment | 30 |
| **Total** | **758** |

The diff exceeds the 400 LOC target because this is a validation/checker
surface. The preflight gates, Stripe-signature generation, webhook posting,
artifact unlock checks, and failure-branch tests need to ship together to avoid
another false-green hosted smoke.
