# PR-Stripe-Webhook-Secret-Rotation

## Why this slice exists

Paid-unlock live validation needs Stripe test-mode Checkout to unlock a real
deflection report through the same webhook trust path as production. Stripe live
and test webhook endpoints use different `whsec_` signing secrets, but ATLAS
currently verifies `/webhooks/stripe` against one configured secret. Replacing
the live secret for a test run would make live unlock fragile, so ATLAS needs to
accept an explicit rotation set before the test-mode paid-unlock smoke can run.

## Scope (this PR)

Ownership lane: ai-content-ops/faq-deflection-paid-unlock
Slice phase: Production hardening

1. Allow `ATLAS_SAAS_STRIPE_WEBHOOK_SECRET` to contain a comma-separated list of
   Stripe webhook signing secrets.
2. Keep fail-closed behavior: missing/invalid signatures still reject before DB
   work.
3. Add focused fixtures proving fallback secret acceptance and all-secret
   rejection.

### Files touched

- `atlas_brain/api/billing.py`
- `tests/test_atlas_billing_stripe_hardening.py`
- `plans/PR-Stripe-Webhook-Secret-Rotation.md`

## Mechanism

The webhook route will parse the configured webhook secret string into ordered
non-empty candidates. Signature verification tries each candidate with
`stripe.Webhook.construct_event(...)` and accepts the first match. If no
candidate matches, the route returns the existing `400 Invalid signature`
response.

The existing Content Ops deflection Stripe paid workflow already enrolls the
Stripe hardening test file in both PR/push path filters and the pytest run-step,
satisfying the atlas_brain CI enrollment rule for the new fixtures in that file.

## Intentional

- No new environment variable: keeping one comma-separated
  `ATLAS_SAAS_STRIPE_WEBHOOK_SECRET` value avoids changing deployment surfaces
  and supports normal secret rotation as well as live/test validation.
- No unsigned or privileged unlock path: paid unlock remains gated only by a
  Stripe-signed `checkout.session.completed` webhook.
- No Stripe API calls in tests: fixtures mock the transport boundary at the
  Stripe module and exercise the route's real verification loop.

## Deferred

- Run the hosted test-mode paid-unlock smoke after this PR lands and the test
  webhook secret plus test report price are provisioned.
- Parked hardening: none.

## Verification

- python -m pytest tests/test_atlas_billing_stripe_hardening.py -q
- python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py -q
- bash scripts/local_pr_review.sh --allow-dirty

## Estimated diff size

| Area | Estimate |
| --- | ---: |
| Code + tests + plan | ~170 |
| **Total** | **~170** |
