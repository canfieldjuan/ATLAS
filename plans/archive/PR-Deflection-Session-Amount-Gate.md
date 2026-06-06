# PR-Deflection-Session-Amount-Gate

## Why this slice exists

Portfolio issue #194 needs multiple FAQ deflection report prices and price
experiments, but the current ATLAS webhook gate is a one-number floor:
`amount_total >= stripe_content_ops_deflection_report_amount_cents`. That lets
higher accidental prices unlock and makes lower prices fail unless ATLAS config
is changed in lockstep before traffic reaches the lower price.

The ATLAS money-safety gate should come first. This slice changes the verified
Stripe webhook path to exact amount matching against a configured allowlist,
while preserving the existing single `$1,500` behavior by default.

## Scope (this PR)

Ownership lane: ai-content-ops/faq-deflection-monetization

Slice phase: Production hardening

1. Add a typed ATLAS SaaS setting for comma-separated allowed deflection report
   Checkout amounts in cents.
2. Change the deflection webhook amount gate from `actual >= floor` to
   `actual in allowed_amounts`.
3. Keep existing deployments compatible: when the new allowlist is empty, the
   existing `stripe_content_ops_deflection_report_amount_cents` setting remains
   the only allowed amount.
4. Fail closed on malformed, empty, non-positive, missing, lower, higher, or
   wrong-currency Checkout sessions.
5. Update the checkout handoff docs/tests so they describe exact allowlisted
   amounts instead of the retired floor.

### Files touched

- `atlas_brain/config.py`
- `atlas_brain/api/billing.py`
- `docs/frontend/content_ops_faq_deflection_checkout_contract.md`
- `scripts/smoke_content_ops_deflection_stripe_paid_unlock.py`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py`
- `plans/PR-Deflection-Session-Amount-Gate.md`

## Mechanism

`SaaSAuthConfig` gains
`stripe_content_ops_deflection_report_allowed_amount_cents`, mapped by the
existing config system to
`ATLAS_SAAS_STRIPE_CONTENT_OPS_DEFLECTION_REPORT_ALLOWED_AMOUNT_CENTS`.

The webhook helper resolves allowed amounts as:

```text
allowed_amounts = parsed allowlist when configured
allowed_amounts = (legacy configured amount,) otherwise
```

Malformed allowlists fail closed by returning no allowed amounts. The paid
unlock path still checks signed Stripe webhook verification, one-time payment
mode, `payment_status == "paid"`, required metadata, UUID-shaped `account_id`,
currency, and report ownership before marking the report paid. This slice only
replaces the floor comparison with exact membership in the resolved server-side
amount set.

## Intentional

- The legacy `stripe_content_ops_deflection_report_amount_cents` setting stays
  in place as the default single-price configuration and compatibility fallback.
- This does not add portfolio variant routing or metadata stamping; that is the
  next repo's slice after ATLAS can safely honor multiple exact amounts.
- No Stripe API call is added to the webhook handler; the gate uses fields
  already present on the signed `checkout.session.completed` event.
- Higher-than-configured amounts are rejected too. Overcharging should surface
  as bad pricing configuration rather than silently unlocking.
- Local review's cross-layer caller hint for `_validate_args` was inspected.
  It is a same-name false positive across unrelated scripts/tests; the touched
  smoke test covers the actual module-local function changed here.

## Deferred

- Portfolio issue #194 follow-up: variant selection and single source of truth
  for buyer-facing price copy in `canfieldjuan/atlas-portfolio`.
- Future hardening if needed: bind variants to signed metadata or Stripe Price
  IDs after the portfolio variant contract exists.
- Parked hardening: none.

## Verification

- `python -m py_compile atlas_brain/api/billing.py atlas_brain/config.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_content_ops_faq_report_contract_docs.py` -- passed with no output.
- `python -m py_compile scripts/smoke_content_ops_deflection_stripe_paid_unlock.py tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py` -- passed with no output.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_content_ops_faq_report_contract_docs.py tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py -q` -- 40 passed, 1 warning.
- `python -m pytest tests/test_atlas_billing_stripe_hardening.py tests/test_b2b_vendor_briefing.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py tests/test_mcp_content_ops_deflection_readonly.py -q` -- 87 passed, 1 warning.
- `bash scripts/run_extracted_pipeline_checks.sh` -- package checks passed; 3058 passed, 10 skipped, 1 warning.
- `bash scripts/local_pr_review.sh --allow-dirty --current-pr-body-file /tmp/atlas-deflection-session-amount-gate-body.md` -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Billing/config gate | ~45 |
| Focused tests | ~85 |
| Smoke script/test | ~10 |
| Checkout contract doc/test | ~15 |
| Plan doc | ~85 |
| **Total** | **~240** |

Under the 400 LOC soft cap.
