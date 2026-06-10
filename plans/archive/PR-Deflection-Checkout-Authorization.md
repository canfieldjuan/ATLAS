# PR-Deflection-Checkout-Authorization

## Why this slice exists

#1386 identifies the paid funnel's root money-path risk: portfolio can create a
Stripe Checkout session using its own report, price, and currency assumptions,
then ATLAS can fail closed after payment because it independently validates
stricter facts. The structural fix is "authorize before charge": ATLAS must be
the source of truth before any Checkout Session is created.

This slice adds the ATLAS-side authorization contract only. It gives the
portfolio repo a single backend preflight to call before it talks to Stripe.
The final diff is above the 400 LOC soft cap because review surfaced a
payment-boundary contract gap: the authorization amount also has to reconcile
with the webhook's accepted amount set before the portfolio repo can safely
consume this contract.

## Scope (this PR)

Ownership lane: deflection/go-live
Slice phase: Production hardening

1. Add a typed ATLAS deflection report Stripe Price ID setting.
2. Add an authenticated ATLAS route that authorizes Checkout only when the
   report exists for the scoped account, is still unpaid/locked, has a full
   artifact, and checkout amount/currency/price config is complete.
3. Return canonical `{amount_cents, currency, price_id}` from ATLAS so the
   portfolio route can build Stripe Checkout without its own price/currency
   judgment.
4. Reconcile the returned amount with the same allowed amount list the Stripe
   webhook uses, so authorization cannot approve a charge amount the unlock
   gate will later reject.
5. Add focused negative tests proving missing, paid, artifactless,
   amount-mismatched, and misconfigured states fail before a charge can be
   attempted.

### Files touched

- `atlas_brain/api/__init__.py`
- `atlas_brain/config.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `plans/PR-Deflection-Checkout-Authorization.md`
- `tests/test_extracted_content_control_surface_api.py`

### Review Contract

Acceptance criteria:
- `POST /content-ops/deflection-reports/{request_id}/checkout-authorization`
  returns 200 only for a scoped report that exists, is unpaid, has an artifact,
  and has non-empty configured `amount_cents`, `currency`, and `price_id`.
- `amount_cents` must be a member of the configured allowed amount set when an
  allow-list exists; otherwise the default amount remains the single accepted
  amount.
- Missing report returns 404.
- Already-paid report returns 409.
- Missing/empty artifact returns 409.
- Missing price ID / invalid amount / invalid currency / malformed allowed
  amount list / amount not accepted by the payment gate returns 503.
- Response contains no full report artifact, ticket text, delivery email, or
  Stripe secret.

Affected surfaces:
- Content Ops deflection report access API.
- ATLAS typed SaaS/Stripe config for the one-time deflection report.
- Future portfolio checkout handoff.

Risk areas:
- This is a payment boundary; fail-open behavior would allow pay-but-locked
  incidents to continue.
- The route must not expose the paid artifact or buyer email while authorizing
  checkout.
- The portfolio repo still needs a follow-up PR to call this route before
  creating Stripe Checkout.

Reviewer rules triggered: R1 Requirements match; R2 Test evidence; R3 Security/auth; R5 Backward compatibility; R8 Idempotency/payment state; R10 Maintainability; R11 Dependencies/config; R12 Deployment safety.

## Mechanism

Add `stripe_content_ops_deflection_report_price_id` to `SaaSAuthConfig`, then
pass the deflection checkout terms from `atlas_brain/api/__init__.py` into
`ContentOpsControlSurfaceApiConfig`. The existing
`stripe_content_ops_deflection_report_allowed_amount_cents` setting is passed
through the same config so the authorization route can reject any
`amount_cents` that the webhook's amount gate would not accept.

The new route resolves the existing `DeflectionReportArtifactStore` and scoped
account, loads the access record, and checks only lock/availability state. If
authorization succeeds, it returns a compact payload:

```json
{
  "request_id": "...",
  "status": "authorized",
  "checkout": {
    "amount_cents": 150000,
    "currency": "usd",
    "price_id": "price_..."
  }
}
```

The route does not call Stripe. It is the pre-payment truth contract the
portfolio checkout route can consume in the next repo slice.

## Intentional

- This PR does not edit `portfolio-ui/api/content-ops/deflection/checkout.js`;
  that is the next #1386 integration slice after the ATLAS contract exists.
- This PR does not constrain Stripe `payment_method_types`; Stripe guidance
  favors Checkout Sessions with dynamic payment methods. Async fulfillment
  remains a separate #1386 webhook slice.
- The route fails closed when no ATLAS price ID is configured. A portfolio
  fallback `price_data` path would preserve the two-decider class this issue is
  trying to remove.
- `price_id` is the authoritative charge input for the future portfolio
  Checkout Session. `amount_cents` is still returned for UI/display and
  contract checks, but this route now verifies that value against the same
  accepted amount set the webhook uses.

## Deferred

- #1386 portfolio slice: call this authorization route before creating Stripe
  Checkout and build the session from ATLAS-returned terms only.
- #1386 webhook slice: fulfill delayed-payment success events and make amount /
  currency / existence mismatches paid-funnel incidents.
- #1386 delivery slice: reconcile result-page URL and delivery scheduling.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_content_control_surface_api.py::test_deflection_checkout_authorization_returns_canonical_terms_only tests/test_extracted_content_control_surface_api.py::test_deflection_checkout_authorization_fails_closed_for_report_state tests/test_extracted_content_control_surface_api.py::test_deflection_checkout_authorization_fails_when_terms_misconfigured tests/test_extracted_content_control_surface_api.py::test_deflection_report_paid_route_uses_trusted_dependency -q`
  - 12 passed.
- `python -m pytest tests/test_extracted_content_control_surface_api.py -q`
  - 143 passed, 1 skipped.
- `bash scripts/validate_extracted_content_pipeline.sh`
  - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt`
  - passed.
- `bash scripts/check_ascii_python.sh`
  - passed.
- `git diff --check`
  - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/__init__.py` | 15 |
| `atlas_brain/config.py` | 4 |
| `extracted_content_pipeline/api/control_surfaces.py` | 106 |
| `plans/PR-Deflection-Checkout-Authorization.md` | 156 |
| `tests/test_extracted_content_control_surface_api.py` | 180 |
| **Total** | **461** |
