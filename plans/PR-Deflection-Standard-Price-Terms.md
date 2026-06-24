# PR-Deflection-Standard-Price-Terms

## Why this slice exists

atlas-portfolio issue #194 now targets an easier standard-price change path:
ATLAS should remain the source of truth for the active deflection report charge,
while portfolio stops carrying a separate public amount that can drift from the
authorized checkout terms. The first ATLAS slice needs a small read contract the
portfolio server can consume before any portfolio display or preflight work
starts.

## Scope (this PR)

Ownership lane: deflection/stripe-monetization
Slice phase: Vertical slice

1. Add a standard deflection pricing terms endpoint on the existing Content Ops
   control-surface router: `GET /content-ops/deflection-reports/pricing/standard`.
2. Return only non-secret pricing terms: `variant`, `status`, `amount_cents`,
   and `currency`. The existing checkout authorization remains the only response
   that includes the Stripe `price_id` used for charge creation.
3. Reuse the existing checkout-term validation so the public pricing terms fail
   closed when amount, allowed amount set, currency, or price ID config is
   invalid.
4. Add extracted-router and Atlas host-mount tests proving the route contract,
   fail-closed behavior, and dependency gates.

### Review Contract
- Acceptance criteria:
  - [ ] `GET /content-ops/deflection-reports/pricing/standard` returns
        `variant=standard`, `status=configured`, `amount_cents`, and
        lower-case `currency` for valid standard checkout config.
  - [ ] The terms response never exposes the Stripe `price_id`.
  - [ ] Invalid amount, currency, allowed amount, or missing price ID config
        fails closed with 503 before portfolio can display the terms.
  - [ ] Existing checkout authorization remains request-specific and still
        returns the Stripe `price_id` only after report-state checks pass.
  - [ ] Atlas host-mount dependencies keep the pricing terms route behind the
        public deflection auth/rate-limit gate.
- Affected surfaces: API / auth / config / payments
- Risk areas: backcompat / security / config drift / payment mismatch
- Reviewer rules triggered: R1, R2, R3, R5, R10, R11, R14

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `plans/PR-Deflection-Standard-Price-Terms.md`
- `tests/test_atlas_content_ops_generated_assets_api.py`
- `tests/test_extracted_content_control_surface_api.py`

## Mechanism

The existing checkout terms helper already centralizes the standard checkout
config guards: positive amount, configured currency, configured Stripe price ID,
and exact allowed-amount membership. This slice keeps that as the authoritative
validator, then projects the validated terms into a non-secret payload:

```json
{
  "variant": "standard",
  "status": "configured",
  "amount_cents": 150000,
  "currency": "usd"
}
```

Because the projection calls the checkout validator first, a missing `price_id`
or stale allowed amount makes the read endpoint return the same 503 class as
checkout authorization instead of letting portfolio display a price ATLAS cannot
charge or unlock.

## Intentional

- The endpoint does not expose `price_id`. Portfolio still has to ask for
  request-specific checkout authorization before creating a Stripe Checkout
  session, which keeps charge creation coupled to an existing report.
- This slice adds only the standard variant path. Generic A/B, cohort routing,
  and public partner checkout remain deferred until ATLAS has variant-aware
  authorization.
- The route uses the existing public deflection dependency set. It is
  non-secret data, but the host mount still keeps the same authenticated
  Content Ops account boundary and rate limit.

## Deferred

- ATLAS config consistency preflight/smoke proving the full `PRICE_ID`,
  `AMOUNT_CENTS`, `ALLOWED_AMOUNT_CENTS`, `CURRENCY`, Stripe Checkout amount,
  and webhook unlock loop agree.
- Portfolio consumption of the ATLAS terms endpoint for standard price display.
- Portfolio checkout/preflight mismatch refusal once display is sourced from
  ATLAS.
- Runbook documentation for the operator change-price procedure.

Parked hardening: none.

## Verification

- `pytest tests/test_extracted_content_control_surface_api.py -k "deflection_standard_pricing_terms or deflection_checkout_authorization or deflection_report_routes_use_public_and_trusted_dependencies"` -- passed, 21 selected.
- `pytest tests/test_atlas_content_ops_generated_assets_api.py -k "public_deflection_routes_use_rate_limit_gate"` -- passed, 1 selected.
- `bash scripts/validate_extracted_content_pipeline.sh` -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `bash scripts/check_ascii_python.sh` -- passed.
- `python scripts/sync_pr_plan.py plans/PR-Deflection-Standard-Price-Terms.md` -- updated the plan from the real diff.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 19 |
| `plans/PR-Deflection-Standard-Price-Terms.md` | 114 |
| `tests/test_atlas_content_ops_generated_assets_api.py` | 1 |
| `tests/test_extracted_content_control_surface_api.py` | 105 |
| **Total** | **239** |
