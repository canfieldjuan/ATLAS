# PR-Deflection-Pricing-Variant-Authorization

## Why this slice exists

atlas-portfolio#194 is partially landed: the easy standard price-change
milestone is complete, but the remaining A/B / multiple simultaneous price path
is blocked because ATLAS still authorizes only one global deflection checkout
price. Portfolio already persists a server-bound `standard` or `partner`
variant at intake and stamps the variant into Stripe metadata, but it still has
to ask ATLAS for the actual `price_id` and amount used to charge.

Root cause: the ATLAS control-surface contract has no variant dimension for
deflection checkout terms. `_deflection_checkout_terms()` reads a single
`price_id` / amount / currency tuple from config, and the checkout authorization
route cannot choose partner-priced terms. This change fixes the root for the
currently implemented `standard` / `partner` variants. Generic multi-arm
experiments remain a future product/config shape, not part of this thin slice.

Diff budget note: this is slightly over the 400 LOC soft cap because the
money-safety contract needs paired positive and negative coverage
(standard-compatible, partner-authorized, unknown-variant rejected,
partner-amount-not-allowlisted rejected) plus the cross-repo checkout contract
doc. The runtime code change remains narrow.

## Scope (this PR)

Ownership lane: deflection/stripe-monetization
Slice phase: Vertical slice

1. Add ATLAS config fields for the partner deflection price ID and amount while
   preserving the existing standard env names and default behavior.
2. Extend the Content Ops control-surface pricing contract so `standard` and
   `partner` public pricing terms can be resolved independently, with no
   `price_id` exposed by public pricing routes.
3. Extend checkout authorization with an optional `price_variant` query
   parameter. Missing/blank means `standard`; unknown or misconfigured variants
   fail closed before any Stripe Checkout session can be created.
4. Prove standard backward compatibility, partner authorization, unknown-variant
   rejection, and partner amount/allowed-set mismatch failure.

### Files touched

- `atlas_brain/api/__init__.py`
- `atlas_brain/config.py`
- `docs/frontend/content_ops_faq_deflection_checkout_contract.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `plans/PR-Deflection-Pricing-Variant-Authorization.md`
- `tests/test_atlas_content_ops_generated_assets_api.py`
- `tests/test_extracted_content_control_surface_api.py`

### Review Contract

Acceptance criteria:

- `POST /content-ops/deflection-reports/{request_id}/checkout-authorization`
  without `price_variant` still returns the standard checkout terms shape used
  today.
- The same route with `price_variant=partner` returns the configured partner
  `amount_cents`, `currency`, and `price_id`, plus a non-secret `variant`
  discriminator.
- Public pricing terms never expose Stripe `price_id`.
- A partner amount that is not present in
  `ATLAS_SAAS_STRIPE_CONTENT_OPS_DEFLECTION_REPORT_ALLOWED_AMOUNT_CENTS` fails
  closed with the existing payment-gate 503 class.
- Unknown variants fail closed with HTTP 400 and do not fall back to standard.

Affected surfaces:

- Hosted Content Ops deflection pricing and checkout authorization routes.
- Host wiring from `atlas_brain.config.SaaSAuthConfig` into
  `ContentOpsControlSurfaceApiConfig`.

Risk areas:

- Money safety: wrong variant must not silently charge a valid but unintended
  price.
- Backward compatibility: existing standard checkout calls and standard pricing
  display must keep working.
- Secret exposure: public pricing endpoints must not leak Stripe Price IDs.

Reviewer rules triggered: R1, R2, R3, R4, R5, R8, R10, R11, R12, R13, R14.

## Mechanism

`ContentOpsControlSurfaceApiConfig` keeps the current standard fields and gains
partner fields. A small resolver normalizes a requested variant, selects the
right amount and `price_id`, reuses the existing currency and allowed-amount set,
and applies the same fail-closed validation as the standard path. The public
pricing routes project only `variant`, `status`, `amount_cents`, and `currency`;
checkout authorization returns the trusted charge terms including `price_id`
only after the report-state checks pass.

The existing `/pricing/standard` route stays intact for portfolio's current
standard display path. A new `/pricing/{price_variant}` route supports the
partner terms needed by the next portfolio slice. Checkout authorization accepts
`price_variant` as an optional query parameter so old callers default to
standard and new callers can explicitly request partner.

## Intentional

- This is not a generic arbitrary experiment catalogue. The repo already has
  `standard` and `partner` portfolio variants; this slice makes those safe to
  authorize from ATLAS.
- Currency remains one shared deflection checkout currency. A per-variant
  currency matrix is not needed for the current USD-only Stripe prices.
- The webhook amount gate remains the exact allowed-set gate. Operators must add
  each live variant amount to the allowlist; the authorization path now refuses a
  variant whose amount is missing from that set.

## Deferred

- atlas-portfolio follow-up: pass the server-bound `priceVariant` to ATLAS
  checkout authorization and remove the standard-only 503 gate for partner when
  hosted config is ready.
- Generic multi-arm/cohort A/B beyond `standard` / `partner` remains deferred
  until the product chooses that paid-surface shape.
- Operator runbook update for enabling partner/test prices in hosted env after
  this ATLAS contract lands.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_extracted_content_control_surface_api.py -k "deflection_standard_pricing_terms or deflection_pricing_terms or deflection_checkout_authorization or deflection_report_routes_use_public_and_trusted_dependencies" -q
  Result: 25 passed, 136 deselected.
- Command: python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -k "deflection_checkout_amount_requires_exact_default_amount or deflection_checkout_amount_uses_allowed_amount_set or deflection_checkout_completion_accepts_lower_allowlisted_amount" -q
  Result: 3 passed, 44 deselected, 1 existing torch warning.
- Command: python -m pytest tests/test_atlas_content_ops_generated_assets_api.py -k "public_deflection_routes_use_rate_limit_gate" -q
  Result: 1 passed, 22 deselected, 1 existing torch warning.
- Command: python -m pytest tests/test_atlas_main_voice_startup.py -q
  Result: 28 passed, 1 existing torch warning.
- Command: python -m pytest tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_deflection_checkout_contract_pins_paid_handoff -q
  Result: 1 passed.
- Command: python -m py_compile atlas_brain/config.py atlas_brain/api/__init__.py extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_control_surface_api.py tests/test_atlas_content_ops_generated_assets_api.py
  Result: passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  Result: passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  Result: passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  Result: passed.
- Command: bash scripts/check_ascii_python.sh
  Result: passed.
- Command: git diff --check
  Result: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/__init__.py` | 6 |
| `atlas_brain/config.py` | 15 |
| `docs/frontend/content_ops_faq_deflection_checkout_contract.md` | 62 |
| `extracted_content_pipeline/api/control_surfaces.py` | 76 |
| `plans/PR-Deflection-Pricing-Variant-Authorization.md` | 158 |
| `tests/test_atlas_content_ops_generated_assets_api.py` | 1 |
| `tests/test_extracted_content_control_surface_api.py` | 163 |
| **Total** | **481** |
