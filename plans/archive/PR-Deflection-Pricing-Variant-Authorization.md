# PR-Deflection-Pricing-Variant-Authorization

## Why this slice exists

atlas-portfolio#194 is partially landed: the easy standard price-change
milestone is complete, but the remaining A/B / multiple simultaneous price path
is blocked because ATLAS still authorizes only one global deflection checkout
price. Portfolio already persists a server-bound `standard` or `partner`
variant at intake and stamps the variant into Stripe metadata, but it still has
to ask ATLAS for the actual `price_id` and amount used to charge.

Root cause: the ATLAS control-surface contract had no variant dimension for
deflection checkout terms, and the webhook completion gate only checked the paid
amount against a global allowlist. `_deflection_checkout_terms()` read a single
`price_id` / amount / currency tuple from config, checkout authorization did not
persist the authorized amount on the report row, and `billing.py` could not tell
a standard-authorized report from a partner-authorized report once both amounts
were globally allowlisted. This change fixes the root for the currently
implemented `standard` / `partner` variants by binding authorization and
completion to the same report row. Generic multi-arm experiments remain a future
product/config shape, not part of this thin slice.

Diff budget note: this is over the 400 LOC soft cap because the money-safety
contract needs paired positive and negative coverage (standard-compatible,
partner-authorized, unknown-variant rejected, partner-amount-not-allowlisted
rejected, wrong-authorized-variant rejected) plus the cross-repo checkout
contract doc and the schema/store binding the webhook now relies on. The runtime
code change remains narrow.

## Scope (this PR)

Ownership lane: deflection/stripe-monetization
Slice phase: Vertical slice
Max files: 16

1. Add ATLAS config fields for the partner deflection price ID and amount while
   preserving the existing standard env names and default behavior.
2. Extend the Content Ops control-surface pricing contract so `standard` and
   `partner` public pricing terms can be resolved independently, with no
   `price_id` exposed by public pricing routes.
3. Extend checkout authorization with an optional `price_variant` query
   parameter. Missing/blank means `standard`; unknown or misconfigured variants
   fail closed before any Stripe Checkout session can be created.
4. Persist the authorized variant, amount, currency, and price ID on the report
   row at checkout authorization time.
5. Bind Stripe webhook completion to the persisted report-specific amount and
   currency when multiple amounts are allowlisted, while preserving the legacy
   single-amount fallback for already-open standard checkouts.
6. Prove standard backward compatibility, partner authorization, unknown-variant
   rejection, partner amount/allowed-set mismatch failure, and wrong-authorized-
   variant rejection.

### Files touched

- `atlas_brain/api/__init__.py`
- `atlas_brain/api/billing.py`
- `atlas_brain/config.py`
- `atlas_brain/storage/migrations/342_content_ops_deflection_checkout_authorization.sql`
- `docs/frontend/content_ops_faq_deflection_checkout_contract.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/PR-Deflection-Pricing-Variant-Authorization.md`
- `tests/maturity_sweep/deflection_product_surface_manifest.json`
- `tests/test_atlas_billing_content_ops_deflection_paid_flow.py`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- `tests/test_atlas_content_ops_generated_assets_api.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
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
- Checkout authorization persists report-specific terms before the portfolio
  creates the Stripe Checkout session.
- A paid session whose amount is globally allowlisted but does not match the
  report's persisted authorized amount leaves the report locked and emits the
  existing paid-funnel terms-mismatch incident.

Affected surfaces:

- Hosted Content Ops deflection pricing and checkout authorization routes.
- Deflection report storage row shape and migration.
- Stripe webhook completion for paid Content Ops deflection reports.
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
only after the report-state checks pass. Successful authorization records the
selected variant, amount, currency, and price ID on
`content_ops_deflection_reports`.

The existing `/pricing/standard` route stays intact for portfolio's current
standard display path. A new `/pricing/{price_variant}` route supports the
partner terms needed by the next portfolio slice. Checkout authorization accepts
`price_variant` as an optional query parameter so old callers default to
standard and new callers can explicitly request partner.

The Stripe webhook still validates mode, paid status, signed metadata, currency,
and the configured allowlist. It then calls `mark_paid()` with the paid amount
and currency, and the store updates the report only if the persisted checkout
authorization matches. When multiple amounts are allowlisted, a report with no
persisted authorization fails closed; when only the legacy standard amount is
allowed, old already-open checkouts without the new columns can still complete.

## Intentional

- This is not a generic arbitrary experiment catalogue. The repo already has
  `standard` and `partner` portfolio variants; this slice makes those safe to
  authorize from ATLAS.
- Currency remains one shared deflection checkout currency. A per-variant
  currency matrix is not needed for the current USD-only Stripe prices.
- The global webhook amount gate remains as a coarse allowlist. Operators must
  add each live variant amount to the allowlist; the authorization path refuses a
  variant whose amount is missing from that set, and the webhook then enforces
  the report-specific persisted amount.

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
- Command: python -m pytest tests/test_alerts.py tests/test_atlas_billing_stripe_hardening.py tests/test_b2b_vendor_briefing.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py tests/test_content_ops_deflection_incidents.py tests/test_mcp_content_ops_deflection_readonly.py -q
  Result: 189 passed, 1 existing torch warning.
- Command: python -m pytest tests/test_content_ops_deflection_report.py::test_postgres_deflection_report_store_round_trips_paid_gate -q
  Result: 1 passed.
- Command: python -m pytest tests/test_atlas_content_ops_generated_assets_api.py -k "public_deflection_routes_use_rate_limit_gate" -q
  Result: 1 passed, 22 deselected, 1 existing torch warning.
- Command: python -m pytest tests/test_atlas_main_voice_startup.py -q
  Result: 28 passed, 1 existing torch warning.
- Command: python -m pytest tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_deflection_checkout_contract_pins_paid_handoff -q
  Result: 1 passed.
- Command: python -m py_compile atlas_brain/config.py atlas_brain/api/__init__.py atlas_brain/api/billing.py extracted_content_pipeline/api/control_surfaces.py extracted_content_pipeline/deflection_report_access.py tests/test_extracted_content_control_surface_api.py tests/test_atlas_content_ops_generated_assets_api.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_content_ops_faq_report_contract_docs.py
  Result: passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  Result: passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh
  Result: 5044 passed, 16 skipped, 1 existing torch warning.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  Result: passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  Result: passed.
- Command: bash scripts/check_ascii_python.sh
  Result: passed.
- Command: python scripts/check_deflection_product_surface_manifest.py
  Result: deflection product surface manifest ok: 38 file(s).
- Command: python scripts/audit_ai_reconciliation.py --current-pr-body-file tmp/pr-deflection-pricing-variant-authorization.md
  Result: passed.
- Command: python scripts/sync_pr_plan.py plans/PR-Deflection-Pricing-Variant-Authorization.md --check
  Result: passed.
- Command: git diff --check
  Result: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/__init__.py` | 6 |
| `atlas_brain/api/billing.py` | 79 |
| `atlas_brain/config.py` | 15 |
| `atlas_brain/storage/migrations/342_content_ops_deflection_checkout_authorization.sql` | 10 |
| `docs/frontend/content_ops_faq_deflection_checkout_contract.md` | 68 |
| `extracted_content_pipeline/api/control_surfaces.py` | 93 |
| `extracted_content_pipeline/deflection_report_access.py` | 190 |
| `plans/PR-Deflection-Pricing-Variant-Authorization.md` | 213 |
| `tests/maturity_sweep/deflection_product_surface_manifest.json` | 1 |
| `tests/test_atlas_billing_content_ops_deflection_paid_flow.py` | 12 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 154 |
| `tests/test_atlas_content_ops_generated_assets_api.py` | 1 |
| `tests/test_content_ops_deflection_report.py` | 2 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 3 |
| `tests/test_extracted_content_control_surface_api.py` | 181 |
| **Total** | **1028** |
