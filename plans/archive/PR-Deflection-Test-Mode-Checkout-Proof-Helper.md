# PR-Deflection-Test-Mode-Checkout-Proof-Helper

## Why this slice exists

#1612 now has a successful no-live-charge Stripe test-mode unlock proof, but the repeatable path is still too manual: the operator has to compose Stripe Checkout metadata, complete the hosted test Checkout, then stitch together the unlock evidence without committing request ids, Checkout ids, URLs, tokens, or report content.

Root cause: the proof flow still depends on operator-composed payment metadata at the Stripe boundary. #1705 made the webhook replay smoke derive the tenant metadata from the persisted report row, but the Checkout creation side of the same proof still needs an operator-safe helper so the next live/test run does not rely on memory or raw IDs in issue comments.

This PR fixes the root within ATLAS scope by adding an operator-run test-mode Checkout proof helper that derives the report account from the database, confirms ATLAS checkout authorization terms, creates a Stripe test-mode Checkout Session with that derived metadata, and writes only a sanitized proof result. It does not change production billing or the atlas-portfolio checkout route.

Diff-budget note: this is intentionally over the soft 400 LOC target. The
operator helper, sanitizer/leak guard, mocked Stripe/HTTP boundary probes, and
CI enrollment are one safety unit; landing the helper without the negative
tests would leave the payment/proof boundary under-proven.

## Scope (this PR)

Ownership lane: content-ops/report-delivery-live-funnel
Slice phase: Functional validation

1. Add an operator-run script for the #1612 no-live-charge proof path: persisted report row -> ATLAS checkout authorization -> Stripe test-mode Checkout Session -> optional paid-artifact polling.
2. Require test-mode Stripe keys only, derive `account_id` from the persisted report row, omit `payment_method_types`, and keep raw Checkout URL/session identifiers out of the JSON proof artifact.
3. Add focused tests that mock Stripe and hosted HTTP transport for successful creation, fail-closed authorization/metadata/key cases, sanitized output, and optional unlock polling.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Deflection-Test-Mode-Checkout-Proof-Helper.md`
- `scripts/run_deflection_test_mode_checkout_proof.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/maturity_sweep/baseline_deflection_lane.json`
- `tests/test_run_deflection_test_mode_checkout_proof.py`

### Review Contract

- Acceptance criteria:
  - [ ] The helper derives the Stripe metadata `account_id` from the persisted `content_ops_deflection_reports` row before creating Checkout.
  - [ ] The helper rejects live-mode Stripe keys and does not pass `payment_method_types` to Checkout Session creation.
  - [ ] The helper confirms ATLAS checkout authorization terms before Stripe creation and fails closed if authorization is missing or malformed.
  - [ ] The helper writes sanitized JSON only: no raw request id, account id, token, database URL, Stripe key, Checkout Session id, Checkout URL, or result URL.
  - [ ] Optional unlock polling can observe locked-then-unlocked artifact status without leaking raw identifiers.
  - [ ] The new tests are enrolled in the extracted pipeline local runner and workflow path filters.
- Affected surfaces: operator scripts, Stripe test-mode proof tooling, hosted ATLAS proof HTTP calls, CI enrollment.
- Risk areas: billing/security metadata, secret leakage, proof artifact truthfulness, idempotency, CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R12, R14.

## Mechanism

The new script reuses the persisted-row account lookup from the existing paid-unlock smoke, then calls the hosted ATLAS checkout-authorization endpoint for the same request. Those two upstream checks establish the tenant/report metadata and price/currency terms before Stripe is touched.

Only test-mode Stripe keys are accepted. Checkout Session creation uses Stripe-hosted Checkout in `payment` mode with one configured price line item, derived report metadata, success/cancel URLs supplied by the operator, and a deterministic idempotency key. It intentionally omits `payment_method_types` so Stripe's Dashboard-configured dynamic payment methods remain in control.

The script can either stop after session creation or poll the paid artifact endpoint until it unlocks. During a real operator run the Checkout URL may be printed to the terminal for local use, but the JSON result records only presence/status booleans and sanitized status values. A final leak guard scans the emitted payload before writing it.

## Intentional

- This is an operator proof helper, not a production endpoint. It lives under `scripts/` and uses mocked Stripe/HTTP in tests.
- The helper creates a direct test-mode Stripe Checkout Session from ATLAS authorization terms instead of calling the atlas-portfolio route. The portfolio route creation was already separately proven; this slice closes the repeatable no-live-charge unlock proof in the ATLAS lane.
- The helper does not automate card entry in Stripe-hosted Checkout. It prints the Checkout URL locally when requested, then waits for webhook-driven unlock if polling is enabled.
- The helper accepts Stripe restricted API keys with an `rk_test_` prefix as well as `sk_test_` keys, and rejects live-mode keys.
- `_open_http_request` intentionally matches the existing one-line sibling operator-script pattern. HTTP status preservation belongs in `_json_request`; the maturity baseline entry accepts the same heuristic shape already grandfathered for those siblings.

## Deferred

- Full browser automation of test-mode Checkout card completion remains deferred; this slice keeps the human-controlled hosted Checkout step.
- The non-blocking #1705 NIT about avoiding database URLs on the CLI is still deferred. This helper follows the existing operator-script convention and keeps the DSN out of artifacts.
- A future atlas-portfolio slice can wrap the same proof around its public checkout route if we want to re-prove route creation and unlock in one place.

Parked hardening: none.

## Verification

- Python compile for the new helper script and its focused test: passed.
- Focused pytest for the new helper plus the existing paid-unlock smoke: 33 passed.
- Extracted pipeline CI enrollment audit: OK, 185 matching tests enrolled.
- Deflection maturity ratchet command from `.github/workflows/maturity_sweep_deflection_content_ops.yml`: passed; the new helper scores 6 and is accepted through the lane baseline to match the existing sibling operator-script pattern.
- Full extracted pipeline bundle: reasoning core 295 passed; extracted content pipeline 4690 passed, 10 skipped, 1 existing torch warning.
- `gitleaks` local binary check: not installed locally; review fix lowered the fabricated Stripe test key fixture to the low-entropy `sk_test_0000000000000000` form while preserving Stripe-key matching/redaction coverage.
- Review-fix transport probe: the one-line sibling wrapper lets `HTTPError` pass through to `_json_request` so status codes are preserved, while generic `URLError` is still sanitized before it reaches the result payload.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `plans/PR-Deflection-Test-Mode-Checkout-Proof-Helper.md` | 91 |
| `scripts/run_deflection_test_mode_checkout_proof.py` | 576 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/maturity_sweep/baseline_deflection_lane.json` | 7 |
| `tests/test_run_deflection_test_mode_checkout_proof.py` | 519 |
| **Total** | **1198** |
