# PR-Deflection-HTTP-Helper-Consolidation

## Why this slice exists

#1722 exposed a structural defect in the deflection report delivery proof
scripts: each operator script owns a local copy of the same HTTP boundary
helpers, and one copy diverged during a review fix. The immediate regression
was fixed in #1722 by conforming that copy back to the sibling pattern, but
the root cause remains: there is no single source for the deflection
operator-script HTTP contract.

Root cause: six deflection operator scripts duplicate the `urlopen` wrapper,
HTTP error body reader, and JSON request handling with small signature and
return-type variations. The duplicated seam makes it easy for a future proof
helper to preserve status codes differently, sanitize errors differently, or
encode webhook/body payloads incorrectly.

This PR fixes the root within the report-delivery lane by moving the shared
HTTP behavior into one owned script module and migrating every current
deflection operator-script caller to it. The migration preserves caller
behavior, including raw Stripe webhook bytes, JSON Mapping bodies, optional
auth headers, and caller-specific redaction.

The diff is over the 400 LOC target because the root fix has to migrate all
six duplicated callers in the same PR. Leaving any one caller on the old local
copy would preserve the divergence class this slice is meant to close.

## Scope (this PR)

Ownership lane: content-ops/report-delivery-live-funnel
Slice phase: Production hardening

1. Add a shared deflection operator-script HTTP helper module for the common
   request opening, HTTP error text extraction, and JSON request result shape.
2. Migrate the six current deflection operator scripts that duplicate those
   helpers:
   `prepare_content_ops_deflection_env.py`,
   `run_deflection_full_report_qa_live_runner.py`,
   `run_deflection_test_mode_checkout_proof.py`,
   `smoke_content_ops_deflection_portfolio_result_page.py`,
   `smoke_content_ops_deflection_stripe_paid_unlock.py`, and
   `smoke_content_ops_deflection_submit_handoff.py`.
3. Add focused helper tests that mock `urllib.request.urlopen` and prove
   status preservation, sanitized transport errors, JSON Mapping body encoding,
   raw webhook bytes, and optional auth/signature headers.
4. Enroll the new helper tests in the extracted pipeline local runner and CI
   explicit test list.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Deflection-HTTP-Helper-Consolidation.md`
- `scripts/_deflection_http.py`
- `scripts/prepare_content_ops_deflection_env.py`
- `scripts/run_deflection_full_report_qa_live_runner.py`
- `scripts/run_deflection_test_mode_checkout_proof.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/smoke_content_ops_deflection_portfolio_result_page.py`
- `scripts/smoke_content_ops_deflection_stripe_paid_unlock.py`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `tests/test_deflection_http_helpers.py`
- `tests/test_run_deflection_test_mode_checkout_proof.py`

### Review Contract

- Acceptance criteria:
  - [ ] `HTTPError` continues to reach the JSON response layer with the real
        HTTP status code intact.
  - [ ] Generic transport errors are redacted through the caller-provided
        redactor before reaching script result payloads.
  - [ ] JSON Mapping request bodies are encoded as JSON with the expected
        content type.
  - [ ] Raw webhook bytes are sent unchanged with the Stripe signature header.
  - [ ] Every migrated script preserves its current response field names and
        existing focused tests still pass.
  - [ ] The new helper test is enrolled in the extracted pipeline runner and
        workflow.
- Affected surfaces: deflection operator scripts, hosted ATLAS proof HTTP
  calls, Stripe webhook proof replay, extracted pipeline CI enrollment.
- Risk areas: payment proof status truthfulness, Stripe webhook replay bytes,
  sanitization, auth/signature headers, refactor parity.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R12, R14.

## Mechanism

Create `scripts/_deflection_http.py` with a shared frozen response dataclass
and helpers for opening HTTP requests, reading HTTP error bodies, and making
JSON-oriented requests. The shared JSON request helper takes optional bearer
token, raw data bytes, JSON body Mapping, Stripe signature, timeout, and a
redactor callback. It returns the same status/payload/text/errors shape the
current callers already use.

The request body contract is explicit: callers pass a Mapping when the helper
should JSON-encode the body, or raw bytes when the caller must preserve the
exact payload. That keeps the submit/env scripts and Stripe webhook replay from
fighting over one ambiguous parameter.

Each migrated script imports the shared module, removes its local duplicate
helpers/dataclass where possible, and keeps its caller-facing CLI/result
contracts unchanged. Existing script tests cover end-to-end parity while the
new helper tests pin the shared edge cases directly.

## Intentional

- This is a script-level shared module under `scripts/`, not an
  `atlas_brain` runtime module. These operator helpers are not synced package
  files, and keeping the helper beside its callers avoids broad app-runtime
  coupling.
- The helper accepts a caller-provided redactor instead of importing one
  sanitizer. The existing scripts have different sensitive-pattern sets, so
  centralizing redaction policy in this slice would be a behavior change.
- The helper keeps one response shape rather than preserving both
  `HttpResult` and `HttpJsonResponse` names. Call sites use equivalent fields,
  and the tests verify those fields continue to exist where the scripts read
  them.

## Deferred

- Cross-repo or runtime API HTTP client consolidation is deferred. This slice
  is deliberately limited to the duplicated deflection operator scripts.
- Removing the deflection maturity baseline entries for older sibling scripts
  is deferred; this slice prevents new divergence without reworking every
  historical heuristic finding in the lane.

Parked hardening: none.

## Verification

- Command passed: python -m py_compile scripts/_deflection_http.py scripts/prepare_content_ops_deflection_env.py scripts/smoke_content_ops_deflection_submit_handoff.py scripts/smoke_content_ops_deflection_stripe_paid_unlock.py scripts/run_deflection_test_mode_checkout_proof.py scripts/smoke_content_ops_deflection_portfolio_result_page.py scripts/run_deflection_full_report_qa_live_runner.py tests/test_deflection_http_helpers.py tests/test_run_deflection_test_mode_checkout_proof.py.
- Command passed: pytest tests/test_deflection_http_helpers.py tests/test_prepare_content_ops_deflection_env.py tests/test_smoke_content_ops_deflection_submit_handoff.py tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py tests/test_run_deflection_test_mode_checkout_proof.py tests/test_smoke_content_ops_deflection_portfolio_result_page.py tests/test_run_deflection_full_report_qa_live_runner.py -q -- 131 passed.
- Command passed: python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 185 matching tests are enrolled.
- Command passed: mapfile -t deflection_lane < <(find atlas_brain scripts extracted_content_pipeline -type f -name '*deflection*.py' | sort -u); python scripts/maturity_sweep_file_lane.py "${deflection_lane[@]}" --tests-root tests --baseline tests/maturity_sweep/baseline_deflection_lane.json --min-score 8 --sensitive-glob '*deflection*' --sensitive-glob '**/billing/**' --sensitive-glob '**/billing*' --sensitive-glob '**/paid*' --sensitive-glob '**/auth/**' --sensitive-glob '**/auth*' --sensitive-glob '**/webhook*' --sensitive-glob '**/webhooks/**' --sensitive-glob '**/payment*' --sensitive-glob '**/invoicing/**' --sensitive-glob '**/*invoice*' --sensitive-glob '**/*deletion*' --
  (ratchet gate passed: no new brittleness above baseline)
- Command passed: bash scripts/run_extracted_pipeline_checks.sh -- 4695 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `plans/PR-Deflection-HTTP-Helper-Consolidation.md` | 152 |
| `scripts/_deflection_http.py` | 168 |
| `scripts/prepare_content_ops_deflection_env.py` | 71 |
| `scripts/run_deflection_full_report_qa_live_runner.py` | 44 |
| `scripts/run_deflection_test_mode_checkout_proof.py` | 58 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `scripts/smoke_content_ops_deflection_portfolio_result_page.py` | 63 |
| `scripts/smoke_content_ops_deflection_stripe_paid_unlock.py` | 71 |
| `scripts/smoke_content_ops_deflection_submit_handoff.py` | 78 |
| `tests/test_deflection_http_helpers.py` | 160 |
| `tests/test_run_deflection_test_mode_checkout_proof.py` | 4 |
| **Total** | **874** |
