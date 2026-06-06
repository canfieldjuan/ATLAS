# PR-Content-Ops-FAQ-Route-Concurrency-Detail-Case-Preflight

## Why this slice exists

PR #1042 fixed the operator runbook so detail-required concurrency smoke cases
are hit-only and miss/liveness cases run separately without detail hydration.
The same contradictory input can still be supplied directly to the smoke CLI:
`--require-detail` plus a case file where a case sets `require_results: false`.
That shape cannot produce a valid detail hydration check because no first
`faq_id` is expected. The tool should reject it during preflight instead of
issuing hosted requests and reporting a later detail failure.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Extend case-file preflight validation for the hosted FAQ route concurrency
   smoke.
2. Reject any case-file row with `require_results: false` when `--require-detail`
   is enabled.
3. Add focused tests proving the detector fails closed and prevents hosted
   request execution.

### Files touched

- `plans/PR-Content-Ops-FAQ-Route-Concurrency-Detail-Case-Preflight.md`
- `scripts/smoke_content_ops_faq_search_route_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`

## Mechanism

`_load_cases(...)` already validates each case-file row before `main(...)`
starts concurrency work. This slice adds one more row-level invariant: detail
mode requires result rows, so a case-level `require_results: false` is invalid
when `args.require_detail` is true. The existing `main(...)` preflight path then
writes the result artifact and exits `2` without calling `_run_concurrent(...)`.

## Intentional

- No change to search-only miss/liveness probes; `require_results: false`
  remains valid when detail hydration is not required.
- No runtime route behavior changes. This only tightens the local smoke CLI
  contract before requests leave the process.
- No runbook changes; PR #1042 already documented the safe operator shape.

## Deferred

Parked hardening: none. `HARDENING.md` was scanned and has no active FAQ search
entries touching this runner.

Live hosted threshold tuning remains deferred until repeated hosted runs provide
stable latency evidence.

## Verification

Local verification:

- python -m pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py
  (64 passed)
- python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-Route-Concurrency-Detail-Case-Preflight.md
  (passed)
- python scripts/audit_extracted_pipeline_ci_enrollment.py
  (122 matching tests enrolled)
- bash scripts/run_extracted_pipeline_checks.sh
  (2563 passed, 7 skipped)
- bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/pr-content-ops-faq-route-concurrency-detail-case-preflight.md
  (passed)

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 77 |
| Smoke script | 4 |
| Tests | 51 |
| **Total** | **132** |
