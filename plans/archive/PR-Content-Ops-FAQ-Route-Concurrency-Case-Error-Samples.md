# PR-Content-Ops-FAQ-Route-Concurrency-Case-Error-Samples

## Why this slice exists

PR #1046 made per-case latency budgets fail closed when a budgeted case has no
request samples. The older per-case error-rate budget still treats a zero-request
case as `0.0` and passes. That leaves one budget path where `--requests` smaller
than the case count can silently skip a case. This slice aligns per-case
error-rate budget behavior with the per-case latency budget.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Make `--max-case-error-rate` fail closed for case summaries with zero
   requests.
2. Preserve the existing over-threshold per-case error behavior.
3. Add focused tests for the zero-request error-budget branch.

### Files touched

- `plans/PR-Content-Ops-FAQ-Route-Concurrency-Case-Error-Samples.md`
- `scripts/smoke_content_ops_faq_search_route_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`

## Mechanism

`_budget_summary(...)` already receives every `cases.summaries[]` row. This
slice checks each case summary's `requests` count before reading
`errors.rate`. If the count is zero, the budget emits `actual: null`, marks the
check failed, and records a deterministic failure string. Cases with samples
continue to use the existing error-rate comparison.

## Intentional

- The behavior changes only when `--max-case-error-rate` is supplied.
- Search-only runs without a per-case error budget remain unchanged.
- No route, database, seed, or runbook changes.

## Deferred

Parked hardening: none. `HARDENING.md` was scanned and has no active FAQ search
entries touching this runner.

Choosing production budget values remains deferred until repeated hosted runs
provide stable evidence.

## Verification

Local verification:

- python -m pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py
  (69 passed)
- python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-Route-Concurrency-Case-Error-Samples.md
  (passed)
- python scripts/audit_extracted_pipeline_ci_enrollment.py
  (122 matching tests enrolled)
- bash scripts/run_extracted_pipeline_checks.sh
  (2568 passed, 7 skipped)
- bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/pr-content-ops-faq-route-concurrency-case-error-samples.md
  (passed)

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 71 |
| Smoke script | 12 |
| Tests | 45 |
| **Total** | **128** |
