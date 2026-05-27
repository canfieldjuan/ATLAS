# PR-Content-Ops-FAQ-Route-Concurrency-Case-Error-Budget

## Why this slice exists

The hosted FAQ search concurrency smoke now reports every case summary and shows
the worst case in the default output. The remaining survivability gap is that
aggregate `--max-error-rate` can still pass when one query/corpus case fails but
the overall request mix is large enough to dilute that failure. This slice adds
an opt-in per-case error-rate budget so operators can fail closed on a bad case
without changing default behavior.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Add `--max-case-error-rate` to the hosted FAQ search route concurrency smoke.
2. Validate the new budget fail-closed during preflight.
3. Evaluate the budget against every `cases.summaries[]` row.
4. Add focused tests for preflight rejection and budget failure when aggregate
   error rate passes but one case fails.

### Files touched

- `plans/PR-Content-Ops-FAQ-Route-Concurrency-Case-Error-Budget.md`
- `scripts/smoke_content_ops_faq_search_route_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`

## Mechanism

The runner already builds compact per-case summaries with each case's error
rate. This PR passes those summaries into `_budget_summary(...)` and, when
`--max-case-error-rate` is supplied, adds one check per case. A case whose
`errors.rate` exceeds the configured value marks the run failed and records a
deterministic budget failure string.

## Intentional

- The new budget is opt-in. No default hosted threshold is introduced.
- This slice only gates per-case error rate. Per-case latency budgets remain
  separate because latency thresholds need live route evidence.
- The budget uses existing summaries instead of reprocessing raw rows, keeping
  printed, JSON, and budget views aligned.

## Deferred

Parked hardening: none.

Per-case latency budgets remain deferred until live hosted runs provide p95/max
threshold evidence.

## Verification

Local verification:

- python -m pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py - 61 passed.
- python -m py_compile scripts/smoke_content_ops_faq_search_route_concurrency.py - passed.
- python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-Route-Concurrency-Case-Error-Budget.md - passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py - passed, 122 matching tests enrolled.
- bash scripts/run_extracted_pipeline_checks.sh - 2560 passed, 7 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/pr-content-ops-faq-route-concurrency-case-error-budget.md - pending.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 70 |
| Runner budget | 45 |
| Tests | 108 |
| **Total** | **223** |
