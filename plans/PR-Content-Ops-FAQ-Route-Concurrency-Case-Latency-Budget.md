# PR-Content-Ops-FAQ-Route-Concurrency-Case-Latency-Budget

## Why this slice exists

The hosted FAQ route concurrency smoke now has aggregate latency budgets,
detail latency budgets, and per-case error budgets. A slow query/corpus case can
still hide when aggregate p95/max latency passes because the case is diluted by
faster traffic. This slice adds opt-in per-case latency budgets so operators can
fail closed on a slow case after mixed-case visibility is already available.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Add opt-in per-case p95 and max latency budget flags to the hosted FAQ route
   concurrency smoke.
2. Validate those flags fail-closed during preflight.
3. Evaluate the budgets against every `cases.summaries[]` latency row.
4. Add the new opt-in flags to the hosted route concurrency runbook.
5. Add focused tests for invalid budgets, per-case latency failure when
   aggregate latency budgets pass, and zero-sample budgeted cases.

### Files touched

- `plans/PR-Content-Ops-FAQ-Route-Concurrency-Case-Latency-Budget.md`
- `docs/extraction/validation/content_ops_faq_route_concurrency_runbook.md`
- `scripts/smoke_content_ops_faq_search_route_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`

## Mechanism

The runner already builds one latency summary per case. This slice adds
`--max-case-p95-ms` and `--max-case-single-request-ms`, threads those values into
`_budget_summary(...)`, and emits deterministic budget checks per case. The new
checks use the existing case summaries instead of recalculating raw request
rows, keeping stdout, result JSON, and budget behavior aligned.
When a case has no latency samples, the per-case latency budget fails closed
with `actual: null` instead of treating missing latency as zero.

## Intentional

- The new budgets are opt-in. No default hosted latency threshold is introduced.
- This does not change aggregate latency budgets or detail latency budgets.
- No hosted route, database, or seed behavior changes.

## Deferred

Parked hardening: none. `HARDENING.md` was scanned and has no active FAQ search
entries touching this runner.

Choosing production SLO values remains deferred until repeated hosted runs
provide stable latency evidence.

## Verification

Local verification:

- python -m pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py
  (68 passed)
- python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-Route-Concurrency-Case-Latency-Budget.md
  (passed)
- python scripts/audit_extracted_pipeline_ci_enrollment.py
  (122 matching tests enrolled)
- bash scripts/run_extracted_pipeline_checks.sh
  (2567 passed, 7 skipped)
- bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/pr-content-ops-faq-route-concurrency-case-latency-budget.md
  (passed)

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 78 |
| Runbook | 4 |
| Smoke script | 55 |
| Tests | 156 |
| **Total** | **293** |
