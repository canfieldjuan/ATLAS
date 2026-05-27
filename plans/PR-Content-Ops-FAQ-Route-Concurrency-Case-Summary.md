# PR-Content-Ops-FAQ-Route-Concurrency-Case-Summary

## Why this slice exists

The hosted FAQ search concurrency smoke can exercise mixed query/corpus cases,
search-detail hydration, and route/detail latency budgets. Its result artifact
still reports most behavior as one aggregate run, so a single slow or failing
case can hide inside the total error rate and latency summary. For live demo
and survivability testing, operators need enough per-case visibility to see
which corpus/query shape failed under concurrent read traffic.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Robust testing

1. Add compact per-case result summaries to the hosted FAQ search concurrency
   smoke result artifact.
2. Report each case's request count, error count/rate, latency summary, detail
   check count, detail failure count, and detail latency summary.
3. Preserve existing aggregate budgets and pass/fail behavior.
4. Add focused tests proving mixed-case failures and detail latency are visible
   in the artifact.

### Files touched

- `plans/PR-Content-Ops-FAQ-Route-Concurrency-Case-Summary.md`
- `scripts/smoke_content_ops_faq_search_route_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`

## Mechanism

The runner already stamps each request row with `case_index` and a case
snapshot. This slice groups completed rows by case index inside
`_summary_payload(...)` and writes a compact `cases.summaries` list alongside
the existing case definitions. The rollups reuse the existing latency, error,
and detail summary helpers so aggregate and per-case math stay aligned.

## Intentional

- This does not add per-case budgets. The goal is visibility for mixed traffic;
  changing pass/fail semantics is a separate production-hardening decision.
- The existing aggregate `errors`, `latency`, `detail`, and `budgets` fields are
  unchanged for backward compatibility with current consumers.
- The case summaries are compact counts and timings, not full duplicate error
  item lists, because the aggregate `errors.items` already carries the first
  failure rows with case snapshots.

## Deferred

Parked hardening: none.

Per-case latency/error budgets remain deferred until a live hosted run shows a
specific threshold worth enforcing. This slice adds the visibility needed to
choose those thresholds from evidence.

## Verification

Local verification:

- python -m pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py - 55 passed.
- python -m py_compile scripts/smoke_content_ops_faq_search_route_concurrency.py - passed.
- python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-Route-Concurrency-Case-Summary.md - passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py - passed, 122 matching tests enrolled.
- bash scripts/run_extracted_pipeline_checks.sh - 2548 passed, 7 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/pr-content-ops-faq-route-concurrency-case-summary.md - passed.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 75 |
| Runner summary | 42 |
| Tests | 132 |
| **Total** | **249** |
