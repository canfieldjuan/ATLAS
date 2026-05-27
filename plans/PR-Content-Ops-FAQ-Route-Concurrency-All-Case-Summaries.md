# PR-Content-Ops-FAQ-Route-Concurrency-All-Case-Summaries

## Why this slice exists

PR #1035 added per-case summaries and PR #1036 surfaced the worst case in the
default hosted FAQ search concurrency output. The summary wiring reused the
`cases.items[:20]` preview cap, so case summaries and the worst-case signal can
ignore case 21+ in a larger mixed-traffic case file. That weakens the operator
visibility we just added. This slice keeps the preview cap but computes compact
summaries for every loaded case.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Robust testing

1. Keep `cases.items` capped to the first 20 case definitions.
2. Change `cases.summaries` to include all loaded cases, not the preview subset.
3. Add a regression test proving a failing case beyond the preview cap appears
   in summaries and drives the worst-case selection.

### Files touched

- `plans/PR-Content-Ops-FAQ-Route-Concurrency-All-Case-Summaries.md`
- `scripts/smoke_content_ops_faq_search_route_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`

## Mechanism

`_summary_payload(...)` already has the full `active_cases` list. This PR passes
that full list into `_case_result_summaries(...)` while leaving
`cases.items[:20]` and `cases.truncated` unchanged. `_worst_case_summary(...)`
then sees the complete compact summary set.

## Intentional

- The full raw case definitions remain capped in `cases.items`; only compact
  per-case counts and timings are emitted for every case.
- No new budgets or pass/fail behavior are introduced.
- This fixes the visibility source instead of adding a special path in
  `_worst_case_summary(...)`.

## Deferred

Parked hardening: none.

Per-case pass/fail budgets remain deferred until live hosted runs provide
threshold evidence.

## Verification

Local verification:

- python -m pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py - 58 passed.
- python -m py_compile scripts/smoke_content_ops_faq_search_route_concurrency.py - passed.
- python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-Route-Concurrency-All-Case-Summaries.md - passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py - passed, 122 matching tests enrolled.
- bash scripts/run_extracted_pipeline_checks.sh - 2557 passed, 7 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/pr-content-ops-faq-route-concurrency-all-case-summaries.md - pending.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 68 |
| Runner summary | 2 |
| Tests | 50 |
| **Total** | **120** |
