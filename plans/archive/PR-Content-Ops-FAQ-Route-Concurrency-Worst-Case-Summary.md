# PR-Content-Ops-FAQ-Route-Concurrency-Worst-Case-Summary

## Why this slice exists

PR #1035 added per-case visibility to the hosted FAQ search concurrency smoke
artifact, but the default terminal output still reports only aggregate errors
and latency. Operators running a live mixed-query smoke without `--json` can
miss which case is failing or slow unless they open the result file. This slice
surfaces the worst per-case signal in the default one-line output while leaving
the JSON artifact and pass/fail semantics unchanged.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Robust testing

1. Add a compact helper that selects the worst case summary from
   `cases.summaries`.
2. Include worst-case index, error count, p95 latency, and max latency in the
   default non-JSON concurrency smoke output.
3. Add focused tests for worst-case selection and printed output.

### Files touched

- `plans/PR-Content-Ops-FAQ-Route-Concurrency-Worst-Case-Summary.md`
- `scripts/smoke_content_ops_faq_search_route_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`

## Mechanism

The runner already computes `cases.summaries` in the result payload. This PR
adds a small pure helper that ranks case summaries by error count, p95 latency,
max latency, and then case index. `_print_summary(...)` uses that helper only
for the human-readable line. JSON output remains the complete structured
payload.

## Intentional

- No pass/fail behavior changes. Existing aggregate budgets remain the only
  budget enforcement in this slice.
- No new command-line flags. The default output should include the operational
  clue automatically.
- The helper ranks by existing summary fields instead of reprocessing raw rows,
  keeping the CLI line aligned with the artifact.

## Deferred

Parked hardening: none.

Per-case pass/fail budgets remain deferred until a live hosted run provides
evidence for thresholds worth enforcing.

## Verification

Local verification:

- python -m pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py - 57 passed.
- python -m py_compile scripts/smoke_content_ops_faq_search_route_concurrency.py - passed.
- python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-Route-Concurrency-Worst-Case-Summary.md - passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py - passed, 122 matching tests enrolled.
- bash scripts/run_extracted_pipeline_checks.sh - 2550 passed, 7 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/pr-content-ops-faq-route-concurrency-worst-case-summary.md - pending.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 71 |
| Runner output | 42 |
| Tests | 85 |
| **Total** | **198** |
