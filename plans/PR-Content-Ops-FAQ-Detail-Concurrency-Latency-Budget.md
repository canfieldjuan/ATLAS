# PR-Content-Ops-FAQ-Detail-Concurrency-Latency-Budget

## Why this slice exists

The hosted FAQ search concurrency smoke can now hydrate generated FAQ detail
documents under load, but its latency budgets still only cover the combined
request row timing. That proves the route returns valid detail payloads, but it
does not give operators a focused fail-closed knob for slow detail fetches.
This slice adds that detail-read budget while staying in robust testing for the
already-built search/detail flow.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Robust testing

1. Add an optional `--max-detail-ms` budget to the hosted FAQ search concurrency
   smoke.
2. Validate the detail budget during preflight so impossible or misleading
   invocations fail before network I/O.
3. Include compact detail latency summary data in the smoke result artifact.
4. Add focused negative and positive tests for the new detector branch.

### Files touched

- `plans/PR-Content-Ops-FAQ-Detail-Concurrency-Latency-Budget.md`
- `scripts/smoke_content_ops_faq_search_route_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`

## Mechanism

The concurrency smoke already records `detail_elapsed_ms` for each detail fetch
when `--require-detail` is set. This PR summarizes those values separately from
the full row timing, then wires `--max-detail-ms` into the existing budget
summary so a single slow detail hydration fails the run with a deterministic
result artifact.

`--max-detail-ms` is only valid with `--require-detail`; otherwise the run would
claim to enforce a budget for work it never performs.

## Intentional

- The existing combined row latency budgets remain unchanged because they still
  measure the full user-facing search-plus-detail request cost.
- The new budget checks max detail fetch latency, not p95, matching the single
  route contract checker's `--max-detail-ms` semantics.
- This does not add a hosted live invocation because credentials and host
  selection remain operator/runtime concerns.

## Deferred

Parked hardening: none.

Per-case detail-required policy remains deferred. Mixed case files can still
include expected miss cases when detail checking is disabled globally; tightening
that shape is separate from measuring detail latency when detail checking is
explicitly enabled.

## Verification

Local verification:

- python -m pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py - 53 passed.
- python -m py_compile scripts/smoke_content_ops_faq_search_route_concurrency.py - passed.
- python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-Detail-Concurrency-Latency-Budget.md - passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py - passed, 122 matching tests enrolled.
- bash scripts/run_extracted_pipeline_checks.sh - 2546 passed, 7 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-content-ops-faq-detail-concurrency-latency-budget-body.md - passed.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 72 |
| Runner budget | 58 |
| Tests | 99 |
| **Total** | **234** |
