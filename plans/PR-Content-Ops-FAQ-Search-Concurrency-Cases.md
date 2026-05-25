# PR-Content-Ops-FAQ-Search-Concurrency-Cases

## Why this slice exists
The hosted FAQ search concurrency smoke now fails closed and exercises real
transport/parser/envelope failures, but it still models only one repeated query.
The next demo-read risk is concurrent users searching different ticket corpora
or statuses at the same time. Operators need that mixed-query pressure without
editing Python or waiting for the larger hosted seeded load-test slice.

This slice adds a small case-file input to the hosted concurrency smoke. The
diff approaches 400 LOC because each new case-file detector branch is pinned
with a focused negative fixture per the checker/auditor rule.

## Scope (this PR)
Ownership lane: content-ops/faq-search
Slice phase: Production hardening.

1. Add `--case-file` JSON support to the hosted FAQ search concurrency smoke.
2. Validate case-file shape fail-closed before issuing requests.
3. Round-robin configured requests across the loaded cases.
4. Include case metadata in each result row and in the summary artifact.
5. Add focused negative fixtures for malformed case files and bad field types.

### Files touched
- `plans/PR-Content-Ops-FAQ-Search-Concurrency-Cases.md`
- `scripts/smoke_content_ops_faq_search_route_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`

## Mechanism
The new `--case-file` flag accepts a JSON list of objects. Each object supports:
`query` (required non-empty string), optional `corpus_id`, optional `status`,
optional positive integer `limit`, and optional boolean `require_results`.
Omitted optional fields inherit the global CLI defaults.

When no case file is supplied, behavior remains the existing single-case smoke.
When a file is supplied, the smoke validates the whole file during preflight and
round-robins `--requests` across the case list. Each request result records the
case index and filters used, so failures can be traced back to a specific query
or corpus filter.

## Intentional
- No hosted data seeding lands here. This is a case-mix runner only.
- No new auth/account override is added; tenant scoping remains token-derived by
  the hosted route.
- Case-file validation is strict and fails before requests. A malformed operator
  input should not produce a partial smoke.

## Deferred
- Hosted seeded end-to-end load testing with isolated corpora and cleanup
  remains a later production-hardening slice.
- Per-case expected source IDs or result IDs remain deferred until we have a
  seeded hosted fixture shape.

## Verification
- pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py -q - 30 passed in 0.07s.
- python -m py_compile scripts/smoke_content_ops_faq_search_route_concurrency.py tests/test_smoke_content_ops_faq_search_route_concurrency.py - Passed.
- git diff --check - Passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-Concurrency-Cases.md - Passed.
- bash scripts/local_pr_review.sh - Passed.

## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | +67 / -0 |
| Smoke script | +127 / -10 |
| Tests | +187 / -3 |
| **Total** | **394** |
