# PR: Content Ops Cache Diagnostics Summary

## Why this slice exists

Content Ops now has an exact-cache policy, adapter, savings rollup, and UI
control. Operators can see total cache hits and avoided spend, but they still
cannot tell why cache did or did not apply: disabled, explicit no-store,
customer-data no-store, miss, hit, store error, and so on.

This slice adds the backend read model for those diagnostics in the existing
usage summary payload. The UI can render the breakdown in a follow-up without
guessing from raw `llm_usage.metadata`.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Add a `by_cache_status` breakdown to `summarize_content_ops_llm_usage`.
2. Group by cache mode, cache reason, cache result, and cache store result from
   Content Ops LLM trace metadata.
3. Reuse the same account/run/request/asset filters as the existing summary,
   model, and asset rollups.
4. Include calls, cost, token totals, and cache savings in each diagnostic row.
5. Add focused fake-pool and Postgres tests for payload shape, account
   filtering, and numeric cache-savings guarding.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Cache-Diagnostics-Summary.md` | Plan doc for the cache diagnostics summary backend slice. |
| `extracted_content_pipeline/content_ops_usage_summary.py` | Add the cache-status usage summary query and payload mapper. |
| `tests/test_extracted_content_ops_usage_summary.py` | Cover summary query shape and Postgres diagnostic rollup payloads. |
| `tests/test_extracted_content_control_surface_api.py` | Pin the API route payload shape with cache diagnostic rows. |

## Mechanism

The summary helper runs a third grouped query after `by_model` and
`by_asset_type`. It projects:

- `metadata ->> 'cache_mode'`
- `metadata ->> 'cache_reason'`
- `metadata ->> 'cache_result'`
- `metadata ->> 'cache_store_result'`

Blank values become `unknown`, matching existing model/asset breakdown
conventions. Each row carries `cost_usd`, `cache_savings_usd`, `calls`,
`input_tokens`, and `output_tokens`. The `cache_savings_usd` expression keeps
the existing guarded numeric JSON check so malformed metadata cannot break the
summary query.

## Intentional

- This does not add UI rendering yet. The backend read model lands first so the
  next UI slice can consume a stable field.
- This does not change cache policy, cache lookup/store behavior, or trace
  metadata names.
- This does not expose raw prompts, responses, or arbitrary metadata.
- This keeps diagnostics in the existing usage summary endpoint instead of
  adding a new route.

## Deferred

- Future PR: render `by_cache_status` in the Intel UI usage area.
- Future PR: persisted tenant cache defaults if operators need always-on cache
  posture.
- Parked hardening: none. Root `HARDENING.md` was scanned; the current
  parked item belongs to the FAQ lane, not this cost-surfacing slice.

## Verification

- python -m pytest tests/test_extracted_content_ops_usage_summary.py tests/test_extracted_content_control_surface_api.py::test_usage_summary_route_returns_content_ops_llm_rollup_with_filters -q — 3 passed, 2 skipped.
- python -m compileall -q extracted_content_pipeline/content_ops_usage_summary.py tests/test_extracted_content_ops_usage_summary.py tests/test_extracted_content_control_surface_api.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- bash scripts/run_extracted_pipeline_checks.sh — 2539 passed, 7 skipped, 1 warning.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas_pr_cache_diagnostics_summary_body.md — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~95 |
| Usage summary query/payload | ~45 |
| Tests | ~95 |
| **Total** | **~235** |
