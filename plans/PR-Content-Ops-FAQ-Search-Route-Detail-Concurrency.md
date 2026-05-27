# PR-Content-Ops-FAQ-Search-Route-Detail-Concurrency

## Why this slice exists

The hosted FAQ search concurrency smoke pressures compact search reads, and the
seeded e2e proves one search result can hydrate through the detail route. The
demo flow uses both together: concurrent users search, select a result, and
hydrate the full generated FAQ report.

This slice adds the thinnest opt-in hosted detail check to the existing route
concurrency smoke so operators can pressure that real read path without a new
runner or any data-seeding behavior.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Robust testing

1. Add `--require-detail` and `--detail-route` to the hosted FAQ search route
   concurrency smoke.
2. When enabled, fetch the first result's FAQ detail after each successful
   search response and validate the canonical detail envelope.
3. Include compact detail status in per-request rows and the summary payload.
4. Keep latency budgets opt-in and unchanged; elapsed request time includes any
   detail fetch when detail mode is enabled.
5. Add focused fixtures for successful detail hydration, missing FAQ ids, and
   malformed detail envelopes.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Search-Route-Detail-Concurrency.md` | Plan contract for this detail-read concurrency slice. |
| `scripts/smoke_content_ops_faq_search_route_concurrency.py` | Add opt-in detail hydration validation to the hosted concurrency smoke. |
| `tests/test_smoke_content_ops_faq_search_route_concurrency.py` | Cover detail success and fail-closed detail branches. |

## Mechanism

`_run_one(...)` keeps its existing search request and envelope checks. If
`--require-detail` is set and search validation produced no errors, it extracts
`results[0].faq_id`, builds the detail URL with the shared contract checker, and
validates the response with the same `contract._validate_detail(...)` helper
used by the single-request checker.

Per-request rows gain `detail_checked`, `detail_faq_id`, and
`detail_elapsed_ms`. The summary reports how many requests attempted detail and
how many detail checks failed.

## Intentional

- No new route, database, seeding, cleanup, or detail schema behavior.
- No default detail mode; existing search-only concurrency behavior is
  unchanged unless operators pass `--require-detail`.
- No default latency SLO. Existing p95/max request budgets remain caller-owned
  and include detail work when detail mode is enabled.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned and has no active FAQ
  search entries touching this runner.
- Detail-specific latency budgets remain deferred until we have a concrete
  hosted detail target.

## Verification

- python -m pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py -q — 46 passed.
- python -m py_compile scripts/smoke_content_ops_faq_search_route_concurrency.py tests/test_smoke_content_ops_faq_search_route_concurrency.py — passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-Route-Detail-Concurrency.md — passed.
- git diff --check — passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py . — 122 matching tests enrolled.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- bash scripts/run_extracted_pipeline_checks.sh — 2515 passed, 7 skipped, 1 warning.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 84 |
| Route concurrency smoke | 74 |
| Tests | 190 |
| **Total** | **348** |
