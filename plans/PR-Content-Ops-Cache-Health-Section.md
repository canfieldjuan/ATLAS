# PR: Content Ops Cache Health Section

## Why this slice exists

The Content Ops cost/caching lane now has cache policy defaults, exact-cache
adapter behavior, cache savings, and Content Ops usage diagnostics. The broader
admin `/admin/costs/cache-health` route still reports the legacy Atlas cache
layers without a Content Ops section, so an operator has to know about the
separate Content Ops summary route to see whether Content Ops cache policy is
working.

This slice wires the existing Content Ops usage read model into the admin cache
health endpoint instead of reimplementing cache-status aggregation.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Product polish

1. Add a `content_ops` section to `/admin/costs/cache-health`.
2. Reuse `summarize_content_ops_llm_usage` as the single source of truth for
   Content Ops calls, cache hits, cache savings, and cache-status rows.
3. Keep the existing cache-health sections unchanged.
4. Add a focused route test that proves the admin endpoint returns the Content
   Ops summary and calls the shared helper with the same `days` window.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Cache-Health-Section.md` | Plan doc for the admin cache-health Content Ops section. |
| `atlas_brain/api/admin_costs.py` | Add the shared Content Ops usage summary to the cache-health response. |
| `tests/test_admin_costs.py` | Cover the new cache-health section and helper wiring. |

## Mechanism

`admin_costs.cache_health(...)` imports and awaits
`summarize_content_ops_llm_usage(pool, days=days)`, then includes the returned
payload under a new top-level `content_ops` key. The helper already owns the
Content Ops `llm_usage` filters and cache-status grouping, so the broader admin
endpoint does not duplicate query text or metadata field names.

The route test monkeypatches the imported helper with an async fake. That keeps
this PR focused on route integration while the existing
`tests/test_extracted_content_ops_usage_summary.py` suite continues to prove
the helper's SQL and payload details.

## Intentional

- This does not rewrite the existing exact/provider/semantic/batch sections.
  They remain the broader Atlas cache-health view.
- This does not add another Content Ops cache-status query in
  `admin_costs.py`; duplicating the helper would create drift.
- This does not add UI rendering yet. The endpoint payload lands first so the
  admin UI can consume one stable field later.

## Deferred

- Future PR: render the new `content_ops` cache-health section in the admin
  cost UI.
- Future PR: fold #1040's recent-call labels into this view after that PR
  merges, if the UI needs drill-down links from the cache-health card.
- Parked hardening: none. Root `HARDENING.md` was scanned; there are no active
  parked items for this ownership lane.

## Verification

- python -m pytest tests/test_admin_costs.py::test_cache_health_rolls_up_exact_prompt_semantic_and_task_reuse -q — 1 passed, 1 warning.
- python -m compileall -q atlas_brain/api/admin_costs.py tests/test_admin_costs.py — passed.
- python -m pytest tests/test_admin_costs.py::test_cache_health_rolls_up_exact_prompt_semantic_and_task_reuse tests/test_extracted_content_ops_usage_summary.py::test_content_ops_usage_summary_contract_against_postgres tests/test_extracted_content_control_surface_api.py::test_usage_summary_route_returns_content_ops_llm_rollup_with_filters -q — 2 passed, 1 skipped, 1 warning.
- git diff --check — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~70 |
| Route wiring | ~5 |
| Test | ~25 |
| **Total** | **~100** |
