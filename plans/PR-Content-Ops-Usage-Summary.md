# PR: Content Ops Usage Summary

## Why this slice exists

PR-Content-Ops-LLM-Usage-Tracing made Content Ops provider calls write shared
llm_usage rows. The next production hardening step is a small read path that
lets operators see recent Content Ops spend before we wire budget gates or exact
caching.

This keeps the cost/caching lane source-first: trace actual calls, then surface
actual spend, then make budget/cache decisions from that data.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Add a Content Ops usage summary helper that aggregates llm_usage rows
   attributed to `content_ops`.
2. Add an optional `/content-ops/usage/summary` route behind a host-provided
   usage pool provider.
3. Support narrow operator filters for recent days, asset type, run id, and
   request id.
4. Return total cost/tokens/cache/failure counts plus provider/model and
   asset-type breakdowns.
5. Add focused tests for provider configuration, filters, and response shape.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Usage-Summary.md` | Plan doc for the first Content Ops cost read path. |
| `atlas_brain/api/__init__.py` | Wire the hosted Content Ops usage route to the Atlas DB pool. |
| `extracted_content_pipeline/content_ops_usage_summary.py` | Query and serialize Content Ops llm_usage rollups. |
| `extracted_content_pipeline/api/control_surfaces.py` | Expose the optional usage summary route. |
| `extracted_content_pipeline/manifest.json` | Register the new usage-summary helper as an owned extracted-package file. |
| `tests/test_extracted_content_control_surface_api.py` | Cover route provider behavior and usage-summary filters/output. |
| `tests/test_atlas_content_ops_generated_assets_api.py` | Cover hosted Atlas wiring for the usage route. |

## Mechanism

The helper builds a fixed SQL filter around rows where
`span_name = 'content_ops.llm.complete'` or `metadata ->> 'product' =
'content_ops'`. Optional filters add exact matches against `asset_type`,
`request_id`, and `run_id` metadata, with `run_id` also checking the promoted
llm_usage run_id column when present.

The route is opt-in via `usage_pool_provider`, matching the existing host-owned
provider pattern in the control-surface API. If the host does not wire a pool,
the route returns 503 instead of pretending usage is available.

Atlas wires the route to its shared DB pool in `atlas_brain/api/__init__.py` so
the hosted `/api/v1/content-ops/usage/summary` path is usable immediately after
merge.

## Intentional

- This is read-only. It does not introduce budget gating or cache behavior.
- This does not add UI yet; the route gives the UI a stable backend contract.
- This does not add new schema. It reads the existing llm_usage fields written
  by the shared tracer.
- This does not capture or expose prompt/response bodies.

## Deferred

- Future PR: add UI/control-surface cards that call this route.
- Future PR: wire `BudgetGate` once usage summary is queryable in the product.
- Future PR: add exact-cache integration with explicit support-ticket privacy
  policy and account scoping.
- Future PR: add execution/run identifiers to Content Ops `/execute` if we need
  first-class run-level grouping beyond caller metadata.
- Parked hardening: none planned.

## Verification

- python -m pytest tests/test_extracted_content_control_surface_api.py -q — 107 passed.
- python -m pytest tests/test_atlas_content_ops_generated_assets_api.py -q — 5 passed.
- python -m compileall -q atlas_brain/api/__init__.py extracted_content_pipeline/content_ops_usage_summary.py extracted_content_pipeline/api/control_surfaces.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <body> — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~105 |
| Atlas host wiring | ~15 |
| Usage summary helper | ~210 |
| API route | ~60 |
| Tests | ~140 |
| Manifest | ~5 |
| **Total** | **~535** |

This exceeds the 400 LOC soft cap because the first read path needs a query
helper, a route seam, manifest inventory, and response-shape tests in the same
slice. UI, budget gating, and cache wiring stay deferred rather than expanding
this PR further.
