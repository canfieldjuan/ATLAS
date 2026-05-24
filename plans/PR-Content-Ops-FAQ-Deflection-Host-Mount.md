# PR-Content-Ops-FAQ-Deflection-Host-Mount

## Why this slice exists

`PR-Content-Ops-FAQ-Deflection-Search-Route` adds the extracted package router,
but the Atlas host still has to mount it before another session can point a demo
at a real deployed URL. The existing Content Ops routes already have the right
auth, tenant-scope, and database-pool bridge; this slice wires the FAQ search
route into that same host surface.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Vertical slice.

1. Import `create_faq_deflection_search_router` in the Atlas API aggregator's
   existing Content Ops optional-import block.
2. Mount `/content-ops/faq-deflection-search` with `get_db_pool`,
   `build_content_ops_scope`, and the same `_capture_content_ops_auth_user`
   dependency used by `/content-ops/*` and `/content-assets/*`.
3. Add a host wiring test that asserts the route is mounted with the shared pool,
   scope provider, and auth dependency.

### Files touched

- `plans/PR-Content-Ops-FAQ-Deflection-Host-Mount.md`
- `atlas_brain/api/__init__.py`
- `tests/test_atlas_content_ops_generated_assets_api.py`

## Mechanism

The host aggregator already wraps Content Ops routes in a defensive import block
so production images without the extracted package keep booting. This slice adds
one more router factory import and one `router.include_router(...)` call inside
that block:

- `pool_provider=get_db_pool`
- `scope_provider=build_content_ops_scope`
- `dependencies=[Depends(_capture_content_ops_auth_user)]`

The dependency preserves the existing B2B growth-plan auth gate and captures the
authenticated user into the ContextVar that `build_content_ops_scope` reads.

## Intentional

- No new standalone bearer-token system is introduced. The host route uses the
  same bearer-authenticated B2B plan dependency as the rest of Content Ops.
- No deployed host URL or environment variable is committed here. That handoff
  depends on deployment after merge.
- No client/demo code changes land here.

## Deferred

- Deployment handoff: after merge, report the deployed `/api/v1/content-ops/faq-deflection-search`
  URL and the existing bearer-token expectation to the demo session.
- `PR-Content-Ops-FAQ-Search-Concurrency-Smoke`: run concurrent filtered
  retrieval against seeded corpora after the hosted endpoint exists.

## Verification

- pytest tests/test_atlas_content_ops_generated_assets_api.py tests/test_extracted_ticket_faq_search_api.py -q - 10 passed, 1 warning.
- python -m py_compile atlas_brain/api/__init__.py tests/test_atlas_content_ops_generated_assets_api.py - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 72 |
| Host mount | 9 |
| Host wiring test | 19 |
| **Total** | **100** |
