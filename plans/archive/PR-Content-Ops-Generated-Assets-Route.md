# PR-Content-Ops-Generated-Assets-Route

## Why this slice exists

Generated Content Ops asset review/export routes already exist in
`extracted_content_pipeline.api.generated_assets`, but Atlas only mounts
the control-surface router. Operators can execute generation from the
host API but cannot review/export generated reports, blog posts,
landing pages, or sales briefs through the same host surface.

## Scope (this PR)

1. Mount `create_generated_asset_router` in `atlas_brain/api/__init__.py`.
2. Reuse the existing Content Ops auth bridge,
   `build_content_ops_scope`, and Atlas database pool.
3. Add Atlas route-registration tests proving `/content-assets/*`
   routes are mounted by the API aggregator and use the shared auth
   dependency wiring.

## Mechanism

The route mount stays inside the existing Content Ops defensive import
block so missing extracted-package dependencies do not break Atlas
startup. It shares the same `Depends(_capture_content_ops_auth_user)`
dependency used by `/content-ops/*`, so tenant scope continues to flow
through the existing ContextVar bridge.

The pool provider is `atlas_brain.storage.database.get_db_pool`; the
extracted generated-asset router already accepts a host pool provider
and performs the database-unavailable guard itself.

## Intentional

- No frontend review UI in this slice.
- No changes to the extracted generated-asset router internals.
- No new review/export behavior; this only exposes the already-tested
  router through Atlas.

## Deferred

- Richer operator review UX in the frontend.
- More detailed generated-asset previews beyond the current API
  payload/export surfaces.

## Verification

- `pytest tests/test_atlas_content_ops_generated_assets_api.py -q`
  - 2 passed, 1 existing torch/pynvml warning.
- `pytest tests/test_extracted_content_asset_api.py tests/test_extracted_campaign_api_hosted_workflow.py::test_hosted_workflow_mounts_generated_asset_router_with_shared_ports -q`
  - 16 passed.
- `git diff --check`
  - Passed.
- Python compile check for `atlas_brain/api/__init__.py` and
  `tests/test_atlas_content_ops_generated_assets_api.py`
  - Passed.

### Files touched

- `atlas_brain/api/__init__.py`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Generated-Assets-Route.md`
- `tests/test_atlas_content_ops_generated_assets_api.py`

## Estimated diff size

| Area | Estimate |
|---|---:|
| Atlas API mount | ~10 LOC |
| Tests | ~65 LOC |
| Plan + coordination | ~60 LOC |
| Total | ~140 LOC |
