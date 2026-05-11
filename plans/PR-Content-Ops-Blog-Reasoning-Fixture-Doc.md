# PR-Content-Ops-Blog-Reasoning-Fixture-Doc

## Why this slice exists

The Content Ops backend catalog and tests now classify `blog_post` as
`reasoning_requirement="optional_host_context"`, but the frontend catalog
fixture and preview API documentation still say `absent`. That stale value can
mislead future UI work and review even though the runtime catalog is correct.

This slice also replaces the stale in-flight row left by PR 475 with the active
coordination row for this corrective PR.

## Scope (this PR)

1. Update the static Content Ops frontend catalog fixture so `blog_post`
   matches the backend catalog.
2. Update the preview API output-catalog table to match the backend catalog.
3. Refresh the coordination in-flight row for this PR.

### Files touched

- `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json`
- `extracted_content_pipeline/docs/control_surface_preview_api.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Blog-Reasoning-Fixture-Doc.md`

## Mechanism

The backend source of truth is `extracted_content_pipeline/control_surfaces.py`.
Its `OUTPUT_CATALOG["blog_post"].reasoning_requirement` is already
`optional_host_context`, and backend tests assert that value. This PR updates
the static fixture and documentation to the same value; no runtime code changes.

## Intentional

- No backend or UI component logic changes. The runtime contract is already
  correct; only static contract material was stale.
- No fixture regeneration. The only stale field found in this slice is the
  `blog_post` reasoning requirement.

## Deferred

- None. This closes the discovered fixture/doc drift for `blog_post`.

## Verification

- `npm ci` in `atlas-intel-ui`
- `npm run build` in `atlas-intel-ui`
- `rg -n "blog_post.*absent|absent.*blog_post|\\| `blog_post` \\| Implemented \\| `absent`" atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json extracted_content_pipeline/docs/control_surface_preview_api.md docs/frontend/content_ops_frontend_contract.md tests extracted_content_pipeline/control_surfaces.py`
- `git diff --check`

## Estimated diff size

4 files, under 120 LOC total.
