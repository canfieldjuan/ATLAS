# PR: Content Ops blog reasoning catalog

## Why this slice exists

`BlogPostGenerationService` already exposes `with_reasoning_context`
and threads a host `CampaignReasoningContextProvider` into its prompt
payload. The control-surface catalog still marks `blog_post` as
`reasoning_requirement="absent"`, so the frontend cannot show the
reasoning-readiness badge for a real reasoning-aware output.

## Scope (this PR)

1. Mark `blog_post` as `optional_host_context` in the output catalog.
2. Update the frontend contract's reasoning vocabulary.
3. Update the catalog API test that locks output reasoning flags.
4. Claim this slice in the extraction coordination table while the PR
   is open.

### Files touched

- `extracted_content_pipeline/control_surfaces.py`
- `docs/frontend/content_ops_frontend_contract.md`
- `tests/test_extracted_content_control_surface_api.py`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Blog-Reasoning-Catalog.md`

## Mechanism

The change is catalog-only. `blog_post` gets the same
`reasoning_requirement="optional_host_context"` value as the other
generated assets that can consume `CampaignReasoningContextProvider`.

Execution behavior is already wired through
`ContentOpsExecutionServices.with_reasoning_context`.

## Intentional

- No changes to `BlogPostGenerationService`; the service already
  supports the provider.
- No frontend code changes; the existing badge is catalog-driven.
- No change to `signal_extraction`, which remains deterministic and
  reasoning-absent.
- No competitive-intelligence files touched.

## Deferred

- Explicit per-step consumed-reasoning audit field in execution
  results.
- Reasoning Context Drawer UI after that backend field exists.

## Verification

- `python -m pytest tests/test_extracted_content_control_surface_api.py`
- `git diff --check`

## Estimated diff size

- 5 files.
- About 65 inserted lines and 4 deleted lines.
- Well below the 400-line soft PR budget.
