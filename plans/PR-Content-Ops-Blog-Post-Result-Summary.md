# PR: Content Ops blog-post result summary

## Why this slice exists

`blog_post` is the remaining implemented Content Ops output without a
frontend execution-result summary. A code read confirmed
`BlogPostGenerationResult.as_dict()` uses the same generated-asset
shape as reports, landing pages, and sales briefs, so the UI can reuse
the shared summary added in the prior slice.

## Scope (this PR)

1. Route `blog_post` execution results through the shared generated
   asset summary.
2. Update the frontend contract so all generated-asset outputs are
   listed together.
3. Claim the slice in the extraction coordination table while the PR
   is open.

### Files touched

- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `docs/frontend/content_ops_frontend_contract.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Blog-Post-Result-Summary.md`

## Mechanism

`ExecutionStepSummary` already routes generated-asset outputs to
`GeneratedAssetSummary`. This PR adds `blog_post` to that output list.

The component defensively reads only `requested`, `generated`,
`skipped`, `saved_ids`, and `errors`; unknown fields stay available in
the raw JSON details block.

## Intentional

- No new blog-specific UI component. The backend result contract is the
  same as the other generated assets.
- No backend changes.
- No changes to `blog_generation.py`; it was read only to verify the
  result contract.
- No competitive-intelligence files touched.

## Deferred

- Rich blog-post preview details, such as title or excerpt, if the
  backend later exposes stable preview fields in the execution result.
- Component-level tests for `ContentOpsNewRun` if the frontend test
  harness is introduced.

## Verification

- `cd atlas-intel-ui && npm ci`
- `cd atlas-intel-ui && npm run build`
- `git diff --check`
- Confirm no non-ASCII added lines in the TypeScript diff.

## Estimated diff size

- 4 files.
- About 40 inserted lines and 2 deleted lines.
- Well below the 400-line soft PR budget.
