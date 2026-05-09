# PR: Content Ops blog reasoning key parity

## Why this slice exists

PR #416 marked `blog_post` as a reasoning-aware output. Review noted a
small parity gap: landing pages write both `reasoning_context` and
`campaign_reasoning_context` into the prompt-visible payload, while
blog generation only writes `reasoning_context`.

The reasoning payload reaches the prompt today under `reasoning_context`;
adding the sibling key keeps reasoning-aware generated assets consistent
for downstream tooling that may inspect `campaign_reasoning_context`
directly.

## Scope (this PR)

1. Add `campaign_reasoning_context` beside `reasoning_context` in the
   enriched blog blueprint payload.
2. Update the existing blog reasoning-provider test to assert both keys
   reach the prompt JSON.
3. Claim this slice in the extraction coordination table while the PR
   is open.

### Files touched

- `extracted_content_pipeline/blog_generation.py`
- `tests/test_extracted_blog_generation.py`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Blog-Reasoning-Key-Parity.md`

## Mechanism

`_blueprint_with_reasoning_context()` now computes the normalized
reasoning payload once, applies existing metadata fields, then writes it
to both `reasoning_context` and `campaign_reasoning_context`.

## Intentional

- No changes to provider resolution, catalog metadata, or frontend UI.
- No changes when the provider is absent or returns empty context.
- No competitive-intelligence files touched.

## Deferred

- Explicit consumed-reasoning audit field in Content Ops execution
  results.
- Reasoning Context Drawer UI after that backend field exists.

## Verification

- `python -m pytest tests/test_extracted_blog_generation.py`
- `git diff --check`

## Estimated diff size

- 4 files.
- About 60 inserted lines and 2 deleted lines.
- Well below the 400-line soft PR budget.
