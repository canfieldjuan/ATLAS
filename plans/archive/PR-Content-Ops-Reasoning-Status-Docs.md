# PR-Content-Ops-Reasoning-Status-Docs

## Why this slice exists

Recent Content Ops slices added consumed-reasoning counts to generated asset
results and surfaced those counts in `ContentOpsStepExecution.reasoning`.
`extracted_content_pipeline/STATUS.md` and one frontend-contract type table
still describe the older state, which makes the product status look behind the
runtime.

## Scope (this PR)

1. Update `STATUS.md` to describe the current execution-result reasoning audit.
2. Update the frontend contract's frozen-dataclass field table for
   `ContentOpsStepExecution`.
3. Keep the Reasoning Context Drawer deferred because raw consumed reasoning
   payloads are still intentionally not exposed.

### Files touched

- `extracted_content_pipeline/STATUS.md`
- `docs/frontend/content_ops_frontend_contract.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

This is a docs-only contract correction. It aligns the docs with the existing
runtime shape:

- `result.reasoning_contexts_used` is a runner-specific count when services
  expose it.
- `reasoning.contexts_used` is the compact per-step audit count.
- Raw consumed reasoning payloads remain out of the execution response.

## Intentional

- No runtime or API shape changes.
- No new Reasoning Context Drawer contract; only counts are documented.

## Deferred

- A future drawer-ready response field carrying raw consumed reasoning context.

## Verification

- `rg -n "do not currently expose consumed reasoning|ContentOpsStepExecution" extracted_content_pipeline/STATUS.md docs/frontend/content_ops_frontend_contract.md`
- `git diff --check`

## Estimated diff size

About 3 files, under 50 changed lines.
