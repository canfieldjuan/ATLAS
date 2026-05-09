# PR: Content Ops reasoning-context contract correction

## Why this slice exists

The frontend contract currently says execution results typically expose
consumed reasoning context in `step.result` and lists a Reasoning
Context Drawer as an MVP screen. A code read shows that is ahead of the
backend contract: `ContentOpsStepExecution.result` receives each
service's summary `as_dict()`, and the generated-asset result summaries
currently expose counts, saved IDs, and errors, not consumed reasoning
payloads.

Leaving the contract as-is would push the UI toward a drawer that has
no reliable runtime data source.

## Scope (this PR)

1. Correct the frontend contract language for host-injected reasoning.
2. Move the Reasoning Context Drawer from MVP screen to deferred work.
3. Record the same caveat in `extracted_content_pipeline/STATUS.md`.
4. Claim this slice in the extraction coordination table while the PR
   is open.

### Files touched

- `docs/frontend/content_ops_frontend_contract.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Reasoning-Context-Contract-Correction.md`

## Mechanism

The docs now separate three facts:

- The catalog can say whether host reasoning is configured.
- Output definitions can say whether an output can consume optional
  host context.
- Execution summaries do not currently expose consumed reasoning
  context; a future backend result field is required before a drawer is
  actionable.

## Intentional

- No UI changes. The output-card reasoning badge already exists and is
  backed by current catalog data.
- No backend changes. This slice corrects the contract before adding a
  new result field.
- No competitive-intelligence files touched.

## Deferred

- Backend execution-result field for consumed reasoning context or a
  compact reasoning audit summary.
- Reasoning Context Drawer UI after that field exists.
- Component tests for the drawer when it becomes implementable.

## Verification

- `git diff --check`
- Documentation-only change; no runtime tests required.

## Estimated diff size

- 4 files.
- About 75 inserted lines and 10 deleted lines.
- Well below the 400-line soft PR budget.
