# Content Ops Reasoning Docs Current

## Why this slice exists

PRs #472-#474 closed the reasoning-source and consumed-context path through the
backend, route boundary, and Atlas Intel UI. A few docs still described the
Reasoning Context Drawer as wholly future work even though the UI now renders
compact consumed-context summaries.

## Scope (this PR)

1. Update status/frontend docs to reflect that compact consumed-context
   rendering has shipped.
2. Keep full drawer/detail UX explicitly deferred.
3. Remove the merged #474 coordination row and claim this slice.

### Files touched

- `extracted_content_pipeline/STATUS.md`
- `docs/frontend/content_ops_frontend_contract.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Reasoning-Docs-Current.md`

## Mechanism

This is documentation-only. It changes stale future-tense language to current
state:

- `reasoning.consumed_contexts` exists in execution responses.
- Atlas Intel UI renders compact summaries for consumed contexts.
- A richer drawer/detail inspector remains deferred.

## Intentional

- No production code or tests change.
- Older plan docs are not rewritten; they are historical records of the PRs
  they described.

## Deferred

- Full consumed-context drawer/detail UX.
- Any additional frontend screenshots or browser automation.

## Verification

- `git diff --check` -> passed

## Estimated diff size

4 files, about +70 / -10 including this plan.
