# PR: Refresh Content Ops Backlog After Reasoning Admin

## Why This Slice Exists

The DB reasoning context admin API, scoped delete path, and visibility events
have merged. The active AI Content Ops backlog still lists that work as open,
which makes the next slice look less clear than it is.

## Scope

Refresh the docs so the completed reasoning-admin work moves to retired
deferrals and the next active Content Ops work item is the generated-asset
preview UX.

### Files Touched

- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Backlog-Reasoning-Admin-Close.md`

## Mechanism

- Retire the merged hosted admin list/upsert, scoped delete, and visibility
  event work.
- Remove the stale active reasoning-admin backlog item.
- Promote generated-asset preview UX as the next recommended Content Ops slice.
- Replace the stale in-flight coordination row with this docs-only claim.

## Intentional

- No production code changes.
- No frontend implementation in this slice.
- No changes to reasoning-core or generated-asset runtime behavior.

## Deferred

- Generated-asset preview UX implementation.
- Frontend component tests for richer result previews.
- Reasoning-core/product-depth work.

## Verification

- `git diff --check`
- `scripts/local_pr_review.sh`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Backlog doc | ~50 |
| Coordination row | ~2 |
| Plan doc | ~50 |
| **Total** | ~102 |
