# Content Ops Backlog Refresh

## Why this slice exists

PR #522 merged the reasoning upsert audit-log slice, but the active backlog
still listed DB reasoning provider hardening as the next task. That would steer
future sessions back into completed work.

## Scope (this PR)

1. Retire the completed DB reasoning provider/admin seams from the active
   backlog.
2. Reorder the remaining Content Ops deferrals around what is actually left.
3. Replace the stale PR #522 coordination row with this docs-only slice.

### Files touched

- `plans/PR-Content-Ops-Backlog-Refresh.md`
- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The backlog now treats DB target-mode filtering, settings selection, upsert,
list/export, stale cleanup, dry-run, audit logging, drawer/detail UI, and batch
review as retired. The remaining active items are:

- reasoning context admin workflow
- optional live-opportunity validation before reasoning upsert
- richer generated-asset previews and frontend tests
- optional batch-review scale hardening
- broader reasoning/source-breadth roadmap work

## Intentional

- No production code changes.
- No test changes.
- No claim that the broader reasoning-core roadmap is complete.

## Deferred

- Implement the next selected backlog item.
- Refresh broader coordination state if we decide to make this backlog the
  canonical next-slice source for all AI Content Ops work.

## Verification

```bash
bash scripts/local_pr_review.sh
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `plans/PR-Content-Ops-Backlog-Refresh.md` | 50 |
| `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | 75 |
| `docs/extraction/coordination/inflight.md` | 2 |
| **Total** | **~127** |
