# PR: Content Ops Backlog Current

## Why this slice exists

PRs #508 and #509 closed the intervention reasoning provider and consumed
reasoning detail UI slices, but the AI Content Ops deferred backlog still lists
both as active. The coordination ledger also still claims the merged #509 UI
slice. That makes the next work item ambiguous and risks another session
reclaiming completed work.

## Scope

1. Move the intervention provider and full reasoning detail UI items from active
   backlog to retired historical deferrals.
2. Renumber the remaining active backlog.
3. Update the current pick recommendation to the next real slice: DB reasoning
   provider hardening.
4. Clear the merged Content Ops UI row from the in-flight ledger.

### Files touched

- `plans/PR-Content-Ops-Backlog-Current.md`
- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

This is a documentation-only synchronization slice. No runtime code changes.
The backlog keeps DB reasoning provider hardening and operator review UX as the
remaining active items; the current recommendation points at DB hardening first
because provider semantics should settle before adding more operator polish.

## Intentional

- No product code changes.
- No new coordination owner row, because this PR itself is the ledger cleanup.
- Operator review UX stays active; richer previews are still useful but lower
  priority than the provider storage semantics.

## Deferred

- The actual DB reasoning provider hardening implementation.
- Any broader roadmap rewrite outside the short AI Content Ops deferral list.

## Verification

- Local PR review passes.
- Diff whitespace check passes.

## Estimated diff size

| Area | Estimated LOC |
| --- | ---: |
| Backlog doc | ~45 |
| Coordination ledger | ~5 |
| Plan doc | ~45 |
| **Total** | **~95** |
