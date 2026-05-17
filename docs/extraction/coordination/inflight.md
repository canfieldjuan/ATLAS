# In-Flight PRs

Last updated: 2026-05-17T17:41Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #568 | Close out Content Ops reasoning-policy backlog | `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/queue.md`, `docs/extraction/coordination/state.md`, `extracted_content_pipeline/STATUS.md`, `plans/PR-ContentOps-Reasoning-Policy-Closeout-2026-05-17.md` | codex-2026-05-17 | Avoid editing Content Ops reasoning-policy closeout/backlog wording while this reconciles the post-#567 state. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
