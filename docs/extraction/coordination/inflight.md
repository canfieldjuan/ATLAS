# In-Flight PRs

Last updated: 2026-05-16T22:32Z by codex-2026-05-16

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #562 | Content Ops reasoning backlog closeout | `docs/extraction/coordination/inflight.md`, `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`, `plans/PR-Content-Ops-Reasoning-Backlog-Closeout.md` | codex-2026-05-16 | Avoid editing the AI Content Ops deferred backlog recommendation while this closes the shipped reasoning-policy arc. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
