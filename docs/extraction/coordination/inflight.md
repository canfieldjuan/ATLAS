# In-Flight PRs

Last updated: 2026-05-17T21:20Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #577 | Reasoning core checks wrapper | `scripts/run_extracted_reasoning_core_checks.sh`, `scripts/run_extracted_pipeline_checks.sh`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `plans/PR-Reasoning-Core-Checks-Wrapper.md` | codex-2026-05-17 | Avoid editing extracted reasoning/core check runners or extraction coordination state |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
