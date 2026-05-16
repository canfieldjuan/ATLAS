# In-Flight PRs

Last updated: 2026-05-16T21:07Z by codex-2026-05-16

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #559 | Content Ops reasoning runtime invariants | `extracted_content_pipeline/reasoning_policy.py`, `extracted_content_pipeline/generation_plan.py`, `extracted_content_pipeline/api/control_surfaces.py`, `tests/test_extracted_content_reasoning_policy.py`, `tests/test_extracted_content_generation_plan.py`, `tests/test_extracted_content_control_surface_api.py`, `extracted_content_pipeline/STATUS.md`, `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`, `plans/PR-Content-Ops-Reasoning-Runtime-Invariants.md` | codex-2026-05-16 | Avoid reasoning preset/runtime packaged-output validation edits. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
