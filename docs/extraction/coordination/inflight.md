# In-Flight PRs

Last updated: 2026-05-16T21:45Z by codex-2026-05-16

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #561 | Content Ops blog narrative pack | `extracted_content_pipeline/api/control_surfaces.py`, `extracted_content_pipeline/content_ops_execution.py`, `extracted_content_pipeline/generation_plan.py`, `extracted_content_pipeline/reasoning_policy.py`, `tests/test_extracted_content_control_surface_api.py`, `tests/test_extracted_content_ops_execution.py`, `tests/test_extracted_content_generation_plan.py`, `tests/test_extracted_content_reasoning_policy.py`, `extracted_content_pipeline/STATUS.md`, `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`, `plans/PR-Content-Ops-Blog-Narrative-Pack.md` | codex-2026-05-16 | Avoid packaged reasoning runtime output/preset edits and blog structured reasoning router changes. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
