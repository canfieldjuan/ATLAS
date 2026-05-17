# In-Flight PRs

Last updated: 2026-05-17T17:18Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #567 | Add email campaign packaged reasoning runtime parity | `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/queue.md`, `plans/PR-ContentOps-Email-Packaged-Reasoning.md`, `extracted_content_pipeline/reasoning_policy.py`, `extracted_content_pipeline/generation_plan.py`, `extracted_content_pipeline/api/control_surfaces.py`, `extracted_content_pipeline/STATUS.md`, `tests/test_extracted_content_reasoning_policy.py`, `tests/test_extracted_content_generation_plan.py`, `tests/test_extracted_content_control_surface_api.py` | codex-2026-05-17 | Avoid editing packaged Content Ops reasoning runtime output allowlists while this aligns email campaign policy with runtime execution. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
