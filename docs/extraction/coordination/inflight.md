# In-Flight PRs

Last updated: 2026-05-19T17:34Z by codex-2026-05-19

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-Live-Provider-Circular-Import | `atlas_brain/auth/__init__.py`, `tests/test_live_provider_circular_import.py`, `plans/PR-Live-Provider-Circular-Import.md` | codex-2026-05-19 | Auth package init, live provider smoke/import-boundary work |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
