# In-Flight PRs

Last updated: 2026-05-14T18:24Z by codex-2026-05-14-content-ops-live-persistence-smoke

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (in flight) | Add Content Ops live execute persistence smoke | NEW: `plans/PR-Content-Ops-Live-Persistence-Smoke.md`. EDIT: `tests/test_extracted_content_ops_live_execute_harness.py`; `docs/extraction/coordination/inflight.md`. | codex-2026-05-14-content-ops-live-persistence-smoke | audit-tooling split PRs; competitive-intelligence extraction files |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
