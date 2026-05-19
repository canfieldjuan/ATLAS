# In-Flight PRs

Last updated: 2026-05-19T04:55Z by codex-2026-05-18

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-Content-Ops-CFPB-Live-Provider-Smoke | `scripts/smoke_content_ops_cfpb_source_postgres.py`, CFPB smoke tests/docs | codex-2026-05-18 | Invoicing route-check PRs; unrelated extraction slices |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
