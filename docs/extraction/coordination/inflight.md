# In-Flight PRs

Last updated: 2026-05-20T17:44Z by codex-2026-05-20

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-Content-Ops-CFPB-Live-Fetch-Compat | `plans/PR-Content-Ops-CFPB-Live-Fetch-Compat.md`; `docs/extraction/coordination/inflight.md`; `scripts/export_content_ops_cfpb_sources.py`; `tests/test_export_content_ops_cfpb_sources.py` | codex-2026-05-20 | Avoid concurrent edits to the CFPB source exporter request/query contract and its tests. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
