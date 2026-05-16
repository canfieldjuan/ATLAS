# In-Flight PRs

Last updated: 2026-05-16T00:35Z by codex-2026-05-15

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops renewal source rows | `extracted_content_pipeline/campaign_source_adapters.py`, `tests/test_extracted_campaign_source_adapters.py`, `extracted_content_pipeline/README.md`, `extracted_content_pipeline/docs/host_install_runbook.md`, `extracted_content_pipeline/STATUS.md`, `docs/extraction/coordination/inflight.md`, `plans/PR-Content-Ops-Renewal-Source-Rows.md` | codex-2026-05-15 | Avoid source-row adapter taxonomy edits |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
