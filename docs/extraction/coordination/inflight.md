# In-Flight PRs

Last updated: 2026-05-15T23:24Z by codex-2026-05-15

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops source bundle ingest | `extracted_content_pipeline/campaign_source_adapters.py`, `tests/test_extracted_campaign_source_adapters.py`, `docs/extraction/coordination/inflight.md`, `plans/PR-Content-Ops-Source-Bundle-Ingest.md` | codex-2026-05-15 | Avoid source-row adapter loader and source-adapter tests |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
