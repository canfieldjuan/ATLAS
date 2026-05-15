# In-Flight PRs

Last updated: 2026-05-15T23:34Z by codex-2026-05-15

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops source bundle example | `extracted_content_pipeline/examples/campaign_source_bundle.json`, `tests/test_extracted_campaign_source_adapters.py`, `tests/test_extracted_content_host_smoke.py`, `tests/test_extracted_campaign_generation_example.py`, `extracted_content_pipeline/README.md`, `extracted_content_pipeline/docs/host_install_runbook.md`, `extracted_content_pipeline/STATUS.md`, `docs/extraction/coordination/inflight.md`, `plans/PR-Content-Ops-Source-Bundle-Example.md` | codex-2026-05-15 | Avoid source-bundle examples/docs and source-row smoke tests |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
