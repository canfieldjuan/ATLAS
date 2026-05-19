# In-Flight PRs

Last updated: 2026-05-19T03:58Z by codex-2026-05-18

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-Content-Ops-Support-Ticket-Examples | `extracted_content_pipeline/examples/support_ticket_sources.csv`, `extracted_content_pipeline/examples/support_ticket_bundle.json`, `extracted_content_pipeline/README.md`, `extracted_content_pipeline/docs/host_install_runbook.md`, `tests/test_extracted_campaign_source_adapters.py`, `docs/extraction/coordination/inflight.md`, `plans/PR-Content-Ops-Support-Ticket-Examples.md` | codex-2026-05-18 | Content Ops support-ticket ingestion examples |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
