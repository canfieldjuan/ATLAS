# In-Flight PRs

Last updated: 2026-05-16T04:29Z by codex-2026-05-15

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops reasoning capability check | `extracted_content_pipeline/api/campaign_operations.py`, `tests/test_extracted_campaign_api_operations.py`, `extracted_content_pipeline/STATUS.md`, `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`, `docs/extraction/coordination/inflight.md`, `plans/PR-Content-Ops-Reasoning-Capability-Check.md` | codex-2026-05-15 | Avoid campaign operations status/readiness edits |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
