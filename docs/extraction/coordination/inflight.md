# In-Flight PRs

Last updated: 2026-05-04T02:01Z by codex-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-D9, in flight) | Add AI Content Ops draft review/status update path | NEW: `extracted_content_pipeline/campaign_postgres_review.py`; NEW: `scripts/review_extracted_campaign_drafts.py`; EDIT: content-pipeline docs/status/manifest/check script; NEW focused review tests | codex-2026-05-03 | Avoid `extracted_reasoning_core/**`, `extracted_content_pipeline/reasoning/**`, `extracted_content_pipeline/docs/reasoning_state_audit.md`, and `extracted_quality_gate/**` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
